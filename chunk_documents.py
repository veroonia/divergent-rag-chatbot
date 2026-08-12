from pathlib import Path
import re
import uuid

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_DIR = Path(__file__).resolve().parent / "data"
PDF_PATH = DATA_DIR / "Divergent.pdf"

# Approximate token targets
PARENT_CHUNK_SIZE = 1500
PARENT_CHUNK_OVERLAP = 150

CHILD_CHUNK_SIZE = 350
CHILD_CHUNK_OVERLAP = 50


def load_book():
    print(f"Loading: {PDF_PATH}")

    loader = PyMuPDF4LLMLoader(
        str(PDF_PATH),
        mode="page"
    )

    documents = loader.load()

    print(f"Pages loaded: {len(documents)}")

    return documents


def clean_text(text):
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def detect_chapter(text):
    match = re.search(
        r"(?i)\bChapter\s+([A-Za-z]+(?:[-\s][A-Za-z]+)*)\b",
        text
    )

    if match:
        return f"Chapter {match.group(1)}"

    return None


def is_front_matter(text):
    """
    Remove pages before the actual novel begins.
    """

    lower = text.lower()

    front_matter_terms = [
        "# divergent",
        "# veronica roth",
        "dedication",
        "contents",
        "table of contents"
    ]

    return any(term in lower for term in front_matter_terms)


def create_page_documents(documents):
    """
    Clean pages, remove front matter and track the current chapter.
    """

    processed_pages = []

    current_chapter = None

    for document in documents:

        text = clean_text(document.page_content)

        if not text:
            continue

        # Skip title / contents / dedication pages
        if is_front_matter(text):
            continue

        detected = detect_chapter(text)

        if detected:
            current_chapter = detected

        page_number = document.metadata.get("page")

        processed_pages.append({
            "text": text,
            "page": page_number,
            "chapter": current_chapter
        })

    return processed_pages


def create_parent_chunks(pages):

    # RecursiveCharacterTextSplitter uses characters.
    # 1500 tokens is roughly 6000 characters for English.
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=6000,
        chunk_overlap=600,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            ""
        ]
    )

    parents = []

    for page in pages:

        chunks = parent_splitter.split_text(page["text"])

        for chunk in chunks:

            parent_id = str(uuid.uuid4())

            parents.append({
                "id": parent_id,
                "text": chunk,
                "metadata": {
                    "source": str(PDF_PATH),
                    "page": page["page"],
                    "chapter": page["chapter"],
                    "type": "parent"
                }
            })

    return parents


def create_child_chunks(parents):

    # ~350 tokens ≈ 1400 characters
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1400,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            ""
        ]
    )

    children = []

    for parent in parents:

        child_chunks = child_splitter.split_text(
            parent["text"]
        )

        for child_number, child_text in enumerate(child_chunks):

            child_id = str(uuid.uuid4())

            children.append({
                "id": child_id,
                "text": child_text,
                "metadata": {
                    "source": parent["metadata"]["source"],
                    "page": parent["metadata"]["page"],
                    "chapter": parent["metadata"]["chapter"],
                    "type": "child",
                    "parent_id": parent["id"],
                    "child_number": child_number
                }
            })

    return children


if __name__ == "__main__":

    documents = load_book()

    print("\nCleaning pages and detecting chapters...")

    pages = create_page_documents(documents)

    print(f"Usable pages: {len(pages)}")

    print("\nCreating parent chunks...")

    parents = create_parent_chunks(pages)

    print(f"Parents created: {len(parents)}")

    print("\nCreating child chunks...")

    children = create_child_chunks(parents)

    print(f"Children created: {len(children)}")

    print("\n" + "=" * 60)
    print("EXAMPLE PARENT")
    print("=" * 60)

    print(parents[0]["text"])

    print("\nMetadata:")
    print(parents[0]["metadata"])

    print("\n" + "=" * 60)
    print("EXAMPLE CHILD")
    print("=" * 60)

    print(children[0]["text"])

    print("\nMetadata:")
    print(children[0]["metadata"])