from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DOCUMENTS_DIR = Path("documents")

WEBPAGES = {
   # "london_attractions": "https://www.visitlondon.com/things-to-do/sightseeing/london-attraction",
    #"london_food": "https://www.visitlondon.com/things-to-do/food-and-drink",
    #"london_itinerary": "https://www.visitlondon.com/things-to-do/visiting-london-for-the-first-time", 
}


# --------------------------------------------------
# Load PDF documents
# --------------------------------------------------

def load_pdf_documents():
    documents = []

    pdf_files = [
        DOCUMENTS_DIR / "london_general_guide.pdf",
        DOCUMENTS_DIR / "london_transportation.pdf",
    ]

    for pdf_file in pdf_files:
        print(f"Loading PDF: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = pdf_file.name
            page.metadata["type"] = "pdf"

        documents.extend(pages)

        print(f"  → {len(pages)} pages loaded")

    return documents


# --------------------------------------------------
# Load webpage documents
# --------------------------------------------------

def load_web_documents():
    documents = []

    for name, url in WEBPAGES.items():
        print(f"\nLoading webpage: {name}")

        loader = WebBaseLoader(url)

        pages = loader.load()

        for page in pages:
            page.metadata["source"] = name
            page.metadata["url"] = url
            page.metadata["type"] = "webpage"

        documents.extend(pages)

        print(f"  → {len(pages)} webpage document(s) loaded")

    return documents


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    pdf_documents = load_pdf_documents()
    web_documents = load_web_documents()

    documents = pdf_documents + web_documents

    print("\n" + "=" * 60)
    print("DOCUMENT LOADING COMPLETE")
    print("=" * 60)

    print(f"Total Document objects: {len(documents)}")

    # --------------------------------------------------
    # Check extracted text
    # --------------------------------------------------

    print("\n--- TEXT EXTRACTION CHECK ---\n")

    for document in documents:

        text = document.page_content.strip()

        print(
            f"{document.metadata['source']} | "
            f"Type: {document.metadata['type']} | "
            f"Characters: {len(text)}"
        )

    # --------------------------------------------------
    # Show samples from every source
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TEXT SAMPLES")
    print("=" * 60)

    for document in documents:

        text = document.page_content.strip()

        if text:

            print(
                f"\n--- {document.metadata['source']} "
                f"({document.metadata['type']}) ---\n"
            )

            print(text[:1000])