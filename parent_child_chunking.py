from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


# ---------------------------------------------------------
# Find all PDFs
# ---------------------------------------------------------

pdf_files = sorted(DATA_DIR.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF files.")

if not pdf_files:
    print("ERROR: No PDF files found in the data folder.")
    raise SystemExit(1)


# ---------------------------------------------------------
# Load documents
# ---------------------------------------------------------

documents = []

for pdf_path in pdf_files:
    print(f"\nLoading: {pdf_path.name}")

    try:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        print(f"→ {len(pages)} pages loaded")

        # Store the source filename in metadata
        for page in pages:
            page.metadata["source_file"] = pdf_path.name

        documents.extend(pages)

    except Exception as error:
        print(f"ERROR loading {pdf_path.name}: {error}")


print(f"\nTotal documents/pages: {len(documents)}")


# ---------------------------------------------------------
# Split documents into chunks
# ---------------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")


# ---------------------------------------------------------
# Show sample chunks
# ---------------------------------------------------------

print("\n--- SAMPLE CHUNKS ---\n")

for i, chunk in enumerate(chunks[:5], start=1):
    print(f"--- Chunk {i} ---")
    print(f"Source: {chunk.metadata.get('source_file')}")
    print(chunk.page_content[:1000])
    print()