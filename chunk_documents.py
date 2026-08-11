from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================
# PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"


# =========================
# LOAD DOCUMENTS
# =========================

documents = []

pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF files.\n")

for pdf_path in pdf_files:
    print(f"Loading: {pdf_path.name}")

    try:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        documents.extend(pages)

        print(f"→ {len(pages)} pages loaded")

    except Exception as e:
        print(f"ERROR loading {pdf_path.name}: {e}")


print(f"\nTotal documents/pages: {len(documents)}")


# =========================
# CHUNK DOCUMENTS
# =========================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")


# =========================
# CHECK RESULTS
# =========================

print("\n--- SAMPLE CHUNKS ---\n")

for i, chunk in enumerate(chunks[:5]):
    print(f"CHUNK {i + 1}")
    print(f"Source: {chunk.metadata.get('source')}")
    print(f"Page: {chunk.metadata.get('page')}")
    print(f"Characters: {len(chunk.page_content)}")
    print("-" * 50)
    print(chunk.page_content[:500])
    print("\n")