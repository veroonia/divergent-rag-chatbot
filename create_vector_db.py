from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# -----------------------------
# SETTINGS
# -----------------------------

DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# -----------------------------
# LOAD PDFs
# -----------------------------

all_documents = []

pdf_files = list(DATA_DIR.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF files.\n")

for pdf_path in pdf_files:

    print(f"Loading: {pdf_path.name}")

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    print(f"→ {len(pages)} pages loaded")

    all_documents.extend(pages)


print(f"\nTotal pages: {len(all_documents)}")


# -----------------------------
# CHUNK DOCUMENTS
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(all_documents)

print(f"Total chunks: {len(chunks)}")


# -----------------------------
# CREATE EMBEDDINGS
# -----------------------------

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


# -----------------------------
# CREATE VECTOR DATABASE
# -----------------------------

print("\nCreating vector database...")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)

print("\nVector database created successfully!")
print(f"Location: {Path(CHROMA_DIR).resolve()}")


# -----------------------------
# VERIFY DATABASE
# -----------------------------

count = vector_db._collection.count()

print(f"Documents actually stored in Chroma: {count}")

if count == 0:
    print("ERROR: Vector database is empty!")
else:
    print("SUCCESS: Documents were stored correctly.")