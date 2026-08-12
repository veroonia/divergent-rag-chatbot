from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# Folder containing your PDFs
DATA_DIR = Path(__file__).resolve().parent / "data"

print(f"Looking for PDFs in: {DATA_DIR}")

pdf_files = list(DATA_DIR.glob("*.pdf"))

if not pdf_files:
    print("ERROR: No PDF files found.")
    exit()

print(f"\nFound {len(pdf_files)} PDF files:\n")

all_documents = []

for pdf_file in pdf_files:
    print(f"Loading: {pdf_file.name}")

    loader = PyPDFLoader(str(pdf_file))
    documents = loader.load()

    print(f"  Pages loaded: {len(documents)}")

    all_documents.extend(documents)

print(f"\nTotal pages loaded: {len(all_documents)}")