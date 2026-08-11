from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading Chroma database...")

vector_db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Check how many documents are actually stored
collection = vector_db._collection

print(f"\nDocuments in Chroma: {collection.count()}")

if collection.count() == 0:
    print("ERROR: The Chroma database is empty.")
else:
    print("Database contains documents.")

    query = "How can I pay for buses in London?"

    print(f"\nQuery: {query}")
    print("Searching...\n")

    results = vector_db.similarity_search(query, k=3)

    print(f"Results returned: {len(results)}\n")

    for i, doc in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(doc.page_content[:1000])
        print()