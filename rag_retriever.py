from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore


# ============================================================
# SETTINGS
# ============================================================

QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "divergent_children"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chroma/Qdrant distance:
# Lower distance = more similar.
#
# This is only used AFTER we already know the question
# is a document-related question.
SIMILARITY_THRESHOLD = 0.65


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

print("Embedding model loaded.")


# ============================================================
# LOAD QDRANT DATABASE
# ============================================================

print("Connecting to Qdrant...")

print(f"Qdrant: {QDRANT_URL}")
print(f"Collection: {COLLECTION_NAME}")

vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    url=QDRANT_URL,
)

print("Qdrant vector database loaded.")


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(query, k=5):
    """
    Retrieve the most relevant document chunks from Qdrant.

    This function does NOT decide whether the question is
    about the book.

    That decision is handled by the document/general router
    in app.py.
    """

    results = vector_db.similarity_search_with_score(
        query,
        k=k,
    )

    return results


# ============================================================
# GET RELEVANT CONTEXT
# ============================================================

def get_relevant_context(query, k=5):
    """
    Retrieve relevant document chunks.

    Lower Qdrant distance means a better semantic match.

    Weak matches are ignored using SIMILARITY_THRESHOLD.
    """

    results = retrieve_documents(
        query,
        k=k,
    )

    if not results:
        return ""

    context_parts = []

    for i, (doc, distance) in enumerate(results, 1):

        # ----------------------------------------------------
        # Ignore weak semantic matches
        # ----------------------------------------------------

        if distance > SIMILARITY_THRESHOLD:
            continue

        # ----------------------------------------------------
        # Extract metadata
        # ----------------------------------------------------

        book = doc.metadata.get(
            "book",
            "Unknown",
        )

        chapter = doc.metadata.get(
            "chapter",
            "Unknown",
        )

        source = doc.metadata.get(
            "source",
            "Unknown",
        )

        # ----------------------------------------------------
        # Build context
        # ----------------------------------------------------

        context_parts.append(
            f"[Source {i}]\n"
            f"Book: {book}\n"
            f"Chapter: {chapter}\n"
            f"Source file: {source}\n"
            f"Relevance score: {distance:.4f}\n\n"
            f"{doc.page_content}"
        )

    if not context_parts:
        return ""

    return "\n\n".join(context_parts)