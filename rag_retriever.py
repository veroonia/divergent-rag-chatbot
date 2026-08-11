from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

CHROMA_DIR = str(Path(__file__).resolve().parent / "chroma_db")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------------

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


# ---------------------------------------------------------
# LOAD CHROMA DATABASE
# ---------------------------------------------------------

print("Loading Chroma database...")

vector_db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)


# ---------------------------------------------------------
# RETRIEVE DOCUMENTS
# ---------------------------------------------------------

def retrieve_documents(query, k=5):
    """
    Retrieve the most relevant document chunks.
    """

    results = vector_db.similarity_search_with_score(
        query,
        k=k
    )

    return results


# ---------------------------------------------------------
# GET RELEVANT CONTEXT
# ---------------------------------------------------------

def get_relevant_context(query, k=5):
    """
    Retrieve document chunks only when they are
    sufficiently relevant to the user's question.
    """

    results = retrieve_documents(query, k=k)

    if not results:
        return ""

    context_parts = []

    # Chroma returns:
    # (Document, distance)
    #
    # Lower distance = more similar.
    
    for i, (doc, distance) in enumerate(results, 1):

        # Ignore weak matches
        if distance > 1.0:
            continue

        context_parts.append(
            f"[Source {i}]\n{doc.page_content}"
        )

    if not context_parts:
        return ""

    return "\n\n".join(context_parts)