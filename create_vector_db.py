import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

from chunk_documents import (
    load_book,
    create_page_documents,
    create_chapters,
    create_parent_chunks,
    create_child_chunks,
)


# ============================================================
# SETTINGS
# ============================================================

BOOK_NAME = "Divergent"

COLLECTION_NAME = "divergent_children"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)


# ============================================================
# LOAD + CHUNK BOOK
# ============================================================

def build_chunks():

    print("=" * 60)
    print("LOADING AND CHUNKING BOOK")
    print("=" * 60)

    # --------------------------------------------------------
    # Load PDF
    # --------------------------------------------------------

    documents = load_book()

    # --------------------------------------------------------
    # Clean pages + detect chapters
    # --------------------------------------------------------

    pages = create_page_documents(
        documents
    )

    print(
        f"\nUsable pages: {len(pages)}"
    )

    # --------------------------------------------------------
    # Group pages into chapters
    # --------------------------------------------------------

    chapters = create_chapters(
        pages
    )

    print(
        f"Chapters detected: {len(chapters)}"
    )

    # --------------------------------------------------------
    # Create parents
    # --------------------------------------------------------

    parents = create_parent_chunks(
        chapters
    )

    print(
        f"Parents created: {len(parents)}"
    )

    # --------------------------------------------------------
    # Create children
    # --------------------------------------------------------

    children = create_child_chunks(
        parents
    )

    print(
        f"Children created: {len(children)}"
    )

    return parents, children


# ============================================================
# CREATE PARENT LOOKUP
# ============================================================

def create_parent_lookup(parents):
    """
    Create a dictionary so that a child can later be mapped
    back to its parent.

    parent_id -> parent information
    """

    parent_lookup = {}

    for parent in parents:

        parent_lookup[
            parent["id"]
        ] = parent

    return parent_lookup


# ============================================================
# CONVERT CHILDREN TO LANGCHAIN DOCUMENTS
# ============================================================

def create_child_documents(
    children,
    parent_lookup
):
    """
    Convert our custom child chunk dictionaries into
    LangChain Documents suitable for Qdrant.

    Only CHILDREN are embedded.

    Each child stores the corresponding parent text in
    metadata so that retrieval can later expand:

        child -> parent
    """

    documents = []

    for child in children:

        parent_id = child["metadata"]["parent_id"]

        parent = parent_lookup.get(
            parent_id
        )

        if parent is None:

            raise ValueError(
                f"Parent {parent_id} "
                f"not found for child "
                f"{child['id']}"
            )

        metadata = {
            # ------------------------------------------------
            # Book
            # ------------------------------------------------

            "book": BOOK_NAME,

            # ------------------------------------------------
            # Hierarchy
            # ------------------------------------------------

            "level": "child",

            "child_id": child["id"],

            "parent_id": parent_id,

            # ------------------------------------------------
            # Chapter
            # ------------------------------------------------

            "chapter": child["metadata"]["chapter"],

            "chapter_index": child["metadata"]["chapter_index"],

            # ------------------------------------------------
            # Position
            # ------------------------------------------------

            "parent_index": child["metadata"]["parent_index"],

            "child_index": child["metadata"]["child_index"],

            # ------------------------------------------------
            # Page information
            # ------------------------------------------------

            "chapter_page_start": child["metadata"][
                "chapter_page_start"
            ],

            "chapter_page_end": child["metadata"][
                "chapter_page_end"
            ],

            # ------------------------------------------------
            # Parent text
            # ------------------------------------------------

            "parent_text": parent["text"],

            # ------------------------------------------------
            # Source
            # ------------------------------------------------

            "source": child["metadata"]["source"],
        }

        documents.append(
            Document(
                page_content=child["text"],
                metadata=metadata
            )
        )

    return documents


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings():

    print("\n" + "=" * 60)
    print("LOADING EMBEDDING MODEL")
    print("=" * 60)

    print(
        f"Model: {EMBEDDING_MODEL}"
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print(
        "Embedding model loaded."
    )

    return embeddings


# ============================================================
# CREATE QDRANT DATABASE
# ============================================================

def create_qdrant(
    child_documents,
    embeddings
):

    print("\n" + "=" * 60)
    print("CREATING QDRANT VECTOR DATABASE")
    print("=" * 60)

    print(
        f"Qdrant: {QDRANT_URL}"
    )

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    print(
        f"Documents to embed: "
        f"{len(child_documents)}"
    )

    vector_store = QdrantVectorStore.from_documents(
        documents=child_documents,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        force_recreate=True,
    )

    print(
        "\nQdrant collection created successfully!"
    )

    return vector_store


# ============================================================
# VERIFY
# ============================================================

def verify_qdrant(
    vector_store,
    expected_count
):

    print("\n" + "=" * 60)
    print("VERIFYING QDRANT")
    print("=" * 60)

    try:

        collection_info = (
            vector_store.client
            .get_collection(
                COLLECTION_NAME
            )
        )

        actual_count = (
            collection_info.points_count
        )

        print(
            f"Expected points: "
            f"{expected_count}"
        )

        print(
            f"Actual points:   "
            f"{actual_count}"
        )

        if actual_count == expected_count:

            print(
                "\nSUCCESS: All child chunks "
                "were stored in Qdrant."
            )

        else:

            print(
                "\nWARNING: Point count does "
                "not match expected count."
            )

    except Exception as e:

        print(
            f"\nCould not verify collection: {e}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. Build hierarchy
    # --------------------------------------------------------

    parents, children = build_chunks()

    # --------------------------------------------------------
    # 2. Create parent lookup
    # --------------------------------------------------------

    parent_lookup = create_parent_lookup(
        parents
    )

    # --------------------------------------------------------
    # 3. Convert children into LangChain Documents
    # --------------------------------------------------------

    child_documents = create_child_documents(
        children,
        parent_lookup
    )

    print(
        f"\nPrepared {len(child_documents)} "
        f"child documents for embedding."
    )

    # --------------------------------------------------------
    # 4. Embeddings
    # --------------------------------------------------------

    embeddings = create_embeddings()

    # --------------------------------------------------------
    # 5. Qdrant
    # --------------------------------------------------------

    vector_store = create_qdrant(
        child_documents,
        embeddings
    )

    # --------------------------------------------------------
    # 6. Verify
    # --------------------------------------------------------

    verify_qdrant(
        vector_store,
        len(child_documents)
    )