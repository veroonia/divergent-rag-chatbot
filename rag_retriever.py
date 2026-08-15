from typing import List, Tuple

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore


# ============================================================
# SETTINGS
# ============================================================

QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "divergent_children"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieve a larger candidate pool first.
RETRIEVAL_K = 20

# A chunk must score >= this to be considered.
SIMILARITY_THRESHOLD = 0.35

# Maximum number of chunks eventually returned to the LLM.
FINAL_K = 5

# Minimum combined score (semantic + lexical + concept) to keep a chunk. 
MIN_COMBINED_SCORE = 0.30


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

print("Embedding model loaded.")


# ============================================================
# LOAD QDRANT
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
# QUESTION TYPE TERMS
# ============================================================

QUESTION_TERMS = {
    "origin": [
        "born",
        "birth",
        "born into",
        "original faction",
        "original family",
        "family",
        "upbringing",
        "grew up",
        "before choosing",
        "before transferring",
        "childhood",
    ],

    "motivation": [
        "why",
        "reason",
        "choose",
        "chose",
        "decision",
        "motivation",
        "wanted",
        "because",
    ],

    "character": [
        "character",
        "personality",
        "traits",
        "relationship",
        "feelings",
        "thinks",
        "believes",
    ],

    "event": [
        "what happened",
        "when",
        "where",
        "event",
        "scene",
        "incident",
    ],
}


# ============================================================
# RETRIEVAL QUERY EXPANSION
# ============================================================

def build_expansion_terms(
    question_type: str | None = None,
) -> list[str]:
    """
    Return domain terms associated with a question subtype.

    IMPORTANT: these are intentionally NOT concatenated onto the
    user's query before embedding. Jamming ~15 keywords onto a
    single sentence and embedding the whole block as one vector
    blurs the embedding (it stops being "about" any one thing
    specifically) and *hurts* semantic recall rather than helping
    it. Instead these terms are embedded as their OWN short query
    in retrieve_documents(), and the two result sets are merged.
    """

    if not question_type:
        return []

    expansion_terms = list(
        QUESTION_TERMS.get(
            question_type,
            [],
        )
    )

    if question_type == "origin":
        expansion_terms.extend(
            [
                "born into faction",
                "original faction",
                "family faction",
                "faction before transfer",
                "where Tris grew up",
                "Abnegation",
                "Dauntless",
            ]
        )

    elif question_type == "motivation":
        expansion_terms.extend(
            [
                "reason",
                "decision",
                "why",
                "motivation",
                "choice",
            ]
        )

    unique_terms = []

    for term in expansion_terms:
        if term not in unique_terms:
            unique_terms.append(term)

    return unique_terms


def build_concept_query(
    question_type: str | None = None,
) -> str | None:
    """
    Build a short, standalone query from expansion terms, meant
    to be embedded on its own (NOT appended to the user query).

    Returns None when there are no terms for this question_type,
    so callers can skip the second search entirely.
    """

    terms = build_expansion_terms(question_type)

    if not terms:
        return None

    return " ".join(terms)


# ============================================================
# NOISE / FRONT-MATTER FILTER
# ============================================================

# Chunks from copyright pages, ISBN blocks, publisher addresses,
# etc. sometimes end up in the vector store as ordinary chunks
# and can occasionally out-rank real narrative passages. These
# markers are distinctive enough that they essentially never
# appear in the novel's actual prose, so a single hit is enough
# to drop the chunk.
FRONT_MATTER_MARKERS = [
    "isbn",
    "library of congress",
    "harpercollins",
    "about the publisher",
    "epub edition",
    "all rights reserved",
    "cataloging-in-publication",
]


def is_front_matter(text: str) -> bool:

    lowered = text.lower()

    for marker in FRONT_MATTER_MARKERS:
        if marker in lowered:
            return True

    return False


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(text: str) -> str:
    return " ".join(
        text.lower().split()
    )


def keyword_score(
    query: str,
    text: str,
) -> float:
    """
    Simple lexical overlap score.

    This is deliberately weaker than semantic similarity.
    """

    query_words = {
        word.strip(".,!?;:\"'()[]")
        for word in normalize_text(query).split()
        if len(word) > 2
    }

    text_words = set(
        normalize_text(text).split()
    )

    if not query_words:
        return 0.0

    overlap = query_words.intersection(
        text_words
    )

    return len(overlap) / len(query_words)


def concept_score(
    query: str,
    text: str,
    question_type: str | None = None,
) -> float:
    """
    Detect whether the retrieved chunk contains concepts
    associated with the type of question.

    This is a small reranking signal, not an answer.
    """

    text_lower = normalize_text(text)

    score = 0.0

    if question_type == "origin":

        origin_terms = [
            "born",
            "family",
            "abnegation",
            "dauntless-born",
            "transfer",
            "transfers",
            "grew up",
            "mother",
            "father",
            "parents",
            "faction",
            "left them",
            "home",
        ]

        for term in origin_terms:
            if term in text_lower:
                score += 1.0

    elif question_type == "motivation":

        motivation_terms = [
            "because",
            "reason",
            "choose",
            "chose",
            "choice",
            "wanted",
            "protect",
            "courage",
            "decision",
        ]

        for term in motivation_terms:
            if term in text_lower:
                score += 1.0

    elif question_type == "character":

        character_terms = [
            "feel",
            "felt",
            "think",
            "thought",
            "believe",
            "relationship",
            "love",
            "hate",
            "afraid",
        ]

        for term in character_terms:
            if term in text_lower:
                score += 1.0

    elif question_type == "event":

        event_terms = [
            "said",
            "says",
            "happened",
            "walked",
            "went",
            "saw",
            "looked",
            "morning",
            "night",
        ]

        for term in event_terms:
            if term in text_lower:
                score += 1.0

    # Normalize roughly to 0-1.
    return min(score / 4.0, 1.0)


# ============================================================
# CHAPTER PRIORITY
# ============================================================

def chapter_number(chapter: str) -> int | None:
    """
    Extract chapter number when metadata contains something
    like 'Chapter SIXTEEN'.

    Returns None when it cannot be determined.
    """

    roman_values = {
        "ONE": 1,
        "TWO": 2,
        "THREE": 3,
        "FOUR": 4,
        "FIVE": 5,
        "SIX": 6,
        "SEVEN": 7,
        "EIGHT": 8,
        "NINE": 9,
        "TEN": 10,
        "ELEVEN": 11,
        "TWELVE": 12,
        "THIRTEEN": 13,
        "FOURTEEN": 14,
        "FIFTEEN": 15,
        "SIXTEEN": 16,
        "SEVENTEEN": 17,
        "EIGHTEEN": 18,
        "NINETEEN": 19,
        "TWENTY": 20,
        "THIRTY": 30,
    }

    if not chapter:
        return None

    words = chapter.upper().split()

    if not words:
        return None

    last_word = words[-1]

    return roman_values.get(last_word)


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    query: str,
    k: int = RETRIEVAL_K,
    question_type: str | None = None,
):
    """
    Retrieve a large candidate pool from Qdrant.

    Runs the user's (clean, unmodified) query as its own search,
    and — if a question_type is known — runs a second, separate
    search using only the domain-concept terms for that subtype.
    The two result sets are merged and deduped, keeping the best
    score per chunk. This widens recall without blurring either
    query's embedding.

    Retrieval itself is semantic.
    Reranking happens afterward.
    """

    primary_results = vector_db.similarity_search_with_score(
        query,
        k=k,
    )

    combined_results = list(primary_results)

    concept_query = build_concept_query(question_type)

    if concept_query:

        concept_results = vector_db.similarity_search_with_score(
            concept_query,
            k=k,
        )

        combined_results.extend(concept_results)

    # ----------------------------------------------------------
    # Dedupe by chunk content, keeping the higher of the two
    # scores when a chunk was returned by both searches.
    # ----------------------------------------------------------

    deduped = {}

    for doc, score in combined_results:

        key = doc.page_content

        if key not in deduped or score > deduped[key][1]:
            deduped[key] = (doc, score)

    return list(deduped.values())


# ============================================================
# RERANK RESULTS
# ============================================================

def rerank_results(
    query: str,
    results,
    question_type: str | None = None,
):
    """
    Rerank retrieved chunks using multiple signals.

    Semantic similarity remains the strongest signal.
    Lexical/concept matches are secondary signals.
    """

    reranked = []

    for doc, score in results:

        text = doc.page_content

        # ----------------------------------------------------
        # Semantic score
        # ----------------------------------------------------

        # Qdrant score (cosine collection):
        # HIGHER = better match, already ~0-1.
        # Just clamp defensively for floating point noise.
        semantic_score = max(
            0.0,
            min(1.0, score),
        )

        # ----------------------------------------------------
        # Keyword score
        # ----------------------------------------------------

        lexical_score = keyword_score(
            query,
            text,
        )

        # ----------------------------------------------------
        # Question concept score
        # ----------------------------------------------------

        concepts = concept_score(
            query,
            text,
            question_type,
        )

        # ----------------------------------------------------
        # Combined score
        # ----------------------------------------------------

        combined_score = (
            semantic_score * 0.65
            + lexical_score * 0.15
            + concepts * 0.20
        )

        reranked.append(
            (
                doc,
                score,
                lexical_score,
                concepts,
                combined_score,
            )
        )

    reranked.sort(
        key=lambda item: item[4],
        reverse=True,
    )

    return reranked


# ============================================================
# GET RELEVANT CONTEXT
# ============================================================

def get_relevant_context(
    query: str,
    k: int = FINAL_K,
    question_type: str | None = None,
):
    """
    Retrieve and rerank document context.

    Returns an empty string when no sufficiently strong
    document evidence is found.
    """

    results = retrieve_documents(
        query=query,
        k=RETRIEVAL_K,
        question_type=question_type,
    )

    if not results:
        return ""

    reranked = rerank_results(
        query=query,
        results=results,
        question_type=question_type,
    )

    context_parts = []

    for i, (
        doc,
        score,
        lexical_score,
        concepts,
        combined_score,
    ) in enumerate(
        reranked,
        1,
    ):

        # ----------------------------------------------------
        # Reject weak semantic matches
        # A LOW score means a POOR match now.
        # ----------------------------------------------------

        if score < SIMILARITY_THRESHOLD:
            continue

        # ----------------------------------------------------
        # Reject extremely weak combined matches
        # ----------------------------------------------------

        if combined_score < MIN_COMBINED_SCORE:
            continue

        # ----------------------------------------------------
        # Reject copyright/ISBN/publisher front-matter chunks
        # ----------------------------------------------------

        if is_front_matter(doc.page_content):
            continue

        # ----------------------------------------------------
        # Metadata
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
        # Context
        # ----------------------------------------------------

        context_parts.append(
            f"[Source {len(context_parts) + 1}]\n"
            f"Book: {book}\n"
            f"Chapter: {chapter}\n"
            f"Source file: {source}\n\n"
            f"{doc.page_content}"
        )

        if len(context_parts) >= k:
            break

    if not context_parts:
        return ""

    return "\n\n".join(
        context_parts
    )


# ============================================================
# DEBUG RETRIEVAL
# ============================================================

def debug_retrieval(
    query: str,
    question_type: str | None = None,
    k: int = 20,
):
    """
    Print retrieval and reranking information.

    Useful when tuning the RAG system.
    """

    print("\n" + "=" * 70)
    print("RETRIEVAL DEBUG")
    print("=" * 70)

    print(f"\nQuestion:")
    print(query)

    print(f"\nQuestion type:")
    print(question_type)

    concept_query = build_concept_query(question_type)

    print("\nPrimary query (embedded as-is):")
    print(query)

    print("\nConcept query (embedded separately, or None):")
    print(concept_query)

    results = retrieve_documents(
        query=query,
        k=k,
        question_type=question_type,
    )

    reranked = rerank_results(
        query=query,
        results=results,
        question_type=question_type,
    )

    print("\n" + "=" * 70)
    print("FINAL RETRIEVAL RESULTS")
    print("=" * 70)

    for i, (
        doc,
        score,
        lexical_score,
        concepts,
        combined_score,
    ) in enumerate(
        reranked,
        1,
    ):

        chapter = doc.metadata.get(
            "chapter",
            "Unknown",
        )

        print(
            f"\nResult {i}"
            f"\nRaw similarity score: {score:.4f}"
            f"\nKeyword score: {lexical_score:.4f}"
            f"\nConcept score: {concepts:.4f}"
            f"\nCombined score: {combined_score:.4f}"
            f"\nChapter: {chapter}"
        )

        print("\nContent preview:")
        print(
            doc.page_content[:500]
        )