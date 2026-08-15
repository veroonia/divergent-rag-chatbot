import json
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from rag_retriever import get_relevant_context

from html_utils import (
    close_assistant_row,
    open_assistant_row,
    render_message_actions,
    render_user_message,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="✨",
    layout="centered",
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GROQ_API_KEY = os.environ.get(
    "GROQ_API_KEY",
    "",
).strip()


# ============================================================
# SETTINGS
# ============================================================

HISTORY_FILE = "chat_history.json"

HISTORY_LOCK = threading.RLock()

# Model used for routing / query rewriting (cheap, simple tasks).
CHAT_MODEL = "openai/gpt-oss-20b"

# Model used for the actual final answer.
# Document QA requires careful multi-passage, multi-hop reading
# (e.g. tracking who said what about whom across several family
# members) — this is a genuine reasoning-capacity requirement,
# not just a prompt-wording issue. openai/gpt-oss-20b was
# repeatedly misattributing quotes even with correct retrieval
# and explicit instructions, so this is bumped to the much
# larger 120b model. llama-3.3-70b-versatile is deprecated on
# Groq as of mid-2026; gpt-oss-120b is the recommended successor.
#
# NOTE: gpt-oss-120b has built-in reasoning. Test that streamed
# output is clean prose and doesn't leak raw reasoning tokens
# into what the user sees — if it does, check Groq's docs for a
# reasoning_effort / reasoning_format parameter to suppress it.
ANSWER_MODEL = "openai/gpt-oss-120b"

# General conversation can be a bit more creative/loose.
GENERAL_TEMPERATURE = 0.7

# Document answers must be a careful, literal reading of the
# retrieved passages — not creative. Keep this at 0 unless you
# have a specific reason to raise it.
DOCUMENT_TEMPERATURE = 0.0

# Maximum answer length.
# This is intentionally much larger than the router's limit.
MAX_ANSWER_TOKENS = 1800

# Number of retrieved chunks.
RETRIEVAL_K = 8


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a helpful, accurate AI assistant.

You can answer two different kinds of questions:

1. GENERAL QUESTIONS
2. DOCUMENT QUESTIONS ABOUT THE NOVEL "DIVERGENT"

============================================================
GENERAL QUESTIONS
============================================================

For general questions, answer normally using your general
knowledge.

Do NOT use the Divergent document context for general questions.

Do NOT mention:
- the document
- retrieval
- RAG
- Qdrant
- document context
- routing

Answer naturally as a normal general-purpose AI assistant.

Give enough detail to properly answer the question.

For simple factual questions, a short answer is fine.

For broader questions, normally provide:
- a direct answer first
- useful explanation
- examples or relevant details when appropriate

Do not unnecessarily restrict general answers to one or two
sentences.

============================================================
DOCUMENT QUESTIONS
============================================================

For questions about Divergent, use ONLY the retrieved document
context supplied in the current request.

Do not use general knowledge to fill gaps.

Do not invent information.

NOVEL-SPECIFIC VOCABULARY:

The book uses specific terms to describe a character's faction
history. Read these literally as direct, explicit statements of
fact — not as something requiring extra inference:

- A character called a "[Faction] transfer" (e.g. "an Abnegation
  transfer," "a Dauntless transfer," "the only Abnegation
  transfer") was BORN AND RAISED in that named faction, then
  chose to leave it for a new faction at the Choosing Ceremony.
  This directly states their birth/origin faction. Treat it the
  same as if the text said "born into [Faction]."
- A character called "[Faction]-born" (e.g. "Dauntless-born")
  was born and raised in that faction and did NOT transfer away
  from it.
- The "Choosing Ceremony" is the event at age sixteen where each
  person selects the faction they will belong to for the rest of
  their life — which may or may not match the faction they were
  born into.

When a retrieved passage uses this vocabulary, use it directly
to answer origin/background questions. Reading this vocabulary
literally is reading the text, not making an unsupported
inference.

Do not assume that a retrieved passage answers the question.

Only state information that is actually supported by the
retrieved context.

Before using any quote or paraphrasing any passage, identify
exactly WHO is speaking and WHO or WHAT the passage is actually
describing. Characters frequently describe someone else's
faction, history, background, or actions — never assume a
passage describes the person the question is about just because
it appears in a relevant retrieved chunk. Trace pronouns and
dialogue attribution carefully before drawing a conclusion.

If several retrieved passages address the question, prioritize
the most explicit, directly-stated passage over an ambiguous
line of dialogue that requires guessing who or what is being
discussed. Do not build an answer around the ambiguous line if
a clearer passage is available.

If passages appear to conflict, do not silently pick one and
present it as certain. State what the clearest, most explicit
passages say, and only mention the ambiguous or conflicting line
if it's needed to explain the discrepancy.

If the retrieved context does not contain enough information,
say clearly:

"The available document context does not contain enough
information to answer that question."

Do not answer the question from your general knowledge when
the retrieved context is insufficient.

When the context does support the answer, answer directly and
naturally.

You may combine information from multiple retrieved passages
when they collectively support the answer.

============================================================
IMPORTANT
============================================================

Always determine the response type from the instructions in
the current request.

If the current request says GENERAL, ignore any document
context completely.

If the current request says DOCUMENT, use only the supplied
retrieved context.
"""


# ============================================================
# ROUTER PROMPT
# ============================================================

ROUTER_PROMPT = """
You are a routing classifier for a chatbot that has access to
the complete novel "Divergent" through a retrieval system.

Your job is to classify the CURRENT user question into exactly
one category:

DOCUMENT
or
GENERAL

============================================================
DOCUMENT
============================================================

Choose DOCUMENT when the user is asking for information that
should come from the novel Divergent.

This includes questions about:

- Tris
- Beatrice Prior
- Tobias
- Four
- Christina
- Caleb
- Peter
- Jeanine
- Eric
- Al
- Will
- Uriah
- characters
- relationships
- family
- factions
- Abnegation
- Dauntless
- Erudite
- Amity
- Candor
- factionless
- initiation
- aptitude tests
- choosing ceremony
- fear landscapes
- Divergence
- plot
- events
- scenes
- chapters
- dialogue
- character motivations
- character decisions
- locations in the novel
- anything that happened in the story

Also classify as DOCUMENT when the question is a follow-up to
an earlier Divergent question.

Examples:

User:
Who is Tris?
DOCUMENT

User:
Why did she choose Dauntless?
DOCUMENT

User:
What about her mother?
DOCUMENT

User:
Why did he do that?
DOCUMENT

User:
What happened next?
DOCUMENT

============================================================
GENERAL
============================================================

Choose GENERAL when the question can be answered independently
of Divergent.

Examples:

What is the capital of France?
GENERAL

Tell me more about Paris.
GENERAL

What are the top places to visit in Paris?
GENERAL

How does photosynthesis work?
GENERAL

Explain machine learning.
GENERAL

Write Python code for a calculator.
GENERAL

============================================================
IMPORTANT
============================================================

Use the recent conversation to resolve pronouns and references.

Words such as:

she
he
her
him
they
it
that
there
why did she
what happened next
what about him

should be classified as DOCUMENT if the recent conversation
clearly established that they refer to a Divergent character,
event, or concept.

However, do NOT classify a question as DOCUMENT merely because
an unrelated earlier conversation happened to mention a Divergent
term.

Return ONLY:

DOCUMENT

or:

GENERAL
"""


# ============================================================
# HISTORY FUNCTIONS
# ============================================================

def _is_valid_history(data):
    """
    Validate the structure of saved chat history.
    """

    if not isinstance(data, list):
        return False

    for item in data:
        if not isinstance(item, dict):
            return False

        if item.get("role") not in {
            "user",
            "assistant",
        }:
            return False

        if not isinstance(
            item.get("content"),
            str,
        ):
            return False

    return True


def load_history():
    """
    Load chat history from JSON.
    """

    with HISTORY_LOCK:

        if not os.path.exists(HISTORY_FILE):
            return []

        try:

            with open(
                HISTORY_FILE,
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

            if _is_valid_history(data):
                return data

            return []

        except (
            json.JSONDecodeError,
            OSError,
        ):

            return []


def save_history(messages):
    """
    Save chat history safely using a temporary file.
    """

    with HISTORY_LOCK:

        directory = (
            os.path.dirname(
                os.path.abspath(
                    HISTORY_FILE
                )
            )
            or "."
        )

        temp_path = None

        try:

            fd, temp_path = tempfile.mkstemp(
                prefix="chat_history_",
                suffix=".tmp",
                dir=directory,
            )

            with os.fdopen(
                fd,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    messages,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            os.replace(
                temp_path,
                HISTORY_FILE,
            )

        except OSError as error:

            print(
                f"History save error: {error}"
            )

            if (
                temp_path
                and os.path.exists(temp_path)
            ):

                try:
                    os.remove(temp_path)

                except OSError:
                    pass


def append_message(
    role,
    content,
):

    with HISTORY_LOCK:

        messages = load_history()

        messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().strftime(
                    "%H:%M"
                ),
            }
        )

        save_history(messages)

        return messages


def clear_history():

    with HISTORY_LOCK:

        try:

            if os.path.exists(
                HISTORY_FILE
            ):

                os.remove(
                    HISTORY_FILE
                )

        except OSError as error:

            st.error(
                f"Could not clear the chat: {error}"
            )


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client():

    if not GROQ_API_KEY:

        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )

    return Groq(
        api_key=GROQ_API_KEY
    )


# ============================================================
# CONVERSATION FORMATTER
# ============================================================

def format_recent_conversation(
    conversation_history,
    limit=8,
):
    """
    Format only the most recent messages for routing and
    retrieval-query rewriting.
    """

    if not conversation_history:
        return "No previous conversation."

    recent_messages = conversation_history[-limit:]

    parts = []

    for message in recent_messages:

        role = message.get(
            "role",
            "",
        ).upper()

        content = message.get(
            "content",
            "",
        )

        if not content:
            continue

        parts.append(
            f"{role}: {content}"
        )

    if not parts:
        return "No previous conversation."

    return "\n".join(parts)


# ============================================================
# QUESTION ROUTER
# ============================================================

def classify_question(
    prompt,
    conversation_history=None,
):
    """
    Determine whether a question is about Divergent or general.

    Uses:
    1. Strong Divergent-specific terms.
    2. Recent conversation context.
    3. LLM classification for ambiguous cases.
    """

    if not prompt or not prompt.strip():
        return "GENERAL"

    prompt_lower = prompt.lower().strip()

    # ========================================================
    # STRONG DOCUMENT TERMS
    # ========================================================

    divergent_terms = [
        "divergent",
        "tris",
        "beatrice",
        "beatrice prior",
        "tobias",
        "four",
        "christina",
        "caleb",
        "peter",
        "jeanine",
        "eric",
        "al",
        "will",
        "uriah",
        "dauntless",
        "abnegation",
        "erudite",
        "amity",
        "candor",
        "faction",
        "factions",
        "factionless",
        "initiation",
        "initiate",
        "aptitude test",
        "choosing ceremony",
        "fear landscape",
        "fear simulation",
        "divergence",
        "divergent serum",
        "dauntless-born",
    ]

    for term in divergent_terms:

        if term in prompt_lower:

            print(
                f"Document keyword detected: {term}"
            )

            return "DOCUMENT"

    # ========================================================
    # RECENT CONVERSATION
    # ========================================================

    conversation_text = format_recent_conversation(
        conversation_history,
        limit=8,
    ).lower()

    recent_document_signal = False

    for term in divergent_terms:

        if term in conversation_text:

            recent_document_signal = True
            break

    # ========================================================
    # LLM ROUTER
    # ========================================================

    try:

        client = get_groq_client()

        router_input = f"""
RECENT CONVERSATION:

{format_recent_conversation(
    conversation_history,
    limit=8,
)}

CURRENT USER QUESTION:

{prompt}

Classify the CURRENT question.

Return exactly one word:

DOCUMENT

or

GENERAL
"""

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": ROUTER_PROMPT,
                },
                {
                    "role": "user",
                    "content": router_input,
                },
            ],
            temperature=0,
            max_tokens=20,
        )

        result = (
            response.choices[0]
            .message
            .content
            .strip()
            .upper()
        )

        print(
            f"LLM router result: {result}"
        )

        if result == "DOCUMENT":
            return "DOCUMENT"

        if result == "GENERAL":
            return "GENERAL"

        # Sometimes models return extra text.
        if "DOCUMENT" in result:
            return "DOCUMENT"

        if "GENERAL" in result:
            return "GENERAL"

    except Exception as error:

        print(
            f"Question router error: "
            f"{type(error).__name__}: {error}"
        )

    # ========================================================
    # SAFE FALLBACK
    # ========================================================

    if recent_document_signal:

        return "DOCUMENT"

    return "GENERAL"


# ============================================================
# DOCUMENT QUESTION SUBTYPE DETECTOR
# ============================================================

# rag_retriever.py has term-expansion and reranking logic keyed
# off a question subtype ("origin", "motivation", "character",
# "event") but nothing was ever detecting or passing that value.
# This is a cheap keyword heuristic — it only needs to be roughly
# right, since it feeds query expansion and a secondary rerank
# signal, not the final answer.

ORIGIN_SIGNALS = [
    "born",
    "birth",
    "family",
    "grew up",
    "growing up",
    "upbringing",
    "originally",
    "original faction",
    "childhood",
    "raised",
]

MOTIVATION_SIGNALS = [
    "why",
    "reason",
    "decision",
    "chose",
    "choose",
    "motivation",
    "wanted",
]

CHARACTER_SIGNALS = [
    "feel",
    "feels",
    "felt",
    "believe",
    "relationship",
    "personality",
    "trait",
    "think",
    "thinks",
    "thought",
]

EVENT_SIGNALS = [
    "what happened",
    "when did",
    "where did",
    "event",
    "scene",
    "incident",
]


def detect_question_subtype(prompt):
    """
    Best-effort keyword classification into one of the subtypes
    rag_retriever.py already knows how to expand/rerank for.

    Returns None when nothing matches — get_relevant_context
    handles that fine, it just skips the extra expansion.
    """

    prompt_lower = prompt.lower()

    for term in ORIGIN_SIGNALS:
        if term in prompt_lower:
            return "origin"

    for term in MOTIVATION_SIGNALS:
        if term in prompt_lower:
            return "motivation"

    for term in CHARACTER_SIGNALS:
        if term in prompt_lower:
            return "character"

    for term in EVENT_SIGNALS:
        if term in prompt_lower:
            return "event"

    return None


# ============================================================
# RETRIEVAL QUERY REWRITER
# ============================================================

def create_retrieval_query(
    prompt,
    conversation_history=None,
):
    """
    Convert a natural user question into a retrieval-friendly
    query.

    This is especially useful for questions such as:

        "what faction was Tris born into?"

    which should become something closer to:

        "Tris Beatrice Prior original birth faction
        family faction before choosing Dauntless"

    The rewritten query is ONLY used for vector retrieval.
    The original user question is still sent to the final model.
    """

    if not prompt:
        return prompt

    try:

        client = get_groq_client()

        conversation_text = format_recent_conversation(
            conversation_history,
            limit=6,
        )

        rewrite_prompt = f"""
You are preparing a search query for a vector database
containing the complete novel Divergent.

Rewrite the user's question into a concise retrieval query.

Your goal is to find the exact passages that answer the question.

Preserve:
- character names
- relationships
- events
- locations
- factions
- chapter concepts
- motivations
- temporal clues
- origin/background information

Resolve pronouns using the conversation when possible.

For example:

User question:
"What faction was Tris born into?"

Good retrieval query:
"Tris Beatrice Prior original faction birth faction
family faction before transferring to Dauntless"

Another example:

User question:
"Why did she choose it?"

If the conversation shows that "she" means Tris and "it"
means Dauntless, produce a query containing those explicit terms.

Do NOT answer the question.

Return ONLY the retrieval query.

RECENT CONVERSATION:
{conversation_text}

USER QUESTION:
{prompt}
"""

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite questions for semantic "
                        "document retrieval. Return only the "
                        "rewritten search query."
                    ),
                },
                {
                    "role": "user",
                    "content": rewrite_prompt,
                },
            ],
            temperature=0,
            max_tokens=150,
        )

        rewritten = (
            response.choices[0]
            .message
            .content
            .strip()
        )

        if rewritten:

            print(
                "Original retrieval query:"
            )

            print(prompt)

            print(
                "Rewritten retrieval query:"
            )

            print(rewritten)

            return rewritten

    except Exception as error:

        print(
            f"Retrieval query rewrite error: "
            f"{type(error).__name__}: {error}"
        )

    # Safe fallback.
    return prompt


# ============================================================
# CSS
# ============================================================

def load_css():

    css_path = (
        Path(__file__)
        .with_name("styles")
        .joinpath("app.css")
    )

    if css_path.exists():

        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


load_css()


# ============================================================
# HERO
# ============================================================

def render_hero():

    return """
    <div class="hero">
        <h1>How can I help you?</h1>
        <p>Ask me anything.</p>
    </div>
    """


def render_missing_key_note():

    return """
    <div class="missing-key-note">
        <strong>Groq API key not configured.</strong>
        <p>Add your GROQ_API_KEY to the .env file to start chatting.</p>
    </div>
    """


# ============================================================
# TOP BAR
# ============================================================

top_bar = st.container()

with top_bar:

    st.markdown(
        '<div class="topbar-wrap">',
        unsafe_allow_html=True,
    )

    left, spacer, right = st.columns(
        [5, 1, 1.2]
    )

    with left:

        st.markdown(
            """
            <div class="wordmark">
                Your Chatbot
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:

        messages_for_button = load_history()

        if st.button(
            "Clear",
            use_container_width=True,
            disabled=not messages_for_button,
        ):

            clear_history()

            st.rerun()

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# LOAD HISTORY
# ============================================================

messages = load_history()


# ============================================================
# HERO
# ============================================================

if not messages:

    st.markdown(
        render_hero(),
        unsafe_allow_html=True,
    )

    if not GROQ_API_KEY:

        st.markdown(
            render_missing_key_note(),
            unsafe_allow_html=True,
        )


# ============================================================
# RENDER CHAT HISTORY
# ============================================================

for message in messages:

    if message["role"] == "user":

        st.markdown(
            render_user_message(
                message["content"],
                message.get(
                    "timestamp",
                    "",
                ),
            ),
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            open_assistant_row(),
            unsafe_allow_html=True,
        )

        st.markdown(
            message["content"]
        )

        st.markdown(
            render_message_actions(),
            unsafe_allow_html=True,
        )

        st.markdown(
            close_assistant_row(),
            unsafe_allow_html=True,
        )


# ============================================================
# CHAT INPUT
# ============================================================

submission = st.chat_input(
    "Ask anything",
    key="main_chat_input",
)


# ============================================================
# EXTRACT PROMPT
# ============================================================

prompt = ""

if submission:

    if hasattr(
        submission,
        "text",
    ):

        prompt = (
            submission.text or ""
        ).strip()

    else:

        prompt = str(
            submission
        ).strip()


# ============================================================
# PROCESS USER QUESTION
# ============================================================

if prompt:

    # ========================================================
    # CHECK API KEY
    # ========================================================

    if not GROQ_API_KEY:

        st.error(
            "No GROQ_API_KEY configured in your .env file."
        )

        st.stop()

    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    messages = append_message(
        "user",
        prompt,
    )

    st.markdown(
        render_user_message(
            prompt,
            messages[-1].get(
                "timestamp",
                "",
            ),
        ),
        unsafe_allow_html=True,
    )

    # ========================================================
    # STEP 1 — CLASSIFY
    # ========================================================

    try:

        question_type = classify_question(
            prompt,
            conversation_history=messages[:-1],
        )

    except Exception as error:

        print(
            f"Question classification error: "
            f"{type(error).__name__}: {error}"
        )

        question_type = "GENERAL"

    print(
        f"Question type: {question_type}"
    )

    # ========================================================
    # STEP 2 — DOCUMENT RETRIEVAL
    # ========================================================

    rag_context = ""

    retrieval_query = prompt

    question_subtype = None

    if question_type == "DOCUMENT":

        # ----------------------------------------------------
        # Detect origin / motivation / character / event so
        # rag_retriever.py's term-expansion and reranking logic
        # for that subtype actually gets used.
        # ----------------------------------------------------

        question_subtype = detect_question_subtype(prompt)

        print(
            f"Question subtype: {question_subtype}"
        )

        # ----------------------------------------------------
        # Rewrite query specifically for retrieval.
        # ----------------------------------------------------

        retrieval_query = create_retrieval_query(
            prompt,
            conversation_history=messages[:-1],
        )

        # ----------------------------------------------------
        # Retrieve from Qdrant.
        # ----------------------------------------------------

        try:

            rag_context = get_relevant_context(
                retrieval_query,
                k=RETRIEVAL_K,
                question_type=question_subtype,
            )

        except Exception as error:

            print(
                f"Retrieval error: "
                f"{type(error).__name__}: {error}"
            )

            rag_context = ""

    # ========================================================
    # DEBUG INFORMATION
    # ========================================================

    if question_type == "DOCUMENT":

        with st.expander(
            "🔎 Retrieved Document Context"
        ):

            st.write(
                "**Original question:**"
            )

            st.write(prompt)

            st.write(
                "**Detected subtype:**"
            )

            st.write(
                question_subtype or "None"
            )

            st.write(
                "**Retrieval query:**"
            )

            st.write(retrieval_query)

            st.write(
                "**Context:**"
            )

            if rag_context:

                st.write(
                    rag_context
                )

            else:

                st.write(
                    "NO RELEVANT DOCUMENT CONTEXT WAS FOUND."
                )

    # ========================================================
    # STEP 3 — BUILD FINAL PROMPT
    # ========================================================

    if question_type == "DOCUMENT":

        if rag_context:

            final_prompt = f"""
The user is asking a question about the novel Divergent.

Answer the user's question using ONLY the retrieved document
context below.

IMPORTANT RULES:

- Use only information supported by the retrieved context.
- Do not use outside knowledge.
- Do not invent facts.
- Remember: "[Faction] transfer" means born/raised in that
  faction, then left it. "[Faction]-born" means born, raised,
  and stayed in that faction. Treat these terms as explicit
  statements of origin, not inferences.
- Do not assume that every retrieved passage is relevant.
- Carefully compare the passages and identify which ones
  actually answer the question.
- Before using any quote or paraphrasing any passage, identify
  exactly WHO is speaking and WHO or WHAT the passage is
  actually describing. Characters frequently describe someone
  else's faction, history, background, or actions — never
  assume a passage describes the person the question is about
  just because it was retrieved as relevant. Trace pronouns and
  dialogue attribution carefully before drawing a conclusion.
- If several passages address the question, prioritize the most
  explicit, directly-stated passage over an ambiguous line of
  dialogue that requires guessing who or what is being
  discussed. Do not build the answer around the ambiguous line
  if a clearer passage is available.
- You may combine information from multiple retrieved passages
  when they collectively support the answer.
- If passages appear to conflict, do not silently pick one and
  present it as certain. State what the clearest, most explicit
  passages say, and only mention the ambiguous or conflicting
  line if it's needed to explain the discrepancy.
- If the retrieved passages do not actually provide enough
  information to answer the question, say so explicitly.
- Do not pretend that an inference is explicitly stated in
  the book.
- Answer naturally and directly.
- Do not mention RAG, Qdrant, retrieval, embeddings, or the
  retrieval process.

RETRIEVED DOCUMENT CONTEXT:

{rag_context}

END RETRIEVED DOCUMENT CONTEXT.

USER QUESTION:

{prompt}
"""

        else:

            final_prompt = f"""
The user is asking about the novel Divergent.

No sufficiently relevant document context was retrieved.

Do NOT answer using general knowledge.

Respond only that the available document context does not
contain enough information to answer the question.

USER QUESTION:

{prompt}
"""

    else:

        final_prompt = f"""
This is a GENERAL question.

Answer it normally using your general knowledge.

IMPORTANT:

- Completely ignore any Divergent document.
- Do not use document context.
- Do not mention documents.
- Do not mention retrieval.
- Do not mention RAG.
- Do not mention this classification.
- Answer naturally and directly.
- Give enough detail to properly answer the question.
- Do not unnecessarily limit the answer to one or two sentences.
- For broad questions, provide a useful explanation and relevant
  details.
- For simple factual questions, remain concise.

USER QUESTION:

{prompt}
"""

    # ========================================================
    # STEP 4 — BUILD API MESSAGES
    # ========================================================

    api_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    # --------------------------------------------------------
    # Add recent conversation history.
    #
    # This allows the model to understand follow-up questions
    # without sending an unnecessarily huge history.
    # --------------------------------------------------------

    recent_messages = messages[:-1][-8:]

    for message in recent_messages:

        api_messages.append(
            {
                "role": message["role"],
                "content": message["content"],
            }
        )

    # --------------------------------------------------------
    # Current question / instructions.
    # --------------------------------------------------------

    api_messages.append(
        {
            "role": "user",
            "content": final_prompt,
        }
    )

    # ========================================================
    # STEP 5 — GENERATE RESPONSE
    # ========================================================

    full_response = ""

    response_succeeded = False

    error_message = None

    st.markdown(
        open_assistant_row(),
        unsafe_allow_html=True,
    )

    placeholder = st.empty()

    # --------------------------------------------------------
    # Document answers use a low, near-deterministic temperature
    # since careful literal reading matters more than variety.
    # General chat keeps the looser temperature.
    # --------------------------------------------------------

    answer_temperature = (
        DOCUMENT_TEMPERATURE
        if question_type == "DOCUMENT"
        else GENERAL_TEMPERATURE
    )

    try:

        client = get_groq_client()

        stream = client.chat.completions.create(
            model=ANSWER_MODEL,
            messages=api_messages,
            temperature=answer_temperature,
            max_tokens=MAX_ANSWER_TOKENS,
            stream=True,
        )

        # ----------------------------------------------------
        # Stream response
        # ----------------------------------------------------

        for chunk in stream:

            if not chunk.choices:
                continue

            delta = (
                chunk.choices[0]
                .delta
                .content
                or ""
            )

            if delta:

                full_response += delta

                placeholder.markdown(
                    full_response + "▌"
                )

        full_response = (
            full_response.strip()
        )

        # ----------------------------------------------------
        # Empty response protection
        # ----------------------------------------------------

        if not full_response:

            raise RuntimeError(
                "The model returned an empty response."
            )

        placeholder.markdown(
            full_response
        )

        st.markdown(
            render_message_actions(),
            unsafe_allow_html=True,
        )

        response_succeeded = True

    except Exception as error:

        placeholder.empty()

        error_message = str(error)

        print(
            f"Groq chat error: "
            f"{type(error).__name__}: {error}"
        )

    finally:

        st.markdown(
            close_assistant_row(),
            unsafe_allow_html=True,
        )

    # ========================================================
    # ERROR DISPLAY
    # ========================================================

    if error_message:

        st.error(
            f"AI Response failed: {error_message}"
        )

        with st.expander(
            "🔍 Debug Info"
        ):

            st.write(
                "**Question type:**"
            )

            st.code(
                question_type
            )

            st.write(
                "**Original question:**"
            )

            st.code(
                prompt
            )

            if question_type == "DOCUMENT":

                st.write(
                    "**Retrieval query:**"
                )

                st.code(
                    retrieval_query
                )

                st.write(
                    "**Retrieved context:**"
                )

                st.code(
                    rag_context
                )

            st.write(
                "**Messages sent to Groq:**"
            )

            st.json(
                api_messages
            )

    # ========================================================
    # SAVE ASSISTANT RESPONSE
    # ========================================================

    if response_succeeded:

        append_message(
            "assistant",
            full_response,
        )