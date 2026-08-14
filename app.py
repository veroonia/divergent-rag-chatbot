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

CHAT_MODEL = "openai/gpt-oss-20b"

TEMPERATURE = 0.7


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a helpful, accurate, and concise AI assistant.

You can answer two types of questions.

1. GENERAL QUESTIONS

If the user asks a general knowledge question that is
unrelated to the document, answer normally using your
general knowledge.

2. DOCUMENT QUESTIONS

If the user asks about the document, use ONLY the
retrieved document context provided in the current request.

IMPORTANT RULES FOR DOCUMENT QUESTIONS:

- Do not use general knowledge to fill missing information.
- Do not invent facts from the document.
- Do not assume that a retrieved passage answers the question.
- Only state information that is supported by the retrieved
  document context.
- If the retrieved context does not contain enough information
  to answer the question, explicitly say that the available
  document context does not contain enough information.
- Never claim that something appears in the document unless
  the retrieved context supports it.
- Be concise and directly answer the user's question.

For general questions, completely ignore the document context.
"""


# ============================================================
# DOCUMENT ROUTING PROMPT
# ============================================================

ROUTER_PROMPT = """
You are a question router for a chatbot that has access to
the novel "Divergent".

Your job is to classify the user's question into exactly
ONE of these two categories:

DOCUMENT
GENERAL

Return ONLY the word:

DOCUMENT

or:

GENERAL


Choose DOCUMENT when the user is asking about:

- Divergent
- Tris
- Beatrice Prior
- Four
- Tobias
- Christina
- Caleb
- Peter
- Dauntless
- Abnegation
- Erudite
- Amity
- Candor
- factions
- initiation
- the characters
- events in the novel
- chapters
- scenes
- relationships between characters
- anything that clearly refers to the contents of the book


Choose GENERAL when the question is unrelated to the novel.

Examples:

"What is the capital of France?"
GENERAL

"Give me five places to visit in Cairo."
GENERAL

"How does photosynthesis work?"
GENERAL

"What faction was Tris born into?"
DOCUMENT

"Why does Tris choose Dauntless?"
DOCUMENT

"Who is Four?"
DOCUMENT

"What happens during Tris's initiation?"
DOCUMENT

"Who is Tris's mother?"
DOCUMENT


Important:

If the question clearly refers to a character, event,
location, faction, chapter, or other element of Divergent,
classify it as DOCUMENT.

Return ONLY DOCUMENT or GENERAL.
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
# QUESTION ROUTER
# ============================================================

def classify_question(prompt):
    """
    Decide whether the question is about Divergent
    or is a general question.

    Returns:

        "DOCUMENT"

    or:

        "GENERAL"
    """

    client = get_groq_client()

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": ROUTER_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        max_tokens=10,
    )

    result = (
        response.choices[0]
        .message
        .content
        .strip()
        .upper()
    )

    # --------------------------------------------------------
    # Keep only the expected classifications
    # --------------------------------------------------------

    if "DOCUMENT" in result:
        return "DOCUMENT"

    if "GENERAL" in result:
        return "GENERAL"

    # --------------------------------------------------------
    # Safe fallback
    # --------------------------------------------------------

    print(
        f"Unexpected router result: {result}"
    )

    return "GENERAL"


# ============================================================
# CSS
# ============================================================

def load_css():

    css_path = (
        Path(__file__)
        .with_name("styles")
        .joinpath("app.css")
    )

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
        <p>Ask me anything about your documents.</p>
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

        messages_for_button = (
            load_history()
        )

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
# NATIVE STREAMLIT CHAT INPUT
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

    # --------------------------------------------------------
    # Check API key
    # --------------------------------------------------------

    if not GROQ_API_KEY:

        st.error(
            "No GROQ_API_KEY configured in your .env file."
        )

        st.stop()


    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

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
    # STEP 1 — CLASSIFY QUESTION
    # ========================================================

    try:

        question_type = classify_question(
            prompt
        )

    except Exception as error:

        print(
            f"Question router error: "
            f"{type(error).__name__}: {error}"
        )

        # Safe fallback:
        # if routing fails, treat the question as general
        # rather than automatically injecting book context.
        question_type = "GENERAL"


    print(
        f"Question type: {question_type}"
    )


    # ========================================================
    # STEP 2 — RETRIEVE DOCUMENT CONTEXT ONLY IF NEEDED
    # ========================================================

    rag_context = ""

    if question_type == "DOCUMENT":

        rag_context = get_relevant_context(
            prompt,
            k=5,
        )


    # ========================================================
    # OPTIONAL DEBUG DISPLAY
    # ========================================================

    if question_type == "DOCUMENT":

        with st.expander(
            "🔎 Retrieved Document Context"
        ):

            if rag_context:

                st.write(
                    rag_context
                )

            else:

                st.write(
                    "NO RELEVANT DOCUMENT CONTEXT WAS FOUND."
                )


    # ========================================================
    # STEP 3 — BUILD GROQ PROMPT
    # ========================================================

    if question_type == "DOCUMENT":

        # ----------------------------------------------------
        # DOCUMENT QUESTION
        # ----------------------------------------------------

        if rag_context:

            rag_prompt = f"""
Answer the user's question using ONLY the retrieved
document context below.

The question is about the document.

IMPORTANT:

- Use only information supported by the retrieved context.
- Do not use your general knowledge to fill missing information.
- Do not invent facts.
- Do not assume that every retrieved passage is relevant.
- If the context does not contain enough information to answer
  the question, explicitly say that the available document
  context does not contain enough information.
- Do not claim that something happened in the book unless the
  retrieved context supports it.

--- RETRIEVED DOCUMENT CONTEXT ---

{rag_context}

--- END RETRIEVED DOCUMENT CONTEXT ---

USER QUESTION:

{prompt}
"""

        else:

            rag_prompt = f"""
The user is asking a question about the document.

However, the retrieval system did not find sufficiently
relevant document context.

Answer ONLY with a concise explanation that the available
document context does not contain enough information to
answer the question.

Do NOT answer using general knowledge.

USER QUESTION:

{prompt}
"""

    else:

        # ----------------------------------------------------
        # GENERAL QUESTION
        # ----------------------------------------------------

        rag_prompt = f"""
Answer the user's question normally using your general
knowledge.

This is a GENERAL question and is not about the document.

IMPORTANT:

- Do not use or mention the document.
- Do not mention retrieval.
- Do not mention RAG.
- Answer naturally and directly.

USER QUESTION:

{prompt}
"""


    # ========================================================
    # BUILD API MESSAGE HISTORY
    # ========================================================

    api_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]


    # --------------------------------------------------------
    # Include previous conversation
    # --------------------------------------------------------

    for message in messages[:-1]:

        api_messages.append(
            {
                "role": message["role"],
                "content": message["content"],
            }
        )


    # --------------------------------------------------------
    # Add current question
    # --------------------------------------------------------

    api_messages.append(
        {
            "role": "user",
            "content": rag_prompt,
        }
    )


    # ========================================================
    # GENERATE ASSISTANT RESPONSE
    # ========================================================

    full_response = ""

    response_succeeded = False

    error_message = None


    st.markdown(
        open_assistant_row(),
        unsafe_allow_html=True,
    )

    placeholder = st.empty()


    try:

        client = get_groq_client()

        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=api_messages,
            temperature=TEMPERATURE,
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
                "**Prompt sent to AI:**"
            )

            st.code(
                prompt
            )

            st.write(
                "**Messages sent:**"
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