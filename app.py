import json
import os
import time
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(page_title="Groq Chat", page_icon="⚡", layout="centered")

load_dotenv()
API_KEY = os.environ.get("GROQ_API_KEY", "")

# ---------------------------------------------------------
# Chat history persistence
# ---------------------------------------------------------
HISTORY_FILE = "chat_history.json"


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_history(messages):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)
    except OSError:
        pass


def clear_history():
    st.session_state.messages = []
    if os.path.exists(HISTORY_FILE):
        try:
            os.remove(HISTORY_FILE)
        except OSError:
            pass

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap');

    #MainMenu, footer, header {visibility: hidden;}
    div[data-testid="stToolbar"] {display: none;}
    div[data-testid="stDecoration"] {display: none;}

    .stApp {
        background: #f7f5f2;
        font-family: 'Inter', sans-serif;
        color: #191816;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 720px;
    }

    /* ---- top bar ---- */
    .wordmark {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.15rem;
        letter-spacing: -0.01em;
        color: #191816;
        padding-top: 0.5rem;
    }
    .wordmark span { color: #ff4d1c; }

    /* ---- hero (empty state) ---- */
    .hero { text-align: center; padding: 4.5rem 0 2rem 0; }
    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2.7rem;
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-bottom: 0.7rem;
        color: #191816;
    }
    .hero h1 .accent { color: #ff4d1c; }
    .hero p {
        color: #706c66;
        font-size: 1.02rem;
    }

    /* ---- chat bubbles ---- */
    div[data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.3rem 0;
    }
    div[data-testid="stChatMessageAvatarUser"] {
        background: #191816 !important;
    }
    div[data-testid="stChatMessageAvatarAssistant"] {
        background: #ff4d1c !important;
    }
    div[data-testid="stChatMessageContent"] {
        background: #ffffff;
        border: 1px solid #e7e3dc;
        border-radius: 14px;
        padding: 0.75rem 1.1rem;
        box-shadow: 0 1px 2px rgba(25, 24, 22, 0.03);
    }

    .speed-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #ff4d1c;
        opacity: 0.85;
        margin-top: 0.2rem;
        padding-left: 0.2rem;
    }

    /* ---- chat input pill ---- */
    div[data-testid="stChatInput"] {
        background: #ffffff;
        border: 1px solid #e7e3dc;
        border-radius: 999px;
        box-shadow: 0 8px 24px -14px rgba(25, 24, 22, 0.25);
    }
    div[data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: #191816 !important;
        padding: 0.85rem 1.2rem !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #a8a39a !important;
    }
    div[data-testid="stChatInput"] button {
        background: #ff4d1c !important;
        border-radius: 50% !important;
    }
    div[data-testid="stChatInput"] button svg {
        fill: #ffffff !important;
    }

    /* ---- header buttons ---- */
    .stButton button, div[data-testid="stPopover"] button {
        background: #ffffff;
        border: 1px solid #e7e3dc;
        color: #706c66;
        border-radius: 999px;
        padding: 0.35rem 1rem;
        font-size: 0.85rem;
    }
    .stButton button:hover, div[data-testid="stPopover"] button:hover {
        border-color: #ff4d1c;
        color: #ff4d1c;
    }

    .no-key-note {
        text-align: center;
        color: #a8a39a;
        font-size: 0.85rem;
        margin-top: -0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# State
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = load_history()
if "settings" not in st.session_state:
    st.session_state.settings = {
        "model": "openai/gpt-oss-20b",
        "temperature": 0.7,
        "system_prompt": "You are a helpful, concise assistant.",
    }


# Top bar: wordmark + clear chat (Pinned)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Pin the first block-container element to the top */
    div[data-testid="stVerticalBlock"] > div:has(.wordmark) {
        position: sticky;
        top: 0;
        z-index: 999;
        background: #f7f5f2;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

top_bar = st.container()
with top_bar:
    left, mid, right = st.columns([5, 1, 1.2])

    with left:
        st.markdown('<div class="wordmark">groq<span>chat</span> ⚡</div>', unsafe_allow_html=True)

    with right:
        if st.button("Clear", use_container_width=True, disabled=not st.session_state.messages):
            clear_history()
            st.rerun()

# ---------------------------------------------------------
# Hero (only when there's no conversation yet)
# ---------------------------------------------------------
if not st.session_state.messages:
    st.markdown(
        """
        <div class="hero">
            <h1>Ask at the <span class="accent">speed</span><br>of thought</h1>
            <p>Groq's LPU inference — answers as fast as you can read them</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not API_KEY:
        st.markdown(
            '<p class="no-key-note">No GROQ_API_KEY found — add it to a .env file '
            "in this folder to enable responses.</p>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------
# Render history
# ---------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "tok_s" in msg:
            st.markdown(f'<div class="speed-tag">{msg["tok_s"]:.0f} tok/s</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Input + streaming response
# ---------------------------------------------------------
prompt = st.chat_input("Ask anything")

if prompt:
    if not API_KEY:
        st.error("No Groq API key configured. Add GROQ_API_KEY to a .env file in this folder.")
        st.stop()

    settings = st.session_state.settings

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    api_messages = [{"role": "system", "content": settings["system_prompt"]}] + [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        start = time.time()
        try:
            client = Groq(api_key=API_KEY)
            stream = client.chat.completions.create(
                model=settings["model"],
                messages=api_messages,
                temperature=settings["temperature"],
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full_response += delta
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"⚠️ Error: {e}"
            placeholder.markdown(full_response)

        elapsed = max(time.time() - start, 0.001)
        tok_s = (len(full_response.split()) / elapsed) * 1.3  # rough tokens/sec estimate

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    save_history(st.session_state.messages)