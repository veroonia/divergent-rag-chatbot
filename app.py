import html
import json
import os
import re
import tempfile
import threading
import asyncio
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
# TTS disabled for now — uncomment when re-enabling Edge TTS voice responses
# import edge_tts

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="✨",
    layout="centered",
)

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()

HISTORY_FILE = "chat_history.json"
HISTORY_LOCK = threading.RLock()

# ---------------------------------------------------------
# Models / voice configuration
# ---------------------------------------------------------
CHAT_MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0.7
SYSTEM_PROMPT = "You are a helpful, concise assistant."

# Edge TTS voice configuration (disabled for now)
# EDGE_TTS_EN_VOICE = "en-US-JennyNeural"
# EDGE_TTS_AR_VOICE = "ar-EG-SalmaNeural"
# MAX_TTS_CHARS = 3000

# ---------------------------------------------------------
# Shared chat history
# ---------------------------------------------------------
def _is_valid_history(data):
    if not isinstance(data, list):
        return False

    for item in data:
        if not isinstance(item, dict):
            return False
        if item.get("role") not in {"user", "assistant"}:
            return False
        if not isinstance(item.get("content"), str):
            return False

    return True


def load_history():
    with HISTORY_LOCK:
        if not os.path.exists(HISTORY_FILE):
            return []

        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)

            return data if _is_valid_history(data) else []
        except (json.JSONDecodeError, OSError):
            return []


def save_history(messages):
    with HISTORY_LOCK:
        directory = os.path.dirname(os.path.abspath(HISTORY_FILE)) or "."
        temp_path = None

        try:
            fd, temp_path = tempfile.mkstemp(
                prefix="chat_history_",
                suffix=".tmp",
                dir=directory,
            )

            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(messages, file, ensure_ascii=False, indent=2)

            os.replace(temp_path, HISTORY_FILE)

        except OSError as error:
            print(f"History save error: {error}")
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


def append_message(role, content):
    with HISTORY_LOCK:
        messages = load_history()
        messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().strftime("%H:%M"),
            }
        )
        save_history(messages)
        return messages


def clear_history():
    with HISTORY_LOCK:
        try:
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
        except OSError as error:
            st.error(f"Could not clear the chat: {error}")


# ---------------------------------------------------------
# Message rendering helpers (bubble layout)
# ---------------------------------------------------------
def render_user_message(content, timestamp=""):
    """Full-width row, bubble pinned to the right, no avatar."""
    safe_content = html.escape(content).replace("\n", "<br>")
    time_html = f'<span class="msg-time">{html.escape(timestamp)}</span>' if timestamp else ""
    st.markdown(
        f'<div class="chat-row user-row">'
        f'<div class="msg-bubble user-bubble">{safe_content}{time_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def open_assistant_row():
    """Opens a full-width, left-aligned row for the assistant's plain-text reply."""
    st.markdown('<div class="chat-row assistant-row"><div class="msg-plain">', unsafe_allow_html=True)


def close_assistant_row():
    st.markdown('</div></div>', unsafe_allow_html=True)


def render_message_actions():
    """Decorative action row under an assistant reply (visual only, not wired up)."""
    st.markdown(
        """
        <div class="msg-actions">
            <span title="Share">📤</span>
            <span title="Regenerate">🔁</span>
            <span title="Copy">📋</span>
            <span title="More">⋯</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# TTS Functions (disabled for now — kept for future re-enabling)
# ---------------------------------------------------------
# def response_language(text):
#     arabic = len(re.findall(r"[\u0600-\u06FF]", text))
#     latin = len(re.findall(r"[A-Za-z]", text))
#     return "ar" if arabic > latin else "en"
#
#
# def generate_edge_tts_audio(text):
#     if not text:
#         return []
#
#     text = re.sub(r"```.*?```", "code omitted", text, flags=re.DOTALL)
#     text = re.sub(r"\s+", " ", text).strip()
#
#     if len(text) > MAX_TTS_CHARS:
#         text = text[:MAX_TTS_CHARS] + "..."
#
#     if not text:
#         return []
#
#     lang = response_language(text)
#     voice = EDGE_TTS_AR_VOICE if lang == "ar" else EDGE_TTS_EN_VOICE
#
#     output_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
#     output_path = output_file.name
#     output_file.close()
#
#     try:
#         async def generate_tts():
#             communicate = edge_tts.Communicate(text, voice)
#             await communicate.save(output_path)
#
#         asyncio.run(generate_tts())
#
#         with open(output_path, "rb") as f:
#             audio_bytes = f.read()
#
#         if not audio_bytes:
#             raise RuntimeError("Edge TTS returned empty audio.")
#
#         return [audio_bytes]
#
#     except Exception as e:
#         print(f"Edge TTS error: {type(e).__name__}: {e}")
#         raise RuntimeError(f"TTS generation failed: {str(e)}")
#     finally:
#         try:
#             os.unlink(output_path)
#         except OSError:
#             pass


# ---------------------------------------------------------
# Get Groq client
# ---------------------------------------------------------
def get_groq_client():
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    return Groq(api_key=GROQ_API_KEY)


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg: #f5f5fb;
        --surface: #ffffff;
        --surface-alt: #f9f8fd;
        --border: #e5e2f5;
        --text: #1c1a2e;
        --text-muted: #6b6885;
        --text-faint: #a5a2c2;
        --accent: #6d5bf6;
        --accent-2: #9b6bf3;
        --accent-soft: #f0edff;
        --accent-gradient: linear-gradient(135deg, #6d5bf6 0%, #a855f7 100%);
        --shadow-sm: 0 1px 2px rgba(28, 26, 46, 0.05);
        --shadow-md: 0 10px 30px -14px rgba(85, 60, 200, 0.28);
        --radius-lg: 20px;
        --radius-md: 14px;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"] {
        display: none;
    }

    .stApp {
        background:
            radial-gradient(1200px 600px at 15% -10%, rgba(109, 91, 246, 0.08), transparent 60%),
            radial-gradient(900px 500px at 100% 0%, rgba(168, 85, 247, 0.07), transparent 55%),
            var(--bg);
        font-family: 'Inter', sans-serif;
        color: var(--text);
    }

    .block-container {
        max-width: 720px;
        padding-top: 1.5rem;
        padding-bottom: 7.5rem;
    }

    /* ---------------- Top bar ---------------- */
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.6rem 0.9rem;
        background: rgba(246, 244, 240, 0.85);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid var(--border);
        border-radius: 999px;
        box-shadow: var(--shadow-sm);
    }

    .wordmark {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: -0.01em;
        color: var(--text);
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }

    .wordmark .bolt {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 26px;
        height: 26px;
        border-radius: 8px;
        background: var(--accent-gradient);
        color: #fff;
        font-size: 0.85rem;
        box-shadow: 0 4px 12px -4px rgba(109, 91, 246, 0.5);
    }

    .wordmark span.accent {
        color: var(--accent);
    }

    div[data-testid="stVerticalBlock"] > div:has(.wordmark) {
        position: sticky;
        top: 0;
        z-index: 999;
        background: transparent;
        padding-top: 0.6rem;
        padding-bottom: 0.6rem;
    }

    /* ---------------- Hero ---------------- */
    .hero {
        text-align: center;
        padding: 4rem 0 2.25rem 0;
    }

    .hero .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--accent);
        background: var(--accent-soft);
        border: 1px solid rgba(109, 91, 246, 0.18);
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        margin-bottom: 1.1rem;
    }

    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2.85rem;
        letter-spacing: -0.02em;
        line-height: 1.08;
        margin-bottom: 0.85rem;
        color: var(--text);
    }

    .hero h1 .accent {
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .hero p {
        color: var(--text-muted);
        font-size: 1.02rem;
        margin: 0 auto;
        max-width: 480px;
    }

    .hero .hint {
        font-size: 0.85rem;
        color: var(--text-faint);
        margin-top: 0.9rem;
    }

    .no-key-note {
        text-align: center;
        color: var(--text-faint);
        font-size: 0.85rem;
        margin-top: 0.75rem;
        background: var(--surface-alt);
        border: 1px dashed var(--border);
        border-radius: var(--radius-md);
        padding: 0.6rem 1rem;
        display: inline-block;
    }

    /* ---------------- Chat rows ---------------- */
    .chat-row {
        display: flex;
        width: 100%;
        margin: 0.35rem 0;
        animation: bubble-in 0.25s ease-out;
    }

    @keyframes bubble-in {
        from { opacity: 0; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ---- User row: bubble pinned to the right ---- */
    .chat-row.user-row {
        justify-content: flex-end;
    }

    .msg-bubble.user-bubble {
        position: relative;
        display: inline-block;
        max-width: 82%;
        background: var(--accent-soft);
        border: 1px solid rgba(109, 91, 246, 0.14);
        border-radius: var(--radius-md) var(--radius-md) 4px var(--radius-md);
        padding: 0.8rem 1.05rem;
        text-align: left;
        line-height: 1.55;
        box-shadow: var(--shadow-sm);
    }

    .msg-bubble.user-bubble .msg-time {
        display: block;
        margin-top: 0.3rem;
        font-size: 0.72rem;
        color: var(--text-faint);
        text-align: right;
    }

    /* ---- Assistant row: plain text, flush left, no bubble ---- */
    .chat-row.assistant-row {
        justify-content: flex-start;
    }

    .msg-plain {
        width: 100%;
        line-height: 1.55;
    }

    .msg-actions {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        margin-top: 0.6rem;
        font-size: 0.95rem;
        color: var(--text-faint);
    }

    .msg-actions span {
        cursor: default;
        opacity: 0.75;
        transition: opacity 0.15s ease, transform 0.15s ease;
    }

    .msg-actions span:hover {
        opacity: 1;
        transform: translateY(-1px);
        color: var(--accent);
    }

    /* ---------------- Buttons ---------------- */
    .stButton > button {
        background: var(--surface);
        border: 1px solid var(--border);
        color: var(--text-muted);
        border-radius: 999px;
        padding: 0.4rem 1.1rem;
        font-size: 0.85rem;
        font-weight: 500;
        transition: all 0.15s ease;
        box-shadow: var(--shadow-sm);
    }

    .stButton > button:hover:not(:disabled) {
        border-color: var(--accent);
        color: var(--accent);
        background: var(--accent-soft);
        transform: translateY(-1px);
    }

    .stButton > button:disabled {
        opacity: 0.45;
    }

    /* ---------------- Chat input ---------------- */
    div[data-testid="stChatInput"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 32px;
        box-shadow: var(--shadow-md);
        padding: 0.25rem 0.35rem;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }

    div[data-testid="stChatInput"]:focus-within {
        border-color: var(--accent);
        box-shadow: 0 14px 34px -14px rgba(109, 91, 246, 0.35);
    }

    /* Strip Streamlit's own inner border/focus-ring so only our pill border shows */
    div[data-testid="stChatInput"] *:focus,
    div[data-testid="stChatInput"] *:focus-within,
    div[data-testid="stChatInput"] *:focus-visible {
        outline: none !important;
        box-shadow: none !important;
    }

    div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] [data-baseweb="textarea"],
    div[data-testid="stChatInput"] [data-baseweb="base-input"],
    div[data-testid="stChatInput"] [data-testid="stChatInputContainer"] {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }

    div[data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: var(--text) !important;
        padding-left: 0.5rem !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: var(--text-faint) !important;
    }

    /* Circular buttons inside the pill (mic + send) */
    div[data-testid="stChatInput"] button {
        border-radius: 50% !important;
        transition: transform 0.15s ease, background 0.15s ease;
        border: none !important;
    }

    div[data-testid="stChatInput"] button:hover {
        transform: scale(1.06);
    }

    /* Send button — solid dark/purple circle with white arrow, like the reference */
    div[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"],
    div[data-testid="stChatInput"] button[kind="chatInputSubmitButton"] {
        background: var(--accent-gradient) !important;
        box-shadow: 0 6px 16px -6px rgba(109, 91, 246, 0.55);
    }

    div[data-testid="stChatInput"] button[data-testid="stChatInputSubmitButton"] svg,
    div[data-testid="stChatInput"] button[kind="chatInputSubmitButton"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }

    /* Mic / other buttons — soft lavender circle */
    div[data-testid="stChatInput"] button:not([data-testid="stChatInputSubmitButton"]):not([kind="chatInputSubmitButton"]) {
        background: var(--accent-soft) !important;
    }

    div[data-testid="stChatInput"] button:not([data-testid="stChatInputSubmitButton"]):not([kind="chatInputSubmitButton"]) svg {
        fill: var(--accent) !important;
        color: var(--accent) !important;
    }

    div[data-testid="stBottom"],
    div[data-testid="stBottom"] > div,
    div[data-testid="stBottomBlockContainer"],
    section[data-testid="stBottom"],
    .stChatFloatingInputContainer {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }

    /* ---------------- Audio + captions ---------------- */
    div[data-testid="stAudio"] {
        margin-top: 0.6rem;
        border-radius: 999px;
        overflow: hidden;
    }

    div[data-testid="stCaptionContainer"] {
        color: var(--text-faint) !important;
        font-size: 0.8rem !important;
    }

    /* ---------------- Expander (debug info) ---------------- */
    div[data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        background: var(--surface-alt);
        box-shadow: var(--shadow-sm);
    }

    /* ---------------- Scrollbar ---------------- */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 999px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-faint);
    }

    @media (max-width: 640px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding-top: 2.5rem;
        }

        .hero h1 {
            font-size: 2.1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Top bar
# ---------------------------------------------------------
top_bar = st.container()

with top_bar:
    st.markdown('<div class="topbar-wrap">', unsafe_allow_html=True)
    left, spacer, right = st.columns([5, 1, 1.2])

    with left:
        st.markdown(
            '<div class="wordmark"><span class="bolt">✨</span>AI <span class="accent">Chatbot</span></div>',
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
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# History
# ---------------------------------------------------------
messages = load_history()


# ---------------------------------------------------------
# Hero
# ---------------------------------------------------------
if not messages:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">✨ AI Chatbot · Powered by Groq</div>
            <h1>Ask at the <span class="accent">speed</span><br>of thought</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not GROQ_API_KEY:
        st.markdown(
            '<div style="text-align:center;">'
            '<p class="no-key-note">⚠️ Missing GROQ_API_KEY in .env file</p>'
            '</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# Render history
# ---------------------------------------------------------
for message in messages:
    if message["role"] == "user":
        render_user_message(message["content"], message.get("timestamp", ""))
    else:
        open_assistant_row()
        st.markdown(message["content"])
        render_message_actions()
        close_assistant_row()

# ---------------------------------------------------------
# Web Speech API STT (disabled for now — kept for future re-enabling)
# ---------------------------------------------------------
# st.iframe("""
# <!doctype html>
# <html><body><script>
# (() => {
#   const parentDocument = window.parent.document;
#   const Recognition = window.parent.SpeechRecognition || window.parent.webkitSpeechRecognition;
#   if (!Recognition) return;
#
#   let recognition;
#   let listening = false;
#   let transcript = '';
#
#   const getChatInput = () => parentDocument.querySelector(
#     '[data-testid="stChatInput"] textarea, [data-testid="stChatInput"] input'
#   );
#
#   function setInputValue(input, value) {
#     const prototype = Object.getPrototypeOf(input);
#     const setter = Object.getOwnPropertyDescriptor(prototype, 'value')?.set;
#     setter.call(input, value);
#     input.dispatchEvent(new Event('input', { bubbles: true }));
#     input.dispatchEvent(new Event('change', { bubbles: true }));
#   }
#
#   function submitTranscript() {
#     const input = getChatInput();
#     if (!input || !transcript.trim()) return;
#     setInputValue(input, transcript.trim());
#     setTimeout(() => {
#       const form = input.closest('form');
#       const submitButton = form?.querySelector('button[type="submit"]');
#       if (submitButton && !submitButton.disabled) submitButton.click();
#     }, 100);
#   }
#
#   function stopListening() {
#     listening = false;
#     if (recognition) {
#       try { recognition.stop(); } catch (_) {}
#     }
#   }
#
#   function startListening() {
#     transcript = '';
#     recognition = new Recognition();
#     recognition.lang = navigator.language || 'en-US';
#     recognition.continuous = false;
#     recognition.interimResults = false;
#     recognition.onstart = () => { listening = true; };
#     recognition.onresult = event => {
#       for (let i = event.resultIndex; i < event.results.length; i += 1) {
#         if (event.results[i].isFinal) transcript += event.results[i][0].transcript;
#       }
#     };
#     recognition.onerror = () => { listening = false; };
#     recognition.onend = () => {
#       const hadTranscript = transcript.trim();
#       listening = false;
#       if (hadTranscript) submitTranscript();
#     };
#     try { recognition.start(); } catch (_) {}
#   }
#
#   function hookMicrophone() {
#     const chatInput = parentDocument.querySelector('[data-testid="stChatInput"]');
#     if (!chatInput) return;
#     const microphone = [...chatInput.querySelectorAll('button')].find(button =>
#       /record|audio|microphone|voice/i.test(button.getAttribute('aria-label') || '')
#     ) || chatInput.querySelector('button');
#     if (!microphone || microphone.dataset.webSpeechHooked) return;
#
#     microphone.dataset.webSpeechHooked = 'true';
#     microphone.addEventListener('click', event => {
#       event.preventDefault();
#       event.stopImmediatePropagation();
#       if (listening) stopListening(); else startListening();
#     }, true);
#   }
#
#   new MutationObserver(hookMicrophone).observe(parentDocument.body, {
#     childList: true, subtree: true
#   });
#   hookMicrophone();
# })();
# </script></body></html>
# """, width=1, height=1, tab_index=-1)


# ---------------------------------------------------------
# Native Streamlit chat input
# ---------------------------------------------------------
submission = st.chat_input(
    "Ask anything",
    # accept_audio and audio_sample_rate disabled along with STT — uncomment to re-enable
    # accept_audio=True,
    # audio_sample_rate=16000,
    key="main_chat_input",  # <--- UNIQUE KEY ADDED HERE
)


# ---------------------------------------------------------
# Process input
# ---------------------------------------------------------
prompt = ""

if submission:
    if hasattr(submission, "text"):
        prompt = (submission.text or "").strip()
    else:
        prompt = str(submission).strip()


# ---------------------------------------------------------
# Generate assistant response
# ---------------------------------------------------------
if prompt:
    if not GROQ_API_KEY:
        st.error("No GROQ_API_KEY configured in your .env file.")
        st.stop()

    # Save the user message
    messages = append_message("user", prompt)

    render_user_message(prompt, messages[-1].get("timestamp", ""))

    api_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *[
            {"role": message["role"], "content": message["content"]}
            for message in messages
        ],
    ]

    full_response = ""
    response_succeeded = False
    error_message = None

    open_assistant_row()
    placeholder = st.empty()

    try:
        client = get_groq_client()
        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=api_messages,
            temperature=TEMPERATURE,
            stream=True,
        )

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_response += delta
                placeholder.markdown(full_response + "▌")

        full_response = full_response.strip()

        if not full_response:
            raise RuntimeError("The model returned an empty response.")

        placeholder.markdown(full_response)
        render_message_actions()
        response_succeeded = True

    except Exception as error:
        placeholder.empty()
        error_message = str(error)
        print(f"Groq chat error: {type(error).__name__}: {error}")

    finally:
        close_assistant_row()

    if error_message:
        st.error(f"AI Response failed: {error_message}")

        # Show the prompt that was sent for debugging
        with st.expander("🔍 Debug Info"):
            st.write("**Prompt sent to AI:**")
            st.code(prompt)
            st.write("**Messages sent:**")
            st.json(api_messages)

    if response_succeeded:
        append_message("assistant", full_response)

        # Voice response disabled for now — uncomment to re-enable Edge TTS playback
        # try:
        #     audio_chunks = generate_edge_tts_audio(full_response)
        #     for index, audio in enumerate(audio_chunks):
        #         st.audio(audio, format="audio/mp3", autoplay=(index == 0))
        #     st.caption("🔊 Voice response (Edge TTS - Free)")
        # except Exception as error:
        #     print(f"TTS error: {type(error).__name__}: {error}")
        #     st.info("💬 Voice generation unavailable, but text response is above.")