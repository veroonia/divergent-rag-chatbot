import json
import os
import tempfile
import threading
import asyncio
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from ui_fragments import (
    close_assistant_row,
    open_assistant_row,
    render_hero,
    render_message_actions,
    render_missing_key_note,
    render_user_message,
    render_wordmark,
)
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
def load_css() -> None:
    css_path = Path(__file__).with_name("styles").joinpath("app.css")
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# Styling
# ---------------------------------------------------------
load_css()


# ---------------------------------------------------------
# Top bar
# ---------------------------------------------------------
top_bar = st.container()

with top_bar:
    st.markdown('<div class="topbar-wrap">', unsafe_allow_html=True)
    left, spacer, right = st.columns([5, 1, 1.2])

    with left:
        st.markdown(render_wordmark(), unsafe_allow_html=True)

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
    st.markdown(render_hero(), unsafe_allow_html=True)

    if not GROQ_API_KEY:
        st.markdown(render_missing_key_note(), unsafe_allow_html=True)


# ---------------------------------------------------------
# Render history
# ---------------------------------------------------------
for message in messages:
    if message["role"] == "user":
        st.markdown(render_user_message(message["content"], message.get("timestamp", "")), unsafe_allow_html=True)
    else:
        st.markdown(open_assistant_row(), unsafe_allow_html=True)
        st.markdown(message["content"])
        st.markdown(render_message_actions(), unsafe_allow_html=True)
        st.markdown(close_assistant_row(), unsafe_allow_html=True)

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

    st.markdown(render_user_message(prompt, messages[-1].get("timestamp", "")), unsafe_allow_html=True)

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

    st.markdown(open_assistant_row(), unsafe_allow_html=True)
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
        st.markdown(render_message_actions(), unsafe_allow_html=True)
        response_succeeded = True

    except Exception as error:
        placeholder.empty()
        error_message = str(error)
        print(f"Groq chat error: {type(error).__name__}: {error}")

    finally:
        st.markdown(close_assistant_row(), unsafe_allow_html=True)

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