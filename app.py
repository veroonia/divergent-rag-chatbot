import json
import os
import re
import tempfile
import threading
import asyncio
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
import edge_tts

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Groq Chat",
    page_icon="⚡",
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

# Edge TTS voice configuration
EDGE_TTS_EN_VOICE = "en-US-JennyNeural"
EDGE_TTS_AR_VOICE = "ar-EG-SalmaNeural"
MAX_TTS_CHARS = 3000

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
        messages.append({"role": role, "content": content})
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
# TTS Functions
# ---------------------------------------------------------
def response_language(text):
    arabic = len(re.findall(r"[\u0600-\u06FF]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return "ar" if arabic > latin else "en"


def generate_edge_tts_audio(text):
    if not text:
        return []

    text = re.sub(r"```.*?```", "code omitted", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    
    if len(text) > MAX_TTS_CHARS:
        text = text[:MAX_TTS_CHARS] + "..."
    
    if not text:
        return []

    lang = response_language(text)
    voice = EDGE_TTS_AR_VOICE if lang == "ar" else EDGE_TTS_EN_VOICE
    
    output_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    output_path = output_file.name
    output_file.close()

    try:
        async def generate_tts():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
        
        asyncio.run(generate_tts())
        
        with open(output_path, "rb") as f:
            audio_bytes = f.read()
        
        if not audio_bytes:
            raise RuntimeError("Edge TTS returned empty audio.")
        
        return [audio_bytes]
        
    except Exception as e:
        print(f"Edge TTS error: {type(e).__name__}: {e}")
        raise RuntimeError(f"TTS generation failed: {str(e)}")
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


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
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

    #MainMenu, footer, header {
        visibility: hidden;
    }

    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"] {
        display: none;
    }

    .stApp {
        background: #f7f5f2;
        font-family: 'Inter', sans-serif;
        color: #191816;
    }

    .block-container {
        max-width: 720px;
        padding-top: 2rem;
        padding-bottom: 7rem;
    }

    .wordmark {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.15rem;
        letter-spacing: -0.01em;
        color: #191816;
        padding-top: 0.5rem;
    }

    .wordmark span {
        color: #ff4d1c;
    }

    .hero {
        text-align: center;
        padding: 4.5rem 0 2rem 0;
    }

    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2.7rem;
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-bottom: 0.7rem;
        color: #191816;
    }

    .hero h1 .accent {
        color: #ff4d1c;
    }

    .hero p {
        color: #706c66;
        font-size: 1.02rem;
    }

    .no-key-note {
        text-align: center;
        color: #a8a39a;
        font-size: 0.85rem;
        margin-top: -0.5rem;
    }

    div[data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.3rem 0;
    }

    div[data-testid="stChatMessageContent"] {
        background: #ffffff;
        border: 1px solid #e7e3dc;
        border-radius: 14px;
        padding: 0.75rem 1.1rem;
        box-shadow: 0 1px 2px rgba(25, 24, 22, 0.03);
    }

    .stButton > button {
        background: #ffffff;
        border: 1px solid #e7e3dc;
        color: #706c66;
        border-radius: 999px;
        padding: 0.35rem 1rem;
        font-size: 0.85rem;
    }

    .stButton > button:hover {
        border-color: #ff4d1c;
        color: #ff4d1c;
    }

    div[data-testid="stChatInput"] {
        background: #ffffff;
        border: 1px solid #e7e3dc;
        border-radius: 28px;
        box-shadow: 0 8px 24px -14px rgba(25, 24, 22, 0.25);
    }

    div[data-testid="stChatInput"] textarea {
        background: transparent !important;
        color: #191816 !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: #a8a39a !important;
    }

    div[data-testid="stChatInput"] button {
        border-radius: 50%;
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

    div[data-testid="stVerticalBlock"] > div:has(.wordmark) {
        position: sticky;
        top: 0;
        z-index: 999;
        background: #f7f5f2;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }

    div[data-testid="stAudio"] {
        margin-top: 0.5rem;
    }

    @media (max-width: 640px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding-top: 3rem;
        }

        .hero h1 {
            font-size: 2.15rem;
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
    left, spacer, right = st.columns([5, 1, 1.2])

    with left:
        st.markdown(
            '<div class="wordmark">groq<span>chat</span> ⚡</div>',
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
            <h1>Ask at the <span class="accent">speed</span><br>of thought</h1>
            <p>Groq-powered chat with Chrome Web Speech API STT and Edge TTS</p>
            <p style="font-size:0.85rem; color:#a8a39a; margin-top:0.5rem;">
                🎤 Click the microphone to speak • 💬 Type to text
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not GROQ_API_KEY:
        st.markdown(
            '<p class="no-key-note">⚠️ Missing GROQ_API_KEY in .env file</p>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# Render history
# ---------------------------------------------------------
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------
# Web Speech API Override - Intercept microphone button
# ---------------------------------------------------------
# This iframe injects JavaScript that intercepts the microphone button click
# and uses Chrome's Web Speech API instead of Streamlit's built-in recording
st.iframe("""
<!DOCTYPE html>
<html>
<head>
<script>
(function() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;
    
    let recognition = null;
    let isListening = false;
    
    // Function to find and hook the microphone button
    function hookMicrophone() {
        const micBtn = document.querySelector('button[data-testid="stChatInput"] button:first-child');
        if (micBtn && !micBtn.dataset.speechHooked) {
            micBtn.dataset.speechHooked = 'true';
            
            micBtn.addEventListener('click', function(e) {
                // Only intercept if it's the mic button (has SVG)
                if (this.querySelector('svg')) {
                    e.stopPropagation();
                    e.preventDefault();
                    
                    if (!recognition) {
                        recognition = new SpeechRecognition();
                        recognition.lang = 'en-US';
                        recognition.continuous = false;
                        recognition.interimResults = true;
                        
                        recognition.onstart = function() {
                            isListening = true;
                            micBtn.style.color = '#dc3545';
                            micBtn.style.animation = 'pulse 1s infinite';
                        };
                        
                        recognition.onresult = function(event) {
                            let finalTranscript = '';
                            for (let i = event.resultIndex; i < event.results.length; i++) {
                                if (event.results[i].isFinal) {
                                    finalTranscript += event.results[i][0].transcript;
                                }
                            }
                            if (finalTranscript) {
                                const input = document.querySelector('input[data-testid="stChatInput"]');
                                if (input) {
                                    input.value = finalTranscript;
                                    input.dispatchEvent(new Event('input', { bubbles: true }));
                                    input.dispatchEvent(new Event('change', { bubbles: true }));
                                    // Click send
                                    const sendBtn = document.querySelector('button[data-testid="stChatInput"] button:last-child');
                                    if (sendBtn) setTimeout(() => sendBtn.click(), 100);
                                }
                            }
                        };
                        
                        recognition.onerror = function() { stopListening(); };
                        recognition.onend = function() { stopListening(); };
                        
                        // Add pulse style
                        const style = document.createElement('style');
                        style.textContent = '@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }';
                        document.head.appendChild(style);
                    }
                    
                    if (isListening) {
                        recognition.stop();
                        stopListening();
                    } else {
                        try { recognition.start(); } catch(e) {}
                    }
                    
                    function stopListening() {
                        isListening = false;
                        micBtn.style.color = '';
                        micBtn.style.animation = 'none';
                        if (recognition) try { recognition.stop(); } catch(e) {}
                    }
                }
            });
        }
    }
    
    // Wait for Streamlit to fully load
    const observer = new MutationObserver(() => {
        hookMicrophone();
    });
    observer.observe(document.body, { childList: true, subtree: true });
    
    // Also try immediately in case it's already loaded
    setTimeout(hookMicrophone, 500);
})();
</script>
</head>
<body>
</body>
</html>
""", width=1, height=1)


# ---------------------------------------------------------
# Native Streamlit chat input
# ---------------------------------------------------------
submission = st.chat_input(
    "Ask anything",
    accept_audio=True,
    audio_sample_rate=16000,
    key="main_chat_input",  # <--- UNIQUE KEY ADDED HERE
)


# ---------------------------------------------------------
# Process input
# ---------------------------------------------------------
prompt = ""

if submission:
    # Extract text from ChatInputValue object
    if hasattr(submission, 'text'):
        prompt = submission.text or ""
    else:
        prompt = str(submission) if submission else ""


# ---------------------------------------------------------
# Generate assistant response
# ---------------------------------------------------------
if prompt:
    if not GROQ_API_KEY:
        st.error("No GROQ_API_KEY configured in your .env file.")
        st.stop()

    # Save the user message
    messages = append_message("user", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    api_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *[
            {"role": message["role"], "content": message["content"]}
            for message in messages
        ],
    ]

    full_response = ""
    response_succeeded = False

    with st.chat_message("assistant"):
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
            response_succeeded = True

        except Exception as error:
            placeholder.empty()
            st.error(f"AI Response failed: {str(error)}")
            print(f"Groq chat error: {type(error).__name__}: {error}")
            
            # Show the prompt that was sent for debugging
            with st.expander("🔍 Debug Info"):
                st.write("**Prompt sent to AI:**")
                st.code(prompt)
                st.write("**Messages sent:**")
                st.json(api_messages)

        if response_succeeded:
            append_message("assistant", full_response)

            # Generate voice response
            try:
                audio_chunks = generate_edge_tts_audio(full_response)
                for index, audio in enumerate(audio_chunks):
                    st.audio(audio, format="audio/mp3", autoplay=(index == 0))
                st.caption("🔊 Voice response (Edge TTS - Free)")
            except Exception as error:
                print(f"TTS error: {type(error).__name__}: {error}")
                st.info("💬 Voice generation unavailable, but text response is above.")