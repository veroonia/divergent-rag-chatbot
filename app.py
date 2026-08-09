import io
import json
import os
import re
import tempfile
import threading
import wave

import streamlit as st
from dotenv import load_dotenv
from groq import Groq


# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI ChatBot",
    layout="centered",
)

load_dotenv()
API_KEY = os.environ.get("GROQ_API_KEY", "").strip()

HISTORY_FILE = "chat_history.json"
HISTORY_LOCK = threading.RLock()

CHAT_MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0.7
SYSTEM_PROMPT = "You are a helpful, concise assistant."

STT_MODEL = "whisper-large-v3-turbo"

TTS_EN_MODEL = "canopylabs/orpheus-v1-english"
TTS_EN_VOICE = "hannah"

TTS_AR_MODEL = "canopylabs/orpheus-arabic-saudi"
TTS_AR_VOICE = "aisha"


# ---------------------------------------------------------
# Shared chat history
# ---------------------------------------------------------
def _is_valid_history(data):
    """Validate the JSON structure before using it as chat history."""
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
    """Load the shared chat history safely."""
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
    """Atomically save history so a partially-written JSON file is avoided."""
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
            st.warning("Your message was sent, but the chat history could not be saved.")
            print(f"History save error: {error}")

            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


def append_message(role, content):
    """
    Reload before appending so different browser sessions/devices do not
    overwrite each other's latest messages.
    """
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
# Groq helpers
# ---------------------------------------------------------
def get_client():
    if not API_KEY:
        raise RuntimeError("No GROQ_API_KEY was found.")
    return Groq(api_key=API_KEY)


def transcribe(audio_file):
    """
    Send Streamlit's native recorded WAV file directly to Groq Whisper.
    No Base64, query parameters, custom JavaScript, or page redirects.
    """
    audio_bytes = audio_file.getvalue()

    if not audio_bytes:
        return ""

    filename = audio_file.name or "recording.wav"

    result = get_client().audio.transcriptions.create(
        file=(filename, audio_bytes),
        model=STT_MODEL,
        response_format="json",
        temperature=0.0,
    )

    return (result.text or "").strip()


def contains_mostly_arabic(text):
    arabic_chars = sum("\u0600" <= char <= "\u06FF" for char in text)
    letters = sum(char.isalpha() for char in text)

    return letters > 0 and arabic_chars / letters >= 0.35


def split_tts_text(text, max_chars=190):
    """
    Orpheus accepts short inputs, so long assistant responses are split
    at sentence/word boundaries before generating audio.
    """
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?؟])\s+", text)
    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) > max_chars:
            words = sentence.split()

            for word in words:
                candidate = word if not current else f"{current} {word}"

                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = word

            continue

        candidate = sentence if not current else f"{current} {sentence}"

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def response_to_wav_bytes(response):
    """Read Groq's WAV response into memory without keeping a permanent file."""
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name

        response.write_to_file(temp_path)

        with open(temp_path, "rb") as audio_file:
            return audio_file.read()

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def merge_wav_files(wav_files):
    """Merge multiple WAV chunks into one playable WAV file."""
    if not wav_files:
        return None

    if len(wav_files) == 1:
        return wav_files[0]

    output = io.BytesIO()

    with wave.open(io.BytesIO(wav_files[0]), "rb") as first:
        params = first.getparams()

    with wave.open(output, "wb") as writer:
        writer.setparams(params)

        for wav_bytes in wav_files:
            with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
                same_format = (
                    reader.getnchannels() == params.nchannels
                    and reader.getsampwidth() == params.sampwidth
                    and reader.getframerate() == params.framerate
                )

                if not same_format:
                    raise ValueError("TTS audio chunks have incompatible WAV formats.")

                writer.writeframes(reader.readframes(reader.getnframes()))

    return output.getvalue()


def generate_tts_audio(text):
    """Generate a single WAV file for an assistant response."""
    clean_text = re.sub(r"```.*?```", "code omitted", text, flags=re.DOTALL)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    if not clean_text:
        return None

    if contains_mostly_arabic(clean_text):
        model = TTS_AR_MODEL
        voice = TTS_AR_VOICE
    else:
        model = TTS_EN_MODEL
        voice = TTS_EN_VOICE

    wav_chunks = []

    for chunk in split_tts_text(clean_text):
        response = get_client().audio.speech.create(
            model=model,
            voice=voice,
            input=chunk,
            response_format="wav",
        )

        wav_chunks.append(response_to_wav_bytes(response))

    return merge_wav_files(wav_chunks)


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

    /* ---- top bar ---- */
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

    /* ---- hero ---- */
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

    /* ---- chat messages ---- */
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

    /* ---- buttons outside the chat input ---- */
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

    /*
    IMPORTANT:
    Do not style every button inside stChatInput.
    Streamlit's native audio-enabled chat input has separate microphone
    and send controls. Keeping those controls native preserves the layout
    shown in the reference image.
    */
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

    /* Keep Streamlit's bottom wrappers visually clean. */
    div[data-testid="stBottom"],
    div[data-testid="stBottom"] > div,
    div[data-testid="stBottomBlockContainer"],
    section[data-testid="stBottom"],
    .stChatFloatingInputContainer {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }

    /* ---- sticky top bar ---- */
    div[data-testid="stVerticalBlock"] > div:has(.wordmark) {
        position: sticky;
        top: 0;
        z-index: 999;
        background: #f7f5f2;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }

    /* ---- mobile ---- */
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
        if st.button("Clear", use_container_width=True):
            clear_history()
            st.rerun()


# ---------------------------------------------------------
# Shared history
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
            <p>Groq-powered chat with native voice input and voice responses</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not API_KEY:
        st.markdown(
            '<p class="no-key-note">No GROQ_API_KEY found — add it to a .env file '
            "in this folder to enable chat, STT, and TTS.</p>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# Render history
# ---------------------------------------------------------
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------
# Native chat input with built-in microphone
# ---------------------------------------------------------
submission = st.chat_input(
    "Ask anything",
    accept_audio=True,
    audio_sample_rate=16000,
    key="chat_input",
)

prompt = ""

if submission:
    typed_text = (submission.text or "").strip()
    recorded_audio = submission.audio

    if recorded_audio is not None:
        if not API_KEY:
            st.error("No Groq API key configured. Add GROQ_API_KEY to your .env file.")
        else:
            with st.spinner("Transcribing your voice..."):
                try:
                    transcript = transcribe(recorded_audio)
                except Exception as error:
                    transcript = ""
                    st.error(
                        "I couldn't transcribe that recording. "
                        "Please check your microphone/API key and try again."
                    )
                    print(f"STT error: {error}")

            if transcript:
                prompt = transcript if not typed_text else f"{typed_text}\n\n{transcript}"
            elif typed_text:
                prompt = typed_text

    else:
        prompt = typed_text


# ---------------------------------------------------------
# Generate assistant response
# ---------------------------------------------------------
if prompt:
    if not API_KEY:
        st.error(
            "No Groq API key configured. Add GROQ_API_KEY to a .env file in this folder."
        )
        st.stop()

    # Save the user's message first, so it is not lost if the API fails.
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
            stream = get_client().chat.completions.create(
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
            st.error("The AI response failed. Please try again.")
            print(f"Chat API error: {error}")

        # Only save a real assistant response, never an API error.
        if response_succeeded:
            append_message("assistant", full_response)

            # Generate and autoplay TTS for the new answer.
            with st.spinner("Generating voice response..."):
                try:
                    tts_audio = generate_tts_audio(full_response)

                    if tts_audio:
                        st.audio(
                            tts_audio,
                            format="audio/wav",
                            autoplay=True,
                        )

                except Exception as error:
                    # The text response still works even if TTS fails.
                    print(f"TTS error: {error}")
                    st.caption("Voice playback is temporarily unavailable.")
