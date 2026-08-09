import os
import io
import json
import base64
from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components
from groq import Groq
from gtts import gTTS
from streamlit_mic_recorder import speech_to_text

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not found in .env file.")
    st.stop()

# Create Groq client
client = Groq(api_key=api_key)

# File to store shared chat history
CHAT_FILE = "chat_history.json"


def load_messages():
    """Load chat history from file."""
    if not os.path.exists(CHAT_FILE):
        return []

    try:
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def save_messages(messages):
    """Save chat history to file."""
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=4, ensure_ascii=False)


def get_response(prompt: str, model_name: str = "openai/gpt-oss-20b") -> str:
    """Send prompt to Groq model."""
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.7,
            max_tokens=2048,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


def text_to_speech(text: str) -> bytes:
    """Convert text to speech MP3 bytes using gTTS."""
    try:
        # Clean markdown formatting before speaking
        clean_text = text.replace("*", "").replace("#", "").replace("`", "")
        tts = gTTS(text=clean_text, lang="en")
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception as e:
        st.error(f"Text-to-Speech Error: {e}")
        return None


def main():
    st.set_page_config(
        page_title="Groq AI Voice Chatbot",
        page_icon="🤖",
        layout="wide",
    )

    # Custom CSS to ensure bottom mic + input bar remains fixed at the bottom throughout the chat
    st.markdown(
        """
        <style>
        /* Fix bottom container (mic + chat input) to viewport bottom */
        div[data-testid="stHorizontalBlock"]:has(iframe[title*="speech_to_text"]) {
            position: fixed;
            bottom: 0px;
            left: 0px;
            right: 0px;
            background: #ffffff;
            padding: 12px 3rem 16px 3rem;
            z-index: 99999;
            box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.08);
            border-top: 1px solid #f0f2f6;
        }
        /* Ensure sidebar open space styling */
        @media (min-width: 992px) {
            section[data-testid="stSidebar"][aria-expanded="true"] + section div[data-testid="stHorizontalBlock"]:has(iframe[title*="speech_to_text"]) {
                left: 18rem;
            }
        }
        /* Add bottom margin to scroll container so chat messages are never covered */
        .main .block-container {
            padding-bottom: 110px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🤖 Groq AI Voice Chatbot")

    # Sidebar options
    with st.sidebar:
        st.header("⚙️ Options")

        st.subheader("🧠 Model Selection")
        model_options = {
            "OpenAI GPT-OSS 20B (Default)": "openai/gpt-oss-20b",
            "Meta Llama 3.3 70B": "llama-3.3-70b-versatile",
            "DeepSeek R1 70B (Advanced Reasoning)": "deepseek-r1-distill-llama-70b",
            "Meta Llama 3.1 8B (Ultra Fast)": "llama-3.1-8b-instant",
            "Mixtral 8x7B": "mixtral-8x7b-32768",
        }
        selected_model_name = st.selectbox(
            "Choose AI Model:",
            options=list(model_options.keys()),
            index=0,
        )
        selected_model_id = model_options[selected_model_name]

        st.markdown("---")
        enable_tts = st.checkbox("🔊 Auto-Play Audio Speech", value=True)

        st.markdown("---")
        if st.button("🗑️ Clear Chat"):
            save_messages([])
            st.rerun()

    # Load shared chat history
    messages = load_messages()

    # Display text messages ONLY
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = None

    # Bottom Input Layout (Web Speech API Mic button + Chat Input side by side)
    col_mic, col_input = st.columns([1.8, 14], vertical_alignment="center")

    with col_mic:
        web_speech_text = speech_to_text(
            language="en",
            start_prompt="🎤 Speak",
            stop_prompt="🛑 Listening...",
            just_once=True,
            use_container_width=True,
            key="web_speech_stt",
        )

    with col_input:
        text_prompt = st.chat_input("Ask me anything or click microphone to speak...")

    # Handle Web Speech API Voice Input
    if web_speech_text and web_speech_text.strip():
        prompt = web_speech_text.strip()

    # Handle Text Input
    if text_prompt:
        prompt = text_prompt

    # Process prompt if available
    if prompt:
        # Save and display user message
        messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response from Groq
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking with {selected_model_name}... 🤖"):
                reply = get_response(prompt, selected_model_id)
                st.markdown(reply)

                # Automatically play audio response invisibly (No visible player controls)
                if enable_tts:
                    with st.spinner("Generating voice audio... 🔊"):
                        speech_bytes = text_to_speech(reply)
                        if speech_bytes:
                            audio_b64 = base64.b64encode(speech_bytes).decode("utf-8")
                            autoplay_html = f"""
                            <audio autoplay style="display:none;">
                                <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                            </audio>
                            """
                            components.html(autoplay_html, height=0)

        # Save assistant text message to chat history and refresh
        messages.append({"role": "assistant", "content": reply})
        save_messages(messages)
        st.rerun()


if __name__ == "__main__":
    main()