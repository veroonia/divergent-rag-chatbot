import os
import json
from dotenv import load_dotenv
import streamlit as st
from groq import Groq

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


def get_response(prompt: str) -> str:
    """Send prompt to Groq."""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.7,
            max_tokens=1024,
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"Error: {e}"


def main():
    st.set_page_config(
        page_title="Groq Chatbot",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 Groq Chatbot")

    # Sidebar
    with st.sidebar:
        st.header("Options")

        if st.button("🗑️ Clear Chat"):
            save_messages([])
            st.rerun()

    # Load shared chat history
    messages = load_messages()

    # Display messages
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):

        # Save user message
        messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )
        save_messages(messages)

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = get_response(prompt)
                st.markdown(reply)

        # Save assistant response
        messages.append(
            {
                "role": "assistant",
                "content": reply
            }
        )
        save_messages(messages)

        # Refresh page to display updated chat
        st.rerun()


if __name__ == "__main__":
    main()