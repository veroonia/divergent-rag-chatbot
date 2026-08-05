import os
from dotenv import load_dotenv
import streamlit as st
from groq import Groq

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")

# Create Groq client
client = Groq(api_key=api_key)


def get_response(prompt: str) -> str:
    """Send a prompt to Groq and return the response."""

    if not api_key:
        return "GROQ_API_KEY not found."

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

    # Store conversation
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):

        # Save and display user message
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = get_response(prompt)
                st.markdown(reply)

        # Save assistant response
        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )


if __name__ == "__main__":
    main()