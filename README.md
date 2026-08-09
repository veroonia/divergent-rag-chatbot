# Simple Chatbot (Streamlit)

A ChatGPT-style chatbot built with Streamlit and Groq's API.

## Installation

Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configure Groq

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your Groq API key.

## Run the application

```powershell
streamlit run app.py
```

## Features

- ChatGPT-style interface
- Conversation history
- Uses the official Groq API
- Secure API key using `.env`

## Project Structure

```
.
├── app.py
├── requirements.txt
├── .env
└── README.md
```

> **Note**
>
> Do **not** commit your `.env` file to GitHub.
