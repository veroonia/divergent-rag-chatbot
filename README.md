# Divergent RAG Chatbot

A ChatGPT-style chatbot built with **Streamlit**, **Groq**, **LangChain**, **Hugging Face embeddings**, and **Qdrant**.

The chatbot can answer both:

- **General questions** using the language model's general knowledge.
- **Document questions** about the _Divergent_ novel using Retrieval-Augmented Generation (RAG) and the indexed document stored in Qdrant.

## Features

- ChatGPT-style Streamlit interface
- Conversation history
- Streaming AI responses
- Groq API integration
- Document-based question answering using RAG
- Qdrant vector database for document retrieval
- Hugging Face sentence-transformer embeddings
- Document relevance filtering using a similarity threshold
- Document/general question routing
- Retrieved document context displayed for document questions
- Chapter and source metadata included in retrieved context
- `.env` support for securely storing API keys
- Docker support

## Technologies

- Python
- Streamlit
- Groq API
- LangChain
- Qdrant
- Hugging Face Embeddings
- Sentence Transformers
- Docker

## Project Structure

````text
.
├── app.py
├── rag_retriever.py
├── html_utils.py
├── ingest.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
├── data/
│   └── Divergent.pdf
├── styles/
│   └── app.css
└── chat_history.json

## Installation

Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
````

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

## Docker

The application can also be run using Docker.

### Build the Docker Image

From the project root:

````powershell
docker build -t divergent-rag-chatbot .



## Qdrant Setup

The chatbot uses Qdrant as its vector database. Make sure Qdrant is running locally at:

http://localhost:6333

The application expects the following collection:

`divergent_children`

### Run Qdrant with Docker

If Qdrant is not already running, start it with:

```powershell
docker run -p 6333:6333 qdrant/qdrant


````
