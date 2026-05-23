# RAG Support Agent

A small RAG support assistant built with Streamlit, LangChain, ChromaDB, Hugging Face embeddings, and Groq chat models.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set your Groq API key:

```powershell
$env:GROQ_API_KEY="your-groq-api-key"
```

For Streamlit Cloud or local Streamlit secrets, copy `.streamlit/secrets.toml.example`
to `.streamlit/secrets.toml`, then add your real key:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

## Add Documents

Put `.txt`, `.md`, or `.pdf` files in the `docs` folder, then build the vector database:

```powershell
python ingest.py
```

## Run The App

```powershell
python -m streamlit run app.py
```

The sidebar lets you upload new `.txt`, `.md`, or `.pdf` files and re-index the knowledge base.

## Run From Terminal

```powershell
python ask.py
```

Type `quit` or `exit` to stop the terminal chat.
