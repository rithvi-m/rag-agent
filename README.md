#RAG Support Agent 🤖

An AI-powered support assistant built with Retrieval-Augmented 
Generation (RAG) — deployed live on Streamlit Cloud.

🌐 Live Demo: rag-agent-klwhga7y6tfnbblzrj2qsh.streamlit.app

What it does
Upload your own documents (.txt, .md, .pdf) and ask questions 
about them. The agent retrieves relevant context and answers 
using Groq's LLM — no hallucinations, grounded responses only.

🛠️ Tech Stack
- Streamlit — frontend UI
- LangChain — RAG pipeline
- ChromaDB — vector database
- Hugging Face — embeddings
- Groq — LLM (chat model)

Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Add your Groq API key in .streamlit/secrets.toml:
GROQ_API_KEY = "your-groq-api-key"

Add Documents
Put .txt, .md, or .pdf files in the docs folder, then run:
python ingest.py

Run The App
python -m streamlit run app.py

---
Built by Rithvi M | github.com/rithvi-m
