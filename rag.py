from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DOCS_DIR = Path("docs")
CHROMA_DIR = Path("chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "llama-3.1-8b-instant"


def get_groq_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY. Set it as an environment variable or add it to "
            "Streamlit secrets before asking questions."
        )
    return api_key


def list_document_files(docs_dir: Path = DOCS_DIR) -> list[Path]:
    if not docs_dir.exists():
        return []
    supported_extensions = {".txt", ".md", ".pdf"}
    return sorted(
        path
        for path in docs_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in supported_extensions
    )


def load_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    documents: list[Document] = []
    for path in list_document_files(docs_dir):
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
        loaded = loader.load()
        for document in loaded:
            document.metadata["source"] = str(path)
        documents.extend(loaded)
    return documents


def split_documents(
    documents: Iterable[Document],
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(list(documents))


def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def ingest_documents(
    docs_dir: Path = DOCS_DIR,
    persist_directory: Path = CHROMA_DIR,
    reset: bool = True,
) -> int:
    documents = load_documents(docs_dir)
    if not documents:
        raise RuntimeError(f"No .txt, .md, or .pdf documents found in {docs_dir}.")

    chunks = split_documents(documents)
    if reset and persist_directory.exists():
        shutil.rmtree(persist_directory)

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        persist_directory=str(persist_directory),
    )
    return len(chunks)


def get_vectorstore(persist_directory: Path = CHROMA_DIR) -> Chroma:
    if not persist_directory.exists():
        raise RuntimeError(
            f"Vector database not found at {persist_directory}. Run `python ingest.py` first."
        )
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=get_embedding_model(),
    )


def get_llm(api_key: str | None = None) -> ChatGroq:
    return ChatGroq(
        api_key=api_key or get_groq_api_key(),
        model_name=os.getenv("GROQ_MODEL", DEFAULT_LLM_MODEL),
        temperature=0,
    )


def format_sources(documents: Iterable[Document]) -> list[str]:
    sources = []
    for document in documents:
        source = document.metadata.get("source", "unknown source")
        if source not in sources:
            sources.append(source)
    return sources


def answer_question(
    question: str,
    llm: ChatGroq | None = None,
    retriever=None,
) -> tuple[str, list[str]]:
    if retriever is None:
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    if llm is None:
        llm = get_llm()

    documents = retriever.invoke(question)
    context = "\n\n".join(
        f"Source: {document.metadata.get('source', 'unknown')}\n{document.page_content}"
        for document in documents
    )
    prompt = (
        "You are a helpful support assistant. Answer the question using only the "
        "context below. If the answer is not in the context, say you do not know "
        "and suggest contacting support.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    response = llm.invoke(prompt)
    return response.content, format_sources(documents)
