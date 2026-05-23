import os
from pathlib import Path

import streamlit as st

from rag import (
    CHROMA_DIR,
    DOCS_DIR,
    answer_question,
    get_llm,
    get_vectorstore,
    ingest_documents,
    list_document_files,
)


st.set_page_config(
    page_title="SupportAI",
    page_icon="AI",
    layout="wide",
)

st.markdown(
    """
<style>
    [data-testid="stSidebar"] { background-color: #1a1a2e; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    .stChatMessage { border-radius: 8px; }
    .stChatInput input { border-radius: 20px !important; }
    header, #MainMenu, footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


def get_secret(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except Exception:
        return None


@st.cache_resource
def load_chain(api_key: str):
    vectorstore = get_vectorstore(CHROMA_DIR)
    llm = get_llm(api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return llm, retriever


def save_uploaded_files(uploaded_files) -> int:
    DOCS_DIR.mkdir(exist_ok=True)
    saved_count = 0
    for uploaded_file in uploaded_files:
        destination = DOCS_DIR / Path(uploaded_file.name).name
        destination.write_bytes(uploaded_file.getbuffer())
        saved_count += 1
    return saved_count


with st.sidebar:
    st.title("SupportAI")
    st.caption("LLaMA + RAG support assistant")
    st.divider()

    st.markdown("**Knowledge base**")
    docs = list_document_files(DOCS_DIR)
    if docs:
        for doc in docs:
            st.markdown(f"- {doc.name}")
    else:
        st.caption("No documents found.")

    uploaded_files = st.file_uploader(
        "Add documents",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Save and re-index", use_container_width=True):
        saved_count = save_uploaded_files(uploaded_files)
        chunk_count = ingest_documents()
        load_chain.clear()
        st.success(f"Saved {saved_count} file(s) and indexed {chunk_count} chunks.")
        st.rerun()

    if st.button("Re-index current docs", use_container_width=True):
        try:
            chunk_count = ingest_documents()
            load_chain.clear()
            st.success(f"Indexed {chunk_count} chunks.")
        except RuntimeError as error:
            st.error(str(error))

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("v1.1")


st.title("Support Assistant")
st.caption("Ask questions that can be answered from your indexed documents.")

api_key = get_secret("GROQ_API_KEY")
if not api_key:
    st.error("Set GROQ_API_KEY in your environment or Streamlit secrets to start chatting.")
    st.stop()

if not CHROMA_DIR.exists():
    try:
        with st.spinner("Building the vector database from docs..."):
            ingest_documents()
    except RuntimeError as error:
        st.warning(f"{error} Add documents in the sidebar, then click re-index.")
        st.stop()

try:
    llm, retriever = load_chain(api_key)
except RuntimeError as error:
    st.error(str(error))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    cols = st.columns(4)
    suggestions = [
        "Reset password",
        "Cancel subscription",
        "Payment issues",
        "Contact support",
    ]
    for index, col in enumerate(cols):
        if col.button(suggestions[index], use_container_width=True):
            st.session_state.messages.append(
                {"role": "user", "content": suggestions[index]}
            )
            st.rerun()

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write("Hello! Ask me a question about the support docs.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            st.caption("Sources: " + ", ".join(message["sources"]))

if question := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = answer_question(question, llm=llm, retriever=retriever)
            except Exception as error:
                answer = (
                    "Groq rejected the API key. Update `.streamlit/secrets.toml` "
                    "with a valid GROQ_API_KEY, then restart Streamlit."
                )
                sources = []
                st.error(answer)
                st.caption(f"Details: {error}")
            else:
                st.write(answer)
                if sources:
                    st.caption("Sources: " + ", ".join(sources))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
