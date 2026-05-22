import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

st.set_page_config(
    page_title="SupportAI",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #1a1a2e; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    .stChatMessage { border-radius: 12px; }
    .stChatInput input { border-radius: 20px !important; }
    header { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/bot.png", width=48)
    st.title("SupportAI")
    st.caption("Powered by LLaMA 3.1 + RAG")
    st.divider()
    st.markdown("**Navigation**")
    st.markdown("💬 **Chat**")
    st.divider()
    st.markdown("**Documents loaded**")
    import os
    docs = os.listdir("docs/") if os.path.exists("docs/") else []
    for doc in docs:
        st.markdown(f"- {doc}")
    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.caption("v1.0 · SupportAI")

# Load models
@st.cache_resource
def load_chain():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )
    llm = ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.1-8b-instant"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    return llm, retriever

llm, retriever = load_chain()

# Main chat area
st.title("Support Assistant")
st.caption("Ask me anything about your account, billing, or technical issues.")

# Suggestion buttons
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.session_state.messages = []
    cols = st.columns(4)
    suggestions = ["Reset password", "Cancel subscription", "Payment issues", "Contact support"]
    for i, col in enumerate(cols):
        if col.button(suggestions[i], use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": suggestions[i]})
            st.rerun()

# Welcome message
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.write("👋 Hello! I'm your Support Assistant. How can I help you today?")
# Show messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
if question := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            docs = retriever.invoke(question)
            context = "\n".join(d.page_content for d in docs)
            response = llm.invoke(
                f"Answer using only this context:\n{context}\n\nQuestion: {question}"
            )
            answer = response.content
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})