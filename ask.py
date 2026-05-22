import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# 1. Load the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Open your ChromaDB
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# 3. Connect to Groq
llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"],
    model_name="llama-3.1-8b-instant"
)

# 4. Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 5. Ask function
def ask(question):
    docs = retriever.invoke(question)
    context = "\n".join(d.page_content for d in docs)
    response = llm.invoke(f"Answer using only this context:\n{context}\n\nQuestion: {question}")
    return response.content

# 6. Chat loop
print("\n🤖 RAG Support Agent Ready! Type 'quit' to exit.\n")

while True:
    question = input("You: ").strip()
    if question.lower() == "quit":
        print("Goodbye!")
        break
    if not question:
        continue
    answer = ask(question)
    print(f"\nAgent: {answer}\n")
