from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Read your text file
loader = TextLoader("docs/support.txt")
documents = loader.load()
print("1. Successfully read your support.txt file!")

# 2. Cut the text into smaller paragraphs (chunks)
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"2. Cut the text into {len(chunks)} small pieces.")

# 3. Choose the tool that turns words into numbers
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Put those pieces into your ChromaDB storage box
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)
print("3. Success! Your pieces are now saved inside ChromaDB.")
