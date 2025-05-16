import os
import requests
from dotenv import load_dotenv

# LangChain Community: loaders, embeddings, vector stores
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# LangChain OpenAI-compatible LLMs (DeepSeek included)
from langchain_openai import ChatOpenAI

# Load your DeepSeek API key from .env
load_dotenv()

# Initialize DeepSeek LLM (OpenAI-compatible)
llm = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",  # Update if DeepSeek base URL differs
    model="deepseek-chat",
)

# Step 1: Load PDF
pdf_path = "documents/automotive_safety_and_homologation.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# Step 2: Split PDF content into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(pages)

# Step 3: Embed chunks into a vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(docs, embeddings)

# Step 4: Build the Retrieval Q&A chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_db.as_retriever())

# Step 5: Ask user for a question
print("Ask a question about the PDF:")
user_query = input("> ")
response = qa_chain.run(user_query)
print("\nAnswer:", response)


# Optional: Load and query data from a remote server
def query_remote_server(url, question):
    print(f"\n[Optional] Fetching data from: {url}")
    try:
        text_data = requests.get(url).text
        text_chunks = splitter.split_text(text_data)
        remote_vectors = FAISS.from_texts(text_chunks, embeddings)
        remote_chain = RetrievalQA.from_chain_type(
            llm=llm, retriever=remote_vectors.as_retriever()
        )
        return remote_chain.run(question)
    except Exception as e:
        return f"Failed to query remote server: {e}"


# Uncomment and use this if you want to try querying remote content
# remote_url = "https://example.com/some.txt"
# remote_answer = query_remote_server(remote_url, "What is this about?")
# print("Remote server answer:", remote_answer)
