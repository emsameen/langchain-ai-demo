import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from co_genticai.config.settings import PINECONE_API_KEY


def write_to_file(filename, embeddings):
    with open(filename, "w") as file:
        file.write(str(embeddings))


# --- STEP 1: Load a sample of the Food.com dataset ---
df = pd.read_csv("data/RAW_recipes.csv")
names = "Name: " + df["name"].fillna("") + "\n"
ingredients = "Ingredients: " + df["ingredients"].fillna("").astype(str) + "\n"
instructions = "Instructions: " + df["steps"].fillna("").astype(str) + "\n"
tags = "Tags: " + df["tags"].fillna("").astype(str) + "\n"

df["text"] = names + ingredients + instructions + tags

texts = df["text"].tolist()
ids = df["id"].astype(str).tolist()

# --- STEP 2: Create embeddings using SentenceTransformers via LangChain ---
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --- STEP 3: Initialize Pinecone (2.x API) ---
# Set your Pinecone environment and index name
PINECONE_ENV = "us-west1-gcp"  # <--- Replace with your environment
INDEX_NAME = "recipes"  # <--- Replace with your index name

from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone client with proxy support (v3.x)
pc = Pinecone(api_key=PINECONE_API_KEY, proxy_url="http://localhost:3128")

# List indexes
index_names = [index.name for index in pc.list_indexes().indexes]
logger.info(f"Available indexes: {index_names}")

# Create index if it doesn't exist
if INDEX_NAME not in index_names:
    logger.info(f"Creating index {INDEX_NAME}")
    pc.create_index(
        INDEX_NAME,
        dimension=384,  # Set to your embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Connect to the index
index = pc.Index(INDEX_NAME)

text_size = len(texts) 
logger.info(f"Text size: {text_size}")
batch_size = 10
batch = []
for i in range(text_size):
    logger.info(f"Processing {i}/{text_size} -> {ids[i]}")
    embedding = embedding_model.embed_query(texts[i])
    batch.append((ids[i], embedding))
    if i < 10:
        write_to_file(f"data/embeddings/{ids[i]}.txt", embedding)
    # Upsert in batches
    if len(batch) == batch_size or i == text_size - 1:
        index.upsert(batch)
        batch = []

logger.info(f"Embeddings upserted to Pinecone {INDEX_NAME} successfully.")
