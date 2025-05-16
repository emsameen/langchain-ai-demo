#!/usr/bin/env python3
"""
Demonstration of using PineconeManager with different embedding models.

This script shows how to:
1. Initialize PineconeManager with different embedding models
2. Compare embeddings from different models
3. Demonstrate how to use different models for different use cases
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[3].resolve()
sys.path.append(str(project_root))

from co_genticai.rag.pinecone_manager import PineconeManager
from co_genticai.config.settings import logger
from co_genticai.rag.vector_storage import VectorStorageHandler
from co_genticai.rag.embedding_manager import EmbeddingManager
from co_genticai.rag.embeddings import EmbeddingOutputType
from co_genticai.rag.filetypes import FileType

# Sample text for demonstration
SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog"


def compare_embedding_dimensions():
    """Compare embedding dimensions from different models."""
    logger.info("Comparing embedding dimensions from different models")
    
    # Select one model from each provider type
    models_to_test = {
        "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
    }
    
    # Only add OpenAI and Google models if the packages are available
    try:
        from langchain_openai import OpenAIEmbeddings
        models_to_test["openai"] = "text-embedding-3-small"
    except ImportError:
        logger.warning("langchain_openai not available, skipping OpenAI models")
        
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        models_to_test["google"] = "textembedding-gecko@latest"
    except ImportError:
        logger.warning("langchain_google_genai not available, skipping Google models")
    
    for provider, model_name in models_to_test.items():
        try:
            # Skip if API keys aren't available for certain providers
            if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
                logger.warning(f"Skipping {model_name} as OPENAI_API_KEY is not set")
                continue
            if provider == "google" and not os.getenv("GOOGLE_API_KEY"):
                logger.warning(f"Skipping {model_name} as GOOGLE_API_KEY is not set")
                continue
            
            logger.info(f"Initializing storage handler for {model_name}")
            storage_handlers[provider] = PineconeManager(
                index_name=f"multi_model_demo_{provider}",
                dimension=384,  # Use consistent dimension for all models
                proxy_url="http://localhost:3128"
            )
            
            logger.info(f"Initializing embedding manager for {model_name}")
            embedding_managers[provider] = EmbeddingManager(
                embedding_model_name=model_name,
                file_type=FileType.TXT,
                file_path="path/to/file.txt",
                output_type=EmbeddingOutputType.FULL
            )
            
            # Generate and upsert embedding
            embeddings = embedding_managers[provider].generate_embeddings()
            storage_handlers[provider].upsert_embeddings(embeddings, document_id="doc_1")
            
            logger.info(f"{provider} model dimension: {len(embeddings['full']['embedding'])}")
        except Exception as e:
            logger.error(f"Error with {provider} model: {e}")


def semantic_search_demo():
    """Demonstrate semantic search with different models."""
    # Sample data
    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Natural language processing helps computers understand human language",
        "Vector databases store and retrieve high-dimensional vectors efficiently",
        "Embeddings represent text as numerical vectors",
        "Pinecone is a vector database optimized for similarity search"
    ]
    ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    
    # Query text
    query = "How do computers process language?"
    
    # Use HuggingFace model (works without API keys)
    logger.info("Semantic search demo with HuggingFace model")
    try:
        manager = PineconeManager(
            index_name="demo-semantic-search",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", 
            proxy_url="http://localhost:3128"
        )
        
        # Upsert data
        logger.info("Upserting sample documents...")
        manager.upsert_from_texts(texts=texts, ids=ids)
        
        # Query
        logger.info(f"Querying with: '{query}'")
        results = manager.query_from_text(query_text=query, top_k=2)
        
        # Display results
        logger.info(f"Results:{results}")
        for match in results.get("matches", []):
            doc_id = match.get("id")
            score = match.get("score")
            doc_index = ids.index(doc_id)
            logger.info(f"Document: {texts[doc_index]}")
            logger.info(f"Score: {score:.4f}")
            logger.info("---")
            
        # Clean up
        logger.info("Cleaning up...")
        manager.delete_all()
        
    except Exception as e:
        logger.error(f"Error in semantic search demo: {e}")


if __name__ == "__main__":
    logger.info("Starting multi-model Pinecone demo")
    
    # Run the embedding dimension comparison
    compare_embedding_dimensions()
    
    # Run the semantic search demo
    semantic_search_demo()
    
    logger.info("Demo completed")
