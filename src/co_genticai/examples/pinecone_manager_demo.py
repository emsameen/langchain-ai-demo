#!/usr/bin/env python3
"""
A simple script demonstrating the usage of the PineconeManager class.

This script shows how to:
1. Initialize a PineconeManager
2. Upsert data to Pinecone
3. Query data from Pinecone
4. Fetch and delete vectors
"""

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

# Sample data for demonstration
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog",
    "A journey of a thousand miles begins with a single step",
    "To be or not to be, that is the question",
    "All that glitters is not gold",
    "The early bird catches the worm"
]
SAMPLE_IDS = ["text1", "text2", "text3", "text4", "text5"]
SAMPLE_METADATA = [
    {"category": "proverb", "language": "english", "words": 9},
    {"category": "proverb", "language": "chinese", "words": 11},
    {"category": "literature", "language": "english", "words": 10},
    {"category": "proverb", "language": "english", "words": 6},
    {"category": "proverb", "language": "english", "words": 7}
]


def main():
    """Main function demonstrating PineconeManager usage."""
    # Step 1: Initialize the storage handler
    logger.info("Initializing Pinecone storage handler")
    storage_handler = PineconeManager(
        index_name="quickstart",
        dimension=384,  # Dimension for all-MiniLM-L6-v2
        proxy_url="http://localhost:3128"
    )
    
    # Step 2: Initialize the embedding manager
    logger.info("Initializing embedding manager")
    embedding_manager = EmbeddingManager(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        file_type=FileType.TXT,
        file_path="path/to/file.txt",
        output_type=EmbeddingOutputType.FULL
    )
    
    # Step 3: Get index statistics before adding data
    stats_before = storage_handler.get_stats()
    logger.info(f"Index stats before upsert: {stats_before}")
    
    # Step 4: Generate embeddings for our sample texts
    logger.info("Generating embeddings for sample texts")
    embeddings = embedding_manager.generate_embeddings(
        texts=SAMPLE_TEXTS,
        ids=SAMPLE_IDS,
        metadatas=SAMPLE_METADATA,
        batch_size=2  # Small batch size for demonstration
    )
    
    # Step 5: Upsert sample data with metadata
    logger.info("Upserting sample texts to Pinecone...")
    storage_handler.upsert_from_embeddings(
        embeddings=embeddings,
        ids=SAMPLE_IDS,
        metadatas=SAMPLE_METADATA,
        batch_size=2  # Small batch size for demonstration
    )
    
    # Step 6: Get index statistics after adding data
    stats_after = storage_handler.get_stats()
    stats_after = manager.get_stats()
    logger.info(f"Index stats after upsert: {stats_after}")
    
    # Step 5: Query the index with a text
    query_text = "What is the meaning of life?"
    logger.info(f"Querying with text: '{query_text}'")
    results = manager.query_from_text(
        query_text=query_text,
        top_k=3,
        include_metadata=True
    )
    
    # Display query results
    logger.info("Query results:")
    for match in results.get("matches", []):
        logger.info(f"ID: {match.get('id')}, Score: {match.get('score'):.4f}")
        logger.info(f"Metadata: {match.get('metadata')}")
        logger.info("---")
    
    # Step 6: Fetch specific vectors by ID
    fetch_id = SAMPLE_IDS[0]
    logger.info(f"Fetching vector with ID: {fetch_id}")
    fetch_result = manager.fetch(fetch_id)
    logger.info(f"Fetched vector metadata: {fetch_result.get('vectors', {}).get(fetch_id, {}).get('metadata')}")
    
    # Step 7: Delete a specific vector
    delete_id = SAMPLE_IDS[-1]
    logger.info(f"Deleting vector with ID: {delete_id}")
    manager.delete(delete_id)
    
    # Step 8: Verify deletion by checking stats
    stats_final = manager.get_stats()
    logger.info(f"Index stats after deletion: {stats_final}")
    
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    main()
