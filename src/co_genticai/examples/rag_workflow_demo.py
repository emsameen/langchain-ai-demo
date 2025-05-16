"""
A comprehensive demo showing the complete RAG workflow:
1. Embedding generation
2. Vector storage
3. Similarity search
"""

import sys
from pathlib import Path
import logging

# Add the project root to the Python path
project_root = Path(__file__).parents[3].resolve()
sys.path.append(str(project_root))

from co_genticai.rag.embeddings import (
    EmbeddingType,
    HuggingFaceModel,
    EmbeddingConfig
)
from co_genticai.rag.embedding_manager import (
    EmbeddingManager,
    FileType,
    EmbeddingOutputType
)
from co_genticai.rag.pinecone_manager import PineconeManager
from co_genticai.config.settings import logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

def main():
    """Main function demonstrating the RAG workflow."""
    try:
        # Step 1: Configure embedding
        logger.info("Configuring embedding")
        embedding_config = EmbeddingConfig(
            type=EmbeddingType.HUGGINGFACE,
            model=HuggingFaceModel.ALL_MINILM_L6
        )
        
        # Step 2: Initialize embedding manager
        logger.info("Initializing embedding manager")
        embedding_manager = EmbeddingManager(
            embedding_config=embedding_config,
            file_type=FileType.MARKDOWN,
            file_path="path/to/your/document.md",
            output_type=EmbeddingOutputType.BOTH,
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Step 3: Initialize vector storage
        logger.info("Initializing vector storage")
        storage_handler = PineconeManager(
            index_name="rag-demo",
            dimension=embedding_config.dimensions,
            metric="cosine",
            cloud="aws",
            region="us-east-1"
        )
        
        # Step 4: Process file and store embeddings
        logger.info("Processing file and storing embeddings")
        document_id = "doc_1"
        embeddings = embedding_manager.process_file(
            storage_handler=storage_handler,
            document_id=document_id
        )
        
        logger.info(f"Stored embeddings for document: {document_id}")
        
        # Step 5: Perform a search
        logger.info("Performing similarity search")
        query_text = "What is the main topic of the document?"
        
        # Generate query embedding
        query_embedding = embedding_manager.embedding.embed_text(query_text)
        
        # Search for similar documents
        results = storage_handler.search(
            query_embedding=query_embedding,
            top_k=3
        )
        
        logger.info("Search results:")
        for i, result in enumerate(results, 1):
            logger.info(f"\nResult {i}:")
            logger.info(f"Score: {result['score']:.4f}")
            logger.info(f"Type: {result['metadata']['type']}")
            logger.info(f"Text: {result['metadata']['text'][:200]}...")
            
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        raise

if __name__ == "__main__":
    main()
