"""Helper utilities for the langchain_rag package."""

import os
from typing import Dict, List, Optional, Union

from loguru import logger


def setup_logger(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Set up logger configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    """
    logger.remove()  # Remove default handler
    
    # Add console handler
    logger.add(
        lambda msg: print(msg, end=""),
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            rotation="10 MB",
            retention="1 week",
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
        )


def load_env_variables(env_file: str = ".env") -> Dict[str, str]:
    """Load environment variables from .env file.
    
    Args:
        env_file: Path to .env file
        
    Returns:
        Dictionary of loaded environment variables
    """
    from dotenv import load_dotenv
    
    # Load .env file
    load_dotenv(env_file)
    
    # Return relevant environment variables
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY", ""),
        "PINECONE_ENVIRONMENT": os.getenv("PINECONE_ENVIRONMENT", ""),
        "HUGGINGFACE_API_KEY": os.getenv("HUGGINGFACE_API_KEY", ""),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
    }


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into chunks of specified size with overlap.
    
    Args:
        text: Text to split
        chunk_size: Size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # Try to find a good breakpoint
        if end < len(text):
            # Try to find the end of a sentence or paragraph
            for breakpoint in [". ", ".\n", "\n\n", " "]:
                pos = text.rfind(breakpoint, start, end)
                if pos != -1:
                    end = pos + len(breakpoint)
                    break
        
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks
