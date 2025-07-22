"""Tests for the utils.helpers module."""

import os
import pytest
from langchain_rag.utils.helpers import chunk_text, load_env_variables


def test_chunk_text():
    """Test the chunk_text function."""
    text = "This is a test text. It should be split into chunks. This is the third sentence."
    chunks = chunk_text(text, chunk_size=30, overlap=5)
    
    # Check we have the expected number of chunks
    assert len(chunks) > 1
    
    # Check the first chunk begins with the beginning of the text
    assert chunks[0].startswith("This is a test text")
    
    # Check overlap works (there should be some overlap between chunks)
    assert any(c[-5:] in chunks[i+1] for i, c in enumerate(chunks[:-1]))


def test_load_env_variables(mock_env_variables):
    """Test loading environment variables."""
    env_vars = load_env_variables()
    
    # Check that all expected environment variables are loaded
    assert "OPENAI_API_KEY" in env_vars
    assert "PINECONE_API_KEY" in env_vars
    assert "PINECONE_ENVIRONMENT" in env_vars
    
    # Check that the values match our mock values
    assert env_vars["OPENAI_API_KEY"] == mock_env_variables["OPENAI_API_KEY"]
    assert env_vars["PINECONE_API_KEY"] == mock_env_variables["PINECONE_API_KEY"]
    assert env_vars["PINECONE_ENVIRONMENT"] == mock_env_variables["PINECONE_ENVIRONMENT"]
