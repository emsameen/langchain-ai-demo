"""Pytest configuration for langchain_rag tests."""

import os
import pytest
from typing import Dict, List

from langchain_core.documents import Document
from langchain_rag.models.schemas import RAGDocument


@pytest.fixture
def sample_documents() -> List[Document]:
    """Fixture providing sample documents for testing."""
    return [
        Document(
            page_content="LangChain is a framework for developing applications powered by language models.",
            metadata={"source": "langchain_docs", "page": 1}
        ),
        Document(
            page_content="RAG (Retrieval Augmented Generation) combines retrieval systems with LLMs.",
            metadata={"source": "rag_docs", "page": 1}
        )
    ]


@pytest.fixture
def sample_rag_documents() -> List[RAGDocument]:
    """Fixture providing sample RAG documents for testing."""
    return [
        RAGDocument(
            content="LangChain is a framework for developing applications powered by language models.",
            metadata={"source": "langchain_docs", "page": 1},
            id="doc1"
        ),
        RAGDocument(
            content="RAG (Retrieval Augmented Generation) combines retrieval systems with LLMs.",
            metadata={"source": "rag_docs", "page": 1},
            id="doc2"
        )
    ]


@pytest.fixture
def mock_env_variables() -> Dict[str, str]:
    """Fixture providing mock environment variables for testing."""
    # Set test environment variables
    os.environ["OPENAI_API_KEY"] = "test-openai-key"
    os.environ["PINECONE_API_KEY"] = "test-pinecone-key"
    os.environ["PINECONE_ENVIRONMENT"] = "test-env"
    
    # Return the variables for use in tests
    return {
        "OPENAI_API_KEY": "test-openai-key",
        "PINECONE_API_KEY": "test-pinecone-key",
        "PINECONE_ENVIRONMENT": "test-env",
    }
    
    # Clean up is handled by pytest's automatic fixture teardown
