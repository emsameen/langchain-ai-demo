"""Tests for the data.loaders module."""

import os
import pytest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_rag.data.loaders import (
    chunk_documents,
    documents_to_rag_documents,
)
from langchain_rag.models.schemas import RAGDocument


def test_documents_to_rag_documents(sample_documents):
    """Test converting LangChain Documents to RAGDocuments."""
    rag_docs = documents_to_rag_documents(sample_documents)
    
    # Check we have the same number of documents
    assert len(rag_docs) == len(sample_documents)
    
    # Check the content and metadata is preserved
    for i, doc in enumerate(sample_documents):
        assert rag_docs[i].content == doc.page_content
        assert rag_docs[i].metadata == doc.metadata
        # Check ID was generated
        assert rag_docs[i].id is not None


def test_chunk_documents_langchain_docs(sample_documents):
    """Test chunking LangChain documents."""
    # Make a longer document for better testing
    long_text = "This is a test document. " * 20
    sample_documents[0].page_content = long_text
    
    chunked_docs = chunk_documents(sample_documents, chunk_size=100, chunk_overlap=20)
    
    # Check we have more documents after chunking
    assert len(chunked_docs) > len(sample_documents)
    
    # Check metadata is preserved and enhanced
    for doc in chunked_docs:
        assert "chunk" in doc.metadata
        assert "chunk_total" in doc.metadata
        
        # Check original metadata is preserved
        if "source" in doc.metadata:
            assert doc.metadata["source"] in ["langchain_docs", "rag_docs"]


def test_chunk_documents_rag_docs(sample_rag_documents):
    """Test chunking RAG documents."""
    # Make a longer document for better testing
    long_text = "This is a test document. " * 20
    sample_rag_documents[0].content = long_text
    
    chunked_docs = chunk_documents(sample_rag_documents, chunk_size=100, chunk_overlap=20)
    
    # Check we have more documents after chunking
    assert len(chunked_docs) > len(sample_rag_documents)
    
    # Check all are RAGDocument instances
    assert all(isinstance(doc, RAGDocument) for doc in chunked_docs)
    
    # Check metadata is preserved and enhanced
    for doc in chunked_docs:
        assert "chunk" in doc.metadata
        assert "chunk_total" in doc.metadata
