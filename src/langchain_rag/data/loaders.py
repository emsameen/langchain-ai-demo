"""Document loaders for the langchain_rag package."""

import os
from typing import Dict, List, Optional, Union

from langchain_community.document_loaders import (
    DirectoryLoader,
    PDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

from langchain_rag.models.schemas import RAGDocument
from langchain_rag.utils.helpers import chunk_text


def load_documents(path: str, recursive: bool = True) -> List[Document]:
    """Load documents from a file or directory.
    
    Args:
        path: Path to file or directory
        recursive: Whether to search subdirectories
        
    Returns:
        List of Document objects
    """
    if os.path.isfile(path):
        return load_file(path)
    elif os.path.isdir(path):
        return load_directory(path, recursive=recursive)
    else:
        raise ValueError(f"Path does not exist: {path}")


def load_file(file_path: str) -> List[Document]:
    """Load a single file based on its extension.
    
    Args:
        file_path: Path to file
        
    Returns:
        List of Document objects
    """
    _, ext = os.path.splitext(file_path.lower())
    
    if ext == ".pdf":
        loader = PDFLoader(file_path)
    elif ext == ".md":
        loader = UnstructuredMarkdownLoader(file_path)
    elif ext in [".txt", ".csv", ".json"]:
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    
    return loader.load()


def load_directory(
    dir_path: str, 
    recursive: bool = True,
    glob_pattern: str = "**/*.*",
) -> List[Document]:
    """Load all documents from a directory.
    
    Args:
        dir_path: Path to directory
        recursive: Whether to search subdirectories
        glob_pattern: Pattern for matching files
        
    Returns:
        List of Document objects
    """
    # Define loaders for different file types
    loaders = {
        ".pdf": PDFLoader,
        ".md": UnstructuredMarkdownLoader,
        ".txt": TextLoader,
        ".csv": TextLoader,
        ".json": TextLoader,
    }
    
    # Create a loader for each supported file type
    loader = DirectoryLoader(
        dir_path,
        glob=glob_pattern,
        loader_cls=lambda x: _get_loader_for_file(x, loaders),
        recursive=recursive,
        show_progress=True,
    )
    
    return loader.load()


def _get_loader_for_file(file_path: str, loaders: Dict) -> Optional:
    """Get the appropriate loader for a file based on its extension.
    
    Args:
        file_path: Path to file
        loaders: Dictionary mapping extensions to loader classes
        
    Returns:
        Loader instance or None if extension not supported
    """
    _, ext = os.path.splitext(file_path.lower())
    
    if ext in loaders:
        return loaders[ext](file_path)
    return None


def documents_to_rag_documents(documents: List[Document]) -> List[RAGDocument]:
    """Convert LangChain Documents to RAGDocuments.
    
    Args:
        documents: List of LangChain Document objects
        
    Returns:
        List of RAGDocument objects
    """
    return [
        RAGDocument(
            content=doc.page_content,
            metadata=doc.metadata,
        )
        for doc in documents
    ]


def chunk_documents(
    documents: List[Union[Document, RAGDocument]],
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> List[Union[Document, RAGDocument]]:
    """Split documents into smaller chunks.
    
    Args:
        documents: List of Document or RAGDocument objects
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunked Document or RAGDocument objects
    """
    chunked_docs = []
    
    for doc in documents:
        if isinstance(doc, Document):
            content = doc.page_content
            metadata = doc.metadata
        else:  # RAGDocument
            content = doc.content
            metadata = doc.metadata
        
        chunks = chunk_text(content, chunk_size, chunk_overlap)
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk"] = i
            chunk_metadata["chunk_total"] = len(chunks)
            
            if isinstance(doc, Document):
                chunked_docs.append(Document(page_content=chunk, metadata=chunk_metadata))
            else:  # RAGDocument
                chunked_docs.append(RAGDocument(content=chunk, metadata=chunk_metadata))
    
    return chunked_docs
