"""Data models and schemas for the langchain_rag package."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

@dataclass
class RAGDocument:
    """Represents a document in the RAG system."""
    
    content: str
    metadata: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)
    id: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        # If no ID is provided, use content hash as ID
        if self.id is None:
            import hashlib
            self.id = hashlib.md5(self.content.encode()).hexdigest()

@dataclass
class RAGQuery:
    """Represents a query in the RAG system."""
    
    text: str
    filters: Dict[str, Union[str, int, float, bool, List]] = field(default_factory=dict)
    top_k: int = 5

@dataclass
class RAGResponse:
    """Represents a response from the RAG system."""
    
    query: str
    answer: str
    sources: List[RAGDocument] = field(default_factory=list)
    metadata: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)
    
    def get_source_texts(self) -> List[str]:
        """Get the text content of all sources.
        
        Returns:
            List of source document contents
        """
        return [doc.content for doc in self.sources]
    
    def get_source_ids(self) -> List[str]:
        """Get the IDs of all sources.
        
        Returns:
            List of source document IDs
        """
        return [doc.id for doc in self.sources]
