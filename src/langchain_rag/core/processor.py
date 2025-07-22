"""Core processor for RAG operations."""

from typing import Dict, List, Optional, Union

from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
from langchain_core.language_models import BaseLLM

from langchain_rag.rag.embeddings import get_embeddings


class RAGProcessor:
    """Main processor for RAG operations.
    
    This class handles the core Retrieval Augmented Generation processing,
    combining retrieval from vector stores with LLM generation.
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        retriever: Optional[BaseRetriever] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 5,
    ):
        """Initialize the RAG processor.
        
        Args:
            llm: Language model to use for generation
            retriever: Optional retriever component. If None, must be set later.
            embedding_model: Name of the embedding model to use
            top_k: Number of documents to retrieve
        """
        self.llm = llm
        self.retriever = retriever
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.qa_chain = None
        
        if self.retriever:
            self._setup_qa_chain()
    
    def _setup_qa_chain(self) -> None:
        """Set up the QA chain with the current retriever and LLM."""
        if not self.retriever:
            raise ValueError("Retriever must be set before setting up QA chain")
            
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
        )
    
    def set_retriever(self, retriever: BaseRetriever) -> None:
        """Set the retriever component.
        
        Args:
            retriever: Retriever component to use
        """
        self.retriever = retriever
        self._setup_qa_chain()
    
    def process(self, query: str) -> Dict[str, Union[str, List[str]]]:
        """Process a query through the RAG pipeline.
        
        Args:
            query: User query to process
            
        Returns:
            Dictionary containing the answer and retrieved sources
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Set a retriever first.")
            
        result = self.qa_chain({"query": query})
        
        return {
            "query": query,
            "answer": result.get("result", ""),
            "sources": result.get("source_documents", []),
        }
