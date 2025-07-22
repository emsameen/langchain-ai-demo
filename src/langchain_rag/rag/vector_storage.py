from typing import List, Dict, Any, Protocol
from abc import abstractmethod
from co_genticai.config.settings import logger


class VectorStorageHandler(Protocol):
    """
    Abstract protocol for vector storage handlers.
    This defines the interface that all storage handlers must implement.
    """

    @abstractmethod
    def __init__(
        self,
        index_name: str,
        dimension: int,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        api_key: Optional[str] = None,
        proxy_url: Optional[str] = None,
    ):
        """
        Initialize the vector storage handler.

        Args:
            index_name: Name of the vector index
            dimension: Dimension of the vectors
            metric: Distance metric for similarity search
            cloud: Cloud provider
            region: Cloud region
            api_key: API key for the storage system
            proxy_url: Optional proxy URL
        """
        pass

    @abstractmethod
    def upsert_embeddings(self, embeddings: Dict[str, Any], document_id: str) -> None:
        """
        Upsert embeddings to the storage system.

        Args:
            embeddings: Dictionary containing embeddings (chunks, full, or both)
            document_id: ID to use for the document
        """
        pass

    @abstractmethod
    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the storage system.

        Args:
            query_embedding: The query embedding to search with
            top_k: Number of results to return
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """
        Delete vectors from the storage system.

        Args:
            ids: List of vector IDs to delete
        """
        pass

    @abstractmethod
    def delete_by_filter(self, filter: Dict[str, Any]) -> None:
        """
        Delete vectors from the storage system using a filter.

        Args:
            filter: Filter dictionary to match vectors for deletion
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the storage system.

        Returns:
            Dictionary containing storage statistics
        """
        pass

    @abstractmethod
    def get_vector(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single vector by ID.

        Args:
            id: Vector ID to retrieve

        Returns:
            Dictionary containing the vector and metadata, or None if not found
        """
        pass

    @abstractmethod
    def fetch(self, ids: List[str]) -> Dict[str, Any]:
        """
        Fetch multiple vectors by their IDs.

        Args:
            ids: List of vector IDs to fetch

        Returns:
            Dictionary mapping IDs to their vectors and metadata
        """
        pass

    """
    Implementation of VectorStorageHandler for Pinecone.
    """

    def __init__(
        self,
        index_name: str,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        api_key: Optional[str] = None,
        proxy_url: Optional[str] = None,
    ):
        """
        Initialize the Pinecone storage handler.

        Args:
            index_name: Name of the Pinecone index
            metric: Distance metric for similarity search
            cloud: Cloud provider
            region: Cloud region
            api_key: Pinecone API key
            proxy_url: Optional proxy URL for Pinecone client
        """
        self.pinecone_manager = PineconeManager(
            index_name=index_name,
            metric=metric,
            cloud=cloud,
            region=region,
            api_key=api_key,
            proxy_url=proxy_url,
        )

    def upsert_embeddings(self, embeddings: Dict[str, Any], document_id: str) -> None:
        """
        Upsert embeddings to Pinecone.

        Args:
            embeddings: Dictionary containing embeddings (chunks, full, or both)
            document_id: ID to use for the document
        """
        try:
            records = []

            # Handle chunk embeddings
            if "chunks" in embeddings:
                for chunk in embeddings["chunks"]:
                    records.append(
                        {
                            "id": f'{document_id}_chunk_{chunk["id"]}',
                            "values": chunk["embedding"],
                            "metadata": {"text": chunk["text"], "type": "chunk"},
                        }
                    )

            # Handle full document embedding
            if "full" in embeddings:
                records.append(
                    {
                        "id": f"{document_id}_full",
                        "values": embeddings["full"]["embedding"],
                        "metadata": {
                            "text": embeddings["full"]["text"],
                            "type": "full",
                        },
                    }
                )

            if records:
                self.pinecone_manager.upsert(records)

        except Exception as e:
            logger.error(f"Error upserting embeddings: {e}")
            raise

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in Pinecone.

        Args:
            query_embedding: The query embedding to search with
            top_k: Number of results to return
        """
        try:
            return self.pinecone_manager.query(
                query_embedding=query_embedding, top_k=top_k
            )
        except Exception as e:
            logger.error(f"Error searching embeddings: {e}")
            raise
