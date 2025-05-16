from typing import List, Dict, Any, Optional, Tuple
from pinecone import Pinecone, ServerlessSpec
from co_genticai.config.settings import PINECONE_API_KEY, logger
from .vector_storage import VectorStorageHandler


class PineconeManager(VectorStorageHandler):
    """
    A class for managing Pinecone vector database operations.

    This class implements the VectorStorageHandler protocol for Pinecone.
    """

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
        Initialize the Pinecone storage handler.

        Args:
            index_name: Name of the Pinecone index
            dimension: Dimension of the vectors
            metric: Distance metric for similarity search
            cloud: Cloud provider
            region: Cloud region
            api_key: Pinecone API key
            proxy_url: Optional proxy URL for Pinecone client
        """
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region

        # Initialize Pinecone client
        pinecone_kwargs = {"api_key": api_key or PINECONE_API_KEY}
        if proxy_url:
            pinecone_kwargs["proxy_url"] = proxy_url

        self.pc = Pinecone(**pinecone_kwargs)

        # Initialize or connect to index
        self._initialize_index()

    def _initialize_index(self) -> None:
        """
        Initialize the Pinecone index if it doesn't exist.
        """
        # List existing indexes
        index_names = [index.name for index in self.pc.list_indexes().indexes]

        # Create index if it doesn't exist
        if self.index_name not in index_names:
            logger.info(f"Creating index {self.index_name}")
            self.pc.create_index(
                self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )

        # Connect to the index
        self.index = self.pc.Index(self.index_name)

    def upsert(
        self, id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Upsert a single vector to the Pinecone index.

        Args:
            id: Unique identifier for the vector
            vector: The vector values
            metadata: Optional metadata to store with the vector
        """
        self.index.upsert([(id, vector, metadata)])

    def upsert_batch(
        self, batch: List[Tuple[str, List[float], Optional[Dict[str, Any]]]]
    ) -> None:
        """
        Upsert a batch of vectors to the Pinecone index.

        Args:
            batch: List of tuples containing (id, vector, metadata)
        """
        self.index.upsert(batch)

    def query(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query the Pinecone index for similar vectors.

        Args:
            query_vector: The query vector
            top_k: Number of results to return
            filter: Optional filter dictionary

        Returns:
            List of matched vectors with scores and metadata
        """
        try:
            results = self.index.query(
                queries=[query_vector], top_k=top_k, filter=filter
            )

            return [
                {"id": match.id, "score": match.score, "metadata": match.metadata}
                for match in results[0].matches
            ]
        except Exception as e:
            logger.error(f"Error querying index: {e}")
            raise

    def delete(self, ids: List[str]) -> None:
        """
        Delete vectors from the Pinecone index.

        Args:
            ids: List of vector IDs to delete
        """
        self.index.delete(ids=ids)

    def delete_by_filter(self, filter: Dict[str, Any]) -> None:
        """
        Delete vectors from the Pinecone index using a filter.

        Args:
            filter: Filter dictionary to match vectors for deletion
        """
        self.index.delete(filter=filter)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Pinecone index.

        Returns:
            Dictionary containing index statistics
        """
        return self.index.describe_index_stats()

    def get_vector(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single vector by ID.

        Args:
            id: Vector ID to retrieve

        Returns:
            Dictionary containing the vector and metadata, or None if not found
        """
        try:
            result = self.index.fetch(ids=[id])
            if result.vectors:
                return {
                    "id": id,
                    "values": result.vectors[id].values,
                    "metadata": result.vectors[id].metadata,
                }
            return None
        except Exception:
            return None

    def fetch(self, ids: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Fetch vectors from the Pinecone index by ID.

        Args:
            ids: Single ID or list of IDs to fetch

        Returns:
            Dictionary containing the fetched vectors
        """
        if isinstance(ids, str):
            ids = [ids]
        return self.index.fetch(ids=ids)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Pinecone index.

        Returns:
            Dictionary containing index statistics
        """
        return self.index.describe_index_stats()

    def close(self) -> None:
        """
        Close the Pinecone client connection.
        """
        # Currently, Pinecone doesn't require explicit connection closing
        pass
