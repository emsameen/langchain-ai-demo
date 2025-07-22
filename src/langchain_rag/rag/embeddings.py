from enum import Enum
from typing import List, Protocol, Optional
from abc import abstractmethod
from langchain_huggingface import HuggingFaceEmbeddings

# Optional imports
try:
    from langchain_openai import OpenAIEmbeddings

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


class EmbeddingType(Enum):
    """Enum for different types of embeddings."""

    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    GOOGLE = "google"


class HuggingFaceModel(Enum):
    """Enum for HuggingFace embedding models."""

    # Smaller models (384 dimensions)
    ALL_MINILM_L6 = (
        "sentence-transformers/all-MiniLM-L6-v2"  # 2021 - Fast, good for general text
    )
    ALL_MINILM_L12 = "sentence-transformers/all-MiniLM-L12-v2"  # 2021 - Larger version of L6, slightly better quality

    # Larger models (768 dimensions)
    ALL_DISTILROBERTA_V1 = "sentence-transformers/all-distilroberta-v1"  # 2021 - Stronger semantic understanding
    ALL_MPNET_BASE = "sentence-transformers/all-mpnet-base-v2"  # 2022 - Best performance for semantic search

    @property
    def dimensions(self) -> int:
        """Return the dimension of the model."""
        model_to_dim = {
            self.ALL_MINILM_L6: 384,
            self.ALL_MINILM_L12: 384,
            self.ALL_DISTILROBERTA_V1: 768,
            self.ALL_MPNET_BASE: 768,
        }
        return model_to_dim[self]


class OpenAIModel(Enum):
    """Enum for OpenAI embedding models."""

    TEXT_EMBEDDING_ADA = "text-embedding-ada-002"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_MEDIUM = "text-embedding-3-medium"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"

    @property
    def dimensions(self) -> int:
        """Return the dimension of the model."""
        model_to_dim = {
            self.TEXT_EMBEDDING_ADA: 1536,
            self.TEXT_EMBEDDING_3_SMALL: 1024,
            self.TEXT_EMBEDDING_3_MEDIUM: 2048,
            self.TEXT_EMBEDDING_3_LARGE: 4096,
        }
        return model_to_dim[self]


class GoogleModel(Enum):
    """Enum for Google embedding models."""

    # Multilingual models
    TEXT_EMBEDDING_MULTILINGUAL = (
        "textembedding-multilingual@latest"  # 2023 - Best for multilingual text
    )

    # English models
    TEXT_EMBEDDING_GEOCKO = (
        "textembedding-gecko@latest"  # 2023 - Latest version, best quality
    )
    TEXT_EMBEDDING_GEOCKO_V1 = "textembedding-gecko@v1"  # 2022 - Stable older version

    @property
    def dimensions(self) -> int:
        """Return the dimension of the model."""
        model_to_dim = {
            self.TEXT_EMBEDDING_GEOCKO: 768,
            self.TEXT_EMBEDDING_MULTILINGUAL: 768,
            self.TEXT_EMBEDDING_GEOCKO_V1: 768,
        }
        return model_to_dim[self]


class EmbeddingConfig:
    """
    Configuration class for embeddings.

    Attributes:
        type: The type of embedding (HUGGINGFACE, OPENAI, or GOOGLE)
        model: The specific model to use (Enum value)
        dimensions: Dimension of the embeddings
        api_key: API key for the provider (if required)
        description: Description of the embedding model
    """

    def __init__(
        self,
        type: EmbeddingType,
        model: Union[HuggingFaceModel, OpenAIModel, GoogleModel],
        api_key: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        Initialize the embedding configuration.

        Args:
            type: Type of embedding
            model: Specific model to use (Enum value)
            api_key: API key for the provider (if required)
            description: Description of the embedding model
        """
        self.type = type
        self.model = model
        self.dimensions = model.dimensions
        self.api_key = api_key
        self.description = description

    @property
    def provider(self) -> str:
        """Return the provider name based on the embedding type."""
        provider_map = {
            EmbeddingType.HUGGINGFACE: "HuggingFace",
            EmbeddingType.OPENAI: "OpenAI",
            EmbeddingType.GOOGLE: "Google",
        }
        return provider_map[self.type]

    @property
    def model_name(self) -> str:
        """Return the model name as string."""
        return self.model.value

    def __str__(self) -> str:
        return (
            f"EmbeddingConfig({self.type.value}, {self.model.value}, {self.dimensions})"
        )

    @property
    def name(self) -> str:
        """Return the model name."""
        return self.model_name

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return self.provider

    def __str__(self) -> str:
        return (
            f"EmbeddingConfig({self.type.value}, {self.model_name}, {self.dimensions})"
        )


class Embedding(Protocol):
    """
    Abstract protocol for embedding implementations.
    This defines the interface that all embedding implementations must implement.
    """

    @abstractmethod
    def __init__(self, config: EmbeddingConfig):
        """Initialize the embedding with configuration."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            List of embedding values
        """
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass


class HuggingFaceEmbedding(Embedding):
    """
    Implementation of Embedding for HuggingFace models.
    """

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize HuggingFace embedding.

        Args:
            config: Embedding configuration
        """
        if config.type != EmbeddingType.HUGGINGFACE:
            raise ValueError(
                f"Invalid config type for HuggingFace embedding: {config.type}"
            )

        self.config = config
        self.model = HuggingFaceEmbeddings(model_name=config.model_name)

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.model.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed_text(text) for text in texts]


class OpenAIEmbedding(Embedding):
    """
    Implementation of Embedding for OpenAI models.
    """

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize OpenAI embedding.

        Args:
            config: Embedding configuration
        """
        if config.type != EmbeddingType.OPENAI:
            raise ValueError(f"Invalid config type for OpenAI embedding: {config.type}")

        if not OPENAI_AVAILABLE:
            raise ImportError(
                "langchain_openai is not installed. Install it with 'pip install langchain_openai'"
            )

        if not config.api_key:
            raise ValueError("OpenAI API key is required for OpenAI embeddings")

        self.config = config
        self.model = OpenAIEmbeddings(
            model_name=config.model_name, openai_api_key=config.api_key
        )

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.model.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed_text(text) for text in texts]


class GoogleEmbedding(Embedding):
    """
    Implementation of Embedding for Google models.
    """

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize Google embedding.

        Args:
            config: Embedding configuration
        """
        if config.type != EmbeddingType.GOOGLE:
            raise ValueError(f"Invalid config type for Google embedding: {config.type}")

        if not GOOGLE_AVAILABLE:
            raise ImportError(
                "langchain_google_genai is not installed. Install it with 'pip install langchain_google_genai'"
            )

        if not config.api_key:
            raise ValueError("Google API key is required for Google embeddings")

        self.config = config
        self.model = GoogleGenerativeAIEmbeddings(
            model_name=config.model_name, google_api_key=config.api_key
        )

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.model.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed_text(text) for text in texts]
