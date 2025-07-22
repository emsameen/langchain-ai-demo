from typing import List, Dict, Any, Optional
from langchain.document_loaders import TextLoader, CSVLoader, JSONLoader
from langchain.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .vector_storage import VectorStorageHandler
from .embeddings import EmbeddingType, EmbeddingConfig
from co_genticai.config.settings import logger
import os


class FileType(Enum):
    """Enum for supported file types."""

    CSV = "csv"
    TXT = "txt"
    PDF = "pdf"
    JSON = "json"
    MARKDOWN = "markdown"


class EmbeddingOutputType(Enum):
    """Enum for different embedding output formats."""

    CHUNKS = "chunks"  # Individual chunks with embeddings
    FULL = "full"  # Full document embedding
    BOTH = "both"  # Both chunks and full document


def get_file_type(file_path: str) -> FileType:
    """
    Get the file type based on the file extension.

    Args:
        file_path: Path to the file

    Returns:
        FileType enum value

    Raises:
        ValueError: If file extension is not supported
    """
    extension = os.path.splitext(file_path)[1].lower()
    file_type_map = {
        ".csv": FileType.CSV,
        ".txt": FileType.TXT,
        ".pdf": FileType.PDF,
        ".json": FileType.JSON,
        ".md": FileType.MARKDOWN,
    }

    if extension not in file_type_map:
        raise ValueError(f"Unsupported file extension: {extension}")

    return file_type_map[extension]


def get_supported_file_extensions() -> List[str]:
    """
    Get a list of supported file extensions.

    Returns:
        List of supported file extensions (including leading .)
    """
    return [f".{file_type.value}" for file_type in FileType]


def is_supported_file_extension(extension: str) -> bool:
    """
    Check if a file extension is supported.

    Args:
        extension: File extension (with or without leading .)

    Returns:
        True if the extension is supported, False otherwise
    """
    extension = extension.lower()
    if not extension.startswith("."):  # Add leading . if missing
        extension = f".{extension}"
    return extension in get_supported_file_extensions()


class EmbeddingManager:
    """
    A class for managing document embeddings.

    This class handles file imports (CSV, TXT, PDF, JSON, Markdown), generates embeddings,
    and provides different output formats for the embeddings.
    """

    def __init__(
        self,
        embedding_config: EmbeddingConfig,
        file_path: str,
        output_type: EmbeddingOutputType = EmbeddingOutputType.CHUNKS,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the EmbeddingManager.

        Args:
            embedding_config: Configuration for the embedding model
            file_path: Path to the input file
            output_type: Type of output to generate (chunks, full document, or both)
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.file_type = get_file_type(file_path)
        self.file_path = file_path
        self.output_type = output_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Initialize embedding implementation based on config
        if embedding_config.type == EmbeddingType.HUGGINGFACE:
            self.embedding = HuggingFaceEmbedding(embedding_config)
        elif embedding_config.type == EmbeddingType.OPENAI:
            self.embedding = OpenAIEmbedding(embedding_config)
        elif embedding_config.type == EmbeddingType.GOOGLE:
            self.embedding = GoogleEmbedding(embedding_config)
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_config.type}")

    def load_file(self) -> List[str]:
        """Load and process the input file."""
        try:
            if self.file_type == FileType.CSV:
                loader = CSVLoader(self.file_path)
                documents = loader.load()
            elif self.file_type == FileType.TXT:
                loader = TextLoader(self.file_path)
                documents = loader.load()
            elif self.file_type == FileType.PDF:
                loader = PyPDFLoader(self.file_path)
                documents = loader.load()
            elif self.file_type == FileType.JSON:
                loader = JSONLoader(self.file_path)
                documents = loader.load()
            elif self.file_type == FileType.MARKDOWN:
                loader = UnstructuredMarkdownLoader(self.file_path)
                documents = loader.load()
            else:
                raise ValueError(f"Unsupported file type: {self.file_type}")

            # Get raw text content
            raw_text = " ".join([doc.page_content for doc in documents])

            # Split into chunks
            chunks = self.text_splitter.split_text(raw_text)

            return chunks, raw_text

        except Exception as e:
            logger.error(f"Error loading file: {e}")
            raise

    def generate_embeddings(self) -> Dict[str, Any]:
        """Generate embeddings based on the specified output type."""
        try:
            chunks, raw_text = self.load_file()

            # Generate embeddings
            chunk_embeddings = self.embedding.embed_texts(chunks)
            full_embedding = self.embedding.embed_text(raw_text)

            # Format output based on output_type
            if self.output_type == EmbeddingOutputType.CHUNKS:
                return {
                    "chunks": [
                        {"id": f"chunk_{i}", "text": chunk, "embedding": embedding}
                        for i, (chunk, embedding) in enumerate(
                            zip(chunks, chunk_embeddings)
                        )
                    ]
                }
            elif self.output_type == EmbeddingOutputType.FULL:
                return {"full": {"text": raw_text, "embedding": full_embedding}}
            else:  # BOTH
                return {
                    "chunks": [
                        {"id": f"chunk_{i}", "text": chunk, "embedding": embedding}
                        for i, (chunk, embedding) in enumerate(
                            zip(chunks, chunk_embeddings)
                        )
                    ],
                    "full": {"text": raw_text, "embedding": full_embedding},
                }

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def process_file(
        self, storage_handler: VectorStorageHandler, document_id: str
    ) -> Dict[str, Any]:
        """
        Process the file, generate embeddings, and store them using the provided storage handler.

        Args:
            storage_handler: The vector storage handler to use for storing embeddings
            document_id: ID to use for the document in storage

        Returns:
            Dictionary containing the generated embeddings
        """
        try:
            # Generate embeddings
            embeddings = self.generate_embeddings()

            # Store embeddings using the provided storage handler
            storage_handler.upsert_embeddings(embeddings, document_id)

            return embeddings

        except Exception as e:
            logger.error(f"Error processing file: {e}")
            raise


class EmbeddingManager:
    """
    A class for managing document embeddings.

    This class handles file imports (CSV, TXT, PDF, JSON, Markdown), generates embeddings,
    and provides different output formats for the embeddings.
    """

    def __init__(
        self,
        embedding_model_name: str,
        file_type: FileType,
        file_path: str,
        output_type: EmbeddingOutputType = EmbeddingOutputType.CHUNKS,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the EmbeddingManager.

        Args:
            embedding_model_name: Name of the embedding model to use
            file_type: Type of file to process
            file_path: Path to the input file
            output_type: Type of output to generate (chunks, full document, or both)
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.file_type = file_type
        self.file_path = file_path
        self.output_type = output_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Initialize embedding model
        self.embedding_model = PineconeManager(
            index_name="dummy",  # Not used for embedding generation
            embedding_model_name=embedding_model_name,
        ).embedding_model

    def load_file(self) -> List[str]:
        """Load and process the input file."""
        try:
            if self.file_type == FileType.CSV:
                loader = CSVLoader(self.file_path)
                documents = loader.load()
            elif self.file_type == FileType.TXT:
                loader = TextLoader(self.file_path)
                documents = loader.load()
            elif self.file_type == FileType.PDF:
                loader = PyPDFLoader(self.file_path)
                documents = loader.load()
            elif self.file_type == FileType.JSON:
                loader = JSONLoader(self.file_path)
                documents = loader.load()
            elif self.file_type == FileType.MARKDOWN:
                loader = UnstructuredMarkdownLoader(self.file_path)
                documents = loader.load()
            else:
                raise ValueError(f"Unsupported file type: {self.file_type}")

            # Get raw text content
            raw_text = " ".join([doc.page_content for doc in documents])

            # Split into chunks
            chunks = self.text_splitter.split_text(raw_text)

            return chunks, raw_text

        except Exception as e:
            logger.error(f"Error loading file: {e}")
            raise

    def generate_embeddings(self) -> Dict[str, Any]:
        """Generate embeddings based on the specified output type."""
        try:
            chunks, raw_text = self.load_file()

            # Generate embeddings
            chunk_embeddings = self.embedding_model.embed_documents(chunks)
            full_embedding = self.embedding_model.embed_query(raw_text)

            # Format output based on output_type
            if self.output_type == EmbeddingOutputType.CHUNKS:
                return {
                    "chunks": [
                        {"id": f"chunk_{i}", "text": chunk, "embedding": embedding}
                        for i, (chunk, embedding) in enumerate(
                            zip(chunks, chunk_embeddings)
                        )
                    ]
                }
            elif self.output_type == EmbeddingOutputType.FULL:
                return {"full": {"text": raw_text, "embedding": full_embedding}}
            else:  # BOTH
                return {
                    "chunks": [
                        {"id": f"chunk_{i}", "text": chunk, "embedding": embedding}
                        for i, (chunk, embedding) in enumerate(
                            zip(chunks, chunk_embeddings)
                        )
                    ],
                    "full": {"text": raw_text, "embedding": full_embedding},
                }

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def process_file(
        self, storage_handler: VectorStorageHandler, document_id: str
    ) -> Dict[str, Any]:
        """
        Process the file, generate embeddings, and store them using the provided storage handler.

        Args:
            storage_handler: The vector storage handler to use for storing embeddings
            document_id: ID to use for the document in storage

        Returns:
            Dictionary containing the generated embeddings
        """
        try:
            # Generate embeddings
            embeddings = self.generate_embeddings()

            # Store embeddings using the provided storage handler
            storage_handler.upsert_embeddings(embeddings, document_id)

            return embeddings

        except Exception as e:
            logger.error(f"Error processing file: {e}")
            raise

    def load_file(self) -> List[str]:
        """Load and process the input file."""
        try:
            if self.file_type == FileType.CSV:
                loader = CSVLoader(self.file_path)
                documents = loader.load()
            elif self.file_type == FileType.TXT:
                loader = TextLoader(self.file_path)
                documents = loader.load()
            elif self.file_type == FileType.PDF:
                loader = PyPDFLoader(self.file_path)
                documents = loader.load()
            elif self.file_type == FileType.JSON:
                loader = JSONLoader(self.file_path)
                documents = loader.load()
            elif self.file_type == FileType.MARKDOWN:
                loader = UnstructuredMarkdownLoader(self.file_path)
                documents = loader.load()
            else:
                raise ValueError(f"Unsupported file type: {self.file_type}")

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            texts = text_splitter.split_documents(documents)

            return [text.page_content for text in texts]

        except Exception as e:
            logger.error(f"Error loading file: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Generate embeddings for the input texts."""
        try:
            embeddings = self.pinecone_manager.embedding_model.embed_documents(texts)
            return [
                {"id": f"doc_{i}", "values": embedding, "metadata": {"text": text}}
                for i, (text, embedding) in enumerate(zip(texts, embeddings))
            ]
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def process_file(self) -> List[Dict[str, Any]]:
        """
        Process the file, generate embeddings, and optionally upsert to Pinecone.

        Returns:
            List of embedding records with metadata
        """
        try:
            # Load and process file
            texts = self.load_file()

            # Generate embeddings
            embeddings = self.generate_embeddings(texts)

            # Optionally upsert to Pinecone
            if self.upsert_enabled:
                self.pinecone_manager.upsert(embeddings)

            return embeddings

        except Exception as e:
            logger.error(f"Error processing file: {e}")
            raise
