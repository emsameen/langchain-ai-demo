"""
Configuration settings for the application.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

# Base directories
ROOT_DIR = Path(__file__).parents[3].resolve()
APP_DIR = ROOT_DIR / "src" / "co_genticai"

# API Keys and model configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEFAULT_MODEL = "gemini-2.0-flash"  # Using the newer, faster flash model
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"

# RAG Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", str(ROOT_DIR / "logs" / "app.log"))

# Set up logging
logger.remove()  # Remove default handler
logger.add(LOG_FILE, level=LOG_LEVEL, rotation="10 MB")  # File handler
logger.add(lambda msg: print(msg, end=""), level=LOG_LEVEL)  # Console handler

# Make sure logs directory exists
log_dir = Path(LOG_FILE).parent
log_dir.mkdir(exist_ok=True, parents=True)
