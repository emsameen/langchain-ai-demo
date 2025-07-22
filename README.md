# LangChain RAG Toolkit

A comprehensive toolkit for building Retrieval Augmented Generation (RAG) applications using LangChain.

## Features

- Document loading and processing from various sources (PDF, Markdown, TXT)
- Text chunking and embedding
- Vector storage integration with Pinecone
- Customizable retrieval pipelines
- Command-line interface for common RAG operations

## Installation

### Prerequisites

- Python 3.9+
- Poetry (for dependency management)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/emsameen/langchain-ai-demo.git
   cd langchain-ai-demo
   ```

2. Install dependencies with Poetry:
   ```bash
   poetry install
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Usage

### Command Line Interface

The package provides a command-line interface for common operations:

```bash
# Display help
langchain-rag --help

# Run a RAG query
langchain-rag rag --query "What is retrieval augmented generation?" --index "my-index"

# Create embeddings from files
langchain-rag embed --data "./data/documents/" --model "sentence-transformers/all-MiniLM-L6-v2"
```

### Python API

```python
from langchain_rag.core.processor import RAGProcessor
from langchain_rag.data.loaders import load_documents
from langchain_openai import OpenAI

# Load and process documents
documents = load_documents("./data/documents/")

# Initialize the RAG processor
llm = OpenAI()
processor = RAGProcessor(llm=llm)

# Process a query
result = processor.process("What is retrieval augmented generation?")
print(result["answer"])
```

## Project Structure

```
langchain-rag/
├── src/langchain_rag/       # Main package
│   ├── cli/                  # Command-line interfaces
│   ├── config/               # Configuration
│   ├── core/                 # Core functionality
│   ├── data/                 # Data handling
│   ├── models/               # Model definitions
│   ├── rag/                  # RAG components
│   ├── services/             # External services integration
│   └── utils/                # Utility functions
├── tests/                    # Test suite
├── docs/                     # Documentation
└── data/                     # Data files
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/
isort src/ tests/
```

## License

Apache 2.0