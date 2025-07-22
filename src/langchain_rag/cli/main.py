#!/usr/bin/env python
"""Command-line interface for langchain_rag."""

import argparse
import sys
from typing import List, Optional

from langchain_rag import __version__


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        args: Command line arguments. Defaults to sys.argv[1:].

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="LangChain-based Retrieval Augmented Generation (RAG) toolkit"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # RAG command
    rag_parser = subparsers.add_parser("rag", help="Run RAG operations")
    rag_parser.add_argument(
        "--query", type=str, help="Query to process through the RAG pipeline"
    )
    rag_parser.add_argument(
        "--index", type=str, default="default", help="Name of vector index to use"
    )

    # Embedding command
    embed_parser = subparsers.add_parser("embed", help="Manage embeddings")
    embed_parser.add_argument(
        "--data", type=str, help="Path to data file or directory to embed"
    )
    embed_parser.add_argument(
        "--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", 
        help="Embedding model to use"
    )

    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """Run the CLI application.

    Args:
        args: Command line arguments. Defaults to sys.argv[1:].

    Returns:
        Exit code.
    """
    parsed_args = parse_args(args)
    
    if parsed_args.command == "rag":
        # TODO: Implement RAG command
        print(f"RAG query: {parsed_args.query} on index: {parsed_args.index}")
        return 0
    elif parsed_args.command == "embed":
        # TODO: Implement embedding command
        print(f"Embedding data: {parsed_args.data} with model: {parsed_args.model}")
        return 0
    else:
        print("Please specify a command. Use --help for more information.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
