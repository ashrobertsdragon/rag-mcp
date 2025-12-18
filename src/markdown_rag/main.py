"""A RAG (Retrieval Augmented Generation) system for markdown files."""

import logging
import sys
from pathlib import Path

from langchain_postgres import PGVector
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from markdown_rag.config import Env, get_cli_args, get_env
from markdown_rag.database import ensure_database_exists
from markdown_rag.initializers import initialize_embeddings
from markdown_rag.mcp import run_mcp
from markdown_rag.models import Command, EmbeddingEngine
from markdown_rag.rag import MarkdownRAG

logger = logging.getLogger("MarkdownRAG")


def start_store(
    directory: Path, settings: Env, embedding_engine: EmbeddingEngine
) -> MarkdownRAG:
    """Initialize the RAG system with a directory of markdown files."""
    ensure_database_exists(settings)

    logger.debug("Initializing embeddings model connection")
    embeddings = initialize_embeddings(embedding_engine, settings)
    logger.debug("Initializing vector store")
    store = PGVector(
        embeddings=embeddings,
        collection_name=directory.name,
        connection=settings.postgres_connection,
        logger=logger,
    )

    logger.debug("Creating database session factory")
    engine = create_engine(settings.postgres_connection)
    session_factory = sessionmaker(bind=engine)

    logger.debug("Initializing RAG system")
    return MarkdownRAG(
        directory,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.CHUNK_OVERLAP,
        vector_store=store,
        embeddings_model=embeddings,
        session_factory=session_factory,
    )


def main() -> None:
    """Entry point for the RAG system."""
    args = get_cli_args()
    env_class = get_env(args.engine)
    settings = env_class(_env_file=args.env_file)

    logging.basicConfig(level=args.level.value)
    logger.debug(f"Log level set to {logger.getEffectiveLevel()}")

    try:
        rag = start_store(args.directory, settings, args.engine)
    except Exception as e:
        logger.exception(f"Failed to start store: {e}", exc_info=False)
        sys.exit(1)
    if args.command == Command.INGEST:
        logger.debug("Ingesting files")
        try:
            rag.ingest()
        except Exception as e:
            logger.exception(f"Failed to ingest files: {e}", exc_info=False)
            sys.exit(1)
    elif args.command == Command.MCP:
        logger.debug("Starting MCP server")
        run_mcp(rag, disabled_tools=settings.DISABLED_TOOLS)
    else:
        logger.error(
            f"Received command {args.command}, expected INGEST or MCP"
        )


if __name__ == "__main__":
    main()
