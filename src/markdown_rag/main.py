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
    settings = (
        env_class(_env_file=args.env_file) if args.env_file else env_class()
    )

    logging.basicConfig(level=args.level.value)
    logger.debug(f"Log level set to {logger.getEffectiveLevel()}")
    logger.debug(f"Starting RAG system in {args.command} mode")

    try:
        rag = start_store(args.directory, settings, args.engine)
    except Exception as e:
        logger.exception(
            f"Failed to start store: ({e.__class__.__name__}) {e}",
            exc_info=False,
        )
        sys.exit(1)

    try:
        match args.command:
            case Command.MCP:
                logger.debug("Starting MCP server")
                run_mcp(rag, settings.DISABLED_TOOLS)
            case Command.INGEST:
                logger.debug("Ingesting files")
                rag.ingest()
            case Command.LIST:
                logger.debug("Listing documents")
                documents = rag.list_documents()
                logger.info("Documents:")
                for doc in documents:
                    logger.info(doc)
            case Command.DELETE:
                if not args.filename:
                    raise ValueError("Filename is required for delete command")
                logger.debug("Deleting document")
                rag.delete_document(args.filename)
            case Command.UPDATE:
                if not args.filename:
                    raise ValueError("Filename is required for update command")
                logger.debug("Updating document")
                rag.refresh_document(args.filename)
            case _:
                logger.error(f"Unknown command {args.command}")
                sys.exit(1)
    except Exception as e:
        logger.exception(f"Failed to run command: {e}", exc_info=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
