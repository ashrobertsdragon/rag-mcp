"""A RAG (Retrieval Augmented Generation) system for markdown files."""

import logging
import sys
from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from mcp.server.fastmcp import FastMCP
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .config import Env, get_cli_args
from .embeddings import RateLimitedEmbeddings
from .models import Command, ErrorResponse, RagResponse
from .rag import MarkdownRAG
from .rate_limiter import RateLimiter, TokenCounter

logger = logging.getLogger("MarkdownRAG")


def start_store(directory: Path, settings: Env) -> MarkdownRAG:
    """Initialize the RAG system with a directory of markdown files."""
    logger.debug("Initializing embeddings model connection")
    base_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="RETRIEVAL_DOCUMENT",
        google_api_key=settings.GOOGLE_API_KEY,
    )
    logger.debug("Initializing the token counter")
    tokenizer = TokenCounter(
        client=base_embeddings.client, model=base_embeddings.model
    )
    logger.debug("Initializing rate limiter")
    rate_limiter = RateLimiter(
        tokenizer=tokenizer,
        max_requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
        max_requests_per_day=settings.RATE_LIMIT_REQUESTS_PER_DAY,
    )

    logger.debug("Wrapping embeddings with rate limiter")
    embeddings = RateLimitedEmbeddings(base_embeddings, rate_limiter)

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
        vector_store=store,
        embeddings_model=embeddings,
        session_factory=session_factory,
    )


def run_mcp(rag: MarkdownRAG, disabled_tools: list[str]) -> None:
    """Run the MCP server."""
    mcp = FastMCP()

    def query(
        query: str, num_results: int = 4
    ) -> list[RagResponse] | ErrorResponse:
        if num_results <= 0:
            return ErrorResponse(
                error=ValueError("num_results must be a positive integer.")
            )
        try:
            return rag.query(query, num_results=num_results)
        except Exception as e:
            logger.exception(f"Failed to query: {e}")
            return ErrorResponse(error=e)

    def list_documents() -> list[str] | ErrorResponse:
        """List all documents in the vector store."""
        try:
            return rag.list_documents()
        except Exception as e:
            logger.exception(f"Failed to list documents: {e}")
            return ErrorResponse(error=e)

    def delete_document(filename: str) -> bool | ErrorResponse:
        """Delete a document from the vector store."""
        try:
            return rag.delete_document(filename)
        except Exception as e:
            logger.exception(f"Failed to delete document: {e}")
            return ErrorResponse(error=e)

    def update_document(filename: str) -> bool | ErrorResponse:
        """Update/refresh a specific document in the vector store."""
        try:
            rag.refresh_document(filename)
            return True
        except Exception as e:
            logger.exception(f"Failed to update document: {e}")
            return ErrorResponse(error=e)

    def refresh_index() -> bool | ErrorResponse:
        """Refresh the entire index (ingest new files)."""
        try:
            rag.ingest()
            return True
        except Exception as e:
            logger.exception(f"Failed to refresh index: {e}")
            return ErrorResponse(error=e)

    for tool in [
        query,
        list_documents,
        delete_document,
        update_document,
        refresh_index,
    ]:
        if tool.__name__ not in disabled_tools:
            mcp.add_tool(tool)  # type: ignore
    mcp.run(transport="stdio")


def main() -> None:
    """Entry point for the RAG system."""
    settings = Env()
    args = get_cli_args()
    logging.basicConfig(level=args.level.value)
    logger.debug(f"Log level set to {logger.getEffectiveLevel()}")
    try:
        rag = start_store(args.directory, settings)
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
