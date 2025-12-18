import logging

from mcp.server.fastmcp import FastMCP

from markdown_rag.models import ErrorResponse, RagResponse
from markdown_rag.rag import MarkdownRAG

logger = logging.getLogger("MarkdownRAG")


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
