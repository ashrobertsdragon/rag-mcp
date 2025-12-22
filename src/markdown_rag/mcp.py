import logging

from mcp.server.fastmcp import FastMCP

from markdown_rag.models import MultiResponse, ToolResponse
from markdown_rag.rag import MarkdownRAG

logger = logging.getLogger("MarkdownRAG")


def run_mcp(rag: MarkdownRAG, disabled_tools: list[str]) -> None:
    """Run the MCP server."""
    mcp = FastMCP()

    def query(query: str, num_results: int = 4) -> ToolResponse:
        if num_results <= 0:
            return ToolResponse(
                success=False,
                error="num_results must be a positive integer.",
            )
        try:
            results: MultiResponse = rag.query(query, num_results=num_results)
            return ToolResponse(data=results)
        except Exception as e:
            logger.exception(f"Failed to query: {e.__class__.__name__}({e})")
            return ToolResponse(
                success=False,
                error=f"{e.__class__.__name__}({e.__class__.__name__}({e}))",
            )

    def list_documents() -> ToolResponse:
        """List all documents in the vector store."""
        try:
            docs: MultiResponse = rag.list_documents()
            return ToolResponse(data=docs)
        except Exception as e:
            logger.exception(
                f"Failed to list documents: {e.__class__.__name__}({e})"
            )
            return ToolResponse(
                success=False,
                error=f"{e.__class__.__name__}({e.__class__.__name__}({e}))",
            )

    def delete_document(filename: str) -> ToolResponse:
        """Delete a document from the vector store."""
        try:
            rag.delete_document(filename)
            return ToolResponse(data=True)
        except Exception as e:
            logger.exception(
                f"Failed to delete document: {e.__class__.__name__}({e})"
            )
            return ToolResponse(
                success=False,
                error=f"{e.__class__.__name__}({e.__class__.__name__}({e}))",
            )

    def update_document(filename: str) -> ToolResponse:
        """Update/refresh a specific document in the vector store."""
        try:
            rag.refresh_document(filename)
            return ToolResponse(data=True)
        except Exception as e:
            logger.exception(
                f"Failed to update document: {e.__class__.__name__}({e})"
            )
            return ToolResponse(
                success=False,
                error=f"{e.__class__.__name__}({e.__class__.__name__}({e}))",
            )

    def refresh_index() -> ToolResponse:
        """Refresh the entire index (ingest new files)."""
        try:
            rag.ingest()
            return ToolResponse(data=True)
        except Exception as e:
            logger.exception(
                f"Failed to refresh index: {e.__class__.__name__}({e})"
            )
            return ToolResponse(
                success=False,
                error=f"{e.__class__.__name__}({e.__class__.__name__}({e}))",
            )

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
