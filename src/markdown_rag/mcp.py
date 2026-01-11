import logging
from collections.abc import Callable

from mcp.server.fastmcp import FastMCP

from markdown_rag.models import MultiResponse, ToolResponse
from markdown_rag.rag import MarkdownRAG

logger = logging.getLogger("MarkdownRAG")


def _run_tool(
    tool: Callable[..., MultiResponse | bool | None],
    tool_name: str,
    kwargs: dict[str, str | int] | None = None,
) -> ToolResponse:
    try:
        result = tool(**kwargs) if kwargs else tool()
        return ToolResponse(data=result)
    except Exception as e:
        logger.exception(
            f"Failed to run {tool_name}: {e.__class__.__name__}({e})"
        )
        return ToolResponse(
            success=False,
            error=f"Failed to run {tool_name}: {e.__class__.__name__}({e})",
        )


def run_mcp(rag: MarkdownRAG, disabled_tools: list[str]) -> None:
    """Run the MCP server."""
    mcp = FastMCP()

    def query(query: str, num_results: int = 4) -> ToolResponse:
        """Retrieve information to help answer a query.

        Args:
            query: The query to ask.
            num_results: The number of results to retrieve.
        """
        if num_results <= 0:
            return ToolResponse(
                success=False,
                error="num_results must be a positive integer.",
            )
        return _run_tool(
            rag.query, "query", {"query": query, "num_results": num_results}
        )

    def list_documents() -> ToolResponse:
        """List all documents in the vector store."""
        return _run_tool(rag.list_documents, "list_documents")

    def delete_document(filename: str) -> ToolResponse:
        """Delete a document from the vector store.

        Args:
            filename: The filename of the document to delete.
        """
        return _run_tool(
            rag.delete_document, "delete_document", {"filename": filename}
        )

    def update_document(filename: str) -> ToolResponse:
        """Update/refresh a specific document in the vector store.

        Args:
            filename: The filename of the document to update.
        """
        return _run_tool(
            rag.refresh_document, "update_document", {"filename": filename}
        )

    def refresh_index() -> ToolResponse:
        """Refresh the entire index (ingest new files)."""
        return _run_tool(rag.ingest, "refresh_index")

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
