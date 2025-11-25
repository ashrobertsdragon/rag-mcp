"""A RAG (Retrieval Augmented Generation) system for markdown files."""

import sys
from collections.abc import Generator
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel


class RagResponse(BaseModel):
    source: str
    content: str


class MarkdownRAG:
    """A RAG (Retrieval Augmented Generation) system for markdown files."""

    def __init__(
        self,
        directory: Path,
        *,
        vector_store: VectorStore,
        embeddings_model: Embeddings,
    ):
        """Initialize the RAG system with a directory of markdown files."""
        self.vector_store = vector_store
        self.embeddings_model = embeddings_model
        self.directory = directory

        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
                ("#####", "Header 5"),
            ],
            strip_headers=False,
        )
        self._recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""],
        )

    def _iterate_paths(self, directory: Path) -> Generator[str, None, None]:
        for pth in directory.iterdir():
            if pth.is_file():
                yield pth.read_text("utf-8")
                continue
            yield from self._iterate_paths(pth)

    def _split_text(self, file: str) -> list[Document]:
        md_docs = self._md_splitter.split_text(file)
        return self._recursive_splitter.split_documents(md_docs)

    def ingest(self) -> None:
        for file in self._iterate_paths(self.directory):
            self.vector_store.add_documents(
                self._split_text(file), metadata={"filename": file}
            )

    def query(self, query: str) -> list[RagResponse]:
        """Retrieve information to help answer a query."""
        docs = self.vector_store.similarity_search(query, k=2)
        return [
            RagResponse(
                source=doc.metadata["filename"], content=doc.page_content
            )
            for doc in docs
        ]


def start_store(directory: Path) -> MarkdownRAG:
    """Initialize the RAG system with a directory of markdown files."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", task_type="RETRIEVAL_DOCUMENT"
    )
    store = PGVector(
        embeddings=embeddings,
        collection_name=directory.name,
        connection="postgresql+psycopg://...",
    )
    return MarkdownRAG(
        directory, vector_store=store, embeddings_model=embeddings
    )


def run_mcp(rag: MarkdownRAG) -> None:
    """Run the MCP server."""
    mcp = FastMCP()

    @mcp.tool()
    def query(query: str) -> list[RagResponse]:
        return rag.query(query)

    mcp.run(transport="stdio")


def main():
    """Entry point for the RAG system."""
    directory = Path(sys.argv[1])
    rag = start_store(directory)
    if len(sys.argv) < 2 or sys.argv[2] not in ["mcp", "ingest"]:
        print("Must specify `ingest` or `mcp`.")
        sys.exit(1)

    if sys.argv[2] == "ingest":
        rag.ingest()
    elif sys.argv[2] == "mcp":
        run_mcp(rag)


if __name__ == "__main__":
    main()
