"""Core RAG system logic."""

import logging
from collections.abc import Generator
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from sqlalchemy import select

from .models import RagResponse

logger = logging.getLogger(__name__)


class MarkdownRAG:
    """A RAG (Retrieval Augmented Generation) system for markdown files."""

    def __init__(
        self,
        directory: Path,
        *,
        vector_store: PGVector,
        embeddings_model: Embeddings,
    ):
        """Initialize the RAG system with a directory of markdown files."""
        self.vector_store = vector_store
        self.embeddings_model = embeddings_model
        self.directory = directory
        logger.debug("Initializing markdown splitter")
        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
                ("#####", "Header 5"),
            ],
            strip_headers=False,
        )
        logger.debug("Initializing recursive text splitter")
        self._recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""],
        )

    def _iterate_paths(
        self, directory: Path
    ) -> Generator[tuple[str, Path], None, None]:
        for pth in directory.iterdir():
            if pth.is_file():
                yield pth.read_text("utf-8"), pth
                continue
            yield from self._iterate_paths(pth)

    def _split_text(self, file: str) -> list[Document]:
        md_docs = self._md_splitter.split_text(file)
        logger.debug(f"Split file into {len(md_docs)} segments")
        docs = self._recursive_splitter.split_documents(md_docs)
        logger.debug(f"Split file into {len(docs)} documents")
        return docs

    def _document_exists(self, metadata: dict[str, str]) -> bool:
        """Check if documents with given metadata exist in vector store."""
        with self.vector_store._make_sync_session() as session:
            collection = self.vector_store.get_collection(session)
            filter_by = [
                self.vector_store.EmbeddingStore.collection_id
                == collection.uuid
            ]

            stmt = (
                select(self.vector_store.EmbeddingStore.id)
                .where(
                    self.vector_store.EmbeddingStore.cmetadata.contains(
                        metadata
                    )
                )
                .filter(*filter_by)
                .limit(1)
            )

            result = session.execute(stmt).first()
            return result is not None

    def ingest(self) -> None:
        """Add documents to the vector store."""
        logger.info(f"Ingesting files from {self.directory}")
        for file, pth in self._iterate_paths(self.directory):
            filename = str(pth.relative_to(self.directory))

            if self._document_exists({"filename": filename}):
                logger.info(f"Skipping {filename} (already in vector store)")
                continue

            logger.info(f"Ingesting {filename}")
            self.vector_store.add_documents(
                self._split_text(file),
                metadata={"filename": filename},
            )

    def query(self, query: str) -> list[RagResponse]:
        """Retrieve information to help answer a query."""
        docs = self.vector_store.similarity_search(query)
        return [
            RagResponse(
                source=doc.metadata["filename"], content=doc.page_content
            )
            for doc in docs
        ]
