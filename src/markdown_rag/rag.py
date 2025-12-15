"""Core RAG system logic."""

import logging
from collections.abc import Callable, Generator
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

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
        session_factory: Callable[[], Session],
    ):
        """Initialize the RAG system with a directory of markdown files."""
        self.vector_store = vector_store
        self.embeddings_model = embeddings_model
        self.directory = directory
        self._session_factory = session_factory
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
        with self._session_factory() as session:
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

    def _add_document(self, filepath: Path, text: str) -> None:
        """Add a single document to the vector store."""
        filename = str(filepath.relative_to(self.directory))

        if self._document_exists({"filename": filename}):
            logger.info(f"Skipping {filename} (already in vector store)")
            return

        logger.info(f"Ingesting {filename}")
        docs = self._split_text(str(text))
        for doc in docs:
            doc.metadata["filename"] = filename

        self.vector_store.add_documents(docs)

    def ingest(self) -> None:
        """Add documents to the vector store."""
        logger.info(f"Ingesting files from {self.directory}")
        for file, pth in self._iterate_paths(self.directory):
            self._add_document(pth, file)

    def query(self, query: str, num_results: int = 4) -> list[RagResponse]:
        """Retrieve information to help answer a query."""
        docs = self.vector_store.similarity_search(query, k=num_results)
        return [
            RagResponse(
                source=doc.metadata["filename"], content=doc.page_content
            )
            for doc in docs
        ]

    def list_documents(self) -> list[str]:
        """List all documents in the vector store."""
        with self._session_factory() as session:
            collection = self.vector_store.get_collection(session)
            stmt = (
                select(
                    self.vector_store.EmbeddingStore.cmetadata[
                        "filename"
                    ].astext
                )
                .filter(
                    self.vector_store.EmbeddingStore.collection_id
                    == collection.uuid
                )
                .distinct()
            )
            result = session.execute(stmt).all()
            return sorted([row[0] for row in result if row[0]])

    def refresh_document(self, filename: str) -> None:
        """Refresh a document in the vector store."""
        file_path = self.directory / filename
        if not file_path.exists():
            raise FileNotFoundError(
                f"File {filename} not found in {self.directory}"
            )

        self.delete_document(filename)
        logger.info(f"Re-ingesting {filename}")
        text = file_path.read_text()
        self._add_document(file_path, text)

    def delete_document(self, filename: str) -> bool:
        """Delete a document from the vector store."""
        with self._session_factory() as session:
            collection = self.vector_store.get_collection(session)
            stmt = (
                delete(self.vector_store.EmbeddingStore)
                .filter(
                    self.vector_store.EmbeddingStore.collection_id
                    == collection.uuid
                )
                .filter(
                    self.vector_store.EmbeddingStore.cmetadata[
                        "filename"
                    ].astext
                    == filename
                )
                .returning(self.vector_store.EmbeddingStore.id)
            )
            result = session.execute(stmt).scalars().all()
            session.commit()
            return len(result) > 0
