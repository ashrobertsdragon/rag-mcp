"""Embeddings wrapper for rate limiting."""

import logging

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from markdown_rag.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class RateLimitedEmbeddings(Embeddings):
    """Wrapper for embeddings model that enforces rate limits.

    Tracks token usage and applies rate limiting before API calls.
    """

    def __init__(
        self,
        embeddings: GoogleGenerativeAIEmbeddings,
        rate_limiter: RateLimiter,
    ):
        """Initialize with an embeddings model and rate limiter.

        Args:
            embeddings: The underlying embeddings model
            rate_limiter: Rate limiter instance for tracking/enforcement
        """
        self.embeddings = embeddings
        self.rate_limiter = rate_limiter

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents with rate limiting.

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors
        """
        results: list[list[float]] = []
        for batch in self.rate_limiter.generate_batches(texts):
            logger.debug(f"Embedding batch of {len(batch)} documents")
            results.extend(self.embeddings.embed_documents(batch))
        return results

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with rate limiting.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        self.rate_limiter.wait_if_needed(text)
        return self.embeddings.embed_query(text)

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying embeddings model."""
        return getattr(self.embeddings, name)
