"""A RAG (Retrieval Augmented Generation) system for markdown files."""

import logging
import sys
import time
from collections import deque
from collections.abc import Callable, Generator
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from functools import lru_cache, partial
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
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PostgresDsn,
    SecretStr,
    field_serializer,
)
from pydantic_settings import (
    BaseSettings,
    CliApp,
    CliPositionalArg,
    SettingsConfigDict,
)

logger = logging.getLogger(__name__)


class Command(StrEnum):
    """Commands for the RAG system."""

    INGEST = "ingest"
    MCP = "mcp"


class LogLevel(IntEnum):
    """Log levels for the RAG system."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40


class RagResponse(BaseModel):
    """MCP response for the RAG system."""

    source: str
    content: str


class ErrorResponse(BaseModel):
    """MCP error response for the RAG system."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    error: Exception


class Env(BaseSettings):
    """Environment variable loading."""

    model_config = SettingsConfigDict(env_file=".env")

    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: SecretStr = Field(default=...)
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: str = Field(default="5432")
    POSTGRES_DB: str = Field(default="embeddings")
    GOOGLE_API_KEY: SecretStr = Field(default=...)

    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(
        default=100, description="Maximum requests per minute"
    )
    RATE_LIMIT_REQUESTS_PER_DAY: int = Field(
        default=1000, description="Maximum requests per day"
    )

    @field_serializer("GOOGLE_API_KEY", "POSTGRES_PASSWORD", when_used="always")
    def dump_secret(self, v: SecretStr) -> str:
        """Get secret value."""
        return v.get_secret_value()

    @property
    def postgres_connection(self) -> str:
        """Postgres connection string."""
        schema = "postgresql+psycopg"
        pw = self.POSTGRES_PASSWORD.get_secret_value()
        auth = f"{self.POSTGRES_USER}:{pw}"
        url = f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}"

        return PostgresDsn(
            f"{schema}://{auth}@{url}/{self.POSTGRES_DB}"
        ).encoded_string()


class CLIArgs(BaseSettings):
    """Command line parser."""

    model_config = SettingsConfigDict(
        cli_exit_on_error=True, cli_parse_args=True
    )
    directory: CliPositionalArg[Path] = Field(default=...)
    command: Command = Field(
        default=Command.MCP, validation_alias=AliasChoices("c", "command")
    )
    level: LogLevel = Field(
        default=LogLevel.WARNING, validation_alias=AliasChoices("l", "level")
    )


@dataclass
class Request:
    """Single embedding request record metadata."""

    timestamp: float
    token_count: int


class RateLimiter:
    """Rate limiter using sliding window algorithm for API request tracking.

    Tracks requests and tokens within specified time windows to enforce limits.
    """

    minute_window: float = 60.0
    day_window: float = 86400.0

    def __init__(
        self,
        tokenizer: Callable[[str], int],
        max_requests_per_minute: int = 100,
        max_tokens_per_minute: int = 30000,
        max_requests_per_day: int = 1000,
    ):
        """Constructor for RateLimiter.

        Args:
            tokenizer: Tokenizer function
            max_requests_per_minute: Maximum requests per minute
            max_tokens_per_minute: Maximum tokens per minute
            max_requests_per_day: Maximum requests per day
        """
        self._requests: deque[Request] = deque()
        self._tokenizer = tokenizer
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_requests_per_day = max_requests_per_day

    def _cleanup_old_records(self, current_time: float) -> None:
        """Remove records outside the daily window."""
        cutoff = current_time - self.day_window
        while self._requests and self._requests[0].timestamp < cutoff:
            self._requests.popleft()

    @lru_cache(maxsize=20)
    def _get_token_count(self, prompt: str) -> int:
        """Request token count from tokenizer.

        Cached for recursive calls.

        Args:
            prompt: Prompt to count tokens for

        Returns:
            int: Token count
        """
        new_tokens = self._tokenizer(prompt)
        return new_tokens

    def _count_recent_tokens(self, tokens: int, current_time: float) -> int:
        """Count tokens within a time window.

        Args:
            tokens: Tokens to add

        Returns:
            int: token_count within the window
        """
        for request in reversed(self._requests):
            if request.timestamp < current_time - self.minute_window:
                break
            tokens += request.token_count
        return tokens

    def _count_recent_requests(self, current_time: float, window: float) -> int:
        """Count requests and within a time window.

        Returns:
            int: request_count within the window
        """
        cutoff = current_time - window
        requests = 0
        for request in reversed(self._requests):
            if request.timestamp < cutoff:
                break
            requests += 1
        return requests

    def _calculate_wait_time_for_limit(
        self, current_time: float, window: float, count: int, max_count: int
    ) -> float:
        """Calculate wait time needed for a specific rate limit."""
        if count < max_count:
            return 0.0

        cutoff = current_time - window
        for request in self._requests:
            timestamp = request.timestamp
            if request.timestamp > cutoff:
                return timestamp - cutoff
        return 0.0

    def _get_usage_and_wait_time(
        self, current_time: float, tokens: int
    ) -> tuple[float, int, int, int]:
        """Get current usage stats and calculate the required wait time."""
        token_requests = self._count_recent_tokens(tokens, current_time)
        minute_requests = self._count_recent_requests(
            current_time, self.minute_window
        )
        day_requests = self._count_recent_requests(
            current_time, self.day_window
        )

        minute_wait = self._calculate_wait_time_for_limit(
            current_time,
            self.minute_window,
            minute_requests,
            self.max_requests_per_minute,
        )
        day_wait = self._calculate_wait_time_for_limit(
            current_time,
            self.day_window,
            day_requests,
            self.max_requests_per_day,
        )
        token_wait = self._calculate_wait_time_for_limit(
            current_time,
            self.minute_window,
            token_requests,
            self.max_tokens_per_minute,
        )

        wait_time = max(minute_wait, day_wait, token_wait)
        return wait_time, token_requests, minute_requests, day_requests

    def _approve_request(
        self,
        current_time: float,
        tokens: int,
        minute_requests: int,
        day_requests: int,
        token_requests: int,
    ) -> None:
        """Log and record an approved request."""
        self._requests.append(
            Request(timestamp=current_time, token_count=tokens)
        )
        logger.debug(
            f"Request allowed: {minute_requests + 1} req/min, "
            f"{day_requests + 1} req/day, {token_requests} tokens/min"
        )

    def _wait_and_retry(
        self,
        wait_time: float,
        minute_requests: int,
        day_requests: int,
        token_requests: int,
        prompt: str,
    ) -> None:
        """Log the wait time and retry the request recursively."""
        logger.info(
            f"Rate limit approaching. Waiting {wait_time:.2f}s "
            f"(requests: {minute_requests}/{self.max_requests_per_minute}/min, "
            f"{day_requests}/{self.max_requests_per_day}/day), "
            f"Tokens: {token_requests} tokens/min"
        )
        time.sleep(wait_time)
        self.wait_if_needed(prompt)

    def wait_if_needed(self, prompt: str) -> None:
        """Block until the request can proceed within rate limits.

        Uses a sliding window algorithm to enforce limits per minute, day, and
        token. Uses recursive calls until the request can proceed.

        Args:
            prompt: Prompt to count tokens for
        """
        current_time = time.time()
        self._cleanup_old_records(current_time)

        tokens = self._get_token_count(prompt)

        (
            wait_time,
            token_requests,
            minute_requests,
            day_requests,
        ) = self._get_usage_and_wait_time(current_time, tokens)

        if wait_time <= 0:
            self._approve_request(
                current_time,
                tokens,
                minute_requests,
                day_requests,
                token_requests,
            )
        else:
            self._wait_and_retry(
                wait_time,
                minute_requests,
                day_requests,
                token_requests,
                prompt,
            )


class RateLimitedEmbeddings(Embeddings):
    """Wrapper for embeddings model that enforces rate limits.

    Tracks token usage and applies rate limiting before API calls.
    """

    def __init__(
        self,
        embeddings: Embeddings,
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
        self.rate_limiter.wait_if_needed("\n".join(texts))
        return self.embeddings.embed_documents(texts)

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

    def ingest(self) -> None:
        """Add documents to the vector store."""
        logger.info(f"Ingesting files from {self.directory}")
        for file, pth in self._iterate_paths(self.directory):
            filename = str(pth.relative_to(self.directory))

            existing_docs = self.vector_store.similarity_search(
                "check", k=1, filter={"filename": {"$eq": filename}}
            )

            if existing_docs:
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


def start_store(directory: Path, settings: Env) -> MarkdownRAG:
    """Initialize the RAG system with a directory of markdown files."""
    logger.debug("Initializing embeddings model connection")
    base_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="RETRIEVAL_DOCUMENT",
        google_api_key=settings.GOOGLE_API_KEY,
    )
    tokenizer = partial(
        base_embeddings.client.models.count_tokens, model=base_embeddings.model
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
    logger.debug("Initializing RAG system")
    return MarkdownRAG(
        directory, vector_store=store, embeddings_model=embeddings
    )


def run_mcp(rag: MarkdownRAG) -> None:
    """Run the MCP server."""
    mcp = FastMCP()

    @mcp.tool()
    def query(query: str) -> list[RagResponse] | ErrorResponse:
        try:
            return rag.query(query)
        except Exception as e:
            logger.exception(f"Failed to query: {e}")
            return ErrorResponse(error=e)

    mcp.run(transport="stdio")


def main() -> None:
    """Entry point for the RAG system."""
    settings = Env()
    args = CliApp.run(CLIArgs)
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
        run_mcp(rag)
    else:
        logger.error(f"Received command {args.command}, expected INGEST or MCP")


if __name__ == "__main__":
    main()
