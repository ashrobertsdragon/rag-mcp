"""A RAG (Retrieval Augmented Generation) system for markdown files."""

import logging
import sys
import time
from collections import deque
from collections.abc import Callable, Generator
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from functools import lru_cache
from pathlib import Path

from google.ai.generativelanguage_v1beta import GenerativeServiceClient
from google.ai.generativelanguage_v1beta.types import Content, Part
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
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
from sqlalchemy import select

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


class TokenCounter:
    """Functor for counting tokens in a prompt."""

    def __init__(self, client: GenerativeServiceClient, model: str):
        """Initialize the token counter.

        Args:
            client: The generative AI client.
            model: The model name to use for tokenization.
        """
        self.client = client
        self.model = model

    def __call__(self, prompt: str) -> int:
        """Count tokens in a prompt.

        Args:
            prompt: The prompt string.

        Returns:
            The number of tokens in the prompt.
        """
        contents = [Content(parts=[Part(text=prompt)])]
        response = self.client.count_tokens(model=self.model, contents=contents)
        return response.total_tokens


@dataclass
class Request:
    """Single embedding request record metadata."""

    timestamp: float
    token_count: int


@dataclass
class UsageStats:
    """Current API usage statistics."""

    minute_requests: int
    day_requests: int
    minute_tokens: int


class UsageTracker:
    """Efficiently tracks API usage with cached statistics.

    Maintains a sliding window of requests and provides O(1) amortized
    access to current usage stats via caching.
    """

    def __init__(self, minute_window: float, day_window: float):
        """Initialize the usage tracker.

        Args:
            minute_window: Time window in seconds for minute-based limits
            day_window: Time window in seconds for day-based limits
        """
        self._requests: deque[Request] = deque()
        self._minute_window = minute_window
        self._day_window = day_window
        self._cached_stats: UsageStats | None = None
        self._cache_time: float = 0.0
        self._cache_ttl: float = 0.1

    def add_request(self, timestamp: float, tokens: int) -> None:
        """Record a new request."""
        self._requests.append(Request(timestamp, tokens))
        self._invalidate_cache()

    def cleanup_old(self, current_time: float) -> None:
        """Remove requests outside the daily window."""
        cutoff = current_time - self._day_window
        while self._requests and self._requests[0].timestamp < cutoff:
            self._requests.popleft()
        self._invalidate_cache()

    def get_stats(self, current_time: float) -> UsageStats:
        """Get current usage statistics with caching."""
        if (
            self._cached_stats
            and (current_time - self._cache_time) < self._cache_ttl
        ):
            return self._cached_stats

        self._cached_stats = self._calculate_stats(current_time)
        self._cache_time = current_time
        return self._cached_stats

    def find_oldest_in_window(
        self, current_time: float, window: float
    ) -> float | None:
        """Find the oldest request timestamp within a window."""
        cutoff = current_time - window
        for req in self._requests:
            if req.timestamp > cutoff:
                return req.timestamp
        return None

    def _calculate_stats(self, current_time: float) -> UsageStats:
        """Calculate usage stats by scanning the deque once."""
        minute_cutoff = current_time - self._minute_window
        day_cutoff = current_time - self._day_window

        minute_requests = 0
        day_requests = 0
        minute_tokens = 0

        for req in reversed(self._requests):
            if req.timestamp >= minute_cutoff:
                minute_requests += 1
                minute_tokens += req.token_count
            if req.timestamp >= day_cutoff:
                day_requests += 1
            else:
                break

        return UsageStats(minute_requests, day_requests, minute_tokens)

    def _invalidate_cache(self) -> None:
        """Invalidate the cached statistics."""
        self._cached_stats = None


@lru_cache(maxsize=20)
def get_token_count(prompt: str, tokenizer: Callable[[str], int]) -> int:
    """Request token count from tokenizer.

    Cached for recursive calls.

    Args:
        prompt: Prompt to count tokens for

    Returns:
        int: Token count
    """
    return tokenizer(prompt)


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
        self._tracker = UsageTracker(self.minute_window, self.day_window)
        self._tokenizer = tokenizer
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_requests_per_day = max_requests_per_day

    def _calculate_wait_time(
        self, current_time: float, tokens: int
    ) -> tuple[float, UsageStats]:
        """Calculate wait time based on current usage and new tokens.

        Args:
            current_time: Current timestamp
            tokens: Number of tokens in the new request

        Returns:
            Tuple of (wait_time, current_usage_stats)
        """
        stats = self._tracker.get_stats(current_time)
        total_tokens = stats.minute_tokens + tokens

        wait_times = []

        if stats.minute_requests >= self.max_requests_per_minute:
            oldest = self._tracker.find_oldest_in_window(
                current_time, self.minute_window
            )
            if oldest:
                wait_times.append(oldest - (current_time - self.minute_window))

        if stats.day_requests >= self.max_requests_per_day:
            oldest = self._tracker.find_oldest_in_window(
                current_time, self.day_window
            )
            if oldest:
                wait_times.append(oldest - (current_time - self.day_window))

        if total_tokens > self.max_tokens_per_minute:
            oldest = self._tracker.find_oldest_in_window(
                current_time, self.minute_window
            )
            if oldest:
                wait_times.append(oldest - (current_time - self.minute_window))

        return max(wait_times, default=0.0), stats

    def _wait_and_log(
        self, wait_time: float, stats: UsageStats, tokens: int
    ) -> None:
        """Wait for the specified time and log the reason.

        Args:
            wait_time: Time to wait in seconds
            stats: Current usage statistics
            tokens: Number of tokens in the pending request
        """
        logger.info(
            f"Rate limit approaching. Waiting {wait_time:.2f}s "
            f"(requests: {stats.minute_requests}/"
            f"{self.max_requests_per_minute}/min, "
            f"{stats.day_requests}/{self.max_requests_per_day}/day), "
            f"Tokens: {stats.minute_tokens + tokens}/"
            f"{self.max_tokens_per_minute} tokens/min"
        )
        time.sleep(wait_time)

    def _process_single_request(self, prompt: str) -> None:
        """Process a single request, waiting if necessary.

        Args:
            prompt: The text prompt to process
        """
        tokens = get_token_count(prompt, self._tokenizer)
        current_time = time.time()
        self._tracker.cleanup_old(current_time)

        wait_time, stats = self._calculate_wait_time(current_time, tokens)

        if wait_time <= 0:
            self._tracker.add_request(current_time, tokens)
            logger.debug(
                f"Request allowed: {stats.minute_requests + 1} req/min, "
                f"{stats.day_requests + 1} req/day, "
                f"{stats.minute_tokens + tokens} tokens/min"
            )
            return

        self._wait_and_log(wait_time, stats, tokens)
        self._process_single_request(prompt)

    def _calculate_batch_size(
        self, texts: list[str], current_time: float
    ) -> int:
        """Calculate max batch size that fits within current limits.

        Args:
            texts: List of text prompts to batch
            current_time: Current timestamp

        Returns:
            Maximum number of texts that can be processed in current batch
        """
        stats = self._tracker.get_stats(current_time)

        available_minute = self.max_requests_per_minute - stats.minute_requests
        available_day = self.max_requests_per_day - stats.day_requests
        available_tokens = self.max_tokens_per_minute - stats.minute_tokens

        max_requests = min(available_minute, available_day)
        if max_requests < 1:
            return 0

        cumulative_tokens = 0
        for i, text in enumerate(texts):
            if i >= max_requests:
                return i

            tokens = get_token_count(text, self._tokenizer)
            if cumulative_tokens + tokens > available_tokens:
                return max(1, i)
            cumulative_tokens += tokens

        return len(texts)

    def generate_batches(
        self, texts: list[str]
    ) -> Generator[list[str], None, None]:
        """Generate batches of texts that respect rate limits.

        Handles waiting and approval internally. Each yielded batch is
        ready to send to the API.

        Args:
            texts: List of texts to batch

        Yields:
            Batches of texts that can be processed without violating rate limits
        """
        while texts:
            current_time = time.time()
            self._tracker.cleanup_old(current_time)

            batch_size = self._calculate_batch_size(texts, current_time)

            if batch_size < 1:
                self.wait_if_needed(texts)
                continue

            batch = texts[:batch_size]
            self.wait_if_needed(batch)
            yield batch

            texts = texts[batch_size:]

    def wait_if_needed(self, request: list[str] | str) -> None:
        """Block until the request can proceed within rate limits.

        Uses a sliding window algorithm to enforce limits per minute, day, and
        token. Processes each request sequentially to ensure rate limits.

        Args:
            request: Prompt or prompts to count tokens for
        """
        requests = request if isinstance(request, list) else [request]
        for prompt in requests:
            self._process_single_request(prompt)


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
