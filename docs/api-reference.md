# API Reference

Complete API documentation for the Markdown RAG system.

## Table of Contents

- [Core Classes](#core-classes)
  - [MarkdownRAG](#markdownrag)
  - [RateLimitedEmbeddings](#ratelimitedembeddings)
  - [RateLimiter](#ratelimiter)
- [Data Models](#data-models)
- [Configuration](#configuration)
- [Utilities](#utilities)

## Core Classes

### MarkdownRAG

**Location:** `src/markdown_rag/rag.py`

A RAG (Retrieval Augmented Generation) system for markdown files with intelligent chunking and vector storage.

#### Constructor

```python
def __init__(
    self,
    directory: Path,
    *,
    vector_store: PGVector,
    embeddings_model: Embeddings,
) -> None: ...
```

**Parameters:**

- `directory` (Path): Path to directory containing markdown files
- `vector_store` (PGVector): Configured PGVector instance for embedding storage
- `embeddings_model` (Embeddings): Embeddings model (typically RateLimitedEmbeddings)

**Example:**

```python
from pathlib import Path
from langchain_postgres import PGVector
from markdown_rag.rag import MarkdownRAG
from markdown_rag.embeddings import RateLimitedEmbeddings

rag = MarkdownRAG(
    directory=Path("./docs"),
    vector_store=vector_store,
    embeddings_model=embeddings_model,
)
```

#### Methods

##### ingest()

Process and store all markdown files from the configured directory.

```python
def ingest(self) -> None: ...
```

**Behavior:**

- Recursively iterates through all files in the directory
- Skips files already in the vector store (based on filename metadata)
- Splits documents using markdown-aware chunking
- Stores embeddings in PostgreSQL with metadata

**Example:**

```python
rag.ingest()
```

**Logging:**

- INFO: File ingestion progress and skipped files
- DEBUG: Document splitting details

##### query()

Retrieve relevant document chunks for a query using semantic similarity.

```python
def query(self, query: str) -> list[RagResponse]: ...
```

**Parameters:**

- `query` (str): Search query text

**Returns:**

- `list[RagResponse]`: List of relevant document chunks with source metadata

**Example:**

```python
results = rag.query("How do I configure authentication?")
for result in results:
    print(f"Source: {result.source}")
    print(f"Content: {result.content}\n")
```

**Default Behavior:**

- Returns top 4 most similar documents (pgvector default)
- Uses cosine similarity for ranking

#### Internal Methods

##### \_iterate_paths()

```python
def _iterate_paths(
    self, directory: Path
) -> Generator[tuple[str, Path], None, None]: ...
```

Recursively yields file contents and paths from a directory.

##### \_split_text()

```python
def _split_text(self, file: str) -> list[Document]: ...
```

Applies two-stage text splitting:

1. Markdown header splitting (##, ###, ####, #####)
1. Recursive character splitting (2000 chars, 50 overlap)

**Parameters:**

- `file` (str): Markdown file content

**Returns:**

- `list[Document]`: List of split documents with metadata

##### \_document_exists()

```python
def _document_exists(self, metadata: dict[str, str]) -> bool: ...
```

Checks if documents with given metadata already exist in the vector store.

**Parameters:**

- `metadata` (dict): Metadata to check (typically `{"filename": "path/to/file.md"}`)

**Returns:**

- `bool`: True if documents exist, False otherwise

---

### RateLimitedEmbeddings

**Location:** `src/markdown_rag/embeddings.py`

Wrapper for embeddings models that enforces API rate limiting with intelligent batching.

#### Constructor

```python
def __init__(
    self,
    embeddings: GoogleGenerativeAIEmbeddings,
    rate_limiter: RateLimiter,
) -> None: ...
```

**Parameters:**

- `embeddings` (GoogleGenerativeAIEmbeddings): Base embeddings model
- `rate_limiter` (RateLimiter): Rate limiter instance

**Example:**

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from markdown_rag.embeddings import RateLimitedEmbeddings
from markdown_rag.rate_limiter import RateLimiter, TokenCounter

base_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", task_type="RETRIEVAL_DOCUMENT"
)

tokenizer = TokenCounter(
    client=base_embeddings.client, model=base_embeddings.model
)

rate_limiter = RateLimiter(
    tokenizer=tokenizer, max_requests_per_minute=100, max_requests_per_day=1000
)

embeddings = RateLimitedEmbeddings(base_embeddings, rate_limiter)
```

#### Methods

##### embed_documents()

Embed multiple documents with automatic batching and rate limiting.

```python
def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
```

**Parameters:**

- `texts` (list[str]): List of document texts to embed

**Returns:**

- `list[list[float]]`: List of embedding vectors

**Behavior:**

- Automatically batches requests based on rate limits
- Waits when necessary to avoid exceeding limits
- Logs batch processing progress

**Example:**

```python
documents = ["First document", "Second document", "Third document"]
embeddings_list = embeddings.embed_documents(documents)
```

##### embed_query()

Embed a single query with rate limiting.

```python
def embed_query(self, text: str) -> list[float]: ...
```

**Parameters:**

- `text` (str): Query text to embed

**Returns:**

- `list[float]`: Embedding vector

**Example:**

```python
query_embedding = embeddings.embed_query("search query")
```

---

### RateLimiter

**Location:** `src/markdown_rag/rate_limiter.py`

Implements a sliding window rate limiting algorithm with token counting and batch optimization.

#### Constructor

```python
def __init__(
    self,
    tokenizer: Callable[[str], int],
    max_requests_per_minute: int = 100,
    max_tokens_per_minute: int = 30000,
    max_requests_per_day: int = 1000,
) -> None: ...
```

**Parameters:**

- `tokenizer` (Callable\[[str], int\]): Function that counts tokens in a string
- `max_requests_per_minute` (int): Maximum API requests per minute (default: 100)
- `max_tokens_per_minute` (int): Maximum tokens per minute (default: 30000)
- `max_requests_per_day` (int): Maximum API requests per day (default: 1000)

**Note:** Rate limit parameters should be configured via environment variables. See [Configuration](#configuration) for details.

#### Methods

##### wait_if_needed()

Block until a request can proceed within rate limits.

```python
def wait_if_needed(self, request: list[str] | str) -> None: ...
```

**Parameters:**

- `request` (list[str] | str): Prompt or list of prompts to process

**Behavior:**

- Counts tokens for each prompt
- Calculates wait time based on current usage
- Sleeps if necessary
- Recursively retries until request can proceed
- Logs rate limit status

**Example:**

```python
rate_limiter.wait_if_needed("Single prompt")

rate_limiter.wait_if_needed(["Prompt 1", "Prompt 2", "Prompt 3"])
```

##### generate_batches()

Generate optimally-sized batches that respect rate limits.

```python
def generate_batches(
    self, texts: list[str]
) -> Generator[list[str], None, None]: ...
```

**Parameters:**

- `texts` (list[str]): List of texts to batch

**Yields:**

- `list[str]`: Batches ready to send to API

**Behavior:**

- Calculates maximum safe batch size
- Waits between batches if needed
- Optimizes throughput while respecting limits

**Example:**

```python
texts = ["Text 1", "Text 2", ..., "Text 100"]

for batch in rate_limiter.generate_batches(texts):
    results = api_call(batch)
    process_results(results)
```

---

### TokenCounter

**Location:** `src/markdown_rag/rate_limiter.py`

Callable class for counting tokens using Google's tokenization API.

#### Constructor

```python
def __init__(self, client: GenerativeServiceClient, model: str) -> None: ...
```

**Parameters:**

- `client` (GenerativeServiceClient): Google Generative AI client
- `model` (str): Model name for tokenization (e.g., "models/gemini-embedding-001")

#### Methods

##### \_\_call\_\_()

Count tokens in a prompt.

```python
def __call__(self, prompt: str) -> int: ...
```

**Parameters:**

- `prompt` (str): Text to tokenize

**Returns:**

- `int`: Number of tokens

**Example:**

```python
from google.ai.generativelanguage_v1beta import GenerativeServiceClient

client = GenerativeServiceClient()
tokenizer = TokenCounter(client, "models/gemini-embedding-001")

token_count = tokenizer("This is a test prompt")
print(f"Tokens: {token_count}")
```

---

### UsageTracker

**Location:** `src/markdown_rag/rate_limiter.py`

Efficiently tracks API usage with sliding windows and cached statistics.

#### Constructor

```python
def __init__(self, minute_window: float, day_window: float) -> None: ...
```

**Parameters:**

- `minute_window` (float): Time window in seconds for minute-based limits (typically 60.0)
- `day_window` (float): Time window in seconds for day-based limits (typically 86400.0)

#### Methods

##### add_request()

```python
def add_request(self, timestamp: float, tokens: int) -> None: ...
```

Record a new API request.

##### get_stats()

```python
def get_stats(self, current_time: float) -> UsageStats: ...
```

Get current usage statistics with caching.

**Returns:**

- `UsageStats`: Current usage across all time windows

##### cleanup_old()

```python
def cleanup_old(self, current_time: float) -> None: ...
```

Remove requests outside the daily window to maintain memory efficiency.

---

## Data Models

**Location:** `src/markdown_rag/models.py`

### Command

```python
class Command(StrEnum):
    INGEST = "ingest"
    MCP = "mcp"
```

Enum for CLI command modes.

### LogLevel

```python
class LogLevel(IntEnum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
```

Enum for logging levels.

### RagResponse

```python
class RagResponse(BaseModel):
    source: str
    content: str
```

Response model for RAG queries.

**Fields:**

- `source` (str): Relative file path of the source document
- `content` (str): Relevant content chunk from the document

**Example:**

```python
response = RagResponse(
    source="docs/setup/auth.md",
    content="## Authentication Configuration\n\nTo configure...",
)
```

### ErrorResponse

```python
class ErrorResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    error: Exception
```

Error response model for MCP server.

**Fields:**

- `error` (Exception): The exception that occurred

---

## Configuration

**Location:** `src/markdown_rag/config.py`

### Env

```python
class Env(BaseSettings):
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: SecretStr = Field(default=...)
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: str = Field(default="5432")
    POSTGRES_DB: str = Field(default="embeddings")
    GOOGLE_API_KEY: SecretStr = Field(default=...)
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(default=100)
    RATE_LIMIT_REQUESTS_PER_DAY: int = Field(default=1000)
```

Environment variable configuration with Pydantic validation.

#### Properties

##### postgres_connection

```python
@property
def postgres_connection(self) -> str: ...
```

Returns PostgreSQL connection string in the format:

```text
postgresql+psycopg://user:password@host:port/database
```

**Example:**

```python
from markdown_rag.config import Env

settings = Env()
connection_string = settings.postgres_connection
```

### CLIArgs

```python
class CLIArgs(BaseSettings):
    directory: CliPositionalArg[Path] = Field(default=...)
    command: Command = Field(default=Command.MCP)
    level: LogLevel = Field(default=LogLevel.WARNING)
```

Command-line argument parser using Pydantic.

**Example:**

```python
from markdown_rag.config import get_cli_args

args = get_cli_args()
print(f"Directory: {args.directory}")
print(f"Command: {args.command}")
print(f"Log level: {args.level}")
```

---

## Utilities

### get_token_count()

**Location:** `src/markdown_rag/rate_limiter.py`

```python
@lru_cache(maxsize=20)
def get_token_count(prompt: str, tokenizer: Callable[[str], int]) -> int: ...
```

Cached token counting function.

**Parameters:**

- `prompt` (str): Text to count tokens for
- `tokenizer` (Callable): Tokenizer function

**Returns:**

- `int`: Token count

**Cache:**

- LRU cache with maximum size of 20 entries
- Useful for recursive calls with repeated prompts

---

## MCP Server Integration

**Location:** `markdown_rag/main.py`

### start_store()

Initialize the RAG system with all dependencies.

```python
def start_store(directory: Path, settings: Env) -> MarkdownRAG: ...
```

**Parameters:**

- `directory` (Path): Path to markdown files
- `settings` (Env): Environment configuration

**Returns:**

- `MarkdownRAG`: Fully initialized RAG system

**Example:**

```python
from pathlib import Path
from markdown_rag.config import Env
from markdown_rag.main import start_store

settings = Env()
rag = start_store(Path("./docs"), settings)
```

### run_mcp()

Start the MCP server with the RAG system.

```python
def run_mcp(rag: MarkdownRAG) -> None: ...
```

**Parameters:**

- `rag` (MarkdownRAG): Initialized RAG system

**Exposed Tools:**

#### query

```python
def query(query: str) -> list[RagResponse] | ErrorResponse: ...
```

MCP tool for querying the RAG system.

**Example MCP Call:**

```json
{
  "tool": "query",
  "arguments": {
    "query": "How do I configure authentication?"
  }
}
```

**Response:**

```json
[
  {
    "source": "docs/auth.md",
    "content": "## Authentication\n\nConfigure authentication by..."
  }
]
```

---

## Type Signatures

### Common Types

```python
from pathlib import Path
from collections.abc import Callable, Generator
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector
from pydantic import SecretStr

Generator[tuple[str, Path], None, None]
Generator[list[str], None, None]
Callable[[str], int]
list[list[float]]
dict[str, str]
```

---

## Error Handling

### Common Exceptions

#### Connection Errors

```python
try:
    rag = start_store(directory, settings)
except Exception as e:
    logger.exception(f"Failed to start store: {e}")
    sys.exit(1)
```

#### Ingestion Errors

```python
try:
    rag.ingest()
except Exception as e:
    logger.exception(f"Failed to ingest files: {e}")
    sys.exit(1)
```

#### Query Errors

```python
try:
    results = rag.query(query)
except Exception as e:
    logger.exception(f"Failed to query: {e}")
    return ErrorResponse(error=e)
```

---

## Performance Considerations

### Batching

The system automatically batches embedding requests to maximize throughput:

```python
for batch in rate_limiter.generate_batches(large_text_list):
    pass
```

### Caching

Token counts are cached using `functools.lru_cache`:

```python
@lru_cache(maxsize=20)
def get_token_count(prompt: str, tokenizer: Callable[[str], int]) -> int:
    return tokenizer(prompt)
```

### Memory Management

Usage tracker automatically cleans up old requests:

```python
self._tracker.cleanup_old(current_time)
```

---

## Logging

All modules use Python's standard logging:

```python
import logging

logger = logging.getLogger("MarkdownRAG")
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

Configure logging level via CLI:

```bash
markdown-rag ./docs --level debug
```
