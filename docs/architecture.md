# Markdown RAG Architecture

## System Overview

The Markdown RAG system is a Retrieval Augmented Generation (RAG) solution that ingests markdown files, stores them as embeddings in a PostgreSQL vector database, and provides semantic search capabilities through an MCP (Model Context Protocol) server.

## Architecture Diagram

```mermaid
graph TB
    subgraph "Client Layer"
        CLI[CLI Arguments]
        MCP_Client[MCP Client]
    end

    subgraph "Application Layer"
        Main[Main Entry Point]
        RAG[MarkdownRAG Core]
        MCP_Server[FastMCP Server]
    end

    subgraph "Processing Layer"
        MDSplitter[Markdown Header Splitter]
        RecSplitter[Recursive Text Splitter]
        RateLimitedEmbed[Rate Limited Embeddings]
    end

    subgraph "Rate Limiting"
        RateLimiter[Rate Limiter]
        TokenCounter[Token Counter]
        UsageTracker[Usage Tracker]
    end

    subgraph "External Services"
        Gemini[Google Gemini API]
        PGVector[PostgreSQL + pgvector]
    end

    CLI --> Main
    Main --> RAG
    Main --> MCP_Server
    MCP_Client --> MCP_Server
    MCP_Server --> RAG

    RAG --> MDSplitter
    MDSplitter --> RecSplitter
    RecSplitter --> RateLimitedEmbed
    RAG --> PGVector

    RateLimitedEmbed --> RateLimiter
    RateLimitedEmbed --> Gemini
    RateLimiter --> TokenCounter
    RateLimiter --> UsageTracker
    TokenCounter --> Gemini
```

## Component Architecture

```mermaid
graph LR
    subgraph "Core Components"
        A[MarkdownRAG]
        B[RateLimitedEmbeddings]
        C[RateLimiter]
    end

    subgraph "Data Models"
        D[Command]
        E[RagResponse]
        F[ErrorResponse]
    end

    subgraph "Configuration"
        G[Env Settings]
        H[CLI Args]
    end

    A --> B
    B --> C
    A --> E
    A --> F
    A --> G
    H --> A
```

## Data Flow

### Ingestion Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant MarkdownRAG
    participant Splitters
    participant RateLimiter
    participant Gemini
    participant PGVector

    User->>CLI: uv run src/markdown_rag/main.py /docs --command ingest
    CLI->>MarkdownRAG: ingest()

    loop For each markdown file
        MarkdownRAG->>MarkdownRAG: Check if exists
        alt Document already exists
            MarkdownRAG->>MarkdownRAG: Skip file
        else New document
            MarkdownRAG->>Splitters: Split by headers
            Splitters->>Splitters: Recursive split
            Splitters->>RateLimiter: Request batch processing
            RateLimiter->>RateLimiter: Calculate batch size
            RateLimiter->>RateLimiter: Wait if needed
            RateLimiter->>Gemini: Generate embeddings
            Gemini->>PGVector: Store vectors + metadata
        end
    end

    PGVector->>User: Ingestion complete
```

### Query Flow

```mermaid
sequenceDiagram
    participant MCP_Client
    participant FastMCP
    participant MarkdownRAG
    participant PGVector
    participant RateLimiter
    participant Gemini

    MCP_Client->>FastMCP: query("How to setup?")
    FastMCP->>MarkdownRAG: query(query_text)
    MarkdownRAG->>RateLimiter: wait_if_needed(query_text)
    RateLimiter->>RateLimiter: Check limits
    RateLimiter->>Gemini: Embed query
    Gemini->>MarkdownRAG: Query embedding
    MarkdownRAG->>PGVector: similarity_search()
    PGVector->>MarkdownRAG: Relevant documents
    MarkdownRAG->>FastMCP: List[RagResponse]
    FastMCP->>MCP_Client: JSON response
```

## Rate Limiting Architecture

```mermaid
graph TB
    subgraph "Request Processing"
        A[Incoming Request]
        B[Token Counter]
        C[Usage Tracker]
    end

    subgraph "Sliding Windows"
        D[Minute Window<br/>60 seconds]
        E[Day Window<br/>86400 seconds]
    end

    subgraph "Limits"
        F[Requests/Minute<br/>Default: 100]
        G[Requests/Day<br/>Default: 1000]
        H[Tokens/Minute<br/>Default: 30000]
    end

    subgraph "Actions"
        I{Within Limits?}
        J[Process Request]
        K[Calculate Wait Time]
        L[Sleep & Retry]
    end

    A --> B
    B --> C
    C --> D
    C --> E
    D --> F
    D --> H
    E --> G
    F --> I
    G --> I
    H --> I
    I -->|Yes| J
    I -->|No| K
    K --> L
    L --> C
```

## Technology Stack

| Layer               | Technology                           | Purpose                                |
| ------------------- | ------------------------------------ | -------------------------------------- |
| **Embeddings**      | Google Gemini (gemini-embedding-001) | Generate semantic embeddings from text |
| **Vector Store**    | PostgreSQL + pgvector                | Store and search vector embeddings     |
| **Framework**       | LangChain                            | Document processing and RAG pipeline   |
| **Server Protocol** | FastMCP                              | Model Context Protocol server          |
| **Configuration**   | Pydantic Settings                    | Type-safe environment config           |
| **Text Processing** | LangChain Text Splitters             | Markdown-aware chunking                |

## Key Design Patterns

### 1. Rate Limiting with Sliding Window

The system uses a sophisticated sliding window algorithm to enforce API rate limits:

- **Dual window tracking**: Separate 60-second and 24-hour windows
- **Token counting**: Counts tokens before making requests
- **Batch optimization**: Calculates maximum safe batch sizes
- **Smart waiting**: Only waits when necessary, with minimal delay

### 2. Embeddings Wrapper Pattern

`RateLimitedEmbeddings` wraps the Google Gemini embeddings model:

- Transparent rate limiting
- Automatic batching
- Seamless integration with LangChain

### 3. Document Deduplication

Before ingesting documents, the system checks if they already exist:

- Uses metadata-based queries
- Prevents duplicate embeddings
- Enables incremental updates

### 4. Hierarchical Text Splitting

Two-stage splitting strategy:

1. **Markdown Header Splitter**: Splits on headers (##, ###, ####, #####)
1. **Recursive Character Splitter**: Further splits large sections (2000 chars, 50 overlap)

This preserves semantic boundaries while maintaining manageable chunk sizes.

## Configuration

### Environment Variables

| Variable                         | Default      | Required | Description                              |
| -------------------------------- | ------------ | -------- | ---------------------------------------- |
| `POSTGRES_USER`                  | `postgres`   | No       | PostgreSQL username                      |
| `POSTGRES_PASSWORD`              | -            | **Yes**  | PostgreSQL password                      |
| `POSTGRES_HOST`                  | `localhost`  | No       | PostgreSQL host                          |
| `POSTGRES_PORT`                  | `5432`       | No       | PostgreSQL port                          |
| `POSTGRES_DB`                    | `embeddings` | No       | Database name                            |
| `GOOGLE_API_KEY`                 | -            | **Yes**  | Google Gemini API key                    |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | `100`        | No       | Max API requests per minute              |
| `RATE_LIMIT_REQUESTS_PER_DAY`    | `1000`       | No       | Max API requests per day                 |
| `DISABLED_TOOLS`                 | -            | No       | Comma-separated list of tools to disable |

### Command Line Options

```bash
uv run --directory path/to/markdown-rag markdown-rag <directory> [OPTIONS]
```

Options:

- `-c, --command {ingest|mcp}`: Operation mode (default: mcp)
- `-l, --level {debug|info|warning|error}`: Log level (default: warning)

## Performance Characteristics

### Ingestion Performance

- **Throughput**: Limited by API rate limits (100 req/min, 30K tokens/min)
- **Batching**: Automatically batches requests to maximize throughput
- **Caching**: LRU cache for token counts (size: 20)
- **Deduplication**: O(1) lookup via PostgreSQL metadata queries

### Query Performance

- **Latency**: ~100-500ms (depends on vector store size)
- **Similarity Search**: pgvector cosine similarity
- **No batching**: Queries are processed individually

### Memory Usage

- **Sliding windows**: O(R) where R = requests in 24 hours
- **Token counting cache**: O(1) - bounded at 20 entries
- **Document processing**: Streams files, minimal memory footprint

## Security Considerations

### Secrets Management

- All secrets stored in environment variables
- `SecretStr` type prevents accidental logging
- PostgreSQL DSN properly encoded

### Input Validation

- Pydantic models validate all inputs
- Type checking enforced (mypy)
- CLI arguments validated before execution

### API Key Protection

- Never logged or exposed
- Passed directly to Google client
- Field serialization controlled

## Error Handling

### Ingestion Errors

- Individual file failures logged but don't stop processing
- Duplicate detection prevents reprocessing
- Rate limit violations trigger automatic retry with backoff

### Query Errors

- Wrapped in `ErrorResponse` model
- Exceptions logged with full stack trace
- MCP client receives structured error response

### Database Errors

- Connection failures propagate to caller
- SQLAlchemy handles connection pooling
- Transactions ensure consistency
