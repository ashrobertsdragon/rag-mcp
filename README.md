# Markdown RAG

A Retrieval Augmented Generation (RAG) system for markdown documentation with intelligent rate limiting and MCP server integration.

## Features

- **Semantic Search**: Vector-based similarity search using Google Gemini embeddings
- **Markdown-Aware Chunking**: Intelligent document splitting that preserves semantic boundaries
- **Rate Limiting**: Sophisticated sliding window algorithm with token counting and batch optimization
- **MCP Server**: Model Context Protocol server for AI assistant integration
- **PostgreSQL Vector Store**: Scalable storage using pgvector extension
- **Incremental Updates**: Smart deduplication prevents reprocessing existing documents
- **Production Ready**: Type-safe configuration, comprehensive logging, and error handling

## Installation

```bash
git clone https://github.com/yourusername/markdown-rag.git
```

## Prerequisites

- Python 3.11+
- PostgreSQL 12+ with [pgvector extension installed](https://github.com/pgvector/pgvector#installation)
- Google Gemini API key
- MCP-compatible client (Claude Desktop, Cline, etc.)

## Quick Start

### 1. Set Up PostgreSQL

```bash
createdb embeddings
```

The pgvector extension will be automatically enabled when you first run the tool.

### 2. Ingest Documents

```bash
cd markdown-rag
uv run markdown-rag /path/to/docs --command ingest
```

**Required environment variables** (create `.env` or export):

```env
POSTGRES_PASSWORD=your_password
GOOGLE_API_KEY=your_gemini_api_key
```

### 3. Configure MCP Client

Add to your MCP client configuration (e.g., `claude_desktop_config.json`). The client will automatically start the server.

**Minimal configuration:**

```json
{
  "mcpServers": {
    "markdown-rag": {
      "command": "uv",
      "args": [
        "run",
        "--directory"
        "/absolute/path/to/markdown-rag",
        "markdown-rag",
        "/absolute/path/to/docs",
        "--command",
        "mcp"
      ],
      "env": {
        "POSTGRES_PASSWORD": "your_password",
        "GOOGLE_API_KEY": "your_api_key"
      }
    }
  }
}
```

**Full configuration:**

```json
{
  "mcpServers": {
    "markdown-rag": {
      "command": "uv",
      "args": [
        "run",
        "--directory"
        "/absolute/path/to/markdown-rag",
        "markdown-rag",
        "/absolute/path/to/docs",
        "--command",
        "mcp"
      ],
      "env": {
        "POSTGRES_USER": "postgres_username",
        "POSTGRES_PASSWORD": "your_password",
        "POSTGRES_HOST": "postgres_url",
        "POSTGRES_PORT": "1234", # Postgres connection URL port number
        "POSTGRES_DB": "embeddings",
        "GOOGLE_API_KEY": "your_api_key",
        "RATE_LIMIT_REQUESTS_PER_MINUTE": "100",
        "RATE_LIMIT_REQUESTS_PER_DAY": "1000",
        "DISABLED_TOOLS": "delete_document,update_document"
      }
    }
  }
}
```

### 4. Query via MCP

The server exposes several tools:

#### query

- Semantic search over documentation
- Arguments: `query` (string), `num_results` (integer, optional, default: 4)

#### list_documents

- List all ingested documents
- Arguments: none

#### delete_document

- Remove a document from the index
- Arguments: `filename` (string)

#### update_document

- Re-ingest a specific document
- Arguments: `filename` (string)

#### refresh_index

- Scan directory and ingest new/modified files
- Arguments: none

To disable tools (e.g., in production), set `DISABLED_TOOLS` environment variable:

```env
DISABLED_TOOLS=delete_document,update_document,refresh_index
```

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
uv run markdown-rag <directory> [OPTIONS]
```

**Arguments:**

- `<directory>`: Path to markdown files directory (required)

**Options:**

- `-c, --command {ingest|mcp}`: Operation mode (default: `mcp`)
  - `ingest`: Process and store documents
  - `mcp`: Start MCP server for queries
- `-l, --level {debug|info|warning|error}`: Logging level (default: `warning`)

**Examples:**

```bash
uv run markdown-rag ./docs --command ingest --level info

uv run markdown-rag /var/docs -c ingest -l debug
```

## Architecture

### System Components

The following diagram shows how the system components interact:

```mermaid
graph TD
    A[MCP Client<br/>Claude, ChatGPT, etc.] --> B[FastMCP Server<br/>Tool: query]
    B --> C[MarkdownRAG]
    C --> D[Text Splitters]
    C --> E[Rate Limited Embeddings]
    E --> F[Google Gemini<br/>Embeddings API]
    C --> G[PostgreSQL<br/>+ pgvector]
```

### Rate Limiting Strategy

The system implements a dual-window sliding algorithm:

- **Request Limits**: Tracks requests per minute and per day
- **Token Limits**: Counts tokens before API calls
- **Batch Optimization**: Calculates maximum safe batch sizes
- **Smart Waiting**: Minimal delays with automatic retry

See [Architecture Documentation](docs/architecture.md) for detailed diagrams.

## Development

### Setup Development Environment

```bash
git clone https://github.com/yourusername/markdown-rag.git
cd markdown-rag
uv sync
```

### Run Linters

```bash
uv run ruff check .

uv run mypy .
```

### Code Style

This project follows:

- **Linting**: Ruff with Google docstring convention
- **Type Checking**: mypy with strict settings
- **Line Length**: 79 characters
- **Import Sorting**: Alphabetical with isort

### Project Structure

```text
markdown-rag/
├── src/markdown_rag/
│   ├── __init__.py
│   ├── main.py              # Entry point and MCP server
│   ├── config.py            # Environment and CLI configuration
│   ├── models.py            # Pydantic data models
│   ├── rag.py               # Core RAG logic
│   ├── embeddings.py        # Rate-limited embeddings wrapper
│   └── rate_limiter.py      # Rate limiting algorithm
├── docs/
│   ├── api-reference.md     # API documentation
│   ├── architecture.md      # Architecture documentation
│   ├── mcp-integration.md   # MCP server integration guide
│   └── user-guide.md        # User guide
├── pyproject.toml           # Project configuration
├── .env                     # Environment variables (not in git)
└── README.md
```

## Troubleshooting

### Common Issues

#### "Failed to start store: connection refused"

PostgreSQL not running or wrong connection settings. Check your connection parameters in environment variables.

#### "Rate limit exceeded"

Adjust rate limits in environment variables:

```env
RATE_LIMIT_REQUESTS_PER_MINUTE=50
RATE_LIMIT_REQUESTS_PER_DAY=500
```

#### "pgvector extension not found"

The pgvector PostgreSQL extension is not installed. Follow the [pgvector installation guide](https://github.com/pgvector/pgvector#installation) for your platform.

#### "Skipping all files (already in vector store)"

Expected behavior. The system prevents duplicate ingestion.

### Logging

```bash
uv run markdown-rag ./docs --command ingest --level debug
```

## Security

### Best Practices

- **Never commit `.env` files** - Add to `.gitignore`
- **Use environment variables** for all secrets
- **Restrict database access** - Use firewall rules
- **Rotate API keys** regularly
- **Use read-only database users** for query-only deployments

### Secrets Management

All secrets use `SecretStr` type to prevent accidental logging:

```python
from pydantic import SecretStr

api_key = SecretStr("secret_value")
```

## Contributing

1. Fork the repository
1. Create a feature branch (`git checkout -b feature/amazing-feature`)
1. Make changes and add tests
1. Run linters (`uv run ruff check .`)
1. Run type checks (`uv run mypy .`)
1. Commit changes (`git commit -m 'feat: add amazing feature'`)
1. Push to branch (`git push origin feature/amazing-feature`)
1. Open a Pull Request

### Commit Message Format

Follow conventional commits:

```text
feat: add new feature
fix: resolve bug
docs: update documentation
refactor: improve code structure
test: add tests
chore: update dependencies
```

## TODOS

- Management of embeddings store via MCP tool.
- Add support for other embeddings models.
- Add support for other vector stores.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - RAG framework
- [Google Gemini](https://ai.google.dev/) - Embedding model
- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity search
- [FastMCP](https://github.com/anthropics/fastmcp) - MCP server framework

## Support

- Documentation: [docs/architecture.md](docs/architecture.md)
- Issues: [GitHub Issues](https://github.com/yourusername/markdown-rag/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/markdown-rag/discussions)
