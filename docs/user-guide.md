# User Guide

Complete guide to using the Markdown RAG system for semantic search over your documentation.

## Table of Contents

- [Getting Started](#getting-started)
- [Installation](#installation)
- [Configuration](#configuration)
- [Ingesting Documents](#ingesting-documents)
- [Configuring MCP Client](#configuring-mcp-client)
- [MCP Tools](#mcp-tools)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Getting Started

Markdown RAG is a semantic search system that allows you to:

1. Ingest markdown documentation into a vector database
1. Query your documentation using natural language
1. Integrate with AI assistants via the MCP (Model Context Protocol) server

### Prerequisites

Before you begin, ensure you have:

- Python 3.11 or higher
- PostgreSQL 12+ with pgvector extension installed
- A Google Gemini API key (for Google embeddings)
- Ollama installed and running (for local embeddings)
- Markdown documentation to index

## Installation

### Prerequisites

Install PostgreSQL and pgvector extension following the [pgvector installation guide](https://github.com/pgvector/pgvector#installation).

### Setup

```bash
git clone https://github.com/yourusername/markdown-rag.git
# Install dependencies
uv sync --google | --ollama
# Optional: Create a PostgreSQL database
createdb embeddings
```

If a database is not present, one will be created for you. The pgvector extension will be automatically enabled when you first run the tool.

## Configuration

### Environment Variables

Create a `.env` file in your project directory:

```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=embeddings

GOOGLE_API_KEY=your_gemini_api_key_here  # Only if using Google
GOOGLE_MODEL=models/gemini-embedding-001 # Optional
GOOGLE_CHUNK_SIZE=2000 # Optional

# Or for Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=mxbai-embed-large
OLLAMA_CHUNK_SIZE=500

RATE_LIMIT_REQUESTS_PER_MINUTE=100
RATE_LIMIT_REQUESTS_PER_DAY=1000
```

### Getting a Google Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
1. Sign in with your Google account
1. Click "Create API Key"
1. Copy the key and add it to your `.env` file

### Configuration Options Explained

| Variable                         | Purpose           | Default      | Notes                                                          |
| -------------------------------- | ----------------- | ------------ | -------------------------------------------------------------- |
| `POSTGRES_USER`                  | Database username | `postgres`   | Use a dedicated user for production                            |
| `POSTGRES_PASSWORD`              | Database password | *required*   | Never commit this to version control                           |
| `POSTGRES_HOST`                  | Database server   | `localhost`  | Use hostname or IP for remote databases                        |
| `POSTGRES_PORT`                  | Database port     | `5432`       | PostgreSQL default port                                        |
| `POSTGRES_DB`                    | Database name     | `[engine]_embeddings` | Defaults to `{engine}_embeddings`              |
| `GOOGLE_API_KEY`                 | Gemini API key    | *required*            | Get from Google AI Studio (if using Google)    |
| `GOOGLE_MODEL`                   | Model name        | `models/gemini...`    | Google embedding model                         |
| `OLLAMA_HOST`                    | Ollama host       | `http://localhost...` | URL of Ollama server                           |
| `OLLAMA_MODEL`                   | Model name        | `mxbai-embed-large`   | Ollama embedding model                         |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | API rate limit    | `100`        | Adjust based on your API quota                                 |
| `RATE_LIMIT_REQUESTS_PER_DAY`    | Daily API limit   | `1000`       | Adjust based on your API quota                                 |
| `DISABLED_TOOLS`                 | Disabled tools    | -            | Comma-separated list (e.g., `delete_document,update_document`) |

## Ingesting Documents

### Basic Ingestion

```bash
cd markdown-rag
uv run markdown-rag /path/to/docs --command ingest
# Or use Ollama
uv run markdown-rag /path/to/docs --command ingest --engine ollama
```

### What Happens During Ingestion

1. **File Discovery**: Recursively scans the directory for markdown files
1. **Duplicate Check**: Skips files already in the database
1. **Text Splitting**: Splits documents at markdown headers and by character count
1. **Embedding Generation**: Creates semantic embeddings using Google Gemini
1. **Storage**: Saves embeddings and metadata to PostgreSQL

### Progress Monitoring

```bash
uv run markdown-rag ./docs --command ingest --level info
```

## Configuring MCP Client

**Minimal configuration:**

```json
{
  "mcpServers": {
    "markdown-rag": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/markdown-rag/",
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
        "--directory",
        "/absolute/path/to/markdown-rag/",
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
        # ... other Google or Ollama settings ...
        "RATE_LIMIT_REQUESTS_PER_MINUTE": "100",
        "RATE_LIMIT_REQUESTS_PER_DAY": "1000",
        "DISABLED_TOOLS": "delete_document,update_document"
      }
    }
  }
}
```

## MCP Tools

### query

Semantic search over documentation.

**Arguments:**

- `query` (string, required)
- `num_results` (integer, optional, default: 4)

**Example:**

```json
{
  "tool": "query",
  "arguments": {
    "query": "How do I configure authentication?",
    "num_results": 4
  }
}
```

### list_documents

List all documents currently in the vector store.

**Example:**

```json
{
  "tool": "list_documents",
  "arguments": {}
}
```

### delete_document

Remove a document from the vector store.

**Arguments:**

- `filename` (string, required)

**Example:**

```json
{
  "tool": "delete_document",
  "arguments": {
    "filename": "docs/old-file.md"
  }
}
```

### update_document

Re-ingest a specific document, updating its embeddings.

**Arguments:**

- `filename` (string, required)

**Example:**

```json
{
  "tool": "update_document",
  "arguments": {
    "filename": "docs/updated-file.md"
  }
}
```

### refresh_index

Scan the directory and ingest any new or modified files.

**Example:**

```json
{
  "tool": "refresh_index",
  "arguments": {}
}
```

### Disabling Tools

Set `DISABLED_TOOLS` environment variable:

```env
DISABLED_TOOLS=delete_document,update_document,refresh_index
```

### Query Tips

**Be specific:**

- ❌ "authentication"
- ✅ "How do I configure OAuth authentication?"

**Use natural language:**

- ❌ "API key config"
- ✅ "Where do I put my API key?"

**Ask about concepts:**

- ✅ "What is the rate limiting strategy?"
- ✅ "How does the system handle errors?"

### Understanding Results

Each result contains:

- `source`: Relative file path (e.g., `docs/api/endpoints.md`)
- `content`: Relevant text chunk with preserved markdown formatting

The system returns the top 4 most semantically similar chunks by default.

## Best Practices

### Document Organization

**Use clear hierarchy:**

```text
docs/
├── getting-started/
│   ├── installation.md
│   └── quick-start.md
├── guides/
│   ├── configuration.md
│   └── deployment.md
└── api/
    └── reference.md
```

**Use meaningful headers:**

```markdown
## Installation on Linux


### Prerequisites

### Installing Dependencies
```

### Writing Searchable Documentation

**Include context:**

```markdown
## User Authentication

The system supports JWT-based authentication for API requests.

### Generating Tokens

To generate an authentication token, send a POST request...
```

**Avoid orphaned headings:**

```markdown
## Database Configuration


Provide description here, not just a heading.

### Connection Settings

Configure the connection string...
```

### Database Maintenance

**Regular backups:**

```bash
pg_dump embeddings > embeddings_backup.sql
```

## Troubleshooting

### Connection Issues

PostgreSQL not running or wrong connection settings. Check your connection parameters.

### API Rate Limits

Adjust rate limits in environment variables:

```env
RATE_LIMIT_REQUESTS_PER_MINUTE=50
RATE_LIMIT_REQUESTS_PER_DAY=500
```

### pgvector Extension Missing

The pgvector PostgreSQL extension is not installed. Follow the [pgvector installation guide](https://github.com/pgvector/pgvector#installation) for your platform.

### Logging and Debugging

```bash
uv run markdown-rag ./docs --command ingest --level debug
```
