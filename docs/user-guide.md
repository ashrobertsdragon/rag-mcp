# User Guide

Complete guide to using the Markdown RAG system for semantic search over your documentation.

## Table of Contents

- [Getting Started](#getting-started)
- [Installation](#installation)
- [Configuration](#configuration)
- [Ingesting Documents](#ingesting-documents)
- [Running the MCP Server](#running-the-mcp-server)
- [Querying Documents](#querying-documents)
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
- A Google Gemini API key
- Markdown documentation to index

## Installation

### Step 1: Install PostgreSQL and pgvector

**On Ubuntu/Debian:**

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo apt install postgresql-17-pgvector
```

**On macOS (Homebrew):**

```bash
brew install postgresql@17
brew install pgvector
```

**On Windows:**

Download PostgreSQL from [postgresql.org](https://www.postgresql.org/download/windows/) and install pgvector following the [official instructions](https://github.com/pgvector/pgvector#windows).

### Step 2: Create Database

```bash
createdb embeddings

psql embeddings -c "CREATE EXTENSION vector;"
```

### Step 3: Install Markdown RAG

**Using uv (recommended):**

```bash
git clone https://github.com/yourusername/markdown-rag.git
cd markdown-rag
uv sync
```

**Using pip:**

```bash
git clone https://github.com/yourusername/markdown-rag.git
cd markdown-rag
pip install -e .
```

### Step 4: Verify Installation

```bash
markdown-rag --help
```

You should see the help message with available commands and options.

## Configuration

### Environment Variables

Create a `.env` file in your project directory:

```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=embeddings

GOOGLE_API_KEY=your_gemini_api_key_here

RATE_LIMIT_REQUESTS_PER_MINUTE=100
RATE_LIMIT_REQUESTS_PER_DAY=1000
```

### Getting a Google Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
1. Sign in with your Google account
1. Click "Create API Key"
1. Copy the key and add it to your `.env` file

### Configuration Options Explained

| Variable                         | Purpose           | Default      | Notes                                            |
| -------------------------------- | ----------------- | ------------ | ------------------------------------------------ |
| `POSTGRES_USER`                  | Database username | `postgres`   | Use a dedicated user for production              |
| `POSTGRES_PASSWORD`              | Database password | *required*   | Never commit this to version control             |
| `POSTGRES_HOST`                  | Database server   | `localhost`  | Use hostname or IP for remote databases          |
| `POSTGRES_PORT`                  | Database port     | `5432`       | PostgreSQL default port                          |
| `POSTGRES_DB`                    | Database name     | `embeddings` | Create separate databases for different projects |
| `GOOGLE_API_KEY`                 | Gemini API key    | *required*   | Get from Google AI Studio                        |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | API rate limit    | `100`        | Adjust based on your API quota                   |
| `RATE_LIMIT_REQUESTS_PER_DAY`    | Daily API limit   | `1000`       | Adjust based on your API quota                   |

## Ingesting Documents

### Basic Ingestion

To ingest all markdown files from a directory:

```bash
markdown-rag /path/to/docs --command ingest
```

**Example:**

```bash
markdown-rag ./documentation --command ingest
```

### What Happens During Ingestion

1. **File Discovery**: Recursively scans the directory for markdown files
1. **Duplicate Check**: Skips files already in the database
1. **Text Splitting**: Splits documents at markdown headers and by character count
1. **Embedding Generation**: Creates semantic embeddings using Google Gemini
1. **Storage**: Saves embeddings and metadata to PostgreSQL

### Progress Monitoring

Enable INFO-level logging to see ingestion progress:

```bash
markdown-rag ./docs --command ingest --level info
```

Output:

```text
INFO:MarkdownRAG:Ingesting files from ./docs
INFO:MarkdownRAG:Ingesting docs/getting-started.md
INFO:MarkdownRAG:Ingesting docs/api/reference.md
INFO:MarkdownRAG:Skipping docs/intro.md (already in vector store)
```

### Incremental Updates

The system automatically skips documents that have already been ingested for efficiency.

### Ingestion Performance

Performance depends on:

- Number of documents
- Document size
- API rate limits

**Typical times:**

- 100 files (~500KB): 2-3 minutes
- 1000 files (~5MB): 20-30 minutes
- 10000 files (~50MB): 3-5 hours

## Running the MCP Server

### Starting the Server

```bash
markdown-rag /path/to/docs --command mcp
```

The server runs on stdio and waits for MCP client connections.

### Integrating with Claude Desktop

Add to your Claude Desktop configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "markdown-rag": {
      "command": "markdown-rag",
      "args": ["/path/to/docs", "--command", "mcp"],
      "env": {
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "your_password",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "embeddings",
        "GOOGLE_API_KEY": "your_api_key"
      }
    }
  }
}
```

### Server Logging

Enable debug logging for troubleshooting:

```bash
markdown-rag ./docs --command mcp --level debug
```

## Querying Documents

### Via MCP Client

When connected via MCP, query using the `query` tool:

**Query:**

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
    "source": "docs/setup/authentication.md",
    "content": "## Authentication Configuration\n\nTo configure authentication..."
  },
  {
    "source": "docs/api/auth.md",
    "content": "### Auth Endpoints\n\nThe authentication API provides..."
  }
]
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

**Problem:** `Failed to start store: connection refused`

**Solution:**

1. Check PostgreSQL is running:

   ```bash
   pg_ctl status
   ```

1. Verify connection settings:

   ```bash
   psql -h localhost -U postgres -d embeddings
   ```

1. Check firewall rules if using remote database

### API Rate Limits

**Problem:** Slow ingestion or rate limit errors

**Solution:**

1. Check current limits:

   ```bash
   grep RATE_LIMIT .env
   ```

1. Reduce limits if hitting API quotas:

   ```env
   RATE_LIMIT_REQUESTS_PER_MINUTE=50
   RATE_LIMIT_REQUESTS_PER_DAY=500
   ```

1. Monitor with debug logging:

   ```bash
   markdown-rag ./docs --command ingest --level debug
   ```

### pgvector Extension Missing

**Problem:** `pgvector extension not found`

**Solution:**

```bash
psql embeddings -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

If installation fails, ensure pgvector is installed for your PostgreSQL version.

**Windows users**: Ensure pgvector is installed via these recommended instructions; [PGVector Installation Guide for Windows 11](https://github.com/ranga-NSL/pgvector4windows/blob/main/pgvector_installation_guide.md)

### Poor Search Results

**Problem:** Queries return irrelevant documents

**Solutions:**

1. **Use more specific queries:**

   - Instead of: "setup"
   - Try: "How do I set up the development environment?"

2. **Check document structure:**

   - Ensure markdown has clear headers
   - Avoid very long sections (>2000 chars without headers)

### Memory Issues

**Problem:** High memory usage during ingestion

**Solution:**

The system streams files and processes them in batches. Memory issues usually indicate:

- Extremely large individual documents
- Database connection issues causing backlog

Try processing in smaller batches:

```bash
markdown-rag ./docs/subset1 --command ingest
markdown-rag ./docs/subset2 --command ingest
```

### Logging and Debugging

**Enable full debug output:**

```bash
markdown-rag ./docs --command ingest --level debug 2>&1 | tee debug.log
```

**Log levels:**

- `debug`: All details including token counts, batch sizes
- `info`: Progress updates, file names
- `warning`: Only warnings and errors (default)
- `error`: Only errors

## Next Steps

- Read the [API Reference](api-reference.md) for programmatic usage
- See [Architecture Documentation](architecture.md) for system details
- Join discussions and report issues on GitHub

## Getting Help

- **Documentation**: Check this guide and the API reference
- **Issues**: Report bugs on [GitHub Issues](https://github.com/yourusername/markdown-rag/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/yourusername/markdown-rag/discussions)
