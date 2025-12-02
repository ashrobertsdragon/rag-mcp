# MCP Server Integration Guide

Complete guide for integrating the Markdown RAG system with Model Context Protocol (MCP) clients.

## Overview

The Markdown RAG system exposes tools via the MCP protocol, allowing AI assistants to search your documentation. The MCP client automatically starts and manages the server process.

## Quick Start

### 1. Ingest Documents First

Before configuring the MCP server, ingest your documentation:

```bash
cd markdown-rag
uv run markdown-rag /path/to/docs --command ingest
```

Set environment variables (`.env` file or export):

```env
POSTGRES_PASSWORD=your_password
GOOGLE_API_KEY=your_gemini_api_key
```

### 2. Configure MCP Client

Add to your MCP client configuration file:

**Claude Desktop locations:**

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

**Minimal configuration:**

```json
{
  "mcpServers": {
    "markdown-rag": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
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
        "--directory",
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

Restart your MCP client after configuration changes.

## Available Tools

### query

Semantic search over ingested markdown documentation.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Natural language search query"
    },
    "num_results": {
      "type": "integer",
      "description": "Number of results to return (default: 4)"
    }
  },
  "required": ["query"]
}
```

#### Example Request

```json
{
  "tool": "query",
  "arguments": {
    "query": "How do I configure authentication?",
    "num_results": 4
  }
}
```

#### Success Response

````json
[
  {
    "source": "docs/setup/authentication.md",
    "content": "## Authentication Configuration\n\nTo configure authentication, set the following environment variables:\n\n- `AUTH_SECRET`: Secret key for JWT signing\n- `AUTH_ISSUER`: JWT issuer identifier"
  },
  {
    "source": "docs/api/auth-endpoints.md",
    "content": "### POST /auth/login\n\nAuthenticate a user and receive a JWT token.\n\n**Request Body:**\n```json\n{\n  \"email\": \"user@example.com\",\n  \"password\": \"secure_password\"\n}\n```"
  }
]
````

#### Error Response

```json
{
  "error": "Exception details here"
}
```

### list_documents

List all documents currently in the vector store.

#### Example Request

```json
{
  "tool": "list_documents",
  "arguments": {}
}
```

#### Success Response

```json
[
  "docs/api/auth.md",
  "docs/setup/installation.md"
]
```

### delete_document

Remove a document from the vector store.

#### Example Request

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

#### Example Request

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

#### Example Request

```json
{
  "tool": "refresh_index",
  "arguments": {}
}
```

## Disabling Tools

You can disable specific tools using the `DISABLED_TOOLS` environment variable. This is useful for restricting write access in production environments.

```env
DISABLED_TOOLS=delete_document,update_document,refresh_index
```

When a tool is disabled, it will not be registered with the MCP server and will be unavailable to clients.

## Usage Examples

### From Claude Desktop

Once configured, you can ask Claude to search your documentation:

**User:** "Can you check our docs for how to set up the development environment?"

**Claude:** *Uses the markdown-rag query tool*

```json
{
  "tool": "query",
  "arguments": {
    "query": "development environment setup"
  }
}
```

**Result:** Claude receives relevant documentation chunks and provides an answer based on your actual docs.

### Common Query Patterns

**Configuration Questions:**

```text
"How do I configure database connections?"
"Where do I put API keys?"
"What environment variables are required?"
```

**How-To Questions:**

```text
"How do I deploy to production?"
"How do I run tests?"
"How do I add a new feature?"
```

**Troubleshooting:**

```text
"How do I fix authentication errors?"
"What to do when the build fails?"
"How do I debug slow queries?"
```

**Architecture Questions:**

```text
"What is the authentication flow?"
"How does the caching system work?"
"What services communicate with the database?"
```

## Logging and Debugging

### Enable Debug Logging

Add `--level debug` to the args in your MCP client configuration:

```json
"args": [
  "run",
  "--directory",
  "/absolute/path/to/markdown-rag",
  "markdown-rag",
  "/absolute/path/to/docs",
  "--command",
  "mcp",
  "--level",
  "debug"
]
```

### Viewing Logs

**Claude Desktop Logs:**

- macOS: `~/Library/Logs/Claude/mcp*.log`
- Windows: `%APPDATA%\Claude\logs\mcp*.log`

## Security Considerations

### Environment Variable Safety

Avoid hardcoding credentials directly in configuration files. Use system environment variables or a `.env` file in a secure location.

## Troubleshooting

### Server Won't Start

Check MCP client logs for errors. Verify configuration syntax, absolute paths, and required environment variables.

### No Results from Queries

Ensure documents are ingested first. Check that collection name matches directory name.

### Rate Limit Errors

Adjust rate limits in environment variables:

```env
RATE_LIMIT_REQUESTS_PER_MINUTE=50
```

## Advanced Configuration

### Multiple Document Collections

Configure multiple server instances in your MCP client, each with different directory paths. Each uses its own collection in PostgreSQL.
