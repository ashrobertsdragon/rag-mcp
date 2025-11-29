# MCP Server Integration Guide

Complete guide for integrating the Markdown RAG system with Model Context Protocol (MCP) clients.

## Overview

The Markdown RAG system exposes a `query` tool via the MCP protocol, allowing AI assistants like Claude to search your documentation and retrieve relevant information.

## Quick Start

Add the server to your MCP client configuration (e.g., Claude Desktop).

### Claude Desktop Integration

Edit your Claude Desktop configuration file:

**Location:**

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

**Configuration:**

```json
{
  "mcpServers": {
    "markdown-rag": {
      "command": "markdown-rag",
      "args": ["/absolute/path/to/docs", "--command", "mcp"],
      "env": {
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "your_password",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "embeddings",
        "GOOGLE_API_KEY": "your_gemini_api_key"
      }
    }
  }
}
```

**Important Notes:**

- Use absolute paths for the directory argument
- Include all required environment variables
- Restart Claude Desktop after configuration changes

### Using with uv

If you installed with uv, use the full path to the executable:

```json
{
  "mcpServers": {
    "markdown-rag": {
      "command": "uv",
      "args": [
        "run",
        "markdown-rag",
        "/absolute/path/to/docs",
        "--command",
        "mcp"
      ],
      "env": {
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "your_password",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "embeddings",
        "GOOGLE_API_KEY": "your_gemini_api_key"
      }
    }
  }
}
```

### Environment Variables via .env

Instead of hardcoding credentials in the configuration, you can use a `.env` file:

```json
{
  "mcpServers": {
    "markdown-rag": {
      "command": "markdown-rag",
      "args": ["/path/to/docs", "--command", "mcp"],
      "cwd": "/path/to/project"
    }
  }
}
```

The server will load environment variables from `.env` in the working directory.

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
    "query": "How do I configure authentication?"
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

## Server Lifecycle

### Starting the Server

```bash
markdown-rag /path/to/docs --command mcp
```

The server:

1. Loads environment variables
2. Connects to PostgreSQL
3. Initializes the embeddings model
4. Sets up rate limiting
5. Starts listening on stdio

### Error Handling

The server logs errors but remains running for:

- Query failures
- Rate limit violations
- Database connection issues (retries)

The server exits for:

- Configuration errors
- Fatal database errors
- Missing dependencies

## Logging and Debugging

### Enable Debug Logging

```json
{
  "mcpServers": {
    "markdown-rag": {
      "command": "markdown-rag",
      "args": [
        "/path/to/docs",
        "--command",
        "mcp",
        "--level",
        "debug"
      ],
      "env": { ... }
    }
  }
}
```

**Log Levels:**

- `debug`: All details including queries, rate limits, embeddings
- `info`: Query results, connection status (default)
- `error`: Errors only

### Viewing Logs

**Claude Desktop Logs:**

- macOS: `~/Library/Logs/Claude/mcp*.log`
- Windows: `%APPDATA%\Claude\logs\mcp*.log`

**Example Debug Output:**

```text
DEBUG:MarkdownRAG:Initializing embeddings model connection
DEBUG:MarkdownRAG:Initializing rate limiter
DEBUG:MarkdownRAG:Initializing vector store
INFO:MarkdownRAG:MCP server started
DEBUG:MarkdownRAG:Received query: "How do I configure auth?"
DEBUG:RateLimiter:Request allowed: 1 req/min, 1 req/day, 24 tokens/min
DEBUG:MarkdownRAG:Found 4 relevant documents
INFO:MarkdownRAG:Query completed in 0.34s
```

### Rate Limit Tuning

Rate limit defaults are based on free tier. Adjust for your tier:

```env
RATE_LIMIT_REQUESTS_PER_MINUTE=100
RATE_LIMIT_REQUESTS_PER_DAY=1000
```

## Security Considerations

### Environment Variable Safety

Never hardcode credentials in configuration files:

```json
{
  "env": {
    "POSTGRES_PASSWORD": "${POSTGRES_PASSWORD}",
    "GOOGLE_API_KEY": "${GOOGLE_API_KEY}"
  }
}
```

Use system environment variables or a `.env` file in a secure location.

### Network Security

**For remote PostgreSQL:**

- Use SSL/TLS connections:

   ```env
   POSTGRES_HOST=db.example.com
   POSTGRES_SSLMODE=require
   ```

- Restrict database access:

   ```sql
   CREATE USER rag_reader WITH PASSWORD 'secure_password';
   GRANT SELECT ON ALL TABLES IN SCHEMA public TO rag_reader;
   ```

- Use firewall rules to limit connections

### API Key Protection

- Store Google API keys securely
- Rotate keys regularly
- Use separate keys for development and production
- Monitor API usage for anomalies

## Troubleshooting

### Server Won't Start

**Problem:** Server fails to start in Claude Desktop

**Solutions:**

- Check logs for specific error:

   ```bash
   tail -f ~/Library/Logs/Claude/mcp*.log
   ```

- Verify configuration syntax (valid JSON)
- Test server manually:

   ```bash
   markdown-rag /path/to/docs --command mcp --level debug
   ```

- Check all paths are absolute
- Verify all environment variables are set

### No Results from Queries

**Problem:** Queries return empty results

**Solutions:**

- Verify documents are ingested:

   ```bash
   markdown-rag /path/to/docs --command ingest --level info
   ```

- Check collection name matches directory name
- Try a broader query
- Verify database contains embeddings:

   ```sql
   SELECT COUNT(*) FROM langchain_pg_embedding;
   ```

### Slow Query Performance

**Problem:** Queries take too long

**Solutions:**

- Check database indexes (advanced):

   ```sql
   CREATE INDEX ON langchain_pg_embedding USING ivfflat (embedding vector_cosine_ops);
   ```

- Monitor rate limiting delays (check logs)
- Use SSD storage for PostgreSQL

### Rate Limit Errors

**Problem:** Frequent rate limit violations

**Solutions:**

- Reduce request limits:

   ```env
   RATE_LIMIT_REQUESTS_PER_MINUTE=50
   ```

- Check API quota usage in Google Cloud Console
- Upgrade API quota if needed
- Review query patterns (avoid rapid repeated queries)

## Advanced Configuration

### Multiple Document Collections

Serve multiple documentation sets:

```json
{
  "mcpServers": {
    "main-docs": {
      "command": "markdown-rag",
      "args": ["/path/to/main-docs", "--command", "mcp"],
      "env": { ... }
    },
    "api-docs": {
      "command": "markdown-rag",
      "args": ["/path/to/api-docs", "--command", "mcp"],
      "env": { ... }
    }
  }
}
```

Each server instance uses its own collection in PostgreSQL.

## Testing MCP Integration

### Manual Testing

Use the MCP Inspector tool:

```bash
npx @modelcontextprotocol/inspector markdown-rag /path/to/docs --command mcp
```

This opens a web interface to test the MCP server directly.

## Best Practices

### Documentation Structure

- **Use clear headers:** Helps with semantic chunking
- **Include context:** Standalone sections are more useful
- **Avoid duplication:** Maintain single source of truth
- **Regular updates:** Re-ingest when docs change

### Query Optimization

- **Be specific:** "How to configure OAuth2?" vs "authentication"
- **Use natural language:** Questions work better than keywords
- **Include context:** "How to debug production errors?" vs "debug"

### Operational Practices

- **Monitor logs:** Check for errors and performance issues
- **Update regularly:** Re-ingest docs after changes
- **Test queries:** Validate results for common questions
- **Version control:** Track configuration changes

## Support and Resources

- **Documentation:** [User Guide](user-guide.md), [API Reference](api-reference.md)
- **Architecture:** [Architecture Documentation](architecture.md)
- **Issues:** [GitHub Issues](https://github.com/yourusername/markdown-rag/issues)
- **MCP Specification:** [Model Context Protocol](https://spec.modelcontextprotocol.io/)
