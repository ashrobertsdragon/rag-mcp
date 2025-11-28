"""Data models for the RAG system."""

from enum import IntEnum, StrEnum

from pydantic import BaseModel, ConfigDict


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
