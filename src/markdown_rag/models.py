"""Data models for the RAG system."""

from enum import IntEnum, StrEnum
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict

DocName: TypeAlias = str


class Command(StrEnum):
    """Commands for the RAG system."""

    INGEST = "ingest"
    MCP = "mcp"


class EmbeddingEngine(StrEnum):
    """Embedding engines for the RAG system."""

    GOOGLE = "google"
    OLLAMA = "ollama"


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


MultiResponse: TypeAlias = list[RagResponse] | list[DocName]


class ToolResponse(BaseModel):
    """Unified response model for MCP tools."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    error: str | None = None
    success: bool = True
    data: MultiResponse | bool | None = None
