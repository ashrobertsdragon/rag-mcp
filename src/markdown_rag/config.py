"""Configuration and command-line argument parsing."""

from pathlib import Path
from typing import ClassVar

from pydantic import (
    AliasChoices,
    Field,
    PostgresDsn,
    SecretStr,
    field_serializer,
    field_validator,
)
from pydantic_settings import (
    BaseSettings,
    CliApp,
    CliPositionalArg,
    SettingsConfigDict,
)

from markdown_rag.models import Command, EmbeddingEngine, LogLevel


class Env(BaseSettings):
    """Base environment settings."""

    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: SecretStr = Field(default=...)
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: str = Field(default="5432")
    POSTGRES_DB: str | None = Field(default=None)

    DISABLED_TOOLS: list[str] = Field(
        default_factory=list,
        description="Comma delimited list of MCP tools to disable",
    )

    CHUNK_OVERLAP: int = Field(default=50)

    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(
        default=100, description="Maximum requests per minute"
    )
    RATE_LIMIT_REQUESTS_PER_DAY: int = Field(
        default=1000, description="Maximum requests per day"
    )

    @field_validator("DISABLED_TOOLS", mode="before")
    @classmethod
    def validate_disabled_tools(cls, v: str | list[str] | None) -> list[str]:
        """Parse and validate comma delimited list."""
        allowed_tools = {
            "query",
            "refresh_index",
            "delete_document",
            "update_document",
            "list_documents",
        }

        def _validate_tool(tool: str) -> str:
            tool = tool.strip().lower()
            if tool not in allowed_tools:
                raise ValueError(f"Invalid tool: {tool}")
            return tool

        if not v:
            return []
        tools = v if isinstance(v, list) else v.split(",")
        return [_validate_tool(tool) for tool in tools if tool.strip()]

    @property
    def embedding_engine(self) -> EmbeddingEngine:
        """Embedding engine to retrieve settings for."""
        raise NotImplementedError

    @property
    def postgres_root_connection(self) -> str:
        """Postgres connection string to without database."""
        schema = "postgresql+psycopg"
        pw = self.POSTGRES_PASSWORD.get_secret_value()
        auth = f"{self.POSTGRES_USER}:{pw}"
        url = f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}"

        return PostgresDsn(f"{schema}://{auth}@{url}").encoded_string()

    @property
    def postgres_connection(self) -> str:
        """Postgres connection string."""
        return f"{self.postgres_root_connection}/{self.db_name}"

    @property
    def db_name(self) -> str:
        """Set default postgres db."""
        return self.POSTGRES_DB or f"{self.embedding_engine}_embeddings"

    @property
    def chunk_size(self) -> int:
        """Get chunk size."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Get string representation."""
        return f"{self.__class__.__name__}({self.__dict__})"

    def __str__(self) -> str:
        """Get string name."""
        return self.__class__.__name__


class GoogleEnv(Env):
    """Google environment settings."""

    GOOGLE_API_KEY: SecretStr = Field(default=...)
    GOOGLE_MODEL: str = Field(default="models/gemini-embedding-001")
    GOOGLE_CHUNK_SIZE: int = Field(default=2000)

    embeding_engine: ClassVar[EmbeddingEngine] = EmbeddingEngine.GOOGLE

    @field_serializer("GOOGLE_API_KEY", when_used="always")
    def dump_google_secret(self, v: SecretStr) -> str:
        """Get secret value."""
        return v.get_secret_value()

    @property
    def chunk_size(self) -> int:
        """Get chunk size."""
        return self.GOOGLE_CHUNK_SIZE


class OllamaEnv(Env):
    """Ollama environment settings."""

    OLLAMA_HOST: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL: str = Field(default="mxbai-embed-large")
    OLLAMA_CHUNK_SIZE: int = Field(default=500)

    embeding_engine: ClassVar[EmbeddingEngine] = EmbeddingEngine.OLLAMA

    @property
    def chunk_size(self) -> int:
        """Get chunk size."""
        return self.OLLAMA_CHUNK_SIZE


class CLIArgs(BaseSettings):
    """Command line parser."""

    model_config = SettingsConfigDict(
        cli_exit_on_error=True, cli_parse_args=True
    )
    directory: CliPositionalArg[Path] = Field(default=...)
    command: Command = Field(
        default=Command.MCP, validation_alias=AliasChoices("c", "command")
    )
    engine: EmbeddingEngine = Field(
        default=EmbeddingEngine.GOOGLE,
        validation_alias=AliasChoices("e", "engine"),
    )
    level: LogLevel = Field(
        default=LogLevel.WARNING, validation_alias=AliasChoices("l", "level")
    )
    env_file: Path = Field(default=Path(".env"))


ENGINES: dict[EmbeddingEngine, type[Env]] = {
    EmbeddingEngine.GOOGLE: GoogleEnv,
    EmbeddingEngine.OLLAMA: OllamaEnv,
}


def get_cli_args() -> CLIArgs:
    """Get command line arguments."""
    return CliApp.run(CLIArgs)


def get_env(engine: EmbeddingEngine) -> type[Env]:
    """Get environment settings."""
    return ENGINES[engine]
