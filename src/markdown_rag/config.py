"""Configuration and command-line argument parsing."""

from pathlib import Path

from pydantic import (
    AliasChoices,
    Field,
    PostgresDsn,
    SecretStr,
    ValidationError,
    field_serializer,
    field_validator,
)
from pydantic_settings import (
    BaseSettings,
    CliApp,
    CliPositionalArg,
    SettingsConfigDict,
)

from .models import Command, LogLevel


class Env(BaseSettings):
    """Environment variable loading."""

    model_config = SettingsConfigDict(env_file=".env")

    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: SecretStr = Field(default=...)
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: str = Field(default="5432")
    POSTGRES_DB: str = Field(default="embeddings")
    GOOGLE_API_KEY: SecretStr = Field(default=...)

    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(
        default=100, description="Maximum requests per minute"
    )
    RATE_LIMIT_REQUESTS_PER_DAY: int = Field(
        default=1000, description="Maximum requests per day"
    )

    DISABLED_TOOLS: str | None = Field(
        default=None,
        description="Comma delimited list of MCP tools to disable",
    )

    @field_serializer(
        "GOOGLE_API_KEY", "POSTGRES_PASSWORD", when_used="always"
    )
    def dump_secret(self, v: SecretStr) -> str:
        """Get secret value."""
        return v.get_secret_value()

    @field_validator("DISABLED_TOOLS", mode="after")
    def validate_disabled_tools(self, v: str | None) -> list[str]:
        """Parse and validate comma delimited list."""
        if v is None:
            return []
        tools = [tool.lower() for tool in v.split(",")]
        allowed_tools = [
            "query",
            "refresh_index",
            "delete_document",
            "update_document",
            "list_documents",
        ]
        for tool in tools:
            if tool not in allowed_tools:
                raise ValidationError(f"Invalid tool: {tool}")
        return tools

    @property
    def postgres_connection(self) -> str:
        """Postgres connection string."""
        schema = "postgresql+psycopg"
        pw = self.POSTGRES_PASSWORD.get_secret_value()
        auth = f"{self.POSTGRES_USER}:{pw}"
        url = f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}"

        return PostgresDsn(
            f"{schema}://{auth}@{url}/{self.POSTGRES_DB}"
        ).encoded_string()


class CLIArgs(BaseSettings):
    """Command line parser."""

    model_config = SettingsConfigDict(
        cli_exit_on_error=True, cli_parse_args=True
    )
    directory: CliPositionalArg[Path] = Field(default=...)
    command: Command = Field(
        default=Command.MCP, validation_alias=AliasChoices("c", "command")
    )
    level: LogLevel = Field(
        default=LogLevel.WARNING, validation_alias=AliasChoices("l", "level")
    )


def get_cli_args() -> CLIArgs:
    """Get command line arguments."""
    return CliApp.run(CLIArgs)
