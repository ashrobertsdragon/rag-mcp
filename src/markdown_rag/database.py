"""Database utilities."""

from logging import getLogger

from sqlalchemy import Engine, create_engine, text

from markdown_rag.config import Env

logger = getLogger(__name__)


def create_database(engine: Engine, user: str, db_name: str) -> None:
    """Create a database."""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
            {"db_name": db_name},
        )
        if result.scalar():
            logger.info(f"Database {db_name} already exists")
            return

        logger.info(f"Creating database: {db_name}")
        quoted_db = engine.dialect.identifier_preparer.quote(db_name)
        quoted_user = engine.dialect.identifier_preparer.quote(user)
        conn.execute(
            text(f"CREATE DATABASE {quoted_db} OWNER = {quoted_user}")
        )


def ensure_database_exists(settings: Env) -> None:
    """Ensure the target database exists."""
    engine = create_engine(
        settings.postgres_root_connection, isolation_level="AUTOCOMMIT"
    )
    create_database(engine, settings.POSTGRES_USER, settings.db_name)

    engine.dispose()
