from sqlalchemy import Engine, inspect, text

from app.utils.logger import get_logger


logger = get_logger(__name__)


DOCUMENT_CHUNK_METADATA_COLUMNS = {
    "search_text": "TEXT",
    "summary": "TEXT",
    "keywords": "JSON",
    "hypothetical_questions": "JSON",
    "structure_types": "JSON",
}


def ensure_document_chunk_metadata_columns(engine: Engine) -> None:
    """
    Add nullable metadata columns introduced after the initial create_all schema.

    This app does not currently use Alembic, and SQLAlchemy create_all will not
    alter tables that already exist. Keep this migration additive and idempotent
    so existing local Docker databases can accept newly enriched chunks.
    """
    inspector = inspect(engine)

    if not inspector.has_table("document_chunks"):
        return

    existing_columns = {
        column["name"]
        for column in inspector.get_columns("document_chunks")
    }
    missing_columns = {
        name: column_type
        for name, column_type in DOCUMENT_CHUNK_METADATA_COLUMNS.items()
        if name not in existing_columns
    }

    with engine.begin() as connection:
        # Multiple Uvicorn workers execute lifespan concurrently. Serialize the
        # additive migration so PostgreSQL does not race while creating the same
        # catalog object despite CREATE INDEX IF NOT EXISTS.
        if engine.dialect.name == "postgresql":
            connection.execute(
                text("SELECT pg_advisory_xact_lock(724194031)")
            )
        if missing_columns:
            logger.info(
                "Applying document chunk metadata schema update: columns=%s",
                sorted(missing_columns),
            )
        for column_name, column_type in missing_columns.items():
            connection.execute(
                text(
                    f"ALTER TABLE document_chunks "
                    f"ADD COLUMN {column_name} {column_type}"
                )
            )
        if engine.dialect.name == "postgresql":
            connection.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_document_chunks_search_fts "
                    "ON document_chunks USING GIN "
                    "(to_tsvector('english', coalesce(search_text, chunk_text)))"
                )
            )
