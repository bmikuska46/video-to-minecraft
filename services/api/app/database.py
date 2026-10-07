from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import Settings


class Base(DeclarativeBase):
    pass


def make_engine(settings: Settings):
    connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
    return create_engine(settings.database_url, pool_pre_ping=True, connect_args=connect_args)


def make_session_factory(engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def session_dependency(factory: sessionmaker[Session]):
    def get_session() -> Iterator[Session]:
        with factory() as session:
            yield session

    return get_session



def add_missing_columns(engine) -> list[str]:
    """Add columns that ``create_all`` cannot add to tables that already exist.

    Only columns with a server default are added, so existing rows stay valid.
    Returns the ``table.column`` names that were added.
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    added = []
    with engine.begin() as connection:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            present = {column["name"] for column in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in present:
                    continue
                if column.server_default is None:
                    raise RuntimeError(f"cannot add {table.name}.{column.name} automatically: no server default")
                column_type = column.type.compile(dialect=engine.dialect)
                default = column.server_default.arg
                connection.execute(text(
                    f"ALTER TABLE {table.name} ADD COLUMN {column.name} {column_type} DEFAULT '{default}'"
                ))
                added.append(f"{table.name}.{column.name}")
    return added
