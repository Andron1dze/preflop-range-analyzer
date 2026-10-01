from pathlib import Path

from sqlalchemy import Engine, create_engine, event

# Сколько миллисекунд писатель ждёт освобождения блокировки, прежде чем упасть
# с "database is locked". Нужно, когда фоновая задача и UI пишут одновременно.
BUSY_TIMEOUT_MS = 5000


def make_engine(path: str | Path) -> Engine:
    """Engine для файловой SQLite-базы: WAL и внешние ключи на каждом соединении."""
    engine = create_engine(f"sqlite:///{Path(path).as_posix()}")
    event.listen(engine, "connect", _configure_connection)
    return engine


def _configure_connection(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    finally:
        cursor.close()
