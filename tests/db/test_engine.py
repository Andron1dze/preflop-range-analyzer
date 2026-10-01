from sqlalchemy import text

from core.db import make_engine


def test_new_connection_uses_wal(db_path):
    engine = make_engine(db_path)
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
    engine.dispose()


def test_new_connection_enforces_foreign_keys(db_path):
    engine = make_engine(db_path)
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1
    engine.dispose()


def test_reader_is_not_blocked_by_open_write_transaction(db_path):
    engine = make_engine(db_path)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE t (x INTEGER)"))
        conn.execute(text("INSERT INTO t VALUES (1)"))

    writer = engine.connect()
    writer.begin()
    writer.execute(text("INSERT INTO t VALUES (2)"))
    try:
        with engine.connect() as reader:
            # WAL: читатель видит последний закоммиченный снимок и не ждёт писателя.
            assert reader.execute(text("SELECT count(*) FROM t")).scalar() == 1
    finally:
        writer.rollback()
        writer.close()
        engine.dispose()
