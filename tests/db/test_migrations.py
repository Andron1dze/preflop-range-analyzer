from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

from core.db import Base, make_engine

ROOT = Path(__file__).resolve().parents[2]


def _schema(engine):
    insp = inspect(engine)
    return {
        table: {
            "columns": {
                c["name"]: (str(c["type"]), c["nullable"]) for c in insp.get_columns(table)
            },
            "pk": tuple(insp.get_pk_constraint(table)["constrained_columns"]),
            "fks": sorted(
                (tuple(fk["constrained_columns"]), fk["referred_table"],
                 fk["options"].get("ondelete"))
                for fk in insp.get_foreign_keys(table)
            ),
            "uniques": sorted(
                (u["name"], tuple(u["column_names"])) for u in insp.get_unique_constraints(table)
            ),
            "checks": sorted(
                (c["name"], " ".join(c["sqltext"].split()))
                for c in insp.get_check_constraints(table)
            ),
            "indexes": sorted(
                (i["name"], tuple(i["column_names"])) for i in insp.get_indexes(table)
            ),
        }
        for table in insp.get_table_names()
        if table != "alembic_version"
    }


def test_upgrade_head_matches_models(tmp_path):
    migrated = tmp_path / "migrated.sqlite"
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{migrated.as_posix()}")
    command.upgrade(cfg, "head")

    reference = make_engine(tmp_path / "reference.sqlite")
    Base.metadata.create_all(reference)
    migrated_engine = make_engine(migrated)
    try:
        assert _schema(migrated_engine) == _schema(reference)
    finally:
        reference.dispose()
        migrated_engine.dispose()
