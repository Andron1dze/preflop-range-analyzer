"""preflop — командная строка анализатора.

    python -m cli.main --db preflop.sqlite init-db
    python -m cli.main --db preflop.sqlite import hands.txt --hero Hero
    python -m cli.main --db preflop.sqlite load-charts charts/btn_rfi.yaml
    python -m cli.main --db preflop.sqlite analyze --chart-set 1 --mapping-version v1 [--config analysis.yaml]

Команды remap пока нет: маппинг решений на узлы ждёт сетку узлов (тикеты 06, 15).
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.orm import Session

from core.adapters.pokerstars import parse_file
from core.analysis.config import AnalysisConfig, load_config
from core.analysis.pipeline import run_analysis
from core.charts.format import ChartValidationError, load_chart_file
from core.charts.store import save_chart_set
from core.db import make_engine
from core.db.persist import import_hands

ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init-db":
        return _init_db(args.db)
    engine = make_engine(args.db)
    try:
        with Session(engine) as session:
            return args.handler(session, args)
    finally:
        engine.dispose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="preflop", description="Анализатор префлоп-диапазонов")
    parser.add_argument("--db", default="preflop.sqlite", help="файл базы SQLite")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("init-db", help="создать или обновить схему базы")

    imp = commands.add_parser("import", help="импортировать HH PokerStars")
    imp.add_argument("file", type=Path)
    imp.add_argument("--hero", required=True, help="ник героя в истории раздач")
    imp.set_defaults(handler=_import)

    charts = commands.add_parser("load-charts", help="загрузить набор чартов (JSON/YAML)")
    charts.add_argument("file", type=Path)
    charts.set_defaults(handler=_load_charts)

    analyze = commands.add_parser("analyze", help="прогнать анализ")
    analyze.add_argument("--chart-set", type=int, required=True)
    analyze.add_argument("--mapping-version", required=True)
    analyze.add_argument("--config", type=Path, help="YAML с параметрами анализа")
    analyze.set_defaults(handler=_analyze)
    return parser


def _init_db(db: str) -> int:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{Path(db).as_posix()}")
    command.upgrade(config, "head")
    print(f"database {db} is up to date")
    return 0


def _import(session: Session, args) -> int:
    parsed = parse_file(args.file.read_text(encoding="utf-8"), hero_name=args.hero)
    report = import_hands(session, parsed.hands, source="pokerstars")
    session.commit()
    errors = [(e.external_id, e.message) for e in parsed.errors] + report.errors
    print(f"imported {report.imported}, duplicates {report.duplicates}, errors {len(errors)}")
    for external_id, message in errors:
        print(f"  hand #{external_id}: {message}")
    return 0


def _load_charts(session: Session, args) -> int:
    try:
        chart = load_chart_file(args.file)
        row = save_chart_set(session, chart)
    except (ChartValidationError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    session.commit()
    note = "" if row.eligible_for_analysis else ", not eligible for analysis"
    print(f"chart set {row.id}: {chart.source}@{chart.version}, {len(chart.nodes)} nodes{note}")
    return 0


def _analyze(session: Session, args) -> int:
    config = load_config(args.config) if args.config else AnalysisConfig()
    try:
        run = run_analysis(
            session, chart_set_id=args.chart_set, mapping_version=args.mapping_version, config=config
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    finally:
        session.commit()
    print(f"run {run.id}: {run.status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
