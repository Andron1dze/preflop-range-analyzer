from pathlib import Path

import yaml
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from cli.main import main
from core.db import make_engine
from core.db.models import AnalysisRun, ChartSet, Hand
from tests.analysis.factories import chart_set_data

FIXTURES = Path(__file__).parents[1] / "adapters" / "fixtures"


def query(db, statement):
    engine = make_engine(db)
    try:
        with Session(engine) as session:
            return session.execute(statement).all()
    finally:
        engine.dispose()


def cli(db, *args):
    return main(["--db", str(db), *args])


def test_init_db_and_import(tmp_path, capsys):
    db = tmp_path / "cli.sqlite"
    assert cli(db, "init-db") == 0
    hh = tmp_path / "hands.txt"
    hh.write_text(
        "\n\n".join((FIXTURES / n).read_text(encoding="utf-8") for n in ("rfi_each_ante.txt", "threebet_jam_bb_ante.txt")),
        encoding="utf-8",
    )

    assert cli(db, "import", str(hh), "--hero", "Hero") == 0
    assert "imported 2, duplicates 0, errors 0" in capsys.readouterr().out
    assert cli(db, "import", str(hh), "--hero", "Hero") == 0
    assert "imported 0, duplicates 2, errors 0" in capsys.readouterr().out
    assert len(query(db, select(Hand.id))) == 2


def test_import_reports_parse_errors(tmp_path, capsys):
    db = tmp_path / "cli.sqlite"
    cli(db, "init-db")
    text = (FIXTURES / "rfi_each_ante.txt").read_text(encoding="utf-8")
    hh = tmp_path / "bad.txt"
    hh.write_text(text.replace("Hero: raises 120 to 220", "Hero: dances"), encoding="utf-8")
    assert cli(db, "import", str(hh), "--hero", "Hero") == 0
    out = capsys.readouterr().out
    assert "imported 0, duplicates 0, errors 1" in out
    assert "250000000001" in out


def test_load_charts(tmp_path, capsys):
    db = tmp_path / "cli.sqlite"
    cli(db, "init-db")
    charts = tmp_path / "charts.yaml"
    charts.write_text(yaml.safe_dump(chart_set_data()), encoding="utf-8")
    assert cli(db, "load-charts", str(charts)) == 0
    assert "not eligible for analysis" in capsys.readouterr().out
    assert query(db, select(ChartSet.source, ChartSet.eligible_for_analysis)) == [("synthetic-test", False)]


def test_invalid_chart_file_fails(tmp_path, capsys):
    db = tmp_path / "cli.sqlite"
    cli(db, "init-db")
    data = chart_set_data()
    del data["chart_set"]["ante"]
    charts = tmp_path / "charts.yaml"
    charts.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert cli(db, "load-charts", str(charts)) == 1
    assert "ante" in capsys.readouterr().err


def test_analyze_refuses_synthetic_charts(tmp_path, capsys):
    db = tmp_path / "cli.sqlite"
    cli(db, "init-db")
    charts = tmp_path / "charts.yaml"
    charts.write_text(yaml.safe_dump(chart_set_data()), encoding="utf-8")
    cli(db, "load-charts", str(charts))
    assert cli(db, "analyze", "--chart-set", "1", "--mapping-version", "m1") == 1
    assert "not eligible" in capsys.readouterr().err


def test_analyze_with_config(tmp_path, capsys):
    db = tmp_path / "cli.sqlite"
    cli(db, "init-db")
    charts = tmp_path / "charts.yaml"
    charts.write_text(yaml.safe_dump(chart_set_data()), encoding="utf-8")
    cli(db, "load-charts", str(charts))
    # Имитируем настоящий набор чартов: CLI анализирует только пригодные.
    engine = make_engine(db)
    with Session(engine) as session:
        session.execute(update(ChartSet).values(eligible_for_analysis=True))
        session.commit()
    engine.dispose()
    config = tmp_path / "analysis.yaml"
    config.write_text(yaml.safe_dump({"stats": {"z_threshold": 3.0}}), encoding="utf-8")

    assert cli(db, "analyze", "--chart-set", "1", "--mapping-version", "m1", "--config", str(config)) == 0
    assert "run 1: done" in capsys.readouterr().out
    [(status, params)] = query(db, select(AnalysisRun.status, AnalysisRun.params))
    assert status == "done"
    assert params["stats"]["z_threshold"] == 3.0


def test_remap(tmp_path, capsys):
    db = tmp_path / "cli.sqlite"
    cli(db, "init-db")
    hh = tmp_path / "hands.txt"
    hh.write_text(
        "\n\n".join((FIXTURES / n).read_text(encoding="utf-8") for n in ("rfi_each_ante.txt", "threebet_jam_bb_ante.txt")),
        encoding="utf-8",
    )
    cli(db, "import", str(hh), "--hero", "Hero")
    cli(db, "load-charts", str(Path(__file__).parents[2] / "charts" / "40bb_core.yaml"))
    capsys.readouterr()

    assert cli(db, "remap", "--chart-set", "1") == 0
    # Раздача 1: UTG и BTN открываются в узлы чарта; BB (25bb против 2.2bb) — w = 0.099 → далёкий.
    # Остальные 14 решений обеих раздач в узлы чарта не попадают.
    assert "grid-v1/cs1/sigma=0.223144/cutoff=0.1: mapped 2, far 1, unmapped 14" in capsys.readouterr().out
