import threading
from pathlib import Path

import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text, update
from sqlalchemy.orm import Session

from core.charts.format import load_chart_file
from core.charts.store import save_chart_set
from core.db import Base, make_engine
from core.db.models import AnalysisRun, ChartSet, DecisionNodeMap, Hand
import web.main as web_main
from web.main import create_app

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests" / "adapters" / "fixtures"
HH = "\n\n".join(
    (FIXTURES / name).read_text(encoding="utf-8") for name in ("rfi_each_ante.txt", "threebet_jam_bb_ante.txt")
)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "web.sqlite"
    engine = make_engine(path)
    Base.metadata.create_all(engine)
    engine.dispose()
    return path


@pytest.fixture
def client(db):
    with TestClient(create_app(db)) as client:
        yield client


def count(db, model):
    engine = make_engine(db)
    try:
        with Session(engine) as session:
            return session.scalar(select(func.count()).select_from(model))
    finally:
        engine.dispose()


def load_charts(db, eligible):
    engine = make_engine(db)
    try:
        with Session(engine) as session:
            chart_set = save_chart_set(session, load_chart_file(ROOT / "charts" / "40bb_core.yaml"))
            if eligible:
                # Имитируем настоящий набор: в анализ пускаются только пригодные чарты.
                session.execute(update(ChartSet).values(eligible_for_analysis=True))
            session.commit()
            return chart_set.id
    finally:
        engine.dispose()


def upload(client, content=HH, hero="Hero", name="hands.txt"):
    return client.post("/hands", files={"file": (name, content.encode("utf-8"), "text/plain")}, data={"hero": hero})


# --- главная -----------------------------------------------------------------


def test_index_has_upload_and_analysis_forms(client, db):
    load_charts(db, eligible=True)
    page = client.get("/")
    assert page.status_code == 200
    assert 'name="file"' in page.text and 'name="hero"' in page.text
    assert 'name="chart_set_id"' in page.text
    assert "synthetic-test@40bb-core-approx-1" in page.text
    assert 'hx-get="/runs"' in page.text


# --- загрузка HH -------------------------------------------------------------


def test_upload_imports_hands(client, db):
    response = upload(client)
    assert response.status_code == 202
    assert "hands.txt" in response.text
    assert count(db, Hand) == 2


@pytest.fixture
def scheduled(monkeypatch):
    """Задачи, поставленные через BackgroundTasks.add_task (сами задачи не выполняются)."""
    tasks = []
    monkeypatch.setattr(BackgroundTasks, "add_task", lambda self, func, *args, **kwargs: tasks.append((func, args)))
    return tasks


def test_upload_runs_import_as_background_task(client, db, scheduled):
    response = upload(client)
    assert response.status_code == 202
    # Обработчик запроса только ставит задачу; импортирует фоновая задача.
    assert count(db, Hand) == 0
    [(func, (engine, text_, hero, filename, log))] = scheduled
    assert func is web_main.import_job
    assert (text_, hero, filename) == (HH, "Hero", "hands.txt")


def test_import_report_is_shown_on_index(client):
    upload(client)
    upload(client)
    page = client.get("/").text
    assert "hands.txt: imported 2, duplicates 0, errors 0" in page
    assert "hands.txt: imported 0, duplicates 2, errors 0" in page


def test_import_errors_are_reported(client):
    upload(client, content=HH.replace("Hero: raises 120 to 220", "Hero: dances"))
    page = client.get("/").text
    assert "imported 1, duplicates 0, errors 1" in page
    assert "250000000001" in page


def test_upload_accepts_bom(client, db):
    response = client.post(
        "/hands", files={"file": ("bom.txt", b"\xef\xbb\xbf" + HH.encode("utf-8"), "text/plain")}, data={"hero": "Hero"}
    )
    assert response.status_code == 202
    assert count(db, Hand) == 2


def test_upload_requires_hero(client):
    response = client.post("/hands", files={"file": ("hands.txt", HH.encode(), "text/plain")})
    assert response.status_code == 422


def test_upload_rejects_non_utf8(client):
    response = client.post("/hands", files={"file": ("bad.txt", b"\xff\xfe\x00bad", "text/plain")}, data={"hero": "Hero"})
    assert response.status_code == 400


# --- запуск анализа ----------------------------------------------------------


def test_run_starts_remap_and_analysis(client, db):
    chart_set_id = load_charts(db, eligible=True)
    upload(client)
    response = client.post("/runs", data={"chart_set_id": chart_set_id})
    assert response.status_code == 202
    assert count(db, DecisionNodeMap) > 0

    engine = make_engine(db)
    with Session(engine) as session:
        [(status, version)] = session.execute(select(AnalysisRun.status, AnalysisRun.mapping_version)).all()
    engine.dispose()
    assert status == "done"
    assert version == f"grid-v1/cs{chart_set_id}/sigma=0.223144/cutoff=0.1"


def test_run_is_a_background_task(client, db, scheduled):
    chart_set_id = load_charts(db, eligible=True)
    assert client.post("/runs", data={"chart_set_id": chart_set_id}).status_code == 202
    assert count(db, AnalysisRun) == 0
    [(func, (engine, scheduled_id))] = scheduled
    assert func is web_main.analysis_job
    assert scheduled_id == chart_set_id


def test_ineligible_chart_set_is_rejected_immediately(client, db):
    chart_set_id = load_charts(db, eligible=False)
    response = client.post("/runs", data={"chart_set_id": chart_set_id})
    assert response.status_code == 400
    assert "not eligible" in response.text
    assert count(db, AnalysisRun) == 0


def test_unknown_chart_set_is_rejected(client):
    response = client.post("/runs", data={"chart_set_id": 999})
    assert response.status_code == 404


# --- список прогонов ---------------------------------------------------------


def add_run(db, **fields):
    engine = make_engine(db)
    with Session(engine) as session:
        session.add(AnalysisRun(**{"mapping_version": "m1", "chart_set_version": "src@1", "params": {}, **fields}))
        session.commit()
    engine.dispose()


def test_runs_fragment_lists_runs_newest_first(client, db):
    add_run(db, status="done", chart_set_version="hrc@1")
    add_run(db, status="failed", chart_set_version="hrc@2")
    page = client.get("/runs")
    assert page.status_code == 200
    assert "<html" not in page.text  # фрагмент для HTMX, не страница
    assert page.text.index("hrc@2") < page.text.index("hrc@1")
    assert "done" in page.text and "failed" in page.text


def test_runs_fragment_when_empty(client):
    assert "No analysis runs yet" in client.get("/runs").text


def test_runs_are_readable_while_background_write_is_open(client, db):
    add_run(db, status="done", chart_set_version="committed@1")
    engine = make_engine(db)
    writer = engine.connect()
    writer.begin()
    writer.execute(text(
        "INSERT INTO analysis_runs (started_at, mapping_version, chart_set_version, params, status) "
        "VALUES ('2026-10-01', 'm', 'uncommitted@1', '{}', 'running')"
    ))
    result = {}
    reader = threading.Thread(target=lambda: result.update(response=client.get("/runs")))
    try:
        reader.start()
        reader.join(timeout=3)
        # WAL: чтение не ждёт открытую транзакцию записи и видит последний коммит.
        assert not reader.is_alive()
        assert "committed@1" in result["response"].text
        assert "uncommitted@1" not in result["response"].text
    finally:
        writer.rollback()
        writer.close()
        engine.dispose()


def test_htmx_is_served_locally_not_from_cdn(client):
    page = client.get("/").text
    assert 'src="/static/htmx.min.js"' in page
    assert "unpkg.com" not in page
    script = client.get("/static/htmx.min.js")
    assert script.status_code == 200 and script.text.startswith("var htmx=")
