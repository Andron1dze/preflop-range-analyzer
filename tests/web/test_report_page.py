import re

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.analysis.config import AnalysisConfig
from core.analysis.pipeline import run_analysis
from core.db import Base, make_engine
from core.db.models import Node
from tests.analysis.factories import RFI, RFI_KEY, VS_OPEN_KEY, add_decisions, load_chart_set, sample_actions
from web.main import create_app

OVERFOLD = {"fold": 0.7, "call": 0.2, "raise": 0.1}


@pytest.fixture
def db_with_run(tmp_path):
    path = tmp_path / "report.sqlite"
    engine = make_engine(path)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        chart_set, nodes = load_chart_set(session)
        add_decisions(
            session, nodes[VS_OPEN_KEY], sample_actions(OVERFOLD, 600, seed=1),
            position="BTN", line="UTG:F,UTG1:F,LJ:F,HJ:F,CO:R2.2", seed=1,
        )
        add_decisions(
            session, nodes[RFI_KEY], sample_actions(RFI, 400, seed=2),
            position="CO", line="UTG:F,UTG1:F,LJ:F,HJ:F", seed=2, stage=None,
        )
        run = run_analysis(
            session, chart_set_id=chart_set.id, mapping_version="m1", config=AnalysisConfig(), allow_synthetic=True
        )
        session.commit()
        ids = {"run": run.id, "vs_open": nodes[VS_OPEN_KEY].id, "rfi": nodes[RFI_KEY].id}
    engine.dispose()
    return path, ids


@pytest.fixture
def client(db_with_run):
    path, _ = db_with_run
    with TestClient(create_app(path)) as client:
        yield client


@pytest.fixture
def ids(db_with_run):
    return db_with_run[1]


def test_unknown_run_is_404(client):
    assert client.get("/runs/999").status_code == 404


def test_report_page_has_run_meta_and_node_summary(client, ids):
    page = client.get(f"/runs/{ids['run']}")
    assert page.status_code == 200
    assert "<html" in page.text
    assert "synthetic-test@1" in page.text and "m1" in page.text and "done" in page.text
    assert VS_OPEN_KEY in page.text and RFI_KEY in page.text
    for header in ("z", "n", "BH-q", "Неизв. стадия"):
        assert f">{header}<" in page.text
    assert ">600<" in page.text  # n по узлу против открытия
    assert ">100%<" in page.text  # у RFI-решений стадия неизвестна


def test_leak_rows_are_marked(client, ids):
    page = client.get(f"/runs/{ids['run']}").text
    leak_rows = re.findall(r'<tr class="node-row leak"[^>]*data-node="(\d+)"', page)
    assert str(ids["vs_open"]) in leak_rows
    assert str(ids["rfi"]) not in leak_rows


def test_node_expands_to_zones_and_groups(client, ids):
    page = client.get(f"/runs/{ids['run']}").text
    assert page.count("<details") >= 5  # по строке на узел и действие
    assert "mix" in page and "suited_connector" in page


def test_coefficient_table_with_intervals(client, ids):
    page = client.get(f"/runs/{ids['run']}").text
    assert "continue_vs_fold" in page and "aggr_vs_call" in page
    assert "intercept" in page and "position=BTN" in page
    assert re.search(r"\[-?\d+\.\d{2}, -?\d+\.\d{2}\]", page)  # доверительный интервал
    assert "частично ожидаемо из-за ICM" in page  # есть признак stage=mid


def test_grid_link_per_node_action(client, ids):
    page = client.get(f"/runs/{ids['run']}").text
    assert f'hx-get="/runs/{ids["run"]}/grid?node_id={ids["vs_open"]}&amp;action=fold"' in page


def test_grid_fragment(client, ids):
    grid = client.get(f"/runs/{ids['run']}/grid", params={"node_id": ids["vs_open"], "action": "fold"})
    assert grid.status_code == 200
    assert "<html" not in grid.text
    cells = re.findall(r'<td class="cell[^"]*"[^>]*>(\w+)</td>', grid.text)
    assert len(cells) == 169
    assert cells[:3] == ["AA", "AKs", "AQs"] and cells[13] == "AKo" and cells[-1] == "22"
    assert grid.text.count("cell over") > grid.text.count("cell under")  # оверфолд
    assert re.search(r'title="[^"]*n=\d+', grid.text)


def test_grid_for_unknown_node_or_action(client, ids):
    assert client.get(f"/runs/{ids['run']}/grid", params={"node_id": 999, "action": "fold"}).status_code == 404
    assert client.get(f"/runs/999/grid", params={"node_id": ids["vs_open"], "action": "fold"}).status_code == 404


def test_runs_list_links_to_reports(client, ids):
    assert f'href="/runs/{ids["run"]}"' in client.get("/runs").text
