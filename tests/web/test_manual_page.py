import html
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.db import Base, make_engine
from core.db.models import Decision, Hand
from web.main import create_app

FIXTURES = Path(__file__).parents[1] / "adapters" / "fixtures"
RFI_JSON = (FIXTURES / "rfi_each_ante.json").read_text(encoding="utf-8")


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "manual.sqlite"
    engine = make_engine(path)
    Base.metadata.create_all(engine)
    engine.dispose()
    return path


@pytest.fixture
def client(db):
    with TestClient(create_app(db)) as client:
        yield client


def rows(db, statement):
    engine = make_engine(db)
    try:
        with Session(engine) as session:
            return session.execute(statement).all()
    finally:
        engine.dispose()


def test_index_has_manual_editor(client):
    page = client.get("/").text
    assert 'hx-post="/manual"' in page
    assert '<textarea name="hand"' in page
    # В редакторе — пример с обязательной стадией (кавычки в textarea экранированы).
    assert '"stage": "mid"' in html.unescape(page)


def test_manual_hand_is_saved_with_stage(client, db):
    response = client.post("/manual", data={"hand": RFI_JSON})
    assert response.status_code == 201
    assert "250000000001" in response.text and "8" in response.text  # номер раздачи и число решений
    assert rows(db, select(Hand.source, Hand.external_id)) == [("manual", "250000000001")]
    stages = set(rows(db, select(Decision.stage, Decision.stage_source)))
    assert stages == {("mid", "manual")}


def test_missing_stage_is_rejected(client, db):
    response = client.post("/manual", data={"hand": RFI_JSON.replace('"stage": "mid"', '"note": "x"')})
    assert response.status_code == 400
    assert "stage" in response.text
    assert rows(db, select(Hand.id)) == []


def test_invalid_json_is_rejected(client):
    response = client.post("/manual", data={"hand": "{oops"})
    assert response.status_code == 400


def test_same_hand_twice_is_a_conflict(client, db):
    assert client.post("/manual", data={"hand": RFI_JSON}).status_code == 201
    response = client.post("/manual", data={"hand": RFI_JSON})
    assert response.status_code == 409
    assert "already" in response.text
    assert len(rows(db, select(Hand.id))) == 1


def test_hands_without_id_are_not_deduplicated(client, db):
    text = RFI_JSON.replace('"id": "250000000001",', "")
    assert client.post("/manual", data={"hand": text}).status_code == 201
    assert client.post("/manual", data={"hand": text}).status_code == 201
    assert len(rows(db, select(Hand.id))) == 2


def test_editor_example_is_a_valid_hand(client, db):
    from web.main import MANUAL_EXAMPLE

    assert client.post("/manual", data={"hand": MANUAL_EXAMPLE}).status_code == 201
