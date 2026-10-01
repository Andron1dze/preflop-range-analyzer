import pytest
from sqlalchemy.orm import Session

from core.db import Base, make_engine


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.sqlite"


@pytest.fixture
def engine(db_path):
    engine = make_engine(db_path)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine):
    with Session(engine) as session:
        yield session
