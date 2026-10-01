"""Веб-интерфейс: загрузка HH, запуск анализа, список прогонов (spec §11).

    uvicorn web.main:app            # база — PREFLOP_DB или preflop.sqlite

Тяжёлая работа — фоновые задачи FastAPI (`BackgroundTasks`) со своей сессией;
страницы только читают готовое. Конкурентное чтение во время записи держит WAL.
"""

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from core.adapters.pokerstars import parse_file
from core.analysis.config import AnalysisConfig
from core.analysis.pipeline import run_analysis
from core.db import make_engine
from core.db.models import AnalysisRun, ChartSet
from core.db.persist import import_hands
from core.mapping.distance import MappingConfig
from core.mapping.remap import remap

WEB = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=WEB / "templates")


@dataclass
class ImportLogEntry:
    filename: str
    summary: str
    errors: list[str] = field(default_factory=list)


def import_job(engine: Engine, text: str, hero: str, filename: str, log: list[ImportLogEntry]) -> None:
    parsed = parse_file(text, hero_name=hero)
    with Session(engine) as session:
        report = import_hands(session, parsed.hands, source="pokerstars")
        session.commit()
    errors = [f"hand #{e.external_id}: {e.message}" for e in parsed.errors]
    errors += [f"hand #{external_id}: {message}" for external_id, message in report.errors]
    summary = f"imported {report.imported}, duplicates {report.duplicates}, errors {len(errors)}"
    log.insert(0, ImportLogEntry(filename, summary, errors))


def analysis_job(engine: Engine, chart_set_id: int) -> None:
    with Session(engine) as session:
        mapping = remap(session, chart_set_id, MappingConfig())
        session.commit()
        try:
            run_analysis(
                session, chart_set_id=chart_set_id, mapping_version=mapping.version, config=AnalysisConfig()
            )
        finally:
            # Упавший прогон тоже сохраняется — со статусом failed.
            session.commit()


def create_app(db_path: str | Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        app.state.engine.dispose()

    app = FastAPI(title="Preflop analyzer", lifespan=lifespan)
    app.state.engine = make_engine(db_path)
    app.state.import_log = []
    app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        with Session(app.state.engine) as session:
            chart_sets = session.scalars(select(ChartSet).order_by(ChartSet.id.desc())).all()
            return TEMPLATES.TemplateResponse(
                request,
                "index.html",
                {"chart_sets": chart_sets, "import_log": app.state.import_log},
            )

    @app.post("/hands", response_class=HTMLResponse, status_code=202)
    async def upload_hands(
        request: Request, background: BackgroundTasks, file: UploadFile, hero: str = Form(...)
    ):
        try:
            text = (await file.read()).decode("utf-8-sig")
        except UnicodeDecodeError:
            return _error(request, f"{file.filename}: file is not UTF-8 text", 400)
        background.add_task(import_job, app.state.engine, text, hero, file.filename, app.state.import_log)
        return TEMPLATES.TemplateResponse(
            request, "upload_started.html", {"filename": file.filename, "hero": hero}, status_code=202
        )

    @app.post("/runs", response_class=HTMLResponse, status_code=202)
    def start_run(request: Request, background: BackgroundTasks, chart_set_id: int = Form(...)):
        with Session(app.state.engine) as session:
            chart_set = session.get(ChartSet, chart_set_id)
            if chart_set is None:
                return _error(request, f"chart set {chart_set_id} not found", 404)
            if not chart_set.eligible_for_analysis:
                return _error(
                    request, f"chart set {chart_set.source}@{chart_set.version} is not eligible for analysis", 400
                )
            label = f"{chart_set.source}@{chart_set.version}"
        background.add_task(analysis_job, app.state.engine, chart_set_id)
        return TEMPLATES.TemplateResponse(request, "run_started.html", {"label": label}, status_code=202)

    @app.get("/runs", response_class=HTMLResponse)
    def runs(request: Request):
        with Session(app.state.engine) as session:
            rows = session.scalars(select(AnalysisRun).order_by(AnalysisRun.id.desc())).all()
            return TEMPLATES.TemplateResponse(request, "runs.html", {"runs": rows})

    return app


def _error(request: Request, message: str, status: int):
    return TEMPLATES.TemplateResponse(request, "error.html", {"message": message}, status_code=status)


app = create_app(os.environ.get("PREFLOP_DB", "preflop.sqlite"))
