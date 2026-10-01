"""Веб-интерфейс: загрузка HH, запуск анализа, список прогонов (spec §11).

    uvicorn web.main:app            # база — PREFLOP_DB или preflop.sqlite

Тяжёлая работа — фоновые задачи FastAPI (`BackgroundTasks`) со своей сессией;
страницы только читают готовое. Конкурентное чтение во время записи держит WAL.
"""

import json
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

from core.adapters.manual import ManualHandError, parse_manual_text
from core.adapters.pokerstars import parse_file
from core.analysis.config import AnalysisConfig
from core.analysis.pipeline import run_analysis
from core.analysis.report import grid_layout, hand_deviations, load_report
from core.db import make_engine
from core.db.models import AnalysisRun, ChartSet
from core.db.persist import import_hands
from core.mapping.distance import MappingConfig
from core.mapping.remap import remap

WEB = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=WEB / "templates")

# Пример для редактора ручного ввода: BTN открывается на 2.2bb, блайнды сбрасывают.
MANUAL_EXAMPLE = json.dumps(
    {
        "id": "live-001",
        "tournament": "Live Main Event",
        "level": 12,
        "blinds": {"sb": 500, "bb": 1000},
        "ante": {"type": "bb", "amount": 1000},
        "button": 6,
        "seats": [
            {"seat": n, "player": name, "stack": 40000}
            for n, name in enumerate(["UTG", "UTG1", "LJ", "HJ", "CO", "Hero", "SB", "BB"], start=1)
        ],
        "hero": "Hero",
        "cards": "AhKh",
        "actions": [
            *({"player": p, "action": "fold"} for p in ["UTG", "UTG1", "LJ", "HJ", "CO"]),
            {"player": "Hero", "action": "raise", "to": 2200},
            {"player": "SB", "action": "fold"},
            {"player": "BB", "action": "fold"},
        ],
        "stage": "mid",
    },
    indent=2,
    ensure_ascii=False,
)


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
                {"chart_sets": chart_sets, "import_log": app.state.import_log, "manual_example": MANUAL_EXAMPLE},
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

    @app.post("/manual", response_class=HTMLResponse, status_code=201)
    def save_manual(request: Request, hand: str = Form(...)):
        try:
            parsed = parse_manual_text(hand)
        except ManualHandError as error:
            return TEMPLATES.TemplateResponse(request, "manual_result.html", {"errors": error.errors}, status_code=400)
        with Session(app.state.engine) as session:
            report = import_hands(session, [parsed], source="manual")
            session.commit()
        label = parsed.hand.external_id or "без номера"
        if report.duplicates:
            return _error(request, f"hand {label} is already imported", 409)
        if report.errors:
            return TEMPLATES.TemplateResponse(
                request, "manual_result.html", {"errors": [m for _, m in report.errors]}, status_code=400
            )
        return TEMPLATES.TemplateResponse(
            request,
            "manual_result.html",
            {"saved": label, "decisions": len(parsed.hand.actions), "stage": parsed.hand.stage},
            status_code=201,
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

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def report(request: Request, run_id: int):
        with Session(app.state.engine) as session:
            data = load_report(session, run_id)
            if data is None:
                return _error(request, f"run {run_id} not found", 404, page=True)
            return TEMPLATES.TemplateResponse(request, "report.html", {"report": data})

    @app.get("/runs/{run_id}/grid", response_class=HTMLResponse)
    def grid(request: Request, run_id: int, node_id: int, action: str):
        with Session(app.state.engine) as session:
            cells = hand_deviations(session, run_id, node_id, action)
        if cells is None:
            return _error(request, f"no data for run {run_id}, node {node_id}, action {action!r}", 404)
        rows = [[_grid_cell(hand, cells[hand]) for hand in row] for row in grid_layout()]
        return TEMPLATES.TemplateResponse(request, "grid.html", {"rows": rows, "action": action})

    return app


def _grid_cell(hand: str, cell) -> dict:
    if not cell.n:
        css, strength = "empty", 0
    else:
        css = "over" if cell.deviation > 0 else "under" if cell.deviation < 0 else "even"
        # Отклонение на 0.5 частоты и больше — максимальная насыщенность.
        strength = round(min(1.0, abs(cell.deviation) * 2) * 100)
    title = f"{hand}: n={cell.n}, факт {cell.observed:.1f}, GTO {cell.expected:.1f}"
    return {"hand": hand, "css": css, "strength": strength, "title": title}


def _error(request: Request, message: str, status: int, *, page: bool = False):
    template = "error_page.html" if page else "error.html"
    return TEMPLATES.TemplateResponse(request, template, {"message": message}, status_code=status)


app = create_app(os.environ.get("PREFLOP_DB", "preflop.sqlite"))
