# 01 — Каркас проекта, модели SQLAlchemy, SQLite WAL

Status: resolved
Type: task
Spec: ../spec.md §2, §3, §10

## Что сделать
- `pyproject.toml`, пакет `core/` (подпакеты из §3 как пустые модули), `tests/`.
- `core/db/`: фабрика engine для SQLite с `journal_mode=WAL` и `foreign_keys=ON` на каждом соединении.
- Модели SQLAlchemy 2.0 (типизированные `Mapped[...]`) для всех таблиц §10.
- Alembic: инициализация и первая миграция, повторяющая модели.

## Критерии приёмки
- Тест: новое соединение к файловой БД возвращает `journal_mode = wal` и `foreign_keys = 1`.
- Тест: `create_all` создаёт все таблицы §10 с ключевыми полями.
- Тест: `decisions.ev_loss`, `stage`, `stage_source` допускают NULL; `actor` обязателен.
- Тест: `chart_sets.model` принимает `chipev` и `icm:<профиль>`, отвергает прочее.
- Тест: `alembic upgrade head` на пустой БД даёт ту же схему, что `create_all`.

## Comments

**2026-10-01 — реализовано (TDD).** 33 теста зелёные (`.venv/Scripts/python -m pytest`).

- `core/db/engine.py`: `make_engine(path)` — на каждом соединении `journal_mode=WAL`, `foreign_keys=ON`, `busy_timeout=5000`.
- `core/db/models.py`: все таблицы §10. Сверх спеки:
  - `hands.external_id` + уникальность `(source, external_id)` — под критерий «без дубликатов» тикета 03;
  - CHECK на `hands.source`, `decisions.stage`, `decisions.stage_source`, `analysis_runs.status`, `node_strategies.freq ∈ [0, 1]`;
  - `decisions.hole_cards` / `hand_class` допускают NULL (карты оппонентов неизвестны);
  - уникальность узла `(chart_set_id, key, stack_bucket, sizing_bucket)` и набора чартов `(source, version)`.
- `chart_sets.model`: `model = 'chipev' OR model GLOB 'icm:?*'` (регистр важен, профиль обязателен).
- Alembic: `migrations/`, `render_as_batch=True`, явная naming convention ограничений. Тест сравнивает схему после `upgrade head` с `create_all`: колонки, PK, FK, UNIQUE, CHECK, индексы. Проверен мутацией (удалённый CHECK ловится).
