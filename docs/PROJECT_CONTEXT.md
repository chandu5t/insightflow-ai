# InsightFlow AI – Project Context

_Paste this file at the start of any new Claude conversation._

## Current status

- **Completed modules:** 1 (Foundation and health check)
- **In progress:** Module 2 (Data ingestion). Code delivered and tested by Claude (137 backend tests, frontend build and lint pass). **Waiting for the developer to run and confirm it.**
- **Next module:** 3 (Profiling, column mapping, analysis tools)
- **Git tag:** `module-1` (add `module-2` after confirmation)
- **Repository:** https://github.com/chandu5t/insightflow-ai (public, so never commit secrets)

## Environment

- Windows 11, VS Code, PowerShell
- Project root: `C:\dev\insightflow-ai`
- Python 3.12, FastAPI 0.141.1, React 19, TypeScript 6, Vite 8 (fill in exact versions from your machine)

## Working agreements

- "TDD" means the **Technical Design Document**. Also use test-driven development where practical.
- One module at a time. The developer confirms a module works before the next begins.
- Each module guide is delivered in a single response (free Claude plan).
- Windows PowerShell commands first. Frontend stays TypeScript (`.tsx` / `.ts`).
- No architecture changes without discussion. Nothing is committed or pushed automatically.
- Never commit `.env` files, API keys, `.venv/`, `node_modules/` or uploaded datasets.

## Module status

| # | Module | Status |
|---|---|---|
| 1 | Foundation and health check | Complete (tag `module-1`) |
| 2 | Data ingestion: CSV/.xlsx upload, validation, preview | Implemented, awaiting developer confirmation |
| 3 | Profiling, column mapping, analysis tools | Planned |
| 4 | Question understanding with Gemini | Planned |
| 5 | LangGraph workflow and result validation | Planned |
| 6 | PostgreSQL integration (+ Docker Compose) | Planned |
| 7 | Basic RAG with pgvector | Planned |
| 8 | Hardening, full Docker, documentation, release | Planned |

## Folder structure (end of Module 2)

```
insightflow-ai/
├── docs/                            PROJECT_CONTEXT.md, DECISIONS.md
├── backend/
│   ├── app/
│   │   ├── main.py                  app, CORS, error handlers, routers
│   │   ├── api/                     health_routes.py, dataset_routes.py
│   │   ├── core/                    config.py (limits), errors.py, logging_config.py
│   │   ├── schemas/                 health_schema.py, error_schema.py, dataset_schema.py
│   │   ├── services/                file_parser.py, dataset_repository.py, dataset_service.py
│   │   └── utils/                   upload_validator.py, table_validator.py
│   ├── tests/                       conftest.py, helpers.py, test_health.py, test_upload_validator.py,
│   │                                test_file_parser.py, test_dataset_repository.py,
│   │                                test_datasets_api.py, test_errors.py
│   ├── data/samples/, data/uploads/ (uploads are git-ignored)
│   └── pytest.ini, requirements.txt, .env.example
└── frontend/src/
    ├── services/api.ts              request(), ApiError, getHealth(), uploadDataset(), getDatasetPreview()
    ├── types/                       api.ts, dataset.ts
    ├── components/                  HealthStatus, FileUpload, DatasetPreview, ErrorMessage (.tsx)
    ├── pages/Dashboard.tsx
    ├── styles/dataset.css
    └── App.tsx, main.tsx, App.css, index.css
```

## API endpoints

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | `{"status","app","version","environment"}` |
| POST | `/datasets/upload` | multipart field `file` (.csv or .xlsx). 201 returns `DatasetSummary`: dataset_id, filename, source_format, row_count, column_count, column_names, uploaded_at, status ("uploaded"), warnings |
| GET | `/datasets/{dataset_id}/preview?rows=N` | Default 5, maximum 20. Returns `DatasetPreview`: metadata + column_names + preview_rows (empty cell = null, long cells cut to 100 chars) + cells_truncated + max_cell_length |

Errors: `{"error","code","details"}`. Codes: UNSUPPORTED_FILE_TYPE, FILE_TOO_LARGE (413), EMPTY_FILE, INVALID_ENCODING, CORRUPTED_FILE, MISSING_HEADERS, DUPLICATE_COLUMNS, NO_DATA_ROWS, TOO_MANY_ROWS, TOO_MANY_COLUMNS, MALFORMED_ROWS, INVALID_DATASET_ID (422), DATASET_NOT_FOUND (404), INVALID_PREVIEW_ROWS (422), VALIDATION_ERROR (422), NOT_FOUND, METHOD_NOT_ALLOWED, HTTP_ERROR, INTERNAL_ERROR (500).

## Environment variables

- `backend/.env`: `ENVIRONMENT`, `LOG_LEVEL`, `CORS_ORIGINS`, and optional `UPLOAD_DIR`, `MAX_UPLOAD_SIZE_MB` (10), `MAX_XLSX_UNCOMPRESSED_MB` (100), `MAX_ROWS` (100000), `MAX_COLUMNS` (100), `PREVIEW_DEFAULT_ROWS` (5), `PREVIEW_MAX_ROWS` (20), `PREVIEW_MAX_CELL_LENGTH` (100)
- `frontend/.env`: `VITE_API_URL`

## Data storage (until Module 6)

`data/uploads/{dataset_id}.csv` (clean UTF-8 CSV, all values as text) + `{dataset_id}.meta.json`. Accessed only through `DatasetRepository` (`JsonDatasetRepository` now, PostgreSQL later). The path is derived from the UUID and never stored or returned. Module 3 reads the clean CSV with Pandas.

## Dependencies

- Backend: fastapi, uvicorn[standard], pydantic-settings, pytest, httpx, **python-multipart, openpyxl** (added in Module 2). Exact versions in `backend/requirements.txt`.
- Frontend: react, react-dom, vite, typescript (no new packages in Module 2).

## Key decisions

See `docs/DECISIONS.md` (D-001 to D-028). Most relevant now: D-019 TypeScript, D-020 CSV + first-sheet XLSX, D-021 capped preview never sent to an LLM, D-022 no Pandas until Module 3, D-023 error format, D-024 storage layout, D-028 Docker deferred to Module 6.

## Testing status

- Backend: 137 tests (7 from Module 1, 130 from Module 2) passed in Claude's environment. Developer confirmation pending.
- Frontend: `npm run build` and `npm run lint` pass. No frontend unit tests.

## Known limitations

- The framework receives the whole upload before our size check runs (see README/limitations).
- No authentication, no delete endpoint, no cleanup of old uploads.
- XLSX formulas: last saved value only. First sheet only. Comma-separated CSV only.
- No Docker files yet (Module 6).