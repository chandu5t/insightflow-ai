# InsightFlow AI – Project Context

*Paste this file at the start of any new Claude conversation.*

## Current status

- **Completed modules:** 1, 2, 3
- **In progress:** None
- **Next module:** 4 (Gemini question understanding)
- **Git tags:** `module-1`, `module-2`, `module-3`
- **Repository:** https://github.com/chandu5t/insightflow-ai
  (public, so never commit secrets)

Module 3 has been implemented and verified:
- Backend: 402 tests passed (137 existing + 265 new).
- Frontend: build and lint passed.
- Manual API testing completed.
- Browser testing completed for CSV and XLSX uploads.
- Dataset profile screen verified.
- Empty-file error handling verified.
- Dynamic preview clearing after failed uploads verified.

## Environment

- Windows 11, VS Code, PowerShell
- Project root: `C:\dev\insightflow-ai`
- Python 3.12
- FastAPI 0.141.1
- React 19
- TypeScript 6
- Vite 8

Exact dependency versions are maintained in:
- `backend/requirements.txt`
- `frontend/package.json`

## Working agreements

- "TDD" means the **Technical Design Document**.
- Use test-driven development where practical.
- One module at a time.
- The developer confirms a module works before the next begins.
- Each module guide is delivered in a single response (free Claude plan).
- Windows PowerShell commands first.
- Frontend remains TypeScript (`.tsx` / `.ts`).
- No architecture changes without discussion.
- Nothing is committed or pushed automatically.
- Never commit `.env` files, API keys, `.venv/`, `node_modules/` or uploaded datasets.
- Python and Pandas calculate analytical values. LLMs must not calculate numerical results directly.
- Module 4 must not send raw preview rows or top categorical values to Gemini.

## Module status

| # | Module | Status |
|---|---|---|
| 1 | Foundation and health check | Complete (tag `module-1`) |
| 2 | Data ingestion: CSV/.xlsx upload, validation, preview | Complete (tag `module-2`) |
| 3 | Profiling, column mapping, analysis tools | Complete (tag `module-3`) |
| 4 | Question understanding with Gemini | Planned |
| 5 | LangGraph workflow and result validation | Planned |
| 6 | PostgreSQL integration (+ Docker Compose) | Planned |
| 7 | Basic RAG with pgvector | Planned |
| 8 | Hardening, full Docker, documentation, release | Planned |

## Folder structure (end of Module 3)

```text
insightflow-ai/

├── docs/
│   ├── PROJECT_CONTEXT.md
│   └── DECISIONS.md
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── health_routes.py
│   │   │   └── dataset_routes.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── errors.py
│   │   │   └── logging_config.py
│   │   ├── schemas/
│   │   │   ├── health_schema.py
│   │   │   ├── error_schema.py
│   │   │   ├── dataset_schema.py
│   │   │   ├── profile_schema.py
│   │   │   └── tool_schema.py
│   │   ├── services/
│   │   │   ├── file_parser.py
│   │   │   ├── dataset_repository.py
│   │   │   ├── dataset_service.py
│   │   │   └── profiling_service.py
│   │   ├── tools/
│   │   │   ├── missing_value_tool.py
│   │   │   ├── profiling_tool.py
│   │   │   ├── revenue.py
│   │   │   ├── aggregation_tool.py
│   │   │   ├── grouping_tool.py
│   │   │   └── ranking_tool.py
│   │   └── utils/
│   │       ├── upload_validator.py
│   │       ├── table_validator.py
│   │       ├── column_mapper.py
│   │       └── dataframe_utils.py
│   │
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── helpers.py
│   │   ├── frames.py
│   │   ├── test_health.py
│   │   ├── test_upload_validator.py
│   │   ├── test_file_parser.py
│   │   ├── test_dataset_repository.py
│   │   ├── test_datasets_api.py
│   │   ├── test_errors.py
│   │   ├── test_column_mapper.py
│   │   ├── test_dataframe_utils.py
│   │   ├── test_missing_value_tool.py
│   │   ├── test_profiling_tool.py
│   │   ├── test_revenue.py
│   │   ├── test_aggregation_tool.py
│   │   ├── test_grouping_tool.py
│   │   ├── test_ranking_tool.py
│   │   ├── test_profile_api.py
│   │   └── test_evaluation.py
│   │
│   ├── data/
│   │   ├── samples/
│   │   ├── uploads/
│   │   └── evaluation/
│   │
│   ├── pytest.ini
│   ├── requirements.txt
│   └── .env.example
│
└── frontend/
    ├── src/
    │   ├── services/
    │   │   └── api.ts
    │   ├── types/
    │   │   ├── api.ts
    │   │   ├── dataset.ts
    │   │   └── profile.ts
    │   ├── components/
    │   │   ├── HealthStatus.tsx
    │   │   ├── FileUpload.tsx
    │   │   ├── DatasetPreview.tsx
    │   │   ├── DatasetProfile.tsx
    │   │   └── ErrorMessage.tsx
    │   ├── pages/
    │   │   └── Dashboard.tsx
    │   ├── styles/
    │   │   ├── dataset.css
    │   │   └── profile.css
    │   ├── App.tsx
    │   ├── main.tsx
    │   ├── App.css
    │   └── index.css
    │
    ├── package.json
    └── .env.example
```

## API endpoints

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | Returns application status, version and environment |
| POST | `/datasets/upload` | Upload a CSV or XLSX file |
| GET | `/datasets/{dataset_id}/preview?rows=N` | Returns preview rows; default 5, maximum 20 |
| GET | `/datasets/{dataset_id}/profile` | Returns dataset types, missing values, duplicates and statistics |

### Profile response

The profile endpoint returns:

- Row and column counts
- Duplicate row count
- Total missing cells
- Per-column data type
- Missing count and percentage
- Unique count
- Numeric statistics
- Datetime range
- Limited top values
- Column role mapping
- Warnings

### Error codes

Existing errors include:

`UNSUPPORTED_FILE_TYPE`, `FILE_TOO_LARGE`, `EMPTY_FILE`,
`INVALID_ENCODING`, `CORRUPTED_FILE`, `MISSING_HEADERS`,
`DUPLICATE_COLUMNS`, `NO_DATA_ROWS`, `TOO_MANY_ROWS`,
`TOO_MANY_COLUMNS`, `MALFORMED_ROWS`, `INVALID_DATASET_ID`,
`DATASET_NOT_FOUND`, `INVALID_PREVIEW_ROWS`, `VALIDATION_ERROR`,
`NOT_FOUND`, `METHOD_NOT_ALLOWED`, `HTTP_ERROR`, `INTERNAL_ERROR`.

Module 3 errors include:

- `DATASET_UNREADABLE` — HTTP 500
- `MISSING_COLUMN` — HTTP 422 through `ToolError`
- `AMBIGUOUS_COLUMN` — HTTP 422 through `ToolError`
- `INSUFFICIENT_DATA` — HTTP 422 through `ToolError`
- `INVALID_NUMERIC_VALUES` — HTTP 422 through `ToolError`
- `INVALID_PARAMETER` — HTTP 422 through `ToolError`

## Environment variables

### Backend

The following variables are available in `backend/.env`:

- `ENVIRONMENT`
- `LOG_LEVEL`
- `CORS_ORIGINS`
- `UPLOAD_DIR`
- `MAX_UPLOAD_SIZE_MB` — default 10
- `MAX_XLSX_UNCOMPRESSED_MB` — default 100
- `MAX_ROWS` — default 100000
- `MAX_COLUMNS` — default 100
- `PREVIEW_DEFAULT_ROWS` — default 5
- `PREVIEW_MAX_ROWS` — default 20
- `PREVIEW_MAX_CELL_LENGTH` — default 100
- `PROFILE_TOP_VALUES` — default 10
- `PROFILE_VALUE_MAX_LENGTH` — default 50

### Frontend

- `VITE_API_URL`

## Data storage (until Module 6)

```text
data/uploads/{dataset_id}.csv
data/uploads/{dataset_id}.meta.json
```

- Uploaded data is stored as clean UTF-8 CSV.
- Values are stored as text.
- The dataset path is derived from the UUID.
- The path is never returned to the frontend.
- Access is handled through `DatasetRepository`.
- Module 3 reads the stored CSV using Pandas.
- PostgreSQL integration is planned for Module 6.

## Dependencies

### Backend

- fastapi
- uvicorn[standard]
- pydantic-settings
- pytest
- httpx
- python-multipart
- openpyxl
- pandas
- numpy
- python-dateutil
- six
- tzdata

Exact versions are maintained in `backend/requirements.txt`.

### Frontend

- react
- react-dom
- vite
- typescript

No additional frontend packages were required for Module 3.

## Analysis Tools (Module 3)

Python and Pandas calculate every number. There is no LLM in these tools.

| Tool | File | What it does |
|---|---|---|
| Profiling | `backend/app/tools/profiling_tool.py` | Types, missing values, statistics and limited value lists |
| Missing values | `backend/app/tools/missing_value_tool.py` | Missing counts and percentages, duplicate rows |
| Aggregation | `backend/app/tools/aggregation_tool.py` | Sum, average, count, total revenue and average order value |
| Grouping | `backend/app/tools/grouping_tool.py` | Sum, average or count per group |
| Ranking | `backend/app/tools/ranking_tool.py` | Top N and bottom N with stable tie handling |
| Column mapper | `backend/app/utils/column_mapper.py` | Maps columns to revenue, quantity, unit price, product, order ID and region |

### Column roles

Column names are matched only from a fixed alias table.

- Capitalization, spaces, hyphens and underscores are ignored.
- Matching columns are mapped to predefined roles.
- If two columns match the same role, `AMBIGUOUS_COLUMN` is raised.
- `column_overrides` is the explicit way to choose a column.
- Module 4 must not independently guess column roles.

### Revenue calculation

Revenue is calculated using the following priority:

1. Direct revenue column.
2. `quantity × unit_price`.
3. `INSUFFICIENT_DATA` if neither option is available.

The calculation method is always returned.

- Negative values are preserved.
- Empty cells are excluded and counted.
- Invalid values stop the calculation.
- Values such as `abc`, `1,200`, `$50`, `inf` and `nan` are not silently corrected or skipped.

### Average order value

Average order value is:

```text
Total revenue / DISTINCT order IDs
```

- Only rows containing both revenue and order ID are used.
- Without an order ID column, the calculation returns `INSUFFICIENT_DATA`.
- A row-count fallback is available only when explicitly enabled with:
  `allow_row_count_fallback=True`.
- Zero orders return `INSUFFICIENT_DATA`.

### Profile rules

- Profiling is computed on demand from the stored CSV.
- Supported detected types include integer, float, boolean, datetime, text and empty.
- Leading-zero numbers are treated as text.
- Only ISO dates are recognised.
- At most 10 top values are returned per column.
- Each top value is limited to 50 characters.
- NaN and infinity are converted to `null` for JSON safety.
- Numeric values are rounded to 6 decimal places.
- Percentages are rounded to 2 decimal places.
- Profile top values and preview rows must never be sent to an LLM.

## Key decisions

See `docs/DECISIONS.md` for decisions D-001 through D-039.

Important decisions include:

- D-019: Frontend uses TypeScript.
- D-020: Support CSV and first-sheet XLSX uploads.
- D-021: Preview data is capped and must not be sent to an LLM.
- D-022: Pandas is introduced in Module 3.
- D-023: Consistent error response format.
- D-024: Dataset storage layout.
- D-028: Docker deferred to Module 6.
- D-029: Analysis tools are plain functions returning Pydantic result models.
- D-030: Only empty cells are missing; invalid numeric text is not silently skipped.
- D-031: Column roles use the fixed alias mapper.
- D-032: Revenue uses direct revenue or quantity multiplied by unit price.
- D-033: Average order value uses distinct order IDs.
- D-034: Tools raise `ToolError`.
- D-035: Profile is computed on demand and limited.
- D-036: Ranking uses competition ranks and stable tie handling.
- D-037: Numbers and percentages use fixed rounding rules.
- D-038: Fixed evaluation files are committed to Git.
- D-039: Docker remains planned for Module 6.

## Testing status

### Backend

- Total tests: **402**
- Existing tests: 137
- New Module 3 tests: 265
- Result: **402 passed**

Run all backend tests:

```powershell
cd C:\dev\insightflow-ai\backend
.\.venv\Scripts\Activate.ps1
pytest -q
```

### Module 3 test files

- `test_column_mapper.py` — 48 tests
- `test_dataframe_utils.py` — 18 tests
- `test_missing_value_tool.py` — 13 tests
- `test_profiling_tool.py` — 39 tests
- `test_revenue.py` — 26 tests
- `test_aggregation_tool.py` — 37 tests
- `test_grouping_tool.py` — 20 tests
- `test_ranking_tool.py` — 22 tests
- `test_profile_api.py` — 15 tests
- `test_evaluation.py` — 27 tests

### Frontend

The following commands passed:

```powershell
cd C:\dev\insightflow-ai\frontend
npm run build
npm run lint
```

Manual browser testing confirmed:

- Backend health status
- CSV upload
- XLSX upload
- Data preview
- Dynamic row selector
- Dataset profile
- Numeric and text classification
- Missing values
- Duplicate rows
- Column mapping
- Error handling when backend is unavailable
- Retry functionality
- Empty-file handling
- Clearing old preview after failed upload

## Known limitations

- The framework receives the whole upload before the size check runs.
- No authentication.
- No delete endpoint.
- No cleanup of old uploads.
- XLSX formulas use the last saved value only.
- XLSX support is limited to the first sheet.
- CSV files must be comma-separated.
- Only empty cells are treated as missing.
- `N/A` and `null` remain text values.
- Currency symbols and thousand separators are not cleaned.
- Invalid numeric text stops calculations with a clear error.
- Day/month dates such as `05/01/2025` are not guessed and remain text.
- The profile is recomputed on every request and has no caching.
- Profiling may take approximately one second for 100,000 rows.
- The profile returns the 10 most common raw values per text column to the browser.
- Profile top values must never be sent to an LLM.
- Groups are case-sensitive; `laptop` and `Laptop` are separate groups.
- Direct revenue is not checked against `quantity × unit_price`.
- Duplicate rows are exact copies of all columns and are not removed from calculations.
- Tools do not yet have dedicated HTTP endpoints.
- Module 4 and Module 5 will wrap the tools into the LangGraph workflow.
- Tested with Pandas 3.0.6.
- Docker files are planned for Module 6.
- PostgreSQL and pgvector are planned for later modules.
- No production deployment configuration exists yet.

## Next module

### Module 4 — Gemini question understanding

Planned responsibilities:

- Accept a natural-language business question.
- Use Gemini to identify the user's intent.
- Identify the requested operation.
- Identify relevant column roles.
- Extract filters, grouping and ranking parameters.
- Produce structured Pydantic output.
- Validate Gemini output.
- Avoid sending raw preview rows and top categorical values.
- Keep numerical calculations inside Python analysis tools.

Module 4 must not introduce architecture changes without discussion.