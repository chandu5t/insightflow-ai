# InsightFlow AI – Business Data Analyst Agent

InsightFlow AI lets you upload a business CSV or Excel file and ask questions in plain English, such as:

> Which region generated the highest revenue?

Python and Pandas perform every calculation. The LLM (Google Gemini) only understands the question and explains results that have already been validated.

> **Status:** Module 7 of 8 (Basic RAG with pgvector).

## Features

Implemented and planned for V1:

- CSV and Excel (`.xlsx`, first sheet) upload with validation and data preview
- Dataset profiling: rows, columns, data types, missing values, duplicate rows, numeric statistics, date ranges, categorical distributions, and automatic column role detection
- Controlled analysis tools: revenue, aggregation, grouping, ranking, and missing-value analysis
- Plain-English questions using Gemini classification with a rule-based fallback
- LangGraph workflow with result validation and number-grounding checks
- Basic RAG for business metric definitions (PostgreSQL + pgvector)
- React frontend with TypeScript
- Pytest tests
- Docker support

## Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, Pydantic |
| Data processing | Pandas, NumPy |
| File parsing | Python `csv` module, openpyxl |
| LLM | Google Gemini |
| Workflow / RAG | LangGraph, LangChain, PostgreSQL, pgvector |
| Frontend | React + TypeScript (Vite) |
| Testing | Pytest |
| DevOps | Git, GitHub, Docker |

## Architecture

React -> FastAPI -> LangGraph workflow (classify, route, execute, validate, explain, respond) -> Pandas tools -> validation and number-grounding checks.

Gemini is used for question classification when configured. A built-in rule-based classifier is used as a fallback. Numerical calculations remain inside Python and Pandas analysis tools.

Detailed diagrams are added in later modules.

## Folder Structure

See `docs/PROJECT_CONTEXT.md` for the current project structure.

## Installation

### Prerequisites

- Python 3.11+ (3.12 recommended)
- Node.js LTS
- Git

### Backend Setup (Windows PowerShell)

```powershell
cd backend

python -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt

Copy-Item .env.example .env
```

### Frontend Setup

```powershell
cd frontend

npm install

Copy-Item .env.example .env
```

## Environment Setup

| File | Variable | Purpose |
|---|---|---|
| `backend/.env` | `ENVIRONMENT` | `development` or `production` |
| `backend/.env` | `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, or `ERROR` |
| `backend/.env` | `CORS_ORIGINS` | Comma-separated frontend URLs |
| `backend/.env` | `MAX_UPLOAD_SIZE_MB` | Maximum upload size; default 10 MB |
| `backend/.env` | `MAX_ROWS` / `MAX_COLUMNS` | Table limits; default 100000 / 100 |
| `backend/.env` | `PREVIEW_DEFAULT_ROWS` / `PREVIEW_MAX_ROWS` | Preview size; default 5 / 20 |
| `backend/.env` | `PROFILE_TOP_VALUES` | Maximum categorical values shown |
| `backend/.env` | `PROFILE_VALUE_MAX_LENGTH` | Maximum displayed categorical value length |
| `backend/.env` | `GEMINI_API_KEY` | Gemini API key; backend only and never commit |
| `backend/.env` | `GEMINI_MODEL` | Gemini model used for classification |
| `backend/.env` | `GEMINI_TIMEOUT_SECONDS` | Gemini request timeout |
| `backend/.env` | `GEMINI_MIN_CONFIDENCE` | Minimum accepted Gemini confidence |
| `backend/.env` | `MAX_QUESTION_LENGTH` | Maximum question length |
| `backend/.env` | `CURRENCY_SYMBOL` | Currency symbol; default `₹` |
| `backend/.env` | `STORAGE_BACKEND` | Storage backend: `json` or `postgres` |
| `backend/.env` | `POSTGRES_USER` | PostgreSQL username |
| `backend/.env` | `POSTGRES_PASSWORD` | PostgreSQL password |
| `backend/.env` | `POSTGRES_DB` | PostgreSQL database name |
| `backend/.env` | `POSTGRES_HOST` | PostgreSQL host |
| `backend/.env` | `POSTGRES_PORT` | PostgreSQL port |
| `backend/.env` | `GEMINI_EMBEDDING_MODEL` | Embedding model; default `gemini-embedding-001` |
| `backend/.env` | `KNOWLEDGE_EMBEDDING_DIMENSIONS` | Embedding dimensions; default `768` |
| `backend/.env` | `KNOWLEDGE_SIMILARITY_THRESHOLD` | Minimum similarity threshold; default `0.6` |
| `backend/.env` | `KNOWLEDGE_TOP_K` | Maximum retrieved knowledge documents; default `3` |
| `frontend/.env` | `VITE_API_URL` | Backend URL used by React |

> **Important:** Never commit `.env` files or API keys.

## Running the Backend

```powershell
cd backend

.\.venv\Scripts\Activate.ps1

uvicorn app.main:app --reload --port 8000
```

API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Running the Frontend

```powershell
cd frontend

npm run dev
```

Open: [http://localhost:5173](http://localhost:5173)

## Running Tests

```powershell
cd backend

.\.venv\Scripts\Activate.ps1

pytest -v
```

Current test status:

- **693 tests passed**
- Backend unit tests and API tests
- Profiling, analysis, classification, dispatching, validation, reconciliation, number-grounding, explanation, evaluation, and PostgreSQL integration tests

Frontend verification:

```powershell
cd frontend

npm run lint

npm run build
```

Both frontend commands pass.

## Workflow and Validation (Module 5)

Each question runs through a LangGraph workflow: classify -> route -> execute -> validate -> explain -> respond.

Failures skip the steps that make no sense (for example, no explanation is written for a result that failed validation).

- **Validation:** finite numbers, row accounting, an independent recomputation of the result, the row count against the upload, and (for groups) totals that add up. Each failed check has a machine-readable code.
- **Number grounding:** every number in an explanation must come from the validated result. Otherwise the answer is rejected with `EXPLANATION_NOT_GROUNDED`.
- **Definitions:** Metric-definition questions use the Module 7 knowledge base when PostgreSQL and Gemini embeddings are configured. If no match is found or the knowledge base is unavailable, the existing stub response is returned.
- The `POST /analysis/query` response format is unchanged.

## PostgreSQL and Docker (Modules 6–7)

For local development without Docker, keep `STORAGE_BACKEND=json` in
`backend/.env` — no database is needed.

To run with PostgreSQL:

```powershell
cd C:\dev\insightflow-ai
docker compose up -d
docker compose ps
```

The backend inside Docker Compose connects to PostgreSQL at `db:5432`
(the Compose service name). From your Windows machine, use `localhost:5433`
with `psql` or a GUI tool.

Dataset metadata and analysis history persist across `docker compose down`
and `up` as long as you don't add `-v`, which deletes the named volume.

```powershell
docker compose logs backend --tail 40
docker compose exec db psql -U insightflow -d insightflow -c "\dt"
docker compose down
```

PostgreSQL stores dataset metadata and analysis history. Uploaded CSV files
remain on disk. The application supports the JSON storage backend for local
development and the PostgreSQL storage backend for Docker Compose.

## API Endpoints

| Method | Path | Description | Module |
|---|---|---|---|
| GET | `/health` | Backend health check | 1 |
| POST | `/datasets/upload` | Upload a CSV or `.xlsx` file | 2 |
| GET | `/datasets/{dataset_id}/preview` | Preview dataset rows | 2 |
| GET | `/datasets/{dataset_id}/profile` | Generate a complete dataset profile | 3 |
| POST | `/analysis/query` | Ask a question about a dataset | 4–5 |
| POST | `/knowledge/search` | Search the built-in metric knowledge base | 7 |

## Analysis Tools

| Tool | Description |
|---|---|
| Revenue Tool | Calculates total revenue using direct revenue or `quantity × unit_price` |
| Aggregation Tool | Supports sum, mean, minimum, maximum, and count |
| Grouping Tool | Groups data by categorical columns and calculates supported metrics |
| Ranking Tool | Ranks grouped results in ascending or descending order |
| Missing Value Tool | Analyzes missing values across dataset columns |
| Profiling Tool | Generates metadata, statistics, types, duplicates, and warnings |

All analysis tools use deterministic Python and Pandas calculations. The LLM does not directly perform numerical calculations.

## Asking Questions (Modules 4–5)

Gemini, or the rule-based fallback, decides what the question means. Python and Pandas calculate the answer, the result is independently validated, and the explanation comes from templates. Number-grounding checks reject explanations containing unsupported numbers.

| Question | Needs |
|---|---|
| What is the total revenue? | A revenue column, or quantity and unit price |
| Revenue by region / Which region generated the highest revenue? | Revenue fields and a region column |
| What is the average unit price? / maximum quantity? | That column |
| How many records are present? | Nothing |
| What is the average order value? | Revenue fields and an order ID |
| Are there missing values? | Nothing |
| What is revenue? | Built-in knowledge base when PostgreSQL and Gemini are configured; otherwise the existing stub response |

Without a Gemini key, the rule-based classifier answers. Filters and forecasts are not supported.

## RAG Knowledge Base (Module 7)

Requires `STORAGE_BACKEND=postgres` and a real `GEMINI_API_KEY`. Seed the knowledge base once after deployment or schema changes:

```powershell
docker compose exec backend python -m app.scripts.seed_knowledge_base
```

Questions such as "What does average order value mean?" can return a retrieved definition with a **Source: InsightFlow AI built-in knowledge base** badge. If there is no matching definition, or PostgreSQL/Gemini is unavailable, the existing stub response is returned instead.

Seeding is explicit and is not performed automatically at application startup.

## Screenshots

Added in Module 8.

## Future Improvements

See the version plan (V1.5, V2.0, V3.0) in the Technical Design Document.

## Author

**Chandrakant Thakare**

- GitHub: [chandu5t](https://github.com/chandu5t)
- LinkedIn: [www.linkedin.com/in/chandrakant-thakare-89994728b](https://www.linkedin.com/in/chandrakant-thakare-89994728b)
