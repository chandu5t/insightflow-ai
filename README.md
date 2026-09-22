# InsightFlow AI – Business Data Analyst Agent

InsightFlow AI lets you upload a business CSV or Excel file and ask questions in plain English, such as:

> Which region generated the highest revenue?

Python and Pandas perform every calculation. The LLM (Google Gemini) only understands the question and explains results that have already been validated.

> **Status:** Module 5 of 8 (LangGraph workflow and result validation).

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

- **633 tests passed**
- Backend unit tests and API tests
- Profiling, analysis, classification, dispatching, validation, reconciliation, number-grounding, explanation, and evaluation tests

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
- **Definitions:** "What is revenue?" is routed to a metric-definition branch. It is a stub until Module 7 and never pretends to have found a definition.
- The `POST /analysis/query` response format is unchanged.

## Docker

Docker support is added progressively. Docker Compose and the full stack are planned for Modules 6 and 8.

## API Endpoints

| Method | Path | Description | Module |
|---|---|---|---|
| GET | `/health` | Backend health check | 1 |
| POST | `/datasets/upload` | Upload a CSV or `.xlsx` file | 2 |
| GET | `/datasets/{dataset_id}/preview` | Preview dataset rows | 2 |
| GET | `/datasets/{dataset_id}/profile` | Generate a complete dataset profile | 3 |
| POST | `/analysis/query` | Ask a question about a dataset | 4–5 |

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
| What is revenue? | Not available yet; Module 7 |

Without a Gemini key, the rule-based classifier answers. Filters and forecasts are not supported.

## Screenshots

Added in Module 8.

## Future Improvements

See the version plan (V1.5, V2.0, V3.0) in the Technical Design Document.

## Author

**Chandrakant Thakare**

- GitHub: [chandu5t](https://github.com/chandu5t)
- LinkedIn: [www.linkedin.com/in/chandrakant-thakare-89994728b](https://www.linkedin.com/in/chandrakant-thakare-89994728b)
