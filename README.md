# InsightFlow AI – Business Data Analyst Agent

InsightFlow AI lets you upload a business CSV or Excel file and ask questions in plain English, such as:

> Which region generated the highest revenue?

Python and Pandas perform every calculation. The LLM (Google Gemini) only understands the question and explains results that have already been validated.

> **Status:** Under development. Module 2 of 8 (data ingestion).

## Features

Planned for V1 (built module by module):

- CSV and Excel (`.xlsx`, first sheet) upload with validation and a small data preview
- Dataset profiling (rows, columns, types, missing values, duplicates)
- Controlled analysis tools: aggregation, grouping, ranking, missing-value analysis
- Natural-language questions using Gemini
- LangGraph workflow with result validation
- Basic RAG for business metric definitions (PostgreSQL + pgvector)
- React frontend
- Pytest tests and Docker support

## Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, Pydantic |
| Data processing | Pandas |
| File parsing | Python `csv` module, openpyxl |
| LLM | Google Gemini |
| Workflow / RAG | LangGraph, LangChain, PostgreSQL, pgvector |
| Frontend | React + TypeScript (Vite) |
| Testing | Pytest |
| DevOps | Git, GitHub, Docker |

## Architecture

React → FastAPI → LangGraph workflow → approved Python tools (Pandas) → result validator → Gemini explanation → JSON response.

Detailed diagrams are added in later modules.

## Folder Structure

See `docs/PROJECT_CONTEXT.md` for the current structure.

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
| `backend/.env` | `CORS_ORIGINS` | Comma-separated frontend URLs allowed to call the API |
| `backend/.env` | `MAX_UPLOAD_SIZE_MB` | Maximum upload size (default: 10 MB) |
| `backend/.env` | `MAX_ROWS` / `MAX_COLUMNS` | Table limits (default: 100000 / 100) |
| `backend/.env` | `PREVIEW_DEFAULT_ROWS` / `PREVIEW_MAX_ROWS` | Preview size (default: 5 / 20) |
| `frontend/.env` | `VITE_API_URL` | Backend URL used by React |

All environment variables listed above are optional unless otherwise specified.

> **Important:** Never commit `.env` files.

## Running the Backend

```powershell
cd backend

.\.venv\Scripts\Activate.ps1

uvicorn app.main:app --reload --port 8000
```

API documentation:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Running the Frontend

```powershell
cd frontend

npm run dev
```

Open:

[http://localhost:5173](http://localhost:5173)

## Running Tests

```powershell
cd backend

.\.venv\Scripts\Activate.ps1

pytest -v
```

## Docker

Docker support is added progressively. Docker Compose and the full stack are planned for Modules 6 and 8.

## API Endpoints

| Method | Path | Description | Module |
|---|---|---|---|
| GET | `/health` | Backend health check | 1 |
| POST | `/datasets/upload` | Upload a CSV or `.xlsx` file | 2 |
| GET | `/datasets/{dataset_id}/preview` | First rows of a dataset (default: 5, maximum: 20) | 2 |

## Screenshots

Added in Module 8.

## Future Improvements

See the version plan (V1.5, V2.0, V3.0) in the Technical Design Document.

## Author

**Chandrakant Thakare**

- GitHub: [chandu5t](https://github.com/chandu5t)
- LinkedIn: [www.linkedin.com/in/chandrakant-thakare-89994728b](https://www.linkedin.com/in/chandrakant-thakare-89994728b)