```markdown
# InsightFlow AI – Project Context

_Paste this file at the start of any new Claude conversation._

## Current status

- **Completed modules:** 1 (Foundation and health check)
- **Next module:** 2 (CSV upload, validation, storage)
- **Git tag:** `module-1` (to be created after commit)

## Environment

- Windows 11, VS Code, PowerShell
- Project root: `C:\dev\insightflow-ai`
- Python: 3.12
- Node.js: 22.13.1
- npm: 10.9.2
- Git: 2.46.1
- Docker: 29.7.2

## Folder structure

```

insightflow-ai/
├── docs/ PROJECT_CONTEXT.md, DECISIONS.md
├── backend/
│ ├── app/
│ │ ├── **init**.py
│ │ ├── main.py FastAPI app, CORS, router registration
│ │ ├── api/
│ │ │ ├── **init**.py
│ │ │ └── health_routes.py GET /health
│ │ ├── core/
│ │ │ ├── **init**.py
│ │ │ ├── config.py Settings (pydantic-settings, reads backend/.env)
│ │ │ └── logging_config.py setup_logging(level)
│ │ └── schemas/
│ │ ├── **init**.py
│ │ └── health_schema.py HealthResponse
│ ├── tests/
│ │ ├── **init**.py
│ │ ├── conftest.py
│ │ └── test_health.py
│ ├── data/
│ │ ├── samples/
│ │ └── uploads/
│ ├── pytest.ini
│ ├── requirements.txt
│ ├── .env.example
│ └── .env (not committed)
└── frontend/
├── src/
│   ├── services/
│   │   └── api.ts getHealth()
│   ├── components/
│   │   └── HealthStatus.tsx
│   ├── pages/
│   │   └── Dashboard.tsx
│   ├── App.tsx
│   ├── main.tsx
│   ├── App.css
│   └── index.css
├── .env.example
├── .env (not committed)
├── index.html
├── package.json
└── vite.config.ts

```

## API endpoints

| Method | Path | Response |
|---|---|---|
| GET | `/health` | `{"status","app","version","environment"}` |

## Environment variables

- `backend/.env`: `ENVIRONMENT`, `LOG_LEVEL`, `CORS_ORIGINS` (comma-separated)
- `frontend/.env`: `VITE_API_URL`

## Dependencies

- Backend: fastapi, uvicorn[standard], pydantic-settings, pytest, httpx (exact versions in `backend/requirements.txt`)
- Frontend: React, React DOM, Vite, TypeScript (exact versions in `frontend/package.json`)

## Database

None yet. Dataset metadata will use JSON files behind a repository interface until Module 6.

## Docker

No Docker files yet. First backend Dockerfile arrives in Module 2.

## Testing status

- 7 backend tests passing (health endpoint, CORS, settings parsing)
- Frontend production build passing
- No frontend tests yet

## Known issues / pending

- FastAPI's default error format (`{"detail": ...}`) differs from the TDD format (`{"error","code","details"}`). Fixed in Module 2 with global exception handlers.
- No authentication (by design for V1, see DECISIONS.md).
```
