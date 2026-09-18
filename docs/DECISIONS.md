# Architecture Decisions

| ID | Decision | Reason |
|---|---|---|
| D-001 | 8 modules. React is built in slices, PostgreSQL and RAG are separate modules, and LangGraph comes after a plain Python pipeline. | Each module is testable, and workflow bugs are separable from logic bugs. |
| D-002 | JSON metadata behind a `DatasetRepository` interface until Module 6, then PostgreSQL. | Keeps the database out of the way until the core workflow works. |
| D-003 | Unsupported and insufficient-data answers return HTTP 200 with a `status` field. Query responses include `validation`, `assumptions`, `calculation_method`. | These are valid requests we cannot answer, not client errors. Also fixes the TDD's missing "validation status". |
| D-004 | Frontend is Vite + React with TypeScript (`.tsx`). | Type safety and maintainability while keeping the frontend beginner-friendly. |
| D-005 | `dataset_id` is a UUID string everywhere. Malformed returns 422, unknown returns 404. | The TDD examples (`dataset_001`) contradicted its own database design. |
| D-006 | Route functions are plain `def`, not `async def`. | Pandas and Gemini calls block. FastAPI runs `def` routes in a thread pool. |
| D-007 | LangChain is used minimally (embeddings/LLM interface). Vector search is our own SQL against `knowledge_documents`. | LangChain's PGVector store creates its own tables and conflicts with the TDD schema. |
| D-008 | Gemini failure fallbacks: rule-based classifier and template explanation. | Reliability, cheaper tests, fewer LLM calls. |
| D-009 | Gemini output must validate against a Pydantic `QueryPlan`, and column names are checked against an allowlist. | Prevents prompt injection and invented columns. |
| D-010 | Dataset profile is computed on demand from the CSV. | The `dataset_columns` table only stores summary data. |
| D-011 | Separate `backend/Dockerfile` and `frontend/Dockerfile`, one `docker-compose.yml`, PostgreSQL via the `pgvector/pgvector` image. | One Dockerfile cannot serve three services. |
| D-012 | RAG is the last V1 feature (Module 7). METRIC_DEFINITION uses a stub until then. | The TDD contradicted itself on V1 vs V1.5 for RAG. |
| D-013 | Settings via pydantic-settings. `CORS_ORIGINS` is a comma-separated string. | Avoids the JSON-list-in-env-var pitfall. |
| D-014 | `data/samples/` is committed. `data/uploads/` contents are git-ignored. | Sample data is shared, user uploads are private. |
| D-015 | No authentication in V1. Local and portfolio use only. | Explicitly out of scope in the TDD. |
| D-016 | `/health` returns `app`, `version`, `environment` in addition to `status`. | Useful for debugging, harmless to expose. |
| D-017 | Virtual environment lives at `backend/.venv`. `requirements.txt` is generated with `pip freeze`. | Reproducible installs. |
| D-018 | Line endings are LF via `.gitattributes`. | Prevents CRLF problems in Docker scripts later. |