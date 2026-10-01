# Troubleshooting

Run commands from `C:\dev\insightflow-ai` unless stated otherwise.

| Issue / symptom | Diagnostic | Likely fix | Verify |
|---|---|---|---|
| Docker Desktop not running | `docker version` | Start Docker Desktop | `docker compose ps` |
| Port already in use | `docker compose ps`; inspect Windows port owner | Change published host port and update `VITE_API_URL`/CORS | Rebuild/recreate; open browser URL |
| DB unhealthy | `docker compose logs db` | Check local Compose `.env`, disk, DB logs | `docker compose exec db pg_isready ...` |
| Backend cannot connect to DB | `docker compose logs backend` | Docker backend must use `db:5432`, not host `localhost:5433` | `docker compose ps` and `/health` |
| Frontend cannot reach backend | Browser developer console | Use host URL `http://localhost:8001` | Retry query/upload |
| CORS error | Browser console and backend `CORS_ORIGINS` | Add exact browser origin | Repeat browser API request |
| Wrong `VITE_API_URL` | Inspect local Compose file/build arg (do not post resolved secrets) | Set host-accessible URL and rebuild frontend | Browser request goes to port 8001 |
| Missing Gemini key | Backend/seed error message | Configure it privately for backend only | Retry seed/query |
| Gemini timeout | Backend logs | Retry after provider/network issue; use existing fallback behavior | Repeat one query |
| Gemini rate limit | Provider/backend error | Wait or resolve quota; avoid repeated seeding | Retry once |
| RAG seed failure | Backend logs, DB health, extension | Confirm DB healthy, extension enabled, key configured | Rerun explicit seed command |
| Empty RAG results | Check row count and seed output | Seed or verify query/threshold; do not lower threshold blindly | Ask a definition question |
| pgvector extension missing | Query `pg_extension` | Existing volumes may need a deliberate extension operation; backup first | Repeat extension query |
| Schema mismatch | Backend startup logs | `create_all()` cannot alter tables; plan a migration/manual repair | Restart and inspect tables |
| Stale image | `docker compose images` and logs | Rebuild targeted images | `docker compose up -d` |
| Existing volume state uncertain | `docker volume ls`; DB queries | Inspect and back up; do not delete as routine troubleshooting | Verify rows after restart |
| Frontend build fails | Run `npm run lint`, `npm run build` | Fix reported source/type error | Rebuild frontend |
| npm dependency failure | `node --version`; `npm ci` | Use compatible Node 22 and committed lockfile | Run `npm ci` then build |
| Backend dependency failure | `python --version`; pip error | Use Python 3.12 venv and supplied requirements | Run backend tests |
| pytest failure | `pytest -q` | Separate unit failures from skipped integration tests; use test DB | Rerun and record counts |
| Environment value ignored | `docker compose config` locally | Root `.env` feeds Compose; `backend/.env` feeds native backend | Check services without publishing resolved config |
| Upload rejected | UI/API error response | Check extension, size, dimensions, encoding, row/column limits | Retry with supported file |
