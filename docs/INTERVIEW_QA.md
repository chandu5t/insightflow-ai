# InsightFlow AI Interview Q&A

## Why FastAPI?
It provides REST endpoints and Pydantic request/response validation. The supplied routes use synchronous functions for blocking file, Pandas, and provider work.

## Why React + TypeScript?
React provides the upload, preview, profile, and question flow. TypeScript models the API boundary and catches some contract errors at build time.

## Why PostgreSQL?
It stores dataset metadata and analysis history. Uploaded CSV files remain on local disk. JSON storage remains available for native development.

## Why pgvector?
Metric definitions are stored with application data and searched by vector similarity without adding a separate vector database.

## Why RAG?
RAG supplies business metric definitions. It does not calculate values from uploaded datasets.

## Why Gemini?
Gemini classifies questions and creates embeddings. Python tools calculate analytical results.

## Why LangGraph?
It makes classification, routing, execution, validation, explanation, and response stages explicit and testable.

## Why deterministic calculations?
Pandas tools calculate directly from the dataset, making results reproducible and independently checkable.

## Why not let the LLM calculate numbers?
The LLM is not the calculation engine. It can miscalculate or invent values; deterministic tools and result validation provide a grounded result.

## How does validation work?
The workflow validates tool output and independently reconciles numerical results against the dataset and metadata.

## How does number grounding work?
The grounding check verifies explanation numbers against validated results. It reduces unsupported numeric claims but has documented limitations.

## How does Docker networking work?
The browser uses a host URL such as `http://localhost:8001`. The backend container uses Compose DNS to connect to PostgreSQL at `db:5432`.

## Why db:5432?
`db` is the Compose service name and `5432` is the PostgreSQL container port. Host-published port 5433 is for host tools.

## Why is localhost different?
Inside a container, `localhost` means that same container. A browser on the host cannot normally resolve Compose service names.

## How is persistence handled?
PostgreSQL uses the `pgdata` named volume. Normal `docker compose down` retains it; `down -v` deletes it.

## Why is RAG seeding manual?
Seeding calls Gemini to produce embeddings and may incur provider usage. It is intentionally separate from normal startup.

## Why is create_all() not migrations?
It creates missing tables but does not apply arbitrary changes to an existing schema. A future schema evolution needs an explicit migration plan.

## What are V1 limitations?
No authentication, local uploads, synchronous processing, limited CSV/XLSX behavior, manual RAG seeding, no ANN index, and no general production deployment configuration.

## What would V2 improve?
Possible future work includes authentication/authorization, more data sources, advanced analytics, broader evaluation, observability, deployment/scaling, and a migration system. These are not V1 features.
