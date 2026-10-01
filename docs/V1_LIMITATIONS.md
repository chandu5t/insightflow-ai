# InsightFlow AI V1 Limitations

The following are current V1 limitations unless local verification proves otherwise.

- No authentication or authorization.
- No delete endpoint or automated upload cleanup.
- Local upload storage.
- XLSX reads only the first worksheet; formulas use saved values.
- CSV parser has documented delimiter/encoding and data-cleaning limits.
- Profile recomputes on each request.
- RAG seeding is manual and requires Gemini embeddings.
- No ANN index for the small knowledge corpus.
- `create_all()` does not migrate arbitrary schema changes.
- No production deployment configuration.
- No frontend test runner or complete automated E2E suite.

Do not claim Module 8 fixes a limitation unless you implement and verify that fix.
