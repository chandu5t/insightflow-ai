# InsightFlow AI backend image (Module 6).
# psycopg[binary] bundles its own libpq, so no libpq-dev/build-essential is needed here.
# Trade-off (documented in DECISIONS D-057): binary wheels are the practical dev choice;
# production deployments would use the non-binary psycopg package against system libpq.

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

# The actual data is bind-mounted from the host by docker-compose.yml; this just
# ensures the path exists before the app's first write.
RUN mkdir -p /app/data/uploads

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]