FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY Pipfile /app/Pipfile

RUN pip install --no-cache-dir pipenv \
    && pipenv install --system --dev --skip-lock

COPY backend/app /app/app
COPY backend/scripts /app/scripts
COPY backend/tests /app/tests
COPY backend/sample_knowledge_base /app/sample_knowledge_base
COPY pytest.ini /app/pytest.ini

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
