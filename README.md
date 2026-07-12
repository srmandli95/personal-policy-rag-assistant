# Document RAG Framework

Document RAG Framework is a production-oriented retrieval-augmented generation framework for document intelligence products. It provides authenticated document ingestion, PostgreSQL/pgvector indexing, citation-backed question answering, durable background jobs, and golden evaluation workflows that can be adapted for enterprise or domain-specific deployments.

The system is structured as a reusable full-stack framework with a React frontend, FastAPI backend, durable worker process, PostgreSQL/pgvector storage, pluggable model services, and OpenAI-backed answer generation. The included Docker Compose setup is the reference development deployment; the architecture is designed to evolve toward managed databases, object storage, horizontal API workers, dedicated job workers, and production observability.

## Core Capabilities

- Google OAuth sign-in with user-scoped documents, chats, datasets, and runs.
- Document upload for PDF, DOCX, TXT, Markdown, and HTML files.
- Durable ingestion pipeline for extraction, chunking, embedding, and indexing.
- Hybrid retrieval using vector search and BM25 keyword search.
- Cross-encoder reranking for stronger evidence ordering.
- LangGraph RAG orchestration with bounded retrieval and generation retries.
- Citation validation and grounded refusal behavior.
- Persistent chat sessions and message history.
- Uploadable golden evaluation datasets with case-level scoring.
- Docker Compose reference stack with frontend, backend, worker, and PostgreSQL.
- Clear path to production deployment with externalized secrets, managed persistence, independent worker scaling, HTTPS cookies, and disabled debug surfaces.

## System Overview

```text
Browser
  |
  v
React + Nginx frontend
  |
  v
FastAPI backend
  |                \
  |                 \ enqueue durable jobs
  v                  v
PostgreSQL + pgvector  Worker
  ^                  |
  |                  v
  +---------- ingestion, embeddings, evaluations

Chat requests:
FastAPI -> LangGraph -> retrieval -> reranking -> OpenAI -> citations -> response
```

## Reference Services

| Service | Development URL | Purpose |
| --- | --- | --- |
| Frontend | `http://localhost:3000` | React workspace and API proxy |
| Backend API | `http://localhost:8000` | FastAPI routes, auth, chat, documents, evaluations |
| API docs | `http://localhost:8000/docs` | OpenAPI documentation |
| PostgreSQL | `localhost:5432` | Relational data and pgvector embeddings |
| Worker | internal | Durable document-processing and evaluation jobs |

## Development Prerequisites

- Docker Desktop with Docker Compose v2
- Git
- Google OAuth 2.0 web client
- OpenAI API key
- Internet access during first build and first model download

Recommended development resources:

- 8 GB RAM minimum
- 10 GB free disk space for Docker images, model cache, uploaded files, and generated artifacts

## Development Quick Start

1. Clone the repository.

   ```bash
   git clone https://github.com/srmandli95/document-rag-framework.git
   cd document-rag-framework
   ```

2. Create the environment file.

   ```bash
   cp .env.example .env
   ```

3. Generate a development JWT secret.

   ```bash
   openssl rand -hex 32
   ```

4. Update `.env` with at least:

   ```dotenv
   OPENAI_API_KEY=your-openai-api-key
   OPENAI_MODEL_NAME=gpt-4o-mini

   JWT_SECRET_KEY=replace-with-generated-secret
   AUTH_COOKIE_SECURE=false
   FRONTEND_URL=http://localhost:3000

   GOOGLE_CLIENT_ID=your-google-client-id
   GOOGLE_CLIENT_SECRET=your-google-client-secret
   GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback
   ```

5. Build and start the stack.

   ```bash
   make build
   ```

6. Open the development workspace.

   ```text
   http://localhost:3000
   ```

## Google OAuth Setup

Create a Google OAuth web application in Google Cloud Console and configure this authorized redirect URI exactly:

```text
http://localhost:8000/auth/google/callback
```

The URI configured in Google Cloud must match `GOOGLE_REDIRECT_URI` in `.env`, including protocol, host, port, and path.

## Production Deployment Model

The current repository ships with a Docker Compose reference deployment, but the runtime boundaries are intentionally production-shaped:

- Serve the React build through a CDN or managed static hosting layer, or keep Nginx as an edge container behind a load balancer.
- Run the FastAPI backend as horizontally scalable API workers behind HTTPS.
- Run one or more dedicated worker replicas for document processing and evaluation workloads.
- Use a managed PostgreSQL service with pgvector enabled, connection pooling, backups, and monitoring.
- Move uploaded files and derived artifacts from local filesystem storage to object storage through the storage abstraction.
- Store secrets in a cloud secrets manager rather than `.env`.
- Set `AUTH_COOKIE_SECURE=true` and terminate TLS at the ingress/load balancer.
- Keep retrieval debug endpoints disabled except in controlled environments.
- Add metrics, traces, structured logs, and queue-depth dashboards before operating at scale.

## Common Commands

Run these from the repository root.

| Command | Description |
| --- | --- |
| `make build` | Build images and start the stack |
| `make run` | Start existing images |
| `make stop` | Stop the stack |
| `make logs` | Follow all Docker Compose logs |
| `make health` | Call the backend health endpoint |
| `make test` | Run backend tests inside the backend container |

Useful direct Docker commands:

```bash
docker compose ps
docker compose logs -f backend
docker compose logs -f worker
docker compose up -d --build backend worker
docker compose up -d --build frontend
```

## Using The Application

### Knowledge Base

1. Sign in with Google.
2. Open the **Knowledge Base** tab.
3. Upload a supported file type.
4. Wait for the document status to become `ready`.

Document processing is asynchronous in Docker Compose. The backend creates a processing job and the worker claims it, extracts text, chunks content, embeds chunks, and stores vectors in PostgreSQL.

### Chat

1. Open the **Chat** tab.
2. Ask a question covered by ready documents.
3. Review the answer and citations.

The RAG graph can rewrite queries, retrieve evidence, rerank candidates, generate an answer, validate support, and retry within configured bounds. If the system cannot ground an answer, it should refuse instead of fabricating unsupported details.

### Evaluations

1. Open the **Evaluations** tab.
2. Copy ready document IDs.
3. Download the JSON template.
4. Upload a golden evaluation dataset.
5. Run the selected dataset.

Example dataset:

```json
{
  "name": "Policy regression set",
  "cases": [
    {
      "id": "supported_case_001",
      "category": "general",
      "question": "What benefit does this policy provide?",
      "expected_answer_contains": ["expected phrase"],
      "expected_document_ids": ["replace-with-ready-document-id"],
      "expected_refusal": false
    },
    {
      "id": "refusal_case_001",
      "category": "general",
      "question": "Ask about information absent from the documents",
      "expected_answer_contains": [],
      "expected_document_ids": [],
      "expected_refusal": true
    }
  ]
}
```

Supported cases require at least one ready document ID. Refusal cases must use an empty `expected_document_ids` list.

## Configuration Highlights

| Setting | Purpose |
| --- | --- |
| `DATABASE_URL` | Synchronous SQLAlchemy connection for most backend and worker persistence |
| `ASYNC_DATABASE_URL` | Async SQLAlchemy connection for async chat persistence paths |
| `JOB_EXECUTION_MODE` | `worker` in Docker Compose, `inline` default for development/tests |
| `ASYNC_RAG_WORKFLOW` | Enables async LangGraph request handling |
| `MAX_UPLOAD_SIZE_MB` | Backend upload limit |
| `EMBEDDING_MODEL_NAME` | Sentence-transformer embedding model used by the default embedding provider |
| `RERANKER_MODEL_NAME` | Cross-encoder reranker model used by the default reranking provider |
| `OPENAI_MODEL_NAME` | Main answer-generation model |
| `OPENAI_AUX_MODEL_NAME` | Auxiliary query rewrite and support-check model |

## Data Persistence

- PostgreSQL data is stored in the Docker volume `postgres_data`.
- Uploaded and generated document artifacts are stored in the repository `data/` directory.
- Stopping the stack does not delete stored data.

To stop services:

```bash
make stop
```

To remove containers and the PostgreSQL volume:

```bash
docker compose down -v
```

This removes persisted database records, embeddings, chat history, evaluation datasets, and evaluation runs. Files under `data/` may need to be removed separately for a fully clean workspace.

## Testing

Backend tests in Docker:

```bash
make test
```

Targeted backend tests with the repository virtual environment:

```bash
PYTHONPATH=backend .venv/bin/pytest backend/tests/test_job_worker.py -q
```

Frontend tests:

```bash
cd frontend
npm test -- --run
```

## Troubleshooting

### Document stays in processing

Check the worker first:

```bash
docker compose ps
docker compose logs --tail=300 worker
```

The worker is responsible for document extraction, chunking, embedding, and worker-mode evaluation runs.

### Evaluation run stays pending or running

Check worker logs and recent evaluation run rows:

```bash
docker compose logs --tail=300 worker
```

Also refresh the frontend after rebuilding, because production frontend assets are served by Nginx.

### OAuth redirect mismatch

Confirm Google Cloud and `.env` both use:

```text
http://localhost:8000/auth/google/callback
```

### OpenAI request failures

Confirm `OPENAI_API_KEY` is present and restart the backend:

```bash
docker compose restart backend
```

### First model operation is slow

The default embedding and reranking models may be downloaded from Hugging Face on first use. Watch backend or worker logs during the first run.

## Project Layout

```text
backend/
  app/
    api/           FastAPI routes
    auth/          OAuth, cookies, JWT helpers, auth dependencies
    db/            SQLAlchemy engines, sessions, migrations
    evaluation/    Golden dataset models, runner, metrics
    generation/    LLM client, prompts, answer generation, citation guard
    graph/         LangGraph RAG state, nodes, workflow
    ingestion/     loaders, extraction, chunking, embedding indexing
    models/        SQLAlchemy ORM models
    repositories/  Persistence helpers
    retrieval/     Vector, BM25, hybrid retrieval, diagnostics
    reranking/     Cross-encoder reranker
    schemas/       Pydantic API schemas
    storage/       File-storage abstraction and local implementation
    workers/       Durable job worker
frontend/
  src/             React workspace, API client, components, tests
scripts/           Database initialization scripts
data/              Development uploaded and generated files
docs/              Architecture and technical design documents
```

## Security Notes

- Do not commit `.env`, OAuth secrets, OpenAI keys, or production JWT secrets.
- Use `AUTH_COOKIE_SECURE=true` behind HTTPS.
- Keep debug endpoints disabled in untrusted environments.
- Replace development database credentials before deploying outside a controlled development environment.
- Scope user data through authenticated user IDs on every document, chat, dataset, and evaluation route.
