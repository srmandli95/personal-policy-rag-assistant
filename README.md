# Document RAG Framework

Document RAG is a local-first document question-answering application. Users can upload documents, wait for automatic extraction and indexing, ask questions with evidence-backed citations, and run reusable golden evaluation datasets against the RAG workflow.

## Features

- Google OAuth authentication with user-scoped documents and chat history
- PDF, DOCX, TXT, Markdown, and HTML ingestion
- Text extraction, chunking, local embeddings, and PostgreSQL/pgvector storage
- Hybrid vector and BM25 retrieval
- Cross-encoder reranking
- LangGraph orchestration with bounded retrieval and answer-generation retries
- Grounding verification and citation validation
- Persistent chat sessions
- Uploadable golden evaluation datasets with case-level results

## Architecture

```text
React + Nginx
      |
      v
FastAPI
      |
      v
LangGraph RAG workflow
      |
      +--> Local embedding model
      +--> BM25 retrieval
      +--> Cross-encoder reranker
      +--> OpenAI LLM
      |
      v
PostgreSQL + pgvector
```

The Docker Compose stack exposes:

| Service | Address | Purpose |
| --- | --- | --- |
| Frontend | `http://localhost:3000` | Browser application and API proxy |
| Backend | `http://localhost:8000` | FastAPI application |
| API documentation | `http://localhost:8000/docs` | OpenAPI/Swagger UI |
| PostgreSQL | `localhost:5432` | Database and vector store |

## Prerequisites

Install the following before starting:

- Git
- Docker Desktop with Docker Compose v2
- A Google Cloud OAuth 2.0 client
- An OpenAI API key
- Internet access during the first build and first model load

Recommended local resources:

- 8 GB RAM minimum
- 10 GB free disk space for Docker images, model caches, and uploaded documents

Node.js, Python, and PostgreSQL do not need to be installed on the host when using the recommended Docker workflow.

## 1. Clone The Repository

```bash
git clone https://github.com/srmandli95/document-rag-framework.git
cd document-rag-framework
```

If the repository is already available locally, run the remaining commands from its root directory, where `docker-compose.yml` and `Makefile` are located.

## 2. Create The Environment File

Copy the provided template:

```bash
cp .env.example .env
```

Generate a local JWT secret:

```bash
openssl rand -hex 32
```

Open `.env` and configure at least these values:

```dotenv
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL_NAME=gpt-4o-mini

JWT_SECRET_KEY=replace-with-the-generated-secret
AUTH_COOKIE_SECURE=false
FRONTEND_URL=http://localhost:3000

GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback
```

Do not commit `.env`. It contains credentials and application secrets.

### Important Defaults

The values in `.env.example` are configured for Docker Compose:

```dotenv
DATABASE_URL=postgresql+psycopg2://personal_policy_rag_assistant:personal_policy_rag_assistant@postgres:5432/personal_policy_rag_assistant
ASYNC_DATABASE_URL=postgresql+asyncpg://personal_policy_rag_assistant:personal_policy_rag_assistant@postgres:5432/personal_policy_rag_assistant

EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L6-v2

MAX_UPLOAD_SIZE_MB=25
RAG_MAX_RETRIEVAL_ATTEMPTS=2
RAG_MAX_GENERATION_ATTEMPTS=2
```

Retry limits must be integers between `1` and `5`.

## 3. Configure Google OAuth

Google OAuth is required to sign in to the browser application.

1. Open the [Google Cloud Console](https://console.cloud.google.com/).
2. Create or select a Google Cloud project.
3. Configure the OAuth consent screen.
4. Create an OAuth client with application type **Web application**.
5. Add the following authorized redirect URI exactly:

   ```text
   http://localhost:8000/auth/google/callback
   ```

6. Copy the generated client ID and client secret into `.env`.

If the OAuth consent screen is in testing mode, add the Google accounts that will use the local application as test users.

The redirect URI in Google Cloud and `GOOGLE_REDIRECT_URI` must match exactly, including protocol, hostname, port, and path.

## 4. Build And Start The Application

Build all images and start the services in the background:

```bash
make build
```

Equivalent Docker command:

```bash
docker compose up -d --build
```

The first build can take several minutes because it installs Python, frontend, and machine-learning dependencies.

Check service status:

```bash
docker compose ps
```

All three services should be running. PostgreSQL should report `healthy`.

Verify the backend:

```bash
make health
```

Expected response:

```json
{"status":"ok","service":"personal-policy-rag-assistant"}
```

## 5. Open And Use The Application

Open:

```text
http://localhost:3000
```

Select **Continue with Google** and complete the OAuth flow.

### Add Documents

1. Open the **Knowledge Base** tab.
2. Upload a PDF, DOCX, TXT, Markdown, or HTML file.
3. Wait for its status to become `ready`.

Document processing runs through validation, extraction, chunking, and embedding. The first embedding operation may take longer while the local sentence-transformer model is downloaded and initialized.

### Ask Questions

1. Open the **Chat** tab.
2. Enter a question covered by the uploaded documents.
3. Review the generated answer and its source citations.

The LangGraph workflow can retry retrieval with a refined query when evidence is insufficient. It can also regenerate an unsupported answer using verifier feedback. When attempts are exhausted, the workflow returns a refusal instead of an unsupported answer.

### Run Evaluations

1. Open the **Evaluations** tab.
2. Copy the IDs of the ready documents referenced by the evaluation.
3. Download the JSON template.
4. Add supported and refusal cases.
5. Upload the completed JSON dataset.
6. Select the dataset and run the evaluation.

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
      "expected_document_ids": ["replace-with-a-ready-document-id"],
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

## Common Commands

Run these commands from the repository root:

| Command | Description |
| --- | --- |
| `make build` | Build images and start/recreate the complete stack |
| `make run` | Start existing images without rebuilding |
| `make stop` | Stop the Docker Compose stack |
| `make logs` | Follow logs from all services |
| `make health` | Call the backend health endpoint |
| `make test` | Run the complete backend test suite in Docker |

Useful Docker commands:

```bash
# Follow backend logs only
docker compose logs -f backend

# Follow frontend proxy logs only
docker compose logs -f frontend

# Restart one service
docker compose restart backend

# Rebuild one service
docker compose up -d --build frontend
```

## Run Tests

The services must be running before using the Make target:

```bash
make test
```

Run frontend tests and a production build directly when Node.js is installed locally:

```bash
cd frontend
npm ci
npm test -- --run
npm run build
```

## Data Persistence

PostgreSQL data is stored in the Docker volume `postgres_data`. Uploaded and generated document files are stored under the repository's `data/` directory through a bind mount.

Stopping services does not delete this data:

```bash
make stop
```

To remove containers and permanently delete the PostgreSQL volume:

```bash
docker compose down -v
```

This destructive command removes local users, document records, chunks, embeddings, chat history, evaluation datasets, and evaluation runs. Files under `data/` must be removed separately if a completely clean file store is required.

## Optional Debug Endpoints

Retrieval and evidence inspection endpoints are disabled by default. Enable them locally by adding these values to `.env`:

```dotenv
ENABLE_DEBUG_ENDPOINTS=true
ENABLE_RETRIEVAL_DEBUG_ENDPOINTS=true
```

Rebuild the backend after changing the flags:

```bash
docker compose up -d --build backend
```

Do not enable debug endpoints in an untrusted production environment.

## Troubleshooting

### Google reports a redirect URI mismatch

Confirm both locations use exactly:

```text
http://localhost:8000/auth/google/callback
```

Check the Google OAuth client configuration and `GOOGLE_REDIRECT_URI` in `.env`.

### Login reports that Google OAuth is not configured

Confirm these values are present and non-empty in `.env`:

```dotenv
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback
```

Then rebuild the backend:

```bash
docker compose up -d --build backend
```

### Answers fail because the OpenAI client is not configured

Set a valid `OPENAI_API_KEY` in `.env` and restart the backend:

```bash
docker compose restart backend
```

### The first query or document takes a long time

The local embedding and cross-encoder models are downloaded from Hugging Face on first use. Follow backend logs to monitor progress:

```bash
docker compose logs -f backend
```

### Upload returns HTTP 413

The backend enforces `MAX_UPLOAD_SIZE_MB`. The Nginx proxy also has a 250 MB request limit. Ensure the file is below both limits. After changing the backend limit, rebuild the backend.

### A document remains in processing or changes to failed

Inspect the background processing logs:

```bash
docker compose logs --tail=300 backend
```

Look for the document name, processing job ID, or messages from extraction, chunking, and embedding.

### A required port is already in use

Check ports `3000`, `8000`, and `5432`:

```bash
lsof -i :3000
lsof -i :8000
lsof -i :5432
```

Stop the conflicting process or update the host-side port mapping in `docker-compose.yml`.

### Docker is using an outdated image

Force a rebuild:

```bash
docker compose up -d --build
```

## Project Layout

```text
backend/
  app/
    api/           FastAPI routes
    evaluation/    RAG evaluation metrics and runner
    generation/    prompts, LLM client, and citation guard
    graph/         LangGraph state, nodes, and routing
    ingestion/     extraction, chunking, and indexing pipeline
    models/        SQLAlchemy database models
    repositories/  persistence operations
    retrieval/     vector, BM25, and hybrid retrieval
    reranking/     cross-encoder reranking
    schemas/       API request and response models
    storage/       file-storage implementations
frontend/
  src/             React application
scripts/           database initialization scripts
data/              local uploaded and generated files
```

## Security Notes

- Never commit `.env`, OAuth client secrets, OpenAI keys, or production JWT secrets.
- Use `AUTH_COOKIE_SECURE=true` behind HTTPS in production.
- Store production secrets in a secrets manager.
- Keep debug endpoints disabled in production.
- Replace the development database credentials before deploying outside a local environment.
