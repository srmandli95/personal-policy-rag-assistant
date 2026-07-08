# RAG concurrency tests

Run these stages against a non-production environment containing an embedded
test document and a valid test-user session cookie:

```bash
LOAD_TEST_USER_ID='test-user-uuid' pipenv run locust -f load_tests/locustfile.py \
  --headless --host http://localhost:8000 --users 10 --spawn-rate 2 --run-time 5m
```

Repeat with `--users 25`, then `--users 50`. Record p50/p95/p99 response time,
failure rate, API/worker CPU and memory, PostgreSQL connection waits, queued job
depth, and OpenAI rate-limit responses. A stage passes only when the error rate
is below 1%, p95 is within the product SLO, memory remains stable, and the job
queue drains after traffic stops.

The profile reads `JWT_SECRET_KEY` from the repository `.env` and creates a
short-lived local test token when `LOAD_TEST_USER_ID` is supplied. Alternatively,
provide a complete token through `LOAD_TEST_COOKIE_VALUE`.
