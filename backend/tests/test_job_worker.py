from types import SimpleNamespace

from app.workers import job_worker


class FakeSession:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_run_one_job_processes_claimed_document_job(monkeypatch):
    db = FakeSession()
    job = SimpleNamespace(
        id="job-1",
        document_id="doc-1",
        user_id="user-1",
        force=False,
        steps=[],
    )
    document = SimpleNamespace(id="doc-1")
    captured = {}

    monkeypatch.setattr(job_worker, "SessionLocal", lambda: db)
    monkeypatch.setattr(job_worker, "_claim_document_job", lambda _db: job)

    def fake_get_document_by_id(**kwargs):
        captured["document_lookup"] = kwargs
        return document

    def fake_process_document(*args, **kwargs):
        captured["process_args"] = args
        captured["process_kwargs"] = kwargs

    monkeypatch.setattr(job_worker, "get_document_by_id", fake_get_document_by_id)
    monkeypatch.setattr(job_worker, "process_document", fake_process_document)

    assert job_worker.run_one_job() is True
    assert captured["document_lookup"] == {
        "db": db,
        "document_id": "doc-1",
        "user_id": "user-1",
    }
    assert captured["process_args"] == (db, document)
    assert captured["process_kwargs"] == {"force": False, "job": job}
    assert db.closed is True
