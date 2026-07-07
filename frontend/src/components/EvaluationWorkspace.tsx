import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type { DocumentRecord, EvaluationDataset, EvaluationRun } from "../types";

interface Props {
  documents: DocumentRecord[];
}

function dateTime(value: string | null | undefined): string {
  return value ? new Date(value).toLocaleString() : "Pending";
}

function makeTemplate(documentId: string): string {
  return JSON.stringify({
    name: "Policy evaluation",
    cases: [
      {
        id: "supported_case_001",
        category: "general",
        question: "What does this document say about the covered topic?",
        expected_answer_contains: ["expected phrase"],
        expected_document_ids: [documentId || "PASTE_READY_DOCUMENT_ID"],
        expected_refusal: false,
      },
      {
        id: "refusal_case_001",
        category: "general",
        question: "Ask about something absent from the uploaded documents",
        expected_answer_contains: [],
        expected_document_ids: [],
        expected_refusal: true,
      },
    ],
  }, null, 2);
}

export function EvaluationWorkspace({ documents }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const copyFeedbackTimer = useRef<number | undefined>(undefined);
  const [datasets, setDatasets] = useState<EvaluationDataset[]>([]);
  const [runs, setRuns] = useState<EvaluationRun[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>();
  const [selectedRunId, setSelectedRunId] = useState<string>();
  const [name, setName] = useState("");
  const [file, setFile] = useState<File>();
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string>();
  const [copiedDocumentId, setCopiedDocumentId] = useState<string>();

  const readyDocuments = useMemo(
    () => documents.filter((document) => document.display_status === "ready"),
    [documents],
  );
  const selectedDataset = datasets.find((dataset) => dataset.dataset_id === selectedDatasetId);
  const datasetRuns = runs.filter((run) => run.dataset_id === selectedDatasetId);
  const selectedRun = runs.find((run) => run.run_id === selectedRunId) || datasetRuns[0];

  const refresh = async () => {
    try {
      const [nextDatasets, nextRuns] = await Promise.all([
        api.listEvaluationDatasets(),
        api.listEvaluationRuns(),
      ]);
      setDatasets(nextDatasets);
      setRuns(nextRuns);
      setSelectedDatasetId((current) => current || nextDatasets[0]?.dataset_id);
      setError(undefined);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Could not load evaluations");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
    return () => window.clearTimeout(copyFeedbackTimer.current);
  }, []);

  const copyDocumentId = async (documentId: string) => {
    try {
      await navigator.clipboard.writeText(documentId);
      setCopiedDocumentId(documentId);
      window.clearTimeout(copyFeedbackTimer.current);
      copyFeedbackTimer.current = window.setTimeout(() => setCopiedDocumentId(undefined), 2000);
    } catch {
      setError("Could not copy the document ID");
    }
  };

  const upload = async () => {
    if (!file) return;
    setUploading(true);
    setError(undefined);
    try {
      const dataset = await api.uploadEvaluationDataset(file, name);
      setDatasets((current) => [dataset, ...current]);
      setSelectedDatasetId(dataset.dataset_id);
      setSelectedRunId(undefined);
      setFile(undefined);
      setName("");
      if (inputRef.current) inputRef.current.value = "";
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : "Evaluation upload failed");
    } finally {
      setUploading(false);
    }
  };

  const run = async () => {
    if (!selectedDatasetId) return;
    setRunning(true);
    setError(undefined);
    try {
      const result = await api.runEvaluationDataset(selectedDatasetId);
      setRuns((current) => [result, ...current]);
      setSelectedRunId(result.run_id);
    } catch (runError) {
      setError(runError instanceof Error ? runError.message : "Evaluation run failed");
    } finally {
      setRunning(false);
    }
  };

  const remove = async (dataset: EvaluationDataset) => {
    if (!window.confirm(`Delete evaluation dataset "${dataset.name}" and its run history?`)) return;
    try {
      await api.deleteEvaluationDataset(dataset.dataset_id);
      const remaining = datasets.filter((item) => item.dataset_id !== dataset.dataset_id);
      setDatasets(remaining);
      setRuns((current) => current.filter((runResult) => runResult.dataset_id !== dataset.dataset_id));
      setSelectedDatasetId(remaining[0]?.dataset_id);
      setSelectedRunId(undefined);
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : "Could not delete evaluation dataset");
    }
  };

  const downloadTemplate = () => {
    const blob = new Blob([makeTemplate(readyDocuments[0]?.document_id || "")], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "golden-evaluation-template.json";
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main className="evaluation-workspace">
      <aside className="evaluation-sidebar">
        <div className="panel-heading">
          <div><p className="eyebrow">Quality</p><h2>Eval datasets</h2></div>
          <span className="count">{datasets.length}</span>
        </div>

        <section className="eval-upload">
          <label>
            Dataset name
            <input value={name} placeholder="Policy regression set" onChange={(event) => setName(event.target.value)} />
          </label>
          <input
            ref={inputRef}
            type="file"
            accept="application/json,.json"
            aria-label="Choose golden evaluation JSON"
            onChange={(event) => setFile(event.target.files?.[0])}
          />
          <button className="primary-button" type="button" disabled={!file || uploading} onClick={() => void upload()}>
            {uploading ? "Validating..." : "Upload dataset"}
          </button>
        </section>

        {error && <div className="error-banner" role="alert">{error}</div>}

        <div className="eval-dataset-list" aria-busy={loading}>
          {!loading && datasets.length === 0 && <p className="muted">No evaluation datasets yet.</p>}
          {datasets.map((dataset) => (
            <article className={`eval-dataset-item ${dataset.dataset_id === selectedDatasetId ? "active" : ""}`} key={dataset.dataset_id}>
              <button
                type="button"
                className="eval-dataset-select"
                onClick={() => { setSelectedDatasetId(dataset.dataset_id); setSelectedRunId(undefined); }}
              >
                <strong>{dataset.name}</strong>
                <span>{dataset.case_count} cases · {dateTime(dataset.created_at)}</span>
              </button>
              <button type="button" className="icon-button" aria-label={`Delete ${dataset.name}`} onClick={() => void remove(dataset)}>×</button>
            </article>
          ))}
        </div>
      </aside>

      <section className="evaluation-content">
        <header className="evaluation-header">
          <div><p className="eyebrow">RAG evaluation</p><h2>{selectedDataset?.name || "Evaluation workspace"}</h2></div>
          <button className="primary-button" type="button" disabled={!selectedDataset || running} onClick={() => void run()}>
            {running ? "Running..." : "Run evaluation"}
          </button>
        </header>

        <section className="eval-section">
          <div className="section-heading">
            <div><h3>Ready document IDs</h3><p className="muted">Use these IDs in supported evaluation cases.</p></div>
            <button className="secondary-button" type="button" onClick={downloadTemplate}>Download JSON template</button>
          </div>
          <div className="document-id-table">
            {readyDocuments.length === 0 && <p className="muted">No ready documents are available.</p>}
            {readyDocuments.map((documentRecord) => (
              <div className="document-id-row" key={documentRecord.document_id}>
                <strong>{documentRecord.original_file_name}</strong>
                <code>{documentRecord.document_id}</code>
                <button
                  className={`secondary-button compact-button copy-button ${copiedDocumentId === documentRecord.document_id ? "copied" : ""}`}
                  type="button"
                  aria-label={copiedDocumentId === documentRecord.document_id ? "Document ID copied" : `Copy document ID for ${documentRecord.original_file_name}`}
                  onClick={() => void copyDocumentId(documentRecord.document_id)}
                >
                  {copiedDocumentId === documentRecord.document_id ? "Copied" : "Copy"}
                </button>
              </div>
            ))}
          </div>
        </section>

        <section className="eval-section eval-format-section">
          <div><h3>Golden evaluation format</h3><p className="muted">Supported cases require document IDs. Refusal cases use an empty list.</p></div>
          <pre>{makeTemplate(readyDocuments[0]?.document_id || "")}</pre>
        </section>

        <section className="eval-section">
          <div className="section-heading">
            <div><h3>Run results</h3><p className="muted">{selectedDataset ? `${datasetRuns.length} runs for this dataset` : "Select a dataset"}</p></div>
            {datasetRuns.length > 0 && (
              <select aria-label="Evaluation run" value={selectedRun?.run_id || ""} onChange={(event) => setSelectedRunId(event.target.value)}>
                {datasetRuns.map((runResult) => <option value={runResult.run_id} key={runResult.run_id}>{dateTime(runResult.started_at)} · {runResult.pass_rate.toFixed(0)}%</option>)}
              </select>
            )}
          </div>

          {!selectedRun && <p className="muted eval-empty-results">Run the selected dataset to see case-level results.</p>}
          {selectedRun && (
            <>
              <div className="eval-summary-strip">
                <div><span>Pass rate</span><strong>{selectedRun.pass_rate.toFixed(1)}%</strong></div>
                <div><span>Passed</span><strong>{selectedRun.passed}</strong></div>
                <div><span>Failed</span><strong>{selectedRun.failed}</strong></div>
                <div><span>Total</span><strong>{selectedRun.total}</strong></div>
              </div>
              <div className="eval-case-results">
                {selectedRun.results.map((result) => (
                  <details className={`eval-case-result ${result.passed ? "passed" : "failed"}`} key={result.id}>
                    <summary>
                      <span className={`result-indicator ${result.passed ? "passed" : "failed"}`}>{result.passed ? "PASS" : "FAIL"}</span>
                      <strong>{result.id}</strong>
                      <span>{result.question}</span>
                    </summary>
                    <div className="eval-case-body">
                      {result.answer && <p>{result.answer}</p>}
                      <dl>
                        <div><dt>Citations</dt><dd>{result.citation_count}</dd></div>
                        <div><dt>Evidence chunks</dt><dd>{result.evidence_chunk_count}</dd></div>
                        <div><dt>Validation</dt><dd>{result.validation_status || "Not set"}</dd></div>
                      </dl>
                      {result.failure_reasons.length > 0 && <ul>{result.failure_reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul>}
                    </div>
                  </details>
                ))}
              </div>
            </>
          )}
        </section>
      </section>
    </main>
  );
}
