import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import type { DocumentRecord, EvaluationDataset, EvaluationRun } from "../types";
import { EvaluationWorkspace } from "./EvaluationWorkspace";

vi.mock("../api/client", () => ({
  api: {
    listEvaluationDatasets: vi.fn(),
    listEvaluationRuns: vi.fn(),
    uploadEvaluationDataset: vi.fn(),
    deleteEvaluationDataset: vi.fn(),
    runEvaluationDataset: vi.fn(),
  },
}));

const documents: DocumentRecord[] = [{
  document_id: "doc-1",
  original_file_name: "policy.pdf",
  file_size_bytes: 1200,
  status: "embedded",
  display_status: "ready",
}];

const dataset: EvaluationDataset = {
  dataset_id: "dataset-1",
  name: "Policy checks",
  original_file_name: "golden.json",
  case_count: 1,
  document_ids: ["doc-1"],
  created_at: "2026-07-06T12:00:00Z",
};

const run: EvaluationRun = {
  run_id: "run-1",
  dataset_id: "dataset-1",
  status: "completed",
  total: 1,
  passed: 1,
  failed: 0,
  pass_rate: 100,
  settings: {},
  results: [{
    id: "coverage-001",
    question: "Is emergency care covered?",
    status: "passed",
    passed: true,
    answer: "Emergency care is covered.",
    expected_refusal: false,
    actual_refusal: false,
    citation_count: 1,
    evidence_chunk_count: 1,
    checks: { citations_present: true },
    failure_reasons: [],
  }],
  started_at: "2026-07-06T12:10:00Z",
  completed_at: "2026-07-06T12:10:10Z",
};

describe("EvaluationWorkspace", () => {
  beforeEach(() => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: vi.fn().mockResolvedValue(undefined) },
    });
    vi.mocked(api.listEvaluationDatasets).mockResolvedValue([dataset]);
    vi.mocked(api.listEvaluationRuns).mockResolvedValue([]);
    vi.mocked(api.runEvaluationDataset).mockResolvedValue(run);
  });

  it("shows ready document IDs and runs a selected dataset", async () => {
    render(<EvaluationWorkspace documents={documents} />);

    expect((await screen.findAllByText("Policy checks")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("doc-1").length).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole("button", { name: "Run evaluation" }));

    await waitFor(() => expect(api.runEvaluationDataset).toHaveBeenCalledWith("dataset-1"));
    expect(await screen.findByText("100.0%")).toBeInTheDocument();
    expect(screen.getByText("coverage-001")).toBeInTheDocument();
  });

  it("uploads a selected JSON dataset", async () => {
    vi.mocked(api.uploadEvaluationDataset).mockResolvedValue({
      ...dataset,
      dataset_id: "dataset-2",
      name: "Uploaded checks",
    });
    render(<EvaluationWorkspace documents={documents} />);
    await screen.findAllByText("Policy checks");

    const file = new File(["[]"], "golden.json", { type: "application/json" });
    fireEvent.change(screen.getByLabelText("Choose golden evaluation JSON"), { target: { files: [file] } });
    fireEvent.click(screen.getByRole("button", { name: "Upload dataset" }));

    await waitFor(() => expect(api.uploadEvaluationDataset).toHaveBeenCalledWith(file, ""));
  });

  it("confirms when a document ID is copied", async () => {
    render(<EvaluationWorkspace documents={documents} />);
    await screen.findAllByText("Policy checks");

    fireEvent.click(screen.getByRole("button", { name: "Copy document ID for policy.pdf" }));

    await waitFor(() => expect(navigator.clipboard.writeText).toHaveBeenCalledWith("doc-1"));
    expect(screen.getByRole("button", { name: "Document ID copied" })).toHaveTextContent("Copied");
  });
});
