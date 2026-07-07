export type DisplayStatus =
  | "uploading"
  | "validating"
  | "extracting"
  | "chunking"
  | "embedding"
  | "ready"
  | "failed";

export interface DocumentRecord {
  document_id: string;
  original_file_name: string;
  file_size_bytes: number;
  status: string;
  display_status: DisplayStatus;
  failure_reason?: string | null;
  created_at?: string | null;
}

export interface Citation {
  document_id?: string;
  document_name?: string;
  source?: string;
  page_number?: number | null;
  chunk_id?: string;
  [key: string]: unknown;
}

export interface ChatMessage {
  message_id?: string;
  question: string;
  answer?: string | null;
  citations: Citation[];
  is_pending?: boolean;
}

export interface ChatSession {
  session_id: string;
  title: string;
  updated_at?: string;
}

export interface AuthUser {
  id: string;
  email: string;
  full_name?: string | null;
  auth_provider: string;
}

export interface EvaluationCase {
  id: string;
  category: string;
  question: string;
  expected_answer_contains: string[];
  expected_citation_document_contains: string[];
  expected_document_ids: string[];
  expected_refusal: boolean;
}

export interface EvaluationDataset {
  dataset_id: string;
  name: string;
  original_file_name: string;
  case_count: number;
  document_ids: string[];
  created_at: string;
  cases?: EvaluationCase[];
}

export interface EvaluationCaseResult {
  id: string;
  question: string;
  status: string;
  passed: boolean;
  answer: string;
  validation_status?: string | null;
  expected_refusal: boolean;
  actual_refusal: boolean;
  citation_count: number;
  evidence_chunk_count: number;
  checks: Record<string, boolean>;
  failure_reasons: string[];
}

export interface EvaluationRun {
  run_id: string;
  dataset_id: string;
  status: string;
  total: number;
  passed: number;
  failed: number;
  pass_rate: number;
  settings: Record<string, unknown>;
  results: EvaluationCaseResult[];
  error_message?: string | null;
  started_at: string;
  completed_at?: string | null;
}
