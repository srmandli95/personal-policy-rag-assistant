import type {
  AuthUser,
  ChatMessage,
  ChatSession,
  Citation,
  DocumentRecord,
  EvaluationDataset,
  EvaluationRun,
} from "../types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    credentials: "include",
    headers: {
      ...options.headers,
    },
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail || `Request failed (${response.status})`);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}

export const api = {
  async me(): Promise<AuthUser> {
    return request<AuthUser>("/auth/me");
  },

  async logout(): Promise<void> {
    await request("/auth/logout", { method: "POST" });
  },

  async googleLogin(): Promise<void> {
    const data = await request<{ authorization_url: string }>("/auth/google/login");
    window.location.assign(data.authorization_url);
  },

  async listDocuments(): Promise<DocumentRecord[]> {
    const data = await request<{ documents: DocumentRecord[] }>("/documents");
    return data.documents;
  },

  async uploadDocument(file: File): Promise<DocumentRecord> {
    const body = new FormData();
    body.append("file", file);
    body.append("category", "general");
    return request<DocumentRecord>("/documents/upload", { method: "POST", body });
  },

  async deleteDocument(documentId: string): Promise<void> {
    await request(`/documents/${documentId}`, { method: "DELETE" });
  },

  async listSessions(): Promise<ChatSession[]> {
    const data = await request<{ sessions: ChatSession[] }>("/chat/sessions");
    return data.sessions;
  },

  async getSession(sessionId: string): Promise<ChatMessage[]> {
    const data = await request<{ messages: ChatMessage[] }>(`/chat/sessions/${sessionId}`);
    return data.messages;
  },

  async deleteSession(sessionId: string): Promise<void> {
    await request(`/chat/sessions/${sessionId}`, { method: "DELETE" });
  },

  async ask(question: string, sessionId?: string): Promise<ChatMessage & { session_id: string }> {
    const data = await request<{
      message_id: string;
      session_id: string;
      question: string;
      answer: string;
      citations: Citation[];
    }>("/chat/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId }),
    });
    return data;
  },

  async listEvaluationDatasets(): Promise<EvaluationDataset[]> {
    const data = await request<{ datasets: EvaluationDataset[] }>("/evaluations/datasets");
    return data.datasets;
  },

  async uploadEvaluationDataset(file: File, name?: string): Promise<EvaluationDataset> {
    const body = new FormData();
    body.append("file", file);
    if (name?.trim()) body.append("name", name.trim());
    return request<EvaluationDataset>("/evaluations/datasets", { method: "POST", body });
  },

  async deleteEvaluationDataset(datasetId: string): Promise<void> {
    await request(`/evaluations/datasets/${datasetId}`, { method: "DELETE" });
  },

  async runEvaluationDataset(datasetId: string): Promise<EvaluationRun> {
    return request<EvaluationRun>(`/evaluations/datasets/${datasetId}/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
  },

  async listEvaluationRuns(datasetId?: string): Promise<EvaluationRun[]> {
    const query = datasetId ? `?dataset_id=${encodeURIComponent(datasetId)}` : "";
    const data = await request<{ runs: EvaluationRun[] }>(`/evaluations/runs${query}`);
    return data.runs;
  },
};
