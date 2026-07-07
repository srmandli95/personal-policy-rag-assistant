import { useRef, useState } from "react";
import type { DocumentRecord } from "../types";

interface Props {
  documents: DocumentRecord[];
  loading: boolean;
  error: string | null;
  onUpload: (file: File) => Promise<void>;
  onDelete: (document: DocumentRecord) => Promise<void>;
}

const allowedExtensions = ".pdf,.docx,.txt,.md,.html";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function KnowledgeBaseWorkspace({ documents, loading, error, onUpload, onDelete }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);

  const uploadFirst = async (files: FileList | null) => {
    const file = files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      await onUpload(file);
      if (inputRef.current) inputRef.current.value = "";
    } finally {
      setUploading(false);
    }
  };

  const confirmDelete = async (document: DocumentRecord) => {
    if (window.confirm(`Delete "${document.original_file_name}" and all generated data?`)) {
      await onDelete(document);
    }
  };

  const readyCount = documents.filter((document) => document.display_status === "ready").length;
  const processingCount = documents.filter((document) => !["ready", "failed"].includes(document.display_status)).length;

  return (
    <main className="knowledge-workspace">
      <header className="knowledge-header">
        <div>
          <p className="eyebrow">Knowledge base</p>
          <h2>Documents</h2>
        </div>
        <div className="knowledge-summary" aria-label="Document summary">
          <div><span>Total</span><strong>{documents.length}</strong></div>
          <div><span>Ready</span><strong>{readyCount}</strong></div>
          <div><span>Processing</span><strong>{processingCount}</strong></div>
        </div>
      </header>

      <section className="knowledge-upload-band">
        <div
          className={`knowledge-drop-zone ${dragging ? "dragging" : ""}`}
          onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            void uploadFirst(event.dataTransfer.files);
          }}
        >
          <div>
            <strong>Upload a document</strong>
            <span>PDF, DOCX, TXT, Markdown, or HTML</span>
          </div>
          <button type="button" className="primary-button" disabled={uploading} onClick={() => inputRef.current?.click()}>
            {uploading ? "Uploading..." : "Choose file"}
          </button>
          <input
            ref={inputRef}
            aria-label="Choose document"
            type="file"
            accept={allowedExtensions}
            hidden
            onChange={(event) => void uploadFirst(event.target.files)}
          />
        </div>
        {error && <div className="error-banner" role="alert">{error}</div>}
      </section>

      <section className="knowledge-documents" aria-busy={loading}>
        <div className="knowledge-table-header">
          <span>Document</span><span>Status</span><span>Size</span><span>Added</span><span>Actions</span>
        </div>
        {!loading && documents.length === 0 && <p className="muted knowledge-empty">No documents uploaded.</p>}
        {documents.map((document) => (
          <article className="knowledge-row" key={document.document_id}>
            <strong title={document.original_file_name}>{document.original_file_name}</strong>
            <span><span className={`status status-${document.display_status}`}>{document.display_status}</span></span>
            <span>{formatBytes(document.file_size_bytes)}</span>
            <span>{document.created_at ? new Date(document.created_at).toLocaleDateString() : "Just now"}</span>
            <button
              type="button"
              className="icon-button"
              aria-label={`Delete ${document.original_file_name}`}
              onClick={() => void confirmDelete(document)}
            >×</button>
            {document.failure_reason && <p className="failure-reason knowledge-failure">{document.failure_reason}</p>}
          </article>
        ))}
      </section>
    </main>
  );
}
