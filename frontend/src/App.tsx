import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "./api/client";
import { ChatPanel } from "./components/ChatPanel";
import { ChatHistoryPanel } from "./components/ChatHistoryPanel";
import { EvaluationWorkspace } from "./components/EvaluationWorkspace";
import { KnowledgeBaseWorkspace } from "./components/KnowledgeBaseWorkspace";
import type { ChatMessage, ChatSession, DocumentRecord } from "./types";

const activeStatuses = new Set(["uploading", "validating", "extracting", "chunking", "embedding"]);
const newChatKey = "__new_chat__";
type WorkspaceView = "chat" | "knowledge-base" | "evaluations";

export function workspaceViewFromHash(hash: string): WorkspaceView {
  if (hash === "#knowledge-base") return "knowledge-base";
  return hash === "#evaluations" ? "evaluations" : "chat";
}

export default function App() {
  const [view, setView] = useState<WorkspaceView>(() => workspaceViewFromHash(window.location.hash));
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [documentError, setDocumentError] = useState<string | null>(null);
  const [documentsLoading, setDocumentsLoading] = useState(true);
  const [messagesByChat, setMessagesByChat] = useState<Record<string, ChatMessage[]>>({});
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>();
  const [chatError, setChatError] = useState<string | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [historyCollapsed, setHistoryCollapsed] = useState(false);
  const activeChatKey = activeSessionId ?? newChatKey;
  const messages = messagesByChat[activeChatKey] ?? [];

  const loadDocuments = useCallback(async () => {
    try {
      setDocuments(await api.listDocuments());
      setDocumentError(null);
    } catch (error) {
      setDocumentError(error instanceof Error ? error.message : "Could not load documents");
    } finally {
      setDocumentsLoading(false);
    }
  }, []);

  const loadSessions = useCallback(async () => {
    try {
      setSessions(await api.listSessions());
      setHistoryError(null);
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : "Could not load chat sessions");
    }
  }, []);

  useEffect(() => {
    void loadDocuments();
    void loadSessions();
  }, [loadDocuments, loadSessions]);

  useEffect(() => {
    const syncViewFromUrl = () => setView(workspaceViewFromHash(window.location.hash));
    window.addEventListener("hashchange", syncViewFromUrl);
    return () => window.removeEventListener("hashchange", syncViewFromUrl);
  }, []);

  const navigateTo = (nextView: WorkspaceView) => {
    setView(nextView);
    window.location.hash = nextView;
  };

  const processing = documents.some((document) => activeStatuses.has(document.display_status));
  useEffect(() => {
    if (!processing) return;
    const interval = window.setInterval(() => void loadDocuments(), 2000);
    return () => window.clearInterval(interval);
  }, [processing, loadDocuments]);

  const readyDocumentCount = useMemo(
    () => documents.filter((document) => document.display_status === "ready").length,
    [documents],
  );

  const upload = async (file: File) => {
    setDocumentError(null);
    try {
      const uploaded = await api.uploadDocument(file);
      setDocuments((current) => [uploaded, ...current.filter((item) => item.document_id !== uploaded.document_id)]);
      await loadDocuments();
    } catch (error) {
      setDocumentError(error instanceof Error ? error.message : "Upload failed");
    }
  };

  const remove = async (document: DocumentRecord) => {
    setDocumentError(null);
    try {
      await api.deleteDocument(document.document_id);
      setDocuments((current) => current.filter((item) => item.document_id !== document.document_id));
    } catch (error) {
      setDocumentError(error instanceof Error ? error.message : "Delete failed");
    }
  };

  const selectSession = async (sessionId: string) => {
    setChatError(null);
    setActiveSessionId(sessionId);
    try {
      const sessionMessages = await api.getSession(sessionId);
      setMessagesByChat((current) => {
        const pendingMessages = (current[sessionId] ?? []).filter((message) => message.is_pending);

        return {
          ...current,
          [sessionId]: [...sessionMessages, ...pendingMessages],
        };
      });
    } catch (error) {
      setChatError(error instanceof Error ? error.message : "Could not load chat");
    }
  };

  const send = async (question: string) => {
    const pendingMessageId = `pending-${Date.now()}-${crypto.randomUUID()}`;
    const sessionId = activeSessionId;
    const chatKey = sessionId ?? newChatKey;

    setChatError(null);
    setMessagesByChat((current) => ({
      ...current,
      [chatKey]: [
        ...(current[chatKey] ?? []),
        {
          message_id: pendingMessageId,
          question,
          citations: [],
          is_pending: true,
        },
      ],
    }));
    try {
      const response = await api.ask(question, sessionId);
      setMessagesByChat((current) => {
        const updatedMessages = (current[chatKey] ?? []).map((message) => (
          message.message_id === pendingMessageId ? response : message
        ));

        if (!sessionId) {
          const { [newChatKey]: _newChatMessages, ...rest } = current;
          return {
            ...rest,
            [response.session_id]: updatedMessages,
          };
        }

        return {
          ...current,
          [chatKey]: updatedMessages,
        };
      });
      setActiveSessionId((current) => (current === sessionId ? response.session_id : current));
      await loadSessions();
    } catch (error) {
      setMessagesByChat((current) => ({
        ...current,
        [chatKey]: (current[chatKey] ?? []).filter((message) => message.message_id !== pendingMessageId),
      }));
      setChatError(error instanceof Error ? error.message : "Could not generate an answer");
    }
  };

  const newChat = () => {
    setActiveSessionId(undefined);
    setMessagesByChat((current) => ({
      ...current,
      [newChatKey]: [],
    }));
    setChatError(null);
  };

  const deleteSession = async (session: ChatSession) => {
    setHistoryError(null);
    try {
      await api.deleteSession(session.session_id);
      setSessions((current) => current.filter((item) => item.session_id !== session.session_id));
      setMessagesByChat((current) => {
        const { [session.session_id]: _deletedMessages, ...rest } = current;
        return rest;
      });
      if (activeSessionId === session.session_id) {
        newChat();
      }
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : "Could not delete chat");
    }
  };

  return (
    <div className="workspace-root">
      <nav className="workspace-tabs" aria-label="Workspace views">
        <button type="button" className={view === "chat" ? "active" : ""} onClick={() => navigateTo("chat")}>Chat</button>
        <button type="button" className={view === "knowledge-base" ? "active" : ""} onClick={() => navigateTo("knowledge-base")}>Knowledge Base</button>
        <button type="button" className={view === "evaluations" ? "active" : ""} onClick={() => navigateTo("evaluations")}>Evaluations</button>
      </nav>
      {view === "chat" ? (
      <div className={`app-shell ${historyCollapsed ? "history-collapsed" : ""}`}>
      <ChatHistoryPanel
        sessions={sessions}
        activeSessionId={activeSessionId}
        error={historyError}
        collapsed={historyCollapsed}
        onToggleCollapsed={() => setHistoryCollapsed((current) => !current)}
        onSelect={selectSession}
        onDelete={deleteSession}
        onNewChat={newChat}
      />
      <ChatPanel
        messages={messages}
        readyDocumentCount={readyDocumentCount}
        loading={messages.some((message) => message.is_pending)}
        error={chatError}
        onSend={send}
      />
      </div>
      ) : view === "knowledge-base" ? (
        <KnowledgeBaseWorkspace
          documents={documents}
          loading={documentsLoading}
          error={documentError}
          onUpload={upload}
          onDelete={remove}
        />
      ) : <EvaluationWorkspace documents={documents} />}
    </div>
  );
}
