import { useEffect, useMemo, useRef, useState } from "react";

type Doc = {
  row?: string | number | null;
  name?: string | null;
  preview?: string | null;
};

type Msg = {
  id: string;
  role: "user" | "assistant";
  text: string;
  docs?: Doc[];
};

const API = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export default function App() {
  const [q, setQ] = useState("");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Msg[]>(() => {
    // restore last chat
    try {
      const raw = localStorage.getItem("rag_chat");
      return raw ? (JSON.parse(raw) as Msg[]) : [];
    } catch {
      return [];
    }
  });
  const listRef = useRef<HTMLDivElement>(null);

  // persist chat
  useEffect(() => {
    localStorage.setItem("rag_chat", JSON.stringify(messages));
  }, [messages]);

  // autoscroll to bottom
  useEffect(() => {
    const el = listRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, loading]);

  const ask = async () => {
    const question = q.trim();
    if (!question || loading) return;

    // add user's message
    const uid = mkId();
    const aid = mkId();
    setMessages((prev) => [
      ...prev,
      { id: uid, role: "user", text: question },
      { id: aid, role: "assistant", text: "Thinking…", docs: [] }, // placeholder
    ]);
    setQ("");
    setLoading(true);

    try {
      const res = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const data = await res.json();

      setMessages((prev) =>
        prev.map((m) =>
          m.id === aid
            ? {
                ...m,
                text: data?.answer ?? "Answer: I don't know\nCitations: (none)",
                docs: (data?.documents as Doc[]) || [],
              }
            : m
        )
      );
    } catch (e: any) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === aid
            ? {
                ...m,
                text:
                  "Answer: Sorry, I couldn’t reach the API.\nCitations: (none)",
                docs: [],
              }
            : m
        )
      );
    } finally {
      setLoading(false);
    }
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask();
    }
  };

  return (
    <div
      style={{
        height: "100dvh",
        maxHeight: "100vw",
        display: "grid",
        gridTemplateRows: "auto 1fr auto",
        gap: 0,
        fontFamily: "Inter, system-ui, sans-serif",
        background: "#0b0b0b",
        color: "#eaeaea",
      }}
    >
      {/* Header */}
      <header
        style={{
          padding: "16px 20px",
          borderBottom: "1px solid #1f1f1f",
        }}
      >
        <h1 style={{ margin: 0, fontSize: 22 }}>
          💻 Laptop Shopping RAG <span style={{ color: "#aaa" }}>(local)</span>
        </h1>
      </header>

      {/* Messages */}
      <div
        ref={listRef}
        style={{
          padding: "16px 20px",
          overflowY: "auto",
        }}
      >
        {messages.length === 0 && (
          <div style={{ opacity: 0.7 }}>
            Ask something like: <code>Price of Acer Aspire 5?</code> or{" "}
            <code>Best budget laptop</code>.
          </div>
        )}

        {messages.map((m) => (
          <Bubble key={m.id} msg={m} />
        ))}
      </div>

      {/* Composer */}
      <footer
        style={{
          padding: "12px 16px",
          borderTop: "1px solid #1f1f1f",
          background: "#0e0e0e",
        }}
      >
        <div style={{ display: "flex", gap: 8 }}>
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Type your question and press Enter…"
            disabled={loading}
            style={{
              flex: 1,
              padding: "12px 14px",
              borderRadius: 10,
              border: "1px solid #2a2a2a",
              background: "#111",
              color: "#eee",
              outline: "none",
            }}
          />
          <button
            onClick={ask}
            disabled={loading || !q.trim()}
            style={{
              padding: "12px 16px",
              borderRadius: 10,
              border: 0,
              background: loading ? "#333" : "#1f6feb",
              color: "white",
              cursor: loading ? "not-allowed" : "pointer",
              minWidth: 90,
            }}
          >
            {loading ? "Thinking…" : "Ask"}
          </button>
        </div>
      </footer>
    </div>
  );
}

/** Chat bubble component (supports context toggle for assistant) */
function Bubble({ msg }: { msg: Msg }) {
  const isUser = msg.role === "user";
  const align = isUser ? "flex-end" : "flex-start";
  const bg = isUser ? "#1f6feb" : "#151515";
  const color = isUser ? "white" : "#eaeaea";
  const border = isUser ? "1px solid #194f9b" : "1px solid #2a2a2a";

  return (
    <div style={{ display: "flex", justifyContent: align, margin: "8px 0" }}>
      <div
        style={{
          maxWidth: 760,
          whiteSpace: "pre-wrap",
          background: bg,
          color,
          border,
          padding: "10px 12px",
          borderRadius: 12,
          boxShadow: "0 1px 0 rgba(0,0,0,0.3)",
        }}
      >
        {msg.text}

        {!isUser && msg.docs && msg.docs.length > 0 && (
          <details style={{ marginTop: 8 }}>
            <summary style={{ cursor: "pointer" }}>Retrieved context</summary>
            <ul style={{ margin: "6px 0 0 16px", padding: 0 }}>
              {msg.docs.slice(0, 4).map((d, i) => (
                <li key={i} style={{ marginTop: 6 }}>
                  <strong>
                    row={d.row ?? "?"}, name="{d.name ?? ""}"
                  </strong>
                  <div style={{ color: "#bdbdbd" }}>
                    {(d.preview ?? "").slice(0, 300)}
                  </div>
                </li>
              ))}
            </ul>
          </details>
        )}
      </div>
    </div>
  );
}

function mkId() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}
