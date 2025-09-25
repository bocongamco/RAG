<<<<<<< HEAD
import React, { useState } from "react";

type DocHit = {
  row?: number | string;
  name?: string;
  preview?: string;
=======
import { useEffect, useMemo, useRef, useState } from "react";

type Doc = {
  row?: string | number | null;
  name?: string | null;
  preview?: string | null;
>>>>>>> origin/main
};

type Msg = {
  id: string;
<<<<<<< HEAD
  user: string;
  answer?: string;
  docs: DocHit[];
  mode: "dense" | "bm25" | "hybrid";
  k: number;
  alpha?: number;
};

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export default function App() {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<"dense" | "bm25" | "hybrid">("hybrid");
  const [alpha, setAlpha] = useState(0.6);
  const [k, setK] = useState(5);
  const [busy, setBusy] = useState(false);
  const [messages, setMessages] = useState<Msg[]>([]);

  async function ask() {
    const q = question.trim();
    if (!q || busy) return;

    setBusy(true);

    // create a stable id for this turn
    const id = (crypto?.randomUUID?.() ?? String(Date.now()));

    // push a placeholder *with* the mode/k/alpha used for this query
    const placeholder: Msg = { id, user: q, answer: undefined, docs: [], mode, k, alpha: mode === "hybrid" ? alpha : undefined };
    setMessages((m) => [placeholder, ...m]);

    try {
      const payload: any = { question: q, mode, k };
      if (mode === "hybrid") payload.alpha = alpha;

      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      let data: { answer: string; documents: DocHit[] } | null = null;
      try { data = await res.json(); } catch { data = null; }

      if (!res.ok || !data) {
        // update only the message with this id
        setMessages((m) =>
          m.map((msg) =>
            msg.id === id
              ? { ...msg, answer: "Answer: Sorry, I couldn’t reach the API.\nCitations: (none)", docs: [] }
              : msg
          )
        );
      } else {
        setMessages((m) =>
          m.map((msg) =>
            msg.id === id ? { ...msg, answer: data!.answer, docs: data!.documents || [] } : msg
          )
        );
      }
    } catch {
      setMessages((m) =>
        m.map((msg) =>
          msg.id === id
            ? { ...msg, answer: "Answer: Sorry, I couldn’t reach the API.\nCitations: (none)", docs: [] }
            : msg
        )
      );
    } finally {
      setBusy(false);
      setQuestion("");
    }
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") ask();
  }

  return (
    <div style={{ minHeight: "100vh", background: "#0f1115", color: "#fff" }}>
      <header style={{ padding: "12px 16px", borderBottom: "1px solid #2a2f3a" }}>
        <strong>💻 Laptop Shopping RAG (local)</strong>
      </header>

      <main style={{ maxWidth: 980, margin: "0 auto", padding: 16 }}>
        {messages.map((msg) => (
          <Bubble key={msg.id} msg={msg} />
        ))}
      </main>

      <footer
        style={{
          position: "sticky",
          bottom: 0,
          left: 0,
          right: 0,
          borderTop: "1px solid #2a2f3a",
          background: "#0f1115",
        }}
      >
        <div
          style={{
            maxWidth: 980,
            margin: "0 auto",
            display: "flex",
            gap: 8,
            alignItems: "center",
            padding: 12,
          }}
        >
          <label style={{ color: "#c7c7c7" }}>
            Mode{" "}
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as any)}
              style={{ background: "#141822", color: "#fff", border: "1px solid #2a2f3a", padding: "4px 6px" }}
            >
              <option value="dense">Dense</option>
              <option value="bm25">BM25</option>
              <option value="hybrid">Hybrid</option>
            </select>
          </label>

          {mode === "hybrid" && (
            <label style={{ color: "#c7c7c7", display: "flex", alignItems: "center", gap: 6 }}>
              α
              <input
                type="number"
                step={0.1}
                min={0}
                max={1}
                value={alpha}
                onChange={(e) => setAlpha(Math.max(0, Math.min(1, Number(e.target.value))))}
                style={{
                  width: 70,
                  background: "#141822",
                  color: "#fff",
                  border: "1px solid #2a2f3a",
                  padding: "4px 6px",
                }}
              />
            </label>
          )}

          <label style={{ color: "#c7c7c7", display: "flex", alignItems: "center", gap: 6 }}>
            k
            <input
              type="number"
              min={1}
              max={20}
              value={k}
              onChange={(e) => setK(Math.max(1, Math.min(20, Number(e.target.value))))}
              style={{
                width: 70,
                background: "#141822",
                color: "#fff",
                border: "1px solid #2a2f3a",
                padding: "4px 6px",
              }}
            />
          </label>

          <input
            placeholder="Type your question and press Enter..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={onKeyDown}
            style={{
              flex: 1,
              background: "#141822",
              color: "#fff",
              border: "1px solid #2a2f3a",
              padding: "10px 12px",
              borderRadius: 8,
            }}
          />

          <button
            onClick={ask}
            disabled={busy}
            style={{
              background: busy ? "#2a2f3a" : "#2563eb",
              color: "#fff",
              padding: "10px 16px",
              border: "none",
              borderRadius: 8,
              cursor: busy ? "not-allowed" : "pointer",
              minWidth: 70,
            }}
          >
            {busy ? "Thinking…" : "Ask"}
=======
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
>>>>>>> origin/main
          </button>
        </div>
      </footer>
    </div>
  );
}

<<<<<<< HEAD
function Bubble({ msg }: { msg: Msg }) {
  const hasAnswer = !!msg.answer;
  const modeLabel =
    msg.mode === "hybrid"
      ? `Hybrid (α=${(msg.alpha ?? 0.6).toFixed(1)})`
      : msg.mode.toUpperCase();

  return (
    <div style={{ marginBottom: 16 }}>
      {/* User bubble */}
      <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 8 }}>
        <div
          style={{
            background: "#1e40af",
            color: "#fff",
            borderRadius: 12,
            padding: "10px 14px",
            maxWidth: 480,
          }}
        >
          {msg.user}
        </div>
      </div>

      {/* Model bubble */}
      <div style={{ display: "flex", justifyContent: "flex-start" }}>
        <div
          style={{
            background: "#151924",
            color: "#fff",
            border: "1px solid #2a2f3a",
            borderRadius: 12,
            padding: 14,
            maxWidth: 680,
            width: "100%",
          }}
        >
          <div style={{ whiteSpace: "pre-wrap" }}>{hasAnswer ? msg.answer : "Thinking…"}</div>

          {msg.docs && msg.docs.length > 0 && (
            <details style={{ marginTop: 10 }} open>
              <summary style={{ cursor: "pointer", color: "#c7c7c7" }}>
                Retrieved context — {modeLabel}, k={msg.k} (showing {msg.docs.length})
              </summary>

              <div style={{ marginTop: 10, display: "grid", gap: 10 }}>
                {msg.docs.map((d, i) => (
                  <div
                    key={i}
                    style={{
                      border: "1px solid #2a2f3a",
                      borderRadius: 8,
                      padding: 10,
                      background: "#0f1320",
                    }}
                  >
                    <div style={{ fontWeight: 600, marginBottom: 6 }}>
                      row={d.row}, name="{d.name}"
                    </div>
                    <div
                      style={{
                        color: "#bdbdbd",
                        whiteSpace: "pre-wrap",
                        wordBreak: "break-word",
                        hyphens: "none",
                      }}
                    >
                      {(d.preview ?? "").slice(0, 800)}
                    </div>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
=======
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
>>>>>>> origin/main
      </div>
    </div>
  );
}
<<<<<<< HEAD
=======

function mkId() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}
>>>>>>> origin/main
