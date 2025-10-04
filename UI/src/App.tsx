import React, { useEffect, useState } from "react";

type DocHit = {
  row?: number | string;
  name?: string;
  preview?: string;
};

type Msg = {
  id: string;
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

  // >>> NEW: pre-fill alpha from the backend (learned value in summary.json) <<<
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/alpha`);
        if (!res.ok) return;
        const data = await res.json();
        if (typeof data?.alpha === "number") setAlpha(data.alpha);
      } catch { /* ignore */ }
    })();
  }, []);

  async function ask() {
    const q = question.trim();
    if (!q || busy) return;

    setBusy(true);

    const id = (crypto?.randomUUID?.() ?? String(Date.now()));
    const placeholder: Msg = { id, user: q, answer: undefined, docs: [], mode, k, alpha: mode === "hybrid" ? alpha : undefined };
    setMessages((m) => [placeholder, ...m]);

    try {
      const payload: any = { question: q, mode, k };
      if (mode === "hybrid") payload.alpha = alpha; // send what user sees

      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      let data: { answer: string; documents: DocHit[] } | null = null;
      try { data = await res.json(); } catch { data = null; }

      if (!res.ok || !data) {
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
                step={0.05}
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
                title="Hybrid weight: 0=BM25, 1=dense. Pre-filled from training summary."
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
          </button>
        </div>
      </footer>
    </div>
  );
}

function Bubble({ msg }: { msg: Msg }) {
  const hasAnswer = !!msg.answer;
  const modeLabel =
    msg.mode === "hybrid"
      ? `Hybrid (α=${(msg.alpha ?? 0.6).toFixed(2)})`
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
      </div>
    </div>
  );
}
