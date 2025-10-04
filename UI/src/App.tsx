import React, { useState, useEffect, useRef } from "react";

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

// If you change footer height, update this too
const FOOTER_PX = 84; // approx height of input bar incl. padding

export default function App() {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<"dense" | "bm25" | "hybrid">("hybrid");
  const [alpha, setAlpha] = useState(0.6);
  const [k, setK] = useState(5);
  const [busy, setBusy] = useState(false);
  const [messages, setMessages] = useState<Msg[]>([]);

  // Auto-scroll anchor
  const bottomRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function ask() {
    const q = question.trim();
    if (!q || busy) return;

    setBusy(true);

    const id = (crypto?.randomUUID?.() ?? String(Date.now()));

    const placeholder: Msg = {
      id,
      user: q,
      answer: undefined,
      docs: [],
      mode,
      k,
      alpha: mode === "hybrid" ? alpha : undefined,
    };
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
      try {
        data = await res.json();
      } catch {
        data = null;
      }

      if (!res.ok || !data) {
        setMessages((m) =>
          m.map((msg) =>
            msg.id === id
              ? {
                  ...msg,
                  answer:
                    "Answer: Sorry, I couldn’t reach the API.\nCitations: (none)",
                  docs: [],
                }
              : msg
          )
        );
      } else {
        setMessages((m) =>
          m.map((msg) =>
            msg.id === id
              ? { ...msg, answer: data!.answer, docs: data!.documents || [] }
              : msg
          )
        );
      }
    } catch {
      setMessages((m) =>
        m.map((msg) =>
          msg.id === id
            ? {
                ...msg,
                answer:
                  "Answer: Sorry, I couldn’t reach the API.\nCitations: (none)",
                docs: [],
              }
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
    <div
      style={{
        minHeight: "100vh",
        background: "#0f1115",
        color: "#fff",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <header
        style={{ padding: "12px 16px", borderBottom: "1px solid #2a2f3a" }}
      >
        <strong>💻 Laptop Shopping RAG (local)</strong>
      </header>

      <main
        style={{
          maxWidth: 980,
          margin: "0 auto",
          padding: 16,
          paddingBottom: FOOTER_PX + 24, // keep last bubble visible above fixed footer
          flex: "1 1 auto",
          width: "100%",
          display: "flex",
          flexDirection: "column",
        }}
      >
        {/* Show oldest at top, newest at bottom */}
        {[...messages].reverse().map((msg) => (
          <Bubble key={msg.id} msg={msg} />
        ))}
        <div ref={bottomRef} />
      </main>

      {/* Fixed footer (always visible) */}
      <footer
        style={{
          position: "fixed",
          left: 0,
          right: 0,
          bottom: 0,
          zIndex: 10,
          borderTop: "1px solid #2a2f3a",
          background: "#0f1115",
          paddingBottom: "env(safe-area-inset-bottom)", // iOS notch safety
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

         

          <label
            style={{
              color: "#c7c7c7",
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            
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
      ? `Hybrid (α=${(msg.alpha ?? 0.6).toFixed(1)})`
      : msg.mode.toUpperCase();

  return (
    <div style={{ marginBottom: 16 }}>
      {/* User bubble */}
      <div
        style={{ display: "flex", justifyContent: "flex-end", marginBottom: 8 }}
      >
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
          <div style={{ whiteSpace: "pre-wrap" }}>
            {/* {hasAnswer ? msg.answer : "Thinking…"} */}
            {hasAnswer ? msg.answer?.split("\n")[0] : "Thinking…"}  {/* Only show the first line (the answer) */}
        
          </div>

          
        </div>
      </div>
    </div>
  );
}
