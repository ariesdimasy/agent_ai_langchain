import { useEffect, useRef, useState } from "react";
import {
  BookOpen,
  Check,
  ChevronRight,
  FileText,
  Lightbulb,
  LoaderCircle,
  MessageCircle,
  Send,
  Sparkles,
  Upload,
  X,
} from "lucide-react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const MAX_FILE_SIZE = 20 * 1024 * 1024;
const SUMMARY_TIMEOUT_MS = 12 * 60 * 1000 + 10_000;

function formatSize(bytes) {
  return bytes < 1024 * 1024
    ? `${Math.max(1, Math.round(bytes / 1024))} KB`
    : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatElapsed(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(remainingSeconds).padStart(2, "0")}`;
}

export default function App() {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadingSeconds, setLoadingSeconds] = useState(0);
  const [summaryProgress, setSummaryProgress] = useState({
    percent: 0,
    message: "Menyiapkan PDF",
  });
  const [chatLoading, setChatLoading] = useState(false);
  const [question, setQuestion] = useState("");
  const [chatMessages, setChatMessages] = useState([]);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  useEffect(() => {
    if (!loading) {
      setLoadingSeconds(0);
      return undefined;
    }
    const timer = window.setInterval(() => {
      setLoadingSeconds((seconds) => seconds + 1);
    }, 1000);
    return () => window.clearInterval(timer);
  }, [loading]);

  function chooseFile(nextFile) {
    if (loading || chatLoading || !nextFile) return;
    setError("");
    setResult(null);
    setChatMessages([]);
    setQuestion("");
    if (!nextFile.name.toLowerCase().endsWith(".pdf")) {
      setFile(null);
      setError("Pilih dokumen dengan format PDF.");
      return;
    }
    if (nextFile.size > MAX_FILE_SIZE) {
      setFile(null);
      setError("Ukuran file maksimal 20 MB.");
      return;
    }
    setFile(nextFile);
  }

  async function handleAsk(event) {
    event.preventDefault();
    const trimmedQuestion = question.trim();
    if (!file || !trimmedQuestion || chatLoading) return;

    setChatLoading(true);
    setError("");
    const formData = new FormData();
    formData.append("file", file);
    formData.append("question", trimmedQuestion);
    formData.append("history", JSON.stringify(chatMessages.slice(-12)));

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Pertanyaan gagal dijawab. Silakan coba lagi.");
      }
      setChatMessages((messages) => [
        ...messages,
        { role: "user", content: trimmedQuestion },
        {
          role: "assistant",
          content: data.answer,
          sources: data.sources || [],
        },
      ]);
      setQuestion("");
    } catch (requestError) {
      setError(
        requestError.message ||
          "Tidak dapat terhubung ke server. Pastikan backend sedang berjalan.",
      );
    } finally {
      setChatLoading(false);
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (!file || loading) return;

    setLoading(true);
    setError("");
    setResult(null);
    setSummaryProgress({ percent: 0, message: "Menyiapkan PDF" });
    const formData = new FormData();
    formData.append("file", file);
    const controller = new AbortController();
    const timeout = window.setTimeout(
      () => controller.abort(),
      SUMMARY_TIMEOUT_MS,
    );

    try {
      const response = await fetch(`${API_URL}/api/summarize/stream`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "Rangkuman gagal dibuat. Silakan coba lagi.");
      }
      if (!response.body) {
        throw new Error("Server tidak mengirimkan stream progres.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let completed = false;
      while (!completed) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

        let eventBoundary = buffer.indexOf("\n\n");
        while (eventBoundary !== -1) {
          const eventText = buffer.slice(0, eventBoundary);
          buffer = buffer.slice(eventBoundary + 2);
          const dataLine = eventText
            .split("\n")
            .find((line) => line.startsWith("data: "));
          if (dataLine) {
            const event = JSON.parse(dataLine.slice(6));
            if (event.type === "progress") {
              setSummaryProgress({
                percent: event.percent,
                message: event.message,
              });
            } else if (event.type === "error") {
              throw new Error(event.message || "Rangkuman gagal dibuat.");
            } else if (event.type === "complete") {
              setSummaryProgress({ percent: 100, message: "Rangkuman selesai" });
              setResult(event.result);
              completed = true;
              break;
            }
          }
          eventBoundary = buffer.indexOf("\n\n");
        }
      }
      if (!completed) {
        throw new Error("Koneksi progres terputus sebelum rangkuman selesai.");
      }
    } catch (requestError) {
      setError(
        requestError.name === "AbortError"
          ? "Proses melewati batas waktu 12 menit dan dibatalkan. Tombol rangkum sudah aktif kembali; coba PDF yang lebih pendek."
          : requestError.message ||
            "Tidak dapat terhubung ke server. Pastikan backend sedang berjalan.",
      );
    } finally {
      window.clearTimeout(timeout);
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Rangkumin beranda">
          <span className="brand-mark"><BookOpen size={19} strokeWidth={2.2} /></span>
          <span>rangkumin<span className="brand-period">.</span></span>
        </a>
        <div className="topbar-note"><span className="status-dot" /> Asisten baca berbasis AI</div>
      </header>

      <main>
        <section className="hero">
          <div className="eyebrow"><Sparkles size={14} /> BACA LEBIH CERDAS</div>
          <h1>Buku tebal.<br /><span>Inti yang jelas.</span></h1>
          <p className="hero-copy">
            Unggah buku PDF untuk membuat rangkuman atau bertanya langsung
            tentang isinya.
          </p>
        </section>

        <section className="workspace" aria-label="Pembuat rangkuman">
          <form className="upload-card" onSubmit={handleSubmit}>
            <div className="card-heading">
              <div>
                <div className="step-label">LANGKAH 01</div>
                <h2>Mulai dari sebuah buku</h2>
              </div>
              <span className="heading-icon"><FileText size={19} /></span>
            </div>

            <div
              className={`dropzone${dragging ? " is-dragging" : ""}${file ? " has-file" : ""}`}
              onDragOver={(event) => {
                event.preventDefault();
                if (!loading && !chatLoading) setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragging(false);
                if (loading || chatLoading) return;
                chooseFile(event.dataTransfer.files[0]);
              }}
            >
              <input
                ref={inputRef}
                type="file"
                accept=".pdf,application/pdf"
                hidden
                disabled={loading || chatLoading}
                onChange={(event) => {
                  chooseFile(event.target.files[0]);
                  event.target.value = "";
                }}
              />
              {file ? (
                <div className="file-selected">
                  <span className="file-icon"><FileText size={21} /></span>
                  <span className="file-details">
                    <strong>{file.name}</strong>
                    <small>{formatSize(file.size)} · PDF</small>
                  </span>
                  <button
                    type="button"
                    className="remove-file"
                    aria-label="Hapus file"
                    disabled={loading || chatLoading}
                    onClick={() => { setFile(null); setResult(null); setChatMessages([]); setQuestion(""); setError(""); }}
                  ><X size={17} /></button>
                </div>
              ) : (
                <div className="drop-content">
                  <span className="upload-icon"><Upload size={21} /></span>
                  <div>
                    <strong>Tarik PDF ke sini</strong>
                    <span>atau <button type="button" className="browse-button" disabled={loading || chatLoading} onClick={() => inputRef.current?.click()}>pilih file</button> dari perangkat</span>
                  </div>
                </div>
              )}
              {!file && <div className="file-hint">PDF · Maksimal 20 MB</div>}
            </div>

            {error && <div className="error-message" role="alert">{error}</div>}

            <button className="submit-button" type="submit" disabled={!file || loading}>
              {loading ? (
                <><LoaderCircle className="spin" size={18} /> Memproses {formatElapsed(loadingSeconds)}</>
              ) : (
                <>Buat rangkuman <ChevronRight size={18} /></>
              )}
            </button>
            <p className="privacy-note"><Check size={13} /> File diproses saat diminta dan tidak disimpan oleh aplikasi.</p>
          </form>

          {file && (
            <section className="chat-card" aria-label="Tanya jawab tentang buku">
              <div className="chat-heading">
                <span className="chat-heading-icon"><MessageCircle size={18} /></span>
                <div>
                  <div className="step-label">TANYA JAWAB</div>
                  <h2>Tanyakan isi buku</h2>
                </div>
              </div>
              <div className="chat-messages" aria-live="polite">
                {chatMessages.length === 0 ? (
                  <p className="chat-empty">Ajukan pertanyaan. Jawaban akan merujuk ke halaman yang relevan.</p>
                ) : (
                  chatMessages.map((message, index) => (
                    <article className={`chat-message ${message.role}`} key={`${index}-${message.role}`}>
                      <div className="message-content">{message.content}</div>
                      {message.sources?.length > 0 && (
                        <small className="message-sources">
                          Sumber: halaman {message.sources.join(", ")}
                        </small>
                      )}
                    </article>
                  ))
                )}
                {chatLoading && (
                  <div className="chat-loading" role="status">
                    <LoaderCircle className="spin" size={15} /> Sedang mencari di dalam buku...
                  </div>
                )}
              </div>
              <form className="chat-form" onSubmit={handleAsk}>
                <textarea
                  aria-label="Pertanyaan tentang buku"
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder="Contoh: Apa gagasan utama pada bab pertama?"
                  maxLength={2000}
                  rows={2}
                  disabled={chatLoading}
                />
                <button
                  type="submit"
                  className="chat-send"
                  aria-label="Kirim pertanyaan"
                  disabled={!question.trim() || chatLoading}
                >
                  {chatLoading ? <LoaderCircle className="spin" size={17} /> : <Send size={17} />}
                </button>
              </form>
            </section>
          )}

          <aside className="info-card">
            <div className="info-icon"><Lightbulb size={18} /></div>
            <div>
              <strong>Jawaban dengan rujukan halaman.</strong>
              <p>Agent mencari bagian relevan dalam PDF dan mencantumkan halaman sumber agar mudah diperiksa.</p>
            </div>
          </aside>

          {loading && (
            <div className="progress-card" role="status">
              <div className="progress-mark"><LoaderCircle className="spin" size={21} /></div>
              <div>
                <strong>Menganalisis PDF dan menunggu rangkuman AI...</strong>
                <p>
                  {summaryProgress.message} · {summaryProgress.percent}% · {formatElapsed(loadingSeconds)}
                </p>
                <progress
                  className="summary-progress"
                  max="100"
                  value={summaryProgress.percent}
                  aria-label={`Progres rangkuman ${summaryProgress.percent}%`}
                />
              </div>
            </div>
          )}

          {result && (
            <section className="result-card" aria-live="polite">
              <div className="result-header">
                <div>
                  <div className="step-label">RANGKUMAN SELESAI</div>
                  <h2>{result.filename}</h2>
                  <p className="result-meta">{result.pages} halaman <span>·</span> {result.chunks} bagian dianalisis</p>
                </div>
                <span className="result-sparkle"><Sparkles size={19} /></span>
              </div>
              <div className="summary-text">{result.summary}</div>
              <button className="secondary-button" onClick={() => { setResult(null); setFile(null); setChatMessages([]); setQuestion(""); }}>
                Rangkum buku lain <ChevronRight size={16} />
              </button>
            </section>
          )}
        </section>
      </main>

      <footer><span>Rangkumin</span><span>Ringkas waktu, tangkap intinya.</span></footer>
    </div>
  );
}
