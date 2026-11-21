
import { useEffect, useRef, useState } from "react";
import "./App.css";

interface AnalysisResult {
  query?: string;
  intent?: string;
  destination?: any;
  reference_location?: any;
  spatial_relationships?: any[];
  ambiguity?: string;
  clarification_required?: boolean;
  clarification_question?: string;
  resolution_status?: string;
  ml_predictions?: any;
  ml_validation?: any;
}

type ApiStatus = "checking" | "online" | "offline";

function App() {
  // Navigation query
  const [query, setQuery] = useState("");
  const [language, setLanguage] = useState("Hinglish");
  const [result, setResult] = useState<AnalysisResult | null>(null);

  // Loading and errors
  const [loading, setLoading] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [error, setError] = useState("");

  // Voice recording
  const [recording, setRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [detectedLanguage, setDetectedLanguage] = useState("");

  // Backend status
  const [apiStatus, setApiStatus] = useState<ApiStatus>("checking");

  // References
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const discardRecordingRef = useRef(false);
  const transcriptionAbortRef = useRef<AbortController | null>(null);
  const analysisAbortRef = useRef<AbortController | null>(null);

  // Check backend health every 10 seconds
  useEffect(() => {
    let mounted = true;

    const checkBackend = async () => {
      try {
        const response = await fetch("http://localhost:8000/health");

        if (!response.ok) {
          throw new Error("Backend unavailable");
        }

        if (mounted) {
          setApiStatus("online");
        }
      } catch {
        if (mounted) {
          setApiStatus("offline");
        }
      }
    };

    checkBackend();

    const interval = window.setInterval(checkBackend, 10000);

    return () => {
      mounted = false;
      window.clearInterval(interval);
    };
  }, []);

  // Recording timer
  useEffect(() => {
    if (!recording) {
      setRecordingSeconds(0);
      return;
    }

    const interval = window.setInterval(() => {
      setRecordingSeconds((seconds) => seconds + 1);
    }, 1000);

    return () => window.clearInterval(interval);
  }, [recording]);

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      transcriptionAbortRef.current?.abort();
      analysisAbortRef.current?.abort();

      if (mediaRecorderRef.current?.state === "recording") {
        mediaRecorderRef.current.stop();
      }

      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  // Start microphone recording
  const startRecording = async () => {
    try {
      setError("");
      setResult(null);
      setDetectedLanguage("");

      if (apiStatus !== "online") {
        throw new Error("Backend is offline. Start the FastAPI server first.");
      }

      if (
        !navigator.mediaDevices?.getUserMedia ||
        !window.MediaRecorder
      ) {
        throw new Error("Your browser does not support audio recording.");
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });

      streamRef.current = stream;
      discardRecordingRef.current = false;
      audioChunksRef.current = [];

      const supportedTypes = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/ogg;codecs=opus",
        "audio/mp4",
      ];

      const mimeType = supportedTypes.find((type) =>
        MediaRecorder.isTypeSupported(type)
      );

      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);

      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        streamRef.current = null;

        // Ignore recordings intentionally cleared by the user
        if (discardRecordingRef.current) {
          discardRecordingRef.current = false;
          return;
        }

        const audioBlob = new Blob(audioChunksRef.current, {
          type: recorder.mimeType || "audio/webm",
        });

        if (!audioBlob.size) {
          setError("No audio was recorded. Please try again.");
          return;
        }

        if (audioBlob.size > 20 * 1024 * 1024) {
          setError("Recording exceeds the 20 MB upload limit.");
          return;
        }

        const extension = recorder.mimeType.includes("ogg")
          ? "ogg"
          : recorder.mimeType.includes("mp4")
            ? "mp4"
            : "webm";

        const formData = new FormData();

        formData.append(
          "audio",
          audioBlob,
          `recording.${extension}`
        );

        const controller = new AbortController();
        transcriptionAbortRef.current = controller;

        setTranscribing(true);
        setError("");

        try {
          const response = await fetch(
            "http://localhost:8000/api/transcribe",
            {
              method: "POST",
              body: formData,
              signal: controller.signal,
            }
          );

          if (!response.ok) {
            const message = await response.text();
            throw new Error(message || "Speech transcription failed.");
          }

          const data = await response.json();

          if (!data.transcription?.trim()) {
            throw new Error("No speech detected. Please try again.");
          }

          // Automatically populate the navigation query
          setQuery(data.transcription.trim());

          // Show Whisper's detected language separately
          setDetectedLanguage(data.language || "Unknown");
        } catch (err) {
          if (err instanceof Error && err.name === "AbortError") {
            return;
          }

          setError(
            err instanceof Error
              ? err.message
              : "Transcription failed. Please try again."
          );
        } finally {
          if (transcriptionAbortRef.current === controller) {
            transcriptionAbortRef.current = null;
            setTranscribing(false);
          }
        }
      };

      recorder.start();
      setRecording(true);
    } catch (err) {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;

      setRecording(false);

      setError(
        err instanceof Error
          ? err.message
          : "Microphone access failed. Check browser permissions."
      );
    }
  };

  // Stop microphone recording
  const stopRecording = () => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  // Analyze navigation query
  const analyzeQuery = async () => {
    if (!query.trim()) {
      setError("Please enter or speak a navigation query.");
      return;
    }

    if (apiStatus !== "online") {
      setError("Backend is offline. Please start FastAPI.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const controller = new AbortController();
    analysisAbortRef.current = controller;

    try {
      const response = await fetch(
        "http://localhost:8000/api/analyze",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            query: query.trim(),
            language: language,
          }),
          signal: controller.signal,
        }
      );

      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || `Request failed: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        return;
      }

      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong. Please try again."
      );
    } finally {
      if (analysisAbortRef.current === controller) {
        analysisAbortRef.current = null;
        setLoading(false);
      }
    }
  };

  // Clear query and all previous states
  const clearQuery = () => {
    // Cancel any active requests
    transcriptionAbortRef.current?.abort();
    analysisAbortRef.current?.abort();

    // Discard current recording instead of transcribing it
    if (mediaRecorderRef.current?.state === "recording") {
      discardRecordingRef.current = true;
      mediaRecorderRef.current.stop();
    }

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    audioChunksRef.current = [];

    setQuery("");
    setResult(null);
    setError("");
    setDetectedLanguage("");
    setRecording(false);
    setTranscribing(false);
    setLoading(false);
  };

  // Format API output
  const displayValue = (value: any) => {
    if (value == null) {
      return "Not identified";
    }

    if (typeof value === "string") {
      return value;
    }

    return JSON.stringify(value, null, 2);
  };

  return (
    <main className="app">

      {/* Navigation bar */}
      <nav className="navbar">
        <div className="brand">
          <span className="brand-icon">M</span>
          <span>MANZIL</span>
        </div>

        <div className="nav-right">
          <span className="nav-tag">
            Multilingual Navigation Understanding
          </span>

          <span className={`api-indicator ${apiStatus}`}>
            <span className="api-dot"></span>

            {apiStatus === "online"
              ? "Backend Online"
              : apiStatus === "offline"
                ? "Backend Offline"
                : "Checking API..."}
          </span>
        </div>
      </nav>

      {/* Hero section */}
      <section className="hero">
        <span className="eyebrow">
          INTELLIGENT NAVIGATION SYSTEM
        </span>

        <h1>Tell us where you want to go.</h1>

        <p>
          Understand natural navigation instructions in Hindi,
          Hinglish and English using hybrid ML and AI.
        </p>

        {/* Query input panel */}
        <div className="search-panel">

          <div className="input-header">
            <label htmlFor="language">QUERY LANGUAGE</label>

            <select
              id="language"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              disabled={recording || transcribing}
            >
              <option>Hinglish</option>
              <option>Hindi</option>
              <option>English</option>
            </select>
          </div>

          {/* Query textarea */}
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Example: Madhapur mein jo mall hai uske peeche wali building ke paas ek ATM hai..."
            rows={4}
          />

          {/* Voice controls */}
          <div className="voice-controls">

            {!recording ? (
              <button
                className="mic-button"
                onClick={startRecording}
                disabled={
                  transcribing ||
                  loading ||
                  apiStatus !== "online"
                }
              >
                🎤 Start Speaking
              </button>
            ) : (
              <button
                className="mic-button recording"
                onClick={stopRecording}
              >
                ⏹ Stop Recording
              </button>
            )}

            {recording && (
              <span>
                <span className="recording-dot">●</span>
                Listening...{" "}
                {Math.floor(recordingSeconds / 60)}:
                {String(recordingSeconds % 60).padStart(2, "0")}
              </span>
            )}

            {transcribing && (
              <span>Converting speech to text...</span>
            )}

            {detectedLanguage && (
              <span>
                Detected language: {detectedLanguage.toUpperCase()}
              </span>
            )}

            {/* Clear button */}
            <button
              className="clear-button"
              onClick={clearQuery}
              disabled={
                !query &&
                !result &&
                !error &&
                !detectedLanguage &&
                !recording &&
                !transcribing &&
                !loading
              }
            >
              Clear ↺
            </button>
          </div>

          {/* Analyze button */}
          <div className="search-footer">
            <span>Describe your destination naturally</span>

            <button
              onClick={analyzeQuery}
              disabled={
                loading ||
                transcribing ||
                recording ||
                !query.trim() ||
                apiStatus !== "online"
              }
            >
              {loading ? "Analyzing..." : "Analyze Query →"}
            </button>
          </div>
        </div>

        {/* Error message */}
        {error && (
          <div className="error-message">
            {error}
          </div>
        )}
      </section>

      {/* Analysis results */}
      {result && (
        <section className="results">

          <div className="results-heading">
            <div>
              <span className="eyebrow">ANALYSIS OUTPUT</span>
              <h2>Navigation understanding</h2>
            </div>

            <span className="status">
              {result.resolution_status || "ANALYZED"}
            </span>
          </div>

          {/* Main results */}
          <div className="result-grid">

            <article className="result-card">
              <span>INTENT</span>
              <h3>{result.intent || "Not identified"}</h3>
            </article>

            <article className="result-card">
              <span>DESTINATION</span>
              <h3>{displayValue(result.destination)}</h3>
            </article>

            <article className="result-card">
              <span>REFERENCE LOCATION</span>
              <h3>{displayValue(result.reference_location)}</h3>
            </article>

            <article className="result-card">
              <span>AMBIGUITY</span>
              <h3>{result.ambiguity || "Unknown"}</h3>
            </article>

          </div>

          {/* Spatial relationships */}
          <article className="detail-card">
            <h3>Spatial relationships</h3>

            {result.spatial_relationships?.length ? (
              <ul>
                {result.spatial_relationships.map((item, index) => (
                  <li key={index}>
                    {displayValue(item)}
                  </li>
                ))}
              </ul>
            ) : (
              <p>No spatial relationships identified.</p>
            )}
          </article>

          {/* Clarification */}
          {result.clarification_required && (
            <article className="clarification-card">
              <h3>Clarification required</h3>

              <p>
                {result.clarification_question ||
                  "More information is needed to identify the destination."}
              </p>
            </article>
          )}

          {/* Technical output */}
          <details className="technical-details">
            <summary>
              View ML predictions and validation
            </summary>

            <h4>ML Predictions</h4>

            <pre>
              {JSON.stringify(result.ml_predictions, null, 2)}
            </pre>

            <h4>ML Validation</h4>

            <pre>
              {JSON.stringify(result.ml_validation, null, 2)}
            </pre>
          </details>

        </section>
      )}

      {/* Footer */}
      <footer>
        MANZIL · Understanding navigation beyond keywords
      </footer>

    </main>
  );
}

export default App;

