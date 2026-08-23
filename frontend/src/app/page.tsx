"use client";

import React, { useState, useEffect, useRef } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface WorkflowStep {
  id: string;
  name: string;
  description: string;
  status: "pending" | "running" | "completed" | "warning" | "error";
  details?: string;
}

const INITIAL_STEPS: WorkflowStep[] = [
  { id: "vision", name: "Vision Analysis", description: "Detecting road damage & estimating severity", status: "pending" },
  { id: "location", name: "Location Resolution", description: "Resolving road coordinates & demo mapping", status: "pending" },
  { id: "road_research", name: "Road & Project Research", description: "Querying structured road maintenance database", status: "pending" },
  { id: "contract_research", name: "Contract / Tender RAG", description: "Retrieving contracts & tenders via hybrid search", status: "pending" },
  { id: "officer_research", name: "Officer Research", description: "Identifying responsible department & officer", status: "pending" },
  { id: "verification", name: "Evidence Verification", description: "Validating cross-agent evidence & detecting conflicts", status: "pending" },
  { id: "complaint", name: "Complaint Generation", description: "Assembling structured complaint record", status: "pending" },
  { id: "quality_evaluation", name: "Quality Evaluation", description: "Calculating deterministic quality score", status: "pending" },
];

export default function RoadWatchDemo() {
  // Backend health state
  const [backendStatus, setBackendStatus] = useState<"checking" | "online" | "offline">("checking");
  const [backendVersion, setBackendVersion] = useState<string>("");

  // Input states
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [locationHint, setLocationHint] = useState<string>("");
  const [isDragOver, setIsDragOver] = useState<boolean>(false);

  // Execution states
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [progressSteps, setProgressSteps] = useState<WorkflowStep[]>(INITIAL_STEPS);
  const [activeLog, setActiveLog] = useState<string[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Result state
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [copiedJson, setCopiedJson] = useState<boolean>(false);
  const [isDownloadingPdf, setIsDownloadingPdf] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Check backend health on mount
  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) {
          const data = await res.json();
          setBackendStatus("online");
          setBackendVersion(data.version || "0.1.0");
        } else {
          setBackendStatus("offline");
        }
      } catch {
        setBackendStatus("offline");
      }
    }
    checkHealth();
  }, []);

  // Handle File Selection
  const handleFileChange = (file: File | null) => {
    if (!file) return;
    const allowed = ["image/jpeg", "image/png", "image/webp", "image/jpg"];
    if (!allowed.includes(file.type)) {
      setErrorMessage("Unsupported file type. Please select a JPG, JPEG, PNG, or WebP image.");
      return;
    }
    setErrorMessage(null);
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  // Helper to generate in-memory dummy image for one-click demo presets
  const loadDemoPreset = (presetName: string, hint: string) => {
    const canvas = document.createElement("canvas");
    canvas.width = 400;
    canvas.height = 300;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      // Asphalt road background
      ctx.fillStyle = "#334155";
      ctx.fillRect(0, 0, 400, 300);
      // Lane divider
      ctx.strokeStyle = "#fbbf24";
      ctx.lineWidth = 4;
      ctx.setLineDash([15, 15]);
      ctx.beginPath();
      ctx.moveTo(200, 0);
      ctx.lineTo(200, 300);
      ctx.stroke();
      // Pothole drawing
      ctx.fillStyle = "#0f172a";
      ctx.beginPath();
      ctx.ellipse(140, 160, 45, 25, Math.PI / 12, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = "#ffffff";
      ctx.font = "14px Arial";
      ctx.fillText(`RoadWatch AI Demo: ${presetName}`, 20, 30);
    }
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `${presetName.toLowerCase().replace(/\s+/g, "_")}.jpg`, { type: "image/jpeg" });
        setSelectedFile(file);
        setPreviewUrl(URL.createObjectURL(file));
        setLocationHint(hint);
        setErrorMessage(null);
      }
    }, "image/jpeg");
  };

  // Reset demo
  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setLocationHint("");
    setIsAnalyzing(false);
    setProgressSteps(INITIAL_STEPS);
    setActiveLog([]);
    setAnalysisResult(null);
    setErrorMessage(null);
  };

  // Run Workflow with Server-Sent Events (SSE)
  const handleAnalyze = async () => {
    if (!selectedFile) {
      setErrorMessage("Please select or drop a road photograph first.");
      return;
    }

    setIsAnalyzing(true);
    setErrorMessage(null);
    setAnalysisResult(null);
    setProgressSteps(INITIAL_STEPS.map((s) => ({ ...s, status: "pending", details: undefined })));
    setActiveLog(["[SYSTEM] Connecting to RoadWatch AI workflow SSE stream..."]);

    const formData = new FormData();
    formData.append("file", selectedFile);
    if (locationHint.trim()) {
      formData.append("location_hint", locationHint.trim());
    }

    try {
      const response = await fetch(`${API_BASE}/api/analyze/stream`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok || !response.body) {
        throw new Error(`Server returned HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      const stateAccumulator: any = {};

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const block of parts) {
          if (!block.trim()) continue;
          let eventType = "message";
          let dataStr = "";

          const lines = block.split("\n");
          for (const line of lines) {
            if (line.startsWith("event: ")) {
              eventType = line.replace("event: ", "").trim();
            } else if (line.startsWith("data: ")) {
              dataStr = line.replace("data: ", "").trim();
            }
          }

          if (!dataStr) continue;

          try {
            const data = JSON.parse(dataStr);
            handleSSEEvent(eventType, data, stateAccumulator);
          } catch (err) {
            console.error("Failed to parse SSE event data:", err, dataStr);
          }
        }
      }
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to execute analysis. Please ensure backend is running.");
      setActiveLog((prev) => [...prev, `[ERROR] ${err.message || "Workflow connection failed."}`]);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Process SSE Events & Update UI Stepper in real-time
  const handleSSEEvent = (event: string, data: any, accumulator: any) => {
    const timestamp = new Date().toLocaleTimeString();

    const updateStep = (id: string, status: WorkflowStep["status"], details?: string) => {
      setProgressSteps((prev) =>
        prev.map((step) => (step.id === id ? { ...step, status, details: details || step.details } : step))
      );
    };

    switch (event) {
      case "workflow_started":
        setActiveLog((prev) => [...prev, `[${timestamp}] 🚀 Workflow started (Run ID: ${data.run_id})`]);
        updateStep("vision", "running", "Analyzing image...");
        break;

      case "vision_completed":
        accumulator.vision_result = data;
        setActiveLog((prev) => [
          ...prev,
          `[${timestamp}] 👁️ Vision: Pothole detected (${data.severity?.toUpperCase()}), Confidence: ${data.confidence}`,
        ]);
        updateStep("vision", "completed", `Detected: ${data.severity?.toUpperCase()} (${Math.round((data.confidence || 0) * 100)}% conf)`);
        updateStep("location", "running", "Resolving location...");
        break;

      case "location_completed":
        accumulator.location_result = data;
        setActiveLog((prev) => [
          ...prev,
          `[${timestamp}] 📍 Location resolved: ${data.estimated_road_name || "Unknown"} via '${data.resolution_method}'`,
        ]);
        updateStep("location", data.confidence > 0 ? "completed" : "warning", `${data.estimated_road_name || "Unresolved"} (${data.resolution_method})`);
        updateStep("road_research", "running", "Looking up road & project...");
        updateStep("contract_research", "running", "Querying hybrid RAG...");
        break;

      case "evidence_found":
        if (data.evidence_type === "road_and_project") {
          accumulator.road_data = data;
          setActiveLog((prev) => [
            ...prev,
            `[${timestamp}] 🛣️ Road DB: ${data.road_name || "N/A"} (Road ID: ${data.road_id || "N/A"}, Project: ${data.project_id || "N/A"})`,
          ]);
          updateStep("road_research", data.road_id ? "completed" : "warning", `Road: ${data.road_id || "None"}, Project: ${data.project_id || "None"}`);
        } else if (data.evidence_type === "contract_and_tender") {
          accumulator.contract_data = data;
          setActiveLog((prev) => [
            ...prev,
            `[${timestamp}] 📄 Hybrid RAG: Contract ${data.best_contract_id || "N/A"} (Tender: ${data.best_tender_reference || "N/A"})`,
          ]);
          updateStep("contract_research", data.best_contract_id ? "completed" : "warning", `Contract: ${data.best_contract_id || "None"}`);
          updateStep("officer_research", "running", "Identifying officer...");
        } else if (data.evidence_type === "officer") {
          accumulator.officer_data = data;
          setActiveLog((prev) => [
            ...prev,
            `[${timestamp}] 👤 Officer: ${data.officer_name || "N/A"} (${data.officer_id || "N/A"}) - ${data.department || ""}`,
          ]);
          updateStep("officer_research", data.officer_id ? "completed" : "warning", `${data.officer_name || "None"} (${data.officer_id || "N/A"})`);
          updateStep("verification", "running", "Verifying evidence...");
        }
        break;

      case "verification_completed":
        accumulator.verification = data;
        setActiveLog((prev) => [
          ...prev,
          `[${timestamp}] ⚖️ Verification: ${data.requires_human_review ? "⚠️ HUMAN REVIEW REQUIRED" : "✅ VERIFIED"} (Confidence: ${data.verification_confidence})`,
        ]);
        updateStep(
          "verification",
          data.requires_human_review ? "warning" : "completed",
          data.requires_human_review ? "Review Required" : "Verified Complete"
        );
        updateStep("complaint", "running", "Assembling complaint...");
        break;

      case "human_review_required":
        setActiveLog((prev) => [...prev, `[${timestamp}] ⚠️ Action needed: Human review flagged.`]);
        break;

      case "complaint_generated":
        accumulator.complaint_record = data;
        setActiveLog((prev) => [...prev, `[${timestamp}] 📋 Complaint generated: ${data.complaint_id}`]);
        updateStep("complaint", "completed", `ID: ${data.complaint_id}`);
        updateStep("quality_evaluation", "running", "Scoring quality...");
        break;

      case "quality_evaluated":
        accumulator.final_quality_score = data.final_quality_score;
        accumulator.quality_explanation = data.quality_explanation;
        setActiveLog((prev) => [
          ...prev,
          `[${timestamp}] 🏆 Quality Score: ${data.final_quality_score}/100`,
        ]);
        updateStep("quality_evaluation", "completed", `Score: ${data.final_quality_score}/100`);
        break;

      case "workflow_completed":
        setActiveLog((prev) => [...prev, `[${timestamp}] ✨ Workflow completed successfully.`]);
        setAnalysisResult({
          ...accumulator,
          ...data,
          vision_result: data.vision_result || accumulator.vision_result,
          location_result: data.location_result || accumulator.location_result,
          road_data: data.road_data || accumulator.road_data,
          contract_data: data.contract_data || accumulator.contract_data,
          officer_data: data.officer_data || accumulator.officer_data,
          verification: data.verification || accumulator.verification,
          complaint_record: data.complaint_record || accumulator.complaint_record,
          final_quality_score: data.final_quality_score ?? accumulator.final_quality_score,
          quality_explanation: data.quality_explanation || accumulator.quality_explanation,
        });
        break;

      case "workflow_error":
        setErrorMessage(data.error || "An unexpected error occurred during workflow execution.");
        setActiveLog((prev) => [...prev, `[${timestamp}] ❌ Error: ${data.error}`]);
        break;

      default:
        break;
    }
  };

  const copyComplaintJson = () => {
    if (analysisResult?.complaint_record) {
      navigator.clipboard.writeText(JSON.stringify(analysisResult.complaint_record, null, 2));
      setCopiedJson(true);
      setTimeout(() => setCopiedJson(false), 2000);
    }
  };

  const handleDownloadPdf = async () => {
    if (!analysisResult) return;
    setIsDownloadingPdf(true);
    try {
      const res = await fetch(`${API_BASE}/api/complaints/pdf`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(analysisResult),
      });
      if (!res.ok) {
        throw new Error(`Failed to generate PDF (HTTP ${res.status})`);
      }
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const cr = analysisResult.complaint_record;
      const complaintId = cr?.complaint_id || analysisResult.run_id || "report";
      const safeId = complaintId.replace(/[^A-Za-z0-9_-]/g, "_");
      a.download = `RoadWatch_Complaint_${safeId}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      console.error("PDF download error:", err);
      alert(err.message || "Failed to download PDF report.");
    } finally {
      setIsDownloadingPdf(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500 selection:text-white">
      {/* 1. Header & Synthetic Data Banner */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-50">
        <div className="bg-amber-500/10 border-b border-amber-500/20 px-4 py-2 text-center text-xs font-medium text-amber-300 flex items-center justify-center gap-2">
          <span>⚠️</span>
          <span>SYNTHETIC DEMO SYSTEM — All roads, tenders, contracts, and officers are strictly fictional for demonstration only.</span>
        </div>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center font-bold text-white shadow-lg shadow-indigo-500/20">
              RW
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                RoadWatch AI
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 font-semibold border border-indigo-500/30">
                  Capstone Demo
                </span>
              </h1>
              <p className="text-xs text-slate-400">AI-powered road issue analysis and complaint generation</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-800 border border-slate-700 text-xs">
              <div
                className={`w-2 h-2 rounded-full ${
                  backendStatus === "online"
                    ? "bg-emerald-400 animate-pulse"
                    : backendStatus === "offline"
                    ? "bg-rose-500"
                    : "bg-amber-400 animate-ping"
                }`}
              />
              <span className="text-slate-300 font-medium">
                {backendStatus === "online"
                  ? `Backend Online (v${backendVersion})`
                  : backendStatus === "offline"
                  ? "Backend Offline (Port 8000)"
                  : "Checking Backend..."}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* 2. Main Content Grid */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 w-full space-y-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Upload & Controls */}
          <section className="lg:col-span-5 space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
              <div>
                <h2 className="text-base font-semibold text-white">1. Road Damage Photograph</h2>
                <p className="text-xs text-slate-400 mt-1">Upload a suspected pothole photograph (JPG, PNG, or WebP).</p>
              </div>

              {/* Drag & Drop Zone */}
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragOver(true);
                }}
                onDragLeave={() => setIsDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setIsDragOver(false);
                  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                    handleFileChange(e.dataTransfer.files[0]);
                  }
                }}
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all duration-200 ${
                  isDragOver
                    ? "border-indigo-500 bg-indigo-500/10"
                    : previewUrl
                    ? "border-slate-700 bg-slate-950"
                    : "border-slate-700 bg-slate-950 hover:border-slate-600 hover:bg-slate-800/40"
                }`}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={(e) => handleFileChange(e.target.files?.[0] || null)}
                  accept="image/jpeg,image/png,image/webp,image/jpg"
                  className="hidden"
                />

                {previewUrl ? (
                  <div className="space-y-3">
                    <img
                      src={previewUrl}
                      alt="Selected road"
                      className="max-h-48 mx-auto rounded-lg object-contain shadow-md border border-slate-800"
                    />
                    <div className="flex items-center justify-between text-xs text-slate-400 px-2">
                      <span className="truncate max-w-[200px] font-mono text-slate-300">{selectedFile?.name}</span>
                      <span className="text-indigo-400 font-medium">Click to change</span>
                    </div>
                  </div>
                ) : (
                  <div className="py-6 space-y-3">
                    <div className="w-12 h-12 rounded-full bg-slate-800 text-indigo-400 flex items-center justify-center mx-auto text-xl border border-slate-700">
                      📷
                    </div>
                    <div>
                      <p className="text-sm font-medium text-slate-200">Drag and drop road photograph here</p>
                      <p className="text-xs text-slate-500 mt-1">or click to browse files from your computer</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Quick Demo Presets */}
              <div className="space-y-2">
                <label className="text-xs font-medium text-slate-400">Quick Demo Presets (One-Click):</label>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <button
                    type="button"
                    onClick={() => loadDemoPreset("Synthetic Road 7", "Synthetic Road 7")}
                    className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-left text-slate-200 transition"
                  >
                    <span className="font-semibold block text-indigo-300">Demo Case 7</span>
                    <span className="text-[11px] text-slate-400">Synthetic Road 7 (RD-007)</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => loadDemoPreset("Synthetic Road 1", "Synthetic Road 1")}
                    className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-left text-slate-200 transition"
                  >
                    <span className="font-semibold block text-indigo-300">Demo Case 1</span>
                    <span className="text-[11px] text-slate-400">Synthetic Road 1 (RD-001)</span>
                  </button>
                </div>
              </div>

              {/* Location Hint Input */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label htmlFor="locationHint" className="text-xs font-semibold text-white">
                    2. Location Hint <span className="text-slate-400 font-normal">(Optional)</span>
                  </label>
                </div>
                <input
                  id="locationHint"
                  type="text"
                  value={locationHint}
                  onChange={(e) => setLocationHint(e.target.value)}
                  placeholder="e.g. Synthetic Road 7 or RD-007"
                  className="w-full px-3.5 py-2.5 rounded-xl bg-slate-950 border border-slate-700 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition"
                />
                <p className="text-[11px] text-slate-500">
                  Provides a location reference for demo mapping when image EXIF GPS metadata is absent.
                </p>
              </div>

              {/* Error Message Display */}
              {errorMessage && (
                <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-start gap-2">
                  <span className="text-rose-400">⚠️</span>
                  <span>{errorMessage}</span>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex gap-3 pt-2">
                <button
                  type="button"
                  onClick={handleAnalyze}
                  disabled={isAnalyzing || !selectedFile}
                  className={`flex-1 py-3 px-4 rounded-xl font-semibold text-sm shadow-lg flex items-center justify-center gap-2 transition-all ${
                    isAnalyzing || !selectedFile
                      ? "bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700"
                      : "bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white shadow-indigo-600/30 cursor-pointer active:scale-[0.99]"
                  }`}
                >
                  {isAnalyzing ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      <span>Executing Multi-Agent Graph...</span>
                    </>
                  ) : (
                    <>
                      <span>🔍 Analyze Road</span>
                    </>
                  )}
                </button>

                {(selectedFile || analysisResult) && !isAnalyzing && (
                  <button
                    type="button"
                    onClick={handleReset}
                    className="py-3 px-4 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm font-medium border border-slate-700 transition cursor-pointer"
                  >
                    Reset
                  </button>
                )}
              </div>
            </div>

            {/* Live SSE Terminal Event Log */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-indigo-400" />
                  Live SSE Event Stream
                </h3>
                <span className="text-[10px] text-slate-500 font-mono">POST /api/analyze/stream</span>
              </div>
              <div className="h-44 overflow-y-auto bg-slate-950 rounded-xl p-3 font-mono text-[11px] space-y-1 text-slate-400 border border-slate-800">
                {activeLog.length === 0 ? (
                  <p className="text-slate-600 italic">No active workflow stream. Click "Analyze Road" to start.</p>
                ) : (
                  activeLog.map((log, idx) => (
                    <div key={idx} className="leading-relaxed">
                      {log}
                    </div>
                  ))
                )}
              </div>
            </div>
          </section>

          {/* Right Column: Workflow Stepper & Results Dashboard */}
          <section className="lg:col-span-7 space-y-6">
            {/* 3. Live Agent Workflow Progress Stepper */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-base font-semibold text-white">Agent Workflow Orchestration</h2>
                  <p className="text-xs text-slate-400">Real-time LangGraph multi-agent execution pipeline</p>
                </div>
                {isAnalyzing && (
                  <span className="text-xs px-2.5 py-1 rounded-full bg-indigo-500/20 text-indigo-300 font-semibold border border-indigo-500/30 animate-pulse">
                    Live Streaming
                  </span>
                )}
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
                {progressSteps.map((step, idx) => (
                  <div
                    key={step.id}
                    className={`p-3 rounded-xl border transition-all ${
                      step.status === "running"
                        ? "bg-indigo-950/40 border-indigo-500/50 shadow-md shadow-indigo-500/10"
                        : step.status === "completed"
                        ? "bg-slate-950/60 border-emerald-500/30"
                        : step.status === "warning"
                        ? "bg-amber-950/20 border-amber-500/30"
                        : "bg-slate-950/30 border-slate-800 opacity-60"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-semibold text-slate-200">
                        {idx + 1}. {step.name}
                      </span>
                      <span>
                        {step.status === "running" && (
                          <div className="w-3.5 h-3.5 border-2 border-indigo-400/30 border-t-indigo-400 rounded-full animate-spin" />
                        )}
                        {step.status === "completed" && <span className="text-emerald-400 text-xs">✓</span>}
                        {step.status === "warning" && <span className="text-amber-400 text-xs">⚠️</span>}
                        {step.status === "pending" && <span className="text-slate-600 text-xs">⏳</span>}
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-400">{step.description}</p>
                    {step.details && (
                      <p className="text-[10px] font-mono text-indigo-300 mt-1 truncate">{step.details}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* 4. Results Dashboard */}
            {analysisResult && (
              <div className="space-y-6 animate-in fade-in duration-300">
                {/* Human Review Banner (if required) */}
                {analysisResult.requires_human_review && (
                  <div className="p-5 rounded-2xl bg-amber-500/10 border-2 border-amber-500/40 shadow-lg space-y-2">
                    <div className="flex items-center gap-2 text-amber-300 font-bold text-sm">
                      <span className="text-lg">⚠️</span>
                      <span>HUMAN REVIEW REQUIRED</span>
                    </div>
                    <p className="text-xs text-amber-200/90 leading-relaxed">
                      The automated verification agent flagged one or more conflicts or gaps in evidence. Human officer
                      review and verification is required before official filing.
                    </p>
                    {analysisResult.complaint_record?.evidence_conflicts?.length > 0 && (
                      <ul className="list-disc list-inside text-xs text-amber-300/80 space-y-0.5 pt-1">
                        {analysisResult.complaint_record.evidence_conflicts.map((conflict: string, i: number) => (
                          <li key={i}>{conflict}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}

                {/* Quality Evaluation Score Card */}
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-bold text-white flex items-center gap-2">
                        <span>🏆</span> Deterministic Quality Evaluation
                      </h3>
                      <p className="text-xs text-slate-400 mt-0.5">Calculated across 8 weighted evidence components</p>
                    </div>
                    <div className="text-right">
                      <span className="text-2xl font-black text-white">
                        {analysisResult.final_quality_score}
                        <span className="text-xs font-medium text-slate-400"> / 100</span>
                      </span>
                    </div>
                  </div>

                  <div className="w-full bg-slate-950 rounded-full h-3 overflow-hidden border border-slate-800">
                    <div
                      className={`h-full transition-all duration-1000 ${
                        analysisResult.final_quality_score >= 80
                          ? "bg-gradient-to-r from-emerald-500 to-teal-400"
                          : analysisResult.final_quality_score >= 60
                          ? "bg-gradient-to-r from-amber-500 to-yellow-400"
                          : "bg-gradient-to-r from-rose-500 to-red-400"
                      }`}
                      style={{ width: `${Math.min(100, Math.max(0, analysisResult.final_quality_score))}%` }}
                    />
                  </div>

                  <p className="text-xs text-slate-300 bg-slate-950 p-3 rounded-xl border border-slate-800 font-mono">
                    {analysisResult.quality_explanation}
                  </p>
                </div>

                {/* Evidence Grid: Detection, Location, Road, Contract, Officer, Verification */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {/* A. Detection Card */}
                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-md space-y-3">
                    <h4 className="text-xs font-bold text-indigo-300 uppercase tracking-wider flex items-center gap-1.5">
                      <span>👁️</span> A. Vision Detection
                    </h4>
                    <div className="space-y-2 text-xs">
                      <div className="flex justify-between items-center py-1 border-b border-slate-800">
                        <span className="text-slate-400">Pothole Detected:</span>
                        <span className={`font-semibold ${analysisResult.vision_result?.pothole_detected ? "text-emerald-400" : "text-slate-200"}`}>
                          {analysisResult.vision_result?.pothole_detected ? "Yes" : (analysisResult.vision_result ? "No" : "N/A")}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-1 border-b border-slate-800">
                        <span className="text-slate-400">Severity:</span>
                        <span className="px-2 py-0.5 rounded text-[11px] font-bold uppercase bg-amber-500/20 text-amber-300 border border-amber-500/30">
                          {analysisResult.vision_result?.severity?.toUpperCase() || "N/A"}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-1 border-b border-slate-800">
                        <span className="text-slate-400">Model Confidence:</span>
                        <span className="font-mono text-slate-200">
                          {analysisResult.vision_result?.confidence !== undefined
                            ? `${Math.round(analysisResult.vision_result.confidence * 100)}%`
                            : "N/A"}
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400 pt-1 italic">
                        {analysisResult.vision_result?.visual_evidence
                          ? `"${analysisResult.vision_result.visual_evidence}"`
                          : "No visual evidence description recorded."}
                      </p>
                    </div>
                  </div>

                  {/* B. Location Card */}
                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-md space-y-3">
                    <h4 className="text-xs font-bold text-indigo-300 uppercase tracking-wider flex items-center gap-1.5">
                      <span>📍</span> B. Location Resolution
                    </h4>
                    <div className="space-y-2 text-xs">
                      <div className="flex justify-between items-center py-1 border-b border-slate-800">
                        <span className="text-slate-400">Road Name:</span>
                        <span className="font-semibold text-slate-200">
                          {analysisResult.location_result?.estimated_road_name || "Unresolved"}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-1 border-b border-slate-800">
                        <span className="text-slate-400">Method:</span>
                        <span className="font-mono text-slate-300">
                          {analysisResult.location_result?.resolution_method || "N/A"}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-1 border-b border-slate-800">
                        <span className="text-slate-400">Confidence:</span>
                        <span className="font-mono text-slate-200">
                          {analysisResult.location_result?.confidence !== undefined
                            ? `${Math.round(analysisResult.location_result.confidence * 100)}%`
                            : "N/A"}
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400 pt-1">
                        {analysisResult.location_result?.notes}
                      </p>
                    </div>
                  </div>

                  {/* C. Road & Maintenance Card */}
                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-md space-y-3">
                    <h4 className="text-xs font-bold text-indigo-300 uppercase tracking-wider flex items-center gap-1.5">
                      <span>🛣️</span> C. Road & Maintenance
                    </h4>
                    <div className="space-y-2 text-xs">
                      <div className="flex justify-between items-center py-1 border-b border-slate-800">
                        <span className="text-slate-400">Road ID:</span>
                        <span className="font-mono font-semibold text-indigo-300">
                          {analysisResult.road_data?.road?.road_id || analysisResult.road_data?.road_id || "None"}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-1 border-b border-slate-800">
                        <span className="text-slate-400">District / Area:</span>
                        <span className="text-slate-200">
                          {analysisResult.road_data?.road
                            ? `${analysisResult.road_data.road.district}, ${analysisResult.road_data.road.area}`
                            : (analysisResult.road_data?.district
                              ? `${analysisResult.road_data.district}${analysisResult.road_data.area ? `, ${analysisResult.road_data.area}` : ""}`
                              : (analysisResult.location_result?.district || "N/A"))}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-1 border-b border-slate-800">
                        <span className="text-slate-400">Active Project:</span>
                        <span className="font-mono text-slate-200">
                          {analysisResult.road_data?.project?.project_id || analysisResult.road_data?.project_id || "None"}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-1">
                        <span className="text-slate-400">Project Status:</span>
                        <span className="font-semibold text-emerald-400">
                          {analysisResult.road_data?.project?.status || analysisResult.road_data?.project_status || (analysisResult.road_data?.project_id ? "Active" : "None")}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* D. Contract / Tender Card */}
                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-md space-y-3">
                    <h4 className="text-xs font-bold text-indigo-300 uppercase tracking-wider flex items-center gap-1.5">
                      <span>📄</span> D. Contract / Tender (RAG)
                    </h4>
                    <div className="space-y-2 text-xs">
                      <div className="flex justify-between items-center py-1 border-b border-slate-800">
                        <span className="text-slate-400">Contract ID:</span>
                        <span className="font-mono font-semibold text-indigo-300">
                          {analysisResult.contract_data?.best_contract_id || "None"}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-1 border-b border-slate-800">
                        <span className="text-slate-400">Tender Reference:</span>
                        <span className="font-mono text-slate-200">
                          {analysisResult.contract_data?.best_tender_reference || "None"}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-1 border-b border-slate-800">
                        <span className="text-slate-400">Contractor:</span>
                        <span className="text-slate-200 truncate max-w-[150px]">
                          {analysisResult.contract_data?.contractor_record?.contractor_name ||
                            analysisResult.contract_data?.contractor_name ||
                            analysisResult.complaint_record?.contractor?.contractor_name ||
                            "None"}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-1">
                        <span className="text-slate-400">Structured Match:</span>
                        <span className="text-emerald-400 font-semibold">
                          {analysisResult.contract_data?.structured_match ? "Yes (Confirmed)" : "Unstructured Only"}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* E. Responsible Officer Card */}
                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-md space-y-3">
                    <h4 className="text-xs font-bold text-indigo-300 uppercase tracking-wider flex items-center gap-1.5">
                      <span>👤</span> E. Responsible Officer
                    </h4>
                    <div className="space-y-2 text-xs">
                      <div className="flex justify-between items-center py-1 border-b border-slate-800">
                        <span className="text-slate-400">Officer Name:</span>
                        <span className="font-semibold text-slate-200">
                          {analysisResult.officer_data?.officer?.officer_name ||
                            analysisResult.officer_data?.officer_name ||
                            analysisResult.complaint_record?.responsible_officer?.officer_name ||
                            "Unassigned"}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-1 border-b border-slate-800">
                        <span className="text-slate-400">Officer ID:</span>
                        <span className="font-mono text-indigo-300">
                          {analysisResult.officer_data?.officer?.officer_id ||
                            analysisResult.officer_data?.officer_id ||
                            analysisResult.complaint_record?.responsible_officer?.officer_id ||
                            "None"}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-1 border-b border-slate-800">
                        <span className="text-slate-400">Department:</span>
                        <span className="text-slate-300 truncate max-w-[180px]">
                          {analysisResult.officer_data?.officer?.department ||
                            analysisResult.officer_data?.department ||
                            analysisResult.complaint_record?.responsible_officer?.department ||
                            "N/A"}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-1">
                        <span className="text-slate-400">Jurisdiction:</span>
                        <span className="text-slate-300">
                          {analysisResult.officer_data?.officer?.jurisdiction ||
                            analysisResult.officer_data?.jurisdiction ||
                            analysisResult.complaint_record?.responsible_officer?.jurisdiction ||
                            analysisResult.location_result?.district ||
                            "N/A"}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* F. Verification Diagnostic Card */}
                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-md space-y-3">
                    <h4 className="text-xs font-bold text-indigo-300 uppercase tracking-wider flex items-center gap-1.5">
                      <span>⚖️</span> F. Verification Status
                    </h4>
                    <div className="space-y-2 text-xs">
                      <div className="flex justify-between items-center py-1 border-b border-slate-800">
                        <span className="text-slate-400">Status:</span>
                        <span
                          className={`font-bold px-2 py-0.5 rounded text-[11px] ${
                            analysisResult.complaint_record?.verification_status === "VERIFIED"
                              ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                              : "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                          }`}
                        >
                          {analysisResult.complaint_record?.verification_status || "N/A"}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-1 border-b border-slate-800">
                        <span className="text-slate-400">Verification Confidence:</span>
                        <span className="font-mono text-slate-200">
                          {analysisResult.complaint_record?.verification_confidence !== undefined
                            ? `${Math.round(analysisResult.complaint_record.verification_confidence * 100)}%`
                            : "N/A"}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-1 border-b border-slate-800">
                        <span className="text-slate-400">Conflicts Detected:</span>
                        <span className="font-mono text-slate-200">
                          {analysisResult.complaint_record?.evidence_conflicts?.length || 0}
                        </span>
                      </div>
                      <div className="flex justify-between items-center py-1">
                        <span className="text-slate-400">Human Review Flag:</span>
                        <span className={analysisResult.requires_human_review ? "text-amber-400 font-bold" : "text-emerald-400"}>
                          {analysisResult.requires_human_review ? "Required" : "Not Required"}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* G. Structured Generated Complaint Card */}
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-bold text-white flex items-center gap-2">
                        <span>📋</span> G. Generated Complaint Record
                      </h3>
                      <p className="text-xs text-slate-400 mt-0.5">Structured record prepared for government complaint filing</p>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={handleDownloadPdf}
                        disabled={isDownloadingPdf || !analysisResult?.complaint_record}
                        className="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-xs font-semibold text-white shadow-md shadow-indigo-600/20 transition cursor-pointer flex items-center gap-1.5"
                      >
                        {isDownloadingPdf ? (
                          <>
                            <span className="animate-spin text-[10px]">⏳</span>
                            <span>Generating PDF...</span>
                          </>
                        ) : (
                          <>
                            <span>📄</span>
                            <span>Download Complaint PDF</span>
                          </>
                        )}
                      </button>
                      <button
                        type="button"
                        onClick={copyComplaintJson}
                        className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-indigo-300 border border-slate-700 transition cursor-pointer"
                      >
                        {copiedJson ? "✓ Copied JSON" : "Copy JSON"}
                      </button>
                    </div>
                  </div>

                  {analysisResult.complaint_record && (
                    <div className="bg-slate-950 rounded-xl p-5 border border-slate-800 space-y-4 text-xs">
                      <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-800">
                        <div>
                          <span className="text-slate-500 font-mono text-[10px] block">COMPLAINT ID</span>
                          <span className="font-mono font-bold text-indigo-400 text-sm">
                            {analysisResult.complaint_record.complaint_id}
                          </span>
                        </div>
                        <div className="text-right">
                          <span className="text-slate-500 font-mono text-[10px] block">TIMESTAMP</span>
                          <span className="text-slate-300 font-mono text-[11px]">
                            {analysisResult.complaint_record.generated_at}
                          </span>
                        </div>
                      </div>

                      <div className="space-y-1">
                        <span className="text-slate-500 font-mono text-[10px] block">ISSUE DESCRIPTION</span>
                        <p className="text-slate-200 leading-relaxed font-sans">
                          {analysisResult.complaint_record.issue_description}
                        </p>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
                        <div className="p-3 bg-slate-900 rounded-lg border border-slate-800">
                          <span className="text-slate-500 text-[10px] block">LOCATION</span>
                          <span className="font-semibold text-slate-200 block truncate">
                            {analysisResult.complaint_record.location_summary}
                          </span>
                        </div>
                        <div className="p-3 bg-slate-900 rounded-lg border border-slate-800">
                          <span className="text-slate-500 text-[10px] block">CONTRACTOR</span>
                          <span className="font-semibold text-slate-200 block truncate">
                            {analysisResult.complaint_record.contractor?.contractor_name || "Unassigned"}
                          </span>
                        </div>
                        <div className="p-3 bg-slate-900 rounded-lg border border-slate-800">
                          <span className="text-slate-500 text-[10px] block">RESPONSIBLE OFFICER</span>
                          <span className="font-semibold text-slate-200 block truncate">
                            {analysisResult.complaint_record.responsible_officer?.officer_name || "Unassigned"}
                          </span>
                        </div>
                      </div>

                      <div className="pt-2 border-t border-slate-800 text-[10px] text-amber-300/70 italic text-center">
                        {analysisResult.complaint_record.disclaimer}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </section>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800 bg-slate-900/50 py-6 text-center text-xs text-slate-500">
        <p>RoadWatch AI — Autonomous Multi-Agent Road Damage Analysis System (Synthetic Demo)</p>
      </footer>
    </div>
  );
}
