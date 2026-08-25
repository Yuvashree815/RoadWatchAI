# RoadWatch AI — Production Deployment Guide
**Backend: Railway • Frontend: Vercel • Source: GitHub (`Yuvashree815/RoadWatchAI`)**

---

## 1. Architecture & Deployment Overview

```
                        ┌─────────────────────────────────────────┐
                        │            User Web Browser             │
                        └──────────────────┬──────────────────────┘
                                           │
                                           ▼
                   ┌───────────────────────────────────────────────────┐
                   │             Frontend — Vercel                     │
                   │  Next.js 16 + React 19 + TypeScript (App Router)   │
                   │  URL: https://<your-project>.vercel.app           │
                   └───────────────────────┬───────────────────────────┘
                                           │
                        API & SSE Stream   │ (NEXT_PUBLIC_API_URL)
                                           ▼
                   ┌───────────────────────────────────────────────────┐
                   │             Backend — Railway                     │
                   │  FastAPI + Uvicorn + Python 3.11                  │
                   │  URL: https://<your-service>.up.railway.app       │
                   ├───────────────────────────────────────────────────┤
                   │  • LangGraph Multi-Agent Workflow Engine          │
                   │  • Google Gemini 3.6 Flash Multimodal Vision      │
                   │  • ChromaDB + BM25 Hybrid Retrieval (RAG)         │
                   │  • LangSmith Live Execution Observability         │
                   │  • ReportLab PDF Complaint Generator              │
                   │  • Gmail SMTP Complaint Email Submission          │
                   └───────────────────────┬───────────────────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
           ┌────────────────────────┐            ┌────────────────────────┐
           │ LangSmith Observability│            │ Gmail SMTP Dispatch    │
           │ (smith.langchain.com)  │            │ (Demo Inbox Delivery)  │
           └────────────────────────┘            └────────────────────────┘
```

---

## 2. Step 1: Deploy Backend to Railway

### A. Create Project on Railway
1. Go to [railway.com](https://railway.com/) and log in with your GitHub account.
2. Click **"New Project"** $\to$ **"Deploy from GitHub repo"**.
3. Select your repository: `Yuvashree815/RoadWatchAI`.
4. Railway will automatically detect the Python environment via `railway.json` / `Procfile` / `requirements.txt`.

### B. Configure Service Settings in Railway
1. In your Railway service dashboard, go to **Settings**:
   - **Root Directory:** `/` (default repository root)
   - **Build Command:** `pip install -r requirements.txt` (or default Nixpacks)
   - **Start Command:** `python -m backend.main`
2. Under **Networking**, click **"Generate Domain"** (e.g. `roadwatch-production.up.railway.app`).
3. Note your Railway public URL (e.g. `https://roadwatch-production.up.railway.app`).

### C. Configure Railway Environment Variables
In Railway Dashboard $\to$ **Variables**, add:

| Variable Name | Production Value | Description |
|:---|:---|:---|
| `GEMINI_API_KEY` | `AIzaSy...` | Your Google Gemini API Key |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Multimodal vision model |
| `LANGSMITH_TRACING` | `true` | Enables LangSmith trace logs |
| `LANGSMITH_API_KEY` | `lsv2_pt_...` | Your LangSmith API Key |
| `LANGSMITH_PROJECT` | `RoadWatchAI` | LangSmith project name |
| `LANGSMITH_ENDPOINT` | `https://api.smith.langchain.com` | LangSmith API endpoint |
| `EMAIL_ENABLED` | `true` | Enables complaint email delivery |
| `MOCK_LLM` | `false` | Real Google Gemini calls |
| `MOCK_EMAIL` | `false` | Real Gmail SMTP transmission |
| `DEMO_MODE` | `true` | Keeps synthetic demo safeguards active |
| `EMAIL_PROVIDER` | `smtp` | Email transmission provider |
| `SMTP_HOST` | `smtp.gmail.com` | Gmail SMTP server |
| `SMTP_PORT` | `587` | STARTTLS port |
| `SMTP_USERNAME` | `your_gmail@gmail.com` | Sender Gmail address |
| `SMTP_PASSWORD` | `your_16_char_app_password` | Gmail 2FA App Password |
| `SMTP_FROM` | `your_gmail@gmail.com` | From header |
| `DEMO_COMPLAINT_EMAIL` | `your_inbox@gmail.com` | Destination demo inbox |
| `FRONTEND_URL` | `https://<your-project>.vercel.app` | Vercel domain for strict CORS |
| `BACKEND_URL` | `https://<your-service>.up.railway.app` | Railway service domain |
| `CHROMA_PERSIST_DIRECTORY` | `./chroma_data` | Vector store directory |

---

## 3. Step 2: Deploy Frontend to Vercel

### A. Import Project in Vercel
1. Go to [vercel.com](https://vercel.com/) and log in with your GitHub account.
2. Click **"Add New..."** $\to$ **"Project"**.
3. Select `Yuvashree815/RoadWatchAI`.
4. In the **Configure Project** screen:
   - **Framework Preset:** `Next.js`
   - **Root Directory:** Click "Edit" and choose `frontend`
   - **Build Command:** `npm run build` (or Next.js default)
   - **Output Directory:** `.next` (default)

### B. Configure Vercel Environment Variables
Under **Environment Variables**, add:

| Key | Value | Description |
|:---|:---|:---|
| `NEXT_PUBLIC_API_URL` | `https://<your-railway-url>.up.railway.app` | Production Railway backend URL (no trailing slash) |

5. Click **"Deploy"**. Vercel will build and deploy your Next.js frontend in ~1 minute.

---

## 4. Verification & Production Health Checks

### 1. Test Backend Health
```bash
curl -i https://<your-railway-url>.up.railway.app/health
```
**Expected Response (HTTP 200):**
```json
{"status":"ok","message":"RoadWatch AI backend is running.","version":"0.1.0"}
```

### 2. Test Configuration Endpoint
```bash
curl -i https://<your-railway-url>.up.railway.app/api/config
```
**Expected Response (HTTP 200):**
Confirms `mock_llm: false`, `mock_email: false`, `email_enabled: true`, and `gemini_configured: true` without exposing keys.

### 3. Test Full Multi-Agent Pipeline from Frontend
1. Open your Vercel frontend URL: `https://<your-project>.vercel.app`.
2. Backend status badge in top navigation should show **"API Online (v0.1.0)"**.
3. Click **"Preset 1 (Demo Case 1)"** or drop a road image.
4. Watch the 9-step agent stepper run in real time.
5. In **Section H**, confirm status badge: **`SUBMITTED ✓`**.
6. Download the PDF report via **"Download Complaint PDF"**.
7. Check your `DEMO_COMPLAINT_EMAIL` inbox for the notification and attached PDF.
8. Check [smith.langchain.com](https://smith.langchain.com/) for the full LangSmith multi-agent execution trace.
