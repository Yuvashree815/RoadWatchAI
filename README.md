# 🛣️ RoadWatch AI — Autonomous Multi-Agent Road Damage Intelligence System

[![Next.js](https://img.shields.io/badge/Frontend-Next.js%2016-black?style=for-the-badge&logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-FF6F00?style=for-the-badge&logo=langchain)](https://langchain-ai.github.io/langgraph/)
[![Google Gemini](https://img.shields.io/badge/Vision%20LLM-Gemini%203.6%20Flash-4285F4?style=for-the-badge&logo=google)](https://ai.google.dev/)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-blue?style=for-the-badge)](https://www.trychroma.com/)
[![LangSmith](https://img.shields.io/badge/Observability-LangSmith-orange?style=for-the-badge&logo=langchain)](https://smith.langchain.com/)
[![Vercel](https://img.shields.io/badge/Deployed-Vercel-black?style=for-the-badge&logo=vercel)](https://road-watch-ai-orcin.vercel.app/)
[![Railway](https://img.shields.io/badge/Deployed-Railway-0B0D0E?style=for-the-badge&logo=railway)](https://roadwatchai-production-c7b6.up.railway.app/)

> **An end-to-end GenAI capstone system that transforms citizen-uploaded road damage photographs into legally structured, verified, and attributable municipal complaints with automated email dispatch and official PDF certificates.**

---

## 🌐 Live Production Links

* **Live Web Application (Vercel):** [https://road-watch-ai-orcin.vercel.app/](https://road-watch-ai-orcin.vercel.app/)
* **Backend API (Railway):** [https://roadwatchai-production-c7b6.up.railway.app/](https://roadwatchai-production-c7b6.up.railway.app/)
* **Interactive Swagger API Docs:** [https://roadwatchai-production-c7b6.up.railway.app/docs](https://roadwatchai-production-c7b6.up.railway.app/docs)
* **Backend Health Check:** [https://roadwatchai-production-c7b6.up.railway.app/health](https://roadwatchai-production-c7b6.up.railway.app/health)

---

## 📌 Problem & Solution Overview

### The Problem
Traditional citizen reporting tools for public infrastructure suffer from:
1. **Unverified Submissions:** High volume of poor-quality or non-damage images causing municipal backlogs.
2. **Missing Accountability:** Citizens and caseworkers do not know which active maintenance project, contractor contract, or government engineer is legally responsible.
3. **Unstructured Output:** Complaints lack formal legal citations, tender references, and quality checks required for administrative action.

### The Solution: RoadWatch AI
RoadWatch AI deploys a coordinated pipeline of **9 autonomous LangGraph agents** that:
* Analyze image damage with **Google Gemini 3.6 Flash Multimodal Vision**.
* Resolve GPS metadata and geographic jurisdiction.
* Retrieve municipal contracts and contractor terms via **Hybrid RAG (ChromaDB + BM25)**.
* Cross-verify evidence across agents to eliminate hallucinations.
* Calculate a transparent **Deterministic Quality Score (0–100)**.
* Generate an official **Complaint PDF Certificate** and automatically dispatch it via **Email** to responsible authorities.

---

## 🏗️ System Architecture & Workflow

```
[ Citizen Image Upload ]
          │
          ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   LangGraph Multi-Agent Pipeline                       │
├────────────────────────────────────────────────────────────────────────┤
│ 1. 👁️ Vision Agent (Gemini 3.6 Flash)                                 │
│    └── Detects pothole presence, severity (Low/Moderate/Critical)      │
│                                                                        │
│ 2. 📍 Location Agent                                                   │
│    └── Resolves GPS EXIF coordinates & geographic jurisdiction         │
│                                                                        │
│ 3. 🛣️ Road Research Agent                                              │
│    └── Queries relational DB for active road maintenance projects      │
│                                                                        │
│ 4. 📄 Contract Research Agent (Hybrid RAG)                             │
│    └── Dense Vector Search (ChromaDB) + Sparse Keyword Match (BM25)    │
│                                                                        │
│ 5. 🏛️ Officer Research Agent                                          │
│    └── Identifies designated municipal authority & chief inspector     │
│                                                                        │
│ 6. ⚖️ Evidence Verification Agent                                      │
│    └── Cross-validates all evidence & flags conflicting data           │
│                                                                        │
│ 7. 📝 Complaint Generation Agent                                       │
│    └── Synthesizes formal municipal complaint with Complaint ID        │
│                                                                        │
│ 8. 📊 Quality Evaluation Agent                                         │
│    └── Computes 8-factor deterministic quality score (0–100)           │
│                                                                        │
│ 9. ✉️ Email Submission & PDF Dispatch                                  │
│    ├── Score >= 50: Generates PDF & delivers via Transactional Email   │
│    └── Score < 50: Routes to Human Review Queue (Safeguard Gate)       │
└────────────────────────────────────────────────────────────────────────┘
          │
          ├──────────────────────────────┬──────────────────────────────┐
          ▼                              ▼                              ▼
  [ Live SSE Stream ]          [ Downloadable PDF ]          [ LangSmith Trace ]
  (Next.js Dashboard)          (Official Certificate)        (Full Observability)
```

---

## 🛠️ Technology Stack

| Layer | Technology | Description |
|:---|:---|:---|
| **Frontend** | Next.js 16, React 19, TypeScript, Tailwind CSS | Real-time reactive dashboard with live Server-Sent Events (SSE) progress stepper and evidence inspector cards. |
| **Backend** | FastAPI, Python 3.11, Uvicorn, Pydantic v2 | High-performance asynchronous REST API supporting streaming endpoints and background tasks. |
| **Multimodal GenAI** | Google Gemini 3.6 Flash (`langchain-google-genai`) | Zero-shot image inspection, severity estimation, and visual evidence extraction. |
| **Orchestration** | LangGraph, LangChain Core | Stateful cyclic/acyclic graph architecture with conditional edges and quality validation gates. |
| **Hybrid RAG** | ChromaDB, BM25 (`rank_bm25`), HuggingFace Embeddings | Hybrid retrieval combining semantic vector similarity with exact keyword matching for tender contracts. |
| **Observability** | LangSmith | End-to-end tracing capturing token consumption, latency, and node execution trees. |
| **PDF Generation** | ReportLab (Backend) + jsPDF (Frontend) | Automated vector PDF document assembly with formatted tables, badges, and disclaimers. |
| **Email Delivery** | Python `smtplib` / TLS | Transactional dispatch with dual-mode live SMTP and zero-latency cloud simulation. |
| **Deployment** | Vercel (Frontend) + Railway (Backend Container) | Distributed edge hosting with automated CI/CD directly from GitHub. |

---

## 📁 Repository Structure

```text
RoadWatchAI/
├── backend/
│   ├── agents/               # Autonomous LangGraph Agent Implementations
│   │   ├── vision_agent.py          # Multimodal Gemini 3.6 Flash vision analysis
│   │   ├── location_agent.py        # GPS EXIF & location resolution
│   │   ├── road_research_agent.py   # Road maintenance database queries
│   │   ├── contract_research_agent.py # Hybrid RAG contract retriever
│   │   ├── officer_research_agent.py  # Responsible authority resolver
│   │   ├── verification_agent.py    # Cross-agent evidence validation
│   │   ├── complaint_agent.py       # Formal complaint record generator
│   │   └── quality_agent.py         # Deterministic 8-factor quality scoring
│   ├── api/                  # FastAPI service and SSE streaming logic
│   ├── database/             # Relational repository & entity models
│   ├── graph/                # LangGraph state & workflow graph definition
│   ├── rag/                  # ChromaDB vector store, BM25 ranker & loaders
│   ├── services/             # Email submission service (SMTP / Mock)
│   ├── utils/                # ReportLab PDF report generation utilities
│   ├── tests/                # Automated pytest test suite (117 tests)
│   ├── config.py             # Dynamic environment settings manager
│   ├── llm.py                # Gemini model factory & MockChatMultimodal
│   ├── main.py               # FastAPI application & entrypoint
│   └── observability.py      # LangSmith tracing setup
├── frontend/
│   ├── src/app/              # Next.js 16 App Router UI & components
│   ├── package.json          # Node dependencies
│   └── tailwind.config.ts    # Styling configuration
├── data/                     # Synthetic municipal datasets (CSVs & Ground Truth)
├── documents/contracts/      # Synthetic government tender contract PDFs
├── DEPLOYMENT.md             # Complete production deployment guide
├── Dockerfile                # Production container specification
├── Procfile                  # Railway / Cloud process definition
├── railway.json              # Railway deployment configuration
└── requirements.txt          # Python dependencies
```

---

## 🚀 Getting Started Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Yuvashree815/RoadWatchAI.git
cd RoadWatchAI
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in your configuration:
```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.6-flash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=RoadWatchAI
EMAIL_ENABLED=true
MOCK_LLM=false
MOCK_EMAIL=true
DEMO_MODE=true
```

### 3. Setup and Run Backend (FastAPI)
```bash
# Create and activate Python virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
python -m backend.main
```
Backend will be live at `http://localhost:8000`.

### 4. Setup and Run Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev
```
Frontend will be live at `http://localhost:3000`.

---

## 🧪 Testing & Verification

The project includes an automated test suite with **117 tests (100% pass rate)** covering agent logic, RAG retrieval, graph compilation, PDF generation, and API endpoints:

```bash
pytest backend/tests/ -v --ignore=backend/tests/live_gemini_test.py
```

Frontend production build check:
```bash
cd frontend
npm run build
```

---

## 🔒 Synthetic Data & Ethics Disclaimer

> **All road names, maintenance contracts, contractor profiles, officer identities, and municipal data used in this demonstration system are strictly synthetic and fictional.**
>
> This project was developed as a GenAI Capstone project to demonstrate advanced engineering principles (Multimodal LLMs, LangGraph orchestration, Hybrid RAG, and Observability) in a realistic public governance scenario.

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
