# RoadWatch AI

## Problem Statement
Road condition monitoring and complaint resolution is often a manual, opaque, and slow process. Citizens report potholes, but correlating these reports with exact road segments, active maintenance contracts, and responsible officers is time-consuming for government departments.

## Project Status
**Current Status:** Milestone 2: Synthetic Data Foundation.
The initial frontend and backend structures have been created, along with a comprehensive synthetic dataset for evaluation and RAG.

## High-Level Architecture
- **Frontend Client:** Next.js (React) application for user interaction and progress tracking.
- **Backend Service:** FastAPI (Python) for processing and SSE streaming.
- **Orchestration Layer:** LangGraph for multi-agent workflows.
- **Intelligence Layer:** LangChain with LLM integration.
- **Relational DB:** Supabase (PostgreSQL) for structured entities.
- **Vector DB:** ChromaDB for unstructured document retrieval (RAG).

## Technology Stack
- **Frontend:** Next.js, React, Tailwind CSS, TypeScript
- **Backend:** FastAPI, Python 3.11+, Pydantic, Pytest
- **AI/ML:** LangChain, LangGraph, ChromaDB
- **Data:** Supabase (PostgreSQL)

## ⚠️ Data Disclaimer
**Synthetic Demo Data:** All government records, contracts, contractor information, and officer details used in this application are strictly **fictional records used for demonstration only**. This prototype demonstrates advanced technical capabilities (GenAI, RAG, workflow orchestration) without claiming access to real, restricted government databases.

## Synthetic Dataset & Relationships
The dataset generated in `data/` and `documents/contracts/` establishes a clear relational structure:
`Road` → `Maintenance Project` → `Contract/Tender` → `Contractor` → `Responsible Officer`

- **Structured Data**: CSV files mapping roads, contracts, contractors, and officers.
- **Unstructured Data**: 12 fictional PDF contracts containing rich text matching the structured data.
- **Ground Truth**: `data/ground_truth.json` maps demo cases to expected entities. This file is **strictly for evaluation/testing** and is not used by the application retrieval logic.

## Database Schema
The planned PostgreSQL schema (`database/schema.sql`) implements referential integrity across the synthetic entities and prepares tables for application state (`analysis_runs`, `complaints`).

## Development Milestones
1. **Milestone 1:** Project Foundation (Git, folder structure, FastAPI setup, Next.js setup, testing scaffolding).
2. **[CURRENT] Milestone 2:** Synthetic Data Foundation (Dataset generation, relational integrity, schema design, PDF generation).
3. **Milestone 3:** Agent Prototyping.
4. **Milestone 4:** LangGraph Orchestration.
5. **Milestone 5:** API & Streaming.
6. **Milestone 6:** Frontend Integration.
7. **Milestone 7:** Quality & Evaluation.
8. **Milestone 8:** Documentation & Deployment.
