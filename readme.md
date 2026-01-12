
---

# 📘 README — Backend  
**vit-qa-bot-backend**

```md
# VIT Q&A Bot — Backend

A FastAPI-based backend for a conversational AI assistant designed to answer VIT-related questions.

This service handles:
- API contracts
- conversation state
- request validation
- chat history management

⚠️ This project is **intentionally incomplete** and is being built in phases.

---

## 🎯 Project Goal

To build a **robust, explainable AI backend** that can evolve from a simple chat API into a full Retrieval-Augmented Generation (RAG) system.

---

## 🧩 Current Phase

### ✅ Phase 1 — API & Conversation State (Complete)

- REST API using FastAPI
- Strict request/response validation (Pydantic)
- Conversation tracking via UUIDs
- In-memory conversation store
- Automatic history trimming
- CORS configuration for browser clients

This phase focuses **only on infrastructure**, not AI.

---

## 🔜 Planned Phases

### 🔄 Phase 2 — Intelligence Layer
- LLM integration
- Prompt construction using chat history
- RAG with vector database (Pinecone / pgvector)
- Source grounding using VIT documents

### 🔐 Phase 3 — Production Hardening
- Persistent storage (DB-backed conversations)
- Authentication and user sessions
- Rate limiting and security controls
- Unit and integration testing
- Observability and logging

---

## 🧠 Architecture Overview

- Stateless HTTP API
- Conversation context passed via `conversation_id`
- Frontend sends only the latest user message
- Backend reconstructs relevant context internally

## 🛠 Tech Stack

- Python 3.10+
- FastAPI
- Pydantic
- Uvicorn

---

## ▶️ Running Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
