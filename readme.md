# VIT Q&A Bot — Backend

RAG-powered conversational AI for answering VIT university queries using official documents.

**Stack:** FastAPI • PostgreSQL + pgvector • OpenAI • Python

---

## Architecture

```
Frontend (Next.js)
       ↓
FastAPI Backend
   ↓       ↓
OpenAI   PostgreSQL + pgvector
(LLM)    (Vector Database)
```

**Query Flow:**
1. User message → Query rewriting (if follow-up)
2. Embed query → Vector similarity search
3. Multi-threshold scoring → Context building
4. GPT-4o-mini generates answer

**Ingestion Pipeline:**
```
PDF → Markdown → Smart Chunking → Embeddings → Database
```

---

## Key Features

- **RAG Pipeline** with semantic search
- **Multi-threshold scoring** (0.65 primary / 0.40 secondary)
- **Smart chunking** using LangChain text splitters
- **Follow-up handling** via query rewriting
- **Deduplication** using SHA-256 fingerprinting
- **Token-based batching** for efficient embedding
- **Batch PDF ingestion** with automatic processing

---

## Project Structure

```
├── app/
│   ├── final_retreval.py      # Vector retrieval
│   ├── retrieval_core.py      # Scoring + context building
│   └── query_rewrite.py       # Follow-up rewriting
├── ingestion/
│   ├── scan.py                # PDF → Markdown (marker-pdf)
│   ├── chunking.py            # Intelligent chunking
│   └── final_ingestion.py     # Embedding + DB insert
├── main.py                    # FastAPI app
└── requirements.txt
```

---

## Ingesting Documents

### 1. Prepare your PDF files

Create a folder and place your PDF documents:

```bash
documents/
├── Academic_Regulations.pdf
├── Hostel_Rules.pdf
├── Placement_Policy.pdf
└── Fee_Structure.pdf
```

### 2. Install ingestion dependencies

```bash
pip install -r requirements-ingest.txt
```

This includes `marker-pdf` for high-quality PDF extraction.

### 3. Run the ingestion pipeline

```python
from ingestion.ingest_folder import ingest_folder

# Process all PDFs in the folder
ingest_folder("documents/")
```

### What happens during ingestion:

```
PDF Files
    ↓
1. SHA-256 Fingerprinting (skip duplicates)
    ↓
2. PDF → Markdown Extraction (marker-pdf)
    ↓
3. LangChain Header Splitting (#, ##, ###, ####)
    ↓
4. TOC Detection & Filtering
    ↓
5. Recursive Chunking (max 2000 chars)
    ↓
6. OpenAI Embedding (text-embedding-3-large)
    ↓
7. Batch Insert to PostgreSQL + pgvector
```

**Features:**
- ✅ Automatic duplicate detection via fingerprinting
- ✅ Smart chunking preserves tables and context
- ✅ TOC sections automatically filtered out
- ✅ Token-based batching (80k tokens/batch)
- ✅ Status tracking (`processing`, `done`, `failed`)

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure .env
OPENAI_API_KEY=your_key
SUPABASE_URL=postgresql://...

# Run server
uvicorn main:app --reload
```

---

## Database Schema

```sql
-- Enable vector extension
CREATE EXTENSION vector;

-- Documents table
CREATE TABLE documents (
    document_id UUID PRIMARY KEY,
    document_name TEXT,
    fingerprint TEXT UNIQUE,
    status TEXT
);

-- Chunks with embeddings
CREATE TABLE document_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id UUID,
    text TEXT,
    embedding vector(3072),  -- OpenAI text-embedding-3-large
    metadata JSONB
);

-- Vector index
CREATE INDEX ON document_chunks USING hnsw (embedding vector_cosine_ops);
```

---

## API Endpoints

### `POST /chat`

```json
// Request
{
  "message": "What is the attendance policy?",
  "conversation_id": "optional-uuid"
}

// Response
{
  "reply": "The attendance policy requires...",
  "conversation_id": "uuid"
}
```

---

## Technical Highlights

**Retrieval Strategy:**
- `score ≥ 0.65` → Direct answer
- `0.40 ≤ score < 0.65` → Related context
- `score < 0.40` → "Not found in documents"

**Chunking Strategy:**
- LangChain `MarkdownHeaderTextSplitter` for header-based splitting
- Split by markdown headers (#, ##, ###, ####)
- Preserve tables (no mid-table splits)
- Auto-detect and remove TOC sections
- Recursive size management (max 2000 chars/chunk)

**Conversation Handling:**
- In-memory conversation store
- Follow-up questions rewritten to standalone queries
- Context-aware responses

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | FastAPI |
| LLM | GPT-4o-mini |
| Embeddings | text-embedding-3-large (3072d) |
| Vector DB | PostgreSQL + pgvector |
| PDF Processing | marker-pdf |
| Text Splitting | LangChain MarkdownHeaderTextSplitter |

---

## Author

**Aditya** — Full-Stack Developer