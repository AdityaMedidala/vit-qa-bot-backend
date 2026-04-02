from openai import OpenAI
import re

PRIMARY_THRESHOLD   = 0.4   # score >= this → full answer
SECONDARY_THRESHOLD = 0.35  # score >= this → partial answer, else out of scope
MAX_CONTEXT_CHUNKS  = 4
MAX_SOURCES         = 2
TOP_K               = 5

_client = OpenAI()  # module-level singleton — avoids per-call connection pool churn

SYSTEM_PROMPT = """You are a helpful assistant for university queries.

Answer the user's question using only the provided context.

Guidelines:
1. Ignore irrelevant personal details and focus on the core topic (fees, dates, courses).
2. For "cost" or "education" questions, look for Tuition Fees, Program Details, or Financials.
3. If the context contains a fee table, summarize or present that data.
4. If the answer is partially available, share what you have.
5. Only say "The document does not provide information" if the context is completely unrelated.
6. If context contains HTML tags like <br>, ignore them and treat the content as plain text.
7. If table column headers appear fragmented or split across cells, reconstruct them logically before presenting.
8. When the context contains a markdown table, ALWAYS reproduce it as a table. Never convert tabular data into bullet points or prose. """
# ── Utilities ─────────────────────────────────────────────────────────────────

def get_section_key(chunk_id: str) -> str:
    return "__".join(chunk_id.split("__")[:-1])

def clean_section(section: str) -> str:
    if not section or section.upper() == "UNKNOWN":
        return ""
    section = re.sub(r"^Module:\d+\s*", "", section, flags=re.IGNORECASE)
    section = re.sub(r"\d+\s*hours?.*",  "", section, flags=re.IGNORECASE)
    section = section.strip(" -–")
    return section[:60] + ("…" if len(section) > 60 else "")

def clean_document(doc: str) -> str:
    return doc.replace("_", " ").replace("-", " ").strip()

def build_context(chunks: list) -> str:
    return "\n\n".join(f"[{item['chunk_id']}]\n {item['text']}" for _, item in chunks)

def extract_sources(chunks: list) -> list[dict]:
    seen, sources = set(), []
    for _, item in chunks:
        doc     = clean_document(item["metadata"].get("document", "Unknown"))
        section = clean_section(item["metadata"].get("level_1", ""))
        key = f"{doc}::{section}"
        if key not in seen:
            seen.add(key)
            sources.append({"document": doc, "section": section, "chunk_id": item["chunk_id"]})
        if len(sources) >= MAX_SOURCES:
            break
    return sources

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = _client.embeddings.create(model="text-embedding-3-large", input=texts)
    return [item.embedding for item in response.data]


# ── Answer generation ─────────────────────────────────────────────────────────

def _messages(query: str, retrieval: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Question:\n{query}\n\nContext:\n{build_context(retrieval['results'])}"},
    ]

def answer_query(query: str, retrieval: dict) -> str:
    if retrieval["mode"] == "none":
        return "The document does not clearly specify this."
    response = _client.chat.completions.create(
        model="gpt-4o-mini", messages=_messages(query, retrieval), temperature=0
    )
    return response.choices[0].message.content.strip()

def stream_answer_query(query: str, retrieval: dict):
    """Yields string tokens from OpenAI stream."""
    if retrieval["mode"] == "none":
        yield "The document does not clearly specify this."
        return
    stream = _client.chat.completions.create(
        model="gpt-4o-mini", messages=_messages(query, retrieval), temperature=0, stream=True
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


# ── Retrieval scoring ─────────────────────────────────────────────────────────

def retrieve_from_scored_chunks(scored: list) -> dict:
    if not scored:
        return {"mode": "none", "top_score": 0, "results": [], "sources": []}

    top_score, top_item = scored[0]

    if top_score < SECONDARY_THRESHOLD:
        return {"mode": "none",    "top_score": top_score, "results": scored[:TOP_K], "sources": []}

    if top_score < PRIMARY_THRESHOLD:
        results = scored[:TOP_K]
        return {"mode": "partial", "top_score": top_score, "results": results, "sources": extract_sources(results)}

    # Full answer — collect chunks above secondary threshold + section siblings
    seen, context = set(), []
    for score, item in scored:
        if score >= SECONDARY_THRESHOLD:
            context.append((score, item))
            seen.add(item["chunk_id"])
        if len(context) >= MAX_CONTEXT_CHUNKS:
            break

    section_key = get_section_key(top_item["chunk_id"])
    for score, item in scored:
        if item["chunk_id"] not in seen and get_section_key(item["chunk_id"]) == section_key:
            context.append((score, item))
        if len(context) >= MAX_CONTEXT_CHUNKS:
            break

    return {"mode": "full", "top_score": top_score, "results": context, "sources": extract_sources(context)}