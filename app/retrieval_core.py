from openai import OpenAI
import re

PRIMARY_THRESHOLD = 0.4  # directly answerable (Cohere scores 0-1)
SECONDARY_THRESHOLD = 0.35  # only include closely relevant chunks
MAX_CONTEXT_CHUNKS = 4
MAX_SOURCES = 2  # max citations shown to user
TOP_K = 5


def get_section_key(chunk_id: str) -> str:
    return "__".join(chunk_id.split("__")[:-1])


def clean_section(section: str) -> str:
    if not section or section.upper() == "UNKNOWN":
        return ""
    # Strip "Module:N " prefix and truncate long headers
    section = re.sub(r"^Module:\d+\s*", "", section, flags=re.IGNORECASE)
    section = re.sub(r"\d+\s*hours?.*", "", section, flags=re.IGNORECASE)
    section = section.strip(" -–")
    return section[:60] + ("…" if len(section) > 60 else "")


def clean_document(doc: str) -> str:
    # "B.Tech_IT_Curriculam_Syllabus_2022-2023" → "B.Tech IT Curriculum Syllabus 2022-23"
    doc = doc.replace("_", " ").replace("-", " ").strip()
    return doc


def extract_sources(chunks: list) -> list[dict]:
    seen = set()
    sources = []
    for _, item in chunks:
        doc = clean_document(item["metadata"].get("document", "Unknown"))
        section = clean_section(item["metadata"].get("level_1", ""))
        key = f"{doc}::{section}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "document": doc,
                "section": section,
                "chunk_id": item["chunk_id"],
            })
        if len(sources) >= MAX_SOURCES:
            break
    return sources


# used for formatting insdie the answer query
def build_context(chunks):
    lines = []
    # _ used to ignore
    for _, items in chunks:
        cid = items["chunk_id"]
        text = items["text"]
        lines.append(f"[{cid}]\n {text}")
    return "\n\n".join(lines)


def answer_query(query: str, retrieval_result):
    client = OpenAI()
    if retrieval_result["mode"] == "none":
        return "The document does not clearly specify this."
    context = build_context(retrieval_result["results"])
    system_prompt = (
        """
         You are a helpful assistant for university queries.
 
         Your Goal: Answer the user's question using the provided Context.
 
         Guidelines:
         1. **Ignore irrelevant user details** (e.g., "my sister", "12th grade", "I'm curious") and focus on the core topic (e.g., fees, dates, courses).
         2. If the user asks about "cost" or "education," look for **Tuition Fees**, **Program Details**, or **Financials** in the context.
         3. If the context contains a fee table, **summarize or present that data** even if it doesn't explicitly say "Total Cost for 4 years".
         4. If the answer is partially available, share what you have.
         5. Only say "The document does not provide information" if the context is completely unrelated (e.g., asking about fees but context is about sports).
         """
    )

    user_prompt = (
        f"Question: \n {query}"
        f"Context: {context}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}

        ],
        # used to provide deterministic model and pick token with highest probability
        temperature=0
    )

    return response.choices[0].message.content.strip()


def retrieve_from_scored_chunks(scored):
    if not scored:
        return {"mode": "none", "top_score": 0, "results": [], "sources": []}

    top_score, top_item = scored[0]

    # out of scope
    if top_score < SECONDARY_THRESHOLD:
        return {
            "mode": "none",
            "top_score": top_score,
            "results": scored[:TOP_K],
            "sources": [],
        }

    # related but unclear
    if top_score < PRIMARY_THRESHOLD:
        results = scored[:TOP_K]
        return {
            "mode": "partial",
            "top_score": top_score,
            "results": results,
            "sources": extract_sources(results),
        }

    # directly answerable — only include chunks above SECONDARY_THRESHOLD
    context = []
    seen_ids = set()

    for score, item in scored:
        if score >= SECONDARY_THRESHOLD:
            context.append((score, item))
            seen_ids.add(item["chunk_id"])
        if len(context) >= MAX_CONTEXT_CHUNKS:
            break

    # section siblings
    section_key = get_section_key(top_item["chunk_id"])
    for score, item in scored:
        if item["chunk_id"] in seen_ids:
            continue
        if get_section_key(item["chunk_id"]) == section_key:
            context.append((score, item))
        if len(context) >= MAX_CONTEXT_CHUNKS:
            break

    return {
        "mode": "full",
        "top_score": top_score,
        "results": context,
        "sources": extract_sources(context),
    }


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    # loops thru results to get the embeddings
    return [item.embedding for item in response.data]