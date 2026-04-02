import os
import cohere
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

from app.db import get_db_connection, release_db_connection
from app.retrieval_core import retrieve_from_scored_chunks, embed_texts

RRF_K         = 60
RERANK_TOP_N  = 10
RERANK_MODEL  = "rerank-v3.5"

_cohere = cohere.Client(os.getenv("COHERE_API_KEY"))


def retrieval_raw(query: str, limit: int = 20) -> list:
    """Hybrid BM25 + vector search fused with RRF."""
    query_emb = embed_texts([query.strip()])[0]
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT chunk_id, text, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM document_chunks ORDER BY score DESC LIMIT %s
        """, (query_emb, limit))
        vector_rows = cur.fetchall()

        cur.execute("""
            SELECT chunk_id, text, metadata,
                   ts_rank(ts, plainto_tsquery('english', %s)) AS score
            FROM document_chunks
            WHERE ts @@ plainto_tsquery('english', %s)
            ORDER BY score DESC LIMIT %s
        """, (query, query, limit))
        bm25_rows = cur.fetchall()
        cur.close()
    finally:
        release_db_connection(conn)

    vector_ranks = {row[0]: i + 1 for i, row in enumerate(vector_rows)}
    bm25_ranks   = {row[0]: i + 1 for i, row in enumerate(bm25_rows)}

    # Deduplicate and fuse
    all_chunks = {
        row[0]: {"chunk_id": row[0], "text": row[1], "metadata": row[2]}
        for row in vector_rows + bm25_rows
    }

    fused = sorted(
        [
            (1 / (RRF_K + vector_ranks.get(cid, limit + 1)) +
             1 / (RRF_K + bm25_ranks.get(cid,   limit + 1)), item)
            for cid, item in all_chunks.items()
        ],
        key=lambda x: x[0], reverse=True,
    )
    return fused[:limit]


def rerank_with_cohere(query: str, fused: list) -> list:
    """
    Cross-encoder reranking on top of RRF results.
    Bi-encoder (vector search) encodes query and chunk separately — fast but
    loses token-level interaction. Cross-encoder sees both together, giving
    more accurate scores. We only run it on the top-20 from RRF (retrieve → rerank).
    """
    if not fused:
        return fused

    response = _cohere.rerank(
        model=RERANK_MODEL, query=query,
        documents=[item["text"] for _, item in fused],
        top_n=RERANK_TOP_N, return_documents=False,
    )

    reranked = [(r.relevance_score, fused[r.index][1]) for r in response.results]

    print(f"DEBUG pre-rerank:  {[item['chunk_id'] for _, item in fused[:5]]}")
    print(f"DEBUG post-rerank: {[item['chunk_id'] for _, item in reranked[:5]]}")
    return reranked


def retrieve_sql(query: str) -> dict:
    fused = retrieval_raw(query)
    if not fused:
        print("DEBUG: No chunks found")
        return {"mode": "none", "top_score": 0, "results": [], "sources": []}

    reranked = rerank_with_cohere(query, fused)
    print(f"DEBUG query: {query}")
    print(f"DEBUG top score: {reranked[0][0]:.4f} | chunk: {reranked[0][1]['text'][:80]}...")
    return retrieve_from_scored_chunks(reranked)


if __name__ == "__main__":
    result = retrieve_sql(input("\nAsk a question\n> "))