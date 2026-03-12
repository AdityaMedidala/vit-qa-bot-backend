from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
import psycopg2

from app.retrieval_core import (
    retrieve_from_scored_chunks,
    build_context,
    embed_texts
)

RRF_K = 60  # standard constant, don't change

def retrieval_raw(query: str, limit: int = 20):
    query = query.strip()
    query_emb = embed_texts([query])[0]
    conn = psycopg2.connect(os.getenv("SUPABASE_URL"))
    cursor = conn.cursor()

    # Vector search — returns ranked results
    cursor.execute('''
        SELECT chunk_id, text, metadata,
               1 - (embedding <=> %s::vector) AS score
        FROM document_chunks
        ORDER BY score DESC
        LIMIT %s
    ''', (query_emb, limit))
    vector_rows = cursor.fetchall()

    # BM25 full-text search — returns ranked results
    cursor.execute('''
        SELECT chunk_id, text, metadata,
               ts_rank(ts, plainto_tsquery('english', %s)) AS score
        FROM document_chunks
        WHERE ts @@ plainto_tsquery('english', %s)
        ORDER BY score DESC
        LIMIT %s
    ''', (query, query, limit))
    bm25_rows = cursor.fetchall()

    cursor.close()
    conn.close()

    # Build rank maps  {chunk_id: rank (1-based)}
    vector_ranks = {row[0]: idx + 1 for idx, row in enumerate(vector_rows)}
    bm25_ranks   = {row[0]: idx + 1 for idx, row in enumerate(bm25_rows)}

    # Collect all chunks seen in either list
    all_chunks = {}
    for row in vector_rows + bm25_rows:
        chunk_id, text, metadata, _ = row
        all_chunks[chunk_id] = {"chunk_id": chunk_id, "text": text, "metadata": metadata}

    # RRF fusion
    fused = []
    for chunk_id, item in all_chunks.items():
        v_rank = vector_ranks.get(chunk_id, limit + 1)
        b_rank = bm25_ranks.get(chunk_id,  limit + 1)
        rrf_score = 1 / (RRF_K + v_rank) + 1 / (RRF_K + b_rank)
        fused.append((rrf_score, item))

    fused.sort(key=lambda x: x[0], reverse=True)
    return fused[:limit]


def retrieve_sql(query: str):
    scored = retrieval_raw(query, limit=20)
    if scored:
        print(f"DEBUG: Query: {query}")
        print(f"DEBUG: Top RRF Score: {scored[0][0]:.4f}")
        print(f"DEBUG: Top Chunk: {scored[0][1]['text'][:100]}...")
    else:
        print("DEBUG: No chunks found")
    return retrieve_from_scored_chunks(scored)


if __name__ == "__main__":
    query_text = input("\n Ask a question \n")
    result = retrieve_sql(query_text)