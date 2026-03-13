from openai import OpenAI
import os
import cohere
from dotenv import load_dotenv
load_dotenv()

from app.db import get_db_connection, release_db_connection
from app.retrieval_core import (
    retrieve_from_scored_chunks,
    build_context,
    embed_texts
)

RRF_K = 60
RERANK_TOP_N = 10      # how many chunks to keep after reranking
RERANK_MODEL = "rerank-v3.5"   # Cohere's latest cross-encoder

cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))


def retrieval_raw(query: str, limit: int = 20):
    query = query.strip()
    query_emb = embed_texts([query])[0]

    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        # Vector search
        cursor.execute('''
            SELECT chunk_id, text, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM document_chunks
            ORDER BY score DESC
            LIMIT %s
        ''', (query_emb, limit))
        vector_rows = cursor.fetchall()

        # BM25 full-text search
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
    finally:
        release_db_connection(conn)

    # Build rank maps
    vector_ranks = {row[0]: idx + 1 for idx, row in enumerate(vector_rows)}
    bm25_ranks   = {row[0]: idx + 1 for idx, row in enumerate(bm25_rows)}

    # Collect all unique chunks
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


def rerank_with_cohere(query: str, fused_chunks: list) -> list:
    """
    Takes RRF-fused chunks, sends to Cohere cross-encoder, returns
    reranked list in same (score, item) format so downstream code is unchanged.

    Cross-encoder vs bi-encoder:
    - Bi-encoder (what we use for vector search): encodes query and chunk
      SEPARATELY, compares via cosine. Fast but loses interaction signal.
    - Cross-encoder (Cohere rerank): feeds [query + chunk] TOGETHER into
      the model. Sees full interaction between query tokens and chunk tokens.
      Much more accurate, but too slow to run on all chunks — so we run it
      only on the top-N from RRF (the "retrieve then rerank" pattern).
    """
    if not fused_chunks:
        return fused_chunks

    documents = [item["text"] for _, item in fused_chunks]

    response = cohere_client.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=RERANK_TOP_N,
        return_documents=False,   # we already have the text
    )

    reranked = []
    for result in response.results:
        original_score, item = fused_chunks[result.index]
        # Use Cohere's relevance score as the new score
        reranked.append((result.relevance_score, item))

    print(f"DEBUG: Pre-rerank order: {[item['chunk_id'] for _, item in fused_chunks[:5]]}")
    print(f"DEBUG: Post-rerank order: {[item['chunk_id'] for _, item in reranked[:5]]}")
    return reranked


def retrieve_sql(query: str):
    # Step 1: Hybrid retrieval + RRF
    fused = retrieval_raw(query, limit=20)

    if not fused:
        print("DEBUG: No chunks found")
        return {"mode": "none", "top_score": 0, "results": [], "sources": []}

    # Step 2: Cohere cross-encoder rerank
    reranked = rerank_with_cohere(query, fused)

    print(f"DEBUG: Query: {query}")
    print(f"DEBUG: Top Cohere score: {reranked[0][0]:.4f}")
    print(f"DEBUG: Top Chunk: {reranked[0][1]['text'][:100]}...")

    # Step 3: Threshold filtering + source extraction
    return retrieve_from_scored_chunks(reranked)


if __name__ == "__main__":
    query_text = input("\n Ask a question \n")
    result = retrieve_sql(query_text)