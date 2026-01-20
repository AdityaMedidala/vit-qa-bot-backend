import json
import math
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv 
load_dotenv()
from ingestion import (
    retrieve_from_scored_chunks,
    build_context,
    answer_query,embed_texts
)

client=OpenAI()

def cosine_simalirity(query_emb,embeddings_matrix):
    query=np.array(query_emb,dtype=np.float64)

    query_norm=query/np.linalg.norm(query)
    matrix_norm=embeddings_matrix /np.linalg.norm(
        embeddings_matrix,axis=1,keepdims=True
    )
    return np.dot(matrix_norm,query_norm)


def load_chunks(path="final_chunks.json"):
    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)
    
def build_vector_store(chunks):
    #list comprehension to get the txt of the chunks 
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    embeddings_matrix=np.array(embeddings,dtype=np.float32)

    store = []
    for chunk, emb in zip(chunks, embeddings):
        store.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "embedding": emb
        })

    return store,embeddings_matrix



def retrieve_local(query: str, store, embedding_matrix):
    query=query.strip()
    if not query:
        return {
            "mode":"none",
            "top_score":"none",
            "result":[]
        }

    query_emb = embed_texts([query])[0]
    scores = cosine_simalirity(query_emb, embedding_matrix)

    scored = list(zip(scores, store))
    scored.sort(key=lambda x: x[0], reverse=True)

    return retrieve_from_scored_chunks(scored)


if __name__ == "__main__":
    chunks = load_chunks("final_chunks.json")
    store,embeddings_matrix = build_vector_store(chunks)

    print(f"Loaded {len(store)} chunks into memory")

    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() == "exit":
            break

        result = retrieve_local(query, store,embeddings_matrix)
        if result["mode"] in ("partial", "full"):
            answer = answer_query(query, result)
            print("\n" + "=" * 50)
            print("ANSWER:")
            print(answer)
            print("=" * 50)
        else:
            print("\nRefused: The document does not provide information related to this question.")
        if result["mode"] == "none" and result["results"]:
            print("Top match was:")
            print(result["results"][0][1]["text"][:150], "...")
