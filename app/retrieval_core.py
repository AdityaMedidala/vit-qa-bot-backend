from openai import OpenAI

PRIMARY_THRESHOLD = 0.65      # directly answerable
SECONDARY_THRESHOLD = 0.40    # for context chunks
MAX_CONTEXT_CHUNKS = 4 
TOP_K = 5


def get_section_key(chunk_id: str) -> str:
    return "__".join(chunk_id.split("__")[:-1])

#used for formatting insdie the answer query
def build_context (chunks):
    lines=[]
    # _ used to ignore
    for _,items in chunks:
        cid=items["chunk_id"]
        text=items["text"]
        lines.append(f"[{cid}]\n {text}")
    return "\n\n".join(lines)


def answer_query(query:str,retrieval_result):
    client=OpenAI()
    if retrieval_result["mode"]=="none":
        return "The document does not clearly specify this."
    context=build_context(retrieval_result["results"])
    system_prompt =(
        """
        You are a question answering system for official university regulations.

        Rules:
        1. Answer ONLY using the provided context.
        2. If the context fully answers the question, provide a clear answer.
        3. If the context does NOT explicitly answer the question, but contains related information:
            - State clearly that the document does not explicitly specify the answer.
            - Then summarize the related information that IS present.
        4. If the context is irrelevant or insufficient, say:
            "The document does not provide information related to this question."
        5. Be concise and factual.
        6. Do NOT guess, infer missing facts, or use outside knowledge.
        """
    )

    user_prompt=(
        f"Question: \n {query}"
        f"Context: {context}"
    )

    response=client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system", "content":system_prompt},
            {"role":"user","content":user_prompt}
        
        ],
        #used to provide deterministic model and pick token with highest probability
        temperature=0
    )

    return response.choices[0].message.content.strip()

def retrieve_from_scored_chunks(scored):
    top_score, top_item = scored[0]

    # out of scope
    if top_score < SECONDARY_THRESHOLD:
        return {
            "mode": "none",
            "top_score": top_score,
            "results": scored[:TOP_K]
        }

    # related but unclear
    if top_score < PRIMARY_THRESHOLD:
        return {
            "mode": "partial",
            "top_score": top_score,
            "results": scored[:TOP_K]
        }
    
    # directly answerable
    context = []
    seen_ids = set()

    # Threshold inclusion
    for score, item in scored:
        if score >= SECONDARY_THRESHOLD:
            context.append((score, item))
            seen_ids.add(item["chunk_id"])

        if len(context) >= MAX_CONTEXT_CHUNKS:
            break

    # Section sibling for context
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
        "results": context
    }

def embed_texts(texts:list[str])->list[list[float]]:
    client=OpenAI()
    response=client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    #loops thru results to get the embeddings
    return [item.embedding for item in response.data]
