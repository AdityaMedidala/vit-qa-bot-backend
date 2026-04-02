from dotenv import load_dotenv
load_dotenv()

import uuid, json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI

from app.final_retreval import retrieve_sql
from app.retrieval_core import answer_query, stream_answer_query
from app.query_rewrite import rewrite_follow_up

app = FastAPI()
client = OpenAI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# In-memory conversation store — resets on restart (fine for now)
conversation_store: dict[str, dict] = {}


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class Source(BaseModel):
    document: str; section: str; chunk_id: str

class ChatResponse(BaseModel):
    reply: str; conversation_id: str; sources: list[Source] = []


def resolve_query(req: ChatRequest) -> tuple[str, str]:
    """Resolves conversation_id and rewrites follow-up queries into standalone ones."""
    cid = req.conversation_id or str(uuid.uuid4())
    state = conversation_store.get(cid)
    if state and "last_query" in state:
        query = rewrite_follow_up(client, state["last_query"], req.message)
    else:
        query = req.message.strip()
    return cid, query


def _retrieval_or_error(query: str) -> dict | None:
    result = retrieve_sql(query)
    if not isinstance(result, dict) or "mode" not in result:
        return None
    return result


@app.get("/")
def health_check():
    return {"status": "backend is running"}


@app.post("/chat")
def chat(req: ChatRequest):
    cid, query = resolve_query(req)

    if not query:
        return ChatResponse(reply="Please ask a relevant question.", conversation_id=cid)

    retrieval = _retrieval_or_error(query)
    if not retrieval:
        return ChatResponse(reply="An internal error occurred.", conversation_id=cid)

    reply = answer_query(query, retrieval)
    conversation_store[cid] = {"last_query": query, "last_answer": reply}
    return ChatResponse(reply=reply, conversation_id=cid, sources=retrieval.get("sources", []))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    cid, query = resolve_query(req)

    def generate():
        yield f"event: meta\ndata: {json.dumps({'conversation_id': cid})}\n\n"

        if not query:
            yield f"event: error\ndata: {json.dumps({'message': 'Please ask a relevant question.'})}\n\n"
            return

        retrieval = _retrieval_or_error(query)
        if not retrieval:
            yield f"event: error\ndata: {json.dumps({'message': 'An internal error occurred.'})}\n\n"
            return

        full_reply = ""
        for token in stream_answer_query(query, retrieval):
            full_reply += token
            yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

        yield f"event: sources\ndata: {json.dumps({'sources': retrieval.get('sources', [])})}\n\n"
        yield f"event: done\ndata: {{}}\n\n"

        conversation_store[cid] = {"last_query": query, "last_answer": full_reply}

    return StreamingResponse(generate(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })