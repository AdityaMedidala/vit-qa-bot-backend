from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from app.final_retreval import retrieve_sql
from app.retrieval_core import answer_query
from app.query_rewrite import rewrite_follow_up
import uuid

client=OpenAI()

app=FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000"
        "https://vit-qa-6txe4spnu-aditya-medidalas-projects.vercel.app/"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str]=None
    

class ChatResponse(BaseModel):
    reply:str
    conversation_id:str

conversation_store: dict[str, dict[str, str]] = {}

@app.get("/")
def health_check():
    print("Health checking")
    return{"Status": "backend is running"}

@app.post("/chat")
def chat(request: ChatRequest):
    conversation_id = request.conversation_id or str(uuid.uuid4())
    is_followup = conversation_id in conversation_store
    if is_followup:
        #used get for redundancy
        state=conversation_store.get(conversation_id)
        #first question
        if not state or "last_query" not in state:
            clean_query=request.message.strip()
        else:
        #follow-up question
            clean_query=rewrite_follow_up(
                client,
                conversation_store[conversation_id]["last_query"],request.message)
    else:
        clean_query=request.message.strip()
    
    if not clean_query:
        return ChatResponse(
            reply="Please reply a relevant question",
            conversation_id=conversation_id
        )
    retrieval=retrieve_sql(clean_query)
    #isInstance for Type chechking
    if not isinstance(retrieval, dict) or "mode" not in retrieval or "results" not in retrieval:
        return ChatResponse(
        reply="An internal error occurred while retrieving information.",
        conversation_id=conversation_id
    )

    reply=answer_query(clean_query,retrieval)

    conversation_store[conversation_id]= {
        "last_query":clean_query,
        "last_answer":reply
    }
    return ChatResponse (
        reply=reply,
        conversation_id=conversation_id
    )
   
