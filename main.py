from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import uuid

app=FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
max_messages=6
conversation_store={}
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str]=None
    

class ChatResponse(BaseModel):
    reply:str
    conversation_id:str


@app.get("/")
def health_check():
    print("Health checking")
    return{"Status": "backend is running"}

@app.post("/chat")
def chat(request: ChatRequest):
    if request.conversation_id is None:
        conversation_id=str(uuid.uuid4())
    else:
        conversation_id=request.conversation_id
    
    if conversation_id not in conversation_store:
        conversation_store[conversation_id]=[]

    conversation_store[conversation_id].append({
        "role":"user",
        "content":request.message
    })

    history1=conversation_store[conversation_id]

    previous_message=None

    for msg in reversed(history1[:-1]):
        if msg["role"] == "user":
            previous_message=msg["content"]
            break
    if previous_message:
        reply="User said "+ previous_message + request.message
    else:
        reply=request.message

    conversation_store[conversation_id].append({
            "role":"assistant",
            "content":reply
        })
    
    conversation_store[conversation_id]=conversation_store[conversation_id][-max_messages:]
    return ChatResponse (
        reply= reply,
        conversation_id=conversation_id
        
    )
   
