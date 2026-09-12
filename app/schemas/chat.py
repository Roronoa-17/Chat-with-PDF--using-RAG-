from pydantic import BaseModel

class ChatRequest(BaseModel):
    thread_id: str
    question: str

class ChatResponse(BaseModel):
    response: str