from fastapi import APIRouter
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.schemas.chat import ChatRequest, ChatResponse
from app.graph.workflow import workflow
from app.core.database import pool

router = APIRouter()

@router.post("/chat", response_model=ChatResponse) 
async def chat_endpoints(request: ChatRequest):
    saver = AsyncPostgresSaver(pool)
    app_graph = workflow.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": request.thread_id}}
    
    result = await app_graph.ainvoke(
        {"question": request.question},
        config=config
    )
    return ChatResponse(response=result["answer"])
