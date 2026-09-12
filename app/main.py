from contextlib import asynccontextmanager
from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.core.database import pool
from app.api.v1.endpoints.chat import router as chat_router
from app.api.v1.endpoints.document import router as document_router
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    await pool.open()
    saver = AsyncPostgresSaver(pool)
    await saver.setup()
    yield
    await pool.close
    
app = FastAPI(title="Enterprise RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints
app.include_router(chat_router, prefix="/api/v1")
app.include_router(document_router, prefix="/api/v1")