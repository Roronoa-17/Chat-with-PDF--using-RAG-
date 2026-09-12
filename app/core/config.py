import os
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

class Settings(BaseModel):
    GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")
    PINECONE_API_KEY: str = os.environ.get("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.environ.get("PINECONE_INDEX_NAME", "enterprise-rage-index")
    DATABASE_URL: str = os.environ.get(
        "DATABASE_URL",
        "postgresql://user:password@localhost:5432/rag_memory"
    )
    
settings = Settings()
