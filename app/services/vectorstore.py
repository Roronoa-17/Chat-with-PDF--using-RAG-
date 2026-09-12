from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.core.config import settings

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", google_api_key=settings.GOOGLE_API_KEY,)

def get_vectorstore() -> PineconeVectorStore:
    return PineconeVectorStore(
        index_name = settings.PINECONE_INDEX_NAME,
        embedding=embeddings
    )