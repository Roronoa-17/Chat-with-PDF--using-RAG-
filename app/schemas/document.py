from pydantic import BaseModel

class DocumentUploadResponse(BaseModel):
    filename: str
    total_chunks: int
    message: str