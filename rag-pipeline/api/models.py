from pydantic import BaseModel

class RAGRequest(BaseModel):
    index_name: str
    namespace: str
    document_url: str

class RAGResponse(BaseModel):
    message: str
    details: dict
