from fastapi import FastAPI
from rag_pipeline.api.endpoints import router

app = FastAPI(
    title="Smart Query RAG API",
    description="API para executar pipelines RAG com Pinecone e Azure OpenAI",
    version="1.0.0",
)

@app.get("/")
def read_root():
    return {"message": "API is running!"}

# Registrar os endpoints
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
