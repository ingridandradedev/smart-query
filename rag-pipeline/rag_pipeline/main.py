from fastapi import FastAPI
from rag_pipeline.api import router
from rag_pipeline.rag import inicializar_pinecone  # Corrigida a importação
from rag_pipeline.config import config

app = FastAPI(
    title="Smart Query RAG API",
    description="API para executar pipelines RAG com Pinecone e Azure OpenAI",
    version="1.0.0",
)

@app.get("/")
def read_root():
    return {"message": "API is running!"}

# Inicializar Pinecone
index = inicializar_pinecone(
    api_key=config["pinecone_api_key"],
    region=config["pinecone_region"],
    index_name="smart-query",
)

# Registrar os endpoints
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
