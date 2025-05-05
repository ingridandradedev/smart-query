import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# --- Carrega variáveis ---
load_dotenv()

AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-gcp")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "smart-query")

# Validações básicas
if not AZURE_API_KEY or not AZURE_ENDPOINT:
    raise RuntimeError("Defina AZURE_OPENAI_API_KEY e AZURE_OPENAI_ENDPOINT no .env")
if not PINECONE_API_KEY:
    raise RuntimeError("Defina PINECONE_API_KEY no .env")

# --- Inicializa Pinecone ---
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(PINECONE_INDEX)

# --- FastAPI app ---
app = FastAPI(title="Ingestão de PDF + Pinecone")

class IngestRequest(BaseModel):
    pdf_url: str

@app.post("/ingest")
def ingest_pdf(req: IngestRequest):
    try:
        # Carrega e divide
        loader = OnlinePDFLoader(req.pdf_url)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        chunks = splitter.split_documents(docs)

        # Embeddings
        embeddings = AzureOpenAIEmbeddings(
            deployment="text-embedding-3-small",
            model="text-embedding-3-small",
            azure_endpoint=AZURE_ENDPOINT,
        )

        # Upsert em Pinecone
        store = PineconeVectorStore(index=index, embedding=embeddings)
        texts = [chunk.page_content for chunk in chunks]
        store.add_texts(texts)

        return {"ingested_chunks": len(chunks)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
