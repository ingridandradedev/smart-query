import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# --- Load environment variables ---
# Load variables from a .env file to access sensitive information like API keys and endpoints.
load_dotenv()

# Azure OpenAI configuration
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")  # API key for Azure OpenAI
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  # Endpoint for Azure OpenAI
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")  # API version (default: 2024-02-01)

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # API key for Pinecone
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-gcp")  # Pinecone environment (default: us-east-1-gcp)
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "smart-query")  # Pinecone index name (default: smart-query)

# --- Basic validations ---
# Ensure that critical environment variables are set; otherwise, raise an error.
if not AZURE_API_KEY or not AZURE_ENDPOINT:
    raise RuntimeError("Defina AZURE_OPENAI_API_KEY e AZURE_OPENAI_ENDPOINT no .env")
if not PINECONE_API_KEY:
    raise RuntimeError("Defina PINECONE_API_KEY no .env")

# --- Initialize Pinecone ---
# Initialize the Pinecone client with the provided API key and environment.
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Check if the specified index exists in Pinecone; if not, create it.
if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX,  # Name of the index
        dimension=1536,  # Dimensionality of the embeddings
        metric="cosine",  # Similarity metric to use (cosine similarity)
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # Serverless configuration
    )
# Retrieve the index for further operations.
index = pc.Index(PINECONE_INDEX)

# --- FastAPI app ---
# Create a FastAPI application for handling PDF ingestion and vector storage.
app = FastAPI(title="Ingestão de PDF + Pinecone")

# Define the request model for the `/ingest` endpoint.
class IngestRequest(BaseModel):
    pdf_url: str  # URL of the PDF to be ingested

@app.post("/ingest")
def ingest_pdf(req: IngestRequest):
    try:
        # --- Load and split the PDF ---
        # Use an online PDF loader to fetch the document from the provided URL.
        loader = OnlinePDFLoader(req.pdf_url)
        docs = loader.load()  # Load the PDF content into a document object.

        # Split the document into smaller chunks for processing.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Maximum size of each chunk
            chunk_overlap=200,  # Overlap between chunks to preserve context
            add_start_index=True  # Include the start index of each chunk
        )
        chunks = splitter.split_documents(docs)  # Split the document into chunks.

        # --- Generate embeddings ---
        # Use Azure OpenAI to generate embeddings for the text chunks.
        embeddings = AzureOpenAIEmbeddings(
            deployment="text-embedding-3-small",  # Deployment name for the embedding model
            model="text-embedding-3-small",  # Model name
            azure_endpoint=AZURE_ENDPOINT,  # Azure endpoint for the OpenAI service
        )

        # --- Upsert into Pinecone ---
        # Create a Pinecone vector store to manage the embeddings.
        store = PineconeVectorStore(index=index, embedding=embeddings)

        # Extract the text content from the chunks and add them to the vector store.
        texts = [chunk.page_content for chunk in chunks]
        store.add_texts(texts)  # Add the text chunks to Pinecone.

        # Return the number of ingested chunks as a response.
        return {"ingested_chunks": len(chunks)}

    except Exception as e:
        # Handle any errors that occur during the process and return a 500 status code.
        raise HTTPException(status_code=500, detail=str(e))
