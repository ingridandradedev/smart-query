# Instalação das dependências necessárias
# !pip install -qU langchain-openai langchain-pinecone pinecone-client

import os
import getpass

from langchain_community.document_loaders import OnlinePDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document

# --- Configurações de credenciais ---

# Azure OpenAI
if not os.getenv("AZURE_OPENAI_API_KEY"):
    os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Digite sua Azure OpenAI API Key: ")
if not os.getenv("AZURE_OPENAI_ENDPOINT"):
    os.environ["AZURE_OPENAI_ENDPOINT"] = input("Digite seu Azure OpenAI Endpoint (ex: https://<seu-recurso>.openai.azure.com/): ")
if not os.getenv("AZURE_OPENAI_API_VERSION"):
    os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"  # ou outra versão suportada

# Pinecone
if not os.getenv("PINECONE_API_KEY"):
    os.environ["PINECONE_API_KEY"] = getpass.getpass("Digite sua Pinecone API Key: ")
pinecone_api_key = os.environ["PINECONE_API_KEY"]

# --- Inicialização Pinecone ---

pc = Pinecone(api_key=pinecone_api_key)

index_name = "smart-query"  # nome do índice Pinecone

# Cria o índice se não existir
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,  # dimensão padrão para embeddings Azure OpenAI
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    import time
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# --- Carregar PDF online ---

pdf_url = "https://faqdrjkeazvchresmldb.supabase.co/storage/v1/object/public/documents//Resumo_Reuniao_Patrick_Ingrid.pdf"  # substitua pela URL do seu PDF online

loader = OnlinePDFLoader(pdf_url)
docs = loader.load()  # carrega o PDF em uma lista de Documentos (um por página)

# --- Chunking recursivo ---

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

chunks = text_splitter.split_documents(docs)

print(f"Total de chunks gerados: {len(chunks)}")

# --- Inicializar embeddings Azure OpenAI ---

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

# --- Inicializar vector store Pinecone ---

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# --- Adicionar documentos ao Pinecone ---

vector_store.add_documents(chunks)

print("Chunks enviados para Pinecone com sucesso!")
