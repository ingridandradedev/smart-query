# Instalação das dependências necessárias
# !pip install -qU langchain-openai langchain-pinecone pinecone-client python-dotenv

import os
from dotenv import load_dotenv  # Importa a biblioteca dotenv

from langchain_community.document_loaders import OnlinePDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document

# --- Carregar variáveis de ambiente do arquivo .env ---
load_dotenv()  # Carrega as variáveis de ambiente do arquivo .env

# --- Configurações de credenciais ---

# Azure OpenAI
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")  # Valor padrão

if not azure_api_key or not azure_endpoint:
    raise ValueError("As variáveis AZURE_OPENAI_API_KEY e AZURE_OPENAI_ENDPOINT devem estar definidas no arquivo .env.")

# Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("A variável PINECONE_API_KEY deve estar definida no arquivo .env.")

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
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
    openai_api_version=azure_api_version,
)

# --- Inicializar vector store Pinecone ---
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# --- Adicionar documentos ao Pinecone ---
vector_store.add_documents(chunks)

print("Chunks enviados para Pinecone com sucesso!")
