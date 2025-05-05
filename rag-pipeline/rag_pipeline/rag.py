import os
import logging
from typing import List, Dict
from dotenv import load_dotenv

from langchain_community.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings  # we'll still use this but with a real Azure SDK client
from langchain.schema import Document

import pinecone
from pinecone import ServerlessSpec   # only needed for create_index spec if you really want serverless

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def carregar_variaveis_ambiente() -> Dict[str, str]:
    load_dotenv()
    config = {
        "azure_api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "azure_api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
        "pinecone_env": os.getenv("PINECONE_ENVIRONMENT"),   # expect you set this in .env
        "pinecone_namespace": os.getenv("PINECONE_NAMESPACE", "default"),
    }
    if not config["azure_api_key"] or not config["azure_endpoint"]:
        raise ValueError("Defina AZURE_OPENAI_API_KEY e AZURE_OPENAI_ENDPOINT no .env")
    if not config["pinecone_api_key"] or not config["pinecone_env"]:
        raise ValueError("Defina PINECONE_API_KEY e PINECONE_ENVIRONMENT no .env")
    return config

def inicializar_pinecone(
    api_key: str,
    environment: str,
    index_name: str = "smart-query",
    dimension: int = 1536
):
    """Inicializa o cliente Pinecone e cria o índice se necessário."""
    logger.info("Inicializando cliente Pinecone via pinecone.init()")
    pinecone.init(api_key=api_key, environment=environment)
    existing = pinecone.list_indexes()
    if index_name not in existing:
        logger.info(f"Criando índice Pinecone: {index_name}")
        pinecone.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            # especifique spec se usar serverless:
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        logger.info(f"Usando índice Pinecone existente: {index_name}")
    return pinecone.Index(index_name)

def carregar_e_dividir_documento(
    url: str,
    tamanho_chunk: int = 1000,
    sobreposicao: int = 200
) -> List[Document]:
    logger.info(f"Carregando documento de {url}")
    loader = OnlinePDFLoader(url)
    docs = loader.load()
    logger.info(f"Dividindo documento em chunks (tamanho={tamanho_chunk}, sobreposicao={sobreposicao})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=tamanho_chunk,
        chunk_overlap=sobreposicao,
        add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Total de chunks gerados: {len(chunks)}")
    return chunks

def inicializar_embeddings(config: Dict[str, str]):
    """Inicializa o modelo de embeddings do Azure OpenAI via Azure SDK."""
    logger.info("Inicializando modelo de embeddings AzureOpenAIEmbeddings")
    from azure.ai.openai import OpenAIClient
    from azure.core.credentials import AzureKeyCredential

    client = OpenAIClient(
        endpoint=config["azure_endpoint"],
        credential=AzureKeyCredential(config["azure_api_key"])
    )
    # aqui usamos AzureOpenAIEmbeddings passando o client diretamente
    return AzureOpenAIEmbeddings(
        deployment="text-embedding-3-small",     # nome do seu deployment
        model="text-embedding-3-small",
        client=client,
        api_version=config["azure_api_version"],
    )

def processar_em_lotes(
    vector_store,
    chunks: List[Document],
    namespace: str,
    tamanho_lote: int = 50
):
    total_lotes = (len(chunks) + tamanho_lote - 1) // tamanho_lote
    for i in range(0, len(chunks), tamanho_lote):
        lote = chunks[i : i + tamanho_lote]
        textos = [c.page_content for c in lote]
        metadados = [
            {
                "source": c.metadata.get("source", "unknown"),
                "page": c.metadata.get("page", 0),
                "start_index": c.metadata.get("start_index", 0),
                "chunk_id": i + idx,
            }
            for idx, c in enumerate(lote)
        ]
        logger.info(f"Processando lote {i//tamanho_lote + 1}/{total_lotes} ({len(textos)} chunks)")
        vector_store.add_texts(texts, metadatas=metadados, namespace=namespace)

def main():
    try:
        config = carregar_variaveis_ambiente()
        index = inicializar_pinecone(
            api_key=config["pinecone_api_key"],
            environment=config["pinecone_env"],
        )
        pdf_url = (
            "https://faqdrjkeazvchresmldb.supabase.co/storage/v1/"
            "object/public/documents/"
            "4f1aaaf5-7801-4617-8147-83e518ea5a6e/"
            "1746385614628_marketing_plan.pdf"
        )
        chunks = carregar_e_dividir_documento(pdf_url)
        embeddings = inicializar_embeddings(config)
        vector_store = PineconeVectorStore(index=index, embedding=embeddings)
        processar_em_lotes(vector_store, chunks, namespace=config["pinecone_namespace"])
        logger.info("Pipeline RAG concluído com sucesso!")
    except Exception as e:
        logger.error(f"Erro no pipeline RAG: {e}")
        raise

if __name__ == "__main__":
    main()