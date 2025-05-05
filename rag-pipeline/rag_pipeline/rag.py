import os
import logging
from typing import List, Dict
from dotenv import load_dotenv

from langchain_community.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document

from pinecone import Pinecone, ServerlessSpec

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def carregar_variaveis_ambiente() -> Dict[str, str]:
    """Carrega variáveis de ambiente do arquivo .env"""
    load_dotenv()
    
    config = {
        "azure_api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "azure_api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
        "pinecone_env": os.getenv("PINECONE_ENVIRONMENT", "us-east-1-gcp"),
        "pinecone_namespace": os.getenv("PINECONE_NAMESPACE", "teste"),
    }
    
    # Validar configurações
    if not config["azure_api_key"] or not config["azure_endpoint"]:
        raise ValueError("Defina AZURE_OPENAI_API_KEY e AZURE_OPENAI_ENDPOINT no .env")
    
    if not config["pinecone_api_key"]:
        raise ValueError("Defina PINECONE_API_KEY no .env")
    
    return config

def inicializar_pinecone(api_key: str, environment: str, index_name: str = "smart-query", dimension: int = 1536):
    """Inicializa o cliente Pinecone e cria o índice se necessário"""
    logger.info("Inicializando cliente Pinecone")
    pc = Pinecone(api_key=api_key, environment=environment)
    
    if index_name not in pc.list_indexes().names():
        logger.info(f"Criando índice Pinecone: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        logger.info(f"Usando índice Pinecone existente: {index_name}")
    
    return pc.Index(index_name)

def carregar_e_dividir_documento(url: str, tamanho_chunk: int = 1000, sobreposicao: int = 200) -> List[Document]:
    """Carrega um documento PDF de uma URL e o divide em chunks"""
    logger.info(f"Carregando documento de {url}")
    loader = OnlinePDFLoader(url)
    docs = loader.load()
    
    logger.info(f"Dividindo documento em chunks (tamanho={tamanho_chunk}, sobreposicao={sobreposicao})")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=tamanho_chunk,
        chunk_overlap=sobreposicao,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Total de chunks gerados: {len(chunks)}")
    
    return chunks

def inicializar_embeddings(config: Dict[str, str]):
    """Inicializa o modelo de embeddings do Azure OpenAI"""
    logger.info("Inicializando modelo de embeddings")
    return AzureOpenAIEmbeddings(
        deployment="text-embedding-3-small",
        model="text-embedding-3-small",
        api_key=config["azure_api_key"],
        api_version=config["azure_api_version"],
        azure_endpoint=config["azure_endpoint"],
    )

def processar_em_lotes(vector_store, chunks: List[Document], namespace: str, tamanho_lote: int = 50):
    """Processa chunks em lotes para evitar problemas de memória"""
    total_lotes = (len(chunks) + tamanho_lote - 1) // tamanho_lote
    
    for i in range(0, len(chunks), tamanho_lote):
        lote_atual = chunks[i:i+tamanho_lote]
        textos = [chunk.page_content for chunk in lote_atual]
        metadados = [
            {
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", 0),
                "start_index": chunk.metadata.get("start_index", 0),
                "chunk_id": i + idx
            } for idx, chunk in enumerate(lote_atual)
        ]
        
        logger.info(f"Processando lote {i//tamanho_lote + 1}/{total_lotes} ({len(textos)} chunks)")
        try:
            vector_store.add_texts(textos, metadatas=metadados, namespace=namespace)
        except Exception as e:
            logger.error(f"Erro ao processar lote {i//tamanho_lote + 1}: {e}")
            raise

def main():
    """Função principal que executa o pipeline RAG"""
    try:
        # Carregar configurações
        config = carregar_variaveis_ambiente()
        
        # Inicializar Pinecone
        index = inicializar_pinecone(
            config["pinecone_api_key"], 
            config["pinecone_env"]
        )
        
        # Carregar e dividir documento
        pdf_url = "https://faqdrjkeazvchresmldb.supabase.co/storage/v1/object/public/documents/4f1aaaf5-7801-4617-8147-83e518ea5a6e/1746385614628_marketing_plan.pdf"
        chunks = carregar_e_dividir_documento(pdf_url)
        
        # Inicializar modelo de embeddings
        embeddings = inicializar_embeddings(config)
        
        # Criar vector store
        vector_store = PineconeVectorStore(index=index, embedding=embeddings)
        
        # Processar chunks em lotes
        processar_em_lotes(vector_store, chunks, namespace=config["pinecone_namespace"])
        
        logger.info("Pipeline RAG concluído com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro no pipeline RAG: {e}")
        raise

if __name__ == "__main__":
    main()