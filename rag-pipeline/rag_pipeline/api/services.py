import logging
from typing import Dict
from rag_pipeline.rag import (
    carregar_variaveis_ambiente,
    inicializar_pinecone,
    carregar_e_dividir_documento,
    inicializar_embeddings,
    processar_em_lotes,
)
from langchain_pinecone import PineconeVectorStore

logger = logging.getLogger(__name__)

def executar_pipeline_rag(index_name: str, namespace: str, document_url: str) -> Dict:
    """
    Executa o pipeline RAG com os parâmetros fornecidos.
    """
    logger.info("Iniciando pipeline RAG")
    
    # Carregar configurações
    config = carregar_variaveis_ambiente()
    
    # Inicializar Pinecone
    index = inicializar_pinecone(
        api_key=config["pinecone_api_key"],
        environment=config["pinecone_env"],
        index_name=index_name,
    )
    
    # Carregar e dividir documento
    chunks = carregar_e_dividir_documento(document_url)
    
    # Inicializar modelo de embeddings
    embeddings = inicializar_embeddings(config)
    
    # Criar vector store
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    
    # Processar chunks em lotes
    processar_em_lotes(vector_store, chunks, namespace=namespace)
    
    logger.info("Pipeline RAG concluído com sucesso!")
    return {"index_name": index_name, "namespace": namespace, "chunks_processed": len(chunks)}
