from typing import Any, Dict, List
from pydantic import BaseModel, Field
import openai
from openai import OpenAI  # nova forma de instanciar o client
from pinecone.grpc import PineconeGRPC as Pinecone
from typing_extensions import Annotated
import os

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg

from react_agent.configuration import Configuration

# =============================================================================
# BASE MODEL PARA A QUERY
# =============================================================================

class KnowledgeBaseQueryParams(BaseModel):
    query: str = Field(..., description="Query em linguagem natural para buscar na knowledge base.")

# =============================================================================
# Função para obter embedding via OpenAI (nova interface)
# =============================================================================

def get_embedding(text: str) -> List[float]:
    """
    Converte um texto em um vetor de embedding utilizando o modelo OpenAI 'text-embedding-3-small'.
    O texto é limpo de quebras de linha para evitar problemas na tokenização.
    """
    clean_text = text.replace("\n", " ")
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[clean_text]
    )
    embedding = response.data[0].embedding
    return embedding

# =============================================================================
# Configurações para OpenAI e Pinecone
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"

# =============================================================================
# Função para consultar o índice no Pinecone
# =============================================================================

def query_pinecone(
    embedding: List[float],
    index_host: str,
    namespace: str,
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Consulta o índice do Pinecone usando o vetor de embedding fornecido.
    Recebe o index_host e o namespace a serem utilizados.
    Retorna os vetores mais similares junto com seus metadados.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=index_host)
    result = index.query(
        namespace=namespace,
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        include_values=True
    )
    return result

# =============================================================================
# Ferramenta: Query Knowledge Base
# =============================================================================

async def query_knowledge_base_tool(
    query: str,
    * ,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Dict[str, Any]:
    """
    Ferramenta para consultar a knowledge base.

    Esta ferramenta recebe uma query em linguagem natural, converte-a em um vetor de embedding
    usando o modelo OpenAI 'text-embedding-3-small', e em seguida consulta o índice do Pinecone.
    Os valores de index_host e namespace devem ser fornecidos na configuração.

    Retorna:
        Um dicionário com a chave 'matches', onde cada item é um dicionário contendo os campos:
            - file_path
            - text
    """
    configuration = Configuration.from_runnable_config(config)
    
    if not configuration.index_host:
        raise ValueError("Configuration 'index_host' is required but was not provided.")
    if not configuration.namespace:
        raise ValueError("Configuration 'namespace' is required but was not provided.")
    
    index_host = configuration.index_host
    namespace = configuration.namespace

    params = KnowledgeBaseQueryParams(query=query)
    embedding = get_embedding(params.query)
    raw_result = query_pinecone(embedding, top_k=5, index_host=index_host, namespace=namespace)
    
    # Filtra para retornar somente 'source' e 'text' dos metadados
    filtered_matches = []
    for match in raw_result.get("matches", []):
        metadata = match.get("metadata", {})
        filtered_matches.append({
            "source": metadata.get("source"),
            "text": metadata.get("text")
        })

    return {"matches": filtered_matches}
