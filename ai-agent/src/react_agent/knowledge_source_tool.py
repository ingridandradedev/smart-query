from typing import Any, Dict, List
from pydantic import BaseModel, Field
import openai
from openai import OpenAI  # new way to instantiate the OpenAI client
import pinecone
from typing_extensions import Annotated
import os

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg

from react_agent.configuration import Configuration

# =============================================================================
# BASE MODEL FOR QUERY PARAMETERS
# =============================================================================

class KnowledgeBaseQueryParams(BaseModel):
    # Defines the structure of the query parameters for the knowledge base tool.
    # The query is a natural language string used to search the knowledge base.
    query: str = Field(..., description="Natural language query to search the knowledge base.")

# =============================================================================
# Function to obtain embeddings via OpenAI (new interface)
# =============================================================================

def get_embedding(text: str) -> List[float]:
    """
    Converts a text into an embedding vector using the OpenAI model 'text-embedding-3-small'.
    The text is cleaned of newline characters to avoid tokenization issues.

    Args:
        text (str): The input text to be converted into an embedding.

    Returns:
        List[float]: The embedding vector representing the input text.
    """
    clean_text = text.replace("\n", " ")  # Removes newline characters from the text.
    client = OpenAI(api_key=OPENAI_API_KEY)  # Initializes the OpenAI client with the API key.
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,  # Specifies the embedding model to use.
        input=[clean_text]  # Sends the cleaned text as input to the model.
    )
    embedding = response.data[0].embedding  # Extracts the embedding vector from the response.
    return embedding

# =============================================================================
# Configuration for OpenAI and Pinecone
# =============================================================================

# Environment variables are used to store sensitive API keys for OpenAI and Pinecone.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"  # Specifies the embedding model to use.

# =============================================================================
# Function to query the Pinecone index
# =============================================================================

def query_pinecone(
    embedding: List[float],
    index_host: str,
    namespace: str,
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Queries the Pinecone index using the provided embedding vector.
    The function requires the index host and namespace to be specified.

    Args:
        embedding (List[float]): The embedding vector to query the index.
        index_host (str): The host address of the Pinecone index.
        namespace (str): The namespace within the Pinecone index.
        top_k (int): The number of top results to retrieve (default is 10).

    Returns:
        Dict[str, Any]: A dictionary containing the most similar vectors along with their metadata.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)  # Initializes the Pinecone client with the API key.
    index = pc.Index(host=index_host)  # Connects to the specified Pinecone index.
    result = index.query(
        namespace=namespace,  # Specifies the namespace to query.
        vector=embedding,  # Embedding vector used for similarity search.
        top_k=top_k,  # Number of top results to retrieve.
        include_metadata=True,  # Includes metadata in the results.
        include_values=True  # Includes vector values in the results.
    )
    return result

# =============================================================================
# Tool: Query Knowledge Base
# =============================================================================

async def query_knowledge_base_tool(
    query: str,
    * ,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Dict[str, Any]:
    """
    Tool to query the knowledge base.

    This tool takes a natural language query, converts it into an embedding vector
    using the OpenAI model 'text-embedding-3-small', and then queries the Pinecone index.
    The values for index_host and namespace must be provided in the configuration.

    Args:
        query (str): The natural language query to search the knowledge base.
        config (RunnableConfig): Configuration object containing index_host and namespace.

    Returns:
        Dict[str, Any]: A dictionary with the key 'matches', where each item is a dictionary
                        containing the fields:
                            - source: The source file or document.
                            - text: The text content of the match.
    """
    # Extracts configuration values from the provided RunnableConfig.
    configuration = Configuration.from_runnable_config(config)
    
    # Validates that the required configuration values are provided.
    if not configuration.index_host:
        raise ValueError("Configuration 'index_host' is required but was not provided.")
    if not configuration.namespace:
        raise ValueError("Configuration 'namespace' is required but was not provided.")
    
    index_host = configuration.index_host  # Retrieves the index host from the configuration.
    namespace = configuration.namespace  # Retrieves the namespace from the configuration.

    # Validates and parses the query parameters using the KnowledgeBaseQueryParams model.
    params = KnowledgeBaseQueryParams(query=query)
    
    # Converts the query into an embedding vector using the OpenAI model.
    embedding = get_embedding(params.query)
    
    # Queries the Pinecone index using the embedding vector.
    raw_result = query_pinecone(embedding, top_k=5, index_host=index_host, namespace=namespace)
    
    # Filters the results to return only the 'source' and 'text' fields from the metadata.
    filtered_matches = []
    for match in raw_result.get("matches", []):
        metadata = match.get("metadata", {})
        filtered_matches.append({
            "source": metadata.get("source"),  # Retrieves the source field from the metadata.
            "text": metadata.get("text")  # Retrieves the text field from the metadata.
        })

    # Returns the filtered matches as the final result.
    return {"matches": filtered_matches}
