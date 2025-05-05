#tools.py

"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example).

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast

# Importing TavilySearchResults for performing web searches.
from langchain_community.tools.tavily_search import TavilySearchResults
# Importing RunnableConfig for configuration management.
from langchain_core.runnables import RunnableConfig
# Importing InjectedToolArg for dependency injection of tool arguments.
from langchain_core.tools import InjectedToolArg
# Importing Annotated for type annotations with additional metadata.
from typing_extensions import Annotated

# Importing custom tools and configurations from the react_agent module.
from react_agent.configuration import Configuration
from react_agent.postgree_sql_tools import list_tables_tool, get_table_columns_tool, execute_sql_query_tool
from react_agent.knowledge_source_tool import query_knowledge_base_tool

async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.

    Args:
        query (str): The search query string.
        config (Annotated[RunnableConfig, InjectedToolArg]): Configuration object injected as a tool argument.

    Returns:
        Optional[list[dict[str, Any]]]: A list of search results, where each result is a dictionary.
    """
    # Extracting configuration settings from the provided RunnableConfig.
    configuration = Configuration.from_runnable_config(config)
    # Wrapping the TavilySearchResults tool with the maximum number of results specified in the configuration.
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    # Performing the asynchronous search operation using the Tavily search engine.
    result = await wrapped.ainvoke({"query": query})
    # Casting the result to the expected type and returning it.
    return cast(list[dict[str, Any]], result)

# TOOLS is a list of callable tools that can be used by the system.
# It includes the search function, tools for interacting with PostgreSQL databases,
# and a tool for querying a knowledge base.
TOOLS: List[Callable[..., Any]] = [
    search, 
    list_tables_tool, 
    get_table_columns_tool, 
    execute_sql_query_tool, 
    query_knowledge_base_tool
]