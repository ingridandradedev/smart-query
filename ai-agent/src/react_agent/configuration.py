from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional

from langchain_core.runnables import RunnableConfig, ensure_config

from react_agent import prompts


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    # The system prompt that defines the context and behavior of the agent.
    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
                           "This prompt sets the context and behavior for the agent."
        },
    )

    # The name of the language model to use for the agent's main interactions.
    # This should follow the format: provider/model-name.
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-3-5-sonnet-20240620",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
                           "Should be in the form: provider/model-name."
        },
    )

    # The maximum number of search results to return for each search query.
    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    # Optional authentication token for Tavily Search.
    tavily_token: Optional[str] = field(
        default=None,
        metadata={
            "description": "Token de autenticação para Tavily Search."
        },
    )

    # User identification fields.
    # Optional user identifier.
    user_id: Optional[str] = field(
        default=None,
        metadata={
            "description": "Identificador do usuário."
        },
    )
    # Optional user name.
    user_name: Optional[str] = field(
        default=None,
        metadata={
            "description": "Nome do usuário."
        },
    )

    # Additional variables for configuration.
    # Optional database schema for PostgreSQL SQL tools.
    database_schema: Optional[str] = field(
        default=None,
        metadata={
            "description": "Database schema to be used in PostgreSQL SQL tools."
        },
    )
    # Optional index host for the knowledge source tool.
    index_host: Optional[str] = field(
        default=None,
        metadata={
            "description": "Index host to be used in the knowledge source tool."
        },
    )
    # Optional namespace for the knowledge source tool.
    namespace: Optional[str] = field(
        default=None,
        metadata={
            "description": "Namespace to be used in the knowledge source tool."
        },
    )

    # PostgreSQL connection fields.
    # Optional PostgreSQL database host.
    postgres_host: Optional[str] = field(
        default=None,
        metadata={
            "description": "Host do banco de dados PostgreSQL."
        },
    )
    # Optional PostgreSQL database port, with a default value of 5432.
    postgres_port: Optional[int] = field(
        default=5432,
        metadata={
            "description": "Porta do banco de dados PostgreSQL."
        },
    )
    # Optional PostgreSQL database name.
    postgres_dbname: Optional[str] = field(
        default=None,
        metadata={
            "description": "Nome do banco de dados PostgreSQL."
        },
    )
    # Optional PostgreSQL database user.
    postgres_user: Optional[str] = field(
        default=None,
        metadata={
            "description": "Usuário do banco de dados PostgreSQL."
        },
    )
    # Optional PostgreSQL database password.
    postgres_password: Optional[str] = field(
        default=None,
        metadata={
            "description": "Senha do banco de dados PostgreSQL."
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """
        Create a Configuration instance from a RunnableConfig object.

        This method takes an optional RunnableConfig object, extracts its
        "configurable" dictionary, and maps its keys to the fields of the
        Configuration class. Only fields defined in the Configuration class
        are included in the resulting instance.
        """
        # Ensure the provided config is valid and has the necessary structure.
        config = ensure_config(config)
        # Extract the "configurable" dictionary from the config.
        configurable = config.get("configurable") or {}
        # Get the set of field names defined in the Configuration class.
        _fields = {f.name for f in fields(cls) if f.init}
        # Create an instance of Configuration using only the matching fields.
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
