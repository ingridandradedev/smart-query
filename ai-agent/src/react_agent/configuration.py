from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional

from langchain_core.runnables import RunnableConfig, ensure_config

from react_agent import prompts


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
                           "This prompt sets the context and behavior for the agent."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-3-5-sonnet-20240620",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
                           "Should be in the form: provider/model-name."
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    tavily_token: Optional[str] = field(
        default=None,
        metadata={
            "description": "Token de autenticação para Tavily Search."
        },
    )

    # Campos para identificação do usuário
    user_id: Optional[str] = field(
        default=None,
        metadata={
            "description": "Identificador do usuário."
        },
    )
    user_name: Optional[str] = field(
        default=None,
        metadata={
            "description": "Nome do usuário."
        },
    )
    # Novos campos para variáveis adicionais
    database_schema: Optional[str] = field(
        default=None,
        metadata={
            "description": "Database schema to be used in PostgreSQL SQL tools."
        },
    )
    index_host: Optional[str] = field(
        default=None,
        metadata={
            "description": "Index host to be used in the knowledge source tool."
        },
    )
    namespace: Optional[str] = field(
        default=None,
        metadata={
            "description": "Namespace to be used in the knowledge source tool."
        },
    )

    # Campos para conexão com o PostgreSQL
    postgres_host: Optional[str] = field(
        default=None,
        metadata={
            "description": "Host do banco de dados PostgreSQL."
        },
    )
    postgres_port: Optional[int] = field(
        default=5432,
        metadata={
            "description": "Porta do banco de dados PostgreSQL."
        },
    )
    postgres_dbname: Optional[str] = field(
        default=None,
        metadata={
            "description": "Nome do banco de dados PostgreSQL."
        },
    )
    postgres_user: Optional[str] = field(
        default=None,
        metadata={
            "description": "Usuário do banco de dados PostgreSQL."
        },
    )
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
        """Create a Configuration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
