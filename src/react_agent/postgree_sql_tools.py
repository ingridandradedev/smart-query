import re
from typing import Any, Dict, List, Union
from pydantic import BaseModel, Field, validator
import psycopg2
from psycopg2.extras import RealDictCursor
from typing_extensions import Annotated
import os

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from react_agent.configuration import Configuration

# =============================================================================
# Configuração do PostgreSQL (exceto schema, que agora virá da config)
# =============================================================================

# Remova as credenciais fixas
# POSTGRES_HOST = "aws-0-sa-east-1.pooler.supabase.com"
# POSTGRES_PORT = 6543
# POSTGRES_DBNAME = "postgres"
# POSTGRES_USER = "postgres.vjhqaxnridfwycamgexo"
# POSTGRES_PASSWORD = "XIAVUxyAvTH2nXEp"

# Substitua por chamadas às variáveis de ambiente
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))  # Valor padrão: 5432
POSTGRES_DBNAME = os.getenv("POSTGRES_DBNAME")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

# =============================================================================
# LISTAR TABELAS (LIST TABLES)
# =============================================================================

def _list_tables(schema: str) -> Dict[str, Any]:
    """
    Lista as tabelas do schema informado do banco de dados PostgreSQL.
    """
    connection = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DBNAME,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )
    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            query = "SELECT table_name FROM information_schema.tables WHERE table_schema = %s"
            cursor.execute(query, (schema,))
            tables = cursor.fetchall()
        return {"tables": [table["table_name"] for table in tables]}
    finally:
        connection.close()

async def list_tables_tool(
    * ,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Dict[str, Any]:
    """
    Ferramenta para listar as tabelas do schema informado no banco de dados PostgreSQL.
    """
    configuration = Configuration.from_runnable_config(config)
    if not configuration.database_schema:
        raise ValueError("Configuration 'database_schema' is required but was not provided.")
    db_schema = configuration.database_schema
    return _list_tables(db_schema)

# =============================================================================
# OBTÉM COLUNAS DE UMA OU MAIS TABELAS
# =============================================================================

class GetTableColumnsParams(BaseModel):
    table_names: Union[str, List[str]] = Field(
        ..., description="Nome ou lista de nomes das tabelas para obter as colunas"
    )

    @validator("table_names", pre=True)
    def ensure_list(cls, v):
        if isinstance(v, str):
            return [v]
        return v

def _get_columns_for_tables(params: GetTableColumnsParams, schema: str) -> Dict[str, Any]:
    """
    Obtém as colunas para cada tabela da lista fornecida, utilizando o schema informado.
    Retorna um dicionário onde cada chave é o nome da tabela e o valor é uma lista de colunas.
    """
    result: Dict[str, Any] = {}
    connection = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DBNAME,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )
    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position;
            """
            for table in params.table_names:
                cursor.execute(query, (schema, table))
                columns = cursor.fetchall()
                result[table] = columns
        return {"tables_columns": result}
    finally:
        connection.close()

async def get_table_columns_tool(
    table_names: Union[str, List[str]],
    * ,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Dict[str, Any]:
    """
    Ferramenta para obter as colunas de uma ou mais tabelas específicas no schema informado do PostgreSQL.
    """
    configuration = Configuration.from_runnable_config(config)
    if not configuration.database_schema:
        raise ValueError("Configuration 'database_schema' is required but was not provided.")
    db_schema = configuration.database_schema
    params = GetTableColumnsParams(table_names=table_names)
    return _get_columns_for_tables(params, db_schema)

# =============================================================================
# EXECUTA QUERY SQL (READ-ONLY)
# =============================================================================

class ExecuteSQLQueryParams(BaseModel):
    query: str = Field(
        ...,
        description=(
            "SQL query a ser executada (apenas comandos read-only são permitidos). "
            "Comandos proibidos: INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE."
            " Comandos permitidos: SELECT, WITH, TABLE, EXPLAIN, VALUES, SHOW, DESCRIBE."
        )
    )

def validate_query(query: str) -> None:
    """
    Valida a query para garantir que não contenha comandos que alteram dados.
    """
    forbidden_keywords = ["insert", "update", "delete", "drop", "alter", "truncate", "create"]
    for keyword in forbidden_keywords:
        pattern = r'\b' + keyword + r'\b'
        if re.search(pattern, query, re.IGNORECASE):
            raise ValueError(f"Queries contendo '{keyword}' não são permitidas.")

def _execute_sql_query(params: ExecuteSQLQueryParams, schema: str) -> Dict[str, Any]:
    """
    Executa uma query SQL read-only no banco de dados PostgreSQL utilizando o schema informado.
    """
    validate_query(params.query)
    connection = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DBNAME,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )
    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            # Define o schema para a sessão
            cursor.execute("SET search_path TO %s", (schema,))
            cursor.execute(params.query)
            try:
                result = cursor.fetchall()
            except psycopg2.ProgrammingError:
                result = []
        return {"result": result}
    finally:
        connection.close()

async def execute_sql_query_tool(
    query: str,
    * ,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Dict[str, Any]:
    """
    Ferramenta para executar uma query SQL read-only no banco de dados PostgreSQL com schema informado.
    """
    configuration = Configuration.from_runnable_config(config)
    if not configuration.database_schema:
        raise ValueError("Configuration 'database_schema' is required but was not provided.")
    db_schema = configuration.database_schema
    params = ExecuteSQLQueryParams(query=query)
    return _execute_sql_query(params, db_schema)
