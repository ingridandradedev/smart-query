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

# Environment variables are used to configure PostgreSQL connection details.
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))  # Default value: 5432
POSTGRES_DBNAME = os.getenv("POSTGRES_DBNAME")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

# =============================================================================
# LIST TABLES
# =============================================================================

def _list_tables(schema: str, config: Configuration) -> Dict[str, Any]:
    """
    Lists all tables within the specified schema in the PostgreSQL database.

    Args:
        schema (str): The schema name to list tables from.
        config (Configuration): Configuration object containing database connection details.

    Returns:
        Dict[str, Any]: A dictionary containing a list of table names.
    """
    # Establish a connection to the PostgreSQL database using the provided configuration.
    connection = psycopg2.connect(
        host=config.postgres_host,
        port=config.postgres_port,
        dbname=config.postgres_dbname,
        user=config.postgres_user,
        password=config.postgres_password
    )
    try:
        # Use a cursor to execute the query and fetch table names from the schema.
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            query = "SELECT table_name FROM information_schema.tables WHERE table_schema = %s"
            cursor.execute(query, (schema,))
            tables = cursor.fetchall()
        # Return the list of table names in a dictionary format.
        return {"tables": [table["table_name"] for table in tables]}
    finally:
        # Ensure the database connection is closed after the operation.
        connection.close()

async def list_tables_tool(
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Dict[str, Any]:
    """
    Tool to list tables within the specified schema in the PostgreSQL database.

    Args:
        config (RunnableConfig): Configuration object injected with database details.

    Returns:
        Dict[str, Any]: A dictionary containing a list of table names.
    """
    # Extract configuration details from the provided RunnableConfig.
    configuration = Configuration.from_runnable_config(config)
    if not configuration.database_schema:
        raise ValueError("Configuration 'database_schema' is required but was not provided.")
    db_schema = configuration.database_schema
    # Call the helper function to list tables.
    return _list_tables(db_schema, config=configuration)

# =============================================================================
# GET COLUMNS FOR ONE OR MORE TABLES
# =============================================================================

class GetTableColumnsParams(BaseModel):
    """
    Model to validate and structure input parameters for fetching table columns.
    """
    table_names: Union[str, List[str]] = Field(
        ..., description="Name or list of table names to fetch columns for"
    )

    @validator("table_names", pre=True)
    def ensure_list(cls, v):
        """
        Ensures that the table_names parameter is always a list, even if a single string is provided.
        """
        if isinstance(v, str):
            return [v]
        return v

def _get_columns_for_tables(params: GetTableColumnsParams, schema: str, config: Configuration) -> Dict[str, Any]:
    """
    Fetches columns for each table in the provided list, using the specified schema.

    Args:
        params (GetTableColumnsParams): Parameters containing table names.
        schema (str): The schema name to fetch columns from.
        config (Configuration): Configuration object containing database connection details.

    Returns:
        Dict[str, Any]: A dictionary where each key is a table name and the value is a list of columns.
    """
    result: Dict[str, Any] = {}
    # Establish a connection to the PostgreSQL database using the provided configuration.
    connection = psycopg2.connect(
        host=config.postgres_host,
        port=config.postgres_port,
        dbname=config.postgres_dbname,
        user=config.postgres_user,
        password=config.postgres_password
    )
    try:
        # Use a cursor to execute the query and fetch column details for each table.
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
        # Return the column details in a dictionary format.
        return {"tables_columns": result}
    finally:
        # Ensure the database connection is closed after the operation.
        connection.close()

async def get_table_columns_tool(
    table_names: Union[str, List[str]],
    * ,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Dict[str, Any]:
    """
    Tool to fetch columns for one or more specific tables within the specified schema in PostgreSQL.

    Args:
        table_names (Union[str, List[str]]): Name or list of table names to fetch columns for.
        config (RunnableConfig): Configuration object injected with database details.

    Returns:
        Dict[str, Any]: A dictionary where each key is a table name and the value is a list of columns.
    """
    # Extract configuration details from the provided RunnableConfig.
    configuration = Configuration.from_runnable_config(config)
    if not configuration.database_schema:
        raise ValueError("Configuration 'database_schema' is required but was not provided.")
    db_schema = configuration.database_schema
    # Validate and structure the input parameters.
    params = GetTableColumnsParams(table_names=table_names)
    # Call the helper function to fetch columns for the tables.
    return _get_columns_for_tables(params, db_schema, config=configuration)

# =============================================================================
# EXECUTE SQL QUERY (READ-ONLY)
# =============================================================================

class ExecuteSQLQueryParams(BaseModel):
    """
    Model to validate and structure input parameters for executing SQL queries.
    """
    query: str = Field(
        ...,
        description=(
            "SQL query to be executed (only read-only commands are allowed). "
            "Prohibited commands: INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE. "
            "Allowed commands: SELECT, WITH, TABLE, EXPLAIN, VALUES, SHOW, DESCRIBE."
        )
    )

def validate_query(query: str) -> None:
    """
    Validates the SQL query to ensure it does not contain prohibited commands.

    Args:
        query (str): The SQL query to validate.

    Raises:
        ValueError: If the query contains any prohibited commands.
    """
    forbidden_keywords = ["insert", "update", "delete", "drop", "alter", "truncate", "create"]
    for keyword in forbidden_keywords:
        pattern = r'\b' + keyword + r'\b'
        if re.search(pattern, query, re.IGNORECASE):
            raise ValueError(f"Queries containing '{keyword}' are not allowed.")

def _execute_sql_query(params: ExecuteSQLQueryParams, schema: str, config: Configuration) -> Dict[str, Any]:
    """
    Executes a read-only SQL query in the PostgreSQL database using the specified schema.

    Args:
        params (ExecuteSQLQueryParams): Parameters containing the SQL query.
        schema (str): The schema name to execute the query within.
        config (Configuration): Configuration object containing database connection details.

    Returns:
        Dict[str, Any]: A dictionary containing the query result.
    """
    # Validate the query to ensure it is read-only.
    validate_query(params.query)
    # Establish a connection to the PostgreSQL database using the provided configuration.
    connection = psycopg2.connect(
        host=config.postgres_host,
        port=config.postgres_port,
        dbname=config.postgres_dbname,
        user=config.postgres_user,
        password=config.postgres_password
    )
    try:
        # Use a cursor to execute the query and fetch results.
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            # Set the schema for the session.
            cursor.execute("SET search_path TO %s", (schema,))
            cursor.execute(params.query)
            try:
                # Fetch all results from the query.
                result = cursor.fetchall()
            except psycopg2.ProgrammingError:
                # Handle cases where the query does not return rows.
                result = []
        # Return the query result in a dictionary format.
        return {"result": result}
    finally:
        # Ensure the database connection is closed after the operation.
        connection.close()

async def execute_sql_query_tool(
    query: str,
    * ,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Dict[str, Any]:
    """
    Tool to execute a read-only SQL query in the PostgreSQL database using the specified schema.

    Args:
        query (str): The SQL query to execute.
        config (RunnableConfig): Configuration object injected with database details.

    Returns:
        Dict[str, Any]: A dictionary containing the query result.
    """
    # Extract configuration details from the provided RunnableConfig.
    configuration = Configuration.from_runnable_config(config)
    if not configuration.database_schema:
        raise ValueError("Configuration 'database_schema' is required but was not provided.")
    db_schema = configuration.database_schema
    # Validate and structure the input parameters.
    params = ExecuteSQLQueryParams(query=query)
    # Call the helper function to execute the query.
    return _execute_sql_query(params, db_schema, config=configuration)
