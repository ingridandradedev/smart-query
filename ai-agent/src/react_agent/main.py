# main.py -> langgraph-app/src/react_agent/main.py

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
import os
from uuid import uuid4
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Retrieve API keys from environment variables
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Importing necessary modules for the agent's functionality
from react_agent.graph import graph
from react_agent.state import State, InputState
from react_agent.configuration import Configuration
from langchain_core.messages import AIMessage, HumanMessage

# Initialize the FastAPI application with metadata
app = FastAPI(
    title="Agent Mark",
    description="Your assistant for marketing data analysis",
    version="1.0"
)

# Define allowed origins for CORS (Cross-Origin Resource Sharing)
origins = [
    "https://smartquery.offgridmartech.com.br",
    # Add other origins as needed
]

# Add CORS middleware to the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODELS

# Define the structure of a message exchanged with the agent
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: Union[str, List[Dict[str, Any]]]

# Define the structure of a request sent to the agent
class AgentRequest(BaseModel):
    messages: List[Message]
    thread_id: Optional[str] = None
    # New fields for user identification
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    # New fields for configuration variables
    database_schema: Optional[str] = None
    index_host: Optional[str] = None
    namespace: Optional[str] = None
    # Fields for PostgreSQL connection parameters
    postgres_host: Optional[str] = None
    postgres_port: Optional[int] = None
    postgres_dbname: Optional[str] = None
    postgres_user: Optional[str] = None
    postgres_password: Optional[str] = None

# Define the structure of a response returned by the agent
class AgentResponse(BaseModel):
    thread_id: str
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    messages: List[Message]

# Define the structure of a single response message
class AgentSingleResponse(BaseModel):
    thread_id: str
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    role: str
    content: Union[str, List[Dict[str, Any]]]

# Helper function to generate a thread ID if not provided
def get_thread_id(thread_id: Optional[str]) -> str:
    return thread_id if thread_id else str(uuid4())

# Helper function to extract the content of a message
def extract_message_content(msg: Union[AIMessage, HumanMessage]) -> Union[str, List[Dict[str, Any]]]:
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        if all(isinstance(item, dict) for item in content):
            return content
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
        return " ".join(text_parts) if text_parts else ""
    if isinstance(content, dict):
        return content.get("text", "")
    return str(content)

# Endpoint to invoke the agent and return a full response
@app.post("/invoke", response_model=AgentResponse)
async def invoke_agent(request: AgentRequest) -> AgentResponse:
    thread_id = get_thread_id(request.thread_id)
    # Configuration dictionary for the agent
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": request.user_id,
            "user_name": request.user_name,
            "database_schema": request.database_schema,
            "index_host": request.index_host,
            "namespace": request.namespace,
            # PostgreSQL connection parameters
            "postgres_host": request.postgres_host,
            "postgres_port": request.postgres_port,
            "postgres_dbname": request.postgres_dbname,
            "postgres_user": request.postgres_user,
            "postgres_password": request.postgres_password,
        }
    }
    # Convert request messages into input state
    input_state = InputState(
        messages=[
            HumanMessage(content=msg.content) if msg.role.lower() == "user" 
            else AIMessage(content=msg.content)
            for msg in request.messages
        ]
    )
    state = State(messages=input_state.messages)
    try:
        # Invoke the agent's graph with the input state and configuration
        output_state = await graph.ainvoke(state, config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # Process the output messages from the agent
    response_messages = []
    for msg in output_state.get("messages", []):
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        content = extract_message_content(msg)
        response_messages.append(Message(role=role, content=content))
    return AgentResponse(
        thread_id=thread_id,
        user_id=request.user_id,
        user_name=request.user_name,
        messages=response_messages
    )

# Endpoint to invoke the agent and return only the last message
@app.post("/invoke_last", response_model=AgentSingleResponse)
async def invoke_agent_last(request: AgentRequest) -> AgentSingleResponse:
    thread_id = get_thread_id(request.thread_id)
    # Configuration dictionary for the agent
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": request.user_id,
            "user_name": request.user_name,
            "database_schema": request.database_schema,
            "index_host": request.index_host,
            "namespace": request.namespace,
            # PostgreSQL connection parameters
            "postgres_host": request.postgres_host,
            "postgres_port": request.postgres_port,
            "postgres_dbname": request.postgres_dbname,
            "postgres_user": request.postgres_user,
            "postgres_password": request.postgres_password,
        }
    }
    # Convert request messages into input state
    input_state = InputState(
        messages=[
            HumanMessage(content=msg.content) if msg.role.lower() == "user" 
            else AIMessage(content=msg.content)
            for msg in request.messages
        ]
    )
    state = State(messages=input_state.messages)
    try:
        # Invoke the agent's graph with the input state and configuration
        output_state = await graph.ainvoke(state, config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # Retrieve the last message from the agent's output
    messages = output_state.get("messages", [])
    if not messages:
        raise HTTPException(status_code=500, detail="No messages returned from the agent.")
    last_msg = messages[-1]
    role = "assistant" if isinstance(last_msg, AIMessage) else "user"
    content = extract_message_content(last_msg)
    return AgentSingleResponse(
        thread_id=thread_id,
        user_id=request.user_id,
        user_name=request.user_name,
        role=role,
        content=content
    )

# Endpoint to stream responses from the agent
@app.post("/stream", response_class=StreamingResponse)
async def stream_agent(request: AgentRequest) -> StreamingResponse:
    thread_id = get_thread_id(request.thread_id)
    # Configuration dictionary for the agent
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": request.user_id,
            "user_name": request.user_name,
            "database_schema": request.database_schema,
            "index_host": request.index_host,
            "namespace": request.namespace,
            # PostgreSQL connection parameters
            "postgres_host": request.postgres_host,
            "postgres_port": request.postgres_port,
            "postgres_dbname": request.postgres_dbname,
            "postgres_user": request.postgres_user,
            "postgres_password": request.postgres_password,
        }
    }
    # Convert request messages into input state
    input_state = InputState(
        messages=[
            HumanMessage(content=msg.content) if msg.role.lower() == "user" 
            else AIMessage(content=msg.content)
            for msg in request.messages
        ]
    )
    state = State(messages=input_state.messages)
    # Generator function to stream chunks of data
    async def generate_chunks():
        try:
            # Stream events from the agent's graph
            async for event in graph.astream_events(state, config=config, version="v2"):
                if event and "messages" in event:
                    last_message = event["messages"][-1]
                    content = extract_message_content(last_message)
                    if isinstance(content, list):
                        yield f"data: {str(content)}\n\n"
                    else:
                        yield f"data: {content}\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(generate_chunks(), media_type="text/event-stream")

# Entry point for running the application
if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI application using Uvicorn
    uvicorn.run("react_agent.main:app", host="0.0.0.0", port=8000, reload=True)