#graph.py 

from datetime import datetime, timezone
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model

# Main node that interacts with the model using the system prompt and the last 6 messages, if available
async def call_model(state: State, config: RunnableConfig) -> Dict[str, List[AIMessage]]:
    # Load configuration from the provided RunnableConfig
    configuration = Configuration.from_runnable_config(config)
    # Load the chat model and bind it to the available tools
    model = load_chat_model(configuration.model).bind_tools(TOOLS)
    # Create a system message with the current system time and user name
    system_message = SystemMessage(
        content=configuration.system_prompt.format(
            system_time=datetime.now(tz=timezone.utc).isoformat(),
            user_name=configuration.user_name or "Unknown User"
        )
    )
    
    # Build the context for the LLM: use only the last 6 messages if available
    if len(state.messages) >= 6:
        messages_to_send = [system_message] + list(state.messages[-6:])
    else:
        messages_to_send = [system_message] + list(state.messages)
    
    # Invoke the model with the prepared messages and configuration
    response = cast(
        AIMessage,
        await model.ainvoke(messages_to_send, config)
    )
    
    # Update the message history: combine the current history with the new response
    new_history = state.messages + [response]
    delete_messages = []
    # If there are more than 6 messages, prepare to remove all except the last 6
    if len(new_history) > 6:
        messages_to_remove = new_history[:-6]  # Keep only the last 6 messages
        delete_messages = [RemoveMessage(id=m.id) for m in messages_to_remove]
    
    # Return the new response along with any messages to be removed
    return {"messages": [response] + delete_messages}

# Routing function: if there is a tool call, route to the "tools" node
def route_model_output(state: State) -> Literal["__end__", "tools"]:
    # Get the last message in the state
    last_message = state.messages[-1]
    # Ensure the last message is an AIMessage
    if not isinstance(last_message, AIMessage):
        raise ValueError(f"Expected AIMessage, but received {type(last_message).__name__}")
    # If the last message contains tool calls, route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, end the process
    return "__end__"

# Graph assembly
builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Add the main model-calling node
builder.add_node(call_model)
# Add the "tools" node, which handles tool-related operations
builder.add_node("tools", ToolNode(TOOLS))

# Define the edges of the graph
builder.add_edge("__start__", "call_model")  # Start by calling the model
builder.add_conditional_edges("call_model", route_model_output)  # Route based on model output
builder.add_edge("tools", "call_model")  # After tools, return to calling the model

# Compile the graph with a memory-based checkpointer
graph = builder.compile(
    checkpointer=MemorySaver(),  # Save memory state for checkpoints
    interrupt_before=[],  # No interruptions before nodes
    interrupt_after=[]    # No interruptions after nodes
)
# Assign a name to the graph
graph.name = "ReAct Agent with Memory"
