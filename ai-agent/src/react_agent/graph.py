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

# Nó principal que chama o modelo usando o system prompt e as últimas 6 mensagens, se disponíveis
async def call_model(state: State, config: RunnableConfig) -> Dict[str, List[AIMessage]]:
    configuration = Configuration.from_runnable_config(config)
    # Carrega o modelo e vincula as ferramentas
    model = load_chat_model(configuration.model).bind_tools(TOOLS)
    system_message = SystemMessage(
        content=configuration.system_prompt.format(
            system_time=datetime.now(tz=timezone.utc).isoformat(),
            user_name=configuration.user_name or "Usuário Desconhecido"
        )
    )
    
    # Constrói o contexto para o LLM: usa somente as últimas 6 mensagens, se disponíveis
    if len(state.messages) >= 6:
        messages_to_send = [system_message] + list(state.messages[-6:])
    else:
        messages_to_send = [system_message] + list(state.messages)
    
    response = cast(
        AIMessage,
        await model.ainvoke(messages_to_send, config)
    )
    
    # Atualiza o histórico: junta o histórico atual com a nova resposta
    new_history = state.messages + [response]
    delete_messages = []
    # Se houver mais de 6 mensagens, prepara remoção de todas exceto as últimas 6
    if len(new_history) > 6:
        messages_to_remove = new_history[:-6]  # mantém apenas as últimas 6
        delete_messages = [RemoveMessage(id=m.id) for m in messages_to_remove]
    
    return {"messages": [response] + delete_messages}

# Função de roteamento: se houver chamada de ferramenta, direciona para o nó "tools"
def route_model_output(state: State) -> Literal["__end__", "tools"]:
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(f"Esperado AIMessage, mas recebeu {type(last_message).__name__}")
    if last_message.tool_calls:
        return "tools"
    return "__end__"

# Montagem do grafo
builder = StateGraph(State, input=InputState, config_schema=Configuration)

builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))

builder.add_edge("__start__", "call_model")
builder.add_conditional_edges("call_model", route_model_output)
builder.add_edge("tools", "call_model")

graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=[],
    interrupt_after=[]
)
graph.name = "ReAct Agent with Memory"
