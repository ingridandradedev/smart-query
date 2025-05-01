#state.py -> langgraph-app/src/react_agent/state.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated

@dataclass
class InputState:
    # Histórico de mensagens recebido na entrada
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)

@dataclass
class State(InputState):
    # Indica se é o último passo da execução
    is_last_step: IsLastStep = field(default=False)
