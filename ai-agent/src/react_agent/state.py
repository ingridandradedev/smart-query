# state.py -> langgraph-app/src/react_agent/state.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated

# Represents the input state of the application, containing the history of messages
@dataclass
class InputState:
    # A sequence of messages received as input. The `Annotated` type is used to associate
    # the `add_messages` function with the `messages` field, which may be used for validation
    # or additional processing.
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)

# Represents the overall state of the application, extending the InputState
@dataclass
class State(InputState):
    # Indicates whether this is the last step of execution. The `IsLastStep` type is used
    # to enforce proper typing and potentially manage state transitions.
    is_last_step: IsLastStep = field(default=False)
