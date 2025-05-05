# utils.py - langgraph-app/src/react_agent/utils.py

"""Utility & helper functions."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage


def get_message_text(msg: BaseMessage) -> str:
    """Extract the text content from a message object.

    Args:
        msg (BaseMessage): A message object that contains content, which can be 
        a string, a dictionary, or a list of strings/dictionaries.

    Returns:
        str: The extracted text content from the message. If the content is a string, 
        it is returned directly. If it is a dictionary, the value of the "text" key 
        is returned. If it is a list, all text elements are concatenated into a single 
        string.
    """
    content = msg.content
    if isinstance(content, str):
        # If the content is a string, return it directly.
        return content
    elif isinstance(content, dict):
        # If the content is a dictionary, return the value of the "text" key, or an empty string if not found.
        return content.get("text", "")
    else:
        # If the content is a list, iterate through it and extract text elements.
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        # Concatenate all text elements into a single string and remove leading/trailing whitespace.
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model based on a fully specified name in the format 'provider/model'.

    Args:
        fully_specified_name (str): A string specifying the provider and model name, 
        separated by a forward slash (e.g., 'openai/gpt-3.5').

    Returns:
        BaseChatModel: An instance of the chat model initialized with the specified provider and model name.
    """
    # Split the input string into provider and model parts using the forward slash as a delimiter.
    provider, model = fully_specified_name.split("/", maxsplit=1)
    # Initialize and return the chat model using the specified provider and model name.
    return init_chat_model(model, model_provider=provider)
