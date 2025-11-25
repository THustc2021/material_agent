from typing import Any, Dict

from langchain_ollama import ChatOllama


def create_chat_model(config: Dict[str, Any], *, temperature: float = 0.0) -> ChatOllama:
    """Create a ChatOllama instance using repository configuration."""
    model = config.get("OLLAMA_MODEL")
    if not model:
        raise ValueError("OLLAMA_MODEL must be defined in the configuration.")

    kwargs: Dict[str, Any] = {"model": model, "temperature": temperature}
    base_url = config.get("OLLAMA_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOllama(**kwargs)
