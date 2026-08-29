from typing import Dict, Type
from src.client.base import BaseLLMClient
from src.client.ollama import OllamaClient

_CLIENT_REGISTRY: Dict[str, Type[BaseLLMClient]] = {
    "ollama": OllamaClient,
}

def get_llm_client(
    provider: str,
    api_base: str,
    model: str,
    api_key: str = "",
    timeout: float = 60.0,
) -> BaseLLMClient:
    """Factory function to get the appropriate LLM client based on the provider."""
    provider_key = provider.lower()
    if provider_key not in _CLIENT_REGISTRY:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. Supported providers are: {list(_CLIENT_REGISTRY.keys())}"
        )
    client_cls = _CLIENT_REGISTRY[provider_key]
    return client_cls(api_base=api_base, model=model, api_key=api_key, timeout=timeout)
