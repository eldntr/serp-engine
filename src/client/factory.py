from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

def get_llm_client(
    provider: str,
    api_base: str,
    model: str,
    api_key: str = "",
    timeout: float = 60.0,
) -> BaseChatModel:
    """Factory function to get the appropriate LangChain Chat Model based on the provider."""
    provider_key = provider.lower()
    if provider_key == "ollama":
        return ChatOllama(
            base_url=api_base,
            model=model,
            timeout=timeout,
        )
    elif provider_key == "openai":
        return ChatOpenAI(
            openai_api_base=api_base,
            model_name=model,
            openai_api_key=api_key,
            timeout=timeout,
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider}'. Supported providers are: ['ollama', 'openai']"
        )
