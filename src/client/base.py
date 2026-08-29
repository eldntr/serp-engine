from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

class BaseLLMClient(ABC):
    """Abstract Base Class for LLM Clients."""
    
    def __init__(self, api_base: str, model: str, api_key: Optional[str] = None, timeout: float = 60.0):
        self.api_base = api_base
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Sends a prompt to the LLM and returns the complete generated string response."""
        pass
