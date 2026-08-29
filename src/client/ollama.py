import json
from typing import AsyncIterator, Optional
import httpx
from src.client.base import BaseLLMClient
from src.core.logger import logger

class OllamaClient(BaseLLMClient):
    """Ollama Client using HTTPX to communicate with Ollama API."""

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        url = f"{self.api_base.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt
        
        # Map any extra parameters (like temperature, top_p, etc.) to 'options'
        if kwargs:
            payload["options"] = kwargs

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
            except Exception as e:
                logger.error(f"Ollama generate error: {e}")
                raise