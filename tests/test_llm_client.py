import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from src.client.factory import get_llm_client
from src.client.ollama import OllamaClient

async def test_ollama_generate():
    print("--- Testing Ollama generate() ---")
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Ini adalah jawaban mock."}
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        
        client = get_llm_client(
            provider="ollama",
            api_base="http://localhost:11434",
            model="qwen2.5:32b-instruct-q4_K_M"
        )
        
        response = await client.generate(
            prompt="Halo, siapa kamu?",
            system_prompt="Kamu adalah asisten AI.",
            temperature=0.7
        )
        
        print(f"Hasil: {response}")
        assert response == "Ini adalah jawaban mock."
        
        # Verify post payload
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        # Check json payload
        json_payload = kwargs.get("json", {})
        assert json_payload["model"] == "qwen2.5:32b-instruct-q4_K_M"
        assert json_payload["prompt"] == "Halo, siapa kamu?"
        assert json_payload["system"] == "Kamu adalah asisten AI."
        assert json_payload["stream"] is False
        assert json_payload["options"] == {"temperature": 0.7}
        print("-> test_ollama_generate PASSED")

async def main():
    await test_ollama_generate()
    print("\nSemua unit test LLM Client berhasil dijalankan!")

if __name__ == "__main__":
    asyncio.run(main())
