import asyncio
from unittest.mock import AsyncMock, patch
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from src.client.factory import get_llm_client

async def test_ollama_client_creation():
    print("--- Testing Ollama Client Creation ---")
    client = get_llm_client(
        provider="ollama",
        api_base="http://localhost:11434",
        model="qwen2.5:32b-instruct-q4_K_M"
    )
    assert isinstance(client, ChatOllama)
    assert client.base_url == "http://localhost:11434"
    assert client.model == "qwen2.5:32b-instruct-q4_K_M"
    print("-> test_ollama_client_creation PASSED")

async def test_openai_client_creation():
    print("--- Testing OpenAI Client Creation ---")
    client = get_llm_client(
        provider="openai",
        api_base="https://api.openai.com/v1",
        model="gpt-4o",
        api_key="test-key"
    )
    assert isinstance(client, ChatOpenAI)
    assert client.openai_api_base == "https://api.openai.com/v1"
    assert client.model_name == "gpt-4o"
    assert client.openai_api_key.get_secret_value() == "test-key"
    print("-> test_openai_client_creation PASSED")

async def test_client_invocation():
    print("--- Testing Client Invocation ---")
    client = get_llm_client(
        provider="ollama",
        api_base="http://localhost:11434",
        model="qwen2.5"
    )
    
    mock_response = AIMessage(content="Ini adalah jawaban mock.")
    with patch.object(ChatOllama, "ainvoke", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = mock_response
        
        response = await client.ainvoke("Halo, siapa kamu?")
        assert response.content == "Ini adalah jawaban mock."
        mock_invoke.assert_called_once_with("Halo, siapa kamu?")
        print("-> test_client_invocation PASSED")

async def main():
    await test_ollama_client_creation()
    await test_openai_client_creation()
    await test_client_invocation()
    print("\nSemua unit test LLM Client berhasil dijalankan!")

if __name__ == "__main__":
    asyncio.run(main())
