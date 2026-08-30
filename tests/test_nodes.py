import asyncio
from unittest.mock import AsyncMock, patch
from src.nodes.query_generator import QueryGeneratorNode, SearchQueries

async def test_query_generator_node_success():
    print("--- Testing QueryGeneratorNode Success Case ---")
    mock_queries = SearchQueries(queries=["resep rendang padang asli", "cara memasak rendang daging sapi"])
    
    mock_structured_llm = AsyncMock()
    mock_structured_llm.return_value = mock_queries
    
    with patch("langchain_ollama.ChatOllama.with_structured_output") as mock_with_structured:
        mock_with_structured.return_value = mock_structured_llm
        
        node = QueryGeneratorNode()
        state = {"prompt": "Bagaimana cara membuat rendang padang yang lezat?"}
        result = await node(state)
        
        print(f"Hasil: {result}")
        assert result == {"queries": ["resep rendang padang asli", "cara memasak rendang daging sapi"]}
        print("-> test_query_generator_node_success PASSED")

async def test_query_generator_node_failure():
    print("--- Testing QueryGeneratorNode Failure/Fallback Case ---")
    mock_structured_llm = AsyncMock()
    mock_structured_llm.side_effect = Exception("LLM API Error")
    
    with patch("langchain_ollama.ChatOllama.with_structured_output") as mock_with_structured:
        mock_with_structured.return_value = mock_structured_llm
        
        node = QueryGeneratorNode()
        state = {"prompt": "Bagaimana cara membuat rendang padang yang lezat?"}
        result = await node(state)
        
        print(f"Hasil: {result}")
        # Fallback should return the original prompt
        assert result == {"queries": ["Bagaimana cara membuat rendang padang yang lezat?"]}
        print("-> test_query_generator_node_failure PASSED")

async def main():
    await test_query_generator_node_success()
    print()
    await test_query_generator_node_failure()
    print("\nSemua unit test QueryGeneratorNode berhasil dijalankan!")

if __name__ == "__main__":
    asyncio.run(main())
