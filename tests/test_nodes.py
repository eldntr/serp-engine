import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from src.nodes.query_generator import QueryGeneratorNode

async def test_node_clean_json():
    print("--- Testing QueryGeneratorNode with Clean JSON response ---")
    mock_response = '["resep rendang padang asli", "cara memasak rendang daging sapi"]'
    
    mock_resp_obj = MagicMock(spec=httpx.Response)
    mock_resp_obj.status_code = 200
    mock_resp_obj.json.return_value = {"response": mock_response}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp_obj
        
        node = QueryGeneratorNode()
        state = {"prompt": "Bagaimana cara membuat rendang padang yang lezat?"}
        result = await node(state)
        
        print(f"Hasil output node state update: {result}")
        assert result == {"queries": ["resep rendang padang asli", "cara memasak rendang daging sapi"]}
        print("-> test_node_clean_json PASSED")

async def test_node_markdown_json():
    print("--- Testing QueryGeneratorNode with Markdown-wrapped JSON response ---")
    mock_response = """
    Berikut adalah query pencariannya:
    ```json
    [
      "perbedaan claude dan chatgpt",
      "keamanan data claude vs chatgpt"
    ]
    ```
    Semoga membantu!
    """
    
    mock_resp_obj = MagicMock(spec=httpx.Response)
    mock_resp_obj.status_code = 200
    mock_resp_obj.json.return_value = {"response": mock_response}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp_obj
        
        node = QueryGeneratorNode()
        state = {"prompt": "Apakah ChatGPT lebih aman dibanding Claude?"}
        result = await node(state)
        
        print(f"Hasil output node state update: {result}")
        assert result == {"queries": ["perbedaan claude dan chatgpt", "keamanan data claude vs chatgpt"]}
        print("-> test_node_markdown_json PASSED")

async def test_node_fallback_list():
    print("--- Testing QueryGeneratorNode with list format response ---")
    mock_response = """
    1. "spesifikasi iphone 15 pro max"
    2. harga pasaran iphone 15 pro max indonesia
    - "fitur baru kamera iphone 15 pro"
    """
    
    mock_resp_obj = MagicMock(spec=httpx.Response)
    mock_resp_obj.status_code = 200
    mock_resp_obj.json.return_value = {"response": mock_response}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp_obj
        
        node = QueryGeneratorNode(max_queries=3)
        state = {"prompt": "Berapa harga dan spesifikasi iphone 15 pro max?"}
        result = await node(state)
        
        print(f"Hasil output node state update: {result}")
        assert result == {"queries": [
            "spesifikasi iphone 15 pro max",
            "harga pasaran iphone 15 pro max indonesia",
            "fitur baru kamera iphone 15 pro"
        ]}
        print("-> test_node_fallback_list PASSED")

async def main():
    await test_node_clean_json()
    print()
    await test_node_markdown_json()
    print()
    await test_node_fallback_list()
    print("\nSemua unit test QueryGeneratorNode berhasil dijalankan!")

if __name__ == "__main__":
    asyncio.run(main())
