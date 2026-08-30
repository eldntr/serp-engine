import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from qdrant_client.models import ScoredPoint
from langchain_core.messages import AIMessage
from src.nodes.query_generator import SearchQueries
from src.nodes.graph import build_agent_graph, AgentState

async def test_full_agent_graph_pipeline():
    """Pengujian end-to-end seluruh StateGraph LangGraph menggunakan mock eksternal (Linear Mode)."""
    print("--- Testing Full Agent Graph Pipeline (mocked - Linear Mode) ---")

    # --- Persiapan mock ---
    mock_queries = SearchQueries(queries=["manfaat kunyit untuk kesehatan", "kandungan senyawa aktif kunyit"])
    mock_structured_llm = AsyncMock()
    mock_structured_llm.return_value = mock_queries

    mock_answer = AIMessage(content="Kunyit mengandung kurkumin [1] yang memiliki sifat anti-inflamasi dan antioksidan.")

    searxng_response_mock = MagicMock(spec=httpx.Response)
    searxng_response_mock.status_code = 200
    searxng_response_mock.json.return_value = {
        "results": [
            {"url": "https://example.com/kunyit-manfaat"},
            {"url": "https://example.com/kunyit-senyawa"},
        ]
    }

    crawl_html = "<html><body><h1>Kunyit</h1><p>" + ("Kunyit mengandung senyawa kurkumin yang bermanfaat. " * 50) + "</p></body></html>"
    crawl_response_mock = MagicMock()
    crawl_response_mock.status_code = 200
    crawl_response_mock.html = crawl_html
    crawl_response_mock.content_length = len(crawl_html)
    crawl_response_mock.error = None

    with (
        patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get,
        patch("langchain_ollama.ChatOllama.with_structured_output") as mock_with_structured,
        patch("langchain_ollama.ChatOllama.ainvoke", new_callable=AsyncMock) as mock_ainvoke,
        patch("src.nodes.crawler.StaticCrawler.fetch", new_callable=AsyncMock) as mock_crawl,
        patch("src.nodes.crawler.DynamicCrawler.start", new_callable=AsyncMock),
        patch("src.nodes.crawler.DynamicCrawler.close", new_callable=AsyncMock),
        patch("src.search.encoders.dense.DenseEncoder") as mock_dense_cls,
        patch("src.search.encoders.sparse.BM25SparseEncoder") as mock_sparse_cls,
        patch("src.search.indexing.qdrant_client.QdrantHybridIndexer") as mock_qdrant_cls,
        patch("src.search.rankers.cross_encoder.NeuralReranker") as mock_reranker_cls,
    ):
        # Routing mock
        mock_get.return_value = searxng_response_mock
        mock_with_structured.return_value = mock_structured_llm
        mock_ainvoke.return_value = mock_answer
        mock_crawl.return_value = crawl_response_mock

        # Setup Qdrant in-memory mock
        mock_qdrant = MagicMock()
        mock_qdrant.init_collection = AsyncMock()
        mock_qdrant.upsert_document = AsyncMock()
        
        mock_point = ScoredPoint(
            id="123",
            version=1,
            score=0.9,
            payload={
                "id": "https://example.com/kunyit-manfaat",
                "url": "https://example.com/kunyit-manfaat",
                "title": "Kunyit Manfaat",
                "text": "Kunyit mengandung senyawa kurkumin yang bermanfaat.",
            }
        )
        mock_qdrant.search_dense = AsyncMock(return_value=[mock_point])
        mock_qdrant.search_sparse = AsyncMock(return_value=[mock_point])
        mock_qdrant_cls.return_value = mock_qdrant

        # Setup encoder mocks
        mock_dense = MagicMock()
        mock_dense.encode.return_value = [[0.1] * 1024]
        mock_dense_cls.return_value = mock_dense

        mock_sparse = MagicMock()
        mock_sparse.encode.return_value = {"kunyit": 0.8, "kurkumin": 0.9}
        mock_sparse_cls.return_value = mock_sparse

        # Setup NeuralReranker mock
        mock_reranker = MagicMock()
        mock_reranker.rerank = MagicMock(return_value=[
            {
                "id": "https://example.com/kunyit-manfaat",
                "url": "https://example.com/kunyit-manfaat",
                "title": "Kunyit Manfaat",
                "text": "Kunyit mengandung senyawa kurkumin yang bermanfaat.",
            }
        ])
        mock_reranker_cls.return_value = mock_reranker

        # Jalankan graph sekuensial linear
        graph = build_agent_graph(indexer=mock_qdrant, max_queries=2, num_results_per_query=2, mode="linear")
        initial_state: AgentState = {
            "prompt": "Apa saja manfaat kunyit untuk kesehatan?",
            "queries": [],
            "urls": [],
            "documents": [],
            "indexed_count": 0,
            "answer": "",
            "messages": [],
        }

        config = {"configurable": {"thread_id": "linear-thread"}}
        result = await graph.ainvoke(initial_state, config=config)

        # Assertions
        print(f"Queries: {result['queries']}")
        print(f"URLs: {result['urls']}")
        print(f"Dokumen ter-crawl: {len(result['documents'])}")
        print(f"Indexed count: {result['indexed_count']}")
        print(f"Answer: {result['answer']}")

        assert isinstance(result["queries"], list)
        assert len(result["queries"]) > 0
        assert isinstance(result["urls"], list)
        assert isinstance(result["documents"], list)
        assert len(result["documents"]) > 0
        assert isinstance(result["answer"], str) and len(result["answer"]) > 0

        print("-> test_full_agent_graph_pipeline PASSED")


async def test_react_agent_graph_pipeline():
    """Pengujian end-to-end graf agen ReAct dinamis menggunakan mock."""
    print("\n--- Testing ReAct Agent Graph Pipeline (mocked - ReAct Mode) ---")

    # --- Persiapan mock messages untuk tool calling loop ---
    msg1 = AIMessage(
        content="",
        tool_calls=[{
            "name": "search_web",
            "args": {"query": "manfaat kunyit"},
            "id": "call_1"
        }]
    )

    msg2 = AIMessage(
        content="",
        tool_calls=[{
            "name": "crawl_and_index_pages",
            "args": {"urls": ["https://example.com/kunyit-manfaat"]},
            "id": "call_2"
        }]
    )

    msg3 = AIMessage(
        content="",
        tool_calls=[{
            "name": "retrieve_relevant_chunks",
            "args": {"query": "manfaat kunyit"},
            "id": "call_3"
        }]
    )

    msg4 = AIMessage(
        content="Kunyit mengandung kurkumin yang bermanfaat untuk kesehatan."
    )

    mock_bound_llm = AsyncMock()
    mock_bound_llm.ainvoke.side_effect = [msg1, msg2, msg3, msg4]

    searxng_response_mock = MagicMock(spec=httpx.Response)
    searxng_response_mock.status_code = 200
    searxng_response_mock.json.return_value = {
        "results": [
            {"url": "https://example.com/kunyit-manfaat"},
        ]
    }

    # Content length must be > 1500 to prevent static-to-dynamic browser fallback in crawl node mock environment
    crawl_html = "<html><body><h1>Kunyit</h1><p>" + ("Kunyit mengandung senyawa kurkumin yang bermanfaat untuk kesehatan. " * 50) + "</p></body></html>"
    crawl_response_mock = MagicMock()
    crawl_response_mock.status_code = 200
    crawl_response_mock.html = crawl_html
    crawl_response_mock.content_length = len(crawl_html)
    crawl_response_mock.error = None

    with (
        patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get,
        patch("langchain_ollama.ChatOllama.bind_tools") as mock_bind_tools,
        patch("src.nodes.crawler.StaticCrawler.fetch", new_callable=AsyncMock) as mock_crawl,
        patch("src.nodes.crawler.DynamicCrawler.start", new_callable=AsyncMock),
        patch("src.nodes.crawler.DynamicCrawler.close", new_callable=AsyncMock),
        patch("src.search.encoders.dense.DenseEncoder") as mock_dense_cls,
        patch("src.search.encoders.sparse.BM25SparseEncoder") as mock_sparse_cls,
        patch("src.search.indexing.qdrant_client.QdrantHybridIndexer") as mock_qdrant_cls,
        patch("src.search.rankers.cross_encoder.NeuralReranker") as mock_reranker_cls,
    ):
        # Setup mocks
        mock_get.return_value = searxng_response_mock
        mock_bind_tools.return_value = mock_bound_llm
        mock_crawl.return_value = crawl_response_mock

        mock_qdrant = MagicMock()
        mock_qdrant.init_collection = AsyncMock()
        mock_qdrant.upsert_document = AsyncMock()
        
        mock_point = ScoredPoint(
            id="123",
            version=1,
            score=0.9,
            payload={
                "id": "https://example.com/kunyit-manfaat",
                "url": "https://example.com/kunyit-manfaat",
                "title": "Kunyit Manfaat",
                "text": "Kunyit mengandung senyawa kurkumin yang bermanfaat.",
            }
        )
        mock_qdrant.search_dense = AsyncMock(return_value=[mock_point])
        mock_qdrant.search_sparse = AsyncMock(return_value=[mock_point])
        mock_qdrant_cls.return_value = mock_qdrant

        mock_dense = MagicMock()
        mock_dense.encode.return_value = [[0.1] * 1024]
        mock_dense_cls.return_value = mock_dense

        mock_sparse = MagicMock()
        mock_sparse.encode.return_value = {"kunyit": 0.8}
        mock_sparse_cls.return_value = mock_sparse

        mock_reranker = MagicMock()
        mock_reranker.rerank = MagicMock(return_value=[
            {
                "id": "https://example.com/kunyit-manfaat",
                "url": "https://example.com/kunyit-manfaat",
                "title": "Kunyit Manfaat",
                "text": "Kunyit mengandung senyawa kurkumin yang bermanfaat.",
            }
        ])
        mock_reranker_cls.return_value = mock_reranker

        # Jalankan graf ReAct
        graph = build_agent_graph(indexer=mock_qdrant, max_queries=2, num_results_per_query=2, mode="react")
        initial_state: AgentState = {
            "prompt": "Apa manfaat kunyit?",
            "queries": [],
            "urls": [],
            "documents": [],
            "indexed_count": 0,
            "answer": "",
            "messages": [],
        }

        config = {"configurable": {"thread_id": "react-thread"}}
        result = await graph.ainvoke(initial_state, config=config)

        # Assertions
        print(f"Queries: {result['queries']}")
        print(f"URLs: {result['urls']}")
        print(f"Indexed count: {result['indexed_count']}")
        print(f"Answer: {result['answer']}")

        assert isinstance(result["queries"], list)
        assert "manfaat kunyit" in result["queries"]
        assert "https://example.com/kunyit-manfaat" in result["urls"]
        assert result["indexed_count"] > 0
        assert result["answer"] == "Kunyit mengandung kurkumin yang bermanfaat untuk kesehatan."

        print("-> test_react_agent_graph_pipeline PASSED")


async def main():
    await test_full_agent_graph_pipeline()
    await test_react_agent_graph_pipeline()
    print("\nSemua unit test Agent Graph (Linear & ReAct) berhasil dijalankan!")

if __name__ == "__main__":
    asyncio.run(main())
