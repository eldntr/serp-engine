import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from src.nodes.graph import build_agent_graph, AgentState

async def test_full_agent_graph_pipeline():
    """Pengujian end-to-end seluruh StateGraph LangGraph menggunakan mock eksternal."""
    print("--- Testing Full Agent Graph Pipeline (mocked) ---")

    # --- Persiapan mock ---
    # Mock 1: LLM generate untuk QueryGeneratorNode
    llm_response_mock = MagicMock(spec=httpx.Response)
    llm_response_mock.status_code = 200
    llm_response_mock.json.return_value = {
        "response": '["manfaat kunyit untuk kesehatan", "kandungan senyawa aktif kunyit"]'
    }

    # Mock 2: SearXNG discovery untuk SearchDiscoveryNode
    searxng_response_mock = MagicMock(spec=httpx.Response)
    searxng_response_mock.status_code = 200
    searxng_response_mock.json.return_value = {
        "results": [
            {"url": "https://example.com/kunyit-manfaat"},
            {"url": "https://example.com/kunyit-senyawa"},
        ]
    }

    # Mock 3: Static crawler fetch untuk CrawlerNode (dua halaman)
    crawl_html = "<html><body><h1>Kunyit</h1><p>" + ("Kunyit mengandung senyawa kurkumin yang bermanfaat. " * 50) + "</p></body></html>"
    crawl_response_mock = MagicMock()
    crawl_response_mock.status_code = 200
    crawl_response_mock.html = crawl_html
    crawl_response_mock.content_length = len(crawl_html)
    crawl_response_mock.error = None

    # Mock 4: LLM generate untuk SynthesizerNode (jawaban akhir)
    synthesizer_llm_mock = MagicMock(spec=httpx.Response)
    synthesizer_llm_mock.status_code = 200
    synthesizer_llm_mock.json.return_value = {
        "response": "Kunyit mengandung kurkumin [1] yang memiliki sifat anti-inflamasi dan antioksidan."
    }

    with (
        patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get,
        patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post,
        patch("src.nodes.crawler.StaticCrawler.fetch", new_callable=AsyncMock) as mock_crawl,
        patch("src.nodes.crawler.DynamicCrawler.start", new_callable=AsyncMock),
        patch("src.nodes.crawler.DynamicCrawler.close", new_callable=AsyncMock),
        patch("src.nodes.indexer.DenseEncoder") as mock_dense_cls,
        patch("src.nodes.indexer.BM25SparseEncoder") as mock_sparse_cls,
        patch("src.nodes.indexer.QdrantHybridIndexer") as mock_qdrant_cls,
    ):
        # Routing mock: GET -> SearXNG, POST -> LLM (dua panggilan berurutan)
        mock_get.return_value = searxng_response_mock
        mock_post.side_effect = [llm_response_mock, synthesizer_llm_mock]
        mock_crawl.return_value = crawl_response_mock

        # Setup Qdrant in-memory mock
        mock_qdrant = MagicMock()
        mock_qdrant.init_collection = AsyncMock()
        mock_qdrant.upsert_document = AsyncMock()
        mock_qdrant_cls.return_value = mock_qdrant

        # Setup encoder mocks
        mock_dense = MagicMock()
        mock_dense.encode.return_value = [[0.1] * 1024]
        mock_dense_cls.return_value = mock_dense

        mock_sparse = MagicMock()
        mock_sparse.encode.return_value = {"kunyit": 0.8, "kurkumin": 0.9}
        mock_sparse_cls.return_value = mock_sparse

        # Jalankan graph
        graph = build_agent_graph(max_queries=2, num_results_per_query=2)
        initial_state: AgentState = {
            "prompt": "Apa saja manfaat kunyit untuk kesehatan?",
            "queries": [],
            "urls": [],
            "documents": [],
            "indexed_count": 0,
            "answer": "",
        }

        result = await graph.ainvoke(initial_state)

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
        assert isinstance(result["answer"], str) and len(result["answer"]) > 0

        print("-> test_full_agent_graph_pipeline PASSED")


async def main():
    await test_full_agent_graph_pipeline()
    print("\nSemua unit test Agent Graph berhasil dijalankan!")

if __name__ == "__main__":
    asyncio.run(main())
