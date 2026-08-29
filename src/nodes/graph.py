from typing import Any, Dict, List, Optional, TypedDict
from langgraph.graph import StateGraph, END
from src.nodes.query_generator import QueryGeneratorNode
from src.nodes.search import SearchDiscoveryNode
from src.nodes.crawler import CrawlerNode
from src.nodes.indexer import IndexerNode
from src.nodes.synthesizer import SynthesizerNode


class AgentState(TypedDict):
    """Representasi state lengkap yang mengalir di sepanjang agent graph."""
    prompt: str                          # Input awal dari user
    queries: List[str]                   # Hasil QueryGeneratorNode
    urls: List[str]                      # Hasil SearchDiscoveryNode
    documents: List[Dict[str, Any]]      # Hasil CrawlerNode
    indexed_count: int                   # Hasil IndexerNode
    answer: str                          # Hasil akhir SynthesizerNode


def build_agent_graph(
    indexer=None,           # Opsional: suntik instance Qdrant yang sudah ada dari app.state
    max_queries: int = 4,
    num_results_per_query: int = 3,
) -> StateGraph:
    """
    Membangun dan mengompilasi StateGraph LangGraph untuk agen RAG otonom.

    Alur kerja:
      prompt
        -> QueryGeneratorNode  (prompt -> queries)
        -> SearchDiscoveryNode (queries -> urls)
        -> CrawlerNode         (urls -> documents)
        -> IndexerNode         (documents -> indexed_count)
        -> SynthesizerNode     (prompt + documents -> answer)
        -> END
    """
    # Inisialisasi semua node
    query_generator = QueryGeneratorNode(max_queries=max_queries)
    search_discovery = SearchDiscoveryNode(num_results_per_query=num_results_per_query)
    crawler = CrawlerNode()
    indexer_node = IndexerNode(indexer=indexer)
    synthesizer = SynthesizerNode()

    # Buat StateGraph dengan AgentState
    graph = StateGraph(AgentState)

    # Daftarkan node
    graph.add_node("query_generator", query_generator)
    graph.add_node("search_discovery", search_discovery)
    graph.add_node("crawler", crawler)
    graph.add_node("indexer", indexer_node)
    graph.add_node("synthesizer", synthesizer)

    # Definisikan alur (edges)
    graph.set_entry_point("query_generator")
    graph.add_edge("query_generator", "search_discovery")
    graph.add_edge("search_discovery", "crawler")
    graph.add_edge("crawler", "indexer")
    graph.add_edge("indexer", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()
