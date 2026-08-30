from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.nodes.query_generator import QueryGeneratorNode
from src.nodes.search import SearchDiscoveryNode
from src.nodes.crawler import CrawlerNode
from src.nodes.indexer import IndexerNode
from src.nodes.retriever import RetrieverNode
from src.nodes.synthesizer import SynthesizerNode


def build_linear_graph(
    AgentState,
    indexer=None,
    max_queries: int = 4,
    num_results_per_query: int = 3,
) -> StateGraph:
    """Builds and compiles the sequential linear pipeline graph."""
    query_generator = QueryGeneratorNode(max_queries=max_queries)
    search_discovery = SearchDiscoveryNode(num_results_per_query=num_results_per_query)
    crawler = CrawlerNode()
    indexer_node = IndexerNode(indexer=indexer)
    retriever = RetrieverNode(indexer=indexer)
    synthesizer = SynthesizerNode()

    graph = StateGraph(AgentState)

    graph.add_node("query_generator", query_generator)
    graph.add_node("search_discovery", search_discovery)
    graph.add_node("crawler", crawler)
    graph.add_node("indexer", indexer_node)
    graph.add_node("retriever", retriever)
    graph.add_node("synthesizer", synthesizer)

    graph.set_entry_point("query_generator")
    graph.add_edge("query_generator", "search_discovery")
    graph.add_edge("search_discovery", "crawler")
    graph.add_edge("crawler", "indexer")
    graph.add_edge("indexer", "retriever")
    graph.add_edge("retriever", "synthesizer")
    graph.add_edge("synthesizer", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
