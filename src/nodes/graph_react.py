from typing import List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.core.config import settings


def build_react_graph(
    AgentState,
    indexer=None,
    max_queries: int = 4,
    num_results_per_query: int = 3,
) -> StateGraph:
    """Builds and compiles the dynamic tool-calling ReAct agent loop graph."""
    graph = StateGraph(AgentState)

    from src.client.factory import get_llm_client
    client = get_llm_client(
        provider=settings.llm.llm_provider,
        api_base=settings.llm.llm_api_base,
        model=settings.llm.llm_model,
        api_key=settings.llm.llm_api_key,
        timeout=settings.llm.llm_timeout,
    )

    from pydantic import BaseModel, Field

    class SearchWebSchema(BaseModel):
        query: str = Field(description="Search query to execute via SearXNG search engine.")

    class CrawlAndIndexPagesSchema(BaseModel):
        urls: List[str] = Field(description="List of URLs to crawl and index into the database.")

    class RetrieveRelevantChunksSchema(BaseModel):
        query: str = Field(description="Query prompt to retrieve the most relevant semantic chunks from Qdrant.")

    llm_with_tools = client.bind_tools([
        {
            "name": "search_web",
            "description": "Search the web for a query to discover new URLs.",
            "parameters": SearchWebSchema.model_json_schema()
        },
        {
            "name": "crawl_and_index_pages",
            "description": "Crawl web pages and index them into Qdrant database.",
            "parameters": CrawlAndIndexPagesSchema.model_json_schema()
        },
        {
            "name": "retrieve_relevant_chunks",
            "description": "Query the Qdrant database to retrieve relevant semantic chunks.",
            "parameters": RetrieveRelevantChunksSchema.model_json_schema()
        }
    ])

    async def agent_node(state: Dict[str, Any]):
        from langchain_core.messages import SystemMessage, HumanMessage

        prompt = state["prompt"]
        messages = state.get("messages", [])

        if not messages:
            system_msg = SystemMessage(
                content="You are an autonomous research assistant. Your task is to resolve the user prompt fully and autonomously by executing tools. "
                        "Do NOT ask the user for permission, choices, or clarification. Always proceed autonomously as follows:\n"
                        "1. If you do not have relevant URLs, call 'search_web' with a search query.\n"
                        "2. Once you have URL search results, immediately call 'crawl_and_index_pages' with the discovered URLs.\n"
                        "3. After indexing, call 'retrieve_relevant_chunks' to get semantic text chunks from the database.\n"
                        "4. Use the retrieved chunks to write a grounded, detailed answer. Cite sources using brackets (e.g. [1], [2]).\n"
                        "Keep calling tools in sequence until you have enough grounded facts to write the final answer. "
                        "Answer in the same language as the query."
            )
            user_msg = HumanMessage(content=prompt)
            messages = [system_msg, user_msg]
            response = await llm_with_tools.ainvoke(messages)
            return {
                "messages": [system_msg, user_msg, response],
                "answer": response.content
            }

        response = await llm_with_tools.ainvoke(messages)
        return {
            "messages": [response],
            "answer": response.content
        }

    async def execute_tools(state: Dict[str, Any]):
        from langchain_core.messages import ToolMessage
        from src.nodes.search import SearchDiscoveryNode
        from src.nodes.crawler import CrawlerNode
        from src.nodes.indexer import IndexerNode
        from src.nodes.retriever import RetrieverNode

        last_message = state["messages"][-1]
        new_messages = []

        queries = list(state.get("queries", []))
        urls = list(state.get("urls", []))
        documents = list(state.get("documents", []))
        indexed_count = state.get("indexed_count", 0)

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            args = tool_call["args"]
            tool_call_id = tool_call["id"]

            if tool_name == "search_web":
                query = args.get("query", "")
                if query not in queries:
                    queries.append(query)
                search_node = SearchDiscoveryNode(num_results_per_query=num_results_per_query)
                search_res = await search_node({"queries": [query]})
                new_urls = search_res.get("urls", [])
                for u in new_urls:
                    if u not in urls:
                        urls.append(u)
                output = f"Discovered URLs for query '{query}': {new_urls}"

            elif tool_name == "crawl_and_index_pages":
                target_urls = args.get("urls", [])
                crawler_node = CrawlerNode()
                crawler_res = await crawler_node({"urls": target_urls})
                crawled_docs = crawler_res.get("documents", [])
                for doc in crawled_docs:
                    if doc not in documents:
                        documents.append(doc)
                indexer_node = IndexerNode(indexer=indexer)
                indexer_res = await indexer_node({"documents": crawled_docs})
                indexed_count += indexer_res.get("indexed_count", 0)
                output = f"Crawled and indexed {len(crawled_docs)} pages. Total chunks indexed: {indexer_res.get('indexed_count', 0)}"

            elif tool_name == "retrieve_relevant_chunks":
                query = args.get("query", "")
                retriever_node = RetrieverNode(indexer=indexer)
                retriever_res = await retriever_node({"prompt": query})
                retrieved_docs = retriever_res.get("documents", [])
                documents = retrieved_docs
                output = f"Retrieved {len(retrieved_docs)} relevant chunks for query '{query}'."

            else:
                output = f"Unknown tool: {tool_name}"

            new_messages.append(ToolMessage(content=output, tool_call_id=tool_call_id))

        return {
            "queries": queries,
            "urls": urls,
            "documents": documents,
            "indexed_count": indexed_count,
            "messages": new_messages
        }

    def route_agent(state: Dict[str, Any]):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "execute_tools"
        return END

    graph.add_node("agent", agent_node)
    graph.add_node("execute_tools", execute_tools)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_agent, ["execute_tools", END])
    graph.add_edge("execute_tools", "agent")

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
