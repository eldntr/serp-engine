from typing import Literal
from fastapi import APIRouter, Request
from pydantic import BaseModel
from src.nodes.graph import build_agent_graph
from src.core.logger import logger

router = APIRouter()


class AgentSearchRequest(BaseModel):
    prompt: str
    max_queries: int = 4
    num_results_per_query: int = 3
    thread_id: str = "default"
    mode: Literal["linear", "react"] = "linear"


class AgentSearchResponse(BaseModel):
    prompt: str
    queries: list[str]
    urls: list[str]
    indexed_count: int
    answer: str


@router.post("/search", response_model=AgentSearchResponse)
async def agent_search(payload: AgentSearchRequest, request: Request):
    """
    Triggers the autonomous LangGraph agent to answer the user question.
    
    Supports:
      - 'linear' mode: deterministic sequential execution.
      - 'react' mode: dynamic tool-calling loop execution.
    """
    indexer = getattr(request.app.state, "indexer", None)

    logger.info(f"Running Agent Graph for prompt: '{payload.prompt}' in mode: '{payload.mode}'")
    
    graph = build_agent_graph(
        indexer=indexer,
        max_queries=payload.max_queries,
        num_results_per_query=payload.num_results_per_query,
        mode=payload.mode,
    )
    
    initial_state = {
        "prompt": payload.prompt,
        "queries": [],
        "urls": [],
        "documents": [],
        "indexed_count": 0,
        "answer": "",
        "messages": [],
    }

    config = {"configurable": {"thread_id": payload.thread_id}}
    final_state = await graph.ainvoke(initial_state, config=config)

    return AgentSearchResponse(
        prompt=final_state["prompt"],
        queries=final_state.get("queries", []),
        urls=final_state.get("urls", []),
        indexed_count=final_state.get("indexed_count", 0),
        answer=final_state.get("answer", ""),
    )
