from fastapi import APIRouter, Request
from pydantic import BaseModel
from src.nodes.graph import build_agent_graph
from src.core.logger import logger

router = APIRouter()


class AgentSearchRequest(BaseModel):
    prompt: str
    max_queries: int = 4
    num_results_per_query: int = 3


class AgentSearchResponse(BaseModel):
    prompt: str
    queries: list[str]
    urls: list[str]
    indexed_count: int
    answer: str


@router.post("/search", response_model=AgentSearchResponse)
async def agent_search(payload: AgentSearchRequest, request: Request):
    """
    Memicu agen RAG otonom LangGraph untuk menjawab pertanyaan user.
    
    Alur:
      1. QueryGeneratorNode  : Mengubah prompt menjadi search queries
      2. SearchDiscoveryNode : Mencari URL via SearXNG & validasi robots.txt
      3. CrawlerNode         : Crawl halaman web & ekstrak teks Markdown
      4. IndexerNode         : Encode & simpan ke Qdrant
      5. SynthesizerNode     : Sintesis jawaban berlandaskan dokumen dari web
    """
    # Gunakan indexer dari app.state jika tersedia (untuk koleksi yang persisten)
    indexer = getattr(request.app.state, "indexer", None)

    logger.info(f"Menjalankan Agent Graph untuk prompt: '{payload.prompt}'")
    
    graph = build_agent_graph(
        indexer=indexer,
        max_queries=payload.max_queries,
        num_results_per_query=payload.num_results_per_query,
    )
    
    initial_state = {
        "prompt": payload.prompt,
        "queries": [],
        "urls": [],
        "documents": [],
        "indexed_count": 0,
        "answer": "",
    }

    final_state = await graph.ainvoke(initial_state)

    return AgentSearchResponse(
        prompt=final_state["prompt"],
        queries=final_state.get("queries", []),
        urls=final_state.get("urls", []),
        indexed_count=final_state.get("indexed_count", 0),
        answer=final_state.get("answer", ""),
    )
