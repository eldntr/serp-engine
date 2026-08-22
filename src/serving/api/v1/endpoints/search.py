import time
from fastapi import APIRouter, Query, Request
from src.serving.schemas.serp import SERPResponse, SERPResultItem

router = APIRouter()


@router.get("", response_model=SERPResponse)
async def search_serp(
    request: Request,
    q: str = Query(..., min_length=2, description="Search query string"),
    top_k: int = Query(5, ge=1, le=20),
    include_summary: bool = Query(True),
):
    start_time = time.perf_counter()

    dense_enc = request.app.state.dense_encoder
    sparse_enc = request.app.state.sparse_encoder
    indexer = request.app.state.indexer
    rrf = request.app.state.rrf
    reranker = request.app.state.reranker
    snippet_extractor = request.app.state.snippet_extractor
    synthesizer = request.app.state.synthesizer

    q_dense = dense_enc.encode(q)[0]
    q_sparse = sparse_enc.encode(q)

    dense_hits = await indexer.search_dense(q_dense, limit=30)
    sparse_hits = await indexer.search_sparse(q_sparse, limit=30)

    fused_candidates = rrf.fuse(dense_hits, sparse_hits, top_n=20)

    candidate_dicts = [
        {
            "id": c.doc_id,
            "text": c.payload.get("text", ""),
            "title": c.payload.get("title", "No Title"),
            "url": c.payload.get("url", ""),
            "metadata": c.payload.get("metadata", {}),
        }
        for c in fused_candidates
    ]

    ranked_docs = reranker.rerank(query=q, candidates=candidate_dicts, top_k=top_k)

    results = []
    for rank_idx, doc in enumerate(ranked_docs, start=1):
        snippet = snippet_extractor.extract(doc["text"], q)
        results.append(
            SERPResultItem(
                rank=rank_idx,
                title=doc["title"],
                url=doc["url"],
                snippet=snippet,
                relevance_score=round(doc.get("cross_score", 0.0), 4),
                metadata=doc.get("metadata", {}),
            )
        )

    summary = None
    if include_summary and results:
        summary = await synthesizer.generate_summary(q, ranked_docs)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return SERPResponse(
        query=q,
        execution_time_ms=round(elapsed_ms, 2),
        total_hits=len(fused_candidates),
        results=results,
        generative_answer=summary,
    )
