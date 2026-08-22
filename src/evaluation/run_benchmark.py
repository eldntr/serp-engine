import asyncio
import time
from src.search.encoders.dense import DenseEncoder
from src.search.encoders.sparse import BM25SparseEncoder
from src.search.indexing.qdrant_client import QdrantHybridIndexer
from src.search.rankers.rrf import ReciprocalRankFusion
from src.search.rankers.cross_encoder import NeuralReranker
from src.evaluation.metrics import IREvaluationMetrics
from src.core.logger import logger
from src.core.config import settings


async def run_benchmark():
    logger.info("=== INITIALIZING BENCHMARK SUITE ===")
    dense_enc = DenseEncoder()
    sparse_enc = BM25SparseEncoder()
    indexer = QdrantHybridIndexer(host=":memory:")
    await indexer.init_collection()

    rrf = ReciprocalRankFusion()
    reranker = NeuralReranker()

    # Ground Truth Dataset
    test_docs = [
        {"id": "doc_python", "text": "Python is an interpreted, high-level, general-purpose programming language."},
        {"id": "doc_fastapi", "text": "FastAPI is a modern, fast web framework for building APIs with Python."},
        {"id": "doc_qdrant", "text": "Qdrant is a vector similarity search engine and database for neural search."},
        {"id": "doc_hybrid", "text": "Hybrid search blends sparse lexical BM25 matching with dense embeddings."},
    ]

    for d in test_docs:
        d_vec = dense_enc.encode(d["text"])[0]
        s_vec = sparse_enc.encode(d["text"])
        await indexer.upsert_document(d["id"], d_vec, s_vec, d)

    test_queries = [
        {"query": "web framework python api", "target": "doc_fastapi", "rel": {"doc_fastapi": 2, "doc_python": 1}},
        {"query": "neural search vector database", "target": "doc_qdrant", "rel": {"doc_qdrant": 2, "doc_hybrid": 1}},
        {"query": "combine bm25 and dense embeddings", "target": "doc_hybrid", "rel": {"doc_hybrid": 2}},
    ]

    all_ranked_ids = []
    ground_truths = []
    ndcg_scores = []
    latencies = []

    logger.info("\n=== RUNNING BENCHMARK EVALUATION ===")
    for item in test_queries:
        q = item["query"]
        t_start = time.perf_counter()

        q_dense = dense_enc.encode(q)[0]
        q_sparse = sparse_enc.encode(q)
        d_hits = await indexer.search_dense(q_dense, limit=10)
        s_hits = await indexer.search_sparse(q_sparse, limit=10)

        fused = rrf.fuse(d_hits, s_hits, top_n=10)
        candidates = [{"id": f.payload["id"], "text": f.payload["text"]} for f in fused]
        reranked = reranker.rerank(q, candidates, top_k=5)

        elapsed = (time.perf_counter() - t_start) * 1000
        latencies.append(elapsed)

        ranked_ids = [r["id"] for r in reranked]
        all_ranked_ids.append(ranked_ids)
        ground_truths.append(item["target"])

        ndcg = IREvaluationMetrics.ndcg_at_k(ranked_ids, item["rel"], k=5)
        ndcg_scores.append(ndcg)

    mrr = IREvaluationMetrics.mean_reciprocal_rank(all_ranked_ids, ground_truths)
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)
    avg_latency = sum(latencies) / len(latencies)

    print("\n" + "=" * 45)
    print("        IR EVALUATION BENCHMARK RESULTS     ")
    print("=" * 45)
    print(f"Mean Reciprocal Rank (MRR)   : {mrr:.4f}")
    print(f"Average NDCG@5               : {avg_ndcg:.4f}")
    print(f"Average Pipeline Latency     : {avg_latency:.2f} ms")
    print("=" * 45)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
