from typing import Any, Dict, List
from src.search.encoders.dense import DenseEncoder
from src.search.encoders.sparse import BM25SparseEncoder
from src.search.indexing.base import BaseIndexer
from src.search.indexing.qdrant_client import QdrantHybridIndexer
from src.search.rankers.rrf import ReciprocalRankFusion
from src.search.rankers.cross_encoder import NeuralReranker
from src.core.logger import logger

class RetrieverNode:
    """LangGraph node to retrieve top K relevant document chunks from the search index database."""
    
    def __init__(self, indexer: BaseIndexer = None, top_k: int = 5):
        self._injected_indexer = indexer
        self.top_k = top_k
        self.dense_encoder = DenseEncoder()
        self.sparse_encoder = BM25SparseEncoder()
        self.rrf = ReciprocalRankFusion()
        self.reranker = NeuralReranker()

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        prompt: str = state.get("prompt", "")
        if not prompt:
            logger.warning("Node 'RetrieverNode' received an empty prompt.")
            return {"documents": []}

        indexer = self._injected_indexer
        if indexer is None:
            logger.warning("RetrieverNode using in-memory Qdrant fallback.")
            indexer = QdrantHybridIndexer(host=":memory:")
            await indexer.init_collection()

        logger.info(f"Retrieving relevant document chunks for prompt: '{prompt}'")
        try:
            q_dense = self.dense_encoder.encode(prompt)[0]
            q_sparse = self.sparse_encoder.encode(prompt)

            dense_hits = await indexer.search_dense(q_dense, limit=30)
            sparse_hits = await indexer.search_sparse(q_sparse, limit=30)

            fused_candidates = self.rrf.fuse(dense_hits, sparse_hits, top_n=20)

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

            if not candidate_dicts:
                logger.warning("No candidate document chunks retrieved from database.")
                return {"documents": []}

            ranked_docs = self.reranker.rerank(query=prompt, candidates=candidate_dicts, top_k=self.top_k)

            logger.success(f"Retrieved {len(ranked_docs)} relevant document chunks from search index")
            return {"documents": ranked_docs}
        except Exception as e:
            logger.error(f"Failed to retrieve document chunks: {e}")
            return {"documents": []}
