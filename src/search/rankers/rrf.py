from typing import Any, Dict, List
from pydantic import BaseModel
from src.core.config import settings


class RankedCandidate(BaseModel):
    doc_id: str
    rrf_score: float
    payload: Dict[str, Any]


class ReciprocalRankFusion:
    def __init__(self, k: int = None):
        self.k = k or settings.retrieval.rrf_k

    def fuse(
        self,
        dense_results: List[Any],
        sparse_results: List[Any],
        top_n: int = 30,
    ) -> List[RankedCandidate]:
        scores: Dict[str, float] = {}
        doc_payloads: Dict[str, Dict[str, Any]] = {}

        for rank, point in enumerate(dense_results, start=1):
            doc_id = str(point.id)
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (self.k + rank))
            doc_payloads[doc_id] = point.payload

        for rank, point in enumerate(sparse_results, start=1):
            doc_id = str(point.id)
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (self.k + rank))
            doc_payloads[doc_id] = point.payload

        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        return [
            RankedCandidate(
                doc_id=doc_id,
                rrf_score=score,
                payload=doc_payloads[doc_id],
            )
            for doc_id, score in sorted_docs[:top_n]
        ]
