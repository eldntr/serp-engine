from typing import Any, Dict, List
import torch
from sentence_transformers import CrossEncoder
from src.core.config import settings


class NeuralReranker:
    def __init__(self, model_name: str = None, device: str = None):
        model_name = model_name or settings.retrieval.cross_encoder_model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        pairs = [[query, cand["text"]] for cand in candidates]
        scores = self.model.predict(pairs)

        for i, candidate in enumerate(candidates):
            candidate["cross_score"] = float(scores[i])

        reranked = sorted(candidates, key=lambda x: x["cross_score"], reverse=True)
        return reranked[:top_k]
