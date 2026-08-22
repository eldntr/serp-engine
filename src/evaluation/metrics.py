import math
from typing import List


class IREvaluationMetrics:
    @staticmethod
    def reciprocal_rank(ranked_doc_ids: List[str], ground_truth_id: str) -> float:
        """Menghitung Reciprocal Rank (RR) untuk single query."""
        for rank, doc_id in enumerate(ranked_doc_ids, start=1):
            if doc_id == ground_truth_id:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def mean_reciprocal_rank(all_ranks: List[List[str]], all_ground_truths: List[str]) -> float:
        """Menghitung Mean Reciprocal Rank (MRR)."""
        if not all_ranks:
            return 0.0
        scores = [
            IREvaluationMetrics.reciprocal_rank(ranked, gt)
            for ranked, gt in zip(all_ranks, all_ground_truths)
        ]
        return sum(scores) / len(scores)

    @staticmethod
    def ndcg_at_k(ranked_doc_ids: List[str], relevance_scores: dict, k: int = 10) -> float:
        """Menghitung Normalized Discounted Cumulative Gain (NDCG@K)."""
        top_k = ranked_doc_ids[:k]
        dcg = 0.0

        for i, doc_id in enumerate(top_k, start=1):
            rel = relevance_scores.get(doc_id, 0)
            dcg += (2**rel - 1) / math.log2(i + 1)

        # Hitung Ideal DCG (IDCG)
        ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum((2**rel - 1) / math.log2(i + 1) for i, rel in enumerate(ideal_rels, start=1))

        if idcg == 0.0:
            return 0.0
        return dcg / idcg
