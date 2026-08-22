import math
import re
from collections import Counter
from typing import Dict, List
from src.core.config import settings


class BM25SparseEncoder:
    def __init__(self, k1: float = None, b: float = None, avg_doc_len: float = None):
        self.k1 = k1 or settings.retrieval.sparse_k1
        self.b = b or settings.retrieval.sparse_b
        self.avg_doc_len = avg_doc_len or settings.retrieval.sparse_avg_doc_len

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def encode(self, text: str) -> Dict[str, float]:
        tokens = self._tokenize(text)
        if not tokens:
            return {}

        doc_len = len(tokens)
        counts = Counter(tokens)
        sparse_vector = {}

        for token, freq in counts.items():
            tf = (freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len)))
            sparse_vector[token] = float(round(tf, 4))

        return sparse_vector
