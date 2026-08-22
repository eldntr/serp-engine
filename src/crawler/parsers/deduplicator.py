import re
from typing import List, Set
from datasketch import MinHash, MinHashLSH


class ContentDeduplicator:
    def __init__(
        self,
        threshold: float = 0.85,
        num_perm: int = 128,
        shingle_size: int = 3,
    ):

        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self._indexed_keys: Set[str] = set()

    def _generate_shingles(self, text: str) -> List[str]:
        normalized = re.sub(r"[^\w\s]", "", text.lower())
        tokens = normalized.split()

        if len(tokens) < self.shingle_size:
            return tokens

        shingles = [
            " ".join(tokens[i : i + self.shingle_size])
            for i in range(len(tokens) - self.shingle_size + 1)
        ]
        return shingles

    def compute_minhash(self, text: str) -> MinHash:
        minhash = MinHash(num_perm=self.num_perm)
        shingles = self._generate_shingles(text)
        for s in shingles:
            minhash.update(s.encode("utf8"))
        return minhash

    def is_duplicate(self, doc_id: str, text: str) -> bool:
        minhash = self.compute_minhash(text)
        candidates = self.lsh.query(minhash)

        return any(cand != doc_id for cand in candidates)

    def insert(self, doc_id: str, text: str) -> bool:
        minhash = self.compute_minhash(text)
        candidates = self.lsh.query(minhash)

        if any(cand != doc_id for cand in candidates):
            return False

        if doc_id not in self._indexed_keys:
            self.lsh.insert(doc_id, minhash)
            self._indexed_keys.add(doc_id)

        return True