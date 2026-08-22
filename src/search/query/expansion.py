import re
from typing import List


class QueryExpansion:
    def __init__(self):
        self.stopwords = {"apa", "itu", "bagaimana", "cara", "yang", "dan", "di", "ke", "what", "is", "how", "to"}

    def clean_query(self, query: str) -> str:
        return re.sub(r"[^\w\s]", "", query).strip()

    def generate_keywords(self, query: str) -> List[str]:
        tokens = self.clean_query(query).lower().split()
        return [t for t in tokens if t not in self.stopwords and len(t) > 2]
