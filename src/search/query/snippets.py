import re
from typing import List


class SnippetExtractor:
    def __init__(self, window_size: int = 35):
        self.window_size = window_size

    def extract(self, text: str, query: str) -> str:
        words = text.split()
        if len(words) <= self.window_size:
            snippet = text
        else:
            query_terms = set(re.findall(r"\w+", query.lower()))
            best_window_idx = 0
            max_matches = -1

            for i in range(0, len(words) - self.window_size + 1, 10):
                window = words[i : i + self.window_size]
                window_tokens = set(re.findall(r"\w+", " ".join(window).lower()))
                matches = len(query_terms.intersection(window_tokens))
                if matches > max_matches:
                    max_matches = matches
                    best_window_idx = i

            selected_words = words[best_window_idx : best_window_idx + self.window_size]
            snippet = "... " + " ".join(selected_words) + " ..."

        for term in re.findall(r"\w+", query):
            if len(term) > 2:
                pattern = re.compile(rf"\b({re.escape(term)})\b", re.IGNORECASE)
                snippet = pattern.sub(r"<mark>\1</mark>", snippet)

        return snippet
