from typing import List, Dict, Any


class GroundedRAGSynthesizer:
    def __init__(self, api_key: str = None):
        self.api_key = api_key

    async def generate_summary(self, query: str, ranked_docs: List[Dict[str, Any]]) -> str:
        if not ranked_docs:
            return "Tidak ada dokumen yang relevan untuk menjawab query ini."

        top_context = ranked_docs[:3]
        synthesis = f"Berdasarkan informasi yang ditemukan untuk pencarian '{query}':\n\n"
        for idx, doc in enumerate(top_context, start=1):
            title = doc.get("title", "No Title")
            snippet = doc.get("snippet", doc.get("text", ""))[:140].replace("<mark>", "").replace("</mark>", "")
            synthesis += f"- [{idx}] {title}: {snippet}...\n"

        synthesis += "\nHasil pencarian di atas disusun berdasarkan relevansi hybrid retrieval dan neural ranking."
        return synthesis
