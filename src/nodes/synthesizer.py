from typing import Any, Dict, List, Optional
from src.client.factory import get_llm_client
from src.core.config import settings
from src.core.logger import logger

class SynthesizerNode:
    """LangGraph node to synthesize a grounded answer from crawled documents using an LLM."""

    def __init__(
        self,
        provider: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.client = get_llm_client(
            provider=provider or settings.llm.llm_provider,
            api_base=api_base or settings.llm.llm_api_base,
            model=model or settings.llm.llm_model,
            api_key=settings.llm.llm_api_key,
            timeout=settings.llm.llm_timeout,
        )

    def _build_context(self, documents: List[Dict[str, Any]], max_chars: int = 6000) -> str:
        """Membangun konteks dari daftar dokumen, dipotong agar tidak melebihi batas token."""
        context_parts = []
        total_chars = 0
        for i, doc in enumerate(documents, start=1):
            title = doc.get("title", "No Title")
            url = doc.get("url", "")
            # Ambil snippet untuk menghemat konteks
            text = doc.get("text", "")[:1000]
            entry = f"[{i}] {title} ({url})\n{text}\n"
            if total_chars + len(entry) > max_chars:
                break
            context_parts.append(entry)
            total_chars += len(entry)
        return "\n---\n".join(context_parts)

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        prompt: str = state.get("prompt", "")
        documents: List[Dict[str, Any]] = state.get("documents", [])

        if not documents:
            logger.warning("Node 'SynthesizerNode' tidak memiliki dokumen untuk sintesis.")
            return {"answer": "Tidak ada dokumen yang berhasil di-crawl untuk menjawab pertanyaan ini."}

        context = self._build_context(documents)

        system_prompt = (
            "You are a grounded AI assistant. Answer the user's question strictly based on the "
            "provided web sources. Cite sources using their number in brackets (e.g., [1], [2]). "
            "If the sources don't contain enough information, say so clearly. "
            "Answer in the same language as the user's question. Be concise and factual."
        )

        user_prompt = (
            f"User Question: {prompt}\n\n"
            f"Web Sources:\n{context}\n\n"
            "Grounded Answer:"
        )

        logger.info(f"Mulai sintesis jawaban untuk prompt: '{prompt}'")
        try:
            answer = await self.client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
            )
            logger.success("Sintesis jawaban berhasil.")
            return {"answer": answer}
        except Exception as e:
            logger.error(f"Gagal melakukan sintesis: {e}")
            return {"answer": f"Terjadi kesalahan saat mensintesis jawaban: {e}"}
