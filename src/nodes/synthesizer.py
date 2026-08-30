from typing import Any, Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from src.client.factory import get_llm_client
from src.core.config import settings
from src.core.logger import logger


class SynthesizerNode:
    """LangGraph node to synthesize a grounded answer from crawled documents using an LLM and LangChain."""

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
        """Builds context from a list of documents, truncated to prevent exceeding token limits."""
        context_parts = []
        total_chars = 0
        for i, doc in enumerate(documents, start=1):
            title = doc.get("title", "No Title")
            url = doc.get("url", "")
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
            logger.warning("Node 'SynthesizerNode' has no documents for synthesis.")
            return {"answer": "No documents were successfully crawled to answer this question."}

        context = self._build_context(documents)

        system_prompt = (
            "You are a grounded AI assistant. Answer the user's question strictly based on the "
            "provided web sources. Cite sources using their number in brackets (e.g., [1], [2]). "
            "If the sources don't contain enough information, say so clearly. "
            "Answer in the same language as the user's question. Be concise and factual."
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "User Question: {prompt}\n\nWeb Sources:\n{context}\n\nGrounded Answer:")
        ])

        logger.info(f"Starting answer synthesis for prompt: '{prompt}'")
        try:
            chain = prompt_template | self.client
            response = await chain.ainvoke({"prompt": prompt, "context": context})
            
            logger.success("Answer synthesis successful.")
            return {"answer": response.content}
        except Exception as e:
            logger.error(f"Failed to perform synthesis: {e}")
            return {"answer": f"An error occurred while synthesizing the answer: {e}"}
