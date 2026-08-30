from typing import Any, Dict, List
from langchain_core.prompts import ChatPromptTemplate
from src.client.factory import get_llm_client
from src.core.config import settings
from src.core.logger import logger


class GroundedRAGSynthesizer:
    """RAG Synthesizer that uses LangChain to generate an answer grounded in search results."""

    def __init__(self, api_key: str = None):
        self.provider = settings.llm.llm_provider
        self.api_base = settings.llm.llm_api_base
        self.model = settings.llm.llm_model
        
        self.llm = get_llm_client(
            provider=self.provider,
            api_base=self.api_base,
            model=self.model,
            api_key=api_key or settings.llm.llm_api_key,
            timeout=settings.llm.llm_timeout,
        )

    async def generate_summary(self, query: str, ranked_docs: List[Dict[str, Any]]) -> str:
        """Synthesizes a structured answer grounded in search results using LangChain."""
        if not ranked_docs:
            return "No relevant documents found to answer this query."

        # Build context from top 5 documents
        context_parts = []
        for idx, doc in enumerate(ranked_docs[:5], start=1):
            title = doc.get("title", "No Title")
            url = doc.get("url", "")
            text = doc.get("text", "")
            # Clean text snippet
            snippet = text[:1500].replace("<mark>", "").replace("</mark>", "")
            context_parts.append(f"[{idx}] {title} ({url})\n{snippet}")
            
        context = "\n---\n".join(context_parts)

        system_prompt = (
            "You are a helpful and grounded assistant. Answer the user's query based strictly on the "
            "provided web sources. Cite sources using their number in brackets (e.g., [1], [2]). "
            "If the sources don't contain enough information, say so clearly. "
            "Answer in the same language as the query (default to Indonesian if not clear). "
            "Be concise, factual, and direct."
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Query: {query}\n\nWeb Sources:\n{context}\n\nGrounded Answer:")
        ])

        logger.info(f"Synthesizing RAG answer for query: '{query}' using LangChain")
        try:
            chain = prompt_template | self.llm
            response = await chain.ainvoke({"query": query, "context": context})
            return response.content
        except Exception as e:
            logger.error(f"Failed to perform RAG synthesis: {e}")
            return "Sorry, an error occurred while synthesizing the answer from search results."
