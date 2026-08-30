from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.client.factory import get_llm_client
from src.core.config import settings
from src.core.logger import logger


class SearchQueries(BaseModel):
    """Schema for structured search query output."""
    queries: List[str] = Field(
        description="A list of keyword-focused, concise, and diverse search engine queries."
    )


class QueryGeneratorNode:
    """LangGraph node to convert a user prompt into a list of web search queries using a structured LLM call."""
    
    def __init__(
        self, 
        provider: Optional[str] = None, 
        api_base: Optional[str] = None, 
        model: Optional[str] = None,
        max_queries: int = 4
    ):
        self.provider = provider or settings.llm.llm_provider
        self.api_base = api_base or settings.llm.llm_api_base
        self.model = model or settings.llm.llm_model
        self.max_queries = max_queries
        
        self.client = get_llm_client(
            provider=self.provider,
            api_base=self.api_base,
            model=self.model,
            api_key=settings.llm.llm_api_key,
            timeout=settings.llm.llm_timeout
        )

    async def generate_search_queries(self, prompt: str) -> List[str]:
        """Receives a prompt and converts it into a list of optimal search keywords using structured output."""
        logger.info(f"Starting to generate search queries for prompt: '{prompt}' using model '{self.model}'")
        
        system_prompt = (
            "You are an expert search query planner. Your task is to analyze a user prompt "
            "and generate a list of search queries that will retrieve the most relevant "
            "and factual information from the web to answer the prompt.\n\n"
            "Guidelines:\n"
            "1. Generate search queries that are keyword-focused, concise, and diverse.\n"
            f"2. Return between 1 and {self.max_queries} queries."
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "User Prompt: {prompt}")
        ])

        try:
            structured_llm = self.client.with_structured_output(SearchQueries)
            chain = prompt_template | structured_llm
            
            result = await chain.ainvoke({"prompt": prompt})
            
            queries = [q.strip() for q in result.queries if q.strip()]
            queries = queries[:self.max_queries]
            
            logger.success(f"Successfully generated {len(queries)} search queries: {queries}")
            return queries
        except Exception as e:
            logger.error(f"Failed to process search queries: {e}")
            return [prompt]

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node signature for LangGraph StateGraph integration.
        
        Receives Graph State (dict) containing the key 'prompt' and returns a 
        state update dict with the key 'queries'.
        """
        prompt = state.get("prompt", "")
        if not prompt:
            logger.warning("Node 'QueryGeneratorNode' received an empty prompt in State.")
            return {"queries": []}
            
        queries = await self.generate_search_queries(prompt)
        return {"queries": queries}
