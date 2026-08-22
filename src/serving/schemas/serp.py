from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class SERPResultItem(BaseModel):
    rank: int
    title: str
    url: str
    snippet: str
    relevance_score: float
    metadata: Dict[str, Any] = {}


class SERPResponse(BaseModel):
    query: str
    execution_time_ms: float
    total_hits: int
    results: List[SERPResultItem]
    generative_answer: Optional[str] = None
