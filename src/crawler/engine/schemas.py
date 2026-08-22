from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict


class CrawlResult(BaseModel):
    url: str
    status_code: int
    html: str
    content_length: int
    engine: str  # "static" atau "dynamic"
    headers: Dict[str, str] = {}
    error: Optional[str] = None