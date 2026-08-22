from pydantic import BaseModel
from typing import Optional, Dict, Any

class CleanedDocument(BaseModel):
    url: str
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    content_markdown: str
    raw_text: str
    word_count: int
    metadata: Dict[str, Any] = {}
