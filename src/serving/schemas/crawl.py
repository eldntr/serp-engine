from pydantic import BaseModel, HttpUrl
from typing import Optional, List


class CrawlTriggerRequest(BaseModel):
    url: str
    force_dynamic: bool = False


class CrawlBatchRequest(BaseModel):
    urls: List[str]
    force_dynamic: bool = False


class CrawlJobResponse(BaseModel):
    job_id: str
    url: str
    status: str
