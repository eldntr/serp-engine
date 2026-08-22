import httpx
from fake_useragent import UserAgent
from typing import Optional
from src.crawler.engine.schemas import CrawlResult
from src.core.config import settings
from src.core.logger import logger
from src.core.exceptions import CrawlerError

ua = UserAgent()


class StaticCrawler:
    def __init__(self, timeout: float = None, follow_redirects: bool = True):
        self.timeout = timeout or (settings.crawler.timeout / 1000.0)
        self.follow_redirects = follow_redirects
        self._headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

    async def fetch(self, url: str) -> CrawlResult:
        headers = self._headers.copy()
        headers["User-Agent"] = ua.random

        async with httpx.AsyncClient(
            http2=True,
            timeout=self.timeout,
            follow_redirects=self.follow_redirects,
            verify=False  # Opsional: cegah crash saat SSL error
        ) as client:
            try:
                response = await client.get(url, headers=headers)
                return CrawlResult(
                    url=str(response.url),
                    status_code=response.status_code,
                    html=response.text,
                    content_length=len(response.text),
                    engine="static",
                    headers=dict(response.headers),
                )
            except Exception as e:
                logger.error(f"Static crawler failed for {url}: {str(e)}")
                return CrawlResult(
                    url=url,
                    status_code=0,
                    html="",
                    content_length=0,
                    engine="static",
                    error=str(e),
                )