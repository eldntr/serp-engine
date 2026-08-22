import urllib.robotparser
from urllib.parse import urlparse
import httpx
from typing import Dict, Optional


class RobotsManager:
    def __init__(self, user_agent: str = "MyCustomSERPBot"):
        self.user_agent = user_agent
        self._parsers: Dict[str, urllib.robotparser.RobotFileParser] = {}

    def _get_robots_url(self, url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    async def can_fetch(self, url: str) -> bool:
        """Mengecek apakah URL target diizinkan untuk di-crawl menurut robots.txt."""
        parsed = urlparse(url)
        domain = parsed.netloc

        if domain not in self._parsers:
            rp = urllib.robotparser.RobotFileParser()
            robots_url = self._get_robots_url(url)

            try:
                async with httpx.AsyncClient(timeout=5.0, verify=False) as client:
                    resp = await client.get(robots_url)
                    if resp.status_code == 200:
                        rp.parse(resp.text.splitlines())
                    else:
                        rp.allow_all = True
            except Exception:
                rp.allow_all = True

            self._parsers[domain] = rp

        return self._parsers[domain].can_fetch(self.user_agent, url)

    def get_crawl_delay(self, url: str) -> Optional[float]:
        """Mengambil nilai crawl-delay jika domain menentukan batas waktu jeda."""
        domain = urlparse(url).netloc
        if domain in self._parsers:
            delay = self._parsers[domain].crawl_delay(self.user_agent)
            return float(delay) if delay else None
        return None