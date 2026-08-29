from typing import Any, Dict, List, Optional
from src.crawler.discovery.searxng import SearXNGDiscovery
from src.crawler.robots import RobotsManager
from src.core.config import settings
from src.core.logger import logger

class SearchDiscoveryNode:
    """LangGraph node to search queries using SearXNG and validate them with robots.txt."""
    
    def __init__(self, base_url: Optional[str] = None, num_results_per_query: int = 3):
        self.base_url = base_url or settings.crawler.searxng_base_url
        self.num_results_per_query = num_results_per_query
        self.discovery = SearXNGDiscovery(base_url=self.base_url)
        self.robots = RobotsManager()

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        queries: List[str] = state.get("queries", [])
        if not queries:
            logger.warning("Node 'SearchDiscoveryNode' menerima list query kosong.")
            return {"urls": []}
            
        discovered_urls: List[str] = []
        
        for q in queries:
            logger.info(f"Mencari URL untuk query: '{q}' via SearXNG")
            try:
                urls = await self.discovery.search(q, num_results=self.num_results_per_query)
                for u in urls:
                    if u not in discovered_urls:
                        # Validasi dengan robots.txt
                        is_allowed = await self.robots.can_fetch(u)
                        if is_allowed:
                            discovered_urls.append(u)
                        else:
                            logger.info(f"URL dilewati karena aturan robots.txt: {u}")
            except Exception as e:
                logger.error(f"Gagal mencari URL untuk query '{q}': {e}")
                
        logger.success(f"Berhasil menemukan {len(discovered_urls)} URL teruji robots.txt")
        return {"urls": discovered_urls}
