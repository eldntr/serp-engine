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
            logger.warning("Node 'SearchDiscoveryNode' received an empty query list.")
            return {"urls": []}
            
        discovered_urls: List[str] = []
        
        for q in queries:
            logger.info(f"Searching URLs for query: '{q}' via SearXNG")
            try:
                urls = await self.discovery.search(q, num_results=self.num_results_per_query)
                for u in urls:
                    if u not in discovered_urls:
                        is_allowed = await self.robots.can_fetch(u)
                        if is_allowed:
                            discovered_urls.append(u)
                        else:
                            logger.info(f"URL skipped due to robots.txt rules: {u}")
            except Exception as e:
                logger.error(f"Failed to search URLs for query '{q}': {e}")
                
        logger.success(f"Successfully discovered {len(discovered_urls)} URLs verified by robots.txt")
        return {"urls": discovered_urls}
