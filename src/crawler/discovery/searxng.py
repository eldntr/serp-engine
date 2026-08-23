import httpx
from typing import List, Dict, Any, Optional

class SearXNGDiscovery:
    def __init__(self, base_url: str = "http://localhost:8080"):
        """
        Jika crawler berjalan di dalam container Docker yang sama di network docker-compose,
        gunakan base_url='http://searxng:8080'.
        Jika dijalankan langsung dari host mesin lokal, gunakan 'http://localhost:8080'.
        """
        self.base_url = base_url.rstrip("/")

    async def search(
        self,
        query: str,
        engines: Optional[List[str]] = None,
        num_results: int = 10,
        language: str = "id-ID"
    ) -> List[str]:
        params: Dict[str, Any] = {
            "q": query,
            "format": "json",
            "language": language,
        }
        
        if engines:
            params["engines"] = ",".join(engines) # e.g. ["google", "bing", "duckduckgo"]

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.base_url}/search", params=params)
            response.raise_for_status()
            data = response.json()

        results = data.get("results", [])
        urls = []
        for item in results:
            url = item.get("url")
            if url and url.startswith("http") and url not in urls:
                urls.append(url)
                if len(urls) >= num_results:
                    break

        return urls