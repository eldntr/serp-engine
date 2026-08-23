import argparse
import asyncio
import time
from src.crawler.discovery.searxng import SearXNGDiscovery
from src.crawler.engine.static import StaticCrawler
from src.crawler.robots import RobotsManager
from tests.utils import save_crawl_result_json


async def main(query: str, num_results: int, base_url: str, debug: bool = False):
    print("=== 0. TESTING URL DISCOVERY (SEARXNG) ===")
    discovery = SearXNGDiscovery(base_url=base_url)
    start_time = time.perf_counter()

    try:
        discovered_urls = await discovery.search(
            query=query,
            num_results=num_results
        )
        elapsed = time.perf_counter() - start_time
        print(f"[DISCOVERY] Query: '{query}'")
        print(f"  -> Found: {len(discovered_urls)} URLs | Time: {elapsed:.3f}s")
        for idx, url in enumerate(discovered_urls, 1):
            print(f"     {idx}. {url}")
    except Exception as e:
        print(f"[DISCOVERY ERROR] Failed to fetch search results from SearXNG: {e}")
        return

    if not discovered_urls:
        print("[WARNING] No URLs discovered. Check your SearXNG container status.")
        return

    print("\n=== 1. TESTING ROBOTS.TXT CHECK ON DISCOVERED URLS ===")
    robots = RobotsManager()
    allowed_urls = []
    for url in discovered_urls:
        is_allowed = await robots.can_fetch(url)
        delay = robots.get_crawl_delay(url)
        print(f"[ROBOTS] {url}")
        print(f"  -> Allowed: {is_allowed} | Crawl Delay: {delay}")
        if is_allowed:
            allowed_urls.append(url)

    print("\n=== 2. TESTING STATIC CRAWLER ON DISCOVERED URLS ===")
    static_crawler = StaticCrawler()
    for url in allowed_urls:
        start_time = time.perf_counter()
        res = await static_crawler.fetch(url)
        elapsed = time.perf_counter() - start_time

        print(f"[{res.engine.upper()}] {url}")
        print(f"  -> Status: {res.status_code} | Size: {res.content_length} chars | Time: {elapsed:.3f}s")
        if res.error:
            print(f"  -> Error: {res.error}")
        else:
            save_crawl_result_json(res, debug=debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test URL Discovery & Crawler Pipeline")
    parser.add_argument("--query", type=str, default="information retrieval evaluation metrics", help="Search query")
    parser.add_argument("--num", type=int, default=3, help="Number of URLs to discover")
    parser.add_argument("--searxng-url", type=str, default="http://localhost:8080", help="SearXNG Base URL")
    parser.add_argument("--debug", action="store_true", help="Print debug information like result paths")

    args = parser.parse_args()
    asyncio.run(
        main(
            query=args.query,
            num_results=args.num,
            base_url=args.searxng_url,
            debug=args.debug
        )
    )