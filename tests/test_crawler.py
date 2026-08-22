import argparse
import asyncio
import time
from src.crawler.engine.static import StaticCrawler
from src.crawler.engine.dynamic import DynamicCrawler
from src.crawler.robots import RobotsManager
from tests.utils import save_crawl_result_json


async def main(debug: bool = False):
    # Target URL:
    # 1. SSR / Static: Wikipedia
    # 2. CSR / Dynamic (SPA): Quotes to Scrape (JS version)
    urls = [
        "https://en.wikipedia.org/wiki/Information_retrieval",
        "https://quotes.toscrape.com/js/",
        "https://www.mims.com/indonesia",  # Biasanya diblokir di robots.txt
    ]

    print("=== 0. TESTING ROBOTS.TXT ===")
    robots = RobotsManager()
    for url in urls:
        is_allowed = await robots.can_fetch(url)
        delay = robots.get_crawl_delay(url)
        print(f"[ROBOTS] {url}")
        print(f"  -> Allowed: {is_allowed} | Crawl Delay: {delay}")

    print("\n=== 1. TESTING STATIC CRAWLER (HTTPX) ===")
    static_crawler = StaticCrawler()
    for url in urls:
        if not await robots.can_fetch(url):
            print(f"[STATIC] {url}\n  -> BLOCKED by robots.txt")
            continue
            
        start_time = time.perf_counter()
        res = await static_crawler.fetch(url)
        elapsed = time.perf_counter() - start_time
        print(f"[{res.engine.upper()}] {url}")
        print(f"  -> Status: {res.status_code} | Size: {res.content_length} chars | Time: {elapsed:.3f}s")
        if res.error:
            print(f"  -> Error: {res.error}")
        else:
            save_crawl_result_json(res, debug=debug)

    print("\n=== 2. TESTING DYNAMIC CRAWLER (PLAYWRIGHT) ===")
    dynamic_crawler = DynamicCrawler(headless=True)
    await dynamic_crawler.start()

    for url in urls:
        if not await robots.can_fetch(url):
            print(f"[DYNAMIC] {url}\n  -> BLOCKED by robots.txt")
            continue

        start_time = time.perf_counter()
        res = await dynamic_crawler.fetch(url)
        elapsed = time.perf_counter() - start_time
        print(f"[{res.engine.upper()}] {url}")
        print(f"  -> Status: {res.status_code} | Size: {res.content_length} chars | Time: {elapsed:.3f}s")
        if res.error:
            print(f"  -> Error: {res.error}")
        else:
            save_crawl_result_json(res, debug=debug)

    await dynamic_crawler.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Crawler Script")
    parser.add_argument("--debug", action="store_true", help="Print debug information like result paths")
    args = parser.parse_args()
    asyncio.run(main(debug=args.debug))