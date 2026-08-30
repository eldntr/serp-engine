import asyncio
from arq.connections import RedisSettings
from loguru import logger
from src.crawler.engine.static import StaticCrawler
from src.crawler.engine.dynamic import DynamicCrawler
from src.crawler.parsers.cleaner import ContentCleaner
from src.crawler.parsers.deduplicator import ContentDeduplicator
from src.crawler.robots import RobotsManager


async def startup(ctx: dict):
    """Initialize crawler engine, parser, and deduplicator when worker starts up."""
    logger.info("Starting Background Crawler Worker Pool...")
    ctx["static_crawler"] = StaticCrawler()
    ctx["dynamic_crawler"] = DynamicCrawler(headless=True)
    await ctx["dynamic_crawler"].start()

    ctx["cleaner"] = ContentCleaner(min_output_length=100)
    ctx["dedup"] = ContentDeduplicator(threshold=0.85)
    ctx["robots"] = RobotsManager(user_agent="MyCustomSERPBot")


async def shutdown(ctx: dict):
    """Clean up browser instances when worker is shut down."""
    logger.info("Closing Crawler Worker Pool...")
    if "dynamic_crawler" in ctx:
        await ctx["dynamic_crawler"].close()


async def crawl_url_task(ctx: dict, url: str, force_dynamic: bool = False) -> dict:
    """Single queue task to process crawling of 1 URL."""
    robots: RobotsManager = ctx["robots"]
    static_crawler: StaticCrawler = ctx["static_crawler"]
    dynamic_crawler: DynamicCrawler = ctx["dynamic_crawler"]
    cleaner: ContentCleaner = ctx["cleaner"]
    dedup: ContentDeduplicator = ctx["dedup"]

    # 1. Robots.txt validation
    is_allowed = await robots.can_fetch(url)
    if not is_allowed:
        logger.warning(f"Access denied by robots.txt: {url}")
        return {"url": url, "status": "blocked_by_robots"}

    delay = robots.get_crawl_delay(url)
    if delay:
        await asyncio.sleep(delay)

    # 2. Ingestion (Static-First with Dynamic Fallback)
    if force_dynamic:
        res = await dynamic_crawler.fetch(url)
    else:
        res = await static_crawler.fetch(url)
        # If content is empty / placeholder, fallback to dynamic
        if res.status_code == 200 and res.content_length < 1500:
            logger.info(f"Content too short, fallback to Playwright: {url}")
            res = await dynamic_crawler.fetch(url)

    if res.status_code != 200 or not res.html:
        return {"url": url, "status": "fetch_failed", "error": res.error}

    # 3. Content Cleansing & Structured Markdown Extraction
    cleaned_doc = cleaner.clean(res.html, url=url)
    if not cleaned_doc:
        return {"url": url, "status": "extraction_empty"}

    # 4. Near-Deduplication (MinHash LSH)
    is_unique = dedup.insert(doc_id=url, text=cleaned_doc.content_markdown)
    if not is_unique:
        logger.info(f"Duplicate detected, skipping: {url}")
        return {"url": url, "status": "duplicate_skipped"}

    logger.success(f"Successfully crawled & cleaned: [{cleaned_doc.word_count} words] - {url}")

    return {
        "url": url,
        "status": "success",
        "title": cleaned_doc.title,
        "word_count": cleaned_doc.word_count,
        "markdown_snippet": cleaned_doc.content_markdown[:200],
    }


class WorkerSettings:
    """ARQ Worker configuration."""
    functions = [crawl_url_task]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings(host="localhost", port=6379)
    max_jobs = 10  # Number of concurrent jobs per worker process