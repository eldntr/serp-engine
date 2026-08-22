import asyncio
from arq.connections import RedisSettings
from loguru import logger
from src.crawler.engine.static import StaticCrawler
from src.crawler.engine.dynamic import DynamicCrawler
from src.crawler.parsers.cleaner import ContentCleaner
from src.crawler.parsers.deduplicator import ContentDeduplicator
from src.crawler.robots import RobotsManager


async def startup(ctx: dict):
    """Inisialisasi engine crawler, parser, dan deduplicator saat worker menyala."""
    logger.info("Memulai Background Crawler Worker Pool...")
    ctx["static_crawler"] = StaticCrawler()
    ctx["dynamic_crawler"] = DynamicCrawler(headless=True)
    await ctx["dynamic_crawler"].start()

    ctx["cleaner"] = ContentCleaner(min_output_length=100)
    ctx["dedup"] = ContentDeduplicator(threshold=0.85)
    ctx["robots"] = RobotsManager(user_agent="MyCustomSERPBot")


async def shutdown(ctx: dict):
    """Membersihkan instance browser saat worker dimatikan."""
    logger.info("Menutup Crawler Worker Pool...")
    if "dynamic_crawler" in ctx:
        await ctx["dynamic_crawler"].close()


async def crawl_url_task(ctx: dict, url: str, force_dynamic: bool = False) -> dict:
    """Tugas antrean tunggal untuk memproses crawling 1 URL."""
    robots: RobotsManager = ctx["robots"]
    static_crawler: StaticCrawler = ctx["static_crawler"]
    dynamic_crawler: DynamicCrawler = ctx["dynamic_crawler"]
    cleaner: ContentCleaner = ctx["cleaner"]
    dedup: ContentDeduplicator = ctx["dedup"]

    # 1. Validasi Robots.txt
    is_allowed = await robots.can_fetch(url)
    if not is_allowed:
        logger.warning(f"Akses ditolak oleh robots.txt: {url}")
        return {"url": url, "status": "blocked_by_robots"}

    delay = robots.get_crawl_delay(url)
    if delay:
        await asyncio.sleep(delay)

    # 2. Ingestion (Static-First dengan Dynamic Fallback)
    if force_dynamic:
        res = await dynamic_crawler.fetch(url)
    else:
        res = await static_crawler.fetch(url)
        # Jika hasil kosong / placeholder, fallback ke dynamic
        if res.status_code == 200 and res.content_length < 1500:
            logger.info(f"Konten terlalu sedikit, fallback ke Playwright: {url}")
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
        logger.info(f"Duplikat terdeteksi, lewati: {url}")
        return {"url": url, "status": "duplicate_skipped"}

    logger.success(f"Berhasil di-crawl & dibersihkan: [{cleaned_doc.word_count} kata] - {url}")

    return {
        "url": url,
        "status": "success",
        "title": cleaned_doc.title,
        "word_count": cleaned_doc.word_count,
        "markdown_snippet": cleaned_doc.content_markdown[:200],
    }


class WorkerSettings:
    """Konfigurasi Worker ARQ."""
    functions = [crawl_url_task]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings(host="localhost", port=6379)
    max_jobs = 10  # Jumlah job concurrent per worker process