import asyncio
from typing import Any, Dict, List
from src.crawler.engine.static import StaticCrawler
from src.crawler.engine.dynamic import DynamicCrawler
from src.crawler.parsers.cleaner import ContentCleaner
from src.crawler.parsers.deduplicator import ContentDeduplicator
from src.core.logger import logger

class CrawlerNode:
    """LangGraph node to crawl web pages from URLs, cleanse HTML to Markdown, and deduplicate content."""
    
    def __init__(self, min_output_length: int = 100, dedup_threshold: float = 0.85):
        self.min_output_length = min_output_length
        self.dedup_threshold = dedup_threshold

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        urls: List[str] = state.get("urls", [])
        if not urls:
            logger.warning("Node 'CrawlerNode' menerima list URL kosong.")
            return {"documents": []}

        logger.info(f"Memulai crawl batch untuk {len(urls)} URL...")
        
        cleaner = ContentCleaner(min_output_length=self.min_output_length)
        dedup = ContentDeduplicator(threshold=self.dedup_threshold)
        static_crawler = StaticCrawler()
        dynamic_crawler = DynamicCrawler(headless=True)
        
        await dynamic_crawler.start()
        
        documents: List[Dict[str, Any]] = []
        
        try:
            for url in urls:
                logger.info(f"Crawling URL: {url}")
                try:
                    # 1. Coba Static Crawler dulu
                    res = await static_crawler.fetch(url)
                    
                    # Fallback ke Playwright jika halaman dinamis/kosong
                    if res.status_code == 200 and res.content_length < 1500:
                        logger.info(f"Konten static sedikit ({res.content_length} karakter), fallback ke Playwright: {url}")
                        res = await dynamic_crawler.fetch(url)
                    elif res.status_code != 200:
                        logger.info(f"Fetch static gagal ({res.status_code}), fallback ke Playwright: {url}")
                        res = await dynamic_crawler.fetch(url)
                        
                    if res.status_code != 200 or not res.html:
                        logger.warning(f"Gagal crawl {url}: status {res.status_code}, error: {res.error}")
                        continue
                        
                    # 2. Pembersihan & ekstraksi Markdown
                    cleaned_doc = cleaner.clean(res.html, url=url)
                    if not cleaned_doc or not cleaned_doc.content_markdown:
                        logger.warning(f"Konten kosong setelah pembersihan: {url}")
                        continue
                        
                    # 3. Deduplikasi Near-Duplicates
                    is_unique = dedup.insert(doc_id=url, text=cleaned_doc.content_markdown)
                    if not is_unique:
                        logger.info(f"Dokumen duplikat diabaikan: {url}")
                        continue
                        
                    documents.append({
                        "id": url,
                        "url": url,
                        "title": cleaned_doc.title or "No Title",
                        "text": cleaned_doc.content_markdown,
                        "word_count": cleaned_doc.word_count
                    })
                    
                except Exception as e:
                    logger.error(f"Error memproses URL {url}: {e}")
        finally:
            await dynamic_crawler.close()
            
        logger.success(f"Berhasil meng-crawl & memproses {len(documents)} dari {len(urls)} URL")
        return {"documents": documents}
