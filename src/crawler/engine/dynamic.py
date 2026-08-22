from playwright.async_api import async_playwright, Browser, BrowserContext
from fake_useragent import UserAgent
from typing import Optional
from src.crawler.engine.schemas import CrawlResult
from src.core.config import settings
from src.core.logger import logger
from src.core.exceptions import CrawlerError

ua = UserAgent()


class DynamicCrawler:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self._playwright = None
        self._browser: Optional[Browser] = None

    async def start(self):
        if not self._browser:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                ],
            )

    async def close(self):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def fetch(
        self,
        url: str,
        wait_until: str = None,
        timeout: float = None,
    ) -> CrawlResult:
        if not self._browser:
            await self.start()

        wait_until = wait_until or settings.crawler.wait_until
        timeout = timeout or settings.crawler.timeout

        context: BrowserContext = await self._browser.new_context(
            user_agent=ua.random,
            viewport={"width": 1280, "height": 800},
            ignore_https_errors=True,
        )
        page = await context.new_page()

        try:
            await page.route(
                "**/*",
                lambda route: route.abort()
                if route.request.resource_type in ["image", "media", "font"]
                else route.continue_(),
            )

            response = await page.goto(url, wait_until=wait_until, timeout=timeout)
            html = await page.content()
            status_code = response.status if response else 200

            return CrawlResult(
                url=page.url,
                status_code=status_code,
                html=html,
                content_length=len(html),
                engine="dynamic",
            )
        except Exception as e:
            logger.error(f"Dynamic crawler failed for {url}: {str(e)}")
            return CrawlResult(
                url=url,
                status_code=0,
                html="",
                content_length=0,
                engine="dynamic",
                error=str(e),
            )
        finally:
            await context.close()