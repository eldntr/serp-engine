from fastapi import APIRouter, HTTPException
from arq import create_pool
from arq.connections import RedisSettings
from src.serving.schemas.crawl import CrawlTriggerRequest, CrawlJobResponse
from src.core.config import settings
from src.core.logger import logger

router = APIRouter()


@router.post("/trigger", response_model=CrawlJobResponse)
async def trigger_crawl(payload: CrawlTriggerRequest):
    try:
        redis = await create_pool(RedisSettings(host=settings.crawler.redis_host, port=settings.crawler.redis_port))
        job = await redis.enqueue_job(
            "crawl_url_task", payload.url, force_dynamic=payload.force_dynamic
        )
        await redis.close()
        logger.info(f"Enqueued crawl task {job.job_id} for {payload.url}")
        return CrawlJobResponse(job_id=job.job_id, url=payload.url, status="queued")
    except Exception as e:
        logger.error(f"Failed to enqueue task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gagal memasukkan task ke queue: {str(e)}")
