import asyncio
from arq import create_pool
from arq.connections import RedisSettings


async def main():
    redis_settings = RedisSettings(host="localhost", port=6379)
    redis = await create_pool(redis_settings)

    test_urls = [
        "https://en.wikipedia.org/wiki/Information_retrieval",
        "https://quotes.toscrape.com/js/",
        "https://en.wikipedia.org/wiki/Special:Search",  # Biasanya diblokir di robots.txt
    ]

    print("=== ENQUEUE CRAWL TASKS KE REDIS ===")
    job_ids = []
    for url in test_urls:
        job = await redis.enqueue_job("crawl_url_task", url)
        print(f"Task dimasukkan: {url} -> Job ID: {job.job_id}")
        job_ids.append(job)

    print("\nMenunggu worker memproses tugas...")
    for job in job_ids:
        result = await job.result(timeout=40, poll_delay=0.5)
        print(f"\n[HASIL JOB {job.job_id}]:")
        print(result)

    await redis.close()


if __name__ == "__main__":
    asyncio.run(main())