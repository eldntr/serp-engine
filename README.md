### Install Playwright Chromium

```bash
playwright install chromium
```
Cara Menjalankan Pengujian
Nyalakan Redis Server (via Docker atau lokal):

Bash
docker run -d -p 6379:6379 --name redis-serp redis:7-alpine
Jalankan Background Worker ARQ di terminal pertama:

Bash
arq src.crawler.queue.redis_tasks.WorkerSettings
Jalankan Pengujian Enqueue Job di terminal kedua:

Bash
python -m tests.test_queue


fix the problem later:
(serp-crawler-engine) eldin@Flow:~/workspace/serp-engine$ python -m tests.test_crawler
=== 0. TESTING ROBOTS.TXT ===
[ROBOTS] https://en.wikipedia.org/wiki/Information_retrieval
  -> Allowed: True | Crawl Delay: None
[ROBOTS] https://quotes.toscrape.com/js/
  -> Allowed: True | Crawl Delay: None
[ROBOTS] https://www.mims.com/indonesia
  -> Allowed: True | Crawl Delay: None

=== 1. TESTING STATIC CRAWLER (HTTPX) ===
[STATIC] https://en.wikipedia.org/wiki/Information_retrieval
  -> Status: 200 | Size: 326715 chars | Time: 0.210s
[STATIC] https://quotes.toscrape.com/js/
  -> Status: 200 | Size: 5806 chars | Time: 1.026s
[STATIC] https://www.mims.com/indonesia
  -> Status: 403 | Size: 5750 chars | Time: 0.944s

=== 2. TESTING DYNAMIC CRAWLER (PLAYWRIGHT) ===
[DYNAMIC] https://en.wikipedia.org/wiki/Information_retrieval
  -> Status: 200 | Size: 601831 chars | Time: 1.719s
[DYNAMIC] https://quotes.toscrape.com/js/
  -> Status: 200 | Size: 8940 chars | Time: 2.195s
[DYNAMIC] https://www.mims.com/indonesia
  -> Status: 0 | Size: 0 chars | Time: 5.055s
  -> Error: Page.goto: net::ERR_NETWORK_CHANGED at https://www.mims.com/indonesia
Call log:
  - navigating to "https://www.mims.com/indonesia", waiting until "networkidle"


to do:
devops with redis or kube or whatever
build fe or web based with llm