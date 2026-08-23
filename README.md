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
python -m tests.test_discovery


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

serp-engine/
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── configs/
│   ├── base.yaml               # Konfigurasi umum (rate limit, concurrency)
│   ├── crawler.yaml            # Header pool, proxy, timeouts
│   └── retrieval.yaml          # Model weights, top-k, RRF constant (k=60)
│
├── data/                       # Local volume / fixtures
│   ├── raw/
│   └── benchmarks/             # Dataset evaluasi (MS MARCO subset / Custom Q&A)
│
├── docker/
│   ├── Dockerfile.crawler
│   ├── Dockerfile.api
│   └── docker-compose.yml      # Orkestrasi Redis, Qdrant, PostgreSQL, & Workers
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/                   # Shared kernel & settings
│   │   ├── config.py           # Pydantic BaseSettings
│   │   ├── exceptions.py
│   │   └── logger.py
│   │
│   ├── crawler/                # Modul 1: Distributed Ingestion
│   │   ├── engine/
│   │   │   ├── dynamic.py      # Playwright async browser pool
│   │   │   └── static.py       # Async HTTPX worker
│   │   ├── parsers/
│   │   │   ├── cleaner.py      # Trafilatura / Readability pipeline
│   │   │   └── deduplicator.py # MinHash LSH / SimHash logic
│   │   ├── queue/
│   │   │   └── redis_tasks.py  # Celery / ARQ task definitions
│   │   └── robots.py           # Politeness & robots.txt parser
│   │
│   ├── search/                 # Modul 2: Information Retrieval & AI
│   │   ├── encoders/
│   │   │   ├── dense.py        # BAAI/bge-m3 dense embedder
│   │   │   └── sparse.py       # BM25 / SPLADE / Sparse token weights
│   │   ├── indexing/
│   │   │   ├── qdrant_client.py# Vector DB connection & payload schemas
│   │   │   └── pg_store.py     # Metadata relational repository
│   │   ├── rankers/
│   │   │   ├── rrf.py          # Reciprocal Rank Fusion implementation
│   │   │   └── cross_encoder.py# BAAI/bge-reranker-large scoring
│   │   └── query/
│   │       ├── expansion.py    # HyDE & synonym rewriting
│   │       └── snippets.py     # Semantic sliding-window extractor
│   │
│   ├── serving/                # Modul 3: Backend API & Agents
│   │   ├── api/
│   │   │   ├── v1/
│   │   │   │   ├── endpoints/
│   │   │   │   │   ├── crawl.py# Trigger & webhook endpoints
│   │   │   │   │   └── search.py# SERP search REST API
│   │   │   │   └── router.py
│   │   ├── schemas/
│   │   │   ├── crawl.py        # Pydantic crawl request/response
│   │   │   └── serp.py         # SERP JSON schema & payload contract
│   │   ├── synthesizer/
│   │   │   └── rag_agent.py    # Grounded generative summary + citations
│   │   └── main.py             # FastAPI entrypoint
│   │
│   └── evaluation/             # Modul 4: IR Metrics & Benchmark
│       ├── metrics.py          # MRR@k, NDCG@k, MAP implementation
│       └── run_benchmark.py    # Evaluasi akurasi retrieval & profiling latency
│
├── tests/
│   ├── test_crawler.py
│   ├── test_retrieval.py
│   └── test_api.py
│
├── .env.example
├── .gitignore
├── pyproject.toml              # Dependency management (Poetry / UV)
└── README.md                   # Dokumentasi arsitektur & cara deploy