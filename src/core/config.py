import yaml
from pathlib import Path
from pydantic_settings import BaseSettings

CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"

def load_yaml(file_name: str) -> dict:
    file_path = CONFIG_DIR / file_name
    if not file_path.exists():
        return {}
    with open(file_path, "r") as f:
        return yaml.safe_load(f) or {}

class BaseConfig(BaseSettings):
    app_name: str = "serp-crawler-engine"
    environment: str = "development"
    rate_limit: int = 100
    concurrency: int = 10

class CrawlerConfig(BaseSettings):
    timeout: int = 30000
    wait_until: str = "networkidle"
    redis_host: str = "localhost"
    redis_port: int = 6379
    user_agents: list = []
    searxng_base_url: str = "http://localhost:8080"

class RetrievalConfig(BaseSettings):
    dense_model_name: str = "BAAI/bge-m3"
    sparse_k1: float = 1.5
    sparse_b: float = 0.75
    sparse_avg_doc_len: float = 200.0
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "serp_documents"
    qdrant_dense_dim: int = 1024
    rrf_k: int = 60
    cross_encoder_model_name: str = "BAAI/bge-reranker-large"
    device: str = None

class LLMConfig(BaseSettings):
    llm_provider: str = "ollama"
    llm_api_base: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:32b-instruct-q4_K_M"
    llm_api_key: str = ""
    llm_timeout: float = 60.0

class Settings:
    def __init__(self):
        base_data = load_yaml("base.yaml")
        crawler_data = load_yaml("crawler.yaml")
        retrieval_data = load_yaml("retrieval.yaml")
        llm_data = load_yaml("llm.yaml")
        
        self.base = BaseConfig(**base_data)
        self.crawler = CrawlerConfig(**crawler_data)
        self.retrieval = RetrievalConfig(**retrieval_data)
        self.llm = LLMConfig(**llm_data)

settings = Settings()
