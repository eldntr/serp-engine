from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.serving.api.v1.router import api_router
from src.search.encoders.dense import DenseEncoder
from src.search.encoders.sparse import BM25SparseEncoder
from src.search.indexing.qdrant_client import QdrantHybridIndexer
from src.search.rankers.rrf import ReciprocalRankFusion
from src.search.rankers.cross_encoder import NeuralReranker
from src.search.query.snippets import SnippetExtractor
from src.serving.synthesizer.rag_agent import GroundedRAGSynthesizer
from src.core.config import settings
from src.core.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.dense_encoder = DenseEncoder()
    app.state.sparse_encoder = BM25SparseEncoder()
    app.state.indexer = QdrantHybridIndexer(host=":memory:")
    await app.state.indexer.init_collection()

    app.state.rrf = ReciprocalRankFusion()
    app.state.reranker = NeuralReranker()
    app.state.snippet_extractor = SnippetExtractor()
    app.state.synthesizer = GroundedRAGSynthesizer()

    logger.info("Initializing search models and qdrant in-memory benchmark collection...")

    sample_docs = [
        {
            "id": "doc-1",
            "url": "https://example.com/vector-search",
            "title": "Memahami Vector Database dan Semantic Search",
            "text": "Vector database seperti Qdrant memungkinkan penyimpanan embedding dense berdimensi tinggi untuk mencari kesamaan semantik dokumen secara efisien pada skala besar.",
        },
        {
            "id": "doc-2",
            "url": "https://example.com/bm25-search",
            "title": "Algoritma BM25 untuk Lexical Exact Match",
            "text": "BM25 adalah algoritma pencarian leksikal berbasis term-frequency yang sangat akurat untuk mencocokkan kata kunci spesifik dan identifier unik.",
        },
        {
            "id": "doc-3",
            "url": "https://example.com/hybrid-search",
            "title": "Panduan Hybrid Search: Dense + Sparse",
            "text": "Hybrid Search menggabungkan keunggulan BM25 dan dense vectors menggunakan Reciprocal Rank Fusion untuk menghasilkan SERP berpresisi tinggi.",
        },
    ]
    for d in sample_docs:
        d_vec = app.state.dense_encoder.encode(d["text"])[0]
        s_vec = app.state.sparse_encoder.encode(d["text"])
        await app.state.indexer.upsert_document(d["id"], d_vec, s_vec, d)

    yield


app = FastAPI(
    title=settings.base.app_name,
    version="1.0.0",
    description="Autonomous Web Crawler & Two-Stage Hybrid Retrieval SERP API",
    lifespan=lifespan,
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/")
def health_check():
    return {"status": "ok", "service": "SERP Engine API"}
