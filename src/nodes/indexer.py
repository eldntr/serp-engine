from typing import Any, Dict, List
from src.search.encoders.dense import DenseEncoder
from src.search.encoders.sparse import BM25SparseEncoder
from src.search.indexing.qdrant_client import QdrantHybridIndexer
from src.core.config import settings
from src.core.logger import logger

class IndexerNode:
    """LangGraph node to encode and index crawled documents into Qdrant vector database."""
    
    def __init__(self, indexer: QdrantHybridIndexer = None):
        # Terima instance yang sudah ada (disuntik dari graph) agar koleksi tidak dibuat ulang
        self._injected_indexer = indexer

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        documents: List[Dict[str, Any]] = state.get("documents", [])
        if not documents:
            logger.warning("Node 'IndexerNode' menerima list dokumen kosong.")
            return {"indexed_count": 0}

        # Gunakan indexer yang disuntik jika tersedia (dari app.state), 
        # jika tidak buat instance baru (fallback untuk testing)
        indexer = self._injected_indexer
        if indexer is None:
            logger.warning("IndexerNode menggunakan in-memory Qdrant fallback.")
            dense_enc = DenseEncoder()
            sparse_enc = BM25SparseEncoder()
            indexer = QdrantHybridIndexer(host=":memory:")
            await indexer.init_collection()
        else:
            dense_enc = DenseEncoder()
            sparse_enc = BM25SparseEncoder()

        logger.info(f"Mulai mengindeks {len(documents)} dokumen ke Qdrant...")
        indexed = 0
        for doc in documents:
            try:
                text = doc.get("text", "")
                d_vec = dense_enc.encode(text)[0]
                s_vec = sparse_enc.encode(text)
                await indexer.upsert_document(doc["id"], d_vec, s_vec, doc)
                indexed += 1
            except Exception as e:
                logger.error(f"Gagal mengindeks dokumen {doc.get('url', '?')}: {e}")

        logger.success(f"Berhasil mengindeks {indexed}/{len(documents)} dokumen")
        return {"indexed_count": indexed}
