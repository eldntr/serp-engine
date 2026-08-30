from typing import Any, Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.search.encoders.dense import DenseEncoder
from src.search.encoders.sparse import BM25SparseEncoder
from src.search.indexing.base import BaseIndexer
from src.search.indexing.qdrant_client import QdrantHybridIndexer
from src.core.config import settings
from src.core.logger import logger

class IndexerNode:
    """LangGraph node to chunk, encode, and index crawled documents into search index using LangChain."""
    
    def __init__(self, indexer: BaseIndexer = None, chunk_size: int = 1000, chunk_overlap: int = 200):
        self._injected_indexer = indexer
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        documents: List[Dict[str, Any]] = state.get("documents", [])
        if not documents:
            logger.warning("Node 'IndexerNode' received an empty document list.")
            return {"indexed_count": 0}

        indexer = self._injected_indexer
        if indexer is None:
            logger.warning("IndexerNode using in-memory Qdrant fallback.")
            dense_enc = DenseEncoder()
            sparse_enc = BM25SparseEncoder()
            indexer = QdrantHybridIndexer(host=":memory:")
            await indexer.init_collection()
        else:
            dense_enc = DenseEncoder()
            sparse_enc = BM25SparseEncoder()

        logger.info(f"Starting to process and index {len(documents)} documents to Qdrant...")
        total_chunks_indexed = 0
        
        for doc in documents:
            try:
                doc_id = doc.get("id") or doc.get("url")
                text = doc.get("text", "")
                
                chunks = self.text_splitter.split_text(text)
                logger.info(f"Splitting document '{doc.get('title')}' ({doc.get('url')}) into {len(chunks)} chunks.")
                
                for idx, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{idx}"
                    
                    chunk_payload = {
                        "id": doc_id,
                        "url": doc.get("url"),
                        "title": doc.get("title", "No Title"),
                        "text": chunk,
                        "chunk_index": idx,
                        "metadata": doc.get("metadata", {}),
                    }
                    
                    d_vec = dense_enc.encode(chunk)[0]
                    s_vec = sparse_enc.encode(chunk)
                    
                    await indexer.upsert_document(chunk_id, d_vec, s_vec, chunk_payload)
                    total_chunks_indexed += 1
                    
            except Exception as e:
                logger.error(f"Failed to index document {doc.get('url', '?')}: {e}")

        logger.success(f"Successfully indexed {total_chunks_indexed} chunks from {len(documents)} documents")
        return {"indexed_count": total_chunks_indexed}
