import uuid
from typing import Any, Dict, List
from qdrant_client import AsyncQdrantClient, models
from src.core.config import settings
from src.core.logger import logger


class QdrantHybridIndexer:
    def __init__(
        self,
        collection_name: str = None,
        host: str = None,
        port: int = None,
        dense_dim: int = None,
    ):
        self.collection_name = collection_name or settings.retrieval.qdrant_collection_name
        self.dense_dim = dense_dim or settings.retrieval.qdrant_dense_dim
        host = host or settings.retrieval.qdrant_host
        port = port or settings.retrieval.qdrant_port
        if host == ":memory:":
            self.client = AsyncQdrantClient(location=":memory:")
        else:
            self.client = AsyncQdrantClient(host=host, port=port)

    async def init_collection(self):
        collections = await self.client.get_collections()
        exists = any(c.name == self.collection_name for c in collections.collections)

        if not exists:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.dense_dim,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )
            logger.info(f"Collection '{self.collection_name}' berhasil dibuat.")

    async def upsert_document(
        self,
        doc_id: str,
        dense_vector: List[float],
        sparse_vector: Dict[str, float],
        payload: Dict[str, Any],
    ):
        indices = [abs(hash(k)) % (2**31) for k in sparse_vector.keys()]
        values = list(sparse_vector.values())

        try:
            point_id = str(uuid.UUID(doc_id))
        except ValueError:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))

        point = models.PointStruct(
            id=point_id,
            vector={
                "dense": dense_vector,
                "sparse": models.SparseVector(indices=indices, values=values),
            },
            payload=payload,
        )

        await self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )

    async def search_dense(self, dense_vector: List[float], limit: int = 50) -> List[models.ScoredPoint]:
        res = await self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vector,
            using="dense",
            limit=limit,
        )
        return res.points

    async def search_sparse(self, sparse_vector: Dict[str, float], limit: int = 50) -> List[models.ScoredPoint]:
        indices = [abs(hash(k)) % (2**31) for k in sparse_vector.keys()]
        values = list(sparse_vector.values())
        res = await self.client.query_points(
            collection_name=self.collection_name,
            query=models.SparseVector(indices=indices, values=values),
            using="sparse",
            limit=limit,
        )
        return res.points
