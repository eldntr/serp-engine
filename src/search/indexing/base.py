from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseIndexer(ABC):
    """Abstract Base Class defining the interface for indexing and search storage engines."""

    @abstractmethod
    async def init_collection(self) -> None:
        """Initialize the storage collection, index, or directory."""
        pass

    @abstractmethod
    async def upsert_document(
        self,
        doc_id: str,
        dense_vector: List[float],
        sparse_vector: Dict[str, float],
        payload: Dict[str, Any],
    ) -> None:
        """Upsert a document chunk along with its dense/sparse vector representations and payload."""
        pass

    @abstractmethod
    async def search_dense(self, dense_vector: List[float], limit: int = 50) -> List[Any]:
        """Perform a dense vector similarity search."""
        pass

    @abstractmethod
    async def search_sparse(self, sparse_vector: Dict[str, float], limit: int = 50) -> List[Any]:
        """Perform a sparse vector search."""
        pass
