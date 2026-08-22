from typing import List, Union
import torch
from sentence_transformers import SentenceTransformer
from src.core.config import settings


class DenseEncoder:
    def __init__(self, model_name: str = None, device: str = None):
        model_name = model_name or settings.retrieval.dense_model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            batch_size=16,
            show_progress_bar=False,
        )
        return embeddings.tolist()
