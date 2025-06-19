from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingInference:
    def __init__(self, path: Path):
        self._path = path
        self._model = SentenceTransformer(self._path.as_posix())
    
    def extract(self, text: str) -> np.ndarray:
        return self._model.encode([text]).reshape(1, -1)
    
    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension() # type: ignore

    @property
    def meta(self):
        return {'path': self._path}