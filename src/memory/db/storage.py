import faiss
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field
from datetime import datetime
import json
from src.ml.inference.embedding import EmbeddingInference
from loguru import logger


class DbItem(BaseModel):
    text: str
    vector: np.ndarray
    meta: dict = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True

class VectorDb:
    def __init__(self, ember: EmbeddingInference, directory: Path):
        self._ember = ember
        self._directory = directory
        self._directory.mkdir(exist_ok=True)

        self._index = faiss.IndexFlatL2(self.dimension)
        self._items: list[DbItem] = []
    
    def add(self, text: str, meta: dict = {}):
        vector = self._ember.extract(text)
        item = DbItem(text=text, vector=vector, meta=meta)
        self._items.append(item)
        self._index.add(vector) # type: ignore

    def search(self, query: str, k: int = 5) -> list[DbItem]:
        vector = self._ember.extract(query)
        distances, indices = self._index.search(
            vector.reshape(1, -1).astype(np.float32),
            k
        ) # type: ignore
        if not self._items:
            return []
        return [self._items[idx] for idx in indices[0]]

    @property
    def dimension(self):
        return self._ember.dimension
    
    def _load(self):
        index_pth = self._directory / "index.faiss"
        items_pth = self._directory / "items.json"

        if index_pth.exists() and items_pth.exists():
            try:
                self._index = faiss.read_index(str(index_pth))
                with open(items_pth, "r", encoding="utf-8") as f:
                    self._items = [DbItem.model_validate(item) for item in json.load(f)]
            except Exception as e:
                logger.error(f"Failed to load vector db: {e}")
    
    def save(self):
        try:
            faiss.write_index(self._index, str(self._directory / "index.faiss"))
            with open(self._directory / "items.json", "w", encoding="utf-8") as f:
                json.dump([item.model_dump() for item in self._items], f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save vector db: {e}")



