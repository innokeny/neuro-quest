import faiss
import numpy as np
import json
import os
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from src.schemas.vector_store import VectorEntry

class LongTermMemory:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 storage_path: str = "data/vector_store"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.entries: List[VectorEntry] = []
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self._load()

    def _save(self):
        """Save the vector store to disk"""
        try:
            faiss.write_index(self.index, os.path.join(self.storage_path, "index.faiss"))
            entries_data = [entry.model_dump_json() for entry in self.entries]
            with open(os.path.join(self.storage_path, "entries.json"), "w", encoding="utf-8") as f:
                json.dump(entries_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving vector store: {e}")
            
    def _load(self):
        """Load the vector store from disk"""
        index_path = os.path.join(self.storage_path, "index.faiss")
        entries_path = os.path.join(self.storage_path, "entries.json")
        
        if os.path.exists(index_path) and os.path.exists(entries_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(entries_path, "r", encoding="utf-8") as f:
                    entries_data = json.load(f)
                    self.entries = [VectorEntry.from_dict(entry) for entry in entries_data]
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error loading vector store: {e}")
                self.index = faiss.IndexFlatL2(self.dimension)
                self.entries = []
        
    def add_entry(self, text: str, metadata: Optional[dict] = None) -> None:
        """Add a new entry to the vector store"""
        vector = self.model.encode([text])[0]
        
        entry = VectorEntry(
            text=text,
            vector=vector.tolist(),
            metadata=metadata or {}
        )
        
        self.index.add(np.array([vector], dtype=np.float32)) # type: ignore
        self.entries.append(entry)
        
        self._save()
        
    def search(self, query: str, k: int = 5) -> List[VectorEntry]:
        """Search for similar entries"""
        if not self.entries:
            return []
            
        query_vector = self.model.encode([query])[0]
        
        distances, indices = self.index.search(
            np.array([query_vector], dtype=np.float32), 
            k
        ) # type: ignore
        
        return [self.entries[idx] for idx in indices[0] if idx < len(self.entries)]
    
    def get_context(self, query: str, k: int = 5) -> str:
        """Get relevant context for a query"""
        similar_entries = self.search(query, k)
        if not similar_entries:
            return ""
            
        context = "\n".join(entry.text for entry in similar_entries)
        return context 