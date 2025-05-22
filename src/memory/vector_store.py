import faiss
import numpy as np
import json
import os
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from ..schemas.vector_store import VectorEntry


class VectorStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 storage_path: str = "data/vector_store"):
        """Initialize vector store with a sentence transformer model"""
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.entries: List[VectorEntry] = []
        self.storage_path = storage_path
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        
        # Load existing data if available
        self._load()
        
    def _save(self):
        """Save the vector store to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, os.path.join(self.storage_path, "index.faiss"))
            
            # Save entries using custom JSON serialization
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
                # Load FAISS index
                self.index = faiss.read_index(index_path)
                
                # Load entries
                with open(entries_path, "r", encoding="utf-8") as f:
                    entries_data = json.load(f)
                    self.entries = [VectorEntry.from_dict(entry) for entry in entries_data]
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error loading vector store: {e}")
                # If there's an error, start with empty store
                self.index = faiss.IndexFlatL2(self.dimension)
                self.entries = []
        
    def add_entry(self, text: str, metadata: Optional[dict] = None) -> None:
        """Add a new entry to the vector store"""
        # Create vector representation
        vector = self.model.encode([text])[0]
        
        # Create entry
        entry = VectorEntry(
            text=text,
            vector=vector.tolist(),
            metadata=metadata or {}
        )
        
        # Add to FAISS index
        self.index.add(np.array([vector], dtype=np.float32))
        self.entries.append(entry)
        
        # Save changes
        self._save()
        
    def search(self, query: str, k: int = 5) -> List[VectorEntry]:
        """Search for similar entries"""
        if not self.entries:
            return []
            
        # Get query vector
        query_vector = self.model.encode([query])[0]
        
        # Search in FAISS
        distances, indices = self.index.search(
            np.array([query_vector], dtype=np.float32), 
            k
        )
        
        # Return matching entries
        return [self.entries[idx] for idx in indices[0] if idx < len(self.entries)]
    
    def get_context(self, query: str, k: int = 5) -> str:
        """Get relevant context for a query"""
        similar_entries = self.search(query, k)
        if not similar_entries:
            return ""
            
        # Combine relevant entries into context
        context = "\n".join(entry.text for entry in similar_entries)
        return context 