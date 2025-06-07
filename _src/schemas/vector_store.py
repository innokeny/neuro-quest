from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class VectorEntry(BaseModel):
    """Schema for entries in the vector database"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "text": "The party enters a dark dungeon",
                "vector": [0.1, 0.2, 0.3],
                "timestamp": "2024-03-23T12:00:00",
                "metadata": {"type": "game_event"}
            }
        }
    )
    
    text: str = Field(..., description="The text content to be vectorized")
    vector: Optional[list[float]] = Field(None, description="Vector representation of the text")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this entry was created")
    metadata: dict = Field(default_factory=dict, description="Additional metadata about the entry")
    
    def model_dump_json(self, **kwargs) -> str:
        """Custom JSON serialization for datetime"""
        kwargs.setdefault('exclude_none', True)
        data = self.model_dump(**kwargs)
        if 'timestamp' in data:
            data['timestamp'] = data['timestamp'].isoformat()
        return data
        
    @classmethod
    def from_dict(cls, data: dict) -> 'VectorEntry':
        """Create an instance from a dictionary, handling datetime conversion"""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data) 