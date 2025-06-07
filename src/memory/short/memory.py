from collections import deque
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, PrivateAttr

class QItem(BaseModel):
    text: str
    _timestamp: str = PrivateAttr(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    
    @property
    def timestamp(self):
        return self._timestamp

class ShortTermMemory:
    def __init__(self, size: int = 10):
        self.memory: deque[QItem] = deque([], maxlen=size)
    
    def add(self, text: str):
        """Add text to short term memory

        :param str text: text to add
        """
        self.memory.append(QItem(text=text))
    
    def get(self, k: Optional[int] = None) -> list[QItem]:
        """get texts from short term memory

        :param Optional[int] k: number of texts to get. If None, return all texts, defaults to None
        :return list[QItem]: list of memory items
        """
        if k is None:
            return list(self.memory)
        return list(self.memory)[-k:]