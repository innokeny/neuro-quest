from collections import deque
from src.schemas.action import Action


class ShortTermMemory:
    def __init__(self, size: int):
        self.memory: deque[Action] = deque([], maxlen=size)
    
    def add(self, action: Action):
        self.memory.append(action)
    
    def get(self) -> list[Action]:
        return list(self.memory)