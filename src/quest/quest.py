from src.memory.short import ShortTermMemory
from src.schemas.action import Action


class Quest:
    def __init__(self, sort_term_memory: ShortTermMemory, model):
        self.sort_term_memory = sort_term_memory
        self.model = model
    
    def get_action(self, text: str) -> Action:
        return Action(role='user', content=text)

    def start(self, text: str) -> Action:
        user_action = self.get_action(text)
        short_memory = self.sort_term_memory.get()
        
        response = self.model.process(user_action, short_memory)
        
        self.sort_term_memory.add(user_action)
        self.sort_term_memory.add(response)

        return response