from pathlib import Path
from typing import Optional
from src.schemas.action import Action

class MockModel:
    
    @staticmethod
    def prompt(action: Action, short: Optional[list[Action]] = None, long: Optional[list[Action]] = None) -> str:
        sort_repr = '' if short is None else '\n'.join(a.representation for a in short)
        long_repr = '' if long is None else '\n'.join(a.representation for a in long)
        text = f'{action.representation}\n{sort_repr}\n{long_repr}'
        return text
    
    def ask(self, text: str) -> str:
        return 'хозяин ударил по голове'

    def process(self, action: Action, short: Optional[list[Action]] = None, long: Optional[list[Action]] = None) -> Action:
        text = self.prompt(action, short, long)
        return Action(role='assistant', content=self.ask(text))

