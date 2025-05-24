from typing import Literal
from pydantic import BaseModel, Field


class Action(BaseModel):
    role: Literal['user', 'assistant'] = Field(..., description='author of text')
    content: str = Field(..., description='text representation of action')

    @property
    def representation(self) -> str:
        return f'{self.role}: {self.content}'