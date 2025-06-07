from pathlib import Path
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers.pipelines.base import Pipeline
from transformers.pipelines import pipeline
from pydantic import BaseModel
from enum import Enum


class NerEntityType(str, Enum):
    UNKNOWN = 'UNKNOWN'
    ORG = 'ORG'
    SPELL = 'SPELL'
    ITEM = 'ITEM'
    ACTION = 'ACTION'
    STATUS = 'STATUS'
    LOC = 'LOC'
    PER = 'PER'
    MON = 'MON'


class NerEntity(BaseModel):
    text: str
    type: NerEntityType


class NerInference:
    def __init__(self, path: Path):
        self._path = path
        self._ner = self._load()
    
    def extract(self, text: str) -> list[NerEntity]:
        output: list[dict] = self._ner(text) # type: ignore
        return [
            NerEntity(
                text=e.get('word', ''),
                type=e.get('entity_group', NerEntityType.UNKNOWN)
            ) for e in output
        ]

    def _load(self) -> Pipeline:
        tokenizer = AutoTokenizer.from_pretrained(self._path)
        model = AutoModelForTokenClassification.from_pretrained(self._path)
        return pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="first")
    
    @property
    def meta(self):
        return {'path': self._path}
