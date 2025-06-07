from pathlib import Path

from src.memory.db.storage import VectorDb
from src.memory.short.memory import ShortTermMemory
from src.ml.inference.embedding import EmbeddingInference
from src.ml.inference.master import MasterInference, Message, MasterConfig, MasterResponse
from src.ml.inference.ner import NerInference

from pydantic import BaseModel, Field

from loguru import logger

class EngineConfig(BaseModel):
    short_memory_size: int = Field(default=5)
    vector_db_path: Path
    number_of_remind_items: int = Field(default=5)
    master_config: MasterConfig
    ner_model_path: Path
    embedding_model_path: Path

class Engine:
    def __init__(self, config: EngineConfig, debug: bool = False):
        if debug != True:
            logger.level("INFO")
                
        self.config = config
        
        self.master = MasterInference(config.master_config)
        logger.debug('Loaded Master Model')
        
        self.ner = NerInference(config.ner_model_path)
        logger.debug(f'Loaded Ner Model: {self.ner.meta}')
        
        self.short_memory = ShortTermMemory(config.short_memory_size)
        
        self.db = VectorDb(
            EmbeddingInference(config.embedding_model_path),
            config.vector_db_path
        )
        logger.debug('Loaded Vector DB')
        
        logger.debug("Engine initialized")

    def remind(self, text: str) -> list[str]:
        """Get texts from long term memory

        :param str text: query text 
        :return list[str]: list of correlated texts from long term memory
        """
        items = self.db.search(text, self.config.number_of_remind_items)
        logger.debug(f"Remind Items: {[item.text for item in items]}")
        return [item.text for item in items]

    def memorize(self, text: str):
        """Add text to long term memory

        :param str text: text to add
        """

        for sentence in text.split("."):
            if sentence.strip() == "":
                continue
            entities = self.ner.extract(sentence)
            logger.debug(f"Entities: {entities}")
            for e in entities:
                self.db.add(sentence, {"type": e.type, "text": e.text})

    def dialog(self, statement: str) -> MasterResponse:
        """Main dialog method 

        :param Step step: dialog step
        :return str: master response
        """

        context: set[str] = set()

        context.update(item.text for item in self.short_memory.get())

        entities = self.ner.extract(statement)
        logger.debug(f"Entities: {entities}")
        for e in entities:
            context.update(self.remind(e.text))
        logger.debug(f"Context: {context}")
        response = self.master.generate(Message(
            context="\n".join(context),
            statement=statement
        ))

        self.short_memory.add(response.text)

        self.memorize(response.text)

        return response

        





