from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel, Field

class Message(BaseModel):
    context: str
    statement: str


class MasterResponse(BaseModel):
    text: str
    meta: dict = Field(default_factory=dict)


class SystemPrompt(BaseModel):
    preambular: str

    def make(self, message: Message)-> str:
        return f"""
You are Dungeon Master and you are talking with a player. Respond to the player's actions corresponding to the context.
Generate a answer of 2-3 sentences to the user's action to continue the story. End this answer with token [END]
Story Hook: {self.preambular}
Context: {message.context}
User Action: {message.statement}
Answer:
""" 
    @property
    def separator(self):
        return "Answer:"


class GenerationConfig(BaseModel):
    temperature: float = Field(default=1.0)
    top_k: int = Field(default=50)
    top_p: float = Field(default=1.0)
    repetition_penalty: float = Field(default=1.0)
    max_new_tokens: int = Field(default=2000)
    stop_strings : str = Field(default="[END]")

class MasterConfig(BaseModel):
    path: Path
    preambular: str
    generation_config: GenerationConfig

    @property
    def prompt(self):
        return SystemPrompt(preambular=self.preambular)

class MasterInference:
    def __init__(self, config: MasterConfig):
        self._path = config.path
        self._prompt = config.prompt
        self._generation_config = config.generation_config
        self._tokenizer = AutoTokenizer.from_pretrained(str(self._path))
        self._model = AutoModelForCausalLM.from_pretrained(str(self._path))
    

    def _response(self, text: str) -> str:
        return text.split(self._prompt.separator)[-1].replace("[END]", "").replace('[end]', "").strip()

    def _generate(self, text: str) -> str:
        """Inner function for generate model output

        :param str text: system prompt
        :param GenerationConfig config: generation config (temperature and etc.)
        :return str:  system prompt with generated text
        """
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(
            **inputs,
            **self._generation_config.model_dump(),
            tokenizer=self._tokenizer
        )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate(self, message: Message) -> MasterResponse:
        prompt = self._prompt.make(message)
        outputs = self._generate(prompt)
        return MasterResponse(text=self._response(outputs))
