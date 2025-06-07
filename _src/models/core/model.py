from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import PeftModel




class DungeonMaster:
    def __init__(self, path: Path, name: str, device: str = "cuda"):
        self.path = path
        self.device = device
        self.name = name
        self._load()
    
    def _load(self):
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.name,
            trust_remote_code=True,
            padding_side="right",
            pad_token="[EOT]" # end of text
        )
        self._tokenizer.add_special_tokens({
            "pad_token": "[EXTRA_0]",
            "eos_token": "[EOT]",
            "bos_token": "[EOT]"
        })

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.name,
            quantization_config=bnb_config,
            device_map={'': torch.cuda.current_device()},
            trust_remote_code=True
        )

        model.resize_token_embeddings(len(self._tokenizer))

        self._model = PeftModel.from_pretrained(model, self.path)
    
    def generate_action(self, context: str, action: str) -> str:
        prompt = f"""
        [im_start]system
        You are a creative Dungeon Master. React to the actions of the players according to the context.
        [im_end]
        [im_start]user
        [CONTEXT] {context}
        [ACTION] {action}
        [im_end]
        [im_start]assistant
        """

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=self._tokenizer.eos_token_id
        )

        full_response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_response.split("[im_start]assistant")[-1].strip()



