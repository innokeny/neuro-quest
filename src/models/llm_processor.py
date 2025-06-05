"""
LLM processor for generating D&D responses using a fine-tuned Qwen model.
"""

from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class LLMProcessor:
    def __init__(self, model_path: Path):
        """
        Initialize the LLM processor with a fine-tuned Qwen model.
        
        Args:
            model_path: Path to the model directory containing adapter files
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Configure quantization for efficient inference
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # Load base model and adapter
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-7B",  # Base model
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        
        # Load adapter weights
        self.model.load_adapter(str(model_path))
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True
        )
        
        # Set default generation parameters
        self.generation_config = {
            "max_new_tokens": 150,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True
        }
    
    def format_context(self, context: str, action: str) -> str:
        """
        Format the context and action into a prompt for the LLM.
        
        Args:
            context: Previous context or state
            action: Current action to process
            
        Returns:
            Formatted prompt string
        """
        return f"""<|im_start|>system
You are a D&D game master. Respond to player actions in character, considering the context and maintaining game consistency.
<|im_end|>
<|im_start|>user
Context: {context}
Action: {action}
<|im_end|>
<|im_start|>assistant
"""
    
    def generate_response(self, context: str, action: str, generation_config: dict = None) -> str:
        """
        Generate a response based on the context and action.
        
        Args:
            context: Previous context or state
            action: Current action to process
            generation_config: Optional generation parameters
            
        Returns:
            Generated response
        """
        # Format the prompt
        prompt = self.format_context(context, action)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        config = generation_config or self.generation_config
        outputs = self.model.generate(
            **inputs,
            **config
        )
        
        # Decode and clean up response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        return response 