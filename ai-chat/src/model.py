from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class ChatModel:
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the chat model with a pre-trained transformer model."""
        self.model_name = model_name or os.getenv("MODEL_NAME", "facebook/opt-350m")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

    def generate_response(self, prompt: str, max_length: int = 100) -> str:
        """Generate a response for the given prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and return the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()

    def chat(self, messages: List[dict]) -> str:
        """Handle a conversation with multiple messages."""
        # Format the conversation history
        conversation = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            conversation += f"{role}: {content}\nassistant: "
        
        return self.generate_response(conversation) 