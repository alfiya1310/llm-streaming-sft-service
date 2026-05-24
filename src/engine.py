import torch
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from peft import PeftModel

class InferenceEngine:
    def __init__(self, base_model_path: str, lora_weights_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        # Configure safe 4-bit loading parameters for the inference baseline
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load the frozen structural layers cleanly on the GPU
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Safe runtime layer fusion
        self.model = PeftModel.from_pretrained(base_model, lora_weights_path)
        self.model.eval()

    def generate_stream(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True
        )
        
        # Run generation loop on a detached system execution background thread
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for new_text in streamer:
            yield f"data: {new_text}\n\n"