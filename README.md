# LLM Streaming SFT Service

A modular, production-grade Machine Learning Engineering pipeline designed to perform Supervised Fine-Tuning (SFT) on open-weight instruction models (Phi-3) and serve them via a low-latency, asynchronous FastAPI streaming API under strict hardware memory constraints.

## Key Architectural Highlights

* **Hardware Optimization (QLoRA):** Compressed model weights to 4-bit NormalFloat (NF4) precision using `bitsandbytes`. This technique successfully dropped the minimum VRAM footprint from 16 GB to **5.5 GB**, allowing training to execute seamlessly on low-cost/free consumer-grade hardware (Nvidia T4).
* **Targeted Parameter Tuning:** Isolated base model weights and injected a parameter-efficient Low-Rank Adaptation (LoRA) layer targeting all linear matrix structures (`q_proj`, `k_proj`, `v_proj`, etc.) to minimize memory usage while preserving base language capabilities.
* **Non-Blocking Multithreaded Engine:** Isolated compute-bound text generation routines inside a detached background execution thread using Hugging Face's `TextIteratorStreamer`, preventing runtime blocking.
* **Low-Latency Serving via FastAPI:** Implemented an asynchronous web layer utilizing `StreamingResponse` to deliver Server-Sent Events (SSE) token-by-token back to the client, aggressively reducing the Time-to-First-Token (TTFT) metric.

---

## Project Structure

* `/config/config.yaml`: Decoupled architecture hyperparameters, training paths, and quantization flags.
* `/src/data_utils.py`: Transforms raw dataset schemas into structured instruction-following conversation boundaries (`<|system|>`, `<|user|>`, `<|assistant|>`).
* `/src/finetune.py`: Script leveraging the modern `trl` library (`SFTConfig` and `SFTTrainer`) to fine-tune the model.
* `/src/engine.py`: Encapsulates the multi-threaded model-streaming wrapper.
* `/src/app.py`: Asynchronous web server hosting production inference endpoints.

---

## Step-by-Step Installation & Quickstart

### 1. Environment Setup
```bash
git clone [https://github.com/YOUR_USERNAME/llm-streaming-sft-service.git](https://github.com/YOUR_USERNAME/llm-streaming-sft-service.git)
cd llm-streaming-sft-service
pip install -r requirements.txt
```

### 2. Execute Fine-Tuning Pipeline
* Run the fine-tuning module to optimize the model parameters against the educational dataset and output localized weights into the /outputs directory:
```bash
python src/finetune.py
```

### 3. Launch Local Asynchronous API Server
```bash
python src/app.py
```

### 4. Query the Live Streaming Endpoint
```bash
import requests

url = "[http://127.0.0.1:8000/v1/chat/stream](http://127.0.0.1:8000/v1/chat/stream)"
payload = {
    "prompt": "<|system|>\nYou are an expert writing coach.<|end|>\n<|user|>\nFix this: She do not like studying history.<|end|>\n<|assistant|>\n"
}

response = requests.post(url, json=payload, stream=True)
for line in response.iter_lines():
    if line:
        decoded = line.decode('utf-8').replace("data: ", "")
        print(decoded, end="", flush=True)
```

## Evaluation & Optimization Performance
Pipeline State,VRAM Footprint,Delivery Style,Architecture Type
Baseline (FP16),~16 GB (Out-of-Memory),Blocked (Full Paragraph),Hardcoded Notebook
Optimized Service,~5.5 GB,Streaming (Token-by-Token),Modular Production Code
