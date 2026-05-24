import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from engine import InferenceEngine

app = FastAPI(title="Production-Grade Student Assistant API", version="1.0.0")

# Initialized dynamically at app startup
engine = None

class GenerationRequest(BaseModel):
    prompt: str

@app.on_event("startup")
async def startup_event():
    global engine
    # Points to model configurations defined in runtime environment
    BASE_MODEL = "unsloth/Phi-3-mini-4k-instruct"
    LORA_PATH = "./outputs/phi3-essay-lora"
    
    if not os.path.exists(LORA_PATH):
        raise RuntimeError(f"Weights not found at {LORA_PATH}. Please run fine-tuning pipeline first.")
        
    engine = InferenceEngine(BASE_MODEL, LORA_PATH)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "hardware": "CUDA-Active"}

@app.post("/v1/chat/stream")
async def stream_generation(request: GenerationRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt string cannot be empty.")
    
    try:
        # Wrap the generator in a production streaming protocol
        return StreamingResponse(engine.generate_stream(request.prompt), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Standard production engine listener loop execution
    uvicorn.run(app, host="0.0.0.0", port=8000)