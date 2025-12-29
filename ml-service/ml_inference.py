from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import socket
import hashlib
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import time
import logging
from typing import Optional
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentiment Analysis API",
    description="Real-time sentiment inference with custom KV-Store caching",
    version="1.0.0"
)

# Request/Response models
class TextInput(BaseModel):
    text: str
    use_cache: bool = True

class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    cached: bool
    inference_time_ms: float

# KV-Store Client
class KVStoreClient:
    """Client for connecting to custom Java KV-Store"""
    
    def __init__(self, host: str = "localhost", port: int = 6379):
        self.host = host
        self.port = port
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info(f"Initialized KV-Store client: {host}:{port}")
    
    def _connect(self) -> socket.socket:
        """Create connection to KV-Store"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        # Skip welcome message
        sock.recv(4096)
        return sock
    
    def _send_command(self, command: str) -> str:
        """Send command and get response"""
        try:
            sock = self._connect()
            sock.sendall(f"{command}\n".encode())
            response = sock.recv(4096).decode()
            sock.close()
            return response
        except Exception as e:
            logger.error(f"KV-Store connection error: {e}")
            return None
    
    def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        response = self._send_command(f"GET {key}")
        if response and not response.startswith("$-1"):
            # Parse Redis-like response: $<length>\r\n<data>\r\n
            lines = response.split("\r\n")
            if len(lines) >= 2:
                self.cache_hits += 1
                logger.info(f"Cache HIT: {key}")
                return lines[1]
        self.cache_misses += 1
        logger.info(f"Cache MISS: {key}")
        return None
    
    def set(self, key: str, value: str) -> bool:
        """Set value in cache"""
        response = self._send_command(f"SET {key} {value}")
        success = response and "+OK" in response
        if success:
            logger.info(f"Cached: {key}")
        return success
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.2f}%"
        }

# ML Model Manager
class SentimentModel:
    """Manages sentiment analysis model"""
    
    def __init__(self):
        logger.info("Loading DistilBERT model...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            'distilbert-base-uncased-finetuned-sst-2-english'
        )
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased-finetuned-sst-2-english'
        )
        self.model.eval()
        self.labels = ['negative', 'positive']
        self.total_inferences = 0
        logger.info("Model loaded successfully")
    
    def predict(self, text: str) -> tuple[str, float, float]:
        """
        Run inference on text
        Returns: (sentiment, confidence, inference_time_ms)
        """
        start_time = time.time()
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, predicted_class = torch.max(predictions, dim=1)
        
        sentiment = self.labels[predicted_class.item()]
        confidence_score = confidence.item()
        inference_time = (time.time() - start_time) * 1000
        
        self.total_inferences += 1
        
        return sentiment, confidence_score, inference_time

# Initialize components
kv_store = KVStoreClient()
model = SentimentModel()

def generate_cache_key(text: str) -> str:
    """Generate deterministic cache key from text"""
    return hashlib.md5(text.encode()).hexdigest()

# API Endpoints

@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(input_data: TextInput):
    """
    Predict sentiment of input text with caching
    """
    text = input_data.text.strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if len(text) > 1000:
        raise HTTPException(status_code=400, detail="Text too long (max 1000 chars)")
    
    # Try cache first
    cached_result = None
    if input_data.use_cache:
        cache_key = generate_cache_key(text)
        cached_result = kv_store.get(cache_key)
    
    if cached_result:
        # Parse cached result: "sentiment:confidence"
        try:
            sentiment, confidence_str = cached_result.split(":")
            confidence = float(confidence_str)
            return PredictionResponse(
                text=text,
                sentiment=sentiment,
                confidence=confidence,
                cached=True,
                inference_time_ms=0.0
            )
        except Exception as e:
            logger.error(f"Cache parse error: {e}")
    
    # Run inference
    sentiment, confidence, inference_time = model.predict(text)
    
    # Cache result
    if input_data.use_cache:
        cache_value = f"{sentiment}:{confidence:.4f}"
        kv_store.set(cache_key, cache_value)
    
    return PredictionResponse(
        text=text,
        sentiment=sentiment,
        confidence=confidence,
        cached=False,
        inference_time_ms=inference_time
    )

@app.get("/stats")
async def get_stats():
    """Get service statistics"""
    return {
        "model": {
            "total_inferences": model.total_inferences
        },
        "cache": kv_store.get_stats()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test KV-Store connection
        kv_store.get("health_check")
        return {
            "status": "healthy",
            "model": "loaded",
            "cache": "connected"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "service": "Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict",
            "stats": "GET /stats",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "ml_inference:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )