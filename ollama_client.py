import asyncio
import time
from typing import List, Dict, Any, Optional, Union
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer
import httpx
import logging
from config import settings

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.timeout = settings.OLLAMA_TIMEOUT
        self.model = settings.OLLAMA_MODEL
        self.client = None
        self.initialize_client()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.last_connection_check = 0
        self.is_ollama_running = False
        
    def initialize_client(self):
        """Initialize the HTTP client with proper timeouts"""
        if self.client:
            try:
                asyncio.run(self.client.aclose())
            except Exception:
                pass
                
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                connect=30.0,
                read=float(self.timeout),
                write=60.0,
                pool=60.0
            )
        )
    
    async def check_ollama_connection(self) -> bool:
        """Check if Ollama service is running"""
        current_time = time.time()
        if current_time - self.last_connection_check < 30:  # Cache check for 30 seconds
            return self.is_ollama_running
            
        self.last_connection_check = current_time
        
        try:
            # First check if the server is reachable
            try:
                response = await self.client.get("/api/version")
                response.raise_for_status()
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                logger.warning(f"Ollama server not reachable at {self.base_url}: {str(e)}")
                self.is_ollama_running = False
                return False
                
            # Then check if the model is available
            try:
                response = await self.client.get("/api/tags")
                response.raise_for_status()
                models = response.json().get('models', [])
                model_available = any(model.get('name', '').startswith(self.model) for model in models)
                if not model_available:
                    logger.warning(f"Model '{self.model}' not found in Ollama. Available models: {[m.get('name') for m in models]}")
                    self.is_ollama_running = False
                    return False
                    
            except Exception as e:
                logger.error(f"Error checking Ollama models: {str(e)}")
                self.is_ollama_running = False
                return False
            self.is_ollama_running = True
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to Ollama: {str(e)}")
            self.is_ollama_running = False
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def generate(
        self,
        model: str = "llama3",
        prompt: str = "",
        system: str = "You are a helpful AI assistant.",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Generate text using the Ollama API with retry logic"""
        try:
            # Check Ollama connection before proceeding
            if not self.is_ollama_running:
                if not await self.check_ollama_connection():
                    raise ConnectionError("Could not connect to Ollama service")
            
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": stream
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error in generate: {str(e)}")
            raise
        
    async def close(self):
        await self.client.aclose()
    
    async def generate(
        self,
        model: str = "llama3",
        prompt: str = "",
        system: str = "You are a helpful AI assistant.",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Generate text using the Ollama API"""
        try:
            # Format the messages for Ollama
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    },
                    "stream": stream
                }
            )
            response.raise_for_status()
            
            # Extract the response content
            result = response.json()
            return {
                "response": result.get("message", {}).get("content", ""),
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(result.get("message", {}).get("content", "").split()),
                    "total_tokens": len(prompt.split()) + len(result.get("message", {}).get("content", "").split())
                }
            }
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}", exc_info=True)
            raise
    
    async def chat(
        self,
        model: str = "llama3",
        messages: List[Dict[str, str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Chat completion using the Ollama API"""
        if messages is None:
            messages = []
            
        try:
            response = await self.client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    },
                    "stream": stream
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a given text using local model"""
        if not text.strip():
            return [0.0] * 384  # Default dimension for all-MiniLM-L6-v2
        return self.embedding_model.encode(text, convert_to_numpy=True).tolist()
