"""Ollama client for local LLM inference."""

from typing import Optional, List, Dict
import requests
import json

from ..utils.logger import log
from ..utils.config import settings


class OllamaClient:
    """Client for Ollama API."""
    
    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None
    ):
        """Initialize Ollama client.
        
        Args:
            host: Ollama server URL
            model: Default model name
        """
        self.host = host or settings.ollama_host
        self.model = model or settings.llm_model
        
        # Verify connection
        self._check_connection()
    
    def _check_connection(self):
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.host}/api/tags")
            if response.status_code == 200:
                log.info(f"Connected to Ollama at {self.host}")
            else:
                log.warning(f"Ollama server returned status {response.status_code}")
        except Exception as e:
            log.error(f"Failed to connect to Ollama: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """Generate completion from prompt.
        
        Args:
            prompt: Input prompt
            model: Model name (defaults to self.model)
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Stream response
            
        Returns:
            Generated text
        """
        model = model or self.model
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
            }
        }
        
        if system:
            payload["system"] = system
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload
            )
            
            if response.status_code == 200:
                if stream:
                    # Handle streaming response
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            if "response" in data:
                                full_response += data["response"]
                    return full_response
                else:
                    result = response.json()
                    return result.get("response", "")
            else:
                log.error(f"Ollama API error: {response.status_code}")
                return ""
                
        except Exception as e:
            log.error(f"Error calling Ollama: {e}")
            return ""
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Chat completion with message history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            Assistant response
        """
        model = model or self.model
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")
            else:
                log.error(f"Ollama chat error: {response.status_code}")
                return ""
                
        except Exception as e:
            log.error(f"Error in Ollama chat: {e}")
            return ""
    
    def list_models(self) -> List[str]:
        """List available models.
        
        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.host}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except Exception as e:
            log.error(f"Error listing models: {e}")
            return []
