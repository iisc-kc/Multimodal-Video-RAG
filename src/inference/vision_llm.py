"""Vision-language model for frame analysis."""

from pathlib import Path
from typing import Optional
import base64
import requests
import json

from ..utils.logger import log
from ..utils.config import settings


class VisionLLM:
    """Vision-language model wrapper for analyzing frames."""
    
    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None
    ):
        """Initialize vision LLM.
        
        Args:
            host: Ollama server URL
            model: Vision model name (llama3.2-vision, qwen2-vl, etc.)
        """
        self.host = host or settings.ollama_host
        self.model = model or settings.vision_model
        
        log.info(f"Initialized Vision LLM: {self.model}")
    
    def analyze_image(
        self,
        image_path: Path,
        prompt: str,
        temperature: float = 0.3
    ) -> str:
        """Analyze image with vision-language model.
        
        Args:
            image_path: Path to image file
            prompt: Question or instruction about the image
            temperature: Sampling temperature
            
        Returns:
            Model response
        """
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Prepare payload for Ollama vision API
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_data],
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                log.error(f"Vision API error: {response.status_code}")
                return ""
                
        except Exception as e:
            log.error(f"Error analyzing image: {e}")
            return ""
    
    def describe_lecture_content(
        self,
        image_path: Path
    ) -> dict:
        """Get structured description of lecture content.
        
        Args:
            image_path: Path to slide/frame
            
        Returns:
            Dictionary with description fields
        """
        prompt = """Analyze this lecture slide/frame and provide:
1. Main Topic: What is the primary subject?
2. Visual Elements: Describe any diagrams, charts, graphs, or images
3. Text Content: Key text visible on the slide
4. Technical Content: Any equations, code, or formulas
5. Learning Objective: What concept is being taught?

Format your response as a structured analysis."""
        
        response = self.analyze_image(image_path, prompt, temperature=0.2)
        
        # Parse response (simplified - could use structured output)
        return {
            "raw_description": response,
            "has_diagram": any(word in response.lower() for word in ["diagram", "chart", "graph", "flowchart"]),
            "has_code": any(word in response.lower() for word in ["code", "function", "algorithm"]),
            "has_equation": any(word in response.lower() for word in ["equation", "formula", "mathematical"])
        }
    
    def answer_visual_question(
        self,
        image_path: Path,
        question: str
    ) -> str:
        """Answer a specific question about an image.
        
        Args:
            image_path: Path to image
            question: Question to answer
            
        Returns:
            Answer text
        """
        return self.analyze_image(image_path, question, temperature=0.3)
    
    def compare_frames(
        self,
        frame1_path: Path,
        frame2_path: Path,
        aspect: str = "content"
    ) -> str:
        """Compare two frames (requires sequential analysis).
        
        Args:
            frame1_path: First frame
            frame2_path: Second frame
            aspect: What to compare (content, progression, etc.)
            
        Returns:
            Comparison description
        """
        # Analyze both frames
        desc1 = self.analyze_image(
            frame1_path,
            f"Describe the {aspect} shown in this lecture slide."
        )
        
        desc2 = self.analyze_image(
            frame2_path,
            f"Describe the {aspect} shown in this lecture slide."
        )
        
        # Note: True comparison would require multi-image input
        # This is a simplified version
        return f"Frame 1: {desc1}\n\nFrame 2: {desc2}"
