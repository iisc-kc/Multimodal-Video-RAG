"""CLIP vision embeddings for frames and images."""

import torch
import numpy as np
from pathlib import Path
from typing import List, Union
from PIL import Image

from transformers import CLIPProcessor, CLIPModel

from ..utils.logger import log
from ..utils.config import settings


class CLIPEmbedder:
    """Generate CLIP embeddings for images."""
    
    def __init__(self, model_name: str = None, device: str = None):
        """Initialize CLIP embedder.
        
        Args:
            model_name: CLIP model name
            device: Device to run model on (cuda/cpu)
        """
        self.model_name = model_name or settings.vision_embedding_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self._load_model()
    
    def _load_model(self):
        """Load CLIP model and processor."""
        log.info(f"Loading CLIP model: {self.model_name} on {self.device}")
        
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        
        self.model.eval()
        log.info("CLIP model loaded successfully")
    
    def embed_image(self, image_path: Union[Path, str]) -> np.ndarray:
        """Generate embedding for a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image embedding vector
        """
        image = Image.open(image_path).convert("RGB")
        
        inputs = self.processor(
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]
    
    def embed_images_batch(
        self, 
        image_paths: List[Union[Path, str]], 
        batch_size: int = 32
    ) -> np.ndarray:
        """Generate embeddings for multiple images in batches.
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            
        Returns:
            Array of image embeddings
        """
        log.info(f"Generating embeddings for {len(image_paths)} images")
        
        all_embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # Load images
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            
            # Process batch
            inputs = self.processor(
                images=images,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            all_embeddings.append(image_features.cpu().numpy())
            
            if (i + batch_size) % 100 == 0:
                log.debug(f"Processed {i + batch_size}/{len(image_paths)} images")
        
        embeddings = np.vstack(all_embeddings)
        log.info(f"Generated {len(embeddings)} image embeddings")
        
        return embeddings
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text (for image-text similarity).
        
        Args:
            text: Input text
            
        Returns:
            Text embedding vector
        """
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()[0]
    
    def compute_similarity(
        self, 
        image_embedding: np.ndarray, 
        text_embedding: np.ndarray
    ) -> float:
        """Compute similarity between image and text embeddings.
        
        Args:
            image_embedding: Image embedding vector
            text_embedding: Text embedding vector
            
        Returns:
            Cosine similarity score
        """
        similarity = np.dot(image_embedding, text_embedding)
        return float(similarity)
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.model.config.projection_dim
