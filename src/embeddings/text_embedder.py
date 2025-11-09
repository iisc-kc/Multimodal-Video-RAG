"""Text embeddings using sentence transformers."""

import numpy as np
from typing import List, Union

from sentence_transformers import SentenceTransformer

from ..utils.logger import log
from ..utils.config import settings


class TextEmbedder:
    """Generate text embeddings using SentenceTransformers."""
    
    def __init__(self, model_name: str = None, device: str = None):
        """Initialize text embedder.
        
        Args:
            model_name: Sentence transformer model name
            device: Device to run model on (cuda/cpu)
        """
        self.model_name = model_name or settings.text_embedding_model
        self.device = device or "cuda"
        
        self._load_model()
    
    def _load_model(self):
        """Load sentence transformer model."""
        log.info(f"Loading text embedding model: {self.model_name}")

        try:
            # Some community models require executing remote config/code from the repo.
            # Tell HuggingFace transformers to allow that by setting trust_remote_code=True.
            # SentenceTransformer forwards kwargs to the underlying transformers loader.
            self.model = SentenceTransformer(
                self.model_name, device=self.device, trust_remote_code=True
            )
        except Exception as exc:
            log.error(
                "Failed to load text embedding model '%s'. If this is a community model, "
                "ensure you trust the repository and set `trust_remote_code=True`. "
                "Original error: %s",
                self.model_name,
                exc,
            )
            raise

        log.info(f"Text embedding model loaded, dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Text embedding vector
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embedding
    
    def embed_texts_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            show_progress: Show progress bar
            
        Returns:
            Array of text embeddings
        """
        log.info(f"Generating embeddings for {len(texts)} text chunks")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress
        )
        
        log.info(f"Generated {len(embeddings)} text embeddings")
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a search query.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding vector
        """
        # Some models have specific query prefixes
        if "nomic" in self.model_name.lower():
            query = f"search_query: {query}"
        
        return self.embed_text(query)
    
    def embed_document(self, document: str) -> np.ndarray:
        """Generate embedding for a document.
        
        Args:
            document: Document text
            
        Returns:
            Document embedding vector
        """
        # Some models have specific document prefixes
        if "nomic" in self.model_name.lower():
            document = f"search_document: {document}"
        
        return self.embed_text(document)
    
    def compute_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
