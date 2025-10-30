"""Agent tools for RAG operations."""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np

from ..embeddings.clip_embedder import CLIPEmbedder
from ..embeddings.text_embedder import TextEmbedder
from ..vector_store.qdrant_client import MultimodalVectorStore
from ..vector_store.schemas import ModalityType
from ..inference.vision_llm import VisionLLM
from ..utils.logger import log
from ..utils.config import settings


class MultimodalRetriever:
    """Multimodal retrieval tool."""
    
    def __init__(
        self,
        vector_store: MultimodalVectorStore,
        text_embedder: TextEmbedder,
        clip_embedder: CLIPEmbedder
    ):
        """Initialize retriever.
        
        Args:
            vector_store: Vector database client
            text_embedder: Text embedding model
            clip_embedder: CLIP embedding model
        """
        self.vector_store = vector_store
        self.text_embedder = text_embedder
        self.clip_embedder = clip_embedder
        
    def retrieve_text(
        self,
        query: str,
        limit: int = 5,
        lecture_id: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None
    ) -> List[Dict]:
        """Retrieve relevant text chunks.
        
        Args:
            query: Search query
            limit: Number of results
            lecture_id: Filter by lecture
            time_range: Time window filter
            
        Returns:
            List of text results
        """
        log.debug(f"Retrieving text for query: {query}")
        
        query_embedding = self.text_embedder.embed_query(query)
        
        results = self.vector_store.search_text(
            query_embedding=query_embedding,
            limit=limit,
            lecture_id=lecture_id,
            time_range=time_range,
            min_score=settings.similarity_threshold
        )
        
        return results
    
    def retrieve_visual(
        self,
        query: str,
        limit: int = 5,
        lecture_id: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
        slides_only: bool = False
    ) -> List[Dict]:
        """Retrieve relevant visual frames.
        
        Args:
            query: Search query (converted to CLIP embedding)
            limit: Number of results
            lecture_id: Filter by lecture
            time_range: Time window filter
            slides_only: Only search slides
            
        Returns:
            List of visual results
        """
        log.debug(f"Retrieving visual content for query: {query}")
        
        query_embedding = self.clip_embedder.embed_text(query)
        
        results = self.vector_store.search_visual(
            query_embedding=query_embedding,
            limit=limit,
            lecture_id=lecture_id,
            time_range=time_range,
            slides_only=slides_only
        )
        
        return results
    
    def retrieve_slides(
        self,
        query: str,
        search_mode: str = "text",
        limit: int = 5,
        lecture_id: Optional[str] = None
    ) -> List[Dict]:
        """Retrieve relevant slides.
        
        Args:
            query: Search query
            search_mode: "text" or "visual"
            limit: Number of results
            lecture_id: Filter by lecture
            
        Returns:
            List of slide results
        """
        log.debug(f"Retrieving slides for query: {query} (mode: {search_mode})")
        
        if search_mode == "visual":
            query_embedding = self.clip_embedder.embed_text(query)
        else:
            query_embedding = self.text_embedder.embed_query(query)
        
        results = self.vector_store.search_slides(
            query_embedding=query_embedding,
            search_mode=search_mode,
            limit=limit,
            lecture_id=lecture_id
        )
        
        return results
    
    def retrieve_multimodal(
        self,
        query: str,
        modalities: List[ModalityType],
        limit_per_modality: int = 3,
        lecture_id: Optional[str] = None
    ) -> Dict[str, List[Dict]]:
        """Retrieve across multiple modalities.
        
        Args:
            query: Search query
            modalities: List of modalities to search
            limit_per_modality: Results per modality
            lecture_id: Filter by lecture
            
        Returns:
            Dictionary mapping modality to results
        """
        log.info(f"Multimodal retrieval for: {query}")
        
        results = {}
        
        if ModalityType.TEXT in modalities:
            results["text"] = self.retrieve_text(
                query, limit=limit_per_modality, lecture_id=lecture_id
            )
        
        if ModalityType.VISUAL in modalities:
            results["visual"] = self.retrieve_visual(
                query, limit=limit_per_modality, lecture_id=lecture_id
            )
        
        if ModalityType.SLIDE in modalities:
            results["slides"] = self.retrieve_slides(
                query, limit=limit_per_modality, lecture_id=lecture_id
            )
        
        return results


class TemporalRetriever:
    """Temporal context retrieval tool."""
    
    def __init__(self, vector_store: MultimodalVectorStore):
        """Initialize temporal retriever.
        
        Args:
            vector_store: Vector database client
        """
        self.vector_store = vector_store
    
    def get_context_around_timestamp(
        self,
        lecture_id: str,
        timestamp: float,
        window: int = None,
        modality: Optional[ModalityType] = None
    ) -> List[Dict]:
        """Get content around a specific time.
        
        Args:
            lecture_id: Lecture identifier
            timestamp: Target timestamp
            window: Time window in seconds
            modality: Filter by modality
            
        Returns:
            List of temporal results
        """
        window = window or settings.temporal_window
        
        log.debug(f"Getting temporal context at {timestamp}s ±{window}s")
        
        return self.vector_store.get_temporal_context(
            lecture_id=lecture_id,
            timestamp=timestamp,
            window=window,
            modality=modality
        )
    
    def find_concept_progression(
        self,
        concept_results: List[Dict],
    ) -> List[Dict]:
        """Order results by timestamp to show concept progression.
        
        Args:
            concept_results: Results from retrieval
            
        Returns:
            Temporally ordered results
        """
        # Sort by timestamp
        sorted_results = sorted(
            concept_results,
            key=lambda x: x.get("payload", {}).get("timestamp", 0)
        )
        
        return sorted_results


class VisualAnalyzer:
    """Visual content analysis tool."""
    
    def __init__(self, vision_llm: VisionLLM):
        """Initialize visual analyzer.
        
        Args:
            vision_llm: Vision-language model
        """
        self.vision_llm = vision_llm
    
    def analyze_frame(
        self,
        frame_path: Path,
        question: str
    ) -> str:
        """Analyze a video frame with VLM.
        
        Args:
            frame_path: Path to frame image
            question: Question about the frame
            
        Returns:
            Analysis response
        """
        log.debug(f"Analyzing frame: {frame_path.name}")
        
        response = self.vision_llm.analyze_image(
            image_path=frame_path,
            prompt=question
        )
        
        return response
    
    def describe_visual_content(
        self,
        frame_path: Path
    ) -> str:
        """Get detailed description of visual content.
        
        Args:
            frame_path: Path to frame image
            
        Returns:
            Description text
        """
        prompt = """Describe this lecture slide/frame in detail. Include:
1. Main topic or concept shown
2. Any diagrams, charts, or visual elements
3. Text content visible
4. Mathematical equations or code if present
5. Overall purpose of this visual"""
        
        return self.analyze_frame(frame_path, prompt)
    
    def identify_diagram_type(
        self,
        frame_path: Path
    ) -> str:
        """Identify the type of diagram shown.
        
        Args:
            frame_path: Path to frame image
            
        Returns:
            Diagram type
        """
        prompt = "What type of diagram or visualization is shown? (e.g., flowchart, architecture diagram, graph, table, equation, code snippet, etc.)"
        
        return self.analyze_frame(frame_path, prompt)


class CrossModalLinker:
    """Cross-modal content linking tool."""
    
    def __init__(self, vector_store: MultimodalVectorStore):
        """Initialize cross-modal linker.
        
        Args:
            vector_store: Vector database client
        """
        self.vector_store = vector_store
    
    def link_text_to_visual(
        self,
        text_result: Dict,
        time_window: int = 5
    ) -> Optional[Dict]:
        """Find visual content corresponding to text.
        
        Args:
            text_result: Text retrieval result
            time_window: Time window to search
            
        Returns:
            Linked visual result
        """
        timestamp = text_result.get("payload", {}).get("timestamp")
        lecture_id = text_result.get("payload", {}).get("lecture_id")
        
        if not timestamp or not lecture_id:
            return None
        
        # Find slides shown during this text
        slides = self.vector_store.get_temporal_context(
            lecture_id=lecture_id,
            timestamp=timestamp,
            window=time_window,
            modality=ModalityType.SLIDE
        )
        
        return slides[0] if slides else None
    
    def link_visual_to_text(
        self,
        visual_result: Dict,
        time_window: int = 5
    ) -> List[Dict]:
        """Find text explanation for visual content.
        
        Args:
            visual_result: Visual retrieval result
            time_window: Time window to search
            
        Returns:
            Linked text results
        """
        timestamp = visual_result.get("payload", {}).get("timestamp")
        lecture_id = visual_result.get("payload", {}).get("lecture_id")
        
        if not timestamp or not lecture_id:
            return []
        
        # Find text spoken during this visual
        text_results = self.vector_store.get_temporal_context(
            lecture_id=lecture_id,
            timestamp=timestamp,
            window=time_window,
            modality=ModalityType.TEXT
        )
        
        return text_results
