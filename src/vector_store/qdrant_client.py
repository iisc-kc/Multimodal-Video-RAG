"""Qdrant vector database client for multimodal storage."""

from typing import List, Dict, Optional, Any
from pathlib import Path
import uuid
import time

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue, Range,
    SearchRequest, NamedVector
)
import numpy as np

from ..utils.logger import log
from ..utils.config import settings
from .schemas import (
    BaseDocument, ModalityType, TextChunk, 
    FrameDocument, SlideDocument, RetrievalResult
)


class MultimodalVectorStore:
    """Vector store for multimodal documents using Qdrant."""
    
    # Collection names
    TEXT_COLLECTION = "lecture_text"
    VISUAL_COLLECTION = "lecture_visual"
    SLIDE_COLLECTION = "lecture_slides"
    
    def __init__(self):
        """Initialize Qdrant client and collections."""
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            timeout=300  # 5 minutes timeout for large batch operations
        )
        
        log.info(f"Connected to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
    
    def create_collections(
        self,
        text_dim: int = 768,
        visual_dim: int = 512,
        recreate: bool = False
    ):
        """Create collections for different modalities.
        
        Args:
            text_dim: Text embedding dimension
            visual_dim: Visual embedding dimension
            recreate: If True, delete existing collections and recreate. If False, skip if exists.
        """
        log.info(f"{'Recreating' if recreate else 'Creating'} Qdrant collections")
        
        try:
            # Text collection
            if recreate or not self.client.collection_exists(self.TEXT_COLLECTION):
                if recreate:
                    self.client.delete_collection(self.TEXT_COLLECTION)
                self.client.create_collection(
                    collection_name=self.TEXT_COLLECTION,
                    vectors_config=VectorParams(
                        size=text_dim,
                        distance=Distance.COSINE
                    )
                )
                log.info(f"Created {self.TEXT_COLLECTION} collection")
            else:
                log.info(f"{self.TEXT_COLLECTION} collection already exists, skipping")
            
            # Visual collection (frames)
            if recreate or not self.client.collection_exists(self.VISUAL_COLLECTION):
                if recreate:
                    self.client.delete_collection(self.VISUAL_COLLECTION)
                self.client.create_collection(
                    collection_name=self.VISUAL_COLLECTION,
                    vectors_config=VectorParams(
                        size=visual_dim,
                        distance=Distance.COSINE
                    )
                )
                log.info(f"Created {self.VISUAL_COLLECTION} collection")
            else:
                log.info(f"{self.VISUAL_COLLECTION} collection already exists, skipping")
            
            # Slide collection (multimodal: visual + text)
            if recreate or not self.client.collection_exists(self.SLIDE_COLLECTION):
                if recreate:
                    self.client.delete_collection(self.SLIDE_COLLECTION)
                self.client.create_collection(
                    collection_name=self.SLIDE_COLLECTION,
                    vectors_config={
                        "visual": VectorParams(size=visual_dim, distance=Distance.COSINE),
                        "text": VectorParams(size=text_dim, distance=Distance.COSINE)
                    }
                )
                log.info(f"Created {self.SLIDE_COLLECTION} collection")
            else:
                log.info(f"{self.SLIDE_COLLECTION} collection already exists, skipping")
            
            log.info("Collections setup complete")
        
        except Exception as e:
            log.error(f"Error creating collections: {e}")
            raise
    
    def add_text_documents(
        self,
        documents: List[TextChunk],
        embeddings: np.ndarray
    ):
        """Add text documents to vector store.
        
        Args:
            documents: List of text documents
            embeddings: Document embeddings
        """
        points = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "lecture_id": doc.lecture_id,
                    "modality": doc.modality.value,
                    "content": doc.content,
                    "timestamp": doc.timestamp,
                    "timestamp_end": doc.timestamp_end,
                    "word_count": doc.word_count,
                    **doc.metadata
                }
            )
            points.append(point)
        
        self.client.upsert(
            collection_name=self.TEXT_COLLECTION,
            points=points
        )
        
        log.info(f"Added {len(points)} text documents")
    
    def add_visual_documents(
        self,
        documents: List[FrameDocument],
        embeddings: np.ndarray
    ):
        """Add visual documents to vector store.
        
        Args:
            documents: List of frame documents
            embeddings: Frame embeddings
        """
        points = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "lecture_id": doc.lecture_id,
                    "modality": doc.modality.value,
                    "content": doc.content,
                    "timestamp": doc.timestamp,
                    "frame_path": doc.frame_path,
                    "frame_id": doc.frame_id,
                    "is_slide": doc.is_slide,
                    **doc.metadata
                }
            )
            points.append(point)
        
        # Batch upsert for large collections (split into smaller chunks for stability)
        batch_size = 50  # Reduced from 100 for better reliability
        max_retries = 3
        total_batches = (len(points) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(points), batch_size):
            batch = points[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            # Retry logic for timeout issues
            for attempt in range(max_retries):
                try:
                    self.client.upsert(
                        collection_name=self.VISUAL_COLLECTION,
                        points=batch,
                        wait=False  # Don't wait for indexing - async mode for speed
                    )
                    if batch_num % 10 == 0 or batch_num == total_batches:
                        log.debug(f"Progress: {batch_num}/{total_batches} batches uploaded")
                    break  # Success, exit retry loop
                except Exception as e:
                    if attempt < max_retries - 1:
                        log.warning(f"Batch {batch_num} attempt {attempt + 1} failed, retrying... ({e})")
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        log.error(f"Failed to upsert batch {batch_num} after {max_retries} attempts: {e}")
                        raise
        
        log.info(f"Added {len(points)} visual documents")
    
    def add_slide_documents(
        self,
        documents: List[SlideDocument],
        visual_embeddings: np.ndarray,
        text_embeddings: np.ndarray
    ):
        """Add slide documents with both visual and text embeddings.
        
        Args:
            documents: List of slide documents
            visual_embeddings: Visual embeddings for slides
            text_embeddings: Text embeddings from OCR
        """
        points = []
        
        for i, doc in enumerate(documents):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "visual": visual_embeddings[i].tolist(),
                    "text": text_embeddings[i].tolist()
                },
                payload={
                    "lecture_id": doc.lecture_id,
                    "modality": doc.modality.value,
                    "content": doc.content,
                    "ocr_text": doc.ocr_text,
                    "timestamp": doc.timestamp,
                    "timestamp_end": doc.timestamp_end,
                    "slide_id": doc.slide_id,
                    "slide_path": doc.slide_path,
                    "has_diagram": doc.has_diagram,
                    "has_code": doc.has_code,
                    **doc.metadata
                }
            )
            points.append(point)
        
        # Batch upsert for large collections (split into smaller chunks for stability)
        batch_size = 50  # Reduced from 100 for better reliability
        max_retries = 3
        total_batches = (len(points) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(points), batch_size):
            batch = points[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            # Retry logic for timeout issues
            for attempt in range(max_retries):
                try:
                    self.client.upsert(
                        collection_name=self.SLIDE_COLLECTION,
                        points=batch,
                        wait=False  # Don't wait for indexing - async mode for speed
                    )
                    if batch_num % 10 == 0 or batch_num == total_batches:
                        log.debug(f"Progress: {batch_num}/{total_batches} batches uploaded")
                    break  # Success, exit retry loop
                except Exception as e:
                    if attempt < max_retries - 1:
                        log.warning(f"Batch {batch_num} attempt {attempt + 1} failed, retrying... ({e})")
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        log.error(f"Failed to upsert batch {batch_num} after {max_retries} attempts: {e}")
                        raise
        
        log.info(f"Added {len(points)} slide documents")
    
    def search_text(
        self,
        query_embedding: np.ndarray,
        limit: int = 5,
        lecture_id: Optional[str] = None,
        time_range: Optional[tuple] = None,
        min_score: Optional[float] = None
    ) -> List[Dict]:
        """Search text collection.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            lecture_id: Filter by lecture ID
            time_range: Filter by time range (start, end)
            min_score: Minimum similarity score
            
        Returns:
            List of search results
        """
        try:
            # Build filter
            filter_conditions = []
            
            if lecture_id:
                filter_conditions.append(
                    FieldCondition(key="lecture_id", match=MatchValue(value=lecture_id))
                )
            
            if time_range:
                start, end = time_range
                filter_conditions.append(
                    FieldCondition(key="timestamp", range=Range(gte=start, lte=end))
                )
            
            query_filter = Filter(must=filter_conditions) if filter_conditions else None
            
            # Search
            results = self.client.search(
                collection_name=self.TEXT_COLLECTION,
                query_vector=query_embedding.tolist(),
                limit=limit,
                query_filter=query_filter,
                score_threshold=min_score
            )
            
            return [self._format_result(r) for r in results]
        
        except Exception as e:
            log.error(f"Error searching text collection: {e}")
            return []
    
    def search_visual(
        self,
        query_embedding: np.ndarray,
        limit: int = 5,
        lecture_id: Optional[str] = None,
        time_range: Optional[tuple] = None,
        slides_only: bool = False
    ) -> List[Dict]:
        """Search visual collection.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            lecture_id: Filter by lecture ID
            time_range: Filter by time range
            slides_only: Only search slide frames
            
        Returns:
            List of search results
        """
        try:
            filter_conditions = []
            
            if lecture_id:
                filter_conditions.append(
                    FieldCondition(key="lecture_id", match=MatchValue(value=lecture_id))
                )
            
            if slides_only:
                filter_conditions.append(
                    FieldCondition(key="is_slide", match=MatchValue(value=True))
                )
            
            if time_range:
                start, end = time_range
                filter_conditions.append(
                    FieldCondition(key="timestamp", range=Range(gte=start, lte=end))
                )
            
            query_filter = Filter(must=filter_conditions) if filter_conditions else None
            
            results = self.client.search(
                collection_name=self.VISUAL_COLLECTION,
                query_vector=query_embedding.tolist(),
                limit=limit,
                query_filter=query_filter
            )
            
            return [self._format_result(r) for r in results]
        
        except Exception as e:
            log.error(f"Error searching visual collection: {e}")
            return []
    
    def search_slides(
        self,
        query_embedding: np.ndarray,
        search_mode: str = "text",  # "text", "visual", or "hybrid"
        limit: int = 5,
        lecture_id: Optional[str] = None
    ) -> List[Dict]:
        """Search slide collection.
        
        Args:
            query_embedding: Query embedding vector
            search_mode: Which embedding to search ("text", "visual", "hybrid")
            limit: Maximum number of results
            lecture_id: Filter by lecture ID
            
        Returns:
            List of search results
        """
        try:
            filter_conditions = []
            
            if lecture_id:
                filter_conditions.append(
                    FieldCondition(key="lecture_id", match=MatchValue(value=lecture_id))
                )
            
            query_filter = Filter(must=filter_conditions) if filter_conditions else None
            
            # Determine vector name
            if search_mode in ["text", "visual"]:
                results = self.client.search(
                    collection_name=self.SLIDE_COLLECTION,
                    query_vector=(search_mode, query_embedding.tolist()),
                    limit=limit,
                    query_filter=query_filter
                )
            else:
                # Hybrid search (not directly supported, would need custom implementation)
                results = self.client.search(
                    collection_name=self.SLIDE_COLLECTION,
                    query_vector=("text", query_embedding.tolist()),
                    limit=limit,
                    query_filter=query_filter
                )
            
            return [self._format_result(r) for r in results]
        
        except Exception as e:
            log.error(f"Error searching slides collection: {e}")
            return []
    
    def get_temporal_context(
        self,
        lecture_id: str,
        timestamp: float,
        window: int = 30,
        modality: Optional[ModalityType] = None
    ) -> List[Dict]:
        """Get content around a specific timestamp.
        
        Args:
            lecture_id: Lecture identifier
            timestamp: Target timestamp
            window: Time window in seconds (before/after)
            modality: Filter by modality
            
        Returns:
            List of documents in temporal window
        """
        collection = self.TEXT_COLLECTION  # Default to text
        
        if modality == ModalityType.VISUAL:
            collection = self.VISUAL_COLLECTION
        elif modality == ModalityType.SLIDE:
            collection = self.SLIDE_COLLECTION
        
        filter_conditions = [
            FieldCondition(key="lecture_id", match=MatchValue(value=lecture_id)),
            FieldCondition(
                key="timestamp",
                range=Range(gte=timestamp - window, lte=timestamp + window)
            )
        ]
        
        # Scroll through results (no vector search, just filter)
        results = self.client.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=filter_conditions),
            limit=100
        )
        
        return [self._format_result_scroll(r) for r in results[0]]
    
    def _format_result(self, result) -> Dict:
        """Format search result."""
        return {
            "id": result.id,
            "score": result.score,
            "payload": result.payload
        }
    
    def _format_result_scroll(self, result) -> Dict:
        """Format scroll result."""
        return {
            "id": result.id,
            "payload": result.payload
        }
    
    def get_collection_stats(self) -> Dict:
        """Get statistics for all collections."""
        stats = {}
        
        for collection in [self.TEXT_COLLECTION, self.VISUAL_COLLECTION, self.SLIDE_COLLECTION]:
            try:
                info = self.client.get_collection(collection)
                stats[collection] = {
                    "points_count": info.points_count,
                    "vectors_count": info.vectors_count
                }
            except Exception as e:
                stats[collection] = {"error": str(e)}
        
        return stats
