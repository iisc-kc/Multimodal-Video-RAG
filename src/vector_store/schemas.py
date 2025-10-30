"""Data schemas for vector store."""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class ModalityType(str, Enum):
    """Modality types for multimodal data."""
    TEXT = "text"
    VISUAL = "visual"
    AUDIO = "audio"
    SLIDE = "slide"


class ContentType(str, Enum):
    """Content types for classification."""
    EXPLANATION = "explanation"
    DIAGRAM = "diagram"
    CODE = "code"
    EQUATION = "equation"
    EXAMPLE = "example"
    DEFINITION = "definition"


class BaseDocument(BaseModel):
    """Base document model."""
    id: str
    lecture_id: str
    modality: ModalityType
    content: str
    timestamp: Optional[float] = None
    timestamp_end: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class TextChunk(BaseDocument):
    """Text chunk from transcription."""
    modality: ModalityType = ModalityType.TEXT
    speaker: Optional[str] = None
    word_count: int = 0
    confidence: Optional[float] = None


class FrameDocument(BaseDocument):
    """Frame/image document."""
    modality: ModalityType = ModalityType.VISUAL
    frame_path: str
    frame_id: int
    slide_id: Optional[int] = None
    is_slide: bool = False


class SlideDocument(BaseDocument):
    """Slide document with OCR text."""
    modality: ModalityType = ModalityType.SLIDE
    slide_path: str
    slide_id: int
    ocr_text: str
    content_type: Optional[ContentType] = None
    has_diagram: bool = False
    has_code: bool = False


class RetrievalResult(BaseModel):
    """Result from vector search."""
    document: BaseDocument
    score: float
    rank: int


class QueryPlan(BaseModel):
    """Agent query plan."""
    query: str
    modalities_to_search: List[ModalityType]
    temporal_filter: Optional[Dict[str, float]] = None
    requires_visual_analysis: bool = False
    requires_cross_modal: bool = False
    reasoning: str = ""
