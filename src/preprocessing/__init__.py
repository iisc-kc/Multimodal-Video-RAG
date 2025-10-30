"""Preprocessing module for video processing."""

from .video_extractor import VideoExtractor
from .slide_detector import SlideDetector
from .transcription import AudioTranscriber
from .ocr_processor import OCRProcessor

__all__ = [
    'VideoExtractor',
    'SlideDetector',
    'AudioTranscriber',
    'OCRProcessor'
]
