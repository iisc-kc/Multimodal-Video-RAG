"""Configuration management for the Multimodal Video RAG system."""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Vector Database
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")

    # Ollama LLM
    ollama_host: str = Field(default="http://localhost:11434", env="OLLAMA_HOST")
    llm_model: str = Field(default="llama3.1:8b-instruct-fp16", env="LLM_MODEL")
    vision_model: str = Field(default="llama3.2-vision:11b", env="VISION_MODEL")

    # Embedding Models
    text_embedding_model: str = Field(
        default="nomic-ai/nomic-embed-text-v1.5", 
        env="TEXT_EMBEDDING_MODEL"
    )
    vision_embedding_model: str = Field(
        default="openai/clip-vit-large-patch14", 
        env="VISION_EMBEDDING_MODEL"
    )

    # Whisper
    whisper_model: str = Field(default="large-v3", env="WHISPER_MODEL")
    whisper_device: str = Field(default="cuda", env="WHISPER_DEVICE")
    use_faster_whisper: bool = Field(default=True, env="USE_FASTER_WHISPER")

    # OCR
    ocr_engine: str = Field(default="easyocr", env="OCR_ENGINE")
    ocr_languages: List[str] = Field(default=["en"], env="OCR_LANGUAGES")

    # Video Processing
    frame_sampling_rate: float = Field(default=1.0, env="FRAME_SAMPLING_RATE")
    min_scene_duration: float = Field(default=2.0, env="MIN_SCENE_DURATION")
    slide_change_threshold: float = Field(default=0.3, env="SLIDE_CHANGE_THRESHOLD")
    video_resolution: int = Field(default=720, env="VIDEO_RESOLUTION")

    # Text Chunking
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    max_chunk_size: int = Field(default=1024, env="MAX_CHUNK_SIZE")

    # Retrieval
    top_k_retrieval: int = Field(default=5, env="TOP_K_RETRIEVAL")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    enable_reranking: bool = Field(default=True, env="ENABLE_RERANKING")
    temporal_window: int = Field(default=30, env="TEMPORAL_WINDOW")

    # Agent
    max_agent_iterations: int = Field(default=5, env="MAX_AGENT_ITERATIONS")
    enable_temporal_reasoning: bool = Field(default=True, env="ENABLE_TEMPORAL_REASONING")
    enable_cross_modal_linking: bool = Field(default=True, env="ENABLE_CROSS_MODAL_LINKING")
    verbose_logging: bool = Field(default=True, env="VERBOSE_LOGGING")

    # Storage Paths
    data_dir: Path = Field(default=Path("./data"), env="DATA_DIR")
    video_input_dir: Path = Field(default=Path("./data/videos"), env="VIDEO_INPUT_DIR")
    processed_output_dir: Path = Field(default=Path("./data/processed"), env="PROCESSED_OUTPUT_DIR")
    index_dir: Path = Field(default=Path("./data/index"), env="INDEX_DIR")
    cache_dir: Path = Field(default=Path("./data/cache"), env="CACHE_DIR")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Path = Field(default=Path("./logs/app.log"), env="LOG_FILE")

    # Web UI
    gradio_port: int = Field(default=7860, env="GRADIO_PORT")
    gradio_share: bool = Field(default=False, env="GRADIO_SHARE")

    # Validators
    @field_validator('top_k_retrieval')
    @classmethod
    def validate_top_k(cls, v):
        if v <= 0:
            raise ValueError('top_k_retrieval must be positive')
        if v > 100:
            raise ValueError('top_k_retrieval should not exceed 100 for performance')
        return v
    
    @field_validator('similarity_threshold')
    @classmethod
    def validate_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('similarity_threshold must be between 0 and 1')
        return v
    
    @field_validator('frame_sampling_rate')
    @classmethod
    def validate_frame_rate(cls, v):
        if v <= 0:
            raise ValueError('frame_sampling_rate must be positive')
        if v > 30:
            raise ValueError('frame_sampling_rate > 30 fps may cause performance issues')
        return v
    
    @field_validator('chunk_size', 'chunk_overlap')
    @classmethod
    def validate_chunk_params(cls, v, info):
        if v <= 0:
            raise ValueError('chunk_size and chunk_overlap must be positive')
        # Check if chunk_size is too large (nomic-embed has 8192 token limit)
        if info.field_name == 'chunk_size' and v > 2048:
            import warnings
            warnings.warn(
                f'chunk_size={v} may exceed model token limits. '
                'Recommended: <= 2048 for safety with nomic-embed-text',
                UserWarning
            )
        return v
    
    @field_validator('temporal_window')
    @classmethod
    def validate_temporal_window(cls, v):
        if v < 0:
            raise ValueError('temporal_window cannot be negative')
        if v > 300:
            raise ValueError('temporal_window > 300 seconds may return too many results')
        return v

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.data_dir,
            self.video_input_dir,
            self.processed_output_dir,
            self.index_dir,
            self.cache_dir,
            self.log_file.parent,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
