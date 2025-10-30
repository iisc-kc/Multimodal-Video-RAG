"""Audio transcription using Whisper."""

from pathlib import Path
from typing import List, Dict, Optional
import json

from ..utils.logger import log
from ..utils.config import settings


class AudioTranscriber:
    """Transcribe audio using Whisper models."""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """Initialize audio transcriber.
        
        Args:
            model_name: Whisper model name (tiny, base, small, medium, large-v3)
            device: Device to run model on (cuda/cpu)
        """
        self.model_name = model_name or settings.whisper_model
        self.device = device or settings.whisper_device
        self.use_faster = settings.use_faster_whisper
        
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model."""
        log.info(f"Loading Whisper model: {self.model_name} on {self.device}")
        
        if self.use_faster:
            try:
                from faster_whisper import WhisperModel
                
                # Use faster-whisper for better performance
                self.model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type="float16" if self.device == "cuda" else "int8"
                )
                self.is_faster = True
                log.info("Using faster-whisper for transcription")
                
            except ImportError:
                log.warning("faster-whisper not available, falling back to openai-whisper")
                self.use_faster = False
        
        if not self.use_faster:
            import whisper
            self.model = whisper.load_model(self.model_name, device=self.device)
            self.is_faster = False
            log.info("Using openai-whisper for transcription")
    
    def transcribe(
        self, 
        audio_path: Path,
        language: str = "en"
    ) -> Dict:
        """Transcribe audio file with timestamps.
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: English)
            
        Returns:
            Dictionary containing full transcription and segments
        """
        log.info(f"Transcribing audio: {audio_path.name}")
        
        if self.is_faster:
            return self._transcribe_faster(audio_path, language)
        else:
            return self._transcribe_openai(audio_path, language)
    
    def _transcribe_faster(self, audio_path: Path, language: str) -> Dict:
        """Transcribe using faster-whisper.
        
        Args:
            audio_path: Path to audio file
            language: Language code
            
        Returns:
            Transcription results
        """
        segments, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True  # Voice activity detection
        )
        
        # Convert segments to list
        segment_list = []
        full_text = []
        
        for segment in segments:
            segment_dict = {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "words": []
            }
            
            # Add word-level timestamps if available
            if hasattr(segment, 'words') and segment.words:
                segment_dict["words"] = [
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability
                    }
                    for word in segment.words
                ]
            
            segment_list.append(segment_dict)
            full_text.append(segment.text.strip())
        
        result = {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "text": " ".join(full_text),
            "segments": segment_list
        }
        
        log.info(f"Transcription complete: {len(segment_list)} segments, {info.duration:.2f}s")
        return result
    
    def _transcribe_openai(self, audio_path: Path, language: str) -> Dict:
        """Transcribe using openai-whisper.
        
        Args:
            audio_path: Path to audio file
            language: Language code
            
        Returns:
            Transcription results
        """
        result = self.model.transcribe(
            str(audio_path),
            language=language,
            verbose=False,
            word_timestamps=True
        )
        
        # Format segments
        segment_list = []
        for segment in result["segments"]:
            segment_dict = {
                "id": segment["id"],
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
                "words": []
            }
            
            # Add word-level timestamps if available
            if "words" in segment:
                segment_dict["words"] = [
                    {
                        "word": word["word"],
                        "start": word["start"],
                        "end": word["end"],
                        "probability": word.get("probability", 1.0)
                    }
                    for word in segment["words"]
                ]
            
            segment_list.append(segment_dict)
        
        formatted_result = {
            "language": result["language"],
            "language_probability": 1.0,
            "duration": segment_list[-1]["end"] if segment_list else 0,
            "text": result["text"],
            "segments": segment_list
        }
        
        log.info(f"Transcription complete: {len(segment_list)} segments")
        return formatted_result
    
    def save_transcription(
        self, 
        transcription: Dict, 
        output_path: Path
    ):
        """Save transcription to JSON file.
        
        Args:
            transcription: Transcription dictionary
            output_path: Path to save JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcription, f, indent=2, ensure_ascii=False)
        
        log.info(f"Transcription saved to {output_path}")
    
    def create_text_chunks(
        self,
        transcription: Dict,
        chunk_size: int = None,
        overlap: int = None
    ) -> List[Dict]:
        """Create overlapping text chunks from transcription.
        
        Args:
            transcription: Transcription dictionary
            chunk_size: Maximum words per chunk
            overlap: Number of overlapping words
            
        Returns:
            List of text chunks with timestamps
        """
        chunk_size = chunk_size or settings.chunk_size
        overlap = overlap or settings.chunk_overlap
        
        segments = transcription["segments"]
        chunks = []
        
        current_chunk = []
        current_words = []
        word_count = 0
        
        for segment in segments:
            words = segment["text"].split()
            
            for word in words:
                current_words.append(word)
                word_count += 1
                
                if word_count >= chunk_size:
                    # Create chunk
                    chunk_text = " ".join(current_words)
                    chunks.append({
                        "text": chunk_text,
                        "start": segment["start"],
                        "end": segment["end"],
                        "word_count": len(current_words)
                    })
                    
                    # Overlap for next chunk
                    current_words = current_words[-overlap:] if overlap > 0 else []
                    word_count = len(current_words)
        
        # Add remaining words
        if current_words:
            chunks.append({
                "text": " ".join(current_words),
                "start": segments[-1]["start"] if segments else 0,
                "end": segments[-1]["end"] if segments else 0,
                "word_count": len(current_words)
            })
        
        log.info(f"Created {len(chunks)} text chunks from transcription")
        return chunks
