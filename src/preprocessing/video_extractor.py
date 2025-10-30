"""Video extraction and frame sampling module."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import subprocess
import json

from moviepy.editor import VideoFileClip
from tqdm import tqdm

from ..utils.logger import log
from ..utils.config import settings


class VideoExtractor:
    """Extract frames and audio from video files."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize the video extractor.
        
        Args:
            output_dir: Directory to save extracted data
        """
        self.output_dir = output_dir or settings.processed_output_dir
        self.frame_rate = settings.frame_sampling_rate
        self.resolution = settings.video_resolution
        
    def extract_video_info(self, video_path: Path) -> Dict:
        """Extract metadata from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing video metadata
        """
        log.info(f"Extracting metadata from {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        metadata = {
            "filename": video_path.name,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration,
            "duration_formatted": self._format_duration(duration)
        }
        
        log.info(f"Video info: {duration:.2f}s, {fps:.2f} FPS, {width}x{height}")
        return metadata
    
    def extract_frames(
        self, 
        video_path: Path, 
        lecture_id: str
    ) -> List[Dict]:
        """Extract frames at specified sampling rate.
        
        Args:
            video_path: Path to video file
            lecture_id: Unique identifier for the lecture
            
        Returns:
            List of dictionaries containing frame info
        """
        log.info(f"Extracting frames from {video_path.name} at {self.frame_rate} FPS")
        
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling interval
        sample_interval = int(video_fps / self.frame_rate)
        
        # Create output directory for frames
        frames_dir = self.output_dir / lecture_id / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        frames_info = []
        frame_idx = 0
        saved_count = 0
        
        pbar = tqdm(total=total_frames, desc="Extracting frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at specified rate
            if frame_idx % sample_interval == 0:
                timestamp = frame_idx / video_fps
                
                # Resize frame
                frame_resized = self._resize_frame(frame)
                
                # Save frame
                frame_filename = f"frame_{saved_count:06d}.jpg"
                frame_path = frames_dir / frame_filename
                cv2.imwrite(str(frame_path), frame_resized)
                
                frames_info.append({
                    "frame_id": saved_count,
                    "timestamp": timestamp,
                    "timestamp_formatted": self._format_duration(timestamp),
                    "path": str(frame_path),
                    "original_frame_idx": frame_idx
                })
                
                saved_count += 1
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        log.info(f"Extracted {saved_count} frames from {total_frames} total frames")
        return frames_info
    
    def extract_audio(
        self, 
        video_path: Path, 
        lecture_id: str
    ) -> Path:
        """Extract audio from video file.
        
        Args:
            video_path: Path to video file
            lecture_id: Unique identifier for the lecture
            
        Returns:
            Path to extracted audio file
        """
        log.info(f"Extracting audio from {video_path.name}")
        
        # Create output directory
        audio_dir = self.output_dir / lecture_id / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        audio_path = audio_dir / "audio.wav"
        
        # Extract audio using moviepy
        try:
            video = VideoFileClip(str(video_path))
            video.audio.write_audiofile(
                str(audio_path),
                codec='pcm_s16le',
                fps=16000,  # 16kHz for Whisper
                nbytes=2,
                verbose=False,
                logger=None
            )
            video.close()
            
            log.info(f"Audio extracted to {audio_path}")
            return audio_path
            
        except Exception as e:
            log.error(f"Error extracting audio: {e}")
            raise
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame while maintaining aspect ratio.
        
        Args:
            frame: Input frame
            
        Returns:
            Resized frame
        """
        height, width = frame.shape[:2]
        
        if height > self.resolution:
            scale = self.resolution / height
            new_width = int(width * scale)
            new_height = self.resolution
            frame = cv2.resize(frame, (new_width, new_height))
        
        return frame
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in HH:MM:SS format.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
