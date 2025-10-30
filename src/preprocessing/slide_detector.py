"""Slide detection and change point identification."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.logger import log
from ..utils.config import settings


class SlideDetector:
    """Detect slide changes in lecture videos."""
    
    def __init__(self, threshold: float = None):
        """Initialize slide detector.
        
        Args:
            threshold: Similarity threshold for slide change detection
        """
        self.threshold = threshold or settings.slide_change_threshold
        self.min_scene_duration = settings.min_scene_duration
        
    def detect_slide_changes(
        self, 
        frames_info: List[Dict]
    ) -> List[Dict]:
        """Detect when slides change in the video.
        
        Args:
            frames_info: List of frame information dictionaries
            
        Returns:
            List of slide segments with start/end times
        """
        log.info(f"Detecting slide changes from {len(frames_info)} frames")
        
        # Load frames and compute histograms
        histograms = []
        for frame_info in frames_info:
            frame = cv2.imread(frame_info["path"])
            hist = self._compute_histogram(frame)
            histograms.append(hist)
        
        # Compute similarity between consecutive frames
        similarities = []
        for i in range(len(histograms) - 1):
            sim = cosine_similarity(
                histograms[i].reshape(1, -1), 
                histograms[i + 1].reshape(1, -1)
            )[0][0]
            similarities.append(sim)
        
        # Detect change points (where similarity drops)
        change_points = [0]  # Start with first frame
        
        for i, sim in enumerate(similarities):
            if sim < (1 - self.threshold):
                # Check minimum scene duration
                prev_change = change_points[-1]
                if frames_info[i + 1]["timestamp"] - frames_info[prev_change]["timestamp"] >= self.min_scene_duration:
                    change_points.append(i + 1)
        
        # Add last frame
        if change_points[-1] != len(frames_info) - 1:
            change_points.append(len(frames_info) - 1)
        
        # Create slide segments
        slides = []
        for i in range(len(change_points) - 1):
            start_idx = change_points[i]
            end_idx = change_points[i + 1]
            
            slide = {
                "slide_id": i,
                "start_frame": start_idx,
                "end_frame": end_idx,
                "start_timestamp": frames_info[start_idx]["timestamp"],
                "end_timestamp": frames_info[end_idx]["timestamp"],
                "duration": frames_info[end_idx]["timestamp"] - frames_info[start_idx]["timestamp"],
                "representative_frame": frames_info[start_idx]["path"]
            }
            slides.append(slide)
        
        log.info(f"Detected {len(slides)} slide segments")
        return slides
    
    def extract_unique_slides(
        self, 
        slides: List[Dict], 
        output_dir: Path
    ) -> List[Path]:
        """Extract representative frames for each unique slide.
        
        Args:
            slides: List of slide segments
            output_dir: Directory to save slide images
            
        Returns:
            List of paths to saved slide images
        """
        log.info(f"Extracting {len(slides)} unique slides")
        
        slides_dir = output_dir / "slides"
        slides_dir.mkdir(parents=True, exist_ok=True)
        
        slide_paths = []
        
        for slide in slides:
            frame_path = Path(slide["representative_frame"])
            slide_filename = f"slide_{slide['slide_id']:03d}.jpg"
            slide_path = slides_dir / slide_filename
            
            # Copy representative frame
            frame = cv2.imread(str(frame_path))
            cv2.imwrite(str(slide_path), frame)
            
            slide_paths.append(slide_path)
            slide["slide_path"] = str(slide_path)
        
        log.info(f"Saved {len(slide_paths)} slide images to {slides_dir}")
        return slide_paths
    
    def _compute_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Compute color histogram for frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Flattened histogram
        """
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Compute histogram for each channel
        hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [60], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [60], [0, 256])
        
        # Normalize and concatenate
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        hist = np.concatenate([hist_h, hist_s, hist_v])
        return hist
    
    def detect_slide_type(self, frame_path: Path) -> str:
        """Classify slide type (title, content, diagram, code, etc.).
        
        Args:
            frame_path: Path to slide image
            
        Returns:
            Slide type classification
        """
        frame = cv2.imread(str(frame_path))
        
        # Simple heuristics (can be improved with ML)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate text density
        edges = cv2.Canny(gray, 50, 150)
        text_density = np.sum(edges > 0) / edges.size
        
        # Heuristic classification
        if text_density < 0.05:
            return "image_heavy"
        elif text_density > 0.15:
            return "code_or_diagram"
        else:
            return "text_content"
