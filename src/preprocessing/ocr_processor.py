"""OCR processing for slides and text extraction."""

from pathlib import Path
from typing import List, Dict, Optional
import json

import cv2
import numpy as np

from ..utils.logger import log
from ..utils.config import settings


class OCRProcessor:
    """Extract text from slides using OCR."""
    
    def __init__(self, engine: Optional[str] = None, languages: Optional[List[str]] = None):
        """Initialize OCR processor.
        
        Args:
            engine: OCR engine to use (easyocr, paddleocr, tesseract)
            languages: List of language codes
        """
        self.engine = engine or settings.ocr_engine
        self.languages = languages or settings.ocr_languages
        
        self._load_ocr_engine()
    
    def _load_ocr_engine(self):
        """Load OCR engine."""
        log.info(f"Loading OCR engine: {self.engine}")
        
        if self.engine == "easyocr":
            import easyocr
            self.reader = easyocr.Reader(self.languages, gpu=True)
            log.info("EasyOCR loaded successfully")
            
        elif self.engine == "paddleocr":
            from paddleocr import PaddleOCR
            self.reader = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                show_log=False
            )
            log.info("PaddleOCR loaded successfully")
            
        elif self.engine == "tesseract":
            import pytesseract
            self.reader = pytesseract
            log.info("Tesseract loaded successfully")
            
        else:
            raise ValueError(f"Unsupported OCR engine: {self.engine}")
    
    def extract_text(self, image_path: Path) -> Dict:
        """Extract text from image using OCR.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        log.debug(f"Extracting text from {image_path.name}")
        
        # Read image
        image = cv2.imread(str(image_path))
        
        # Preprocess image for better OCR
        preprocessed = self._preprocess_image(image)
        
        # Extract text based on engine
        if self.engine == "easyocr":
            return self._extract_easyocr(preprocessed, image_path)
        elif self.engine == "paddleocr":
            return self._extract_paddleocr(preprocessed, image_path)
        elif self.engine == "tesseract":
            return self._extract_tesseract(preprocessed, image_path)
    
    def _extract_easyocr(self, image: np.ndarray, image_path: Path) -> Dict:
        """Extract text using EasyOCR.
        
        Args:
            image: Preprocessed image
            image_path: Original image path
            
        Returns:
            Extraction results
        """
        results = self.reader.readtext(image)
        
        text_blocks = []
        full_text = []
        
        for bbox, text, confidence in results:
            text_blocks.append({
                "text": text,
                "confidence": float(confidence),
                "bbox": bbox
            })
            full_text.append(text)
        
        return {
            "image_path": str(image_path),
            "full_text": " ".join(full_text),
            "text_blocks": text_blocks,
            "num_blocks": len(text_blocks)
        }
    
    def _extract_paddleocr(self, image: np.ndarray, image_path: Path) -> Dict:
        """Extract text using PaddleOCR.
        
        Args:
            image: Preprocessed image
            image_path: Original image path
            
        Returns:
            Extraction results
        """
        results = self.reader.ocr(image, cls=True)
        
        text_blocks = []
        full_text = []
        
        if results and results[0]:
            for line in results[0]:
                bbox, (text, confidence) = line
                text_blocks.append({
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": bbox
                })
                full_text.append(text)
        
        return {
            "image_path": str(image_path),
            "full_text": " ".join(full_text),
            "text_blocks": text_blocks,
            "num_blocks": len(text_blocks)
        }
    
    def _extract_tesseract(self, image: np.ndarray, image_path: Path) -> Dict:
        """Extract text using Tesseract.
        
        Args:
            image: Preprocessed image
            image_path: Original image path
            
        Returns:
            Extraction results
        """
        # Get detailed data
        data = self.reader.image_to_data(
            image, 
            output_type=self.reader.Output.DICT
        )
        
        text_blocks = []
        full_text = []
        
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 0:  # Only confident detections
                text = data['text'][i].strip()
                if text:
                    text_blocks.append({
                        "text": text,
                        "confidence": float(data['conf'][i]) / 100.0,
                        "bbox": [
                            data['left'][i],
                            data['top'][i],
                            data['width'][i],
                            data['height'][i]
                        ]
                    })
                    full_text.append(text)
        
        return {
            "image_path": str(image_path),
            "full_text": " ".join(full_text),
            "text_blocks": text_blocks,
            "num_blocks": len(text_blocks)
        }
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        return thresh
    
    def process_slides_batch(
        self,
        slide_paths: List[Path],
        output_dir: Path
    ) -> List[Dict]:
        """Process multiple slides in batch.
        
        Args:
            slide_paths: List of slide image paths
            output_dir: Directory to save OCR results
            
        Returns:
            List of OCR results
        """
        log.info(f"Processing {len(slide_paths)} slides with OCR")
        
        results = []
        
        for slide_path in slide_paths:
            try:
                ocr_result = self.extract_text(slide_path)
                results.append(ocr_result)
                
            except Exception as e:
                log.error(f"Error processing {slide_path.name}: {e}")
                results.append({
                    "image_path": str(slide_path),
                    "full_text": "",
                    "text_blocks": [],
                    "num_blocks": 0,
                    "error": str(e)
                })
        
        # Save results
        output_path = output_dir / "ocr_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        log.info(f"OCR results saved to {output_path}")
        return results
    
    def detect_code_blocks(self, text: str) -> List[str]:
        """Detect code blocks in extracted text.
        
        Args:
            text: Extracted text
            
        Returns:
            List of detected code snippets
        """
        # Simple heuristic: look for common code patterns
        code_indicators = [
            'def ', 'class ', 'import ', 'function',
            'return', 'if ', 'for ', 'while ',
            '{', '}', '()', '=>'
        ]
        
        lines = text.split('\n')
        code_blocks = []
        current_block = []
        in_code = False
        
        for line in lines:
            # Check if line looks like code
            if any(indicator in line for indicator in code_indicators):
                in_code = True
                current_block.append(line)
            elif in_code:
                if line.strip():
                    current_block.append(line)
                else:
                    # End of code block
                    if current_block:
                        code_blocks.append('\n'.join(current_block))
                        current_block = []
                    in_code = False
        
        if current_block:
            code_blocks.append('\n'.join(current_block))
        
        return code_blocks
