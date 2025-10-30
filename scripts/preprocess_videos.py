"""Batch video preprocessing script."""

import argparse
from pathlib import Path
import json
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.video_extractor import VideoExtractor
from src.preprocessing.slide_detector import SlideDetector
from src.preprocessing.transcription import AudioTranscriber
from src.preprocessing.ocr_processor import OCRProcessor
from src.utils.logger import log
from src.utils.config import settings


def process_video(
    video_path: Path,
    lecture_id: str,
    output_dir: Path
):
    """Process a single video file.
    
    Args:
        video_path: Path to video file
        lecture_id: Unique lecture identifier
        output_dir: Output directory for processed data
    """
    log.info(f"\n{'='*60}\nProcessing: {video_path.name}\nLecture ID: {lecture_id}\n{'='*60}")
    
    lecture_output_dir = output_dir / lecture_id
    lecture_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Extract video metadata
    extractor = VideoExtractor(output_dir)
    metadata = extractor.extract_video_info(video_path)
    
    # Step 2: Extract frames
    frames_info = extractor.extract_frames(video_path, lecture_id)
    
    # Step 3: Extract audio
    audio_path = extractor.extract_audio(video_path, lecture_id)
    
    # Step 4: Detect slide changes
    slide_detector = SlideDetector()
    slides = slide_detector.detect_slide_changes(frames_info)
    slide_paths = slide_detector.extract_unique_slides(slides, lecture_output_dir)
    
    # Step 5: Transcribe audio
    transcriber = AudioTranscriber()
    transcription = transcriber.transcribe(audio_path)
    transcriber.save_transcription(
        transcription,
        lecture_output_dir / "audio" / "transcription.json"
    )
    
    # Create text chunks
    text_chunks = transcriber.create_text_chunks(transcription)
    
    # Step 6: OCR on slides
    ocr_processor = OCRProcessor()
    ocr_results = ocr_processor.process_slides_batch(
        slide_paths,
        lecture_output_dir
    )
    
    # Step 7: Save combined metadata
    combined_metadata = {
        "lecture_id": lecture_id,
        "video_info": metadata,
        "frames_count": len(frames_info),
        "slides_count": len(slides),
        "audio_duration": transcription["duration"],
        "text_chunks_count": len(text_chunks),
        "processing_complete": True
    }
    
    with open(lecture_output_dir / "metadata.json", 'w') as f:
        json.dump(combined_metadata, f, indent=2)
    
    # Save frames info
    with open(lecture_output_dir / "frames_info.json", 'w') as f:
        json.dump(frames_info, f, indent=2)
    
    # Save slides info
    with open(lecture_output_dir / "slides_info.json", 'w') as f:
        json.dump(slides, f, indent=2)
    
    # Save text chunks
    with open(lecture_output_dir / "text_chunks.json", 'w') as f:
        json.dump(text_chunks, f, indent=2)
    
    log.info(f"✓ Processing complete for {lecture_id}")
    log.info(f"  - Frames: {len(frames_info)}")
    log.info(f"  - Slides: {len(slides)}")
    log.info(f"  - Duration: {metadata['duration_formatted']}")
    log.info(f"  - Text chunks: {len(text_chunks)}")
    
    return combined_metadata


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Preprocess lecture videos")
    parser.add_argument(
        "--input",
        type=str,
        default="./data/videos",
        help="Input directory containing videos"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Process a single video file"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.video:
        # Process single video
        video_path = Path(args.video)
        if not video_path.exists():
            log.error(f"Video file not found: {video_path}")
            return
        
        lecture_id = video_path.stem
        process_video(video_path, lecture_id, output_dir)
    else:
        # Process all videos in directory
        video_files = list(input_dir.glob("*.mp4")) + \
                     list(input_dir.glob("*.avi")) + \
                     list(input_dir.glob("*.mkv")) + \
                     list(input_dir.glob("*.mov"))
        
        if not video_files:
            log.warning(f"No video files found in {input_dir}")
            log.info("Supported formats: .mp4, .avi, .mkv, .mov")
            return
        
        log.info(f"Found {len(video_files)} videos to process")
        
        for video_path in video_files:
            lecture_id = video_path.stem
            try:
                process_video(video_path, lecture_id, output_dir)
            except Exception as e:
                log.error(f"Error processing {video_path.name}: {e}")
                continue
        
        log.info("\n" + "="*60)
        log.info("✓ All videos processed successfully!")
        log.info("="*60)


if __name__ == "__main__":
    main()
