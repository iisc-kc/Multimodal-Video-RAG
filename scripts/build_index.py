"""Build vector index from preprocessed data."""

import argparse
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.vector_store.qdrant_client import MultimodalVectorStore
from src.vector_store.schemas import TextChunk, FrameDocument, SlideDocument, ModalityType
from src.embeddings.text_embedder import TextEmbedder
from src.embeddings.clip_embedder import CLIPEmbedder
from src.utils.logger import log
from src.utils.config import settings


def build_index(processed_dir: Path):
    """Build vector index from processed lecture data.
    
    Args:
        processed_dir: Directory containing processed lecture data
    """
    log.info(f"Building index from {processed_dir}")
    
    # Initialize components
    vector_store = MultimodalVectorStore()
    text_embedder = TextEmbedder()
    clip_embedder = CLIPEmbedder()
    
    # Create collections
    vector_store.create_collections(
        text_dim=text_embedder.embedding_dim,
        visual_dim=clip_embedder.embedding_dim
    )
    
    # Get all lecture directories
    lecture_dirs = [d for d in processed_dir.iterdir() if d.is_dir()]
    
    if not lecture_dirs:
        log.warning(f"No lecture directories found in {processed_dir}")
        return
    
    log.info(f"Found {len(lecture_dirs)} lectures to index")
    
    for lecture_dir in tqdm(lecture_dirs, desc="Indexing lectures"):
        lecture_id = lecture_dir.name
        
        try:
            # Load metadata
            metadata_path = lecture_dir / "metadata.json"
            if not metadata_path.exists():
                log.warning(f"No metadata found for {lecture_id}, skipping")
                continue
            
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            # Index text chunks
            text_chunks_path = lecture_dir / "text_chunks.json"
            if text_chunks_path.exists():
                index_text_chunks(
                    text_chunks_path,
                    lecture_id,
                    vector_store,
                    text_embedder
                )
            
            # Index frames
            frames_info_path = lecture_dir / "frames_info.json"
            if frames_info_path.exists():
                index_frames(
                    frames_info_path,
                    lecture_id,
                    vector_store,
                    clip_embedder
                )
            
            # Index slides
            slides_info_path = lecture_dir / "slides_info.json"
            ocr_results_path = lecture_dir / "ocr_results.json"
            if slides_info_path.exists() and ocr_results_path.exists():
                index_slides(
                    slides_info_path,
                    ocr_results_path,
                    lecture_id,
                    vector_store,
                    text_embedder,
                    clip_embedder
                )
            
            log.info(f"✓ Indexed {lecture_id}")
            
        except Exception as e:
            log.error(f"Error indexing {lecture_id}: {e}")
            continue
    
    # Print statistics
    stats = vector_store.get_collection_stats()
    log.info("\n" + "="*60)
    log.info("Index Statistics:")
    for collection, info in stats.items():
        log.info(f"  {collection}: {info.get('points_count', 0)} documents")
    log.info("="*60)


def index_text_chunks(
    chunks_path: Path,
    lecture_id: str,
    vector_store: MultimodalVectorStore,
    text_embedder: TextEmbedder
):
    """Index text chunks."""
    with open(chunks_path) as f:
        chunks_data = json.load(f)
    
    # Create document objects
    documents = []
    for i, chunk in enumerate(chunks_data):
        doc = TextChunk(
            id=f"{lecture_id}_text_{i}",
            lecture_id=lecture_id,
            content=chunk["text"],
            timestamp=chunk.get("start", 0),
            timestamp_end=chunk.get("end", 0),
            word_count=chunk.get("word_count", 0)
        )
        documents.append(doc)
    
    # Generate embeddings
    texts = [doc.content for doc in documents]
    embeddings = text_embedder.embed_texts_batch(texts, show_progress=False)
    
    # Add to vector store
    vector_store.add_text_documents(documents, embeddings)
    
    log.debug(f"  Added {len(documents)} text chunks")


def index_frames(
    frames_info_path: Path,
    lecture_id: str,
    vector_store: MultimodalVectorStore,
    clip_embedder: CLIPEmbedder
):
    """Index video frames."""
    with open(frames_info_path) as f:
        frames_data = json.load(f)
    
    # Create document objects
    documents = []
    frame_paths = []
    
    for frame_info in frames_data:
        frame_path = Path(frame_info["path"])
        if not frame_path.exists():
            continue
        
        doc = FrameDocument(
            id=f"{lecture_id}_frame_{frame_info['frame_id']}",
            lecture_id=lecture_id,
            content=f"Frame at {frame_info['timestamp_formatted']}",
            timestamp=frame_info["timestamp"],
            frame_path=str(frame_path),
            frame_id=frame_info["frame_id"]
        )
        documents.append(doc)
        frame_paths.append(frame_path)
    
    if not documents:
        return
    
    # Generate embeddings
    embeddings = clip_embedder.embed_images_batch(frame_paths, batch_size=32)
    
    # Add to vector store
    vector_store.add_visual_documents(documents, embeddings)
    
    log.debug(f"  Added {len(documents)} frames")


def index_slides(
    slides_info_path: Path,
    ocr_results_path: Path,
    lecture_id: str,
    vector_store: MultimodalVectorStore,
    text_embedder: TextEmbedder,
    clip_embedder: CLIPEmbedder
):
    """Index slides with both visual and text embeddings."""
    with open(slides_info_path) as f:
        slides_data = json.load(f)
    
    with open(ocr_results_path) as f:
        ocr_data = json.load(f)
    
    # Create document objects
    documents = []
    slide_paths = []
    ocr_texts = []
    
    for slide_info, ocr_result in zip(slides_data, ocr_data):
        slide_path = Path(slide_info.get("slide_path", ""))
        if not slide_path.exists():
            continue
        
        ocr_text = ocr_result.get("full_text", "")
        
        doc = SlideDocument(
            id=f"{lecture_id}_slide_{slide_info['slide_id']}",
            lecture_id=lecture_id,
            content=f"Slide {slide_info['slide_id']}: {ocr_text[:100]}",
            timestamp=slide_info["start_timestamp"],
            timestamp_end=slide_info["end_timestamp"],
            slide_id=slide_info["slide_id"],
            slide_path=str(slide_path),
            ocr_text=ocr_text
        )
        documents.append(doc)
        slide_paths.append(slide_path)
        ocr_texts.append(ocr_text if ocr_text else "empty slide")
    
    if not documents:
        return
    
    # Generate embeddings
    visual_embeddings = clip_embedder.embed_images_batch(slide_paths, batch_size=16)
    text_embeddings = text_embedder.embed_texts_batch(ocr_texts, show_progress=False)
    
    # Add to vector store
    vector_store.add_slide_documents(documents, visual_embeddings, text_embeddings)
    
    log.debug(f"  Added {len(documents)} slides")


def main():
    """Main indexing function."""
    parser = argparse.ArgumentParser(description="Build vector index from processed data")
    parser.add_argument(
        "--data",
        type=str,
        default="./data/processed",
        help="Directory containing processed lecture data"
    )
    
    args = parser.parse_args()
    
    processed_dir = Path(args.data)
    
    if not processed_dir.exists():
        log.error(f"Processed data directory not found: {processed_dir}")
        log.info("Run preprocess_videos.py first to process videos")
        return
    
    build_index(processed_dir)
    
    log.info("\n✓ Index building complete!")
    log.info("You can now query the system using app/cli.py")


if __name__ == "__main__":
    main()
