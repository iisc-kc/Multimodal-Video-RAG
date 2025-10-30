# System Architecture

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  CLI App     │  │  Gradio UI   │  │   API        │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                   Agentic Control Layer                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           MultimodalRAGAgent                        │   │
│  │  ┌────────────┐  ┌──────────┐  ┌──────────────┐   │   │
│  │  │Query       │  │Planning  │  │Synthesis     │   │   │
│  │  │Analysis    │  │          │  │              │   │   │
│  │  └────────────┘  └──────────┘  └──────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌──────▼───────┐  ┌────────▼──────┐
│   Retrieval    │  │   Analysis   │  │   Reasoning   │
│     Tools      │  │    Tools     │  │     Tools     │
│                │  │              │  │               │
│ • Multimodal   │  │ • Visual     │  │ • Temporal    │
│   Retriever    │  │   Analyzer   │  │   Retriever   │
│ • Text Search  │  │ • VLM        │  │ • Cross-Modal │
│ • Visual Search│  │   Analysis   │  │   Linker      │
└────────┬───────┘  └──────┬───────┘  └───────┬───────┘
         │                 │                  │
┌────────▼─────────────────▼──────────────────▼─────────┐
│               Storage & Inference Layer                │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐ │
│  │   Qdrant     │  │   Ollama    │  │  Embedding   │ │
│  │  (Vectors)   │  │   (LLMs)    │  │   Models     │ │
│  └──────────────┘  └─────────────┘  └──────────────┘ │
└────────────────────────────────────────────────────────┘
                            │
┌────────────────────────────────────────────────────────┐
│                    Data Layer                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐      │
│  │  Frames    │  │  Slides    │  │ Transcripts│      │
│  │  (Images)  │  │  (OCR)     │  │  (Text)    │      │
│  └────────────┘  └────────────┘  └────────────┘      │
└────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow

### 1. Preprocessing Pipeline

```
Video File
    │
    ├─► VideoExtractor ─────► Frames (1 FPS)
    │                          │
    │                          ├─► CLIP Embeddings
    │                          └─► Frame Metadata
    │
    ├─► VideoExtractor ─────► Audio (WAV)
    │                          │
    │                          └─► Whisper ─────► Transcript + Timestamps
    │                                             │
    │                                             └─► Text Chunks
    │
    └─► SlideDetector ──────► Unique Slides
                               │
                               ├─► OCR ─────► Text Content
                               └─► CLIP ────► Visual Embeddings
```

### 2. Indexing Pipeline

```
Processed Data
    │
    ├─► Text Chunks ────► TextEmbedder ────► Qdrant (text_collection)
    │
    ├─► Frames ─────────► CLIPEmbedder ────► Qdrant (visual_collection)
    │
    └─► Slides ─────────┬─► CLIPEmbedder ──┐
                        └─► TextEmbedder ──┴─► Qdrant (slide_collection)
                                                [Multimodal: visual + text]
```

### 3. Query Processing Pipeline

```
User Query
    │
    └─► 1. Query Analysis (LLM)
         │  - Detect modalities needed
         │  - Identify temporal requirements
         │  - Determine if VLM needed
         │
         └─► 2. Retrieval Planning
              │
              ├─► Text Retrieval
              │    └─► Vector Search (text embeddings)
              │
              ├─► Visual Retrieval
              │    └─► Vector Search (CLIP embeddings)
              │
              └─► Slide Retrieval
                   └─► Hybrid Search (visual + text)
              │
              └─► 3. Enhancement
                   │
                   ├─► Visual Analysis (if needed)
                   │    └─► VisionLLM analyzes frames
                   │
                   ├─► Temporal Reasoning (if needed)
                   │    └─► Sort by timestamp, find progression
                   │
                   └─► Cross-Modal Linking (if needed)
                        └─► Link text ↔ visual content
              │
              └─► 4. Synthesis
                   │
                   └─► LLM combines all sources
                        └─► Final Answer + Citations
```

## 🧩 Component Details

### Vector Store Collections

#### 1. Text Collection
- **Vectors**: Text embeddings (768D - nomic-embed-text)
- **Metadata**:
  - `lecture_id`: str
  - `timestamp`: float
  - `timestamp_end`: float
  - `content`: str (actual text)
  - `word_count`: int

#### 2. Visual Collection
- **Vectors**: CLIP embeddings (512D)
- **Metadata**:
  - `lecture_id`: str
  - `timestamp`: float
  - `frame_path`: str
  - `frame_id`: int
  - `is_slide`: bool

#### 3. Slide Collection
- **Vectors**: Named vectors
  - `visual`: CLIP embedding (512D)
  - `text`: Text embedding (768D)
- **Metadata**:
  - `lecture_id`: str
  - `timestamp`: float
  - `timestamp_end`: float
  - `slide_id`: int
  - `slide_path`: str
  - `ocr_text`: str
  - `has_diagram`: bool
  - `has_code`: bool

### Agent Decision Flow

```python
# Simplified pseudocode of agent logic

def query(user_query):
    # Step 1: Analyze
    analysis = llm.analyze_query(user_query)
    # Returns: {modalities, temporal, visual_analysis, cross_modal}
    
    # Step 2: Retrieve
    results = {}
    if 'text' in analysis.modalities:
        results['text'] = retrieve_text(user_query)
    if 'visual' in analysis.modalities:
        results['visual'] = retrieve_visual(user_query)
    if 'slides' in analysis.modalities:
        results['slides'] = retrieve_slides(user_query)
    
    # Step 3: Enhance
    if analysis.visual_analysis:
        for visual_result in results['visual'][:2]:
            frame_path = visual_result.frame_path
            analysis = vision_llm.analyze(frame_path, user_query)
            results['visual_analysis'].append(analysis)
    
    if analysis.temporal:
        results = sort_by_timestamp(results)
        results['progression'] = identify_concept_flow(results)
    
    if analysis.cross_modal:
        for text_result in results['text']:
            linked_visual = find_visual_at_same_time(text_result)
            results['links'].append({text_result, linked_visual})
    
    # Step 4: Synthesize
    answer = llm.synthesize(
        query=user_query,
        text=results['text'],
        visual=results['visual'],
        analysis=results['visual_analysis'],
        links=results['links']
    )
    
    return answer
```

## 📊 Model Specifications

| Model | Purpose | Size | Device | Speed |
|-------|---------|------|--------|-------|
| **nomic-embed-text-v1.5** | Text embeddings | 137M params | GPU/CPU | ~1000 docs/sec |
| **CLIP ViT-L/14** | Vision embeddings | 428M params | GPU | ~100 imgs/sec |
| **Whisper Large-v3** | Transcription | 1.5B params | GPU | ~10x realtime |
| **Llama 3.1 8B** | Main reasoning | 8B params | GPU | ~20 tokens/sec |
| **Llama 3.2 Vision 11B** | Frame analysis | 11B params | GPU | ~15 tokens/sec |

## 🎯 Key Design Decisions

### 1. Why Separate Collections?
- **Performance**: Optimized index per modality
- **Flexibility**: Different embedding dimensions
- **Scalability**: Can scale collections independently

### 2. Why Temporal Metadata?
- Enables "before/after" queries
- Supports concept progression tracking
- Allows temporal filtering

### 3. Why Agentic Approach?
- **Not all queries need all modalities**: Save compute
- **Dynamic tool selection**: Better accuracy
- **Explainability**: Show reasoning steps

### 4. Why Local Models?
- **Privacy**: No data leaves your machine
- **Cost**: No API fees
- **Control**: Customize models as needed

## 🔍 Retrieval Strategies

### Hybrid Search for Slides
```python
# Slides have BOTH visual and text embeddings
# Can search by either modality

# Text-based search (OCR content)
results = search_slides(query_embedding, mode="text")

# Visual-based search (diagram/image)
results = search_slides(query_embedding, mode="visual")
```

### Temporal Window Retrieval
```python
# Get content around a timestamp
context = get_temporal_context(
    lecture_id="lecture_03",
    timestamp=245.3,  # Result timestamp
    window=30  # ±30 seconds
)
# Returns text spoken and slides shown in that window
```

### Cross-Modal Linking
```python
# Find what was SHOWN when something was SAID
text_result = search_text("Q-learning update rule")
visual_result = link_to_visual(text_result)
# Returns slide shown at that timestamp
```

## 🚀 Performance Optimizations

1. **Batch Embedding**: Process images in batches of 32
2. **Caching**: Cache frequently accessed embeddings
3. **Lazy Loading**: Only analyze frames when needed
4. **Quantization**: Use FP16 for faster inference
5. **Frame Sampling**: 1 FPS instead of full frame rate

## 🔮 Future Enhancements

1. **Concept Knowledge Graph**: Build graph of lecture concepts
2. **Reranking**: Use cross-encoder for better ranking
3. **Multi-Lecture Search**: Search across entire course
4. **Auto Quiz Generation**: Generate questions from content
5. **Personalized Learning**: Track user progress
6. **Real-time Processing**: Process videos as they stream
