# 🎥 Agentic Multimodal Video RAG System

A production-ready agentic RAG system for video lectures using **100% open-source models**.

## 🌟 Features

- **Multimodal Processing**: Video frames (CLIP), audio (Whisper), slides (OCR)
- **Temporal Reasoning**: Understand concept progression across lectures
- **Agentic Control**: LangGraph-based intelligent retrieval planning
- **Slide-Video Alignment**: Match slides with spoken explanations
- **Local Inference**: No API costs, full privacy
- **Diagram Understanding**: Vision-language models for technical diagrams

## 🧠 Open-Source Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| **Vision Encoder** | `CLIP ViT-L/14` | Frame embeddings |
| **Audio Transcription** | `Whisper Large-v3` | Speech-to-text |
| **Text Embeddings** | `nomic-embed-text-v1.5` | Text chunks |
| **Vision-Language** | `Llama-3.2-11B-Vision` or `Qwen2-VL-7B` | Frame analysis |
| **Reasoning LLM** | `Llama-3.1-8B-Instruct` | Agent orchestration |
| **OCR** | `PaddleOCR` or `EasyOCR` | Slide text extraction |
| **Vector DB** | `Qdrant` | Multimodal storage |

## 📁 Project Structure

```
multimodal-video-rag/
├── src/
│   ├── preprocessing/
│   │   ├── video_extractor.py      # Frame & audio extraction
│   │   ├── slide_detector.py       # Slide change detection
│   │   ├── transcription.py        # Whisper integration
│   │   └── ocr_processor.py        # OCR for slides
│   ├── embeddings/
│   │   ├── clip_embedder.py        # CLIP vision embeddings
│   │   ├── text_embedder.py        # Text embeddings
│   │   └── multimodal_fusion.py    # Cross-modal alignment
│   ├── vector_store/
│   │   ├── qdrant_client.py        # Vector DB operations
│   │   └── schemas.py              # Data models
│   ├── agent/
│   │   ├── tools.py                # RAG tools (retrieval, analysis)
│   │   ├── graph.py                # LangGraph state machine
│   │   └── prompts.py              # Agent prompts
│   ├── inference/
│   │   ├── ollama_client.py        # Ollama integration
│   │   └── vision_llm.py           # Vision-LLM wrapper
│   └── utils/
│       ├── config.py               # Configuration management
│       └── logger.py               # Logging setup
├── scripts/
│   ├── preprocess_videos.py       # Batch video processing
│   ├── build_index.py             # Build vector index
│   └── evaluate.py                # Evaluation metrics
├── app/
│   ├── cli.py                     # Command-line interface
│   └── gradio_app.py              # Web UI (optional)
├── data/
│   ├── videos/                    # Input videos
│   ├── processed/                 # Extracted data
│   └── index/                     # Vector DB storage
├── tests/
│   └── test_queries.json          # Evaluation queries
├── requirements.txt
├── docker-compose.yml             # Qdrant + Ollama
├── .env.example
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repo-url>
cd multimodal-video-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama (for local LLM inference)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b-instruct-fp16
ollama pull llama3.2-vision:11b  # Or qwen2-vl:7b
```

### 2. Start Services

```bash
# Start Qdrant vector database
docker-compose up -d

# Verify Ollama is running
ollama list
```

### 3. Process Videos

```bash
# Place your lecture videos in data/videos/
# Then run preprocessing
python scripts/preprocess_videos.py --input data/videos --output data/processed

# Build vector index
python scripts/build_index.py --data data/processed
```

### 4. Query the System

```bash
# CLI interface
python app/cli.py

# Or launch web UI
python app/gradio_app.py
```

## 💻 Usage Examples

```python
from src.agent.graph import MultimodalRAGAgent

agent = MultimodalRAGAgent()

# Example queries
queries = [
    "What is the definition of Q-learning?",
    "Show me the diagram where he explained the neural network architecture",
    "What did the professor say after showing the reward function graph?",
    "Compare the MCTS explanation in lecture 3 vs lecture 5"
]

for query in queries:
    response = agent.query(query)
    print(f"Q: {query}")
    print(f"A: {response['answer']}")
    print(f"Sources: {response['sources']}")
    print("-" * 80)
```

## ⚙️ Configuration

Edit `.env` file:

```bash
# Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Ollama
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=llama3.1:8b-instruct-fp16
VISION_MODEL=llama3.2-vision:11b

# Processing
FRAME_SAMPLING_RATE=1  # frames per second
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

## 📊 Evaluation

```bash
# Run evaluation on test queries
python scripts/evaluate.py --test-file tests/test_queries.json

# Metrics:
# - Retrieval Precision@K
# - Temporal Accuracy
# - Modality Selection Accuracy
# - Answer Quality (LLM-as-judge)
```

## 🎯 Advanced Features

- **Temporal Knowledge Graph**: Concept progression tracking
- **Auto Quiz Generation**: Generate questions from lectures
- **Personalized Summaries**: Based on user interactions
- **Multi-Lecture Reasoning**: Cross-lecture concept comparison

## 🐳 Docker Deployment

```bash
# Build and run entire stack
docker-compose up --build

# Access at http://localhost:7860
```

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md first.

## 📚 Citation

If you use this project, please cite:
```bibtex
@software{multimodal_video_rag_2025,
  author = {Your Name},
  title = {Agentic Multimodal Video RAG System},
  year = {2025},
  url = {https://github.com/yourusername/multimodal-video-rag}
}
```

## 🙏 Acknowledgments

- OpenAI CLIP
- OpenAI Whisper
- Meta Llama
- Alibaba Qwen
- LangChain/LangGraph
- Qdrant

---

**Built with ❤️ for AI Education**
