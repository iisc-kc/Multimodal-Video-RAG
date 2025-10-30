# 🚀 Quick Start Guide

## Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- Docker and Docker Compose
- 20GB+ free disk space

## Installation Steps

### 1. Clone and Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install CLIP from source
pip install git+https://github.com/openai/CLIP.git
```

### 2. Install Ollama and Models

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull llama3.1:8b-instruct-fp16    # Main LLM (4.7GB)
ollama pull llama3.2-vision:11b          # Vision model (7.9GB)
# Alternative: ollama pull qwen2-vl:7b   # Qwen vision model

# Verify installation
ollama list
```

### 3. Start Vector Database

```bash
# Start Qdrant
docker-compose up -d

# Verify Qdrant is running
curl http://localhost:6333/collections
```

### 4. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env if needed (default settings should work)
nano .env
```

## Processing Your Lectures

### Step 1: Add Videos

```bash
# Place your lecture videos in the data/videos directory
mkdir -p data/videos
# Copy your .mp4, .avi, or .mkv files here
```

### Step 2: Preprocess Videos

This extracts frames, audio, transcribes with Whisper, detects slides, and runs OCR.

```bash
# Process all videos
python scripts/preprocess_videos.py --input data/videos --output data/processed

# Or process a single video
python scripts/preprocess_videos.py --video data/videos/lecture_01.mp4
```

**Time estimate:** ~10-15 minutes per hour of video

### Step 3: Build Vector Index

This generates embeddings and populates the vector database.

```bash
python scripts/build_index.py --data data/processed
```

**Time estimate:** ~5-10 minutes per lecture

## Using the System

### Interactive CLI

```bash
python app/cli.py
```

Example session:
```
🔍 Query: What is Q-learning?

⏳ Processing...

📝 Answer:
Q-learning is a model-free reinforcement learning algorithm that learns the value
of actions in states. The Q-learning update rule is:

Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
               a'

📚 Sources:
1. [text] lecture_03 @ 245.3s (score: 0.892)
2. [slides] lecture_03 @ 240.0s (score: 0.876)
```

### Single Query

```bash
python app/cli.py --query "Explain the neural network diagram"
```

### Filter by Lecture

```bash
python app/cli.py --query "What is MCTS?" --lecture lecture_05
```

## Troubleshooting

### Common Issues

**1. Qdrant connection error**
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Restart if needed
docker-compose restart qdrant
```

**2. Ollama model not found**
```bash
# List available models
ollama list

# Pull missing model
ollama pull llama3.1:8b-instruct-fp16
```

**3. CUDA out of memory**
```bash
# Edit .env to use smaller models or CPU
WHISPER_MODEL=medium  # Instead of large-v3
WHISPER_DEVICE=cpu
```

**4. Slow processing**
```bash
# Reduce frame sampling rate in .env
FRAME_SAMPLING_RATE=0.5  # 1 frame every 2 seconds instead of 1/sec
```

### Verify Installation

```bash
# Check Python packages
pip list | grep -E "transformers|qdrant|whisper|langchain"

# Check Ollama
ollama list

# Check Qdrant
curl http://localhost:6333/
```

## Performance Tips

### GPU Acceleration

Make sure CUDA is available:
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

### Batch Processing

Process multiple videos in parallel (if you have enough GPU memory):
```bash
# Process videos in separate terminals
python scripts/preprocess_videos.py --video data/videos/lecture_01.mp4 &
python scripts/preprocess_videos.py --video data/videos/lecture_02.mp4 &
```

### Model Selection

**For limited resources (< 8GB VRAM):**
- WHISPER_MODEL=medium
- LLM_MODEL=llama3.1:8b-instruct-q4_0  # Quantized version
- VISION_MODEL=llava:7b

**For high performance (16GB+ VRAM):**
- WHISPER_MODEL=large-v3
- LLM_MODEL=llama3.1:8b-instruct-fp16
- VISION_MODEL=qwen2-vl:7b

## Next Steps

1. **Test the system** with example queries
2. **Evaluate** using `scripts/evaluate.py`
3. **Customize** prompts in `src/agent/prompts.py`
4. **Extend** with additional tools or features

## Getting Help

- Check logs in `logs/app.log`
- Enable verbose mode: `verbose on` in CLI
- Review configuration in `.env`

Happy querying! 🎉
