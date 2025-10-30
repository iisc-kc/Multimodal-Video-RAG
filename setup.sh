#!/bin/bash
# Setup script for Multimodal Video RAG System

set -e

echo "🚀 Setting up Multimodal Video RAG System"
echo "=========================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "❌ Error: Python 3.9+ required. Found: $python_version"
    exit 1
fi
echo "✓ Python version OK: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "✓ Virtual environment created"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✓ Pip upgraded"

# Install dependencies
echo ""
echo "Installing Python packages..."
echo "(This may take 5-10 minutes)"
pip install -r requirements.txt
echo "✓ Python packages installed"

# Install CLIP from source
echo ""
echo "Installing CLIP from source..."
pip install git+https://github.com/openai/CLIP.git
echo "✓ CLIP installed"

# Create .env file
echo ""
echo "Creating .env configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ .env file created"
else
    echo "⚠ .env file already exists, skipping"
fi

# Check for Docker
echo ""
echo "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first:"
    echo "   https://docs.docker.com/get-docker/"
    exit 1
fi
echo "✓ Docker is installed"

# Check for Ollama
echo ""
echo "Checking Ollama installation..."
if ! command -v ollama &> /dev/null; then
    echo "⚠ Ollama not found. Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✓ Ollama installed"
else
    echo "✓ Ollama is installed"
fi

# Pull Ollama models
echo ""
echo "Downloading LLM models (this may take 10-20 minutes)..."
echo "Pulling Llama 3.1 8B Instruct..."
ollama pull llama3.1:8b-instruct-fp16

echo "Pulling Llama 3.2 Vision 11B..."
ollama pull llama3.2-vision:11b

echo "✓ Models downloaded"

# Start Qdrant
echo ""
echo "Starting Qdrant vector database..."
docker-compose up -d
sleep 5

# Check Qdrant health
if curl -s http://localhost:6333/ > /dev/null; then
    echo "✓ Qdrant is running"
else
    echo "❌ Failed to start Qdrant"
    exit 1
fi

# Create necessary directories
echo ""
echo "Creating data directories..."
mkdir -p data/videos data/processed data/index data/cache logs
echo "✓ Directories created"

# Summary
echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Place lecture videos in data/videos/"
echo ""
echo "3. Process videos:"
echo "   python scripts/preprocess_videos.py"
echo ""
echo "4. Build index:"
echo "   python scripts/build_index.py"
echo ""
echo "5. Start querying:"
echo "   python app/cli.py"
echo ""
echo "For detailed instructions, see QUICKSTART.md"
echo "=========================================="
