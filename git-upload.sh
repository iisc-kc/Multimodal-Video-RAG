#!/bin/bash
# Script to initialize Git repository and prepare for GitHub upload

echo "🚀 Preparing project for GitHub upload"
echo "======================================="

# Navigate to project directory
cd /tmp/multimodal-video-rag

# Initialize Git repository
echo "Initializing Git repository..."
git init

# Add all files
echo "Adding files to Git..."
git add .

# Create initial commit
echo "Creating initial commit..."
git commit -m "Initial commit: Agentic Multimodal Video RAG System

Features:
- Complete multimodal processing pipeline (video, audio, slides)
- Open-source model stack (CLIP, Whisper, Llama)
- Agentic RAG with temporal reasoning
- Cross-modal linking and visual analysis
- Production-ready code with comprehensive documentation"

echo ""
echo "✅ Git repository initialized!"
echo ""
echo "========================================="
echo "📤 To upload to GitHub, run these commands:"
echo "========================================="
echo ""
echo "1. Create a new repository on GitHub:"
echo "   Go to: https://github.com/new"
echo "   Repository name: multimodal-video-rag"
echo "   Description: Agentic Multimodal Video RAG for Lecture Understanding"
echo "   Keep it Public or Private (your choice)"
echo "   DO NOT initialize with README (we already have one)"
echo ""
echo "2. Then run these commands:"
echo ""
echo "   cd /tmp/multimodal-video-rag"
echo "   git remote add origin https://github.com/YOUR_USERNAME/multimodal-video-rag.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "Replace YOUR_USERNAME with your actual GitHub username"
echo ""
echo "========================================="
echo ""
echo "Alternative: Use GitHub CLI (if installed)"
echo "========================================="
echo ""
echo "If you have GitHub CLI installed (gh):"
echo ""
echo "   cd /tmp/multimodal-video-rag"
echo "   gh repo create multimodal-video-rag --public --source=. --remote=origin"
echo "   git push -u origin main"
echo ""
echo "========================================="
