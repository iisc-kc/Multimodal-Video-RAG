"""Command-line interface for querying the RAG system."""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.agent.graph import MultimodalRAGAgent
from src.utils.logger import log
from src.utils.config import settings


def interactive_mode(agent: MultimodalRAGAgent):
    """Run interactive query mode."""
    print("\n" + "="*80)
    print("🎥 Multimodal Video RAG System - Interactive Mode")
    print("="*80)
    print("\nCommands:")
    print("  - Type your question to query the system")
    print("  - 'exit' or 'quit' to exit")
    print("  - 'verbose on/off' to toggle detailed output")
    print("  - 'help' for examples")
    print("="*80 + "\n")
    
    verbose = settings.verbose_logging
    
    while True:
        try:
            query = input("\n🔍 Query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if query.lower() == 'help':
                show_examples()
                continue
            
            if query.lower().startswith('verbose'):
                if 'on' in query.lower():
                    verbose = True
                    print("✓ Verbose mode enabled")
                elif 'off' in query.lower():
                    verbose = False
                    print("✓ Verbose mode disabled")
                continue
            
            # Process query
            print("\n⏳ Processing...")
            
            result = agent.query(query, verbose=verbose)
            
            # Display results
            print("\n" + "="*80)
            print("📝 Answer:")
            print("="*80)
            print(result["answer"])
            
            print("\n" + "-"*80)
            print("📚 Sources:")
            print("-"*80)
            for i, source in enumerate(result["sources"], 1):
                timestamp = source.get("timestamp", 0)
                modality = source.get("modality", "unknown")
                score = source.get("score", 0)
                print(f"{i}. [{modality}] @ {timestamp:.1f}s (score: {score:.3f})")
            
            if verbose and result.get("analysis"):
                print("\n" + "-"*80)
                print("🧠 Query Analysis:")
                print("-"*80)
                analysis = result["analysis"]
                print(f"Modalities: {[m.value for m in analysis.get('modalities', [])]}")
                print(f"Temporal: {analysis.get('temporal', False)}")
                print(f"Visual Analysis: {analysis.get('visual_analysis', False)}")
                print(f"Cross-Modal: {analysis.get('cross_modal', False)}")
                if analysis.get('reasoning'):
                    print(f"Reasoning: {analysis['reasoning']}")
            
            print("="*80)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            log.error(f"Error processing query: {e}")
            print(f"\n❌ Error: {e}")
            continue


def show_examples():
    """Show example queries."""
    print("\n" + "="*80)
    print("📖 Example Queries:")
    print("="*80)
    
    examples = [
        "What is the definition of Q-learning?",
        "Explain the neural network architecture shown in the diagram",
        "What did the professor say after showing the reward function?",
        "Show me the slide about backpropagation",
        "Compare the MCTS explanation in lecture 3 and lecture 5",
        "What is the UCB formula?",
        "Explain the diagram shown at timestamp 15:30",
        "What concepts were covered before reinforcement learning?"
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
    
    print("="*80)


def single_query_mode(agent: MultimodalRAGAgent, query: str, lecture_id: str = None):
    """Process a single query."""
    log.info(f"Processing query: {query}")
    
    result = agent.query(query, lecture_id=lecture_id, verbose=True)
    
    print("\n" + "="*80)
    print("📝 Answer:")
    print("="*80)
    print(result["answer"])
    
    print("\n" + "-"*80)
    print("📚 Sources:")
    print("-"*80)
    for i, source in enumerate(result["sources"], 1):
        timestamp = source.get("timestamp", 0)
        modality = source.get("modality", "unknown")
        score = source.get("score", 0)
        lecture = source.get("lecture_id", "unknown")
        print(f"{i}. [{modality}] {lecture} @ {timestamp:.1f}s (score: {score:.3f})")
    
    print("="*80)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Multimodal Video RAG System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode:
    python app/cli.py

  Single query:
    python app/cli.py --query "What is reinforcement learning?"

  Filter by lecture:
    python app/cli.py --query "Explain Q-learning" --lecture lecture_03
        """
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to process"
    )
    
    parser.add_argument(
        "--lecture",
        type=str,
        help="Filter results by lecture ID"
    )
    
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Show example queries"
    )
    
    args = parser.parse_args()
    
    if args.examples:
        show_examples()
        return
    
    # Initialize agent
    print("\n🚀 Initializing Multimodal RAG Agent...")
    print("  - Loading embeddings models...")
    print("  - Connecting to vector database...")
    print("  - Setting up LLM inference...")
    
    try:
        agent = MultimodalRAGAgent()
        print("✓ Agent initialized successfully!\n")
    except Exception as e:
        log.error(f"Failed to initialize agent: {e}")
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. Qdrant is running (docker-compose up -d)")
        print("  2. Ollama is running with required models")
        print("  3. Index has been built (python scripts/build_index.py)")
        return
    
    if args.query:
        # Single query mode
        single_query_mode(agent, args.query, args.lecture)
    else:
        # Interactive mode
        interactive_mode(agent)


if __name__ == "__main__":
    main()
