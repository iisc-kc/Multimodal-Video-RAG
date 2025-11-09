"""
Gradio Web UI for Multimodal Video RAG System
"""

import gradio as gr
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.graph import MultimodalRAGAgent
from src.utils.logger import log
from src.utils.config import settings


class GradioApp:
    def __init__(self):
        """Initialize the Gradio app with RAG agent."""
        self.agent = None
        self.initialize_agent()
    
    def initialize_agent(self):
        """Initialize the RAG agent."""
        try:
            log.info("Initializing Multimodal RAG Agent for Gradio UI")
            self.agent = MultimodalRAGAgent()
            log.info("Agent initialized successfully")
            return True
        except Exception as e:
            log.error(f"Failed to initialize agent: {e}")
            return False
    
    def process_query(self, query: str, lecture_id: str = None, verbose: bool = False):
        """Process a query and return formatted results.
        
        Args:
            query: User query
            lecture_id: Optional lecture ID filter
            verbose: Show detailed analysis
            
        Returns:
            Tuple of (answer, sources, analysis)
        """
        if not self.agent:
            return "❌ Error: Agent not initialized. Please restart the application.", "", ""
        
        if not query.strip():
            return "Please enter a question.", "", ""
        
        try:
            # Process query
            result = self.agent.query(
                query,
                lecture_id=lecture_id if lecture_id else None,
                verbose=True
            )
            
            # Format answer
            answer = result["answer"]
            
            # Format sources
            sources_text = "📚 **Sources:**\n\n"
            for i, source in enumerate(result["sources"], 1):
                timestamp = source.get("timestamp", 0)
                modality = source.get("modality", "unknown")
                score = source.get("score", 0)
                lecture = source.get("lecture_id", "unknown")
                
                # Format time as MM:SS
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"
                
                sources_text += f"{i}. **[{modality}]** `{lecture}` @ {time_str} (score: {score:.3f})\n"
            
            # Format analysis if verbose
            analysis_text = ""
            if verbose and result.get("analysis"):
                analysis = result["analysis"]
                analysis_text = "🧠 **Query Analysis:**\n\n"
                analysis_text += f"• **Modalities:** {[m.value for m in analysis.get('modalities', [])]}\n"
                analysis_text += f"• **Temporal Reasoning:** {analysis.get('temporal', False)}\n"
                analysis_text += f"• **Visual Analysis:** {analysis.get('visual_analysis', False)}\n"
                analysis_text += f"• **Cross-Modal Linking:** {analysis.get('cross_modal', False)}\n"
                if analysis.get('reasoning'):
                    analysis_text += f"\n**Reasoning:** {analysis['reasoning']}\n"
            
            return answer, sources_text, analysis_text
            
        except Exception as e:
            log.error(f"Error processing query: {e}")
            return f"❌ Error: {str(e)}", "", ""


def create_app():
    """Create and configure the Gradio interface."""
    
    app_instance = GradioApp()
    
    with gr.Blocks(
        title="🎥 Multimodal Video RAG",
        theme=gr.themes.Soft(),
        css="""
        .main-container {max-width: 1200px; margin: auto;}
        .answer-box {font-size: 16px; line-height: 1.6;}
        """
    ) as demo:
        
        gr.Markdown("""
        # 🎥 Multimodal Video RAG System
        
        Ask questions about lecture videos. The system searches across:
        - 📝 **Text transcripts** (audio transcription)
        - 🖼️ **Visual frames** (key frames from videos)
        - 📊 **Slides** (extracted presentation slides)
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="E.g., What is prompt engineering? Explain the RAG architecture...",
                    lines=2
                )
            
            with gr.Column(scale=1):
                lecture_filter = gr.Textbox(
                    label="Lecture Filter (Optional)",
                    placeholder="E.g., Lec_04",
                    lines=1
                )
        
        with gr.Row():
            verbose_check = gr.Checkbox(label="Show detailed analysis", value=False)
            submit_btn = gr.Button("🔍 Search", variant="primary", scale=1)
        
        gr.Markdown("---")
        
        # Output sections
        answer_output = gr.Markdown(
            label="Answer",
            elem_classes=["answer-box"]
        )
        
        with gr.Accordion("📚 Sources", open=True):
            sources_output = gr.Markdown()
        
        with gr.Accordion("🧠 Query Analysis", open=False):
            analysis_output = gr.Markdown()
        
        # Example queries
        gr.Markdown("### 💡 Example Questions:")
        gr.Examples(
            examples=[
                ["What is prompt engineering?"],
                ["Explain the RAG architecture"],
                ["What is fine-tuning?"],
                ["Explain agentic AI"],
                ["What are LLM training methods?"],
                ["What is AI safety?"],
                ["Explain LLMOps"],
            ],
            inputs=query_input,
        )
        
        # Event handlers
        submit_btn.click(
            fn=app_instance.process_query,
            inputs=[query_input, lecture_filter, verbose_check],
            outputs=[answer_output, sources_output, analysis_output]
        )
        
        query_input.submit(
            fn=app_instance.process_query,
            inputs=[query_input, lecture_filter, verbose_check],
            outputs=[answer_output, sources_output, analysis_output]
        )
        
        # Footer
        gr.Markdown("""
        ---
        **System Status:**
        - ✅ Models: CLIP, Whisper, Llama 3.1/3.2, Nomic Embeddings
        - ✅ Vector DB: Qdrant (localhost:6333)
        - ✅ LLM: Ollama (localhost:11434)
        
        *Built with 100% open-source models*
        """)
    
    return demo


def main():
    """Launch the Gradio app."""
    demo = create_app()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=settings.gradio_port,
        share=settings.gradio_share,
        show_error=True
    )


if __name__ == "__main__":
    main()
