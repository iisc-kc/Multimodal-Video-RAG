"""Simplified agentic RAG orchestrator."""

from typing import Dict, List, Optional
from pathlib import Path
import re

from ..vector_store.qdrant_client import MultimodalVectorStore
from ..embeddings.text_embedder import TextEmbedder
from ..embeddings.clip_embedder import CLIPEmbedder
from ..inference.ollama_client import OllamaClient
from ..inference.vision_llm import VisionLLM
from ..agent.tools import (
    MultimodalRetriever,
    TemporalRetriever,
    VisualAnalyzer,
    CrossModalLinker
)
from ..agent.prompts import (
    QUERY_ANALYSIS_PROMPT,
    SYNTHESIS_PROMPT,
    TEMPORAL_REASONING_PROMPT
)
from ..vector_store.schemas import ModalityType
from ..utils.logger import log
from ..utils.config import settings


class MultimodalRAGAgent:
    """Agentic multimodal RAG system."""
    
    def __init__(self):
        """Initialize the agent and all components."""
        log.info("Initializing Multimodal RAG Agent")
        
        # Initialize components
        self.vector_store = MultimodalVectorStore()
        self.text_embedder = TextEmbedder()
        self.clip_embedder = CLIPEmbedder()
        self.llm = OllamaClient()
        self.vision_llm = VisionLLM()
        
        # Initialize tools
        self.retriever = MultimodalRetriever(
            self.vector_store,
            self.text_embedder,
            self.clip_embedder
        )
        self.temporal_retriever = TemporalRetriever(self.vector_store)
        self.visual_analyzer = VisualAnalyzer(self.vision_llm)
        self.cross_modal_linker = CrossModalLinker(self.vector_store)
        
        log.info("Agent initialized successfully")
    
    def query(
        self,
        query: str,
        lecture_id: Optional[str] = None,
        verbose: bool = None
    ) -> Dict:
        """Process a user query through the agentic pipeline.
        
        Args:
            query: User question
            lecture_id: Optional lecture filter
            verbose: Show reasoning steps
            
        Returns:
            Dictionary with answer and metadata
        """
        verbose = verbose if verbose is not None else settings.verbose_logging
        
        log.info(f"\n{'='*60}\nProcessing query: {query}\n{'='*60}")
        
        # Step 1: Analyze query to determine modalities and approach
        analysis = self._analyze_query(query)
        
        if verbose:
            log.info(f"Query Analysis:\n{analysis}")
        
        # Step 2: Retrieve from appropriate modalities
        retrieved_data = self._retrieve_multimodal(
            query,
            analysis,
            lecture_id
        )
        
        # Step 3: Enhance with visual analysis if needed
        if analysis.get("visual_analysis", False):
            retrieved_data = self._enhance_with_visual_analysis(
                retrieved_data,
                query
            )
        
        # Step 4: Apply temporal reasoning if needed
        if analysis.get("temporal", False):
            retrieved_data = self._apply_temporal_reasoning(retrieved_data)
        
        # Step 5: Link modalities if needed
        if analysis.get("cross_modal", False):
            retrieved_data = self._link_cross_modal(retrieved_data)
        
        # Step 6: Synthesize final answer
        answer = self._synthesize_answer(
            query,
            retrieved_data
        )
        
        return {
            "query": query,
            "answer": answer,
            "sources": self._format_sources(retrieved_data),
            "analysis": analysis,
            "retrieved_data": retrieved_data
        }
    
    def _analyze_query(self, query: str) -> Dict:
        """Analyze query to determine retrieval strategy.
        
        Args:
            query: User query
            
        Returns:
            Analysis dictionary
        """
        # Use LLM for query analysis
        prompt = QUERY_ANALYSIS_PROMPT.format(query=query)
        
        response = self.llm.generate(
            prompt,
            temperature=0.3,
            max_tokens=300
        )
        
        # Parse response
        analysis = {
            "modalities": [],
            "temporal": False,
            "visual_analysis": False,
            "cross_modal": False,
            "reasoning": ""
        }
        
        # Simple parsing (could be improved with structured output)
        if "MODALITIES:" in response:
            modalities_line = re.search(r"MODALITIES:\s*\[(.*?)\]", response)
            if modalities_line:
                modalities_text = modalities_line.group(1)
                if "text" in modalities_text.lower():
                    analysis["modalities"].append(ModalityType.TEXT)
                if "visual" in modalities_text.lower():
                    analysis["modalities"].append(ModalityType.VISUAL)
                if "slide" in modalities_text.lower():
                    analysis["modalities"].append(ModalityType.SLIDE)
        
        # Fallback: if no modalities detected, use all
        if not analysis["modalities"]:
            analysis["modalities"] = [ModalityType.TEXT, ModalityType.VISUAL, ModalityType.SLIDE]
        
        analysis["temporal"] = "TEMPORAL: yes" in response
        analysis["visual_analysis"] = "VISUAL_ANALYSIS: yes" in response
        analysis["cross_modal"] = "CROSS_MODAL: yes" in response
        
        reasoning_match = re.search(r"REASONING:\s*(.+)", response, re.DOTALL)
        if reasoning_match:
            analysis["reasoning"] = reasoning_match.group(1).strip()
        
        return analysis
    
    def _retrieve_multimodal(
        self,
        query: str,
        analysis: Dict,
        lecture_id: Optional[str]
    ) -> Dict:
        """Retrieve from multiple modalities.
        
        Args:
            query: Search query
            analysis: Query analysis
            lecture_id: Lecture filter
            
        Returns:
            Retrieved data from all modalities
        """
        results = {
            "text": [],
            "visual": [],
            "slides": []
        }
        
        modalities = analysis.get("modalities", [])
        
        if ModalityType.TEXT in modalities:
            results["text"] = self.retriever.retrieve_text(
                query,
                limit=settings.top_k_retrieval,
                lecture_id=lecture_id
            )
            log.info(f"Retrieved {len(results['text'])} text results")
        
        if ModalityType.VISUAL in modalities:
            results["visual"] = self.retriever.retrieve_visual(
                query,
                limit=settings.top_k_retrieval,
                lecture_id=lecture_id
            )
            log.info(f"Retrieved {len(results['visual'])} visual results")
        
        if ModalityType.SLIDE in modalities:
            results["slides"] = self.retriever.retrieve_slides(
                query,
                limit=settings.top_k_retrieval,
                lecture_id=lecture_id
            )
            log.info(f"Retrieved {len(results['slides'])} slide results")
        
        return results
    
    def _enhance_with_visual_analysis(
        self,
        retrieved_data: Dict,
        query: str
    ) -> Dict:
        """Enhance results with VLM analysis.
        
        Args:
            retrieved_data: Retrieved documents
            query: User query
            
        Returns:
            Enhanced data with visual analysis
        """
        visual_analyses = []
        
        # Analyze top visual results
        for result in retrieved_data.get("visual", [])[:2]:
            frame_path = result.get("payload", {}).get("frame_path")
            if frame_path and Path(frame_path).exists():
                analysis = self.visual_analyzer.analyze_frame(
                    Path(frame_path),
                    query
                )
                visual_analyses.append({
                    "frame_path": frame_path,
                    "timestamp": result.get("payload", {}).get("timestamp"),
                    "analysis": analysis
                })
        
        # Analyze top slides
        for result in retrieved_data.get("slides", [])[:2]:
            slide_path = result.get("payload", {}).get("slide_path")
            if slide_path and Path(slide_path).exists():
                analysis = self.visual_analyzer.analyze_frame(
                    Path(slide_path),
                    query
                )
                visual_analyses.append({
                    "slide_path": slide_path,
                    "timestamp": result.get("payload", {}).get("timestamp"),
                    "analysis": analysis
                })
        
        retrieved_data["visual_analysis"] = visual_analyses
        return retrieved_data
    
    def _apply_temporal_reasoning(
        self,
        retrieved_data: Dict
    ) -> Dict:
        """Apply temporal ordering and reasoning.
        
        Args:
            retrieved_data: Retrieved documents
            
        Returns:
            Data with temporal ordering
        """
        # Combine all results and sort by timestamp
        all_results = []
        
        for modality in ["text", "visual", "slides"]:
            for result in retrieved_data.get(modality, []):
                timestamp = result.get("payload", {}).get("timestamp")
                if timestamp is not None:
                    all_results.append(result)
        
        # Sort by timestamp
        all_results.sort(key=lambda x: x.get("payload", {}).get("timestamp", 0))
        
        retrieved_data["temporal_order"] = all_results
        return retrieved_data
    
    def _link_cross_modal(
        self,
        retrieved_data: Dict
    ) -> Dict:
        """Link content across modalities.
        
        Args:
            retrieved_data: Retrieved documents
            
        Returns:
            Data with cross-modal links
        """
        links = []
        
        # For each text result, find related visual
        for text_result in retrieved_data.get("text", [])[:3]:
            visual = self.cross_modal_linker.link_text_to_visual(text_result)
            if visual:
                links.append({
                    "text": text_result,
                    "visual": visual,
                    "type": "text_to_visual"
                })
        
        retrieved_data["cross_modal_links"] = links
        return retrieved_data
    
    def _synthesize_answer(
        self,
        query: str,
        retrieved_data: Dict
    ) -> str:
        """Synthesize final answer from all retrieved data.
        
        Args:
            query: User query
            retrieved_data: All retrieved and processed data
            
        Returns:
            Final answer
        """
        # Format retrieved data for prompt
        text_results = self._format_results(retrieved_data.get("text", []))
        visual_results = self._format_results(retrieved_data.get("visual", []))
        slide_results = self._format_results(retrieved_data.get("slides", []))
        visual_analysis = self._format_visual_analysis(
            retrieved_data.get("visual_analysis", [])
        )
        
        prompt = SYNTHESIS_PROMPT.format(
            query=query,
            text_results=text_results,
            visual_results=visual_results,
            slide_results=slide_results,
            visual_analysis=visual_analysis
        )
        
        answer = self.llm.generate(
            prompt,
            temperature=0.5,
            max_tokens=800
        )
        
        return answer
    
    def _format_results(self, results: List[Dict]) -> str:
        """Format results for prompt."""
        if not results:
            return "No results found."
        
        formatted = []
        for i, result in enumerate(results[:5], 1):
            payload = result.get("payload", {})
            score = result.get("score", 0)
            timestamp = payload.get("timestamp", 0)
            content = payload.get("content", "")
            
            formatted.append(
                f"{i}. [Score: {score:.3f}, Time: {timestamp:.1f}s]\n   {content[:200]}..."
            )
        
        return "\n\n".join(formatted)
    
    def _format_visual_analysis(self, analyses: List[Dict]) -> str:
        """Format visual analysis for prompt."""
        if not analyses:
            return "No visual analysis available."
        
        formatted = []
        for analysis in analyses:
            timestamp = analysis.get("timestamp", 0)
            text = analysis.get("analysis", "")
            formatted.append(f"[Time: {timestamp:.1f}s]\n{text}")
        
        return "\n\n".join(formatted)
    
    def _format_sources(self, retrieved_data: Dict) -> List[Dict]:
        """Format sources for output."""
        sources = []
        
        for modality in ["text", "visual", "slides"]:
            for result in retrieved_data.get(modality, [])[:3]:
                payload = result.get("payload", {})
                sources.append({
                    "modality": modality,
                    "timestamp": payload.get("timestamp"),
                    "score": result.get("score"),
                    "lecture_id": payload.get("lecture_id")
                })
        
        return sources
