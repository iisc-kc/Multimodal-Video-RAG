"""Simplified agentic RAG orchestrator."""

from typing import Dict, List, Optional
from pathlib import Path
import re
import json

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
        
        # Step 4: Apply temporal reasoning only if needed
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
        # Use LLM for query analysis with JSON output
        prompt = QUERY_ANALYSIS_PROMPT.format(query=query)
        
        try:
            response = self.llm.generate(
                prompt,
                temperature=0.3,
                max_tokens=500
            )
            
            # Try to parse JSON response
            # Clean up response (sometimes LLM adds markdown code blocks)
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            analysis = json.loads(response)
            
            # Validate and convert modalities to ModalityType
            modality_map = {
                "text": ModalityType.TEXT,
                "visual": ModalityType.VISUAL,
                "slides": ModalityType.SLIDE,
                "slide": ModalityType.SLIDE
            }
            
            modalities = []
            for mod in analysis.get("modalities", []):
                if mod.lower() in modality_map:
                    modalities.append(modality_map[mod.lower()])
            
            # Fallback: if no valid modalities, search all
            if not modalities:
                modalities = [ModalityType.TEXT, ModalityType.VISUAL, ModalityType.SLIDE]
            
            # Enhancement: Always include slides when searching text for better context
            # Slides often contain key diagrams, definitions, and visual explanations
            if ModalityType.TEXT in modalities and ModalityType.SLIDE not in modalities:
                modalities.append(ModalityType.SLIDE)
                log.info("Added slides to retrieval for comprehensive context")
            
            return {
                "modalities": modalities,
                "temporal": analysis.get("temporal", False),
                "visual_analysis": analysis.get("visual_analysis", False),
                "cross_modal": analysis.get("cross_modal", False),
                "reasoning": analysis.get("reasoning", "")
            }
            
        except (json.JSONDecodeError, KeyError, Exception) as e:
            log.warning(f"Failed to parse LLM analysis as JSON: {e}. Falling back to regex parsing.")
            
            # Fallback to regex parsing (legacy behavior)
            analysis = {
                "modalities": [],
                "temporal": False,
                "visual_analysis": False,
                "cross_modal": False,
                "reasoning": ""
            }
            
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
            if frame_path:
                frame_path = Path(frame_path)
                if not frame_path.exists():
                    log.warning(f"Frame path not found: {frame_path}")
                    continue
                    
                try:
                    analysis = self.visual_analyzer.analyze_frame(
                        frame_path,
                        query
                    )
                    visual_analyses.append({
                        "frame_path": str(frame_path),
                        "timestamp": result.get("payload", {}).get("timestamp"),
                        "analysis": analysis
                    })
                except Exception as e:
                    log.error(f"Error analyzing frame {frame_path}: {e}")
        
        # Analyze top slides
        for result in retrieved_data.get("slides", [])[:2]:
            slide_path = result.get("payload", {}).get("slide_path")
            if slide_path:
                slide_path = Path(slide_path)
                if not slide_path.exists():
                    log.warning(f"Slide path not found: {slide_path}")
                    continue
                    
                try:
                    analysis = self.visual_analyzer.analyze_frame(
                        slide_path,
                        query
                    )
                    visual_analyses.append({
                        "slide_path": str(slide_path),
                        "timestamp": result.get("payload", {}).get("timestamp"),
                        "analysis": analysis
                    })
                except Exception as e:
                    log.error(f"Error analyzing slide {slide_path}: {e}")
        
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
        
        if not all_results:
            log.debug("No results with timestamps for temporal ordering")
            return retrieved_data
        
        # Sort by timestamp
        all_results.sort(key=lambda x: x.get("payload", {}).get("timestamp", 0))
        
        retrieved_data["temporal_order"] = all_results
        log.debug(f"Applied temporal ordering to {len(all_results)} results")
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
        
        try:
            # For each text result, find related visual
            for text_result in retrieved_data.get("text", [])[:3]:
                try:
                    visual = self.cross_modal_linker.link_text_to_visual(text_result)
                    if visual:
                        links.append({
                            "text": text_result,
                            "visual": visual,
                            "type": "text_to_visual"
                        })
                except Exception as e:
                    log.warning(f"Failed to link text to visual: {e}")
                    continue
            
            retrieved_data["cross_modal_links"] = links
            log.debug(f"Created {len(links)} cross-modal links")
        
        except Exception as e:
            log.error(f"Error in cross-modal linking: {e}")
            retrieved_data["cross_modal_links"] = []
        
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
        # Check if we have any results
        total_results = (len(retrieved_data.get("text", [])) + 
                        len(retrieved_data.get("visual", [])) + 
                        len(retrieved_data.get("slides", [])))
        
        if total_results == 0:
            # Get collection stats for debugging
            try:
                stats = self.vector_store.get_collection_stats()
                stats_msg = "\n\n📊 **Collection Status:**\n"
                for coll, info in stats.items():
                    count = info.get('points_count', 0)
                    stats_msg += f"  - {coll}: {count} documents\n"
            except:
                stats_msg = ""
            
            return (
                "I couldn't find any relevant information in the indexed lecture materials "
                "to answer your question.\n\n"
                "**Possible reasons:**\n"
                "1. The topic wasn't covered in the indexed lectures\n"
                "2. Try rephrasing with different keywords or broader terms\n"
                "3. The lectures may not be fully indexed yet\n"
                "4. Lower the similarity threshold in .env (current: "
                f"{settings.similarity_threshold})\n\n"
                "**Tips:**\n"
                "• Use simpler, more general terms\n"
                "• Ask about specific lecture topics you know exist\n"
                "• Check if the video preprocessing completed successfully"
                f"{stats_msg}"
            )
        
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
            max_tokens=1200  # Increased from 800 to allow more comprehensive answers
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
            lecture_id = payload.get("lecture_id", "unknown")
            
            # Include FULL content (not truncated) so LLM has complete context
            # Limit to first 1500 chars if content is extremely long
            content_display = content[:1500] if len(content) > 1500 else content
            
            formatted.append(
                f"{i}. **{lecture_id}** [Score: {score:.3f}, Time: {timestamp:.1f}s]\n   {content_display}"
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
