"""Agent prompts for different reasoning stages."""

QUERY_ANALYSIS_PROMPT = """You are an AI assistant analyzing user queries about lecture videos.

User Query: {query}

Analyze this query and determine:
1. What information is the user seeking?
2. Which modalities should be searched? (text transcript, visual frames, slides)
3. Does this require temporal reasoning? (e.g., "after", "before", "during")
4. Does this require visual analysis? (e.g., "show me", "diagram", "chart")
5. Does this need cross-modal linking? (e.g., "what was said when showing...")

Provide your analysis in this format:
MODALITIES: [list of: text, visual, slides]
TEMPORAL: yes/no
VISUAL_ANALYSIS: yes/no
CROSS_MODAL: yes/no
REASONING: brief explanation of your analysis
"""

PLANNING_PROMPT = """You are a planning agent for a multimodal RAG system.

User Query: {query}

Query Analysis:
{analysis}

Available Tools:
1. retrieve_text(query, lecture_id, time_range) - Search transcript
2. retrieve_visual(query, lecture_id, time_range) - Search video frames
3. retrieve_slides(query, lecture_id) - Search slides
4. analyze_frame(frame_path, question) - Use VLM to analyze a frame
5. get_temporal_context(lecture_id, timestamp, window) - Get content around time
6. link_modalities(result) - Find related content across modalities

Create a step-by-step plan to answer the query:
STEP 1: [action]
STEP 2: [action]
...
"""

SYNTHESIS_PROMPT = """You are synthesizing information from multiple sources to answer a user's question.

User Query: {query}

Retrieved Information:

TEXT RESULTS:
{text_results}

VISUAL RESULTS:
{visual_results}

SLIDE RESULTS:
{slide_results}

VISUAL ANALYSIS:
{visual_analysis}

Based on all the information above, provide a comprehensive answer to the user's query.
Include:
1. Direct answer to the question
2. Relevant timestamps and source references
3. Visual descriptions if applicable
4. Additional context that might be helpful

Format your response clearly with timestamps and source citations.
"""

TEMPORAL_REASONING_PROMPT = """Given these results ordered by timestamp, identify the progression of concepts:

Results:
{ordered_results}

Describe:
1. How the concept develops over time
2. Key moments or transitions
3. The relationship between different timestamps
"""

VISUAL_DESCRIPTION_PROMPT = """Analyze this lecture slide/frame in detail:

Describe:
1. Main visual elements (diagrams, charts, text)
2. Technical content (equations, code, formulas)
3. The concept being illustrated
4. How this visual aids understanding

Be specific and educational in your description.
"""

FOLLOWUP_CLARIFICATION_PROMPT = """The user's query is ambiguous or could benefit from clarification.

Query: {query}

Potential Issues:
{issues}

Available Information:
{available_info}

Generate a helpful clarifying question or suggest alternatives to help the user refine their query.
"""

RERANKING_PROMPT = """You are re-ranking search results based on relevance to the query.

Query: {query}

Results:
{results}

Rank these results from most to least relevant. Consider:
1. Semantic relevance to the query
2. Completeness of information
3. Temporal context if applicable

Output the result IDs in ranked order: [id1, id2, id3, ...]
"""
