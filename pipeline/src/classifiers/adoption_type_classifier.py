"""Adoption type classifier for AIRO pipeline.

Classifies AI adoption mentions into three categories:
- non_llm: Traditional AI/ML (computer vision, predictive analytics)
- llm: Large Language Models (GPT, chatbots, text generation)
- agentic: Autonomous AI agents
"""

from typing import Any, Dict, List, Tuple

from .base_classifier import BaseClassifier


class AdoptionTypeClassifier(BaseClassifier):
    """3-category classifier for AI adoption types.

    Categories:
    - non_llm: Traditional AI/ML (computer vision, predictive analytics,
               fraud detection, recommendation systems)
    - llm: Large Language Models (GPT, BERT, chatbots, text generation, NLP)
    - agentic: Agentic AI (autonomous agents, self-directed systems)

    Output:
    - adoption_types: List of detected types
    - confidence: 0.0-1.0
    - evidence: Quotes for each type
    - vendors: List of AI vendors/products mentioned
    """

    CLASSIFIER_TYPE = "adoption"

    def get_prompt(self, text: str, metadata: Dict[str, Any]) -> str:
        """Generate the classification prompt for adoption type detection."""
        firm_name = metadata.get("firm_name", "Unknown Company")
        report_year = metadata.get("report_year", "Unknown")
        sector = metadata.get("sector", "Unknown")

        # Truncate text if too long
        max_chars = 30000
        if len(text) > max_chars:
            text = text[:15000] + "\n\n[...content truncated...]\n\n" + text[-15000:]

        prompt = f"""You are an expert analyst classifying the types of AI technologies mentioned in company annual reports.

## CONTEXT
Company: {firm_name}
Sector: {sector}
Report Year: {report_year}

## TASK
Analyze this annual report and classify the types of AI adoption mentioned.

## AI ADOPTION CATEGORIES

### non_llm (Traditional AI/ML)
Technologies that do NOT involve large language models:
- Computer vision and image recognition
- Predictive analytics and forecasting
- Recommendation systems
- Fraud detection and anomaly detection
- Robotic process automation (RPA)
- Machine learning for optimization
- Pattern recognition
- Algorithmic trading (non-LLM)

### llm (Large Language Models)
Technologies based on language models:
- GPT, ChatGPT, GPT-4
- BERT and transformer-based models
- Chatbots and conversational AI
- Text generation and summarization
- Natural Language Processing (NLP)
- Generative AI for text/content
- AI assistants (Copilot, Claude, Gemini)

### agentic (Agentic AI)
Autonomous AI systems:
- AI agents that take actions independently
- Self-directed automation
- Autonomous decision-making systems
- Multi-agent systems
- AI systems with planning and execution
- Auto-GPT or similar autonomous frameworks

## REPORT EXCERPT
\"\"\"
{text}
\"\"\"

## INSTRUCTIONS
1. Read the report and identify all AI adoption mentions
2. Classify each mention into one or more categories
3. Extract evidence quotes for each category found
4. Note any AI vendors or products mentioned (OpenAI, Microsoft Azure, Google, AWS, etc.)
5. A company may have multiple adoption types - list all that apply

## OUTPUT FORMAT
Return a JSON object:
{{
    "adoption_types": ["non_llm", "llm", "agentic"],  // List of detected types
    "confidence": 0.0-1.0,
    "evidence": {{
        "non_llm": ["quote about traditional ML..."],
        "llm": ["quote about language models..."],
        "agentic": ["quote about autonomous agents..."]
    }},
    "vendors": ["OpenAI", "Microsoft", "Google"],  // AI vendors mentioned
    "reasoning": "Brief explanation"
}}

If a category is not mentioned, omit it from adoption_types and evidence.
If no AI adoption is mentioned at all, return empty adoption_types.

Return ONLY valid JSON, no additional text.
"""
        return prompt

    def parse_result(
        self, response: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Parse the adoption type classification response.

        Returns:
            Tuple of (primary_label, confidence, evidence_list, reasoning)
        """
        adoption_types = response.get("adoption_types", [])
        confidence = response.get("confidence", 0.5)
        evidence_dict = response.get("evidence", {})
        reasoning = response.get("reasoning", "")
        vendors = response.get("vendors", [])

        # Ensure adoption_types is a list
        if isinstance(adoption_types, str):
            adoption_types = [adoption_types] if adoption_types else []
        elif not isinstance(adoption_types, list):
            adoption_types = []

        # Primary label is the list joined, or "none" if empty
        if adoption_types:
            primary_label = ",".join(sorted(adoption_types))
        else:
            primary_label = "none"

        # Flatten evidence from dict to list
        evidence = []
        if isinstance(evidence_dict, dict):
            for category, quotes in evidence_dict.items():
                if isinstance(quotes, list):
                    for quote in quotes:
                        evidence.append(f"[{category}] {quote}")
                elif isinstance(quotes, str):
                    evidence.append(f"[{category}] {quotes}")
        elif isinstance(evidence_dict, list):
            evidence = evidence_dict

        # Add vendor info to reasoning if present
        if vendors:
            reasoning = f"{reasoning} Vendors mentioned: {', '.join(vendors)}"

        return primary_label, confidence, evidence, reasoning



