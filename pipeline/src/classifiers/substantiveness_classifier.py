"""Substantiveness classifier for AIRO pipeline.

Classifies AI mentions as boilerplate, contextual, or substantive.
"""

from typing import Any, Dict, List, Tuple

from .base_classifier import BaseClassifier


class SubstantivenessClassifier(BaseClassifier):
    """Substantiveness classifier for AI disclosures.

    Categories:
    - boilerplate: Generic legal phrasing applicable to any company
    - contextual: Sector-relevant but non-specific
    - substantive: Named systems, quantified impact, concrete mitigations

    Output:
    - substantiveness: Primary classification
    - confidence: 0.0-1.0
    - evidence: Quotes demonstrating the substantiveness level
    - substantive_ratio: Approximate % of substantive content
    """

    CLASSIFIER_TYPE = "substantiveness"

    def get_prompt(self, text: str, metadata: Dict[str, Any]) -> str:
        """Generate the classification prompt for substantiveness detection."""
        firm_name = metadata.get("firm_name", "Unknown Company")
        report_year = metadata.get("report_year", "Unknown")
        sector = metadata.get("sector", "Unknown")

        # Truncate text if too long
        max_chars = 30000
        if len(text) > max_chars:
            text = text[:15000] + "\n\n[...content truncated...]\n\n" + text[-15000:]

        prompt = f"""You are an expert analyst assessing the SUBSTANTIVENESS of AI disclosures in company annual reports.

## CONTEXT
Company: {firm_name}
Sector: {sector}
Report Year: {report_year}

## TASK
Analyze the AI-related content in this report and classify its overall substantiveness.

## SUBSTANTIVENESS LEVELS

### boilerplate
Generic, templated language that could apply to ANY company:
- "AI may pose risks to our operations"
- "We are monitoring developments in artificial intelligence"
- "AI presents both opportunities and challenges"
- Standard legal disclaimers about technology
- Vague references without specifics
- No named systems, metrics, or actions

### contextual
Sector or company-relevant but still non-specific:
- References to industry-specific AI applications
- General statements about AI strategy
- Mentions of AI governance without specifics
- Discussion of AI in sector context
- Some relevance but lacks concrete details

### substantive
Specific, concrete, and verifiable information:
- Named AI systems or products (e.g., "our GPT-4 powered chatbot")
- Quantified metrics (e.g., "reduced processing time by 40%")
- Specific incidents or case studies
- Detailed mitigation steps with named controls
- Budget allocations or investment figures
- Named vendors or partners
- Measurable outcomes

## REPORT EXCERPT
\"\"\"
{text}
\"\"\"

## INSTRUCTIONS
1. Identify all AI-related content in the report
2. Classify the OVERALL substantiveness level
3. Extract quotes that best demonstrate the substantiveness level
4. Estimate what percentage of AI content is truly substantive

## OUTPUT FORMAT
Return a JSON object:
{{
    "substantiveness": "boilerplate" | "contextual" | "substantive",
    "confidence": 0.0-1.0,
    "evidence": {{
        "boilerplate_examples": ["generic quote..."],
        "contextual_examples": ["sector-relevant quote..."],
        "substantive_examples": ["specific quote with details..."]
    }},
    "substantive_ratio": 0.0-1.0,  // Estimated % of content that is substantive
    "indicators": ["list of substantive indicators found"],
    "reasoning": "Brief explanation of classification"
}}

Classify as:
- "boilerplate" if content is entirely or mostly generic
- "contextual" if content has some relevance but lacks specifics
- "substantive" if content includes concrete, verifiable details

Return ONLY valid JSON, no additional text.
"""
        return prompt

    def parse_result(
        self, response: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Parse the substantiveness classification response.

        Returns:
            Tuple of (primary_label, confidence, evidence_list, reasoning)
        """
        substantiveness = response.get("substantiveness", "boilerplate")
        confidence = response.get("confidence", 0.5)
        evidence_dict = response.get("evidence", {})
        reasoning = response.get("reasoning", "")
        substantive_ratio = response.get("substantive_ratio", 0.0)
        indicators = response.get("indicators", [])

        # Validate substantiveness value
        valid_levels = {"boilerplate", "contextual", "substantive"}
        if substantiveness not in valid_levels:
            substantiveness = "boilerplate"

        primary_label = substantiveness

        # Flatten evidence from dict to list
        evidence = []
        if isinstance(evidence_dict, dict):
            for level, quotes in evidence_dict.items():
                if isinstance(quotes, list):
                    for quote in quotes:
                        evidence.append(f"[{level}] {quote}")
                elif isinstance(quotes, str) and quotes:
                    evidence.append(f"[{level}] {quotes}")
        elif isinstance(evidence_dict, list):
            evidence = evidence_dict

        # Add ratio and indicators to reasoning
        reasoning_parts = [reasoning]
        if substantive_ratio:
            reasoning_parts.append(f"Substantive ratio: {substantive_ratio:.0%}")
        if indicators:
            reasoning_parts.append(f"Indicators: {', '.join(indicators[:5])}")

        full_reasoning = " | ".join(filter(None, reasoning_parts))

        return primary_label, confidence, evidence, full_reasoning






