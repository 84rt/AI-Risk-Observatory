"""Binary harms classifier for AIRO pipeline.

Classifies whether a company report mentions AI-related harms.
"""

from typing import Any, Dict, List, Tuple

from .base_classifier import BaseClassifier


class HarmsClassifier(BaseClassifier):
    """Binary classifier for AI-related harms detection.

    Answers: "Does this report mention AI-related harms?"

    Output:
    - harms_mentioned: true/false
    - confidence: 0.0-1.0
    - evidence: Up to 5 key quotes supporting classification
    """

    CLASSIFIER_TYPE = "harms"

    def get_prompt(self, text: str, metadata: Dict[str, Any]) -> str:
        """Generate the classification prompt for harms detection."""
        firm_name = metadata.get("firm_name", "Unknown Company")
        report_year = metadata.get("report_year", "Unknown")
        sector = metadata.get("sector", "Unknown")

        # Truncate text if too long (keep most relevant parts)
        max_chars = 30000
        if len(text) > max_chars:
            # Keep beginning and end, as AI mentions may be in risk sections
            text = text[:15000] + "\n\n[...content truncated...]\n\n" + text[-15000:]

        prompt = f"""You are an expert analyst for the UK AI Safety Institute, analyzing company annual reports for mentions of AI-related HARMS.

## CONTEXT
Company: {firm_name}
Sector: {sector}
Report Year: {report_year}

## TASK
Analyze this annual report excerpt and determine if it mentions any AI-related HARMS.

AI-related harms include:
- Risks to safety, security, or privacy from AI systems
- Bias, discrimination, or unfair outcomes from AI
- Job displacement or workforce impacts from AI
- Misinformation or content authenticity issues
- Cybersecurity threats enabled by AI
- Operational failures or errors from AI systems
- Regulatory or legal risks from AI use
- Reputational damage from AI incidents
- Environmental impacts from AI infrastructure
- Any negative consequences discussed in relation to AI/ML

## REPORT EXCERPT
\"\"\"
{text}
\"\"\"

## INSTRUCTIONS
1. Carefully read the report excerpt
2. Identify any mentions of AI-related harms (risks, dangers, negative consequences)
3. Extract up to 5 specific quotes that demonstrate harms being mentioned
4. Determine your confidence level

## OUTPUT FORMAT
Return a JSON object with these fields:
{{
    "harms_mentioned": true/false,
    "confidence": 0.0-1.0,
    "evidence": ["quote1", "quote2", ...],
    "reasoning": "Brief explanation of your classification"
}}

If harms ARE mentioned, set harms_mentioned to true and provide evidence quotes.
If NO harms are mentioned, set harms_mentioned to false with empty evidence.

Return ONLY valid JSON, no additional text.
"""
        return prompt

    def parse_result(
        self, response: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Parse the harms classification response.

        Returns:
            Tuple of (primary_label, confidence, evidence_list, reasoning)
        """
        harms_mentioned = response.get("harms_mentioned", False)
        confidence = response.get("confidence", 0.5)
        evidence = response.get("evidence", [])
        reasoning = response.get("reasoning", "")

        # Ensure evidence is a list
        if isinstance(evidence, str):
            evidence = [evidence] if evidence else []
        elif not isinstance(evidence, list):
            evidence = []

        # Primary label is the boolean value as string
        primary_label = "true" if harms_mentioned else "false"

        return primary_label, confidence, evidence, reasoning






