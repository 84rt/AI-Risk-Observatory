"""Vendor classifier for AIRO pipeline.

Extracts AI vendor/provider mentions from company reports.
"""

from typing import Any, Dict, List, Tuple

from .base_classifier import BaseClassifier


class VendorClassifier(BaseClassifier):
    """AI vendor/provider extraction classifier.

    Extracts mentions of AI vendors and products from reports.

    Target Vendors:
    - OpenAI (GPT, ChatGPT, DALL-E)
    - Microsoft (Azure OpenAI, Copilot, Azure ML)
    - Google (Gemini, Vertex AI, Google Cloud AI)
    - Amazon (Bedrock, SageMaker, AWS AI)
    - Anthropic (Claude)
    - Meta (LLaMA)
    - Internal/Custom solutions
    - Other vendors

    Output:
    - vendors: List of vendors with product details
    - confidence: 0.0-1.0
    - evidence: Quotes for each vendor mention
    """

    CLASSIFIER_TYPE = "vendor"

    def get_prompt(self, text: str, metadata: Dict[str, Any]) -> str:
        """Generate the classification prompt for vendor extraction."""
        firm_name = metadata.get("firm_name", "Unknown Company")
        report_year = metadata.get("report_year", "Unknown")
        sector = metadata.get("sector", "Unknown")

        # Truncate text if too long
        max_chars = 30000
        if len(text) > max_chars:
            text = text[:15000] + "\n\n[...content truncated...]\n\n" + text[-15000:]

        prompt = f"""You are an expert analyst extracting AI VENDOR and PRODUCT mentions from company annual reports.

## CONTEXT
Company: {firm_name}
Sector: {sector}
Report Year: {report_year}

## TASK
Identify all AI vendors, products, and platforms mentioned in this report.

## VENDOR CATEGORIES

### Major AI Providers
- **OpenAI**: GPT, GPT-4, ChatGPT, DALL-E, Whisper
- **Microsoft**: Azure OpenAI, Copilot, Azure Machine Learning, GitHub Copilot
- **Google**: Gemini, Vertex AI, Google Cloud AI, TensorFlow, BERT
- **Amazon**: AWS Bedrock, SageMaker, Amazon AI services
- **Anthropic**: Claude
- **Meta**: LLaMA, PyTorch (framework)
- **IBM**: Watson
- **Salesforce**: Einstein AI

### Cloud AI Platforms
- Microsoft Azure
- Google Cloud Platform
- Amazon Web Services
- Oracle Cloud

### Specialized AI Tools
- DataRobot, H2O.ai, C3.ai
- UiPath, Automation Anywhere (RPA)
- Palantir

### Internal/Custom
- In-house AI systems
- Proprietary models
- Custom-built solutions

## REPORT EXCERPT
\"\"\"
{text}
\"\"\"

## INSTRUCTIONS
1. Read the report carefully for vendor/product mentions
2. Note explicit mentions (e.g., "we use Azure OpenAI")
3. Note implicit mentions (e.g., "our LLM partner" might indicate OpenAI)
4. Capture evidence quotes for each vendor
5. Note whether the company uses internal/custom solutions

## OUTPUT FORMAT
Return a JSON object:
{{
    "vendors_found": true/false,
    "confidence": 0.0-1.0,
    "vendors": [
        {{
            "vendor": "OpenAI",
            "products": ["GPT-4", "ChatGPT"],
            "usage_type": "direct" | "via_azure" | "mentioned_only",
            "evidence": "Exact quote mentioning the vendor"
        }}
    ],
    "internal_ai": {{
        "mentioned": true/false,
        "evidence": "Quote about internal AI development"
    }},
    "reasoning": "Summary of AI vendor landscape for this company"
}}

If NO vendors are explicitly mentioned, set vendors_found to false.
Include ALL vendors mentioned, even if just referenced in passing.

Return ONLY valid JSON, no additional text.
"""
        return prompt

    def parse_result(
        self, response: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Tuple[str, float, List[str], str]:
        """Parse the vendor extraction response.

        Returns:
            Tuple of (primary_label, confidence, evidence_list, reasoning)
        """
        vendors_found = response.get("vendors_found", False)
        confidence = response.get("confidence", 0.5)
        vendors = response.get("vendors", [])
        internal_ai = response.get("internal_ai", {})
        reasoning = response.get("reasoning", "")

        # Build primary label from vendor list
        if vendors_found and vendors:
            vendor_names = [v.get("vendor", "unknown") for v in vendors if isinstance(v, dict)]
            primary_label = ",".join(sorted(set(vendor_names)))
        elif internal_ai.get("mentioned"):
            primary_label = "internal_only"
        else:
            primary_label = "none"

        # Extract evidence from vendors
        evidence = []
        if isinstance(vendors, list):
            for v in vendors:
                if isinstance(v, dict):
                    vendor_name = v.get("vendor", "Unknown")
                    products = v.get("products", [])
                    vendor_evidence = v.get("evidence", "")
                    usage = v.get("usage_type", "")

                    if products:
                        products_str = ", ".join(products)
                        evidence.append(f"[{vendor_name}: {products_str}] {vendor_evidence}")
                    else:
                        evidence.append(f"[{vendor_name}] {vendor_evidence}")

        # Add internal AI evidence
        if internal_ai.get("mentioned") and internal_ai.get("evidence"):
            evidence.append(f"[Internal] {internal_ai['evidence']}")

        return primary_label, confidence, evidence, reasoning



