#!/usr/bin/env python3
#%% SETUP & IMPORTS
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Union, Literal

from google import genai
from pydantic import BaseModel, Field

# Allow importing pipeline modules
PIPELINE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PIPELINE_DIR))

from src.classifiers.schemas import MentionTypeResponse
from src.classifiers import MentionTypeClassifier, AdoptionTypeClassifier
from src.classifiers.llm_classifier_v2 import LLMClassifierV2

from visualize_helper import visualize_all, visualize_summary_table
from src.utils.prompt_loader import get_prompt_messages


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env_local() -> None:
    repo_root = REPO_ROOT
    env_path = repo_root / ".env.local"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_local()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


#%% EXAMPLE 1: Recipe extraction with structured output
class Ingredient(BaseModel):
    name: str = Field(description="Name of the ingredient.")
    quantity: str = Field(description="Quantity of the ingredient, including units.")


class Recipe(BaseModel):
    recipe_name: str = Field(description="The name of the recipe.")
    prep_time_minutes: Optional[int] = Field(
        description="Optional time in minutes to prepare the recipe."
    )
    ingredients: List[Ingredient]
    instructions: List[str]



prompt = """
Please extract the recipe from the following text.
The user wants to make delicious chocolate chip cookies.
They need 2 and 1/4 cups of all-purpose flour, 1 teaspoon of baking soda,
1 teaspoon of salt, 1 cup of unsalted butter (softened), 3/4 cup of granulated sugar,
3/4 cup of packed brown sugar, 1 teaspoon of vanilla extract, and 2 large eggs.
For the best part, they'll need 2 cups of semisweet chocolate chips.
First, preheat the oven to 375°F (190°C). Then, in a small bowl, whisk together the flour,
baking soda, and salt. In a large bowl, cream together the butter, granulated sugar, and brown sugar
until light and fluffy. Beat in the vanilla and eggs, one at a time. Gradually beat in the dry
ingredients until just combined. Finally, stir in the chocolate chips. Drop by rounded tablespoons
onto ungreased baking sheets and bake for 9 to 11 minutes.
"""

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=prompt,
    config={
        "response_mime_type": "application/json",
        "response_json_schema": Recipe.model_json_schema(),
    },
)

recipe = Recipe.model_validate_json(response.text)
print(recipe)


#%% EXAMPLE 2: Moderation decision with structured output
class SpamDetails(BaseModel):
    reason: str = Field(description="The reason why the content is considered spam.")
    spam_type: Literal["phishing", "scam", "unsolicited promotion", "other"] = Field(
        description="The type of spam."
    )


class NotSpamDetails(BaseModel):
    summary: str = Field(description="A brief summary of the content.")
    is_safe: bool = Field(description="Whether the content is safe for all audiences.")


class ModerationResult(BaseModel):
    decision: Union[SpamDetails, NotSpamDetails]


prompt = """
Please moderate the following content and provide a decision.
Content: 'Congratulations! You've won a free cruise to the Bahamas. Click here to claim your prize: www.definitely-not-a-scam.com'
"""

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=prompt,
    config={
        "response_mime_type": "application/json",
        "response_json_schema": ModerationResult.model_json_schema(),
    },
)

result = ModerationResult.model_validate_json(response.text)
print(result)


#%% EXAMPLE 3: Mention type classification (aligned with BaseClassifier)
text = """
We are expanding our AI-driven analytics platform across business units.
Our vendor partnership with CloudCo provides the underlying LLM services.
We also discuss risks around data privacy and model misuse.
"""

system_prompt, user_prompt = get_prompt_messages(
    "mention_type",
    reasoning_policy="short",
    firm_name="Example Corp",
    sector="Technology",
    report_year="2024",
    report_section="Strategic Report",
    text=text.strip(),
)

combined_prompt = (
    f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
    if system_prompt
    else user_prompt
)

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=combined_prompt,
    config={
        "response_mime_type": "application/json",
        "response_json_schema": MentionTypeResponse.model_json_schema(),
    },
)

mention_result = MentionTypeResponse.model_validate_json(response.text)
print(mention_result)


#%% EXAMPLE 4: Mention type comparison (baseline vs v2) on first 10 chunks
def load_chunks(path: Path, limit: int | None = None) -> list[dict]:
    chunks: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
            if limit and len(chunks) >= limit:
                break
    return chunks


HUMAN_ANNOTATIONS = REPO_ROOT / "data" / "golden_set" / "human" / "annotations.jsonl"
MAX_CHUNKS = 10

chunks = load_chunks(HUMAN_ANNOTATIONS, limit=MAX_CHUNKS)
print(f"\nLoaded {len(chunks)} chunks for comparison.")

baseline = MentionTypeClassifier(
    run_id="baseline-mention-v1",
    model_name="gemini-3-flash-preview",
    temperature=0.0,
    thinking_budget=0,
    use_openrouter=False,
)
adoption_clf = AdoptionTypeClassifier(
    run_id="baseline-adoption-v1",
    model_name="gemini-3-flash-preview",
    temperature=0.0,
    thinking_budget=0,
    use_openrouter=False,
)
v2 = LLMClassifierV2(
    run_id="mention-v2",
    model_name="gemini-3-flash-preview",
    temperature=0.0,
    thinking_budget=0,
    use_openrouter=False,
)
adoption_clf_v2 = AdoptionTypeClassifier(
    run_id="adoption-v2",
    model_name="gemini-3-flash-preview",
    temperature=0.0,
    thinking_budget=0,
    use_openrouter=False,
)


def extract_mention_types(classification: object) -> list[str]:
    if isinstance(classification, dict) and "mention_types" in classification:
        return [
            str(mt.value) if hasattr(mt, "value") else str(mt)
            for mt in classification["mention_types"]
        ]
    return []


def compare_sets(human: list[str], llm: list[str]) -> str:
    h = set(human)
    l = set(llm)
    if h == l:
        return "EXACT"
    if h & l:
        return "PARTIAL"
    return "DIFF"


def extract_adoption_types(adoption_classification: object, threshold: float = 0.0) -> list[str]:
    if not isinstance(adoption_classification, dict):
        return []
    confidences = adoption_classification.get("adoption_confidences", {}) or {}
    if not isinstance(confidences, dict):
        return []
    return [
        k for k, v in confidences.items()
        if isinstance(v, (int, float)) and v > threshold
    ]


baseline_results: list[tuple[dict, dict]] = []
v2_results: list[tuple[dict, dict]] = []

counts = {
    "baseline": {"EXACT": 0, "PARTIAL": 0, "DIFF": 0},
    "v2": {"EXACT": 0, "PARTIAL": 0, "DIFF": 0},
}

for chunk in chunks:
    metadata = {
        "firm_id": chunk.get("company_id", "Unknown"),
        "firm_name": chunk.get("company_name", "Unknown"),
        "report_year": chunk.get("report_year", 0),
        "sector": "Unknown",
        "report_section": (
            chunk.get("report_sections", ["Unknown"])[0]
            if chunk.get("report_sections")
            else "Unknown"
        ),
    }

    human = chunk.get("mention_types", []) or []

    base_result = baseline.classify(chunk["chunk_text"], metadata)
    v2_result = v2.classify(chunk["chunk_text"], metadata)

    base_labels = extract_mention_types(base_result.classification)
    v2_labels = extract_mention_types(v2_result.classification)

    base_adoption_types: list[str] = []
    v2_adoption_types: list[str] = []
    if "adoption" in base_labels:
        adoption_result = adoption_clf.classify(
            chunk["chunk_text"],
            {**metadata, "mention_types": base_labels},
        )
        base_adoption_types = extract_adoption_types(adoption_result.classification)
    if "adoption" in v2_labels:
        adoption_result_v2 = adoption_clf_v2.classify(
            chunk["chunk_text"],
            {**metadata, "mention_types": v2_labels},
        )
        v2_adoption_types = extract_adoption_types(adoption_result_v2.classification)

    base_match = compare_sets(human, base_labels)
    v2_match = compare_sets(human, v2_labels)

    counts["baseline"][base_match] += 1
    counts["v2"][v2_match] += 1

    baseline_results.append(
        (
            chunk,
            {
                "mention_types": base_labels,
                "adoption_types": base_adoption_types,
                "confidence": base_result.confidence_score,
                "reasoning": base_result.reasoning,
            },
        )
    )
    v2_results.append(
        (
            chunk,
            {
                "mention_types": v2_labels,
                "adoption_types": v2_adoption_types,
                "confidence": v2_result.confidence_score,
                "reasoning": v2_result.reasoning,
            },
        )
    )

    print(
        f"- {chunk['company_name']} ({chunk['report_year']}): "
        f"human={sorted(human)} | v1={sorted(base_labels)} ({base_match}) | "
        f"v2={sorted(v2_labels)} ({v2_match})"
    )

total = len(chunks)
print("\nSummary (mention_types vs human):")
for key in ("baseline", "v2"):
    exact = counts[key]["EXACT"]
    partial = counts[key]["PARTIAL"]
    diff = counts[key]["DIFF"]
    agree = exact + partial
    print(
        f"  {key}: exact={exact} ({exact/total:.0%}), "
        f"partial={partial} ({partial/total:.0%}), "
        f"diff={diff} ({diff/total:.0%}), "
        f"agreement={agree} ({agree/total:.0%})"
    )

print("\nBaseline visualization:")
visualize_summary_table(baseline_results)
visualize_all(baseline_results, show_text=False)

print("\nV2 visualization:")
visualize_summary_table(v2_results)
visualize_all(v2_results, show_text=False)
