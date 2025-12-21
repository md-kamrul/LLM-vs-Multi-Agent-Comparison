import json
import os
import random
from pathlib import Path
from textwrap import dedent
from typing import Dict, List

from google.adk.agents.llm_agent import Agent


def _load_dataset_records(dataset_path: Path) -> List[Dict[str, str]]:
    """Return cleaned article-reference pairs from the JSONL dataset."""
    records: List[Dict[str, str]] = []
    try:
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                article = payload.get("article")
                reference = payload.get("reference")
                if not article or not reference:
                    continue
                records.append(
                    {
                        "article": " ".join(article.split()),
                        "reference": " ".join(reference.split()),
                    }
                )
    except FileNotFoundError:
        return []
    return records


def _select_random_examples(records: List[Dict[str, str]], sample_size: int) -> List[Dict[str, str]]:
    if not records:
        return []
    if len(records) <= sample_size:
        random.shuffle(records)
        return records
    return random.sample(records, sample_size)


def _build_few_shot_block(records: List[Dict[str, str]]) -> str:
    if not records:
        return "Dataset not available. Skip few-shot priming and rely on user input alone."
    sections: List[str] = []
    for index, record in enumerate(records, start=1):
        sections.append(
            f"Example {index} Article:\n{record['article']}\nExample {index} Summary:\n{record['reference']}"
        )
    return "\n\n".join(sections)


_MODULE_DIR = Path(__file__).resolve().parent
_DATASET_PATH = _MODULE_DIR.parent / "Dataset" / "cnn_dailymail_test.jsonl"
_SAMPLE_SIZE = 100

seed_value = os.getenv("MODEL_TRAINING_AGENT_SEED")
if seed_value:
    try:
        random.seed(int(seed_value))
    except ValueError:
        random.seed(seed_value)

_few_shot_examples = _select_random_examples(
    _load_dataset_records(_DATASET_PATH),
    _SAMPLE_SIZE,
)


model_training_agent = Agent(
    model="gemini-2.5-flash",
    name="model_training_agent",
    description="An agent that primes the summarization model using few-shot examples.",
    instruction=dedent(
        f"""
                You are a highly skilled model training agent. Your task is to analyze the embedded few-shot examples and produce a concise style guide that downstream summarization agents must follow.

                Context:
                - The dataset is located at "{_DATASET_PATH.relative_to(_MODULE_DIR.parent)}".
                - You see a randomly sampled set of {_SAMPLE_SIZE} article/reference pairs from this dataset below.
                - Each pair has an "article" (source text) and "reference" (gold summary).

                Your goals:
                - Infer the typical summary style used in these examples, including:
                        - Typical summary length in sentences.
                        - Tone (e.g., neutral, factual, headline-like, etc.).
                        - Level of detail vs. brevity.
                        - Preference for lexical overlap vs. paraphrasing.
                - Turn these observations into a compact style guide that another agent can apply when generating summaries for new articles.

                Few-shot training pairs (for you to analyze):

                {_build_few_shot_block(_few_shot_examples)}

                Now, based ONLY on the examples above, output a single JSON object with EXACTLY this structure and nothing else:

                {{
                    "style_guidelines": [
                        "<bullet-point rule 1>",
                        "<bullet-point rule 2>",
                        "<bullet-point rule 3>"
                    ],
                    "length_hint_sentences": <integer>,
                    "lexical_overlap_preference": "high" | "medium" | "low"
                }}

                Rules:
                - Do not explain your reasoning.
                - Do not add any text before or after the JSON object.
                - Ensure the JSON is syntactically valid.
        """
    ),
)