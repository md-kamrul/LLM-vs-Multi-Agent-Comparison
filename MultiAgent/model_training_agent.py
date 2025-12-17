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
_SAMPLE_SIZE = 200

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
        You are a highly skilled model training agent. Your task is to prime the model with representative few-shot examples so that downstream summarization agents produce concise, faithful summaries.

        - Load the dataset located at "{_DATASET_PATH.relative_to(_MODULE_DIR.parent)}".
        - Use the "article" field as the source text and the "reference" field as the expected summary. Ignore any "id" field that may appear.
        - Rely on the following randomly sampled set of {_SAMPLE_SIZE} examples to capture the dataset style. Treat them as canonical demonstrations for few-shot prompting and prefer lexical overlap with the provided summaries when reasonable.
        - The dataset is already split for you; prioritize pattern learning over evaluation metrics.
        - Maintain consistency with the tone, length, and factual grounding shown in the examples.

        Few-shot training pairs:

        {_build_few_shot_block(_few_shot_examples)}

    - Use these examples to prime the model for generating high-quality summaries in subsequent tasks.

    - after training the model with few-shot examples, pass control to summary_generator_agent for generating summary. so that it can use the trained model for better summary generation.

    - No output is required from you for this task.
        """
    ),
)