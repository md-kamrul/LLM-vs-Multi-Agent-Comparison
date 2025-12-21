import argparse

from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu


def compute_scores(reference: str, summary: str) -> None:
    """Compute and print BLEU and ROUGE scores for a single pair.

    This is a local, offline helper script. It does not call the multi-agent
    system or any remote APIs; it just compares two strings and prints the
    scores to stdout, so you can see them directly in the VS Code terminal.
    """

    # BLEU (sacrebleu returns 0–100, convert to 0–1)
    bleu_result = corpus_bleu([summary], [[reference]])
    bleu = bleu_result.score / 100.0

    # ROUGE-1/2/L F1 using rouge-score
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge = scorer.score(reference, summary)

    rouge1_f = rouge["rouge1"].fmeasure
    rouge2_f = rouge["rouge2"].fmeasure
    rougeL_f = rouge["rougeL"].fmeasure

    print("=== Evaluation Scores ===")
    print(f"BLEU:    {bleu:.4f}")
    print(f"ROUGE-1: {rouge1_f:.4f}")
    print(f"ROUGE-2: {rouge2_f:.4f}")
    print(f"ROUGE-L: {rougeL_f:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute BLEU and ROUGE scores between a reference summary "
            "(e.g., from the dataset) and a generated summary (e.g., "
            "from the multi-agent system)."
        )
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference summary text (ground truth, e.g., from CNN/DailyMail).",
    )
    parser.add_argument(
        "--summary",
        required=True,
        help="Generated summary text to evaluate.",
    )

    args = parser.parse_args()
    compute_scores(args.reference, args.summary)


if __name__ == "__main__":
    main()
