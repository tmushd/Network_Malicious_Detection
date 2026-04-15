#!/usr/bin/env python3
"""Run the Week 12 novelty experiment for the modular malicious URL repo."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from network_malicious_detection.constants import PAPER_METRICS
from network_malicious_detection.data import load_hf_dataset, make_task_splits
from network_malicious_detection.hybrid_fusion import evaluate_binary_hybrid_fusion
from network_malicious_detection.reporting import save_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Week 12 novelty experiment.")
    parser.add_argument("--dataset-name", default="bgspaditya/byt-mal-minpro")
    parser.add_argument("--train-size", type=int, default=6000)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--electra-model", default="bgspaditya/malurl-electra")
    parser.add_argument("--electra-batch-size", type=int, default=64)
    parser.add_argument("--electra-max-length", type=int, default=128)
    parser.add_argument("--electra-device", default="auto")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true", help="Use local HF caches only.")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "week12_novelty"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    data = load_hf_dataset(
        dataset_name=args.dataset_name,
        seed=args.seed,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
    )
    binary = make_task_splits(data, "binary")

    results = evaluate_binary_hybrid_fusion(
        train_df=binary.train,
        val_df=binary.val,
        test_df=binary.test,
        seed=args.seed,
        max_workers=args.max_workers,
        electra_model=args.electra_model,
        batch_size=args.electra_batch_size,
        max_length=args.electra_max_length,
        device=args.electra_device,
        local_files_only=args.offline,
    )

    metrics_rows = []
    for model_name, payload in results.items():
        row = {"task": "binary", "model": model_name.upper()}
        row.update(payload["metrics"])
        metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows).sort_values("model").reset_index(drop=True)
    metrics_df.to_csv(output_dir / "week12_metrics.csv", index=False)

    paper_binary = PAPER_METRICS["binary"]["ELECTRA"]
    comparison_rows = []
    for model_name in ("rf", "electra", "hybrid_fusion"):
        acc = results[model_name]["metrics"]["accuracy"]
        comparison_rows.append(
            {
                "model": model_name.upper(),
                "paper_reference_model": "ELECTRA",
                "paper_accuracy": paper_binary["accuracy"],
                "reproduced_accuracy": acc,
                "improvement_vs_paper": acc - paper_binary["accuracy"],
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "week12_comparison.csv", index=False)

    save_confusion_matrix(
        matrix=results["hybrid_fusion"]["confusion"],
        class_names=["benign", "malicious"],
        title="Hybrid Fusion (Week 12 novelty)",
        output_path=output_dir / "confusion_binary_hybrid_fusion.png",
    )

    report_lines = [
        "# Week 12 Novelty Report",
        "",
        "## What changed",
        "",
        "- Replaced the plain standalone baseline block with a new hybrid fusion block.",
        "- The new block combines Random Forest lexical probabilities with pretrained ELECTRA probabilities.",
        "- A logistic-regression meta-model is trained on the validation split and evaluated on held-out test data.",
        "",
        "## Binary Metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Comparison Against Parent Paper ELECTRA Accuracy",
        "",
        comparison_df.to_markdown(index=False),
        "",
        "## Key takeaway",
        "",
        f"- Hybrid fusion binary accuracy: {results['hybrid_fusion']['metrics']['accuracy']:.4f}",
        f"- Parent paper ELECTRA binary accuracy: {paper_binary['accuracy']:.4f}",
        f"- Improvement vs parent paper binary accuracy: {results['hybrid_fusion']['metrics']['accuracy'] - paper_binary['accuracy']:.4f}",
        "",
    ]
    (output_dir / "week12_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(metrics_df.to_string(index=False))
    print()
    print(comparison_df.to_string(index=False))
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
