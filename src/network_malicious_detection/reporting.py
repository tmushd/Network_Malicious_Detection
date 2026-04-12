"""Result serialization and report generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .constants import PAPER_METRICS


def save_confusion_matrix(
    matrix,
    class_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_paper_comparison(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for task in ("multiclass", "binary"):
        for model_name, paper_values in PAPER_METRICS[task].items():
            matched = metrics_df[(metrics_df["task"] == task) & (metrics_df["model"] == model_name)]
            if matched.empty:
                continue
            observed = matched.iloc[0]
            for metric in ("accuracy", "precision", "recall", "f1_weighted"):
                rows.append(
                    {
                        "task": task,
                        "model": model_name,
                        "metric": metric,
                        "paper": paper_values[metric],
                        "reproduced": float(observed[metric]),
                        "abs_diff": abs(float(observed[metric]) - paper_values[metric]),
                    }
                )
    return pd.DataFrame(rows)


def write_markdown_report(
    output_path: Path,
    run_name: str,
    metrics_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        f"# Reproducibility Report: {run_name}",
        "",
        "## Model Metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
    ]

    if not comparison_df.empty:
        lines.extend(
            [
                "## Paper vs Reproduced",
                "",
                comparison_df.to_markdown(index=False),
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")

