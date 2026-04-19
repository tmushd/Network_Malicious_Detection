#!/usr/bin/env python3
"""Print saved run artifacts to the terminal (no re-run required)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _df_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    safe = df.astype(object).where(pd.notnull(df), None)
    return safe.to_dict(orient="records")


def main() -> int:
    parser = argparse.ArgumentParser(description="Print results from outputs/reproducibility to the terminal.")
    parser.add_argument("--output-dir", type=str, default="outputs/reproducibility")
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only JSON (no markdown tables).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (repo_root / args.output_dir).resolve()

    metrics_csv = output_dir / "metrics.csv"
    if not metrics_csv.exists():
        raise SystemExit(f"Missing: {metrics_csv} (run scripts/reproduce.py first)")

    metrics_df = pd.read_csv(metrics_csv)
    comparison_csv = output_dir / "paper_comparison.csv"
    comparison_df = pd.read_csv(comparison_csv) if comparison_csv.exists() else pd.DataFrame()

    if not args.json_only:
        print("=== Metrics Summary ===")
        cols = [c for c in ("task", "model", "accuracy", "precision", "recall", "f1_weighted", "eval_rows", "device") if c in metrics_df.columns]
        print(metrics_df[cols].astype(object).where(pd.notnull(metrics_df[cols]), "").to_markdown(index=False))
        if not comparison_df.empty:
            print("\n=== Paper vs Reproduced (Abs Diff) ===")
            print(comparison_df.to_markdown(index=False))

        print("\n=== Artifacts ===")
        for p in (
            output_dir / "metrics.csv",
            output_dir / "paper_comparison.csv",
            output_dir / "reproducibility_report.md",
            output_dir / "run_metadata.json",
        ):
            if p.exists():
                print(str(p))
        for p in sorted(output_dir.glob("confusion_*.png")):
            print(str(p))

    summary = {
        "artifacts_dir": str(output_dir),
        "artifacts": {
            "metrics_csv": str(output_dir / "metrics.csv"),
            "paper_comparison_csv": str(output_dir / "paper_comparison.csv"),
            "reproducibility_report_md": str(output_dir / "reproducibility_report.md"),
            "run_metadata_json": str(output_dir / "run_metadata.json"),
        },
        "metrics": _df_records(metrics_df),
        "paper_comparison": _df_records(comparison_df),
    }
    print("\n" + json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

