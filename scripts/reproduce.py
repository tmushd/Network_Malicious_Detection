#!/usr/bin/env python3
"""Run modular reproducibility experiments for malicious URL detection."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from network_malicious_detection.data import load_hf_dataset, make_task_splits
from network_malicious_detection.electra_eval import evaluate_pretrained_electra
from network_malicious_detection.features import build_lexical_features
from network_malicious_detection.metrics import dump_json
from network_malicious_detection.models import fit_and_evaluate_classical
from network_malicious_detection.reporting import (
    build_paper_comparison,
    save_confusion_matrix,
    write_markdown_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce malicious URL detection metrics.")
    parser.add_argument("--dataset-name", default="bgspaditya/byt-mal-minpro")
    parser.add_argument("--task", choices=["multiclass", "binary", "both"], default="both")
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--electra-eval-size", type=int, default=500)
    parser.add_argument("--electra-model", default="bgspaditya/malurl-electra")
    parser.add_argument("--electra-batch-size", type=int, default=128)
    parser.add_argument("--electra-max-length", type=int, default=128)
    parser.add_argument("--electra-device", default="auto", help="auto|cpu|mps|cuda")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-electra", action="store_true")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "reproducibility"))
    parser.add_argument(
        "--no-print-json",
        action="store_false",
        dest="print_json",
        help="Disable printing a JSON summary to stdout at the end of the run.",
    )
    parser.set_defaults(print_json=True)
    return parser.parse_args()


def stratified_subset(df: pd.DataFrame, label_col: str, size: int, seed: int) -> pd.DataFrame:
    if size <= 0 or size >= len(df):
        return df.reset_index(drop=True)
    sampled, _ = train_test_split(df, train_size=size, stratify=df[label_col], random_state=seed)
    return sampled.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Avoid stale artifacts when re-running into the same output directory.
    for stale in (
        "metrics.csv",
        "paper_comparison.csv",
        "reproducibility_report.md",
        "run_metadata.json",
    ):
        try:
            (output_dir / stale).unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass
    for stale_path in output_dir.glob("confusion_*.png"):
        try:
            stale_path.unlink()
        except Exception:
            pass

    args.max_workers = max(1, min(int(args.max_workers), 8))
    for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[env_var] = str(args.max_workers)

    print("Loading dataset...")
    data = load_hf_dataset(
        dataset_name=args.dataset_name,
        seed=args.seed,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
    )
    print(
        f"Data loaded | train={len(data.train):,} val={len(data.val):,} test={len(data.test):,}"
    )

    tasks = ["multiclass", "binary"] if args.task == "both" else [args.task]
    rows: list[dict] = []
    run_metadata = {
        "generated_at": datetime.now().isoformat(),
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "train_size": len(data.train),
        "val_size": len(data.val),
        "test_size": len(data.test),
        "tasks": tasks,
        "electra_model": args.electra_model,
    }

    for task in tasks:
        print(f"\n=== Task: {task} ===")
        task_splits = make_task_splits(data, task)

        if not args.skip_baselines:
            print("Building lexical features...")
            x_train = build_lexical_features(task_splits.train)
            x_test = build_lexical_features(task_splits.test)
            y_train = task_splits.train["label"].to_numpy()
            y_test = task_splits.test["label"].to_numpy()

            print("Training/evaluating classical models...")
            classical_results = fit_and_evaluate_classical(
                task=task,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                class_count=len(task_splits.class_names),
                seed=args.seed,
                max_workers=args.max_workers,
            )

            for model_name, result in classical_results.items():
                metrics = result["metrics"]
                rows.append(
                    {
                        "task": task,
                        "model": model_name,
                        "accuracy": metrics["accuracy"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1_weighted": metrics["f1_weighted"],
                        "f1_macro": metrics["f1_macro"],
                        "eval_rows": len(task_splits.test),
                        "device": None,
                    }
                )
                save_confusion_matrix(
                    matrix=result["confusion"],
                    class_names=task_splits.class_names,
                    title=f"{model_name} ({task})",
                    output_path=output_dir / f"confusion_{task}_{model_name}.png",
                )

        if not args.skip_electra:
            electra_test = stratified_subset(
                task_splits.test,
                label_col="label",
                size=args.electra_eval_size,
                seed=args.seed,
            )
            print(
                f"Evaluating ELECTRA checkpoint on {len(electra_test):,} samples..."
            )
            electra_result = evaluate_pretrained_electra(
                test_df=electra_test,
                task=task,
                class_names=task_splits.class_names,
                model_name=args.electra_model,
                batch_size=args.electra_batch_size,
                max_length=args.electra_max_length,
                device=args.electra_device,
            )
            metrics = electra_result["metrics"]
            rows.append(
                {
                    "task": task,
                    "model": "ELECTRA",
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_weighted": metrics["f1_weighted"],
                    "f1_macro": metrics["f1_macro"],
                    "eval_rows": len(electra_test),
                    "device": electra_result["device"],
                }
            )
            save_confusion_matrix(
                matrix=electra_result["confusion"],
                class_names=task_splits.class_names,
                title=f"ELECTRA ({task})",
                output_path=output_dir / f"confusion_{task}_ELECTRA.png",
            )

    metrics_df = pd.DataFrame(rows).sort_values(["task", "model"]).reset_index(drop=True)
    comparison_df = build_paper_comparison(metrics_df)

    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    if not comparison_df.empty:
        comparison_df.to_csv(output_dir / "paper_comparison.csv", index=False)

    run_metadata["runtime_seconds"] = round(time.time() - start, 2)
    run_metadata["metrics_rows"] = len(metrics_df)
    dump_json(output_dir / "run_metadata.json", run_metadata)

    write_markdown_report(
        output_path=output_dir / "reproducibility_report.md",
        run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        metrics_df=metrics_df,
        comparison_df=comparison_df,
    )

    print("\nDone.")
    print(f"Saved metrics: {output_dir / 'metrics.csv'}")
    if not comparison_df.empty:
        print(f"Saved comparison: {output_dir / 'paper_comparison.csv'}")
    print(f"Saved report: {output_dir / 'reproducibility_report.md'}")

    if args.print_json:
        def _df_records(df: pd.DataFrame) -> list[dict]:
            if df.empty:
                return []
            safe = df.astype(object).where(pd.notnull(df), None)
            return safe.to_dict(orient="records")

        summary = {
            "run_metadata": run_metadata,
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


if __name__ == "__main__":
    main()
