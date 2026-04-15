# Proof Artifacts (Safe Laptop Run)

These files are generated evidence from a low-resource reproducibility run.

## Command used

```bash
python3 scripts/reproduce.py \
  --train-size 5000 \
  --val-size 800 \
  --test-size 1500 \
  --electra-eval-size 500 \
  --max-workers 2 \
  --electra-device cpu \
  --output-dir outputs/repro_submission_safe
```

## Core evidence files

- `metrics_safe_run.csv`: reproduced model metrics.
- `paper_comparison_safe_run.csv`: paper vs reproduced deltas.
- `reproducibility_report_safe_run.md`: markdown report summary.
- `run_metadata_safe_run.json`: exact runtime configuration and timestamp.

## Confusion matrix evidence

- `confusion_multiclass_ELECTRA.png`
- `confusion_multiclass_LGBM.png`
- `confusion_multiclass_RF.png`
- `confusion_multiclass_XGB.png`
- `confusion_binary_ELECTRA.png`
- `confusion_binary_LGBM.png`
- `confusion_binary_RF.png`
- `confusion_binary_XGB.png`

## Week 12 novelty evidence

The repo also includes a Week 12 enhancement run under:

- `week12_novelty/week12_metrics.csv`
- `week12_novelty/week12_comparison.csv`
- `week12_novelty/week12_report.md`
- `week12_novelty/confusion_binary_hybrid_fusion.png`

This novelty run adds a new `HYBRID_FUSION` block that combines Random Forest lexical probabilities with pretrained ELECTRA probabilities.
