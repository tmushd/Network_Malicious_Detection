# Network_Malicious_Detection (Modular Reproducibility Platform)

This repository is now a **fully runnable and modular implementation platform** for reproducing results from:

**Türk & Kılıçaslan (2025), _Malicious URL Detection with Advanced Machine Learning and Optimization-Supported Deep Learning Models_ (ELECTRA paper).**

It is structured so you can easily replace model blocks (e.g., swap `ELECTRA` with your own method) without rewriting the whole pipeline.

## What this implementation gives you

- Reproducible pipeline for:
  - `multiclass` malicious URL detection
  - `binary` malicious URL detection
- Classical baselines (from lexical URL features):
  - `LGBM`
  - `XGB`
  - `RF`
- ELECTRA reproduction via checkpoint evaluation:
  - default model: `bgspaditya/malurl-electra`
- Automatic proof artifacts:
  - metrics CSV
  - paper-vs-reproduced comparison CSV
  - confusion matrix figures
  - markdown reproducibility report

## Repository structure

```text
scripts/reproduce.py                     # Main CLI for experiments
src/network_malicious_detection/
  constants.py                           # Paper target metrics + label order
  data.py                                # Dataset loading + task label prep
  features.py                            # Modular lexical feature engineering
  models.py                              # Classical model training/eval
  electra_eval.py                        # ELECTRA evaluation module
  metrics.py                             # Shared metric helpers
  reporting.py                           # Confusion matrix + reports
outputs/reproducibility/                 # Generated proof artifacts
```

## Dataset

Default dataset source:
- Hugging Face: `bgspaditya/byt-mal-minpro`

This maps to the same malicious URL benchmark family used in the paper and includes train/val/test splits.

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the default **safe profile** (small slices + low CPU pressure):

```bash
python3 scripts/reproduce.py
```

3. Key outputs:

- `outputs/reproducibility/metrics.csv`
- `outputs/reproducibility/paper_comparison.csv`
- `outputs/reproducibility/reproducibility_report.md`
- `outputs/reproducibility/confusion_*.png`

## Useful run options

```bash
# Only multiclass
python3 scripts/reproduce.py --task multiclass

# Only binary
python3 scripts/reproduce.py --task binary

# Faster run (smaller splits)
python3 scripts/reproduce.py --train-size 80000 --test-size 10000 --electra-eval-size 3000

# Control CPU usage
python3 scripts/reproduce.py --max-workers 2

# Skip ELECTRA if you only want classical baselines
python3 scripts/reproduce.py --skip-electra
```

## Parent paper reference metrics used for comparison

The code compares reproduced metrics against the paper's reported values for:
- `LGBM`
- `XGB`
- `RF`
- `ELECTRA`

for both `multiclass` and `binary` scenarios.

## Notes for coursework deliverable

If your assignment asks for public GitHub evidence:
1. Run `scripts/reproduce.py`.
2. Commit generated artifacts in `outputs/reproducibility/`.
3. Push to a public GitHub repo.
4. Share the repo URL as the reproducibility proof.
