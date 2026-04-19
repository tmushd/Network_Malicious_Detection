# Network Malicious Detection

## Repo

This repository is a reproducible experimentation pipeline for malicious URL detection, designed to make model evaluation and paper-style reporting easy to run, audit, and share.

It reproduces and compares against results from:
Turk & Kilicaslan (2025), *Malicious URL Detection with Advanced Machine Learning and Optimization-Supported Deep Learning Models*.

### At A Glance

| Area | What this project does |
| --- | --- |
| Problem | Detects malicious URLs for security triage and automation |
| Tasks | Binary classification (benign vs malicious) and multiclass classification (malicious categories) |
| Dataset | Downloads a benchmark dataset on first run from Hugging Face (`bgspaditya/byt-mal-minpro`) |
| Models | Classical baselines on lexical features (`RF`, `XGB`, `LGBM`) and ELECTRA checkpoint evaluation (`bgspaditya/malurl-electra`) |
| Reproducibility | Single CLI entrypoint with fixed seed + controlled CPU usage |
| Outputs | Metrics CSVs, confusion matrix images, and a markdown reproducibility report |

Engineering highlights:
- Single entrypoint CLI: `scripts/reproduce.py` (parameterized sizes, seed, CPU limits, and model switches)
- Modular code layout under `src/network_malicious_detection/` (data, features, models, metrics, reporting)
- Proof-first outputs in `outputs/reproducibility/` so results can be reviewed without rerunning training

### What You Get Out of This Repo

This is optimized as a portfolio-ready reproducibility project:
- Runs end-to-end without manual data wrangling (downloads dataset + pretrained model weights automatically)
- Produces “paper appendix” artifacts (tables + plots + a report) that are easy to attach to a write-up
- Makes it easy to swap components (e.g., replace the checkpoint or baseline model code) without breaking the pipeline shape

### Key Artifacts

After a run, `outputs/reproducibility/` contains:
- `metrics.csv`: per-task metrics for each model
- `paper_comparison.csv`: paper vs reproduced comparison table (where applicable)
- `reproducibility_report.md`: markdown summary for quick review
- `confusion_*.png`: confusion matrices for each model/task

Optional extension included (no extra setup required):
- Week 12 hybrid fusion module (`src/network_malicious_detection/hybrid_fusion.py`) with proof artifacts under `proof/`

## Run

### 0) Install prerequisites (macOS / Windows)

You need:
- Git
- Python 3.10+ (with `pip` and `venv`)

macOS:
- Install Git (via Xcode Command Line Tools): `xcode-select --install`
- Install Python 3.10+: download and install from https://www.python.org/downloads/

Windows:
- Install Git for Windows: https://git-scm.com/download/win
- Install Python 3.10+: download and install from https://www.python.org/downloads/ (check "Add python.exe to PATH")

Verify installs:
- `git --version`
- macOS: `python3 --version`
- Windows: `py --version`

### 1) Clone

```bash
git clone https://github.com/tmushd/Network_Malicious_Detection.git
cd Network_Malicious_Detection
```

### 2) Environment setup (macOS)
Install Homebrew:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Then install LightGBM / XGBoost (required)

```bash
brew install libomp
```

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

### 2) Environment setup (Windows PowerShell)

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
py -m pip install -U pip
pip install -r requirements.txt
```

Notes:
- First run requires internet (downloads the dataset and the pretrained ELECTRA checkpoint from Hugging Face).
- If `torch` fails to install from `requirements.txt`, install PyTorch first (per your CPU/GPU) and then re-run `pip install -r requirements.txt`.
- Seeds are fixed (`--seed 42`), but some ML libraries can still show tiny metric differences across hardware/OS.

### 3) Quick run (downsized, single command)

```bash
python scripts/reproduce.py --task both --train-size 2000 --val-size 300 --test-size 500 --electra-eval-size 200 --max-workers 2 --seed 42 --output-dir outputs/reproducibility
```

### 4) Reproduce our run (exact command + code version)

Code version (commit): `24af7509b4098dd4bd0d03fe95aaaeb163041b7f`

```bash
python scripts/reproduce.py --dataset-name bgspaditya/byt-mal-minpro --task both --train-size 8000 --val-size 1000 --test-size 2000 --electra-eval-size 500 --electra-model bgspaditya/malurl-electra --electra-batch-size 128 --electra-max-length 128 --electra-device auto --max-workers 4 --seed 42 --output-dir outputs/reproducibility
```

### Outputs

After either run completes, look in `outputs/reproducibility/`:
- `metrics.csv`
- `paper_comparison.csv`
- `reproducibility_report.md`
- `run_metadata.json`
- `confusion_*.png`
