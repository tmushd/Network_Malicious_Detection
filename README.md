# Network Malicious Detection

## Repo

Reproducible pipeline for the malicious URL detection results in:
Türk & Kılıçaslan (2025), *Malicious URL Detection with Advanced Machine Learning and Optimization-Supported Deep Learning Models*.

What’s in here:
- Tasks: `binary` + `multiclass` malicious URL detection
- Baselines (lexical URL features): `RF`, `XGB`, `LGBM`
- Transformer checkpoint evaluation: ELECTRA (`bgspaditya/malurl-electra`)
- Default dataset source: Hugging Face `bgspaditya/byt-mal-minpro` (downloaded on first run)
- Main entrypoint: `scripts/reproduce.py` (writes all artifacts to `outputs/reproducibility/`)

## Run

### 1) Clone

```bash
git clone https://github.com/tmushd/Network_Malicious_Detection.git
cd Network_Malicious_Detection
```

### 2) Environment setup (macOS)

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
