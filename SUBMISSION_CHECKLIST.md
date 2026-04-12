# Week 11 Submission Checklist

Use this file as your final pre-submit checklist.

## 1. Assignment requirement: public GitHub code

- [ ] Repo is public.
- [ ] TA/instructor can open it without requesting access.

## 2. Assignment requirement: implementation platform (modular)

- [x] Modular pipeline exists in `src/network_malicious_detection/`.
- [x] Single runner exists: `scripts/reproduce.py`.
- [x] Model blocks are separable:
  - lexical-feature ML models (`models.py`)
  - ELECTRA evaluation (`electra_eval.py`)
  - reporting (`reporting.py`)

## 3. Assignment requirement: proof of similar results to parent paper

- [x] Proof files are included in `proof/`.
- [x] Safe reproducibility metrics are included:
  - `proof/metrics_safe_run.csv`
  - `proof/paper_comparison_safe_run.csv`
  - `proof/reproducibility_report_safe_run.md`
- [x] ELECTRA reproduced near paper value:
  - Multiclass accuracy `0.988` vs paper `0.99`
  - Binary accuracy `0.99` vs paper `0.99`

## 4. Publish steps (run on your machine)

If you are not logged in:

```bash
gh auth login
```

Create your own public repo and push this project:

```bash
cd /Users/vayu/Documents/Playground/Network_Malicious_Detection
git add README.md requirements.txt scripts src proof SUBMISSION_CHECKLIST.md
git commit -m "Add modular reproducibility platform and proof artifacts"
gh repo create <your-username>/Network_Malicious_Detection_340W --public --source=. --remote=student-origin --push
```

Or use the helper script:

```bash
./scripts/publish_public_repo.sh <your-username> Network_Malicious_Detection_340W
```

Then submit:

- [ ] Public GitHub URL in Canvas.

## 5. Optional TA quick-run command (safe on laptop)

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
