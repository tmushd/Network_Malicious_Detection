"""Shared constants for reproducibility experiments."""

CLASS_ORDER = ["benign", "defacement", "malware", "phishing"]
BINARY_CLASS_ORDER = ["benign", "malicious"]

PAPER_METRICS = {
    "multiclass": {
        "LGBM": {"accuracy": 0.96, "precision": 0.95, "recall": 0.93, "f1_weighted": 0.94},
        "XGB": {"accuracy": 0.96, "precision": 0.96, "recall": 0.94, "f1_weighted": 0.94},
        "RF": {"accuracy": 0.97, "precision": 0.96, "recall": 0.95, "f1_weighted": 0.95},
        "ELECTRA": {"accuracy": 0.99, "precision": 0.99, "recall": 0.99, "f1_weighted": 0.99},
    },
    "binary": {
        "LGBM": {"accuracy": 0.93, "precision": 0.94, "recall": 0.92, "f1_weighted": 0.93},
        "XGB": {"accuracy": 0.94, "precision": 0.95, "recall": 0.92, "f1_weighted": 0.94},
        "RF": {"accuracy": 0.95, "precision": 0.96, "recall": 0.95, "f1_weighted": 0.95},
        "ELECTRA": {"accuracy": 0.99, "precision": 0.99, "recall": 0.99, "f1_weighted": 0.99},
    },
}

