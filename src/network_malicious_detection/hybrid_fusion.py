"""Week 12 novelty block: hybrid lexical + ELECTRA fusion."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .electra_eval import predict_binary_electra_probabilities
from .features import build_lexical_features
from .metrics import compute_metrics, make_confusion
from .models import build_classical_models


def evaluate_binary_hybrid_fusion(
    train_df,
    val_df,
    test_df,
    seed: int,
    max_workers: int,
    electra_model: str,
    batch_size: int,
    max_length: int,
    device: str = "auto",
    local_files_only: bool = False,
) -> dict:
    rf = build_classical_models("binary", seed=seed, max_workers=max_workers)["RF"]

    x_train = build_lexical_features(train_df)
    x_val = build_lexical_features(val_df)
    x_test = build_lexical_features(test_df)
    y_train = train_df["label"].to_numpy(dtype=int)
    y_val = val_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)

    rf.fit(x_train, y_train)
    rf_val_prob = rf.predict_proba(x_val)[:, 1]
    rf_test_prob = rf.predict_proba(x_test)[:, 1]
    rf_test_pred = (rf_test_prob >= 0.5).astype(int)

    electra_val = predict_binary_electra_probabilities(
        test_df=val_df,
        model_name=electra_model,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        local_files_only=local_files_only,
    )
    electra_test = predict_binary_electra_probabilities(
        test_df=test_df,
        model_name=electra_model,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        local_files_only=local_files_only,
    )

    fusion_train = np.column_stack([rf_val_prob, electra_val["y_prob"]])
    fusion_test = np.column_stack([rf_test_prob, electra_test["y_prob"]])

    fusion_model = LogisticRegression(max_iter=1000, random_state=seed)
    fusion_model.fit(fusion_train, y_val)
    fusion_prob = fusion_model.predict_proba(fusion_test)[:, 1]
    fusion_pred = (fusion_prob >= 0.5).astype(int)

    return {
        "rf": {
            "metrics": compute_metrics(y_test, rf_test_pred),
            "y_pred": rf_test_pred,
            "y_prob": rf_test_prob,
        },
        "electra": {
            "metrics": compute_metrics(y_test, electra_test["y_pred"]),
            "y_pred": electra_test["y_pred"],
            "y_prob": electra_test["y_prob"],
            "device": electra_test["device"],
        },
        "hybrid_fusion": {
            "metrics": compute_metrics(y_test, fusion_pred),
            "y_pred": fusion_pred,
            "y_prob": fusion_prob,
            "confusion": make_confusion(y_test, fusion_pred, class_count=2),
            "weights": fusion_model.coef_.tolist(),
            "bias": fusion_model.intercept_.tolist(),
        },
    }
