"""Classical model training."""

from __future__ import annotations

from typing import Dict

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .metrics import compute_metrics, make_confusion


def build_classical_models(task: str, seed: int, max_workers: int) -> Dict[str, object]:
    is_binary = task == "binary"
    models: Dict[str, object] = {
        "LGBM": LGBMClassifier(
            random_state=seed,
            n_estimators=220,
            learning_rate=0.08,
            num_leaves=63,
            objective="binary" if is_binary else "multiclass",
            n_jobs=max_workers,
            verbose=-1,
        ),
        "XGB": XGBClassifier(
            random_state=seed,
            n_estimators=220,
            max_depth=7,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            objective="binary:logistic" if is_binary else "multi:softmax",
            eval_metric="logloss" if is_binary else "mlogloss",
            n_jobs=max_workers,
        ),
        "RF": RandomForestClassifier(
            n_estimators=220,
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=max_workers,
        ),
    }
    return models


def fit_and_evaluate_classical(
    task: str,
    x_train,
    y_train,
    x_test,
    y_test,
    class_count: int,
    seed: int,
    max_workers: int,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    models = build_classical_models(task, seed, max_workers=max_workers)

    for name, model in models.items():
        if name == "XGB" and task == "multiclass":
            model.set_params(num_class=int(len(np.unique(y_train))))

        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        if task == "binary":
            pred = np.asarray(pred).astype(int)
        metrics = compute_metrics(y_test, pred)
        conf = make_confusion(y_test, pred, class_count=class_count)
        results[name] = {
            "metrics": metrics,
            "y_pred": pred,
            "confusion": conf,
        }
    return results
