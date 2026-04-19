"""Classical model training."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .metrics import compute_metrics, make_confusion


def build_classical_models(task: str, seed: int, max_workers: int) -> Dict[str, object]:
    """Build baseline models.

    Compiled deps (LightGBM/XGBoost) are imported lazily so the pipeline can still
    run on machines where those wheels are unavailable or unstable.
    """

    is_binary = task == "binary"

    try:
        from lightgbm import LGBMClassifier  # type: ignore
    except Exception:
        LGBMClassifier = None  # type: ignore

    try:
        from xgboost import XGBClassifier  # type: ignore
    except Exception:
        XGBClassifier = None  # type: ignore

    models: Dict[str, object] = {
        "RF": RandomForestClassifier(
            n_estimators=220,
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=max_workers,
        ),
    }

    if LGBMClassifier is not None:
        models["LGBM"] = LGBMClassifier(
            random_state=seed,
            n_estimators=220,
            learning_rate=0.08,
            num_leaves=63,
            objective="binary" if is_binary else "multiclass",
            n_jobs=max_workers,
            verbose=-1,
        )

    if XGBClassifier is not None:
        models["XGB"] = XGBClassifier(
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
        )

    return models


def _fit_predict_one(
    task: str,
    model_name: str,
    seed: int,
    max_workers: int,
    x_train,
    y_train,
    x_test,
    y_test,
    class_count: int,
) -> Optional[dict]:
    models = build_classical_models(task, seed, max_workers=max_workers)
    if model_name not in models:
        return None
    model = models[model_name]

    if task == "multiclass" and model_name in ("XGB", "LGBM"):
        # Be explicit for multiclass boosters; avoids ambiguity in native libs.
        n_classes = int(len(np.unique(y_train)))
        try:
            model.set_params(num_class=n_classes)
        except Exception:
            pass

    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    if task == "binary":
        pred = np.asarray(pred).astype(int)
    metrics = compute_metrics(y_test, pred)
    conf = make_confusion(y_test, pred, class_count=class_count)
    return {
        "metrics": metrics,
        "y_pred": pred,
        "confusion": conf,
    }


def _subprocess_worker(child_conn, payload: dict) -> None:
    # Runs in a child process. If a native crash occurs inside a compiled model
    # (e.g., LightGBM/XGBoost segfault), only the child dies and the pipeline
    # can continue.
    try:
        res = _fit_predict_one(**payload)
        child_conn.send({"ok": True, "result": res})
    except Exception as e:
        child_conn.send({"ok": False, "error": repr(e)})
    finally:
        try:
            child_conn.close()
        except Exception:
            pass


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
    """Fit classical baselines and return metrics + confusion matrices.

    Important: we run each baseline in its own subprocess so a native crash in
    LightGBM/XGBoost cannot take down the entire reproducibility run.
    """

    import multiprocessing as mp

    results: dict[str, dict] = {}
    model_names = list(build_classical_models(task, seed, max_workers=max_workers).keys())
    ctx = mp.get_context("spawn")

    for model_name in model_names:
        payload = {
            "task": task,
            "model_name": model_name,
            "seed": seed,
            "max_workers": max_workers,
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
            "class_count": class_count,
        }

        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_subprocess_worker, args=(child_conn, payload))
        proc.start()
        proc.join()

        if proc.exitcode != 0:
            print(f"[WARN] Baseline '{model_name}' crashed (exitcode={proc.exitcode}); skipping.")
            try:
                parent_conn.close()
            except Exception:
                pass
            continue

        if parent_conn.poll(0.1):
            msg = parent_conn.recv()
        else:
            msg = {"ok": False, "error": "no result returned"}
        try:
            parent_conn.close()
        except Exception:
            pass

        if not msg.get("ok"):
            err = msg.get("error", "unknown error")
            print(f"[WARN] Baseline '{model_name}' failed ({err}); skipping.")
            continue

        model_result = msg.get("result")
        if model_result is None:
            print(f"[WARN] Baseline '{model_name}' returned no result; skipping.")
            continue

        results[model_name] = model_result

    return results
