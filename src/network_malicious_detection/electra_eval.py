"""ELECTRA inference/evaluation utilities."""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
import torch
from huggingface_hub.utils import disable_progress_bars
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as hf_logging

from .metrics import compute_metrics, make_confusion


def _choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_electra(model_name: str, resolved_device: str, local_files_only: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    ).to(resolved_device)
    model.eval()
    return tokenizer, model


def _predict_logits(
    urls: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str,
    local_files_only: bool = False,
):
    resolved_device = _choose_device(device)
    tokenizer, model = _load_electra(model_name, resolved_device, local_files_only=local_files_only)
    logits_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(urls), batch_size):
            batch_urls = urls[start : start + batch_size]
            encoded = tokenizer(
                batch_urls,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(resolved_device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            logits_chunks.append(logits.cpu().numpy())

    merged = np.concatenate(logits_chunks, axis=0) if logits_chunks else np.empty((0, 0))
    id2label_raw = model.config.id2label or {}
    id2label = {int(k): str(v).lower() for k, v in id2label_raw.items()}
    if not id2label:
        id2label = {idx: str(idx) for idx in range(model.config.num_labels)}

    return {
        "logits": merged,
        "id2label": id2label,
        "device": resolved_device,
    }


def predict_binary_electra_probabilities(
    test_df,
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str = "auto",
    local_files_only: bool = False,
) -> dict:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1" if local_files_only else os.environ.get("HF_HUB_OFFLINE", "0"))
    disable_progress_bars()
    hf_logging.set_verbosity_error()
    try:
        hf_logging.disable_progress_bar()
    except AttributeError:
        pass

    urls = test_df["url"].astype(str).tolist()
    raw = _predict_logits(
        urls=urls,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        local_files_only=local_files_only,
    )
    logits = raw["logits"]
    id2label = raw["id2label"]
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

    benign_idx = next((idx for idx, name in id2label.items() if name == "benign"), 0)
    positive_prob = 1.0 - probs[:, benign_idx]
    y_pred = (positive_prob >= 0.5).astype(int)

    return {
        "y_prob": positive_prob,
        "y_pred": y_pred,
        "device": raw["device"],
    }


def evaluate_pretrained_electra(
    test_df,
    task: Literal["multiclass", "binary"],
    class_names: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str = "auto",
    local_files_only: bool = False,
) -> dict:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    disable_progress_bars()
    hf_logging.set_verbosity_error()
    try:
        hf_logging.disable_progress_bar()
    except AttributeError:
        pass

    urls = test_df["url"].astype(str).tolist()
    raw = _predict_logits(
        urls=urls,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        local_files_only=local_files_only,
    )
    logits = raw["logits"]
    id2label = raw["id2label"]
    preds = np.argmax(logits, axis=1).tolist()

    if task == "multiclass":
        name_to_idx = {name: idx for idx, name in enumerate(class_names)}
        converted_preds = []
        for pred_id in preds:
            pred_name = id2label.get(int(pred_id), "").strip().lower()
            if pred_name in name_to_idx:
                converted_preds.append(name_to_idx[pred_name])
            else:
                converted_preds.append(int(pred_id))
        y_pred = np.asarray(converted_preds, dtype=int)
    else:
        converted_preds = []
        for pred_id in preds:
            pred_name = id2label.get(int(pred_id), "").strip().lower()
            converted_preds.append(0 if pred_name == "benign" else 1)
        y_pred = np.asarray(converted_preds, dtype=int)

    y_true = test_df["label"].to_numpy(dtype=int)
    metrics = compute_metrics(y_true, y_pred)
    confusion = make_confusion(y_true, y_pred, class_count=len(class_names))

    return {
        "metrics": metrics,
        "y_pred": y_pred,
        "confusion": confusion,
        "device": raw["device"],
    }
