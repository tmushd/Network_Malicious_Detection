"""Dataset loading and task-specific label preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from .constants import BINARY_CLASS_ORDER, CLASS_ORDER


@dataclass
class DataSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass
class TaskSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    class_names: list[str]


def _clean_split(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.dropna(subset=["url", "type"]).copy()
    cleaned["url"] = cleaned["url"].astype(str).str.strip()
    cleaned = cleaned[cleaned["url"].str.len() > 0]
    cleaned = cleaned.drop_duplicates(subset=["url", "type"]).reset_index(drop=True)
    return cleaned


def _sample_split(
    df: pd.DataFrame,
    label_col: str,
    sample_size: Optional[int],
    seed: int,
) -> pd.DataFrame:
    if sample_size is None or sample_size <= 0 or sample_size >= len(df):
        return df.reset_index(drop=True)
    sampled, _ = train_test_split(
        df,
        train_size=sample_size,
        random_state=seed,
        stratify=df[label_col],
    )
    return sampled.reset_index(drop=True)


def load_hf_dataset(
    dataset_name: str,
    seed: int,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
    test_size: Optional[int] = None,
) -> DataSplits:
    dataset = load_dataset(dataset_name)
    train_df = _clean_split(pd.DataFrame(dataset["train"]))
    val_df = _clean_split(pd.DataFrame(dataset["val"]))
    test_df = _clean_split(pd.DataFrame(dataset["test"]))

    train_df = _sample_split(train_df, "type", train_size, seed)
    val_df = _sample_split(val_df, "type", val_size, seed)
    test_df = _sample_split(test_df, "type", test_size, seed)

    return DataSplits(train=train_df, val=val_df, test=test_df)


def make_task_splits(data: DataSplits, task: str) -> TaskSplits:
    if task not in {"multiclass", "binary"}:
        raise ValueError(f"Unsupported task: {task}")

    if task == "multiclass":
        label_map = {name: idx for idx, name in enumerate(CLASS_ORDER)}
        class_names = CLASS_ORDER
        convert = lambda s: s
    else:
        label_map = {name: idx for idx, name in enumerate(BINARY_CLASS_ORDER)}
        class_names = BINARY_CLASS_ORDER
        convert = lambda s: np.where(s == "benign", "benign", "malicious")

    def transform(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["label_name"] = convert(out["type"])
        out["label"] = out["label_name"].map(label_map).astype(int)
        return out

    return TaskSplits(
        train=transform(data.train),
        val=transform(data.val),
        test=transform(data.test),
        class_names=class_names,
    )

