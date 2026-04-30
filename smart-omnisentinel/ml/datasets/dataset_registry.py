"""
ml/datasets/dataset_registry.py
---------------------------------
Central registry mapping dataset names to their loader modules.
Used by the training pipeline to select data sources.

Supported datasets:
    rwf2000     - RWF-2000 violence detection (train/val)
    urfall      - UR Fall Detection dataset
    custom      - Custom annotated CCTV footage (CSV-based)

Usage:
    from ml.datasets.dataset_registry import get_loader
    loader = get_loader("rwf2000")
    loader.process_dataset(input_root="data/raw/RWF-2000",
                           output_root="data/processed")
"""
from __future__ import annotations

from typing import Any

REGISTRY = {
    "rwf2000": "ml.datasets.rwf2000_loader",
    "urfall": "ml.datasets.urfall_loader",
    "custom": "ml.datasets.custom_loader",
}


def get_loader(name: str) -> Any:
    """
    Return the loader module for the given dataset name.
    Raises ValueError if dataset is not registered.
    """
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(REGISTRY.keys())}"
        )
    import importlib
    return importlib.import_module(REGISTRY[name])


def list_datasets() -> list:
    return list(REGISTRY.keys())
