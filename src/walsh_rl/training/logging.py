"""JSON output for training and evaluation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def save_json(data: dict, path: str | Path) -> None:
    serializable = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, np.generic):
            serializable[key] = value.item()
        else:
            serializable[key] = value
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)
