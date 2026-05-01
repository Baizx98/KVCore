from __future__ import annotations

import json
from pathlib import Path
from typing import Generator, Iterable

import torch
from safetensors import safe_open


def safetensors_weights_iterator(
    weight_files: Iterable[Path],
    prefix: str = "",
    device: str = "cpu",
) -> Generator[tuple[str, torch.Tensor], None, None]:
    for filepath in weight_files:
        with safe_open(filepath, framework="pt", device=device) as weights_file:
            for name in weights_file.keys():
                yield prefix + name, weights_file.get_tensor(name)


def pt_weights_iterator(
    weight_files: Iterable[Path],
    prefix: str = "",
    device: str = "cpu",
) -> Generator[tuple[str, torch.Tensor], None, None]:
    for filepath in weight_files:
        state_dict = torch.load(
            filepath,
            map_location=device,
            weights_only=True,
        )
        for name, tensor in state_dict.items():
            yield prefix + name, tensor
        del state_dict


def get_weight_files_from_index(
    model_path: Path,
    index_name: str,
) -> list[Path]:
    index_path = model_path / index_name
    if not index_path.exists():
        return []

    index_data = json.loads(index_path.read_text())
    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Invalid weight index file: {index_path}")

    filenames = sorted(set(weight_map.values()))
    return [model_path / filename for filename in filenames]


def filter_duplicate_safetensors_files(
    weight_files: list[Path],
    model_path: Path,
) -> list[Path]:
    index_files = [
        model_path / "model.safetensors.index.json",
        model_path / "consolidated.safetensors.index.json",
    ]
    indexed: set[str] = set()
    for index_path in index_files:
        if index_path.exists():
            index_data = json.loads(index_path.read_text())
            weight_map = index_data.get("weight_map")
            if isinstance(weight_map, dict):
                indexed.update(weight_map.values())
    if not indexed:
        return weight_files
    filtered = [path for path in weight_files if path.name in indexed]
    return filtered or weight_files


__all__ = [
    "safetensors_weights_iterator",
    "pt_weights_iterator",
    "get_weight_files_from_index",
    "filter_duplicate_safetensors_files",
]
