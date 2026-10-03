"""Détection du matériel de calcul (CUDA / MPS / CPU)."""

from __future__ import annotations

import importlib.util
from functools import lru_cache

from . import config


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


@lru_cache(maxsize=1)
def get_device() -> str:
    if config.DEVICE != "auto":
        return config.DEVICE
    if not torch_available():
        return "cpu"
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def system_info() -> dict:
    info: dict = {"device": get_device(), "torch": None, "cuda": False, "gpu": None, "vram_gb": None}
    if not torch_available():
        return info
    import torch

    info["torch"] = torch.__version__
    if torch.cuda.is_available():
        info["cuda"] = True
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["vram_gb"] = round(props.total_memory / 1024**3, 1)
    return info


def free_memory() -> None:
    import gc

    gc.collect()
    if torch_available():
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
