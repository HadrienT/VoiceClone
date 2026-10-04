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
        return _best_cuda()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _best_cuda() -> str:
    """Avec plusieurs GPU, prend celui qui a le plus de mémoire libre (les autres peuvent être occupés)."""
    import torch

    if torch.cuda.device_count() < 2:
        return "cuda"
    try:
        free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    except Exception:
        return "cuda"
    return f"cuda:{free.index(max(free))}"


def cuda_index(device: str) -> int:
    """Indice du GPU d'un nom de périphérique ("cuda" -> 0, "cuda:1" -> 1)."""
    _, _, idx = device.partition(":")
    return int(idx) if idx.isdigit() else 0


def system_info() -> dict:
    info: dict = {"device": get_device(), "torch": None, "cuda": False, "gpu": None, "vram_gb": None}
    if not torch_available():
        return info
    import torch

    info["torch"] = torch.__version__
    if torch.cuda.is_available():
        info["cuda"] = True
        idx = cuda_index(info["device"]) if info["device"].startswith("cuda") else 0
        props = torch.cuda.get_device_properties(idx)
        info["gpu"] = props.name
        info["vram_gb"] = round(props.total_memory / 1024**3, 1)
        info["vram_free_gb"] = round(torch.cuda.mem_get_info(idx)[0] / 1024**3, 1)
        info["gpus"] = gpu_list(idx)
    return info


def gpu_list(selected: int | None = None) -> list[dict]:
    """Mémoire de chaque GPU (tous processus confondus) et part utilisée par VoiceClone."""
    import torch

    out = []
    for i in range(torch.cuda.device_count()):
        try:
            free, total = torch.cuda.mem_get_info(i)
            out.append({"index": i, "name": torch.cuda.get_device_name(i), "selected": i == selected,
                        "total_gb": round(total / 1024**3, 2), "free_gb": round(free / 1024**3, 2),
                        "used_gb": round((total - free) / 1024**3, 2),
                        "voiceclone_gb": round(torch.cuda.memory_reserved(i) / 1024**3, 2)})
        except Exception as exc:
            out.append({"index": i, "error": str(exc)})
    return out


def _cuda_ready() -> bool:
    if not torch_available():
        return False
    import sys

    if "torch" not in sys.modules:  # ne pas importer torch juste pour ça
        return False
    import torch

    return torch.cuda.is_available()


def free_vram_gb() -> float | None:
    dev = get_device()
    if not dev.startswith("cuda") or not _cuda_ready():
        return None
    import torch

    return torch.cuda.mem_get_info(cuda_index(dev))[0] / 1024**3


def allocated_gb(dev: str) -> float | None:
    if not dev.startswith("cuda") or not _cuda_ready():
        return None
    import torch

    return torch.cuda.memory_allocated(cuda_index(dev)) / 1024**3


def free_memory() -> None:
    import gc

    gc.collect()
    if torch_available():
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
