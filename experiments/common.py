"""Shared helpers: result paths, JSON output, machine info, plot style."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "docs" / "figures"


def machine() -> dict:
    cpu = platform.processor() or ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "unknown"
    import jax

    return {
        "cpu": cpu,
        "cores": os.cpu_count(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "commit": commit,
    }


def _jsonable(o):
    import numpy as np

    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def save(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps({"machine": machine(), **payload}, indent=2, default=_jsonable))
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )
    return plt


# Fixed color per system, used in every figure.
COLORS = {
    "spmd": "#4C72B0",
    "async": "#DD8452",
    "sync": "#8C8C8C",
    "pathways": "#DD8452",
    "theory": "#55A868",
    "accent": "#C44E52",
}
