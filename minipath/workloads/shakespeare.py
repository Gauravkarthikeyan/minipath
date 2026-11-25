"""Tiny Shakespeare (Karpathy's char-rnn corpus): ~1.1M characters of real text."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
CACHE = Path(os.environ.get("MINIPATH_DATA", Path(__file__).resolve().parents[2] / "data"))


def load() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (train_tokens, val_tokens, vocab). 90/10 split, uint8 ids."""
    path = CACHE / "tinyshakespeare.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.rename(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"tinyshakespeare checksum mismatch: {digest}")
    text = raw.decode("utf-8")
    vocab = sorted(set(text))
    lut = {c: i for i, c in enumerate(vocab)}
    ids = np.array([lut[c] for c in text], dtype=np.uint8)
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:], vocab
