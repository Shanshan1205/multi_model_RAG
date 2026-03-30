from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable

import fitz
import numpy as np


def stable_id(*parts: object) -> str:
    raw = "::".join(str(p) for p in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path, default: object | None = None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_vector(vector: Iterable[float]) -> list[float]:
    arr = np.array(list(vector), dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    a = np.array(vec1, dtype=float)
    b = np.array(vec2, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def weighted_fusion(v1: list[float], v2: list[float], alpha: float) -> list[float]:
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    if a.shape != b.shape:
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]
    fused = alpha * a + (1.0 - alpha) * b
    return normalize_vector(fused.tolist())


def text_tokens(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fffA-Za-z0-9_\.\-%]+", text.lower())


def overlap_ratio(query: str, text: str) -> float:
    q = set(text_tokens(query))
    t = set(text_tokens(text))
    if not q or not t:
        return 0.0
    return len(q & t) / max(len(q), 1)


def encode_file_to_data_url(path: str | Path, mime: str = "image/png") -> str:
    raw = Path(path).read_bytes()
    return f"data:{mime};base64,{base64.b64encode(raw).decode('utf-8')}"


def rect_to_bbox(rect: fitz.Rect | tuple[float, float, float, float]) -> list[float]:
    if isinstance(rect, fitz.Rect):
        return [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
    return [float(x) for x in rect]


def union_bboxes(boxes: list[list[float]]) -> list[float] | None:
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return [x0, y0, x1, y1]


def bbox_area(bbox: list[float] | None) -> float:
    if not bbox:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def short_text(text: str, max_chars: int = 400) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
