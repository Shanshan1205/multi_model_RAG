from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    slice_dir: Path
    store_dir: Path
    manifest_dir: Path
    text_store_path: Path
    image_store_path: Path
    openai_api_key: str
    openai_base_url: str | None
    openai_chat_model: str
    openai_vision_model: str
    openai_embed_model: str
    top_k: int
    text_retrieval_k: int
    image_retrieval_k: int
    table_image_alpha: float
    image_text_alpha: float
    rerank_beta: float
    page_dpi: int
    enable_formula_recognition: bool
    enable_table_detection: bool
    enable_layout_image_slicing: bool
    enable_vision_summary: bool


base_dir = Path(__file__).resolve().parent.parent
configured_data_dir = os.getenv("DATA_DIR", str(base_dir / "data"))
data_dir = Path(configured_data_dir).resolve()
raw_dir = data_dir / "raw"
processed_dir = data_dir / "processed"
slice_dir = processed_dir / "slices"
store_dir = data_dir / "stores"
manifest_dir = data_dir / "manifests"
for folder in [data_dir, raw_dir, processed_dir, slice_dir, store_dir, manifest_dir]:
    folder.mkdir(parents=True, exist_ok=True)

settings = Settings(
    base_dir=base_dir,
    data_dir=data_dir,
    raw_dir=raw_dir,
    processed_dir=processed_dir,
    slice_dir=slice_dir,
    store_dir=store_dir,
    manifest_dir=manifest_dir,
    text_store_path=store_dir / "text_store.json",
    image_store_path=store_dir / "image_store.json",
    openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
    openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
    openai_vision_model=os.getenv("OPENAI_VISION_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")),
    openai_embed_model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large"),
    top_k=int(os.getenv("TOP_K", "6")),
    text_retrieval_k=int(os.getenv("TEXT_RETRIEVAL_K", "8")),
    image_retrieval_k=int(os.getenv("IMAGE_RETRIEVAL_K", "8")),
    table_image_alpha=float(os.getenv("TABLE_IMAGE_ALPHA", "0.65")),
    image_text_alpha=float(os.getenv("IMAGE_TEXT_ALPHA", "0.55")),
    rerank_beta=float(os.getenv("RERANK_BETA", "0.7")),
    page_dpi=int(os.getenv("PAGE_DPI", "160")),
    enable_formula_recognition=os.getenv("ENABLE_FORMULA_RECOGNITION", "1") == "1",
    enable_table_detection=os.getenv("ENABLE_TABLE_DETECTION", "1") == "1",
    enable_layout_image_slicing=os.getenv("ENABLE_LAYOUT_IMAGE_SLICING", "1") == "1",
    enable_vision_summary=os.getenv("ENABLE_VISION_SUMMARY", "1") == "1",
)
