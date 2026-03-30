from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


BBox = list[float]


@dataclass
class Block:
    block_id: str
    doc_id: str
    page_num: int
    block_type: str
    text: str = ""
    bbox: BBox | None = None
    image_path: str | None = None
    structured_data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedDocument:
    doc_id: str
    source_pdf: str
    total_pages: int
    text_blocks: list[Block]
    table_blocks: list[Block]
    image_blocks: list[Block]
    formula_blocks: list[Block]

    @property
    def all_blocks(self) -> list[Block]:
        return self.text_blocks + self.table_blocks + self.image_blocks + self.formula_blocks

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_pdf": self.source_pdf,
            "total_pages": self.total_pages,
            "text_blocks": [b.to_dict() for b in self.text_blocks],
            "table_blocks": [b.to_dict() for b in self.table_blocks],
            "image_blocks": [b.to_dict() for b in self.image_blocks],
            "formula_blocks": [b.to_dict() for b in self.formula_blocks],
        }


@dataclass
class IndexRecord:
    record_id: str
    store_name: str
    doc_id: str
    page_num: int
    block_id: str
    block_type: str
    bbox: BBox | None
    embedding: list[float]
    display_text: str
    structural_text: str
    image_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IndexRecord":
        return cls(**payload)


@dataclass
class SearchHit:
    record_id: str
    store_name: str
    doc_id: str
    page_num: int
    block_type: str
    bbox: BBox | None
    display_text: str
    structural_text: str
    image_path: str | None
    metadata: dict[str, Any]
    semantic_score: float
    structural_score: float
    final_score: float


@dataclass
class SearchResponse:
    question: str
    results: list[SearchHit]
