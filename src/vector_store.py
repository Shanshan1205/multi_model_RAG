from __future__ import annotations

from pathlib import Path

from src.schemas import IndexRecord
from src.utils import cosine_similarity, load_json, save_json


class VectorStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.records: dict[str, IndexRecord] = {}
        self._load()

    def _load(self) -> None:
        payload = load_json(self.path, default={"records": []})
        records = payload.get("records", []) if isinstance(payload, dict) else []
        self.records = {item["record_id"]: IndexRecord.from_dict(item) for item in records}

    def _flush(self) -> None:
        save_json(self.path, {"records": [record.to_dict() for record in self.records.values()]})

    def upsert_records(self, records: list[IndexRecord]) -> None:
        for record in records:
            self.records[record.record_id] = record
        self._flush()

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[IndexRecord, float]]:
        scored: list[tuple[IndexRecord, float]] = []
        for record in self.records.values():
            score = cosine_similarity(query_embedding, record.embedding)
            scored.append((record, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
