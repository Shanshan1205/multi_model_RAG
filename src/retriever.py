from __future__ import annotations

import re

from src.config import settings
from src.embedder import OpenAIEmbedder
from src.schemas import IndexRecord, SearchHit, SearchResponse
from src.utils import overlap_ratio
from src.vector_store import VectorStore


class MultiModalRetriever:
    def __init__(self, text_store: VectorStore, image_store: VectorStore, embedder: OpenAIEmbedder) -> None:
        self.text_store = text_store
        self.image_store = image_store
        self.embedder = embedder

    def search(self, question: str, top_k: int) -> SearchResponse:
        query_embedding = self.embedder.embed_text(question)
        text_candidates = self.text_store.search(query_embedding, settings.text_retrieval_k)
        image_candidates = self.image_store.search(query_embedding, settings.image_retrieval_k)

        merged: dict[str, SearchHit] = {}
        for record, semantic in text_candidates + image_candidates:
            hit = self._build_hit(question, record, semantic)
            existing = merged.get(hit.record_id)
            if existing is None or hit.final_score > existing.final_score:
                merged[hit.record_id] = hit

        results = sorted(merged.values(), key=lambda item: item.final_score, reverse=True)[:top_k]
        return SearchResponse(question=question, results=results)

    def _build_hit(self, question: str, record: IndexRecord, semantic_score: float) -> SearchHit:
        structural_score = self._structural_score(question, record)
        final_score = settings.rerank_beta * structural_score + (1.0 - settings.rerank_beta) * semantic_score
        return SearchHit(
            record_id=record.record_id,
            store_name=record.store_name,
            doc_id=record.doc_id,
            page_num=record.page_num,
            block_type=record.block_type,
            bbox=record.bbox,
            display_text=record.display_text,
            structural_text=record.structural_text,
            image_path=record.image_path,
            metadata=record.metadata,
            semantic_score=float(semantic_score),
            structural_score=float(structural_score),
            final_score=float(final_score),
        )

    def _structural_score(self, question: str, record: IndexRecord) -> float:
        query = question.lower()
        structure = (record.structural_text or "").lower()
        display = (record.display_text or "").lower()

        score = 0.0
        score += 0.35 * overlap_ratio(query, structure)
        score += 0.15 * overlap_ratio(query, display)

        wants_table = any(token in query for token in ["表", "table", "行", "列", "排名", "对比", "数值"])
        wants_chart = any(token in query for token in ["图", "chart", "趋势", "增长", "下降", "波动", "占比"])
        wants_formula = any(token in query for token in ["公式", "推导", "latex", "符号"])

        if record.block_type == "table":
            score += 0.25
            if wants_table:
                score += 0.2
            if re.search(r"rows=\d+|columns=\d+|header=", structure):
                score += 0.15
        elif record.block_type == "image":
            score += 0.1
            if wants_chart:
                score += 0.25
            if any(token in structure for token in ["chart", "bar", "line", "pie", "trend", "axis", "流程图", "示意图"]):
                score += 0.15
        elif record.block_type == "formula":
            score += 0.1
            if wants_formula:
                score += 0.3
        else:
            score += 0.05

        return min(score, 1.0)
