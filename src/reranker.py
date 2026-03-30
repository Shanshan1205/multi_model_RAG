from __future__ import annotations

import os
from typing import cast

from src.config import settings
from src.schemas import SearchHit


class BGEReranker:
    def __init__(self) -> None:
        self._model = None
        self._load_error: str | None = None

    def rerank(self, question: str, hits: list[SearchHit]) -> list[SearchHit]:
        if not hits:
            return hits

        if not settings.enable_model_reranker:
            for hit in hits:
                hit.model_rerank_score = 0.0
                hit.final_score = hit.base_score
            return sorted(hits, key=lambda item: item.final_score, reverse=True)

        model = self._get_model()
        if model is None:
            for hit in hits:
                hit.model_rerank_score = 0.0
                hit.final_score = hit.base_score
                hit.metadata["reranker_error"] = self._load_error or "BGE reranker unavailable"
            return sorted(hits, key=lambda item: item.final_score, reverse=True)

        pairs = [[question, self._build_passage(hit)] for hit in hits]

        try:
            raw_scores = model.compute_score(pairs, normalize=settings.reranker_normalize)
            if isinstance(raw_scores, (int, float)):
                score_list = [float(raw_scores)]
            else:
                score_list = [float(score) for score in cast(list[float], raw_scores)]
        except Exception as exc:
            for hit in hits:
                hit.model_rerank_score = 0.0
                hit.final_score = hit.base_score
                hit.metadata["reranker_error"] = f"compute_score failed: {exc}"
            return sorted(hits, key=lambda item: item.final_score, reverse=True)

        reranked: list[SearchHit] = []
        for hit, score in zip(hits, score_list, strict=False):
            clipped = max(0.0, min(1.0, score)) if settings.reranker_normalize else score
            hit.model_rerank_score = clipped
            hit.final_score = (
                settings.reranker_weight * hit.model_rerank_score
                + (1.0 - settings.reranker_weight) * hit.base_score
            )
            reranked.append(hit)

        return sorted(reranked, key=lambda item: item.final_score, reverse=True)

    def _build_passage(self, hit: SearchHit) -> str:
        bbox_text = "None" if hit.bbox is None else ",".join(f"{value:.2f}" for value in hit.bbox)
        return (
            f"block_type={hit.block_type}\n"
            f"store_name={hit.store_name}\n"
            f"page_num={hit.page_num}\n"
            f"bbox={bbox_text}\n"
            f"display_text:\n{hit.display_text}\n\n"
            f"structural_text:\n{hit.structural_text}"
        )

    def _get_model(self):
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            return None

        try:
            from FlagEmbedding import FlagReranker
        except Exception as exc:
            self._load_error = (
                "无法导入 FlagEmbedding。请先执行: pip install -U FlagEmbedding。"
                f" 原始错误: {exc}"
            )
            return None

        try:
            os.environ.setdefault("HF_HOME", str(settings.reranker_cache_dir))
            kwargs: dict[str, object] = {"use_fp16": settings.reranker_use_fp16}
            if settings.reranker_device:
                kwargs["device"] = settings.reranker_device
            self._model = FlagReranker(settings.reranker_model, **kwargs)
            return self._model
        except Exception as exc:
            self._load_error = (
                f"加载 BGE reranker 失败: model={settings.reranker_model}, "
                f"device={settings.reranker_device or 'auto'}, "
                f"use_fp16={settings.reranker_use_fp16}. 错误: {exc}"
            )
            return None
