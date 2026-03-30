from __future__ import annotations

from openai import OpenAI

from src.config import settings
from src.schemas import IndexRecord, ParsedDocument
from src.utils import normalize_vector, stable_id, weighted_fusion


class OpenAIEmbedder:
    def __init__(self) -> None:
        client_kwargs = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url
        self.client = OpenAI(**client_kwargs) if settings.openai_api_key else None

    def embed_text(self, text: str) -> list[float]:
        clean_text = (text or "").strip() or "empty"
        if not self.client:
            return self._local_fallback_embedding(clean_text)
        response = self.client.embeddings.create(model=settings.openai_embed_model, input=clean_text)
        return normalize_vector(response.data[0].embedding)

    def build_index_records(self, document: ParsedDocument) -> tuple[list[IndexRecord], list[IndexRecord]]:
        text_records: list[IndexRecord] = []
        image_records: list[IndexRecord] = []

        for block in document.text_blocks:
            emb = self.embed_text(block.text)
            text_records.append(
                IndexRecord(
                    record_id=stable_id(block.block_id, "text_store"),
                    store_name="text_store",
                    doc_id=block.doc_id,
                    page_num=block.page_num,
                    block_id=block.block_id,
                    block_type=block.block_type,
                    bbox=block.bbox,
                    embedding=emb,
                    display_text=block.text,
                    structural_text=block.metadata.get("source", "page_text"),
                    metadata=block.metadata,
                )
            )

        for block in document.formula_blocks:
            formula_text = block.structured_data.get("latex") or block.text
            emb = self.embed_text(formula_text)
            text_records.append(
                IndexRecord(
                    record_id=stable_id(block.block_id, "formula_text_store"),
                    store_name="text_store",
                    doc_id=block.doc_id,
                    page_num=block.page_num,
                    block_id=block.block_id,
                    block_type="formula",
                    bbox=block.bbox,
                    embedding=emb,
                    display_text=formula_text,
                    structural_text="formula_latex",
                    image_path=block.image_path,
                    metadata={**block.metadata, **block.structured_data},
                )
            )

        for block in document.table_blocks:
            markdown = block.structured_data.get("markdown", block.text)
            structure_summary = block.structured_data.get("structure_summary", "")
            vision_summary = block.structured_data.get("vision_summary", "")
            table_text_embedding = self.embed_text(markdown)
            table_image_embedding = self.embed_text(vision_summary)
            fused = weighted_fusion(table_image_embedding, table_text_embedding, settings.table_image_alpha)
            table_structural_text = f"{structure_summary}\n{vision_summary}".strip()

            text_records.append(
                IndexRecord(
                    record_id=stable_id(block.block_id, "table_text_store"),
                    store_name="text_store",
                    doc_id=block.doc_id,
                    page_num=block.page_num,
                    block_id=block.block_id,
                    block_type="table",
                    bbox=block.bbox,
                    embedding=table_text_embedding,
                    display_text=markdown,
                    structural_text=table_structural_text,
                    image_path=block.image_path,
                    metadata={**block.metadata, **block.structured_data},
                )
            )
            image_records.append(
                IndexRecord(
                    record_id=stable_id(block.block_id, "table_image_store"),
                    store_name="image_store",
                    doc_id=block.doc_id,
                    page_num=block.page_num,
                    block_id=block.block_id,
                    block_type="table",
                    bbox=block.bbox,
                    embedding=fused,
                    display_text=vision_summary or markdown,
                    structural_text=table_structural_text,
                    image_path=block.image_path,
                    metadata={**block.metadata, **block.structured_data, "fusion_alpha": settings.table_image_alpha},
                )
            )

        for block in document.image_blocks:
            image_summary = block.structured_data.get("vision_summary", block.text)
            semantic_embedding = self.embed_text(image_summary)
            structure_embedding = self.embed_text(f"image {image_summary}")
            fused = weighted_fusion(structure_embedding, semantic_embedding, settings.image_text_alpha)
            image_records.append(
                IndexRecord(
                    record_id=stable_id(block.block_id, "image_store"),
                    store_name="image_store",
                    doc_id=block.doc_id,
                    page_num=block.page_num,
                    block_id=block.block_id,
                    block_type="image",
                    bbox=block.bbox,
                    embedding=fused,
                    display_text=image_summary,
                    structural_text=image_summary,
                    image_path=block.image_path,
                    metadata={**block.metadata, **block.structured_data, "fusion_alpha": settings.image_text_alpha},
                )
            )

        return text_records, image_records

    def _local_fallback_embedding(self, text: str) -> list[float]:
        dims = 128
        values = [0.0] * dims
        for idx, ch in enumerate(text.encode("utf-8")):
            values[idx % dims] += (ch % 31) / 31.0
        return normalize_vector(values)
