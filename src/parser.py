from __future__ import annotations

import re
import shutil
from pathlib import Path

import fitz
from openai import OpenAI

from src.config import settings
from src.schemas import Block, ParsedDocument
from src.utils import bbox_area, encode_file_to_data_url, rect_to_bbox, save_json, short_text, stable_id


class MultiModalParser:
    def __init__(self) -> None:
        client_kwargs = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url
        self.client = OpenAI(**client_kwargs) if settings.openai_api_key else None

    def parse(self, pdf_path: str) -> ParsedDocument:
        src_pdf = Path(pdf_path).resolve()
        doc_id = stable_id(src_pdf.name, src_pdf.stat().st_size)
        copied_pdf = settings.raw_dir / f"{doc_id}_{src_pdf.name}"
        if src_pdf != copied_pdf:
            shutil.copy2(src_pdf, copied_pdf)

        doc = fitz.open(str(src_pdf))
        text_blocks: list[Block] = []
        table_blocks: list[Block] = []
        image_blocks: list[Block] = []
        formula_blocks: list[Block] = []

        doc_slice_dir = settings.slice_dir / doc_id
        doc_slice_dir.mkdir(parents=True, exist_ok=True)

        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            page_num = page_index + 1
            page_data = page.get_text("dict")

            text_blocks.extend(self._extract_text_blocks(doc_id, page_num, page_data))
            if settings.enable_table_detection:
                table_blocks.extend(self._extract_table_blocks(doc_id, page, page_num, doc_slice_dir))
            if settings.enable_layout_image_slicing:
                image_blocks.extend(self._extract_image_blocks(doc_id, page, page_num, doc_slice_dir))
            if settings.enable_formula_recognition:
                formula_blocks.extend(self._extract_formula_blocks(doc_id, page, page_num, page_data, doc_slice_dir))

        parsed = ParsedDocument(
            doc_id=doc_id,
            source_pdf=str(copied_pdf),
            total_pages=len(doc),
            text_blocks=text_blocks,
            table_blocks=table_blocks,
            image_blocks=image_blocks,
            formula_blocks=formula_blocks,
        )
        save_json(settings.manifest_dir / f"{doc_id}.json", parsed.to_dict())
        return parsed

    def _extract_text_blocks(self, doc_id: str, page_num: int, page_data: dict) -> list[Block]:
        blocks: list[Block] = []
        for block_idx, block in enumerate(page_data.get("blocks", [])):
            if block.get("type") != 0:
                continue
            lines = []
            line_boxes = []
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                line_text = "".join(span.get("text", "") for span in spans).strip()
                if line_text:
                    lines.append(line_text)
                    line_boxes.append(rect_to_bbox(line["bbox"]))
            merged = "\n".join(lines).strip()
            if not merged or len(merged) < 8:
                continue
            bbox = rect_to_bbox(block["bbox"])
            block_type = self._classify_text_block(merged)
            blocks.append(
                Block(
                    block_id=stable_id(doc_id, page_num, "text", block_idx, merged[:50]),
                    doc_id=doc_id,
                    page_num=page_num,
                    block_type=block_type,
                    text=merged,
                    bbox=bbox,
                    metadata={"source": "page_text", "char_count": len(merged)},
                )
            )
        return blocks

    def _extract_table_blocks(self, doc_id: str, page: fitz.Page, page_num: int, out_dir: Path) -> list[Block]:
        blocks: list[Block] = []
        try:
            tables = page.find_tables()
        except Exception:
            return blocks

        for idx, table in enumerate(getattr(tables, "tables", [])):
            bbox = rect_to_bbox(table.bbox)
            if bbox_area(bbox) < 1000:
                continue
            markdown = table.to_markdown(clean=False)
            image_path = out_dir / f"page_{page_num:03d}_table_{idx:02d}.png"
            self._save_region_image(page, bbox, image_path)
            structure_summary = self._build_table_structure_summary(markdown)
            vision_summary = self._describe_visual_region(
                image_path=image_path,
                role="table",
                prompt=(
                    "请分析这张表格截图，返回简洁中文摘要，重点包括：表头、关键字段、主要数值模式、"
                    "是否适合回答趋势/对比/排名类问题。"
                ),
            )
            blocks.append(
                Block(
                    block_id=stable_id(doc_id, page_num, "table", idx, bbox),
                    doc_id=doc_id,
                    page_num=page_num,
                    block_type="table",
                    text=markdown,
                    bbox=bbox,
                    image_path=str(image_path),
                    structured_data={
                        "markdown": markdown,
                        "structure_summary": structure_summary,
                        "vision_summary": vision_summary,
                    },
                    metadata={"source": "page.find_tables"},
                )
            )
        return blocks

    def _extract_image_blocks(self, doc_id: str, page: fitz.Page, page_num: int, out_dir: Path) -> list[Block]:
        blocks: list[Block] = []
        drawings = page.get_drawings()
        for idx, drawing in enumerate(drawings):
            rect = drawing.get("rect")
            if not rect:
                continue
            bbox = rect_to_bbox(rect)
            if bbox_area(bbox) < 4000:
                continue
            image_path = out_dir / f"page_{page_num:03d}_figure_{idx:02d}.png"
            self._save_region_image(page, bbox, image_path)
            summary = self._describe_visual_region(
                image_path=image_path,
                role="chart_or_figure",
                prompt=(
                    "请判断这是不是图表、示意图、流程图或普通图片。"
                    "用 JSON 风格文本返回 chart_type、title_guess、keywords、trend_or_relation、summary。"
                ),
            )
            blocks.append(
                Block(
                    block_id=stable_id(doc_id, page_num, "image", idx, bbox),
                    doc_id=doc_id,
                    page_num=page_num,
                    block_type="image",
                    text=summary,
                    bbox=bbox,
                    image_path=str(image_path),
                    structured_data={"vision_summary": summary},
                    metadata={"source": "page_drawings"},
                )
            )
        return blocks

    def _extract_formula_blocks(
        self,
        doc_id: str,
        page: fitz.Page,
        page_num: int,
        page_data: dict,
        out_dir: Path,
    ) -> list[Block]:
        blocks: list[Block] = []
        for idx, block in enumerate(page_data.get("blocks", [])):
            if block.get("type") != 0:
                continue
            text = " ".join(
                "".join(span.get("text", "") for span in line.get("spans", []))
                for line in block.get("lines", [])
            ).strip()
            if not self._looks_like_formula(text):
                continue
            bbox = rect_to_bbox(block["bbox"])
            image_path = out_dir / f"page_{page_num:03d}_formula_{idx:02d}.png"
            self._save_region_image(page, bbox, image_path)
            latex = self._describe_visual_region(
                image_path=image_path,
                role="formula",
                prompt="请把图片中的公式识别为 LaTeX，只返回 LaTeX 公式本身。",
            )
            blocks.append(
                Block(
                    block_id=stable_id(doc_id, page_num, "formula", idx, bbox),
                    doc_id=doc_id,
                    page_num=page_num,
                    block_type="formula",
                    text=latex or text,
                    bbox=bbox,
                    image_path=str(image_path),
                    structured_data={"latex": latex or text, "raw_text": text},
                    metadata={"source": "formula_heuristic"},
                )
            )
        return blocks

    def _save_region_image(self, page: fitz.Page, bbox: list[float], output_path: Path) -> None:
        rect = fitz.Rect(*bbox)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect, alpha=False)
        pix.save(str(output_path))

    def _classify_text_block(self, text: str) -> str:
        stripped = text.strip()
        if len(stripped) < 40 and not stripped.endswith("。"):
            return "title"
        return "text"

    def _looks_like_formula(self, text: str) -> bool:
        if len(text) > 160 or len(text) < 3:
            return False
        formula_signals = ["=", "+", "-", "∑", "√", "λ", "β", "α", "≤", "≥", "^", "_"]
        hits = sum(1 for token in formula_signals if token in text)
        return hits >= 2 or bool(re.search(r"[A-Za-z]\([^)]+\)", text))

    def _build_table_structure_summary(self, markdown: str) -> str:
        rows = [line for line in markdown.splitlines() if line.strip()]
        header = rows[0] if rows else ""
        column_count = header.count("|") - 1 if header else 0
        row_count = max(len(rows) - 2, 0) if len(rows) >= 2 else 0
        return f"rows={row_count}; columns={column_count}; header={short_text(header, 120)}"

    def _describe_visual_region(self, image_path: Path, role: str, prompt: str) -> str:
        if not self.client or not settings.enable_vision_summary:
            return f"[{role}] {image_path.name}"
        try:
            response = self.client.responses.create(
                model=settings.openai_vision_model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": encode_file_to_data_url(image_path),
                            },
                        ],
                    }
                ],
                max_output_tokens=300,
            )
            return (response.output_text or "").strip() or f"[{role}] {image_path.name}"
        except Exception as exc:
            return f"[{role}] vision_summary_failed: {exc}"
