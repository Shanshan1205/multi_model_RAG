from __future__ import annotations

from openai import OpenAI

from src.config import settings
from src.schemas import SearchHit
from src.utils import encode_file_to_data_url


class OpenAIAnswerGenerator:
    def __init__(self) -> None:
        client_kwargs = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url
        self.client = OpenAI(**client_kwargs) if settings.openai_api_key else None

    def generate(self, question: str, results: list[SearchHit]) -> str:
        if not results:
            return "没有检索到可用证据。"

        context_lines = []
        for idx, item in enumerate(results, start=1):
            context_lines.append(
                f"[证据{idx}] page={item.page_num}; type={item.block_type}; store={item.store_name}; "
                f"semantic={item.semantic_score:.4f}; structural={item.structural_score:.4f}; final={item.final_score:.4f}\n"
                f"内容: {item.display_text}\n"
                f"结构: {item.structural_text}\n"
            )

        if not self.client:
            refs = "、".join(sorted({f"第{item.page_num}页" for item in results}))
            return (
                "未检测到 OPENAI_API_KEY，因此返回基于检索结果的保守摘要。\n\n"
                f"问题：{question}\n\n"
                + "\n".join(context_lines[:4])
                + f"\n参考来源：{refs}"
            )

        content = [
            {
                "type": "input_text",
                "text": (
                    "你是一个严谨的多模态PDF分析助手。请严格依据提供的证据回答。\n\n"
                    f"用户问题：{question}\n\n"
                    "证据如下：\n"
                    + "\n".join(context_lines)
                    + "\n回答要求：\n"
                    "1. 优先引用表格/图表的结构信息；文本语义为辅。\n"
                    "2. 回答中指出关键结论来自哪一页。\n"
                    "3. 若证据不足，明确说明信息不足。\n"
                    "4. 最后附上“参考来源：第X页 ...”。"
                ),
            }
        ]

        for item in results:
            if item.image_path:
                content.append({"type": "input_text", "text": f"下面这张图片来自第 {item.page_num} 页的 {item.block_type} 区域。"})
                content.append({"type": "input_image", "image_url": encode_file_to_data_url(item.image_path)})

        response = self.client.responses.create(
            model=settings.openai_chat_model,
            input=[{"role": "user", "content": content}],
            max_output_tokens=900,
        )
        return (response.output_text or "").strip()
