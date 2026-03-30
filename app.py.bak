import argparse
from pathlib import Path

from src.config import settings
from src.embedder import OpenAIEmbedder
from src.generator import OpenAIAnswerGenerator
from src.parser import MultiModalParser
from src.retriever import MultiModalRetriever
from src.vector_store import VectorStore


def build_retriever() -> MultiModalRetriever:
    text_store = VectorStore(settings.text_store_path)
    image_store = VectorStore(settings.image_store_path)
    embedder = OpenAIEmbedder()
    return MultiModalRetriever(text_store=text_store, image_store=image_store, embedder=embedder)


def index_document(pdf_path: str) -> None:
    parser = MultiModalParser()
    embedder = OpenAIEmbedder()
    text_store = VectorStore(settings.text_store_path)
    image_store = VectorStore(settings.image_store_path)

    document = parser.parse(pdf_path)
    text_records, image_records = embedder.build_index_records(document)

    text_store.upsert_records(text_records)
    image_store.upsert_records(image_records)

    print("\n========== INDEX SUMMARY ==========")
    print(f"doc_id        : {document.doc_id}")
    print(f"pages         : {document.total_pages}")
    print(f"text records  : {len(text_records)}")
    print(f"image records : {len(image_records)}")
    print(f"text store    : {settings.text_store_path}")
    print(f"image store   : {settings.image_store_path}")


def query_document(question: str, top_k: int | None = None) -> None:
    retriever = build_retriever()
    generator = OpenAIAnswerGenerator()
    response = retriever.search(question=question, top_k=top_k or settings.top_k)
    answer = generator.generate(question=question, results=response.results)

    print("\n========== ANSWER ==========")
    print(answer)

    print("\n========== RETRIEVED RESULTS ==========")
    for idx, item in enumerate(response.results, start=1):
        print(
            f"[{idx}] score={item.final_score:.4f} semantic={item.semantic_score:.4f} "
            f"struct={item.structural_score:.4f} store={item.store_name} page={item.page_num} "
            f"type={item.block_type} bbox={item.bbox}"
        )
        print(item.display_text[:500])
        if item.image_path:
            print(f"image_path={item.image_path}")
        print("-" * 100)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Multi Modal RAG Lite")
    cli.add_argument("--mode", choices=["index", "query"], required=True)
    cli.add_argument("--pdf", type=str, help="Path to a PDF for indexing")
    cli.add_argument("--question", type=str, help="Question for querying")
    cli.add_argument("--top_k", type=int, default=None)
    args = cli.parse_args()

    if args.mode == "index":
        if not args.pdf:
            raise ValueError("index 模式必须提供 --pdf")
        if not Path(args.pdf).exists():
            raise FileNotFoundError(args.pdf)
        index_document(args.pdf)
    else:
        if not args.question:
            raise ValueError("query 模式必须提供 --question")
        query_document(args.question, top_k=args.top_k)
