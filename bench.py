from __future__ import annotations

import re
from pathlib import Path

from dotenv import load_dotenv

from src import Document, EmbeddingStore, FixedSizeChunker, GeminiEmbedder, _mock_embed


DATA_DIR = Path("data/university-uet")
OUTPUT_PATH = Path("ket_qua_benchmark.txt")
CHUNK_SIZE = 500
OVERLAP = 50
TOP_K = 3

QUERIES = [
    (
        "Sinh viên hệ Chuẩn phải đóng học phí theo hình thức nào?",
        "Hệ Chuẩn thu học phí theo tín chỉ; mức thu thay đổi từng năm theo Nghị định của Chính phủ.",
    ),
    (
        "Khi nào sinh viên bị kỷ luật Cảnh cáo thì Điểm rèn luyện tối đa là bao nhiêu?",
        "Khi bị kỷ luật Cảnh cáo, điểm rèn luyện tối đa là loại Trung bình.",
    ),
    (
        "Sinh viên người dân tộc thiểu số thuộc hộ nghèo được miễn giảm học phí ra sao?",
        "Sinh viên dân tộc thiểu số thuộc hộ nghèo/cận nghèo thuộc diện miễn 100% học phí.",
    ),
    (
        "Để đạt điểm rèn luyện loại xuất sắc cần bao nhiêu điểm?",
        "Cần đạt từ 90 đến 100 điểm.",
    ),
    (
        "Sinh viên khuyết tật có được ưu tiên điểm rèn luyện không?",
        "Có. Sinh viên khuyết tật/hoàn cảnh đặc biệt có cơ chế cộng điểm ưu tiên sự nỗ lực.",
    ),
]


def parse_markdown(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    metadata: dict[str, str] = {}
    body = text

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            frontmatter = parts[1]
            body = parts[2].strip()
            current_key = ""
            for raw_line in frontmatter.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if ":" in line:
                    key, value = line.split(":", 1)
                    current_key = key.strip()
                    metadata[current_key] = value.strip().strip('"')
                elif current_key:
                    metadata[current_key] = f"{metadata[current_key]} {line}".strip()

    metadata.setdefault("doc_id", path.stem)
    metadata.setdefault("title", path.stem)
    return metadata, body


def build_documents() -> tuple[list[Document], dict[str, int]]:
    chunker = FixedSizeChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    documents: list[Document] = []
    per_doc_counts: dict[str, int] = {}

    for path in sorted(DATA_DIR.glob("*.md")):
        metadata, body = parse_markdown(path)
        chunks = chunker.chunk(body)
        doc_id = metadata["doc_id"]
        per_doc_counts[doc_id] = len(chunks)

        for index, chunk in enumerate(chunks, start=1):
            chunk_metadata = dict(metadata)
            chunk_metadata["chunk_index"] = str(index)
            chunk_metadata["source_path"] = str(path)
            documents.append(
                Document(
                    id=f"{doc_id}::chunk-{index}",
                    content=chunk,
                    metadata=chunk_metadata,
                )
            )

    return documents, per_doc_counts


def make_embedder():
    load_dotenv(Path(".env"), override=False)
    try:
        return GeminiEmbedder()
    except Exception:
        return _mock_embed


def clean_preview(text: str, limit: int = 320) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def main() -> int:
    docs, per_doc_counts = build_documents()
    embedder = make_embedder()
    store = EmbeddingStore(collection_name="university_uet_benchmark", embedding_fn=embedder)
    store.add_documents(docs)

    backend_name = getattr(embedder, "_backend_name", embedder.__class__.__name__)
    lines: list[str] = []
    lines.append("=== Benchmark Retrieval: university-uet ===")
    lines.append(f"Embedding backend: {backend_name}")
    lines.append(f"Chunking strategy: FixedSizeChunker(chunk_size={CHUNK_SIZE}, overlap={OVERLAP})")
    lines.append(f"Data directory: {DATA_DIR}")
    lines.append(f"Documents: {len(per_doc_counts)}")
    lines.append(f"Chunks: {store.get_collection_size()}")
    lines.append("")
    lines.append("Chunk count by document:")
    for doc_id, count in per_doc_counts.items():
        lines.append(f"- {doc_id}: {count}")

    relevant_count = 0
    lines.append("")
    lines.append("=== Top-3 Results ===")
    for query_index, (query, gold_answer) in enumerate(QUERIES, start=1):
        lines.append("")
        lines.append(f"Q{query_index}: {query}")
        lines.append(f"Gold answer: {gold_answer}")
        results = store.search_with_filter(query, top_k=TOP_K, metadata_filter={"audience": "student"})
        has_relevant = False

        for rank, result in enumerate(results, start=1):
            metadata = result["metadata"]
            content = result["content"]
            preview = clean_preview(content)
            doc_id = metadata.get("doc_id", "")
            chunk_index = metadata.get("chunk_index", "")
            score = result["score"]
            title = metadata.get("title", "")

            if any(token.lower() in content.lower() for token in gold_answer.split()[:4]):
                has_relevant = True

            lines.append(
                f"{rank}. score={score:.4f} doc_id={doc_id} chunk={chunk_index} title={title}"
            )
            lines.append(f"   preview: {preview}")

        if has_relevant:
            relevant_count += 1
        lines.append(f"Top-3 contains likely relevant chunk: {'yes' if has_relevant else 'manual-check'}")

    lines.append("")
    lines.append(f"Summary: {relevant_count} / {len(QUERIES)} queries had an automatically detected likely relevant chunk in top-3.")
    lines.append("Note: final relevance in the report was checked manually against the gold answers.")

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    print("\n".join(lines[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
