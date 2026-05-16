"""Annotate existing chunk JSONL files with TOC-derived metadata.

Reads a chunks JSONL file (output of chunk_documents.py), looks up each
section_header in data/srd_toc.yaml, and stamps entity_type, context_path,
and chunk_strategy onto each matched chunk.

Usage:
    poetry run python -m src.scripts.annotate_chunks data/chunks/SRD_CC_v5.2.1_chunks.jsonl
    poetry run python -m src.scripts.annotate_chunks <input> --toc data/srd_toc.yaml
    poetry run python -m src.scripts.annotate_chunks <input> --output <out.jsonl>
"""

import argparse
import json
import sys
from pathlib import Path

from src.settings import logger
from src.toc.models import TocDocument
from src.toc.parser import build_index
from src.toc.parser import load_toc


def load_annotator(toc_path: Path) -> TocDocument:
    return load_toc(toc_path)


def annotate_chunk(chunk: dict, document: TocDocument) -> dict:
    title = chunk.get("section_header", "")
    nodes, parents = build_index(document)
    node = next((n for n in nodes.values() if n.title == title), None)
    if node is None:
        return chunk
    result = dict(chunk)
    # Resolve entity_type by walking ancestors
    entity_type: str | None = None
    current_id: str | None = node.id
    while current_id is not None:
        n = nodes.get(current_id)
        if n is None:
            break
        if n.entity_type is not None:
            entity_type = n.entity_type
            break
        current_id = parents.get(current_id)
    # Resolve chunk_strategy by walking ancestors
    chunk_strategy = None
    current_id = node.id
    while current_id is not None:
        n = nodes.get(current_id)
        if n is None:
            break
        if n.chunk_strategy is not None:
            chunk_strategy = n.chunk_strategy
            break
        current_id = parents.get(current_id)
    if entity_type is not None:
        result["entity_type"] = entity_type
    result["context_path"] = node.id
    if chunk_strategy is not None:
        result["chunk_strategy"] = chunk_strategy.value
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stamp chunk JSONL with entity_type and context_path from the TOC."
    )
    parser.add_argument("input", type=str, help="Path to input chunks JSONL file")
    parser.add_argument(
        "--toc",
        type=str,
        default="data/srd_toc.yaml",
        help="Path to the SRD TOC YAML file (default: data/srd_toc.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output path for annotated JSONL. "
            "Defaults to <input_stem>_annotated.jsonl in the same directory."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    input_path = Path(args.input)
    toc_path = Path(args.toc)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    if not toc_path.exists():
        logger.error(f"TOC file not found: {toc_path}")
        return 1

    output_path = (
        Path(args.output)
        if args.output
        else input_path.parent / f"{input_path.stem}_annotated.jsonl"
    )

    logger.info(f"Loading TOC from {toc_path}")
    document = load_annotator(toc_path)

    annotated_count = 0
    total_count = 0

    with (
        open(input_path, encoding="utf-8") as infile,
        open(output_path, "w", encoding="utf-8") as outfile,
    ):
        for line in infile:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            result = annotate_chunk(chunk, document)
            outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
            total_count += 1
            if "context_path" in result:
                annotated_count += 1

    logger.info(f"Annotated {annotated_count}/{total_count} chunks. Output: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
