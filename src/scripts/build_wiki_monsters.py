"""Build the fabled-campaigns Wiki monsters dataset from the SRD stat blocks.

Workflow:
    1. Split `monster_az.md` into one stat block per `#### ` heading.
    2. Extract each block into a `WikiMonster` with Gemini structured output. A response
       that fails validation is retried once with the validation error in the prompt.
    3. Cross-check every record against its source block (src/wiki/crosscheck.py).
    4. If nothing failed, write the records sorted by name as a JSON list, ready to copy
       over fabled-campaigns' `data/monsters.json`. Otherwise log every problem and write
       nothing.

Example Usage:
    python -m src.scripts.build_wiki_monsters
    python -m src.scripts.build_wiki_monsters --max-rate 60 --output data/wiki/monsters.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import jinja2
from aiolimiter import AsyncLimiter
from pydantic import ValidationError
from tqdm.asyncio import tqdm

from src.services.gemini import gemini_async_retry
from src.settings import client
from src.settings import config
from src.settings import jinja2_env_async
from src.settings import logger
from src.wiki.crosscheck import mismatches
from src.wiki.models import WikiMonster
from src.wiki.splitter import StatBlock
from src.wiki.splitter import split_stat_blocks

ROOT = Path(__file__).parents[2]
RESPONSE_SCHEMA = WikiMonster.model_json_schema()


async def extract_monster(
    block: StatBlock,
    template: jinja2.Template,
    model_id: str,
    thinking_budget: int,
    limiter: AsyncLimiter,
) -> WikiMonster:
    """Extract one stat block, retrying once with the validation error on failure."""

    @gemini_async_retry()
    async def _generate(contents: str):
        async with limiter:
            return await client.aio.models.generate_content(
                model=model_id,
                contents=contents,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": RESPONSE_SCHEMA,
                    "thinking_config": {"thinking_budget": thinking_budget},
                },
            )

    prompt = await template.render_async(
        category=block.category, group=block.group, text=block.text
    )
    contents = prompt
    for attempt in range(2):
        response = await _generate(contents)
        try:
            return WikiMonster.model_validate_json(response.text)
        except ValidationError as error:
            if attempt:
                raise
            contents = (
                f"{prompt}\n\nYour previous answer failed validation. "
                f"Fix these errors:\n\n{error}"
            )


async def main(
    source: Path, output: Path, model_id: str, thinking_budget: int, max_rate: int
) -> int:
    blocks = split_stat_blocks(source.read_text())
    template = jinja2_env_async.get_template("extract_wiki_monster.md")
    limiter = AsyncLimiter(max_rate, 60)
    logger.info(f"Extracting {len(blocks)} stat blocks with {model_id}")

    async def _extract(block: StatBlock) -> WikiMonster | Exception:
        try:
            return await extract_monster(block, template, model_id, thinking_budget, limiter)
        except Exception as error:
            return error

    results = await tqdm.gather(*map(_extract, blocks), desc="Extracting", unit="block")

    records, problems = [], []
    for block, result in zip(blocks, results):
        if isinstance(result, Exception):
            problems.append(f"{block.name}: extraction failed: {result}")
            continue
        record = result.model_dump()
        problems.extend(mismatches(record, block))
        records.append(record)

    if problems:
        for problem in problems:
            logger.error(problem)
        logger.error(f"{len(problems)} problems found; {output} was not written")
        return 1

    records.sort(key=lambda record: record["name"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(records)} monsters to {output}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the Wiki monsters dataset from the SRD stat blocks."
    )
    parser.add_argument("--source", type=Path, default=ROOT / "monster_az.md")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "wiki" / "monsters.json"
    )
    parser.add_argument(
        "--model",
        default=config.GENERATION_MODEL,
        help=f"Gemini model identifier (default: {config.GENERATION_MODEL}).",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=0,
        help="Token budget for Gemini's thinking (default: 0).",
    )
    parser.add_argument(
        "--max-rate",
        type=int,
        default=30,
        help="Max API calls per 60-second window (default: 30).",
    )
    args = parser.parse_args()

    if args.thinking_budget < 0:
        parser.error("--thinking-budget must be non-negative")
    if args.max_rate < 1:
        parser.error("--max-rate must be at least 1")

    sys.exit(
        asyncio.run(
            main(args.source, args.output, args.model, args.thinking_budget, args.max_rate)
        )
    )
