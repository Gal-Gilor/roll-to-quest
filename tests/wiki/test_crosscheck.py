import json
from pathlib import Path

import pytest

from src.wiki.crosscheck import expected_fields
from src.wiki.crosscheck import mismatches
from src.wiki.splitter import split_stat_blocks

SOURCE = Path(__file__).parents[2] / "monster_az.md"
BLOCKS = {block.name: block for block in split_stat_blocks(SOURCE.read_text())}
SAMPLE = json.loads((Path(__file__).parent / "monsters_sample.json").read_text())


@pytest.mark.parametrize("record", SAMPLE, ids=[r["slug"] for r in SAMPLE])
def test_sample_matches_source(record):
    assert mismatches(record, BLOCKS[record["name"]]) == []


def test_every_block_parses():
    for block in BLOCKS.values():
        expected_fields(block)


def test_reads_lair_xp():
    expected = expected_fields(BLOCKS["Aboleth"])
    assert (expected["cr"], expected["xp"], expected["xpInLair"]) == ("10", 5900, 7200)


@pytest.mark.parametrize(
    "override",
    [
        {"ac": 14},
        {"xpInLair": 11000},
        {"skills": None},
        {"saves": {**SAMPLE[0]["saves"], "dex": 3}},
        {"body": SAMPLE[0]["body"].replace("+4", "+5", 1)},
    ],
)
def test_reports_changed_fields(override):
    record = SAMPLE[0] | override
    [message] = mismatches(record, BLOCKS[record["name"]])
    assert message.startswith(f"{record['name']}: {next(iter(override))} is")
