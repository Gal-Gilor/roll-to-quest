"""Pydantic models and parsing helpers for Fabled Campaigns Wiki content.

These models describe the *published* shape of SRD content as it is rendered on
the Fabled Campaigns Wiki (the Next.js app reads ``data/magic-items.json`` and
types it with ``types/wiki.ts``). They are intentionally separate from the
knowledge-graph extraction models in :mod:`src.extraction.models`, which capture
a different, normalization-oriented shape.

The models serialize to camelCase so that ``model_dump(by_alias=True)`` produces
JSON that drops straight into the Wiki's ``data/*.json`` files, and so that the
generated JSON schema can be handed to ``google-genai`` as a ``response_schema``
for structured output.
"""

from src.wiki.models import MagicItem

__all__ = ["MagicItem"]
