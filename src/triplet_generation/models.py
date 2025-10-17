from pydantic import BaseModel
from pydantic import Field


class AnchorNegativePair(BaseModel):
    """Anchor-Negative representing Gemini's response schema."""

    anchor: str = Field(
        ...,
        description=(
            "The reference text you're starting from—the point of comparison. "
            "In retrieval systems, this is typically your query or question. "
            "The anchor represents 'what you're looking for.'"
        ),
    )
    negative: str = Field(
        ...,
        description=(
            "A text that is irrelevant or incorrect for the anchor. "
            "A negative that is deceptively similar to the anchor—it looks related on the "
            "surface (lexically similar, same domain, overlapping keywords) but "
            "is actually semantically different or factually incorrect."
        ),
    )


class Triplet(BaseModel):
    """Complete triplet with anchor, positive, and negative."""

    anchor: str = Field(
        ...,
        description=(
            "The reference text you're starting from—the point of comparison. "
            "In retrieval systems, this is typically your query or question. "
            "The anchor represents 'what you're looking for.'"
        ),
    )
    positive: str = Field(
        ...,
        description=(
            "A text that is semantically relevant, related, or correct for the anchor. "
            "This is the complete source chunk text."
        ),
    )
    negative: str = Field(
        ...,
        description=(
            "A text that is irrelevant or incorrect for the anchor. "
            "A negative that is deceptively similar to the anchor—it looks related on the "
            "surface (lexically similar, same domain, overlapping keywords) but "
            "is actually semantically different or factually incorrect."
        ),
    )
