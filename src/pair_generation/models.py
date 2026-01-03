from pydantic import BaseModel
from pydantic import Field


class AnchorOnly(BaseModel):
    """Response schema from Gemini for anchor-only generation.

    This model represents the minimal response when generating only queries/anchors
    without negatives, optimized for cost-efficient pair generation.
    """

    anchor: str = Field(
        ...,
        description=(
            "A natural language query or question that the document chunk can answer. "
            "Frame this as a realistic search query a user would ask."
        ),
    )


class AnchorPositivePair(BaseModel):
    """Final output format for anchor-positive training pairs.

    This model represents the complete pair used for training embedding models
    without hard negatives. The positive is always the source document chunk.
    """

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
