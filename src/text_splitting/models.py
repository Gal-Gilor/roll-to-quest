from pydantic import BaseModel
from pydantic import Field


class Figure(BaseModel):
    """Represents a figure in a Markdown document."""

    title: str | None = Field(None, description="If existgs, the title of the figure.")
    caption: str | None = Field(None, description="If exists, the caption of the figure.")
    description: str | None = Field(
        None, description="A highly detailed description of the figure."
    )
    relevant_section: str | None = Field(
        None,
        description="available, the header of the section where the figure is located.",
    )


class Table(BaseModel):
    """Represents a table in a Markdown document."""

    title: str | None = Field(None, description="If exists, the title of the table.")
    caption: str | None = Field(None, description="If exists, the caption of the table.")
    table: str | None = Field(None, description="Markdown formatted table")
    relevant_section: str | None = Field(
        None,
        description="If exists, the header of the section where the table is located.",
    )


class Page(BaseModel):
    text: str = Field(
        description="The page content excluding the text in the figures and tables."
    )
    page_number: int
    tables: list[Table | None] = Field(
        default_factory=list, description="If exist, the tables in the page."
    )
    figures: list[Figure | None] = Field(
        default_factory=list, description="if exist, the figures in the page."
    )


class PDF(BaseModel):
    pages: list[Page]
