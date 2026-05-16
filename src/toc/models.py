from enum import StrEnum

from pydantic import BaseModel
from pydantic import Field


class NodeType(StrEnum):
    CHAPTER = "chapter"
    SECTION = "section"
    SUBSECTION = "subsection"
    ENTRY = "entry"


class ChunkStrategy(StrEnum):
    ENTRY = "entry"
    ROW = "row"
    SECTION = "section"


class TocNode(BaseModel):
    id: str
    title: str
    node_type: NodeType
    chunk_strategy: ChunkStrategy | None = None
    entity_type: str | None = None
    page: int | None = None
    children: list["TocNode"] = Field(default_factory=list)


class TocDocument(BaseModel):
    document: str
    sections: list[TocNode]
