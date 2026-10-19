"""Validated source inventory; a candidate is not a downloaded or indexed document."""

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, TypeAdapter

DEFAULT_CATALOG = Path(__file__).resolve().parent / "sources.json"


class Source(BaseModel):
    """Source identity and applicability, independent of ingestion state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    manufacturer: str = Field(min_length=1)
    model: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: HttpUrl
    download_url: HttpUrl | None = None
    expected_sha256: str | None = None
    format: Literal["pdf", "html_manual"] = "pdf"
    category: Literal["over-ear", "true-wireless", "neckband"] = "over-ear"
    machine_translated: bool = False
    excluded_pages: list[int] = Field(default_factory=list)
    language: str = Field(min_length=1)
    region: str = Field(min_length=1)
    document_version: str = Field(min_length=1)
    firmware_version: str | None = None
    verified_on: date
    status: Literal["candidate"] = "candidate"


def load_sources(path: Path = DEFAULT_CATALOG) -> list[Source]:
    """Load a catalog without network access or database side effects."""
    sources = TypeAdapter(list[Source]).validate_json(path.read_text(encoding="utf-8"))
    if len({source.id for source in sources}) != len(sources):
        raise ValueError("Source ids must be unique")
    return sources
