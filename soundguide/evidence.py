"""Contracts for stored metadata, ranked chunks and answer evidence."""

from datetime import datetime
from typing import Literal, NotRequired

from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict

from soundguide.errors import IndexNotReadyError


class ChunkMetadata(TypedDict):
    """Provenance written at ingestion; page is null for web manuals."""

    source_id: str
    document_id: str
    content_sha256: str
    model: str
    manufacturer: str
    language: str
    region: str
    format: Literal["pdf", "html_manual"]
    machine_translated: bool
    document_version: str
    firmware_version: str | None
    source_url: str
    page: int | None
    pipeline_version: str
    chunk_index: int
    section_title: str | None


class IndexedSourceState(TypedDict):
    """Committed index metadata returned by PostgreSQL."""

    source_id: str
    model: str
    profile: str
    indexed_at: datetime
    content_sha256: str | None
    chunks: int


class RankedChunk(TypedDict):
    """A stored chunk plus scores or context links added by ranking."""

    id: str
    content: str
    metadata: ChunkMetadata
    cosine_similarity: float
    rrf_score: NotRequired[float]
    bm25_score: NotRequired[float]
    ranks: NotRequired[dict[str, int]]
    context_neighbor_of: NotRequired[str]


class EvidenceChunk(RankedChunk):
    """A ranked chunk with an authoritative citation location."""

    citation_url: str


METADATA_ADAPTER = TypeAdapter(ChunkMetadata)
EVIDENCE_ADAPTER = TypeAdapter(list[EvidenceChunk])


def parse_chunk_metadata(payload: str) -> ChunkMetadata:
    """Fail closed on corrupt or obsolete metadata at the storage boundary."""
    try:
        return METADATA_ADAPTER.validate_json(payload, strict=True)
    except ValidationError as exc:
        raise IndexNotReadyError(
            "Indexed chunk metadata is invalid; reindex the manual"
        ) from exc
