"""Read local download metadata and check committed manual indexes."""

import json

from soundguide.catalog import Source, load_sources
from soundguide.errors import IndexNotReadyError, ManifestError
from soundguide.settings import Settings
from soundguide.store import Store


def downloaded_checksum(source: Source, settings: Settings) -> str | None:
    """Read only the manifest for the given source; missing is not malformed."""
    path = settings.data_dir / "raw" / source.id / "download.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except ValueError as exc:
        raise ManifestError("Manual download manifest is invalid") from exc
    checksum = record.get("sha256") if isinstance(record, dict) else None
    if not isinstance(checksum, str) or not checksum:
        raise ManifestError("Manual download manifest is missing a checksum")
    return checksum


def index_is_ready(
    source: Source, state: dict, checksum: str | None, profile: str
) -> bool:
    """Use the same provenance predicate for listing and retrieval."""
    return (
        state["profile"] == profile
        and state["chunks"] > 0
        and state["content_sha256"] == source.expected_sha256
        and (checksum is None or checksum == state["content_sha256"])
    )


async def list_source_statuses(settings: Settings) -> list[dict]:
    """Build the full inventory only for callers that actually request it."""
    states = {}
    if settings.database_url:
        states = {row["source_id"]: row for row in await Store(settings).status()}
    profile = settings.profile
    result = []
    for source in load_sources():
        item = source.model_dump(mode="json")
        checksum = downloaded_checksum(source, settings)
        item.update(status="downloaded" if checksum else "candidate", chunks=0)
        state = states.get(source.id)
        if state is not None:
            ready = index_is_ready(source, state, checksum, profile)
            item.update(
                status="indexed" if ready else "needs_reindex",
                chunks=state["chunks"],
                indexed_at=state["indexed_at"],
            )
        result.append(item)
    return result


async def require_indexed_source(
    store: Store, model: str, language: str, version: str
) -> Source:
    """Check only manuals matching the requested model, language and version."""
    settings = store.settings
    profile = settings.profile
    for source in load_sources():
        if (source.model, source.language, source.document_version) != (
            model,
            language,
            version,
        ):
            continue
        states = await store.status(source.id)
        if states and index_is_ready(
            source, states[0], downloaded_checksum(source, settings), profile
        ):
            return source
    raise IndexNotReadyError("This model's manual is not indexed or needs reindexing")
