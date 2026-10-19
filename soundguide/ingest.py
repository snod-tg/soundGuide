"""Index verified local downloads. Run: uv run python -m soundguide.ingest."""

import asyncio
import hashlib
import json
from pathlib import Path

from soundguide.catalog import load_sources
from soundguide.embeddings import HttpEmbeddings
from soundguide.prepare import prepare_document
from soundguide.settings import get_settings
from soundguide.store import Store


async def ingest_source(source, settings, store, embeddings):
    """Verify downloaded bytes, embed externally, then commit the complete source."""
    manifest = settings.data_dir / "raw" / source.id / "download.json"
    record = json.loads(manifest.read_text())
    path = Path(record["path"]).resolve()
    if not path.is_relative_to((settings.data_dir / "raw").resolve()):
        raise ValueError("Downloaded file is outside the local raw directory")
    payload = path.read_bytes()
    if (
        record["source_id"] != source.id
        or record["model"] != source.model
        or record["requested_url"] != str(source.download_url or source.url)
        or hashlib.sha256(payload).hexdigest() != record["sha256"]
        or (source.expected_sha256 and source.expected_sha256 != record["sha256"])
    ):
        raise ValueError("Download identity/hash mismatch; fetch the source again")
    prepared = await asyncio.to_thread(
        lambda: asyncio.run(
            prepare_document(
                source,
                payload,
                "application/json"
                if source.format == "html_manual"
                else "application/pdf",
            )
        )
    )
    prepared["download"] = record
    prepared["provenance_verification"] = "official_https_download_sha256"
    vectors = await embeddings.aembed_documents(
        [chunk["embedding_text"] for chunk in prepared["chunks"]]
    )
    await store.replace(source, prepared, vectors)
    prepared["status"] = "indexed"
    prepared["indexed"] = True
    prepared["embedding_profile"] = settings.profile
    output = settings.data_dir / "prepared" / (source.id + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(prepared, ensure_ascii=False, indent=2))
    temporary.replace(output)
    print(f"{source.model}: indexed {len(vectors)} chunks", flush=True)


async def main():
    """Index catalogued official manuals without invoking a generation model."""
    settings = get_settings()
    store = Store(settings)
    await store.setup()
    embeddings = HttpEmbeddings(settings)
    for source in load_sources():
        await ingest_source(source, settings, store, embeddings)


if __name__ == "__main__":
    asyncio.run(main())
