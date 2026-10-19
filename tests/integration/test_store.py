"""Run: SG_RUN_POSTGRES_TESTS=1 uv run pytest tests/integration."""

import os
from dataclasses import replace
from urllib.parse import urlsplit
from uuid import uuid4

import asyncpg
import pytest

from soundguide.catalog import load_sources
from soundguide.prepare import prepare_document
from soundguide.settings import get_settings
from soundguide.store import Store

pytestmark = pytest.mark.skipif(
    os.getenv("SG_RUN_POSTGRES_TESTS") != "1",
    reason="Opt in to isolated PostgreSQL tests",
)


async def test_real_pgvector_isolation_idempotency_rollback_and_profile_guard():
    """Exercise real transactions and vectors without calling any embedding model."""
    settings = get_settings()
    dsn = os.environ.get("SG_TEST_DATABASE_URL", "")
    if (
        urlsplit(dsn).path != "/soundguide_test"
        or urlsplit(dsn).hostname != "127.0.0.1"
    ):
        pytest.fail("Refusing to modify any database except local soundguide_test")
    settings = replace(settings, database_url=dsn)
    store = Store(settings)
    await store.setup()
    sources = [
        s.model_copy(update={"id": s.id + "-" + uuid4().hex}) for s in load_sources()
    ]
    prepared = []
    vector = [1.0] + [0.0] * 511
    try:
        for source in sources:
            data = await prepare_document(
                source, b"original test content", "text/plain"
            )
            data["download"] = {"sha256": "synthetic-test-only"}
            prepared.append(data)
            await store.replace(source, data, [vector])
            await store.replace(source, data, [vector])
        hits = await store.search(vector, sources[0].model, 10, "zh-CN", "v1")
        assert len(hits) == 1
        assert hits[0]["metadata"]["model"] == sources[0].model
        assert hits[0]["cosine_similarity"] == pytest.approx(1)
        broken = await prepare_document(
            sources[0], b"changed test content", "text/plain"
        )
        broken["download"] = {"sha256": "synthetic-revision"}
        with pytest.raises(asyncpg.DataError):
            await store.replace(sources[0], broken, [[1.0]])
        hits = await store.search(vector, sources[0].model, 10, "zh-CN", "v1")
        assert hits[0]["id"] == prepared[0]["chunks"][0]["id"]
        assert hits[0]["content"] == "original test content"
        changed = Store(replace(settings, embedding_revision="different-weights"))
        with pytest.raises(ValueError, match="profile has changed"):
            await changed.search(vector, sources[0].model, 10, "zh-CN", "v1")
    finally:
        async with store.connection() as conn:
            await conn.execute(
                "DELETE FROM sg_documents WHERE source_id=ANY($1::text[])",
                [s.id for s in sources],
            )
