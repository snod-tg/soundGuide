"""Offline tests against the real inherited parser, with synthetic input only."""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from soundguide.catalog import load_sources
from soundguide.prepare import prepare_document
from soundguide.server import APP
from soundguide.settings import Settings, get_settings


async def test_api_reports_candidates_without_model_or_database_credentials(tmp_path):
    """Expose honest source state without constructing external clients."""
    APP.dependency_overrides[get_settings] = lambda: Settings(data_dir=tmp_path)
    async with AsyncClient(
        transport=ASGITransport(app=APP), base_url="http://test"
    ) as client:
        assert (await client.get("/health")).json() == {
            "status": "ok",
            "service": "soundguide",
        }
        response = await client.get("/api/sources")
    APP.dependency_overrides.clear()
    assert response.status_code == 200
    assert {source["model"] for source in response.json()} == {
        source.model for source in load_sources()
    }
    assert all(source["status"] == "candidate" for source in response.json())


async def test_real_parser_preserves_source_and_stable_ids_without_cross_model_mix():
    """Exercise HTML parsing and splitting inherited from LangConnect."""
    source, other = load_sources()[:2]
    # Synthetic parser fixture; not official product instructions or an evaluation answer.
    payload = (
        "<html><body><h1>测试文档</h1>"
        + "型号与版本信息仅用于测试。" * 150
        + "</body></html>"
    ).encode()
    first = await prepare_document(source, payload, "text/html")
    second = await prepare_document(source, payload, "text/html")
    other_model = await prepare_document(other, payload, "text/html")
    assert len(first["chunks"]) > 1
    assert first["chunks"] == second["chunks"]
    assert first["document_id"] != other_model["document_id"]
    assert len({chunk["id"] for chunk in first["chunks"]}) == len(first["chunks"])
    assert all(chunk["metadata"]["model"] == source.model for chunk in first["chunks"])
    assert all(
        chunk["metadata"]["source_url"] == str(source.download_url or source.url)
        for chunk in first["chunks"]
    )
    assert first["indexed"] is False
    assert first["status"] == "prepared"
    assert first["provenance_verification"] == "operator_supplied_file"
    assert "<body>" not in first["chunks"][0]["content"]


async def test_updated_document_or_version_gets_new_identity():
    """Do not reuse document identities after a revision."""
    source = load_sources()[0]
    first = await prepare_document(source, b"original content", "text/plain")
    changed = await prepare_document(source, b"updated content", "text/plain")
    revised = await prepare_document(
        source.model_copy(update={"document_version": "v2"}),
        b"original content",
        "text/plain",
    )
    assert (
        len({first["document_id"], changed["document_id"], revised["document_id"]}) == 3
    )


async def test_empty_and_unsupported_documents_are_rejected():
    """Reject unsupported content instead of reporting successful preparation."""
    source = load_sources()[0]
    for payload, content_type in (
        (b"", "text/plain"),
        (b"   ", "text/plain"),
        (b"content", "image/png"),
    ):
        with pytest.raises(ValueError, match="Only HTML|Document must|No text"):
            await prepare_document(source, payload, content_type)


def test_duplicate_source_ids_are_rejected(tmp_path):
    """Prevent ambiguous source identity in the catalog."""
    source = load_sources()[0].model_dump(mode="json")
    path = tmp_path / "sources.json"
    path.write_text(json.dumps([source, source]))
    with pytest.raises(ValueError, match="unique"):
        load_sources(path)
