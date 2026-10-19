"""Offline contract tests; no external models or PostgreSQL connections."""

from dataclasses import replace

import httpx
import pytest

from soundguide.embeddings import HttpEmbeddings
from soundguide.settings import Settings


@pytest.mark.parametrize("failure", ["dimension", "zero", "cardinality", "revision"])
async def test_embedding_provider_contract_fails_closed(monkeypatch, failure):
    """Do not persist incomplete or incompatible provider responses."""
    settings = Settings(
        database_url="unused-offline",
        embedding_key="synthetic-key",
        embedding_revision="expected",
    )

    def respond(request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "model": settings.embedding_model,
                    "revision": "wrong" if failure == "revision" else "expected",
                    "dimensions": 512,
                },
            )
        vector = [1.0] + [0.0] * 511
        if failure == "dimension":
            vector = vector[:-1]
        if failure == "zero":
            vector = [0.0] * 512
        return httpx.Response(
            200,
            json={
                "data": []
                if failure == "cardinality"
                else [{"index": 0, "embedding": vector}]
            },
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs),
    )
    with pytest.raises(ValueError, match="profile|cardinality|invalid vector"):
        await HttpEmbeddings(settings).aembed_documents(["synthetic content"])


def test_embedding_fingerprint_tracks_weights_and_query_preprocessing():
    """Changing weights or query handling requires a different compatible index."""
    settings = Settings(embedding_revision="one")
    assert settings.profile != replace(settings, embedding_revision="two").profile
    assert settings.profile != replace(settings, query_prefix="").profile
    assert "synthetic-secret" not in repr(
        replace(settings, embedding_key="synthetic-secret")
    )


def test_pdf_cleanup_preserves_steps_and_removes_navigation():
    """Do not turn related-topic links into apparent answer evidence."""
    from soundguide.prepare import clean_pdf_page

    text = "帮助指南\n无线降噪立体声耳机\nWH-1000XM5\n初始化耳机\n1\n关闭耳机。\n2\n按住两个按钮。\n相关主题\n连接电脑\nCopyright 2022 Sony Corporation\n140"
    cleaned = clean_pdf_page(text, 140)
    assert cleaned.startswith("初始化耳机")
    assert "\n1\n" in cleaned
    assert "\n2\n" in cleaned
    assert "相关主题" not in cleaned
    assert "连接电脑" not in cleaned
    assert "Copyright" not in cleaned


def test_official_downloader_rejects_untrusted_urls():
    """Redirects may never escape the official guide host."""
    from soundguide.download import official_url

    assert official_url("https://helpguide.sony.net/manual.pdf")
    assert not official_url("http://helpguide.sony.net/manual.pdf")
    assert not official_url("https://helpguide.sony.net.attacker.example/manual.pdf")
    assert not official_url("https://user:password@helpguide.sony.net/manual.pdf")
    assert not official_url("https://127.0.0.1/manual.pdf")


async def test_search_requires_model_and_rejects_empty_or_unconfigured_requests(
    tmp_path,
):
    """Reject ambiguous requests before any provider or database call."""
    from httpx import ASGITransport, AsyncClient

    from soundguide.server import APP
    from soundguide.settings import get_settings

    APP.dependency_overrides[get_settings] = lambda: Settings(data_dir=tmp_path)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=APP), base_url="http://test"
        ) as client:
            assert (
                await client.post("/api/search", json={"query": "测试"})
            ).status_code == 422
            assert (
                await client.post(
                    "/api/search", json={"query": "测试", "model": "other-model"}
                )
            ).status_code == 422
            assert (
                await client.post(
                    "/api/search", json={"query": "   ", "model": "WH-1000XM4"}
                )
            ).status_code == 422
            assert (
                await client.post(
                    "/api/search", json={"query": "测试", "model": "WH-1000XM4"}
                )
            ).status_code == 409
    finally:
        APP.dependency_overrides.clear()


@pytest.mark.parametrize("failure", ["redirect", "checksum"])
async def test_downloader_does_not_publish_unverified_downloads(
    tmp_path, monkeypatch, failure
):
    """Reject off-domain redirects and revised bytes before creating a manifest."""
    from soundguide.catalog import load_sources
    from soundguide.download import download_source

    source = next(source for source in load_sources() if source.format == "pdf")

    def respond(request) -> httpx.Response:
        if failure == "redirect":
            return httpx.Response(
                302, headers={"location": "https://attacker.example/manual.pdf"}
            )
        return httpx.Response(200, content=b"%PDF-synthetic-changed-document")

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs),
    )
    with pytest.raises(ValueError, match="allowlist|changed"):
        await download_source(source, tmp_path)
    assert not list(tmp_path.rglob("download.json"))


async def test_tampered_file_is_rejected_before_embedding_or_storage(tmp_path):
    """A stale download manifest must not authorize modified local bytes."""
    import json

    from soundguide.catalog import load_sources
    from soundguide.ingest import ingest_source

    source = load_sources()[0]
    raw = tmp_path / "raw" / source.id
    raw.mkdir(parents=True)
    file = raw / "manual.pdf"
    file.write_bytes(b"tampered local file")
    (raw / "download.json").write_text(
        json.dumps(
            {
                "source_id": source.id,
                "model": source.model,
                "requested_url": str(source.download_url or source.url),
                "path": str(file),
                "sha256": source.expected_sha256,
            }
        )
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        await ingest_source(source, Settings(data_dir=tmp_path), None, None)
