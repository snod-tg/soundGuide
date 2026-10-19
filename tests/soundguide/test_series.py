"""Multi-model ingestion and HTML chapter provenance, using synthetic manuals only."""

import hashlib
import json
from collections import Counter

import httpx
import pytest

from soundguide.catalog import load_sources
from soundguide.download import download_source
from soundguide.prepare import prepare_document
from soundguide.retrieval import fuse
from soundguide.server import APP
from soundguide.settings import Settings, get_settings


def test_series_inventory_is_pinned_and_translation_is_explicit():
    """Account for each intended family and distinguish the three machine translations."""
    sources = load_sources()
    assert len(sources) == len({source.model for source in sources}) == 15
    assert Counter(source.category for source in sources) == {
        "over-ear": 8,
        "true-wireless": 5,
        "neckband": 2,
    }
    assert {source.model for source in sources if source.format == "html_manual"} == {
        "MDR-1000X"
    }
    assert {source.model for source in sources if source.machine_translated} == {
        "WH-1000XM4C",
        "WH-1000XX",
        "WF-1000XM6",
    }
    assert all(len(source.expected_sha256 or "") == 64 for source in sources)


@pytest.fixture
def html_source():
    """Reuse a real source identity with unpinned synthetic content for offline testing."""
    return next(s for s in load_sources() if s.format == "html_manual").model_copy(
        update={"expected_sha256": None}
    )


@pytest.fixture
def chapter_html():
    """Make body text visibly distinct from unwanted navigation and executable markup."""
    return '<html><main><h2 id="id_title">合成测试章节</h2><div id="id_content"><p>仅用于测试的证据。</p><nav>导航菜单</nav><script>untrusted()</script></div></main></html>'


async def test_html_download_follows_only_own_manual_and_preserves_chapter_urls(
    html_source, chapter_html, tmp_path, monkeypatch
):
    """Do not ingest another product's links, navigation, or invented PDF page numbers."""
    base = str(html_source.url).rsplit("/", 1)[0] + "/"
    visited = []

    def respond(request) -> httpx.Response:
        visited.append(str(request.url))
        if request.url.path.endswith("index.html"):
            return httpx.Response(
                200,
                text=f'<h1>{html_source.model}</h1><a href="contents/a.html">A</a><a href="contents/b.html">B</a><a href="https://attacker.example/contents/a.html">X</a><a href="../../other/contents/c.html">Y</a>',
            )
        return httpx.Response(200, text=chapter_html)

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: original(transport=httpx.MockTransport(respond), **kw),
    )
    record = await download_source(html_source, tmp_path)
    payload = (
        tmp_path / "raw" / html_source.id / (record["sha256"] + ".json")
    ).read_bytes()
    prepared = await prepare_document(html_source, payload, "application/json")
    assert record["chapters"] == 2
    assert len(visited) == 3
    assert {chunk["metadata"]["source_url"] for chunk in prepared["chunks"]} == {
        base + "contents/a.html",
        base + "contents/b.html",
    }
    assert all(chunk["metadata"]["page"] is None for chunk in prepared["chunks"])
    assert all(
        "导航菜单" not in chunk["content"] and "untrusted()" not in chunk["content"]
        for chunk in prepared["chunks"]
    )


@pytest.mark.parametrize("failure", ["missing", "redirect"])
async def test_incomplete_html_download_preserves_previous_manifest(
    html_source, chapter_html, tmp_path, monkeypatch, failure
):
    """Partial downloads must never replace the authoritative complete archive manifest."""
    directory = tmp_path / "raw" / html_source.id
    directory.mkdir(parents=True)
    manifest = directory / "download.json"
    manifest.write_text("previous complete snapshot")

    def respond(request) -> httpx.Response:
        if request.url.path.endswith("index.html"):
            return httpx.Response(
                200,
                text=f'<h1>{html_source.model}</h1><a href="contents/a.html">A</a><a href="contents/b.html">B</a>',
            )
        if request.url.path.endswith("b.html"):
            return (
                httpx.Response(404)
                if failure == "missing"
                else httpx.Response(
                    302,
                    headers={"location": "https://attacker.example/contents/b.html"},
                )
            )
        return httpx.Response(200, text=chapter_html)

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: original(transport=httpx.MockTransport(respond), **kw),
    )
    with pytest.raises((httpx.HTTPStatusError, ValueError)):
        await download_source(html_source, tmp_path)
    assert manifest.read_text() == "previous complete snapshot"


@pytest.mark.parametrize("failure", ["hash", "other-model"])
async def test_html_archive_rejects_tampering_and_cross_manual_chapters(
    html_source, chapter_html, failure
):
    """An archive's inner chapter identity is checked before parsing and embedding."""
    page = {
        "url": str(html_source.url).replace("index.html", "contents/a.html"),
        "html": chapter_html,
        "sha256": hashlib.sha256(chapter_html.encode()).hexdigest(),
    }
    if failure == "hash":
        page["html"] += "tampered"
    else:
        page["url"] = (
            "https://helpguide.sony.net/mdr/another-model/v1/zh-cn/contents/a.html"
        )
    archive = {
        "format": "sony_html_manual_v1",
        "model": html_source.model,
        "homepage_url": str(html_source.url),
        "pages": [page],
    }
    with pytest.raises(ValueError, match="URL/hash"):
        await prepare_document(
            html_source, json.dumps(archive).encode(), "application/json"
        )


def test_html_fusion_limits_per_chapter_instead_of_entire_manual():
    """Null PDF pages must not cause all HTML chapters to share a two-chunk limit."""
    rows = [
        {
            "id": str(i),
            "content": "合成测试证据",
            "metadata": {
                "source_id": "synthetic",
                "page": None,
                "source_url": f"https://helpguide.sony.net/test/{i}.html",
            },
        }
        for i in range(5)
    ]
    assert len(fuse(rows, "合成测试证据", 5)) == 5


async def test_unindexed_catalogued_model_is_rejected_before_embedding(
    monkeypatch, tmp_path
):
    """Catalog membership alone does not authorize answering from a missing index."""
    from soundguide import server

    async def inventory(*args: object) -> list[dict]:
        return [
            {
                "model": "WF-1000XM6",
                "language": "zh-CN",
                "document_version": "v1",
                "status": "candidate",
            }
        ]

    async def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Do not call an embedding service for an unindexed model")

    monkeypatch.setattr(server, "sources", inventory)
    monkeypatch.setattr(server, "retrieve", forbidden)
    APP.dependency_overrides[get_settings] = lambda: Settings(
        database_url="unused",
        embedding_key="synthetic",
        embedding_revision="synthetic",
        data_dir=tmp_path,
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=APP), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/search", json={"model": "WF-1000XM6", "query": "合成测试"}
            )
        assert response.status_code == 409
    finally:
        APP.dependency_overrides.clear()
