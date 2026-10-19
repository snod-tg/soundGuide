"""Download official manual snapshots. Run: uv run python -m soundguide.download."""

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from bs4 import BeautifulSoup

from soundguide.catalog import load_sources

DATA = Path(__file__).resolve().parents[1] / ".data"
MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_HTML_BYTES = 1024 * 1024
MAX_CHAPTERS = 200


def official_url(url: str) -> bool:
    """Only fetch HTTPS documents from the selected official guide host."""
    parsed = urlsplit(url)
    return (
        parsed.scheme == "https"
        and parsed.hostname == "helpguide.sony.net"
        and parsed.port in (None, 443)
        and not parsed.username
        and not parsed.password
    )


async def fetch_official(client, url: str, limit: int, prefix: str | None = None):
    """Validate every redirect before fetching, with bounded decompressed bytes."""
    for _ in range(5):
        if not official_url(url) or (prefix and not url.startswith(prefix)):
            raise ValueError("Document URL is outside the official allowlist")
        async with client.stream("GET", url) as response:
            if response.is_redirect:
                url = str(response.url.join(response.headers["location"]))
                continue
            response.raise_for_status()
            body = bytearray()
            async for block in response.aiter_bytes():
                body.extend(block)
                if len(body) > limit:
                    raise ValueError("Official document exceeds the byte limit")
            return bytes(body), response
    raise ValueError("Too many redirects")


async def html_snapshot(source, client):
    """Archive only chapters linked by this manual's homepage, never a general crawl."""
    home_url = str(source.url)
    prefix = home_url.rsplit("/", 1)[0] + "/"
    homepage, response = await fetch_official(client, home_url, MAX_HTML_BYTES, prefix)
    soup = BeautifulSoup(homepage, "html.parser")
    if source.model not in soup.get_text(" ", strip=True):
        raise ValueError("Manual homepage does not identify the expected model")
    links = sorted(
        {
            str(response.url.join(anchor["href"]))
            for anchor in soup.select("a[href]")
            if str(response.url.join(anchor["href"])).startswith(prefix + "contents/")
            and urlsplit(str(response.url.join(anchor["href"]))).path.endswith(".html")
            and not urlsplit(str(response.url.join(anchor["href"]))).query
            and not urlsplit(str(response.url.join(anchor["href"]))).fragment
        }
    )
    if not links or len(links) > MAX_CHAPTERS:
        raise ValueError("Manual must contain 1 to 200 chapter links")
    semaphore = asyncio.Semaphore(4)

    async def chapter(url) -> dict:
        async with semaphore:
            payload, chapter_response = await fetch_official(
                client, url, MAX_HTML_BYTES, prefix + "contents/"
            )
        parsed = BeautifulSoup(payload, "html.parser")
        if not parsed.select_one("main #id_content") or not parsed.select_one(
            "main #id_title"
        ):
            raise ValueError("Manual chapter is missing its expected body/title")
        return {
            "url": str(chapter_response.url),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "html": payload.decode("utf-8"),
        }

    pages = await asyncio.gather(*(chapter(url) for url in links))
    archive = {
        "format": "sony_html_manual_v1",
        "model": source.model,
        "homepage_url": home_url,
        "homepage_html": homepage.decode("utf-8"),
        "pages": pages,
    }
    payload = json.dumps(archive, ensure_ascii=False, sort_keys=True).encode()
    if len(payload) > MAX_PDF_BYTES:
        raise ValueError("Complete HTML manual exceeds 20 MiB")
    return payload, response, len(pages)


async def download_source(source, directory: Path = DATA):
    """Publish a complete verified PDF or chapter archive, preserving previous snapshots."""
    url = str(source.download_url or source.url)
    async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
        if source.format == "html_manual":
            payload, response, chapters = await html_snapshot(source, client)
            extension = ".json"
        else:
            payload, response = await fetch_official(client, url, MAX_PDF_BYTES)
            if not payload.startswith(b"%PDF-"):
                raise ValueError("Official response is not a PDF")
            chapters = None
            extension = ".pdf"
    digest = hashlib.sha256(payload).hexdigest()
    if source.expected_sha256 and digest != source.expected_sha256:
        raise ValueError(
            "Official document changed; review source metadata before replacing the pinned snapshot"
        )
    target = directory / "raw" / source.id / (digest + extension)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    record = {
        "source_id": source.id,
        "model": source.model,
        "requested_url": url,
        "final_url": str(response.url),
        "fetched_at": datetime.now(UTC).isoformat(),
        "sha256": digest,
        "bytes": len(payload),
        "path": str(target.resolve()),
        "content_type": "application/json"
        if source.format == "html_manual"
        else response.headers.get("content-type"),
        "etag": response.headers.get("etag"),
        "last_modified": response.headers.get("last-modified"),
        "format": source.format,
        "chapters": chapters,
    }
    manifest = target.parent / "download.json"
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    temporary.replace(manifest)
    return record


async def main():
    """Download all manuals; reuse only matching, verified local snapshots."""
    for source in load_sources():
        manifest = DATA / "raw" / source.id / "download.json"
        if manifest.exists():
            record = json.loads(manifest.read_text())
            cached = Path(record["path"]).resolve()
            if (
                cached.is_relative_to((DATA / "raw").resolve())
                and cached.is_file()
                and record["source_id"] == source.id
                and record["model"] == source.model
                and record["requested_url"] == str(source.download_url or source.url)
                and hashlib.sha256(cached.read_bytes()).hexdigest()
                == record["sha256"]
                == source.expected_sha256
            ):
                print(f"{source.model}: verified cached snapshot", flush=True)
                continue
        record = await download_source(source)
        print(
            f"{source.model}: downloaded {record['bytes']} bytes; sha256={record['sha256']}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
