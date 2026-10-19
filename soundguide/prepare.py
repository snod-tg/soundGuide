"""Prepare an operator-supplied document using LangConnect's real parser/splitter.

Run: uv run python -m soundguide.prepare --help
This does not fetch, embed, index or verify that the supplied file matches its URL.
"""

import argparse
import asyncio
import hashlib
import json
import re
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
from fastapi import UploadFile
from langchain_community.document_loaders.parsers import PDFMinerParser
from langchain_core.documents import Document
from langchain_core.documents.base import Blob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from starlette.datastructures import Headers

from langconnect.services.document_processor import process_document
from soundguide import PIPELINE_VERSION
from soundguide.catalog import Source, load_sources

MAX_BYTES = 20 * 1024 * 1024
CONTENT_TYPES = ("text/html", "text/plain", "application/pdf", "application/json")


def clean_pdf_page(text: str, page: int) -> str:
    """Remove Sony navigation/footer noise while preserving numbered instructions."""
    lines = []
    for line in text.splitlines():
        value = line.strip()
        if value == "相关主题":
            break
        if (
            value in {"帮助指南", "SONY", "1000X THE COLLEXION"}
            or value.startswith(
                (
                    "无线降噪立体声耳机",
                    "无线立体声耳机",
                    "请注意\uff0c本帮助指南是由机器翻译",
                )
            )
            or re.fullmatch(r"(?:WH|WF|WI|MDR)-1000X[A-Z0-9]*(?: / [A-Z0-9-]+)?", value)
            or re.fullmatch(r"型号[\uff1a:]?\s*YY\d+", value)
            or "Copyright" in value
        ):
            continue
        lines.append(line)
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[-1].strip() == str(page):
        lines.pop()
    return "\n".join(lines).strip()


def parse_html_manual(source: Source, payload: bytes) -> list[Document]:
    """Extract each archived chapter body with its real URL, never invented PDF pages."""
    archive = json.loads(payload)
    if (
        source.format != "html_manual"
        or archive.get("format") != "sony_html_manual_v1"
        or archive.get("model") != source.model
        or archive.get("homepage_url") != str(source.url)
    ):
        raise ValueError("HTML manual identity mismatch")
    prefix = str(source.url).rsplit("/", 1)[0] + "/contents/"
    documents = []
    seen = set()
    for page in archive["pages"]:
        url = page["url"]
        parsed = urlsplit(url)
        if (
            not url.startswith(prefix)
            or not parsed.path.endswith(".html")
            or parsed.query
            or parsed.fragment
            or url in seen
            or hashlib.sha256(page["html"].encode()).hexdigest() != page["sha256"]
        ):
            raise ValueError("HTML chapter URL/hash mismatch")
        seen.add(url)
        soup = BeautifulSoup(page["html"], "html.parser")
        body = soup.select_one("main #id_content")
        title = soup.select_one("main #id_title")
        if body is None or title is None:
            raise ValueError("HTML chapter is missing its body/title")
        for junk in body.select("script, style, nav, .relatedTopic, .related-topic"):
            junk.decompose()
        for image in body.select("img[alt]"):
            image.replace_with(image.get("alt", ""))
        text = body.get_text("\n", strip=True)
        if not text:
            raise ValueError("HTML chapter contains no extractable text")
        heading = title.get_text(" ", strip=True)
        documents.append(
            Document(
                page_content=heading + "\n" + text,
                metadata={"source_url": url, "section_title": heading[:80]},
            )
        )
    if not documents:
        raise ValueError("HTML manual contains no chapters")
    return documents


async def prepare_document(source: Source, payload: bytes, content_type: str) -> dict:
    """Parse/split a file and attach stable provenance and applicability metadata."""
    if content_type not in CONTENT_TYPES:
        raise ValueError(
            "Only HTML, plain text, PDF and archived HTML manuals are supported"
        )
    if not payload or len(payload) > MAX_BYTES:
        raise ValueError("Document must contain 1 byte to 20 MiB")
    content_hash = hashlib.sha256(payload).hexdigest()
    identity = json.dumps(
        {
            **source.model_dump(mode="json", exclude={"verified_on", "status"}),
            "content_sha256": content_hash,
            "pipeline_version": PIPELINE_VERSION,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    document_id = hashlib.sha256(identity.encode()).hexdigest()
    upload = UploadFile(
        file=BytesIO(payload),
        headers=Headers({"content-type": content_type}),
    )
    try:
        if content_type == "application/json":
            documents = parse_html_manual(source, payload)
        elif content_type == "application/pdf":
            documents = PDFMinerParser(mode="page").parse(
                Blob(data=payload, mimetype=content_type)
            )
        else:
            documents = await process_document(upload)
        if content_type == "application/pdf":
            cleaned = []
            title = source.title
            for doc in documents:
                page = doc.metadata["page"] + 1
                if page in source.excluded_pages:
                    continue
                original = doc.page_content
                doc.page_content = clean_pdf_page(original, page)
                if not doc.page_content:
                    continue
                if "帮助指南" in original.splitlines()[:6]:
                    title = doc.page_content.splitlines()[0][:80]
                doc.metadata["section_title"] = title
                cleaned.append(doc)
            documents = cleaned
        documents = RecursiveCharacterTextSplitter(
            chunk_size=350, chunk_overlap=60
        ).split_documents(documents)
    finally:
        await upload.close()
    documents = [doc for doc in documents if doc.page_content.strip()]
    if not documents:
        raise ValueError("No text could be extracted from the document")
    chunks = [
        {
            "id": hashlib.sha256(f"{document_id}:{index}".encode()).hexdigest(),
            "content": doc.page_content,
            "embedding_text": (
                doc.metadata.get("section_title", "") + "\n" + doc.page_content
            ).strip(),
            "metadata": {
                "source_id": source.id,
                "document_id": document_id,
                "content_sha256": content_hash,
                "model": source.model,
                "manufacturer": source.manufacturer,
                "language": source.language,
                "region": source.region,
                "format": source.format,
                "machine_translated": source.machine_translated,
                "document_version": source.document_version,
                "firmware_version": source.firmware_version,
                "source_url": doc.metadata.get("source_url")
                or str(source.download_url or source.url),
                "page": doc.metadata.get("page", -1) + 1
                if "page" in doc.metadata
                else None,
                "pipeline_version": PIPELINE_VERSION,
                "chunk_index": index,
                "section_title": doc.metadata.get("section_title"),
            },
        }
        for index, doc in enumerate(documents)
    ]
    return {
        "schema_version": 2,
        "status": "prepared",
        "indexed": False,
        "provenance_verification": "operator_supplied_file",
        "prepared_at": datetime.now(UTC).isoformat(),
        "document_id": document_id,
        "chunks": chunks,
    }


def main() -> None:
    """Prepare a local source file and refuse to overwrite an existing artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--content-type", required=True, choices=CONTENT_TYPES)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sources = {source.id: source for source in load_sources()}
    if args.source_id not in sources:
        parser.error("Unknown source id; add a source to soundguide/sources.json first")
    if args.input.stat().st_size > MAX_BYTES:
        parser.error("Document exceeds 20 MiB")
    report = asyncio.run(
        prepare_document(
            sources[args.source_id], args.input.read_bytes(), args.content_type
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"Prepared {len(report['chunks'])} chunks; not embedded or indexed: {args.output}"
    )


if __name__ == "__main__":
    main()
