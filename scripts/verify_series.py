"""Run: uv run python scripts/verify_series.py --output .data/evals/series.json.

Verify actual indexed snapshots, model isolation and citation locations without generation.
This is an ingestion/retrieval smoke check, not an answer-accuracy benchmark.
"""

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from soundguide import PIPELINE_VERSION
from soundguide.catalog import Source, load_sources
from soundguide.evidence import EVIDENCE_ADAPTER, EvidenceChunk
from soundguide.settings import Settings, get_settings

ROOT = Path(__file__).resolve().parents[1]


QUERIES = ("如何配对和连接蓝牙设备", "初始化耳机以恢复出厂设置", "如何对耳机充电")


def describe_source(source: Source, state: dict, settings: Settings) -> dict:
    """Check committed, downloaded and prepared state before retrieving."""
    if state["status"] != "indexed" or state["chunks"] <= 0:
        raise RuntimeError(f"Missing current index for {source.model}")
    record = json.loads(
        (settings.data_dir / "raw" / source.id / "download.json").read_text()
    )
    if (
        hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
        != source.expected_sha256
    ):
        raise RuntimeError(f"Local snapshot changed for {source.model}")
    prepared = json.loads(
        (settings.data_dir / "prepared" / (source.id + ".json")).read_text()
    )
    if (
        prepared["embedding_profile"] != settings.profile
        or len(prepared["chunks"]) != state["chunks"]
    ):
        raise RuntimeError(f"Prepared/committed index mismatch for {source.model}")
    locations = {
        chunk["metadata"]["page"] or chunk["metadata"]["source_url"]
        for chunk in prepared["chunks"]
    }
    return {
        "model": source.model,
        "category": source.category,
        "format": source.format,
        "machine_translated": source.machine_translated,
        "manual_url": str(source.url),
        "download_url": str(source.download_url) if source.download_url else None,
        "sha256": record["sha256"],
        "bytes": record["bytes"],
        "fetched_at": record["fetched_at"],
        "chapters": record.get("chapters"),
        "excluded_pages": source.excluded_pages,
        "indexed_locations": len(locations),
        "indexed_chunks": state["chunks"],
        "indexed_at": state["indexed_at"],
    }


def citation_is_valid(hit: EvidenceChunk, source: Source) -> bool:
    """Check the location appropriate to this manual's format."""
    page = hit["metadata"]["page"]
    if source.format == "html_manual":
        chapter_prefix = str(source.url).rsplit("/", 1)[0] + "/contents/"
        return page is None and hit["citation_url"].startswith(chapter_prefix)
    return (
        page is not None
        and page > 0
        and page not in source.excluded_pages
        and hit["citation_url"] == f"{source.download_url}#page={page}"
    )


def evidence_errors(hit: EvidenceChunk, source: Source) -> list[str]:
    """Name each failing scope or location invariant."""
    expected = {
        "model": source.model,
        "source_id": source.id,
        "content_sha256": source.expected_sha256,
        "language": source.language,
        "machine_translated": source.machine_translated,
        "pipeline_version": PIPELINE_VERSION,
        "document_version": source.document_version,
    }
    mismatches = [
        key for key, value in expected.items() if hit["metadata"][key] != value
    ]
    if not citation_is_valid(hit, source):
        mismatches.append("citation_url/page")
    return mismatches


def verify_query(client: httpx.Client, source: Source, query: str) -> dict:
    """Retrieve evidence without generation and report the failing fields."""
    started = time.monotonic()
    response = client.post(
        "/api/search", json={"model": source.model, "query": query, "top_k": 3}
    )
    response.raise_for_status()
    hits = EVIDENCE_ADAPTER.validate_python(response.json()["results"], strict=True)
    if not hits:
        raise RuntimeError(f"No retrieved evidence for {source.model}: {query}")
    for hit in hits:
        mismatches = evidence_errors(hit, source)
        if mismatches:
            fields = ", ".join(mismatches)
            raise RuntimeError(
                f"Invalid evidence for {source.model}: {fields}; query: {query}"
            )
    return {
        "model": source.model,
        "query": query,
        "passed": True,
        "seconds": round(time.monotonic() - started, 4),
        "hits": [
            {
                "section_title": hit["metadata"]["section_title"],
                "page": hit["metadata"]["page"],
                "url": hit["citation_url"],
            }
            for hit in hits
        ],
    }


def collect_report(client: httpx.Client, settings: Settings) -> dict:
    """Verify each catalogued manual and summarize successful checks."""
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "notice": "Ingestion, retrieval connectivity and applicability checks only; no generation or semantic accuracy score.",
        "pipeline_version": PIPELINE_VERSION,
        "embedding_profile": settings.profile,
        "catalog_sha256": hashlib.sha256(
            (ROOT / "soundguide/sources.json").read_bytes()
        ).hexdigest(),
        "sources": [],
        "retrieval": [],
    }
    response = client.get("/api/sources")
    response.raise_for_status()
    inventory = {row["id"]: row for row in response.json()}
    for source in load_sources():
        state = inventory.get(source.id)
        if state is None:
            raise RuntimeError(f"Missing inventory entry for {source.model}")
        report["sources"].append(describe_source(source, state, settings))
        report["retrieval"].extend(
            verify_query(client, source, query) for query in QUERIES
        )
        print(
            f"{source.model}: {state['chunks']} indexed chunks; 3/3 scope checks passed",
            flush=True,
        )
    report["summary"] = {
        "indexed_models": len(report["sources"]),
        "indexed_chunks": sum(row["indexed_chunks"] for row in report["sources"]),
        "retrieval_checks": len(report["retrieval"]),
        "passed_checks": sum(row["passed"] for row in report["retrieval"]),
        "generation_calls": 0,
    }
    return report


def main() -> None:
    """Write the verification report after all manual checks succeed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = get_settings()
    output = args.output or settings.data_dir / "evals" / "series.json"
    with httpx.Client(base_url=args.base_url, timeout=90, trust_env=False) as client:
        report = collect_report(client, settings)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(report["summary"])


if __name__ == "__main__":
    main()
