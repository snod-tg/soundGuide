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
from soundguide.catalog import load_sources
from soundguide.settings import get_settings

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Fail on missing/stale indexes, mixed model evidence or invalid citation locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--output", type=Path, default=ROOT / ".data/evals/series.json")
    args = parser.parse_args()
    settings = get_settings()
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
    with httpx.Client(base_url=args.base_url, timeout=90, trust_env=False) as client:
        response = client.get("/api/sources")
        response.raise_for_status()
        inventory = {row["id"]: row for row in response.json()}
        for source in load_sources():
            state = inventory[source.id]
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
                raise RuntimeError(
                    f"Prepared/committed index mismatch for {source.model}"
                )
            locations = {
                chunk["metadata"]["page"] or chunk["metadata"]["source_url"]
                for chunk in prepared["chunks"]
            }
            row = {
                "model": source.model,
                "category": source.category,
                "format": source.format,
                "machine_translated": source.machine_translated,
                "manual_url": str(source.url),
                "download_url": str(source.download_url)
                if source.download_url
                else None,
                "sha256": record["sha256"],
                "bytes": record["bytes"],
                "fetched_at": record["fetched_at"],
                "chapters": record.get("chapters"),
                "excluded_pages": source.excluded_pages,
                "indexed_locations": len(locations),
                "indexed_chunks": state["chunks"],
                "indexed_at": state["indexed_at"],
            }
            report["sources"].append(row)
            for query in (
                "如何配对和连接蓝牙设备",
                "初始化耳机以恢复出厂设置",
                "如何对耳机充电",
            ):
                started = time.monotonic()
                response = client.post(
                    "/api/search",
                    json={"model": source.model, "query": query, "top_k": 3},
                )
                response.raise_for_status()
                hits = response.json()["results"]
                passed = bool(hits) and all(
                    hit["metadata"]["model"] == source.model
                    and hit["metadata"]["source_id"] == source.id
                    and hit["metadata"]["content_sha256"] == source.expected_sha256
                    and hit["metadata"]["language"] == source.language
                    and hit["metadata"]["machine_translated"]
                    == source.machine_translated
                    and hit["metadata"]["pipeline_version"] == PIPELINE_VERSION
                    and (
                        hit["metadata"]["page"] is None
                        and hit["citation_url"].startswith(
                            str(source.url).rsplit("/", 1)[0] + "/contents/"
                        )
                        if source.format == "html_manual"
                        else hit["metadata"]["page"] not in source.excluded_pages
                        and hit["citation_url"]
                        == str(source.download_url) + f"#page={hit['metadata']['page']}"
                    )
                    for hit in hits
                )
                report["retrieval"].append(
                    {
                        "model": source.model,
                        "query": query,
                        "passed": passed,
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
                )
                if not passed:
                    raise RuntimeError(
                        f"Retrieval scope/location check failed for {source.model}"
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(report["summary"])


if __name__ == "__main__":
    main()
