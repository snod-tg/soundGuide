"""Run: uv run python scripts/doctor.py. Read-only diagnosis; never starts model processes."""

import asyncio
from urllib.parse import urlsplit

import asyncpg
import httpx

from soundguide.settings import get_settings


async def main():
    """Check independent dependencies while keeping credentials and provider errors private."""
    settings = get_settings()
    healthy = True
    try:
        settings.require_ready()
        settings.require_generation()
    except ValueError:
        print("FAIL configuration: check private .env against .env.example")
        raise SystemExit(1) from None
    async with httpx.AsyncClient(trust_env=False, timeout=8) as client:
        for name, url, key in (
            ("embedding", settings.embedding_url.removesuffix("/v1") + "/health", ""),
            (
                "generation",
                settings.generation_url + "/models",
                settings.generation_key,
            ),
            ("application", "http://127.0.0.1:8001/api/sources", ""),
        ):
            try:
                response = await client.get(
                    url, headers={"Authorization": "Bearer " + key} if key else {}
                )
                response.raise_for_status()
                data = response.json()
                if name == "embedding" and (
                    data.get("model"),
                    data.get("revision"),
                    data.get("dimensions"),
                ) != (
                    settings.embedding_model,
                    settings.embedding_revision,
                    settings.dimensions,
                ):
                    raise ValueError("Index profile mismatch")
                if name == "generation" and settings.generation_model not in {
                    item["id"] for item in data.get("data", [])
                }:
                    raise ValueError("Configured generation model is missing")
                if name == "application":
                    count = sum(item["status"] == "indexed" for item in data)
                    print(f"OK application: {count}/{len(data)} manuals indexed")
                else:
                    print(
                        f"OK {name}: {urlsplit(url).hostname}:{urlsplit(url).port or 443}"
                    )
            except (httpx.HTTPError, ValueError, KeyError):
                healthy = False
                print(
                    f"FAIL {name}: check its service/API connection"
                    + (" and SSH forwarding" if name == "generation" else "")
                )
    try:
        connection = await asyncpg.connect(settings.database_url, timeout=5)
        try:
            await connection.fetchval("SELECT 1")
        finally:
            await connection.close()
        print("OK database")
    except (asyncpg.PostgresError, OSError):
        healthy = False
        print("FAIL database: check the dedicated SoundGuide cluster and credentials")
    if not healthy:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
