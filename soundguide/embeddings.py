"""LangChain Embeddings implementation backed exclusively by an external HTTP API."""

import asyncio
import math

import httpx
from langchain_core.embeddings import Embeddings

from soundguide.settings import Settings

MAX_CHARACTERS = 450


class HttpEmbeddings(Embeddings):
    """Validate model revision, response cardinality and vector dimensions."""

    def __init__(self, settings: Settings) -> None:
        """Bind the explicit external provider configuration."""
        self.settings = settings
        settings.require_ready()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed bounded batches without silently truncating supplied documents."""
        if any(not text.strip() or len(text) > MAX_CHARACTERS for text in texts):
            raise ValueError("Embedding input must contain 1 to 450 characters")
        s = self.settings
        async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
            health = await client.get(s.embedding_url.removesuffix("/v1") + "/health")
            health.raise_for_status()
            info = health.json()
            if (info.get("model"), info.get("revision"), info.get("dimensions")) != (
                s.embedding_model,
                s.embedding_revision,
                s.dimensions,
            ):
                raise ValueError(
                    "Embedding provider does not match the configured index profile"
                )
            vectors = []
            for start in range(0, len(texts), 16):
                batch = texts[start : start + 16]
                response = await client.post(
                    s.embedding_url + "/embeddings",
                    headers={"Authorization": "Bearer " + s.embedding_key},
                    json={
                        "model": s.embedding_model,
                        "input": batch,
                        "encoding_format": "float",
                    },
                )
                response.raise_for_status()
                rows = response.json()["data"]
                if len(rows) != len(batch) or sorted(
                    row["index"] for row in rows
                ) != list(range(len(batch))):
                    raise ValueError("Embedding response cardinality/index mismatch")
                for row in sorted(rows, key=lambda item: item["index"]):
                    vector = row["embedding"]
                    if (
                        len(vector) != s.dimensions
                        or any(
                            not isinstance(v, (float, int))
                            or isinstance(v, bool)
                            or not math.isfinite(v)
                            for v in vector
                        )
                        or not any(v != 0 for v in vector)
                    ):
                        raise ValueError(
                            "Embedding response contains an invalid vector"
                        )
                    vectors.append(vector)
            return vectors

    async def aembed_query(self, text: str) -> list[float]:
        """Apply BGE's short-query instruction only to queries."""
        return (await self.aembed_documents([self.settings.query_prefix + text]))[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Provide LangChain's synchronous interface."""
        return asyncio.run(self.aembed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        """Provide LangChain's synchronous query interface."""
        return asyncio.run(self.aembed_query(text))
