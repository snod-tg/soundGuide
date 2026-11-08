"""LangChain Embeddings implementation backed exclusively by an external HTTP API."""

import asyncio
import math

import httpx
from langchain_core.embeddings import Embeddings

from soundguide.errors import EmbeddingContractError, InvalidQueryError
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
            raise InvalidQueryError("Embedding input must contain 1 to 450 characters")
        s = self.settings
        async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
            await self.verify_provider_profile(client)
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
                vectors.extend(read_vectors(response, len(batch), s.dimensions))
            return vectors

    async def verify_provider_profile(self, client: httpx.AsyncClient) -> None:
        """Keep revision verification separate from batch requests."""
        s = self.settings
        response = await client.get(s.embedding_url.removesuffix("/v1") + "/health")
        response.raise_for_status()
        info = response_object(response)
        if (info.get("model"), info.get("revision"), info.get("dimensions")) != (
            s.embedding_model,
            s.embedding_revision,
            s.dimensions,
        ):
            raise EmbeddingContractError(
                "Embedding provider does not match the configured index profile"
            )

    async def aembed_query(self, text: str) -> list[float]:
        """Apply BGE's short-query instruction only to queries."""
        return (await self.aembed_documents([self.settings.query_prefix + text]))[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Provide LangChain's synchronous interface."""
        return asyncio.run(self.aembed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        """Provide LangChain's synchronous query interface."""
        return asyncio.run(self.aembed_query(text))


def response_object(response: httpx.Response) -> dict:
    """Require a JSON object rather than exposing decoder or attribute errors."""
    try:
        data = response.json()
    except ValueError as exc:
        raise EmbeddingContractError(
            "Embedding provider did not return valid JSON"
        ) from exc
    if not isinstance(data, dict):
        raise EmbeddingContractError("Embedding provider did not return a JSON object")
    return data


def read_vectors(
    response: httpx.Response, size: int, dimensions: int
) -> list[list[float]]:
    """Validate response indexes before restoring input order."""
    rows = response_object(response).get("data")
    if (
        not isinstance(rows, list)
        or len(rows) != size
        or not all(
            isinstance(row, dict) and type(row.get("index")) is int for row in rows
        )
    ):
        raise EmbeddingContractError("Embedding response cardinality/index mismatch")
    ordered = sorted(rows, key=lambda row: row["index"])
    if [row["index"] for row in ordered] != list(range(size)):
        raise EmbeddingContractError("Embedding response cardinality/index mismatch")
    vectors = []
    for row in ordered:
        vector = row.get("embedding")
        if not valid_vector(vector, dimensions):
            raise EmbeddingContractError(
                "Embedding response contains an invalid vector"
            )
        vectors.append(vector)
    return vectors


def valid_vector(vector: object, dimensions: int) -> bool:
    """Reject missing, malformed and zero vectors without coercion."""
    return (
        isinstance(vector, list)
        and len(vector) == dimensions
        and all(
            type(value) in (int, float) and math.isfinite(value) for value in vector
        )
        and any(value != 0 for value in vector)
    )
