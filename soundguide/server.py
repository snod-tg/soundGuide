"""SoundGuide source status and filtered vector retrieval API."""

import json
from pathlib import Path
from typing import Annotated, Literal

import asyncpg
import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from soundguide import __version__
from soundguide.answer import (
    GenerationUnavailableError,
    InvalidAnswerError,
    clarification_for,
    generate,
)
from soundguide.catalog import load_sources
from soundguide.retrieval import expand_query, retrieve
from soundguide.settings import Settings, get_settings
from soundguide.store import Store

APP = FastAPI(
    title="SoundGuide API",
    description="耳机官方手册检索与带证据引用的问答。",
    version=__version__,
)
WEB_DIR = Path(__file__).with_name("web")
APP.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@APP.get("/", include_in_schema=False)
async def home() -> FileResponse:
    """Serve the single-turn demo without another frontend service."""
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@APP.get("/health")
async def health() -> dict:
    """Report process liveness, not database/model readiness."""
    return {"status": "ok", "service": "soundguide"}


@APP.get("/api/sources")
async def sources(settings: Annotated[Settings, Depends(get_settings)]) -> list[dict]:
    """Report downloaded and committed indexed state, never just planned state."""
    states = {}
    if settings.database_url:
        try:
            states = {row["source_id"]: row for row in await Store(settings).status()}
        except (ValueError, OSError, asyncpg.PostgresError) as exc:
            raise HTTPException(503, "SoundGuide storage is unavailable") from exc
    result = []
    for source in load_sources():
        item = source.model_dump(mode="json")
        item.update(status="candidate", chunks=0)
        manifest = settings.data_dir / "raw" / source.id / "download.json"
        downloaded = None
        if manifest.exists():
            downloaded = json.loads(manifest.read_text())
            item["status"] = "downloaded"
        if source.id in states:
            state = states[source.id]
            item.update(
                status="indexed"
                if state["profile"] == settings.profile
                and state["chunks"] > 0
                and state["content_sha256"] == source.expected_sha256
                and (
                    downloaded is None
                    or downloaded.get("sha256") == state["content_sha256"]
                )
                else "needs_reindex",
                chunks=state["chunks"],
                indexed_at=state["indexed_at"],
            )
        result.append(item)
    return result


class ProductRequest(BaseModel):
    """Accept catalogued models only; no duplicated hard-coded model lists."""

    model: str = Field(min_length=1, max_length=40)

    @field_validator("model")
    @classmethod
    def known_model(cls, value: str) -> str:
        """Reject unknown models before reaching an external service."""
        if value not in {source.model for source in load_sources()}:
            raise ValueError("Unknown product model")
        return value


class SearchRequest(ProductRequest):
    """An explicit model is required to prevent mixing near-identical manuals."""

    query: str = Field(min_length=1, max_length=300)
    strategy: Literal["vector", "hybrid"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=20)
    language: Literal["zh-CN"] = "zh-CN"
    document_version: Literal["v1"] = "v1"


@APP.post("/api/search")
async def search(
    body: SearchRequest, settings: Annotated[Settings, Depends(get_settings)]
) -> dict:
    """Return real retrieved evidence with exact applicability filters and PDF pages."""
    if not body.query.strip():
        raise HTTPException(422, "Query must contain non-whitespace text")
    try:
        settings.require_ready()
        inventory = await sources(settings)
        if not any(
            item["model"] == body.model
            and item["language"] == body.language
            and item["document_version"] == body.document_version
            and item["status"] == "indexed"
            for item in inventory
        ):
            raise HTTPException(
                409, "This model's manual is not indexed or needs reindexing"
            )
        hits = await retrieve(
            body.query,
            body.model,
            body.top_k,
            body.language,
            body.document_version,
            body.strategy,
            settings,
        )
    except ValueError as exc:
        raise HTTPException(
            409,
            "Embedding configuration/index mismatch; check configuration or reindex",
        ) from exc
    except (httpx.HTTPError, OSError, asyncpg.PostgresError) as exc:
        raise HTTPException(503, "SoundGuide retrieval service is unavailable") from exc
    return {
        "query": body.query,
        "model": body.model,
        "embedding_profile": settings.profile,
        "strategy": body.strategy,
        "expanded_query": expand_query(body.query)
        if body.strategy == "hybrid"
        else body.query,
        "results": hits,
    }


class AnswerRequest(ProductRequest):
    """A single-turn question with an explicit product model."""

    question: str = Field(min_length=1, max_length=300)
    strategy: Literal["vector", "hybrid"] = "hybrid"


@APP.post("/api/answer")
async def answer(
    body: AnswerRequest, settings: Annotated[Settings, Depends(get_settings)]
) -> dict:
    """Retrieve eligible evidence, generate once, and validate all returned citations."""
    clarification = clarification_for(body.question)
    if clarification:
        return {
            "question": body.question,
            "model": body.model,
            "status": "clarification_needed",
            "answer": clarification,
            "citations": [],
            "model_called": False,
            "retrieval": {"strategy": "none", "evidence_count": 0},
        }
    retrieval = await search(
        SearchRequest(
            query=body.question, model=body.model, top_k=8, strategy=body.strategy
        ),
        settings,
    )
    try:
        response = await generate(
            body.question, body.model, retrieval["results"], settings
        )
    except InvalidAnswerError as exc:
        raise HTTPException(
            502,
            "Model output failed citation or completeness validation; no answer was published",
        ) from exc
    except ValueError as exc:
        raise HTTPException(503, "Generation model API is not configured") from exc
    except (GenerationUnavailableError, httpx.HTTPError, OSError) as exc:
        raise HTTPException(503, "Generation model API is unavailable") from exc
    return {
        "question": body.question,
        "model": body.model,
        **response,
        "retrieval": {
            "strategy": body.strategy,
            "embedding_profile": settings.profile,
            "evidence_count": len(retrieval["results"]),
        },
    }
