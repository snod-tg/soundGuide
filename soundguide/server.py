"""SoundGuide source status and filtered vector retrieval API."""

from pathlib import Path
from typing import Annotated, Literal

import asyncpg
import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
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
from soundguide.errors import (
    ConfigurationError,
    EmbeddingContractError,
    IndexNotReadyError,
    InvalidQueryError,
    ManifestError,
    RetrievalLimitError,
    SoundGuideError,
)
from soundguide.inventory import list_source_statuses
from soundguide.retrieval import expand_query, retrieve
from soundguide.settings import Settings, get_settings

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


ERROR_STATUS = {
    ConfigurationError: 503,
    EmbeddingContractError: 502,
    IndexNotReadyError: 409,
    InvalidQueryError: 422,
    ManifestError: 503,
    RetrievalLimitError: 503,
}


@APP.exception_handler(SoundGuideError)
async def application_error(_request: Request, exc: SoundGuideError) -> JSONResponse:
    """Translate expected service errors at the HTTP boundary."""
    return JSONResponse({"detail": str(exc)}, status_code=ERROR_STATUS[type(exc)])


async def dependency_unavailable(_request: Request, _exc: Exception) -> JSONResponse:
    """Keep infrastructure error details out of the public response."""
    return JSONResponse(
        {"detail": "SoundGuide dependency is unavailable"}, status_code=503
    )


for error_type in (httpx.HTTPError, OSError, asyncpg.PostgresError):
    APP.add_exception_handler(error_type, dependency_unavailable)


@APP.get("/api/sources")
async def sources(settings: Annotated[Settings, Depends(get_settings)]) -> list[dict]:
    """Report downloaded and committed indexed state, never just planned state."""
    return await list_source_statuses(settings)


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
    hits = await retrieve(
        body.query,
        body.model,
        body.top_k,
        body.language,
        body.document_version,
        body.strategy,
        settings,
    )
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
    hits = await retrieve(
        body.question, body.model, 8, "zh-CN", "v1", body.strategy, settings
    )
    try:
        response = await generate(body.question, body.model, hits, settings)
    except InvalidAnswerError as exc:
        raise HTTPException(
            502,
            "Model output failed citation or completeness validation; no answer was published",
        ) from exc
    except (GenerationUnavailableError, httpx.HTTPError, OSError) as exc:
        raise HTTPException(503, "Generation model API is unavailable") from exc
    return {
        "question": body.question,
        "model": body.model,
        **response,
        "retrieval": {
            "strategy": body.strategy,
            "embedding_profile": settings.profile,
            "evidence_count": len(hits),
        },
    }
