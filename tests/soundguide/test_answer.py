"""Citation and model-output contract tests without real model requests."""

from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError
from pydantic import ValidationError

import soundguide.answer as answering
from soundguide.answer import Draft, InvalidAnswerError, generate, render
from soundguide.settings import Settings


@pytest.fixture
def hit():
    """Provide synthetic evidence, never an official product instruction."""
    return {
        "id": "test-chunk",
        "content": "仅用于测试:按住测试按钮七秒钟。不要执行此测试指令。",
        "metadata": {
            "model": "WH-1000XM5",
            "page": 140,
            "section_title": "测试章节",
            "document_version": "v1",
            "content_sha256": "a" * 64,
        },
        "citation_url": "https://helpguide.sony.net/test.pdf#page=140",
    }


def draft(text="按住测试按钮七秒钟。", evidence_id="E1"):
    """Build a candidate statement with explicit evidence references."""
    return Draft(
        status="answered",
        claims=[{"text": text, "evidence": [{"evidence_id": evidence_id}]}],
    )


def test_host_assigns_real_citation_urls_and_deduplicates_references(hit):
    """Repeated support should use the same citation label and actual source URL."""
    proposal = draft()
    proposal.claims.append(proposal.claims[0])
    result = render(proposal, [hit], "WH-1000XM5")
    assert len(result["citations"]) == 1
    assert result["answer"].count("[1]") == 2
    assert result["citations"][0]["url"] == hit["citation_url"]
    assert result["citations"][0]["chunk_id"] == "test-chunk"
    assert result["citations"][0]["quote"] == hit["content"]


@pytest.mark.parametrize("change", ["unknown", "blank", "wrong-model", "url", "label"])
def test_unverified_or_misowned_citations_are_rejected(hit, change):
    """Never publish a model-invented citation or a quote from a different model."""
    proposal = draft()
    if change == "unknown":
        proposal = draft(evidence_id="E9")
    if change == "blank":
        hit["content"] = " " * 8
    if change == "wrong-model":
        hit["metadata"]["model"] = "WH-1000XM4"
    if change == "url":
        proposal = draft(text="访问https://attacker.example查看操作")
    if change == "label":
        proposal = draft(text="操作步骤[9]")
    with pytest.raises(InvalidAnswerError):
        render(proposal, [hit], "WH-1000XM5")


def test_refusal_cannot_smuggle_uncited_claims():
    """Insufficient-evidence output must not contain unsupported product answers."""
    proposal = draft()
    proposal.status = "insufficient_evidence"
    with pytest.raises(InvalidAnswerError, match="claims"):
        render(proposal, [], "WH-1000XM5")


async def test_empty_retrieval_never_calls_the_model(monkeypatch):
    """A no-hit result is handled without model configuration or token usage."""

    def forbidden(**kwargs: object) -> None:
        raise AssertionError("Model should not be constructed")

    monkeypatch.setattr(answering, "ChatOpenAI", forbidden)
    result = await generate("unknown question", "WH-1000XM5", [], Settings())
    assert result["status"] == "insufficient_evidence"
    assert not result["citations"]


@pytest.mark.parametrize(
    "failure", ["truncated", "malformed", "provider", "sdk_length", "sdk_filter"]
)
async def test_failed_generation_is_not_published_and_not_retried(
    hit, monkeypatch, failure
):
    """Invalid completion or a provider failure must stop after one request."""
    calls = []

    class FakeModel:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def ainvoke(self, messages) -> SimpleNamespace:
            calls.append(messages)
            if failure == "provider":
                raise APIConnectionError(
                    request=httpx.Request("POST", "http://test-model")
                )
            if failure == "sdk_length":
                from openai.types.chat import ChatCompletion

                raise answering.LengthFinishReasonError(
                    completion=ChatCompletion(
                        id="fixture",
                        choices=[],
                        created=0,
                        model="fixture",
                        object="chat.completion",
                    )
                )
            if failure == "sdk_filter":
                raise answering.ContentFilterFinishReasonError
            return SimpleNamespace(
                content="not valid JSON",
                response_metadata={
                    "finish_reason": "length" if failure == "truncated" else "stop"
                },
                usage_metadata={},
            )

    monkeypatch.setattr(answering, "ChatOpenAI", FakeModel)
    settings = Settings(
        generation_url="http://test-model/v1",
        generation_model="fixture",
        generation_key="synthetic-key",
    )
    with pytest.raises((InvalidAnswerError, answering.GenerationUnavailableError)):
        await generate("测试", "WH-1000XM5", [hit], settings)
    assert len(calls) == 1


def test_model_cannot_supply_its_own_quote_or_url():
    """Quotations and links are backend-owned, not provider-controlled fields."""
    with pytest.raises(ValidationError):
        Draft(
            status="answered",
            claims=[
                {
                    "text": "test",
                    "evidence": [{"evidence_id": "E1", "quote": "invented quote"}],
                }
            ],
        )


@pytest.mark.parametrize("failure", ["citation", "provider"])
async def test_answer_route_hides_failed_model_output(
    hit, monkeypatch, tmp_path, failure
):
    """API errors must not expose unvalidated model text or provider details."""
    from soundguide import server
    from soundguide.server import APP
    from soundguide.settings import get_settings

    async def retrieved(*args: object, **kwargs: object) -> list[dict]:
        return [hit]

    async def rejected(*args: object, **kwargs: object) -> None:
        if failure == "citation":
            raise InvalidAnswerError("unvalidated private draft")
        raise answering.GenerationUnavailableError("private provider detail")

    async def inventory(*args: object) -> list[dict]:
        return [
            {
                "model": "WH-1000XM5",
                "language": "zh-CN",
                "document_version": "v1",
                "status": "indexed",
            }
        ]

    monkeypatch.setattr(server, "retrieve", retrieved)
    monkeypatch.setattr(server, "sources", inventory)
    monkeypatch.setattr(server, "generate", rejected)
    APP.dependency_overrides[get_settings] = lambda: Settings(
        database_url="unused",
        embedding_key="fake",
        embedding_revision="fake",
        data_dir=tmp_path,
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=APP), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/answer", json={"model": "WH-1000XM5", "question": "测试问题"}
            )
        assert response.status_code == (502 if failure == "citation" else 503)
        assert "private" not in response.text
        assert "citations" not in response.json()
    finally:
        APP.dependency_overrides.clear()


@pytest.mark.parametrize(
    "question",
    [
        "怎么设置\uff1f",
        "连不上。",
        "声音不对",
        "我想调整一下。",
        "连接有问题。",
        "声音不正常。",
    ],
)
async def test_vague_question_is_clarified_without_retrieval_or_model(
    monkeypatch, tmp_path, question
):
    """Known empty-intent inputs should ask a useful question at zero model cost."""
    from soundguide import server
    from soundguide.server import APP
    from soundguide.settings import get_settings

    async def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("No retrieval or model call is needed")

    monkeypatch.setattr(server, "retrieve", forbidden)
    monkeypatch.setattr(server, "generate", forbidden)
    APP.dependency_overrides[get_settings] = lambda: Settings(data_dir=tmp_path)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=APP), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/answer",
                json={"model": "WH-1000XM5", "question": question},
            )
        assert response.status_code == 200
        assert response.json()["status"] == "clarification_needed"
        assert response.json()["model_called"] is False
        assert response.json()["citations"] == []
    finally:
        APP.dependency_overrides.clear()
