"""Call the configured model once for an evidence-grounded answer."""

import json

import httpx
from langchain_openai import ChatOpenAI
from openai import APIError, ContentFilterFinishReasonError, LengthFinishReasonError

from soundguide.citations import Draft, InvalidAnswerError, render
from soundguide.evidence import EvidenceChunk
from soundguide.prompts import PROMPT
from soundguide.settings import Settings


class GenerationUnavailableError(RuntimeError):
    """External model transport/provider failure without leaking its response."""


async def generate(
    question: str, model: str, hits: list[EvidenceChunk], settings: Settings
) -> dict:
    """Make at most one external model call; never expose unvalidated model output."""
    if not hits:
        return {
            **render(Draft(status="insufficient_evidence", claims=[]), [], model),
            "model_called": False,
        }
    settings.require_generation()
    schema = Draft.model_json_schema()
    schema["$defs"]["Reference"]["properties"]["evidence_id"] = {
        "type": "string",
        "enum": [f"E{i}" for i in range(1, len(hits) + 1)],
    }
    evidence = [
        {
            "id": f"E{i}",
            "model": model,
            "section_title": hit["metadata"].get("section_title"),
            "page": hit["metadata"].get("page"),
            "document_version": hit["metadata"]["document_version"],
            "firmware_version": hit["metadata"].get("firmware_version"),
            "machine_translated": hit["metadata"].get("machine_translated", False),
            "content": hit["content"],
        }
        for i, hit in enumerate(hits, 1)
    ]
    messages = PROMPT.format_messages(
        model=model,
        question=question,
        evidence=json.dumps(evidence, ensure_ascii=False, separators=(",", ":")),
    )
    async with httpx.AsyncClient(trust_env=False, timeout=60) as client:
        llm = ChatOpenAI(
            model=settings.generation_model,
            base_url=settings.generation_url,
            api_key=settings.generation_key,
            temperature=0,
            max_completion_tokens=settings.generation_max_tokens,
            max_retries=0,
            timeout=60,
            use_responses_api=False,
            http_async_client=client,
            model_kwargs={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "soundguide_answer",
                        "strict": True,
                        "schema": schema,
                    },
                }
            },
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        try:
            response = await llm.ainvoke(messages)
        except (LengthFinishReasonError, ContentFilterFinishReasonError) as exc:
            raise InvalidAnswerError("Generation did not finish normally") from exc
        except APIError as exc:
            raise GenerationUnavailableError(
                "Generation model API is unavailable"
            ) from exc
    if response.response_metadata.get("finish_reason") != "stop" or not isinstance(
        response.content, str
    ):
        raise InvalidAnswerError("Generation did not finish normally")
    try:
        draft = Draft.model_validate_json(response.content)
    except ValueError as exc:
        raise InvalidAnswerError("Generation did not match the output schema") from exc
    result = render(draft, hits, model)
    return {
        **result,
        "model_called": True,
        "generation_model": settings.generation_model,
        "usage": response.usage_metadata,
    }
