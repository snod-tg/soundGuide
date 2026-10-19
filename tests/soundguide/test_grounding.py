"""Regression checks for evidence boundaries, output constraints and benchmark metrics."""

import json
from types import SimpleNamespace

import soundguide.answer as answering
from soundguide.retrieval import add_neighbors, fuse
from soundguide.settings import Settings


def test_explicit_initialization_prefers_instructions_to_navigation_mentions():
    """Do not confuse a cross-reference mentioning an operation with that operation."""
    rows = [
        {
            "id": str(i),
            "content": "删除配对信息请参考初始化耳机以恢复出厂设置。",
            "metadata": {
                "source_id": "synthetic",
                "page": i,
                "section_title": "手机配对" if i < 3 else "初始化耳机以恢复出厂设置",
            },
        }
        for i in range(4)
    ]
    assert fuse(rows, "初始化时有哪些条件", 3)[0]["id"] == "3"


def test_neighbors_preserve_top_three_and_never_cross_sources_or_pages():
    """A missing condition may be nearby, but another product/page must not be attached."""

    def row(index: int, page: int = 1, source: str = "source-a") -> dict:
        return {
            "id": f"{source}-{page}-{index}",
            "content": str(index),
            "metadata": {
                "source_id": source,
                "page": page,
                "source_url": "https://helpguide.sony.net/test.pdf",
                "section_title": "测试章节",
                "chunk_index": index,
            },
        }

    seeds = [row(10), row(20, 2), row(30, 3)]
    other_page = row(11, 4)
    other_source = row(9, source="source-b")
    neighbor = row(11)
    result = add_neighbors(seeds, [*seeds, neighbor, other_page, other_source], 8)
    assert result[:3] == seeds
    assert [r["id"] for r in result] == [r["id"] for r in seeds] + [neighbor["id"]]
    assert result[-1]["content"] == neighbor["content"]
    assert result[-1]["context_neighbor_of"] == seeds[0]["id"]


def test_neighbors_do_not_evict_lower_ranked_evidence():
    """A relevant rank-eight instruction must not be replaced by rank-one context."""

    def row(index: int) -> dict:
        return {
            "id": str(index),
            "content": str(index),
            "metadata": {
                "source_id": "a",
                "page": 1,
                "source_url": "a",
                "section_title": "same",
                "chunk_index": index,
            },
        }

    seeds = [row(i * 10) for i in range(8)]
    assert add_neighbors(seeds, [*seeds, row(1), row(2)], 8) == seeds


async def test_real_output_schema_bounds_references_before_host_validation(monkeypatch):
    """Constrain actual provider output to existing ids and at most three references."""
    calls = []

    class FakeModel:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

        async def ainvoke(self, messages) -> SimpleNamespace:
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "status": "insufficient_evidence",
                        "claims": [],
                        "clarification": None,
                    }
                ),
                response_metadata={"finish_reason": "stop"},
                usage_metadata={},
            )

    monkeypatch.setattr(answering, "ChatOpenAI", FakeModel)
    hit = {
        "content": "仅用于测试的证据",
        "metadata": {"section_title": "测试章节", "page": 1, "document_version": "v1"},
    }
    await answering.generate(
        "测试",
        "WH-1000XM5",
        [hit],
        Settings(
            generation_url="http://test/v1",
            generation_model="test",
            generation_key="synthetic",
        ),
    )
    response_format = calls[0]["model_kwargs"]["response_format"]
    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]["schema"]
    assert schema["$defs"]["Reference"]["properties"]["evidence_id"]["enum"] == ["E1"]
    assert schema["$defs"]["Claim"]["properties"]["evidence"]["maxItems"] == 3
