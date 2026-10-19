"""Deterministic lexical ranking and fusion tests on synthetic documents."""

from soundguide.retrieval import bm25_scores, expand_query, fuse, tokenize


def test_tokenization_and_explicit_domain_aliases():
    """Map colloquial feature descriptions to the manuals' vocabulary."""
    assert "多点" in tokenize("多点连接")
    assert "mac" in tokenize("连接MAC电脑")
    assert "多点连接" in expand_query("如何同时连接手机和电脑")
    assert "初始化" in expand_query("恢复出厂设置")
    assert expand_query("打开耳机") == "打开耳机"


def test_bm25_matches_specific_terms_and_length_normalizes():
    """Prefer a relevant short document to unrelated or excessively padded text."""
    scores = bm25_scores(
        "多点连接", ["多点连接", "多点连接 " + "普通文字 " * 100, "恢复出厂设置"]
    )
    assert scores[0] > scores[1] > scores[2]
    assert bm25_scores("query", []) == []
    assert bm25_scores("query", [""]) == [0]


def test_fusion_deduplicates_and_limits_one_page_coverage():
    """Avoid duplicate chunk ids and prevent one page from consuming the context."""
    rows = [
        {
            "id": str(i),
            "content": "多点连接说明",
            "metadata": {
                "source_id": "synthetic",
                "page": 1 if i < 4 else 2,
                "section_title": "多点连接",
            },
            "cosine_similarity": 1 - i * 0.01,
        }
        for i in range(5)
    ]
    hits = fuse(rows, "同时连接", 5)
    assert len(hits) == 3
    assert len({hit["id"] for hit in hits}) == len(hits)
    assert sum(hit["metadata"]["page"] == 1 for hit in hits) == 2
    assert all(hit["rrf_score"] > 0 for hit in hits)
