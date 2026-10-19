"""Small-corpus Chinese BM25 and vector retrieval fused with reciprocal ranks."""

import math
import re
from collections import Counter, defaultdict

from soundguide.embeddings import HttpEmbeddings
from soundguide.settings import Settings
from soundguide.store import Store

RRF_K = 60
CANDIDATE_LIMIT = 30
MAX_CORPUS = 10000
MAX_PER_PAGE = 2
BM25_K1 = 1.2
BM25_B = 0.75
NEIGHBOR_WINDOW = 2


def tokenize(text: str) -> list[str]:
    """Use Chinese character bigrams plus Latin/numeric terms; no dictionary download."""
    tokens = []
    for part in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            tokens.extend(part[i : i + 2] for i in range(len(part) - 1))
            if len(part) == 1:
                tokens.append(part)
        else:
            tokens.append(part)
    return tokens


def expand_query(query: str) -> str:
    """Apply explicit domain vocabulary aliases, not an unbounded model rewrite."""
    additions = []
    if "同时" in query and ("连接" in query or "连" in query):
        additions.append("多点连接 连接2个设备")
    if "恢复出厂" in query:
        additions.append("初始化耳机")
    if "听不到声音" in query or "没声音" in query:
        additions.append("没有声音")
    if "广播" in query:
        additions.append("快速提醒模式")
    if "耳塞" in query and any(word in query for word in ("清洁", "清洗", "水洗")):
        additions.append("更换耳塞 干布清洁")
    return query + " " + " ".join(additions) if additions else query


def intent_titles(query: str) -> tuple[str, ...]:
    """Prefer an explicit requested operation over navigation mentions of that operation."""
    if "初始化" in query or "恢复出厂" in query:
        return ("初始化耳机以恢复出厂设置",)
    if "广播" in query or "快速提醒" in query:
        return ("快速听到环境声音\uff08快速提醒模式\uff09",)
    if "耳塞" in query and any(word in query for word in ("清洁", "清洗", "水洗")):
        return ("更换耳塞", "保养")
    if "充电" in query and any(
        word in query for word in ("开机", "听歌", "播放", "接口")
    ):
        return ("对耳机充电", "充电")
    return ()


def bm25_scores(query: str, texts: list[str]) -> list[float]:
    """Compute length-normalized BM25 with positive Robertson-style IDF."""
    bags = [Counter(tokenize(text)) for text in texts]
    if not bags:
        return []
    average = sum(sum(bag.values()) for bag in bags) / len(bags)
    if not average:
        return [0.0] * len(bags)
    scores = [0.0] * len(bags)
    for term in set(tokenize(query)):
        frequency = sum(term in bag for bag in bags)
        if not frequency:
            continue
        inverse = math.log(1 + (len(bags) - frequency + 0.5) / (frequency + 0.5))
        for index, bag in enumerate(bags):
            count = bag.get(term, 0)
            if count:
                normalization = BM25_K1 * (
                    1 - BM25_B + BM25_B * sum(bag.values()) / average
                )
                scores[index] += (
                    inverse * count * (BM25_K1 + 1) / (count + normalization)
                )
    return scores


def fuse(candidates: list[dict], query: str, top_k: int) -> list[dict]:
    """Fuse bounded rank lists, retaining at most two chunks from one PDF page."""
    texts = [
        (row["metadata"].get("section_title") or "") * 3 + " " + row["content"]
        for row in candidates
    ]
    lexical = bm25_scores(expand_query(query), texts)
    sparse = sorted(
        (i for i, score in enumerate(lexical) if score > 0),
        key=lambda i: (-lexical[i], candidates[i]["id"]),
    )[:CANDIDATE_LIMIT]
    scores = defaultdict(float)
    ranks = defaultdict(dict)
    for method, indices in (
        ("vector", list(range(min(CANDIDATE_LIMIT, len(candidates))))),
        ("bm25", sparse),
    ):
        for rank, index in enumerate(indices, 1):
            scores[index] += 1 / (RRF_K + rank)
            ranks[index][method] = rank
    preferred = intent_titles(query)
    for index in scores:
        if candidates[index]["metadata"].get("section_title") in preferred:
            scores[index] += 2 / (RRF_K + 1)
    pages = Counter()
    results = []
    for index in sorted(scores, key=lambda i: (-scores[i], candidates[i]["id"])):
        row = candidates[index]
        page = (
            row["metadata"]["source_id"],
            row["metadata"].get("page") or row["metadata"].get("source_url"),
        )
        if pages[page] >= MAX_PER_PAGE:
            continue
        pages[page] += 1
        results.append(
            {
                **row,
                "rrf_score": scores[index],
                "bm25_score": lexical[index],
                "ranks": ranks[index],
            }
        )
        if len(results) == top_k:
            break
    return results


def add_neighbors(seeds: list[dict], candidates: list[dict], limit: int) -> list[dict]:
    """Preserve every ranked seed; neighbors may only fill unused evidence slots."""
    result = seeds[:limit]
    if len(result) >= limit:
        return result
    seen = {row["id"] for row in result}
    for anchor in result[:3]:
        metadata = anchor["metadata"]
        neighbors = [
            row
            for row in candidates
            if row["id"] not in seen
            and row["metadata"]["source_id"] == metadata["source_id"]
            and row["metadata"].get("section_title") == metadata.get("section_title")
            and row["metadata"].get("page") == metadata.get("page")
            and row["metadata"].get("source_url") == metadata.get("source_url")
            and abs(row["metadata"]["chunk_index"] - metadata["chunk_index"])
            <= NEIGHBOR_WINDOW
        ]
        neighbors.sort(
            key=lambda row: (
                abs(row["metadata"]["chunk_index"] - metadata["chunk_index"]),
                row["metadata"]["chunk_index"],
            )
        )
        for row in neighbors[:2]:
            if len(result) >= limit:
                return result
            result.append({**row, "context_neighbor_of": anchor["id"]})
            seen.add(row["id"])
    for row in seeds:
        if row["id"] not in seen and len(result) < limit:
            result.append(row)
            seen.add(row["id"])
    return result


async def retrieve(  # noqa: PLR0913
    query: str,
    model: str,
    top_k: int,
    language: str,
    version: str,
    strategy: str,
    settings: Settings,
) -> list[dict]:
    """Read one model's candidates and never mix embedding profiles or models."""
    if strategy not in {"vector", "hybrid"}:
        raise ValueError("Unknown retrieval strategy")
    vector = await HttpEmbeddings(settings).aembed_query(
        expand_query(query) if strategy == "hybrid" else query
    )
    rows = await Store(settings).search(
        vector,
        model,
        top_k if strategy == "vector" else MAX_CORPUS + 1,
        language,
        version,
    )
    if strategy == "hybrid":
        if len(rows) > MAX_CORPUS:
            raise ValueError("Corpus exceeds the in-memory BM25 limit")
        rows = add_neighbors(fuse(rows, query, top_k), rows, top_k)
    for row in rows:
        metadata = row["metadata"]
        row["citation_url"] = metadata["source_url"] + (
            f"#page={metadata['page']}" if metadata.get("page") else ""
        )
    return rows
