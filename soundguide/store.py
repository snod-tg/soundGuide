"""Transactional source replacement and exact cosine search using pgvector."""

import json
from contextlib import asynccontextmanager

import asyncpg

from soundguide.catalog import Source
from soundguide.errors import IndexNotReadyError
from soundguide.evidence import IndexedSourceState, RankedChunk, parse_chunk_metadata
from soundguide.settings import Settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS sg_documents (
  source_id text PRIMARY KEY, document_id text UNIQUE NOT NULL,
  model text NOT NULL, language text NOT NULL, document_version text NOT NULL,
  profile text NOT NULL, provenance jsonb NOT NULL,
  indexed_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS sg_chunks (
  id text PRIMARY KEY,
  source_id text NOT NULL REFERENCES sg_documents(source_id) ON DELETE CASCADE,
  content text NOT NULL, metadata jsonb NOT NULL, embedding vector(512) NOT NULL
);
CREATE INDEX IF NOT EXISTS sg_chunks_source ON sg_chunks(source_id);
"""


class Store:
    """Use only the configured SoundGuide database; never switch to an upstream default."""

    def __init__(self, settings: Settings) -> None:
        """Bind the dedicated application database configuration."""
        self.settings = settings
        settings.require_ready()

    @asynccontextmanager
    async def connection(self):
        """Bound connection lifetime, including failed API requests."""
        conn = await asyncpg.connect(self.settings.database_url, timeout=5)
        try:
            yield conn
        finally:
            await conn.close()

    async def setup(self):
        """Create application tables; the extension is provisioned by the database admin."""
        async with self.connection() as conn, conn.transaction():
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtext('soundguide-schema'))"
            )
            await conn.execute(SCHEMA)

    async def replace(self, source, prepared, vectors):
        """Commit the document and all chunks together, preserving old data on failure."""
        chunks = prepared["chunks"]
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("Missing chunks or vectors")
        async with self.connection() as conn, conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", source.id)
            await conn.execute("DELETE FROM sg_documents WHERE source_id=$1", source.id)
            await conn.execute(
                "INSERT INTO sg_documents(source_id,document_id,model,language,document_version,profile,provenance) VALUES($1,$2,$3,$4,$5,$6,$7::jsonb)",
                source.id,
                prepared["document_id"],
                source.model,
                source.language,
                source.document_version,
                self.settings.profile,
                json.dumps(prepared["download"]),
            )
            await conn.executemany(
                "INSERT INTO sg_chunks(id,source_id,content,metadata,embedding) VALUES($1,$2,$3,$4::jsonb,$5::text::vector)",
                [
                    (
                        chunk["id"],
                        source.id,
                        chunk["content"],
                        json.dumps(chunk["metadata"]),
                        json.dumps(vector),
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )

    async def status(self, source_id: str | None = None) -> list[IndexedSourceState]:
        """Return actual committed counts and embedding profile for each source."""
        async with self.connection() as conn:
            rows = await conn.fetch(
                """
                SELECT d.source_id, d.model, d.profile, d.indexed_at,
                       d.provenance->>'sha256' AS content_sha256, count(c.id) AS chunks
                FROM sg_documents d LEFT JOIN sg_chunks c USING (source_id)
                WHERE ($1::text IS NULL OR d.source_id = $1)
                GROUP BY d.source_id
                ORDER BY d.source_id
                """,
                source_id,
            )
            return [
                IndexedSourceState(
                    source_id=row["source_id"],
                    model=row["model"],
                    profile=row["profile"],
                    indexed_at=row["indexed_at"],
                    content_sha256=row["content_sha256"],
                    chunks=row["chunks"],
                )
                for row in rows
            ]

    async def search(  # noqa: PLR0913
        self,
        vector,
        model: str,
        top_k: int,
        language: str,
        version: str,
        *,
        source: Source | None = None,
    ) -> list[RankedChunk]:
        """Filter applicability before ranking and reject mismatched embedding profiles."""
        async with self.connection() as conn:
            source_id = source.id if source else None
            expected_checksum = source.expected_sha256 if source else None
            mismatch = await conn.fetchval(
                """
                SELECT count(*) FROM sg_documents
                WHERE model = $1 AND language = $2 AND document_version = $3
                  AND profile <> $4 AND ($5::text IS NULL OR source_id = $5)
                """,
                model,
                language,
                version,
                self.settings.profile,
                source_id,
            )
            if mismatch:
                raise IndexNotReadyError(
                    "Indexed embedding profile has changed; reindex this model first"
                )
            rows = await conn.fetch(
                """SELECT c.id,c.content,c.metadata,(c.embedding <=> $1::text::vector) AS distance
              FROM sg_chunks c JOIN sg_documents d USING(source_id)
              WHERE d.model=$2 AND d.language=$3 AND d.document_version=$4 AND d.profile=$5
                AND ($7::text IS NULL OR (
                    d.source_id = $7
                    AND d.provenance->>'sha256' IS NOT DISTINCT FROM $8::text
                ))
              ORDER BY c.embedding <=> $1::text::vector, c.id LIMIT $6""",
                json.dumps(vector),
                model,
                language,
                version,
                self.settings.profile,
                top_k,
                source_id,
                expected_checksum,
            )
            return [
                {
                    "id": row["id"],
                    "content": row["content"],
                    "metadata": parse_chunk_metadata(row["metadata"]),
                    "cosine_similarity": 1 - float(row["distance"]),
                }
                for row in rows
            ]
