"""Explicit independent database and embedding API configuration."""

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from soundguide import PIPELINE_VERSION

DIMENSIONS = 512
MIN_GENERATION_TOKENS = 128
MAX_GENERATION_TOKENS = 4096

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default="", repr=False)
    embedding_url: str = "http://127.0.0.1:8091/v1"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_key: str = field(default="", repr=False)
    embedding_revision: str = ""
    dimensions: int = 512
    query_prefix: str = "为这个句子生成表示以用于检索相关文章："  # noqa: RUF001
    generation_url: str = ""
    generation_model: str = ""
    generation_key: str = field(default="", repr=False)
    generation_max_tokens: int = 1536
    data_dir: Path = ROOT / ".data"

    def require_ready(self):
        """Fail closed rather than substituting fake embeddings."""
        if (
            not self.database_url
            or not self.embedding_key
            or not self.embedding_revision
        ):
            raise ValueError(
                "Configure SoundGuide database and embedding credentials/revision first"
            )
        if self.dimensions != DIMENSIONS:
            raise ValueError("Initial schema requires 512-dimensional embeddings")

    def require_generation(self):
        """Require explicit generation API settings without coupling model lifecycles."""
        if (
            not self.generation_url
            or not self.generation_model
            or not self.generation_key
        ):
            raise ValueError("Configure the generation model API first")
        if (
            not MIN_GENERATION_TOKENS
            <= self.generation_max_tokens
            <= MAX_GENERATION_TOKENS
        ):
            raise ValueError("Generation token limit must be between 128 and 4096")

    @property
    def profile(self) -> str:
        """Identify the weights and preprocessing used by an index."""
        values = [
            self.embedding_model,
            self.embedding_revision,
            self.dimensions,
            self.query_prefix,
            PIPELINE_VERSION,
        ]
        return hashlib.sha256(
            json.dumps(values, ensure_ascii=False).encode()
        ).hexdigest()


def get_settings() -> Settings:
    """Load local configuration without exposing credentials in API responses."""
    load_dotenv(ROOT / ".env", override=False)
    return Settings(
        database_url=os.getenv("SG_DATABASE_URL", ""),
        embedding_url=os.getenv("SG_EMBEDDING_BASE_URL", "http://127.0.0.1:8091/v1"),
        embedding_model=os.getenv("SG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
        embedding_key=os.getenv("SG_EMBEDDING_API_KEY", ""),
        embedding_revision=os.getenv("SG_EMBEDDING_REVISION", ""),
        generation_url=os.getenv("SG_GENERATION_BASE_URL", ""),
        generation_model=os.getenv("SG_GENERATION_MODEL", ""),
        generation_key=os.getenv("SG_GENERATION_API_KEY", ""),
        generation_max_tokens=int(os.getenv("SG_GENERATION_MAX_TOKENS", "1536")),
        dimensions=int(os.getenv("SG_EMBEDDING_DIMENSIONS", "512")),
    )
