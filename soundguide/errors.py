"""Failures shared by the application services and translated at the HTTP boundary."""


class SoundGuideError(ValueError):
    """Base for expected application failures with safe messages."""


class ConfigurationError(SoundGuideError):
    """Required runtime configuration is missing or invalid."""


class IndexNotReadyError(SoundGuideError):
    """The requested manual does not have a compatible committed index."""


class InvalidQueryError(SoundGuideError):
    """The retrieval input cannot be processed."""


class EmbeddingContractError(SoundGuideError):
    """The embedding provider response violates the configured contract."""


class RetrievalLimitError(SoundGuideError):
    """The corpus exceeds the capacity of the requested retrieval strategy."""


class ManifestError(SoundGuideError):
    """A local download manifest could not be interpreted."""
