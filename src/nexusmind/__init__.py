"""NexusMind public package exports."""

from .knowledge import (
    Document,
    KnowledgeSource,
    KnowledgeSourceType,
    compute_content_hash,
)

__all__ = [
    "Document",
    "KnowledgeSource",
    "KnowledgeSourceType",
    "compute_content_hash",
    "__version__",
]

__version__ = "0.1.0"

