"""NexusMind public package exports."""

from .knowledge import (
    Document,
    KnowledgeSource,
    KnowledgeSourceType,
    compute_content_hash,
)
from .knowledge_ingestion import (
    DEFAULT_MAX_DOCUMENTS,
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_TOTAL_BYTES,
    DEFAULT_SUPPORTED_EXTENSIONS,
    DocumentCountLimitError,
    FileTooLargeError,
    InvalidTextEncodingError,
    KnowledgeIngestionError,
    KnowledgeSourceAdapter,
    LocalDirectoryAdapter,
    LocalFileAdapter,
    LocalIngestionLimits,
    PathEscapeError,
    SourceNotFoundError,
    SourceTypeError,
    SymlinkSourceError,
    TotalBytesLimitError,
    UnsupportedFileTypeError,
)

__all__ = [
    "Document",
    "KnowledgeSource",
    "KnowledgeSourceType",
    "compute_content_hash",
    "DEFAULT_MAX_DOCUMENTS",
    "DEFAULT_MAX_FILE_BYTES",
    "DEFAULT_MAX_TOTAL_BYTES",
    "DEFAULT_SUPPORTED_EXTENSIONS",
    "DocumentCountLimitError",
    "FileTooLargeError",
    "InvalidTextEncodingError",
    "KnowledgeIngestionError",
    "KnowledgeSourceAdapter",
    "LocalDirectoryAdapter",
    "LocalFileAdapter",
    "LocalIngestionLimits",
    "PathEscapeError",
    "SourceNotFoundError",
    "SourceTypeError",
    "SymlinkSourceError",
    "TotalBytesLimitError",
    "UnsupportedFileTypeError",
    "__version__",
]

__version__ = "0.1.0"

