from enum import Enum
class StopReason(str, Enum):
    MODEL_COMPLETED = "model_completed"
    MODEL_FAILED = "model_failed"
    TOOL_FAILED = "tool_failed"
    LIMIT_EXCEEDED = "limit_exceeded"
    APPROVAL_DENIED = "approval_denied"
    RUNTIME_ERROR = "runtime_error"
    CANCELLED = "cancelled"
