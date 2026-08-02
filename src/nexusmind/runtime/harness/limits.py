from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HarnessLimits:
    max_model_turns: int = 8
    max_tool_calls_total: int = 32
    max_tool_arguments_bytes_per_call: int = 1024 * 1024
    max_tool_arguments_bytes_total: int = 4 * 1024 * 1024
    max_tool_result_bytes_per_call: int = 1024 * 1024
    max_tool_result_bytes_total: int = 4 * 1024 * 1024
    max_json_nodes_per_payload: int = 100_000
    max_json_depth: int = 100

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("Harness limits must be positive integers")
        if self.max_json_nodes_per_payload < 3:
            raise ValueError("Harness limits must allow a minimal tool result envelope")
