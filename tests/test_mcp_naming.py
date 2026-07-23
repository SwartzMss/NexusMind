from nexusmind.mcp.naming import mcp_tool_local_name


def test_normal_name_maps_to_valid_local_name() -> None:
    assert mcp_tool_local_name("demo", "echo").startswith("demo__echo_")


def test_dot_slash_and_digit_names_are_mapped() -> None:
    name = mcp_tool_local_name("demo", "1.admin/tools.list")

    assert name[0].isalpha()
    assert "/" not in name
    assert "." not in name


def test_long_name_is_truncated_with_hash() -> None:
    name = mcp_tool_local_name("demo", "tool/" + ("x" * 200))

    assert len(name) <= 64


def test_normalization_collision_keeps_distinct_hashes() -> None:
    first = mcp_tool_local_name("demo", "admin/tools")
    second = mcp_tool_local_name("demo", "admin.tools")

    assert first != second


def test_mapping_is_deterministic() -> None:
    assert mcp_tool_local_name("demo", "admin/tools") == mcp_tool_local_name("demo", "admin/tools")

