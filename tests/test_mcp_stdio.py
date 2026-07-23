import os
import asyncio
import time
from contextlib import asynccontextmanager

import pytest

from nexusmind.mcp import stdio
from nexusmind.mcp.config import MCPStdioServerConfig
from nexusmind.mcp.errors import MCPConnectionError


def test_stdio_client_uses_devnull_for_errlog(monkeypatch) -> None:
    opened = {}

    class FakeFile:
        def close(self):
            opened["closed"] = True

    def fake_open(path, mode, encoding):
        opened["path"] = path
        opened["mode"] = mode
        opened["encoding"] = encoding
        return FakeFile()

    monkeypatch.setattr(stdio, "open", fake_open, raising=False)
    errlog = stdio._open_errlog()

    assert opened == {"path": os.devnull, "mode": "w", "encoding": "utf-8"}
    errlog.close()
    assert opened["closed"] is True


def test_stdio_initialize_uses_single_connect_timeout_budget(monkeypatch) -> None:
    events: list[str] = []

    class FakeErrlog:
        def close(self):
            events.append("errlog.close")

    class FakeStdioContext:
        async def __aenter__(self):
            events.append("stdio.enter")
            await asyncio.sleep(0.02)
            return "read", "write"

        async def __aexit__(self, exc_type, exc, tb):
            events.append("stdio.exit")

    class FakeSessionContext:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            events.append("session.enter")
            await asyncio.sleep(0.02)
            return self

        async def __aexit__(self, exc_type, exc, tb):
            events.append("session.exit")

        async def initialize(self):
            events.append("session.initialize")
            await asyncio.sleep(0.02)

    monkeypatch.setattr(stdio, "_open_errlog", lambda: FakeErrlog())
    monkeypatch.setattr(stdio, "stdio_client", lambda params, errlog: FakeStdioContext())
    monkeypatch.setattr(stdio, "ClientSession", FakeSessionContext)

    @asynccontextmanager
    async def fake_timeout(seconds):
        started = time.monotonic()
        yield
        if time.monotonic() - started > seconds:
            raise asyncio.TimeoutError()

    monkeypatch.setattr(stdio, "_same_task_timeout", fake_timeout)

    async def run():
        config = MCPStdioServerConfig(server_id="demo", command="python", connect_timeout=0.05)
        async with stdio.MCPStdioClient(config):
            pass

    with pytest.raises(MCPConnectionError, match="timed out"):
        asyncio.run(run())

    assert events == ["stdio.enter", "session.enter", "session.initialize", "session.exit", "stdio.exit", "errlog.close"]


def test_stdio_external_cancel_propagates_and_cleans_all_resources(monkeypatch) -> None:
    events: list[str] = []

    class FakeErrlog:
        def close(self):
            events.append("errlog.close")

    class FakeStdioContext:
        async def __aenter__(self):
            events.append("stdio.enter")
            return "read", "write"

        async def __aexit__(self, exc_type, exc, tb):
            events.append("stdio.exit")

    class FakeSessionContext:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            events.append("session.enter")
            return self

        async def __aexit__(self, exc_type, exc, tb):
            events.append("session.exit")
            raise asyncio.CancelledError()

        async def initialize(self):
            events.append("session.initialize")
            raise asyncio.CancelledError()

    monkeypatch.setattr(stdio, "_open_errlog", lambda: FakeErrlog())
    monkeypatch.setattr(stdio, "stdio_client", lambda params, errlog: FakeStdioContext())
    monkeypatch.setattr(stdio, "ClientSession", FakeSessionContext)

    async def run():
        config = MCPStdioServerConfig(server_id="demo", command="python", connect_timeout=1)
        async with stdio.MCPStdioClient(config):
            pass

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())

    assert events == ["stdio.enter", "session.enter", "session.initialize", "session.exit", "stdio.exit", "errlog.close"]


def test_stdio_cleanup_attempts_all_resources_when_session_exit_raises_base_exception(monkeypatch) -> None:
    events: list[str] = []

    class FakeErrlog:
        def close(self):
            events.append("errlog.close")

    class FakeStdioContext:
        async def __aexit__(self, exc_type, exc, tb):
            events.append("stdio.exit")

    class FakeSessionContext:
        async def __aexit__(self, exc_type, exc, tb):
            events.append("session.exit")
            raise BaseExceptionGroup("cancel group", [asyncio.CancelledError()])

    async def run():
        client = stdio.MCPStdioClient(MCPStdioServerConfig(server_id="demo", command="python"))
        client._session_context = FakeSessionContext()
        client._stdio_context = FakeStdioContext()
        client._errlog = FakeErrlog()
        await client._cleanup(raise_errors=False)

    asyncio.run(run())

    assert events == ["session.exit", "stdio.exit", "errlog.close"]
