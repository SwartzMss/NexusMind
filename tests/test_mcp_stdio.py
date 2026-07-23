import os

from nexusmind.mcp import stdio


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
