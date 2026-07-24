# NexusMind

NexusMind is starting with a small, provider-decoupled model runtime. This baseline supports:

- NexusMind-owned message and runtime event contracts.
- A `ChatModel` abstraction with async streaming.
- An OpenAI-compatible HTTP adapter.
- A provider-neutral tool registry and executor baseline.
- A CLI entrypoint for streaming chat output.
- A fake model for offline tests.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

If you prefer requirements files for local tooling, install the same dependency set with:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Configure

Copy `.env.example` to `.env` or export the variables in your shell:

```powershell
$env:NEXUSMIND_MODEL_BASE_URL = "https://api.openai.com/v1"
$env:NEXUSMIND_MODEL_API_KEY = "your-api-key"
$env:NEXUSMIND_MODEL_NAME = "gpt-4.1-mini"
$env:NEXUSMIND_MODEL_TIMEOUT = "60"
```

The CLI intentionally does not print API keys in errors.

## Run

```powershell
nexusmind chat "介绍一下你自己"
```

You can also start a one-turn prompt interactively:

```powershell
nexusmind chat
```

## Tools

The tool system runs without model configuration or API keys.

List registered built-in tools:

```powershell
nexusmind tools list
```

Call the built-in echo tool:

```powershell
nexusmind tools call echo '{"text":"hello"}'
```

## MCP Stdio Tools

MCP server configuration is read from a JSON file. Treat this file as sensitive if it contains environment variables or secrets.

Example `mcp.json`:

```json
{
  "servers": {
    "demo": {
      "transport": "stdio",
      "command": "python",
      "args": ["tests/fixtures/mcp_echo_server.py"],
      "cwd": null,
      "env": {}
    }
  }
}
```

List tools exposed by the server:

```powershell
nexusmind mcp tools --config mcp.json --server demo
```

Call a discovered MCP tool through the NexusMind registry and executor:

```powershell
nexusmind mcp call --config mcp.json --server demo --tool demo__echo_290c9db7d5 --arguments '{"text":"hello"}'
```

## Model Tool Calls

NexusMind can pass registered `ToolDefinition` values to an OpenAI-compatible chat model, parse streamed `tool_calls` into provider-neutral events, execute requested tools through `ToolExecutor`, and feed structured `role=tool` results back into the next model turn. The runtime uses bounded single-agent loop limits to prevent unbounded model turns or tool result growth.

## Test

```powershell
pytest
```

