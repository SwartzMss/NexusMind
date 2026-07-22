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

## Test

```powershell
pytest
```

