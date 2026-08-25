# Windows Portable Runtime Design

## Goal

Distribute NexusMind as a Windows portable command-line runtime that does not
require a separately installed Python interpreter, keeps mutable user data out
of the application bundle, and leaves enough local diagnostics to investigate
startup, synchronization, query, and unexpected runtime failures.

## Scope

The deliverable is a PyInstaller `onedir` build wrapped in a portable ZIP. It
retains the existing `nexusmind` CLI and does not add a GUI, installer, update
mechanism, cloud service, license management, or model distribution system.

The portable package contains the executable, bundled Python runtime, and
Python dependencies. It does not contain user configuration, databases,
downloaded models, logs, or other mutable runtime data.

## Runtime Layout

The default runtime root is `%USERPROFILE%\.nexusmind`. A non-empty
`NEXUSMIND_RUNTIME_DIR` environment variable overrides the root, making tests,
managed deployments, and portable launch scripts deterministic. Relative
overrides are rejected so mutable data cannot silently depend on the process
working directory.

At process startup NexusMind creates and validates this layout:

```text
<runtime-root>/
├── config/
├── data/
├── logs/
│   └── nexusmind.log
└── models/
```

Runtime path discovery and directory creation live in a focused module rather
than in CLI command handlers. Failure to resolve or create the layout is a
startup failure with a concise terminal message. When a log file cannot be
opened, the message cannot point to a usable log and states that logging could
not be initialized.

Existing explicit CLI flags such as `--state-db`, `--checkpoint-db`, and
`--lease-db` keep their current semantics. Documentation recommends paths under
the runtime `data` directory but this change does not silently opt users into
persistence or rewrite caller-provided paths.

## Logging

The executable entry point initializes logging before dispatching to the
existing CLI. Logs go to `<runtime-root>/logs/nexusmind.log` through a bounded
rotating file handler. Each line is structured JSON containing a timestamp,
level, logger, event name, and bounded diagnostic fields. Rotation prevents an
unbounded long-lived log file; a small fixed number of backups is retained.

The logging API exposes explicit lifecycle events for:

- process startup and normal exit;
- knowledge synchronization start, completion, and failure;
- knowledge query start, completion, and failure;
- unexpected executable-boundary failures.

The current CLI does not expose first-class knowledge sync or query commands,
so reusable logging helpers define those lifecycle events for current library
consumers and future CLI integration. This PR does not invent new commands.

Log records must not include API keys, authorization values, full prompts,
document contents, tool outputs, or complete exception arguments supplied by
remote services. Operation names, counts, durations, exception types, and
bounded identifiers are allowed. Unexpected failures include a traceback in
the local log while the terminal receives only a stable message and the log
path.

## Executable Boundary and Error Handling

The Python package keeps `nexusmind.cli:main` as the testable command parser and
adds a thin executable entry function. That function:

1. resolves and creates the runtime layout;
2. configures runtime logging;
3. records startup metadata that is not sensitive;
4. invokes the existing CLI;
5. records the exit status; and
6. catches otherwise unhandled `Exception` values, logs the traceback, prints
   a friendly diagnostic with the absolute log path, and returns a non-zero
   exit code.

`KeyboardInterrupt` and other `BaseException` subclasses are not converted into
unexpected application failures. Existing handled configuration, workspace,
MCP, persistence, and command errors retain their current CLI messages and exit
codes.

## Packaging and Release Artifact

PyInstaller is a development-only dependency. A checked-in spec builds a
console-mode `onedir` application named `nexusmind`; it gathers package metadata
and hidden imports required by dependencies instead of relying on a developer
machine's Python installation at runtime.

A PowerShell build script creates a clean staging directory, invokes PyInstaller
from the repository root, smoke-tests the produced executable with `--help`, and
archives the complete application directory as a portable ZIP. Build output
remains ignored by Git.

A Windows GitHub Actions workflow installs the supported build Python version,
installs the project and packaging dependencies, runs the focused tests, builds
the artifact, repeats the executable smoke test with an isolated runtime root,
and uploads the ZIP. Linux CI continues to validate platform-neutral runtime
and logging logic but is not treated as proof that the Windows binary works.

## Testing

Unit tests cover:

- default and overridden runtime-root resolution;
- rejection of invalid relative overrides;
- creation of the stable directory layout;
- deterministic structured log fields, sensitive-field exclusion, and bounded
  rotation configuration;
- lifecycle logging for startup, sync, query, success, and failure;
- executable-boundary behavior for normal exits and unexpected exceptions;
- preservation of `KeyboardInterrupt` behavior; and
- static packaging configuration and documented portable-build entry points.

The Windows workflow supplies the end-to-end acceptance check: the generated
executable starts without invoking a system Python command, creates runtime
directories outside the bundle, writes a startup log, and returns success for
`--help`. The build fails if the executable or ZIP is missing.

## Acceptance Mapping

- **Runs without Python installation:** the PyInstaller `onedir` artifact
  bundles the interpreter and dependencies, verified by the Windows smoke test.
- **Runtime logs are generated:** executable startup creates the rotating JSON
  log and records startup/exit events.
- **User data is separated from executable:** all mutable directories resolve
  under the user runtime root or its explicit absolute override.
- **Failures are diagnosable:** lifecycle helpers record sync/query failures and
  the executable boundary records unexpected startup/runtime tracebacks.
- **Portable package can be generated:** the PowerShell script and Windows CI
  produce and upload a ZIP containing the complete `onedir` application.
