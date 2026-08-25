# Portable Runtime Next to Executable Design

## Goal

Make the default mutable runtime root for the frozen Windows portable CLI a
`.nexusmind` directory beside `nexusmind.exe`, so the extracted application and
its local data can be moved together as one portable directory.

## Runtime Root Resolution

Runtime-root resolution keeps three explicit precedence levels:

1. A non-empty `NEXUSMIND_RUNTIME_DIR` remains the highest-priority override
   and must resolve to an absolute path.
2. A PyInstaller-frozen process, identified by `sys.frozen`, defaults to
   `Path(sys.executable).parent / ".nexusmind"`.
3. A normal source or installed-Python process continues to default to
   `Path.home() / ".nexusmind"`.

The frozen default must not depend on the current working directory. Launching
the same executable from PowerShell, Explorer, a shortcut, or another directory
therefore resolves the same runtime root.

## Portable Layout

The generated `onedir` package retains PyInstaller's `_internal` directory and
creates mutable runtime state alongside it:

```text
nexusmind/
├── nexusmind.exe
├── _internal/
└── .nexusmind/
    ├── config/
    ├── data/
    ├── logs/
    │   └── nexusmind.log
    └── models/
```

The `.nexusmind` directory is created on first executable startup, not embedded
in the ZIP. The executable directory must be writable. If it is read-only,
runtime initialization fails with the existing controlled startup message;
users can recover by moving the portable directory to a writable location or
setting `NEXUSMIND_RUNTIME_DIR` to an absolute writable path.

Because data is now colocated with the portable application, deleting or
replacing the whole extracted directory can delete local logs, configuration,
models, and data. Documentation must tell users to preserve `.nexusmind` during
upgrades or back it up before replacing the directory.

## Compatibility

Source development behavior remains unchanged, avoiding writes beside the
system or virtual-environment Python executable. Existing managed deployments
that set `NEXUSMIND_RUNTIME_DIR` also remain unchanged.

This change does not migrate an existing `%USERPROFILE%\.nexusmind` directory.
The new frozen executable starts with its adjacent `.nexusmind`; users who need
old data must copy it explicitly or temporarily point the environment override
at the old absolute directory. Automatic migration is outside this focused
default-path change.

## Packaging and Tests

Unit tests simulate frozen execution by setting `sys.frozen` and
`sys.executable`, then verify that the root resolves beside the executable.
They also verify that the environment override still wins and that non-frozen
execution retains the user-home default.

The PowerShell portable smoke test must launch the executable from a working
directory outside the bundle without setting `NEXUSMIND_RUNTIME_DIR`. It then
checks for `dist/nexusmind/.nexusmind/logs/nexusmind.log`. This provides a real
Windows assertion that resolution follows the executable rather than the
shell's current directory.

README documentation is updated to show the adjacent layout, writable-directory
requirement, upgrade/data-loss warning, override behavior, and lack of automatic
migration from the previous user-profile default.
