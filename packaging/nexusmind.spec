# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata


ROOT = Path(SPECPATH).parent
ENTRY = "src/nexusmind/runtime_entrypoint.py"
datas = (
    collect_data_files("certifi")
    + copy_metadata("httpx")
    + copy_metadata("python-dotenv")
)
hiddenimports = collect_submodules("nexusmind")

analysis = Analysis(
    [str(ROOT / ENTRY)],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="nexusmind",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
)

bundle = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="nexusmind",
)
