from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest


def test_pyinstaller_spec_targets_desktop_entry_and_onedir() -> None:
    text = Path("packaging/nexusmind.spec").read_text(encoding="utf-8")

    assert "src/nexusmind/desktop.py" in text.replace("\\", "/")
    assert "COLLECT(" in text
    assert "console=True" in text


def test_pyinstaller_spec_resolves_repository_root_from_spec_directory() -> None:
    spec = Path("packaging/nexusmind.spec").resolve()
    tree = ast.parse(spec.read_text(encoding="utf-8"))
    root_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ROOT" for target in node.targets)
    )

    root = eval(
        compile(ast.Expression(root_assignment.value), str(spec), "eval"),
        {"Path": Path, "SPECPATH": str(spec.parent)},
    )

    assert root == spec.parent.parent


def test_portable_script_builds_smoke_tests_and_archives() -> None:
    text = Path("scripts/build-portable.ps1").read_text(encoding="utf-8")

    assert "PyInstaller" in text
    assert "nexusmind.exe" in text
    assert "--help" in text
    assert "Compress-Archive" in text


def test_portable_script_smoke_tests_executable_relative_runtime() -> None:
    text = Path("scripts/build-portable.ps1").read_text(encoding="utf-8")

    assert "Remove-Item Env:NEXUSMIND_RUNTIME_DIR" in text
    assert "Push-Location $smokeWorkingDirectory" in text
    assert 'Join-Path $bundlePath ".nexusmind\\logs\\nexusmind.log"' in text


def test_portable_script_removes_smoke_runtime_before_archiving() -> None:
    text = Path("scripts/build-portable.ps1").read_text(encoding="utf-8")
    cleanup = "Remove-Item -Path $smokeRuntimeRoot -Recurse -Force"

    assert cleanup in text
    assert text.index(cleanup) < text.index("Compress-Archive")


def test_ci_uploads_portable_directory_without_nested_zip() -> None:
    text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "path: dist/nexusmind/" in text
    assert "path: dist/nexusmind-windows-portable.zip" not in text


def test_portable_script_runs_extracted_archive_knowledge_base_e2e() -> None:
    text = Path("scripts/build-portable.ps1").read_text(encoding="utf-8")

    required = (
        "Expand-Archive",
        "smoke-fixture.md",
        '"create"',
        '"source", "add"',
        '"sync"',
        '"search"',
        '"inspect"',
        '"--json"',
        "ConvertFrom-Json",
        "registered_source_count",
        "canonical_source_count",
        "document_count",
    )
    for marker in required:
        assert marker in text
    assert '@("create", $knowledgeBasePath)' in text
    assert '"--name"' not in text
    assert text.index("Compress-Archive") < text.index("Test-PortableArchive -ArchivePath")


@pytest.mark.parametrize("scenario", ["runner-temp", "system-temp", "command-failure", "inside-checkout"])
def test_portable_archive_smoke_is_isolated_and_cleans_up(tmp_path, scenario) -> None:
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("PowerShell is required to exercise portable smoke orchestration")
    script = Path("scripts/build-portable.ps1").resolve()
    repository = tmp_path / "checkout"
    repository.mkdir()
    archive = repository / "portable.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("nexusmind/nexusmind.exe", b"test placeholder; launcher is stubbed")
    temp_root = tmp_path / "runner-temp"
    temp_root.mkdir()
    observations = tmp_path / "observations.json"
    harness = tmp_path / "exercise-smoke.ps1"
    harness.write_text(r'''
param($Script, $Archive, $Repository, $Observations, $Scenario)
$ErrorActionPreference = "Stop"
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Script, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw "Invalid PowerShell syntax: $errors" }
$definition = $ast.Find({ param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq "Test-PortableArchive"
}, $true)
if (-not $definition) { throw "Missing Test-PortableArchive function" }
. ([scriptblock]::Create($definition.Extent.Text))
$script:calls = @()
function Invoke-PortableCommand {
    param($Executable, [string[]]$Arguments)
    $script:calls += [pscustomobject]@{ executable = $Executable; cwd = (Get-Location).Path; arguments = $Arguments }
    if ($Scenario -eq "command-failure") { throw "Injected portable failure" }
    if ($Arguments[0] -eq "search") { return '[{"text":"release-smoke-token"}]' }
    if ($Arguments[0] -eq "inspect") { return '{"status":{"registered_source_count":1,"canonical_source_count":1,"document_count":1}}' }
    return '{}'
}
Set-Location -LiteralPath $Repository
$before = (Get-Location).Path
$failure = $null
try { Test-PortableArchive -ArchivePath $Archive -RepositoryRoot $Repository }
catch { $failure = $_.Exception.Message }
[pscustomobject]@{ before = $before; after = (Get-Location).Path; failure = $failure; calls = @($script:calls) } |
    ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Observations -Encoding utf8
''', encoding="utf-8")
    env = os.environ.copy()
    if scenario == "system-temp":
        env.pop("RUNNER_TEMP", None)
    else:
        env["RUNNER_TEMP"] = str(repository if scenario == "inside-checkout" else temp_root)
    subprocess.run([pwsh, "-NoProfile", "-File", str(harness), str(script), str(archive),
                    str(repository), str(observations), scenario], env=env, check=True, capture_output=True, encoding="utf-8", timeout=30)

    result = json.loads(observations.read_text(encoding="utf-8-sig"))
    assert Path(result["before"]) == Path(result["after"]) == repository
    if scenario == "inside-checkout":
        assert "outside" in result["failure"]
        assert result["calls"] == []
    else:
        assert result["failure"] == ("Injected portable failure" if scenario == "command-failure" else None)
        calls = result["calls"]
        assert [call["arguments"][0] for call in calls] == (["create"] if scenario == "command-failure" else ["create", "source", "sync", "search", "inspect"])
        smoke_root = Path(calls[0]["cwd"]).parent
        assert not smoke_root.is_relative_to(repository)
        assert smoke_root.name.startswith("nexusmind-release-smoke-")
        if scenario != "system-temp":
            assert smoke_root.parent == temp_root
        for call in calls:
            assert Path(call["cwd"]) == smoke_root / "cwd"
            assert Path(call["executable"]).is_relative_to(smoke_root / "extracted")
        assert not smoke_root.exists()
    assert list(temp_root.iterdir()) == []
