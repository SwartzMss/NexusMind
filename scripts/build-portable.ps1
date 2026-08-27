$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$distRoot = Join-Path $repositoryRoot "dist"
$bundlePath = Join-Path $distRoot "nexusmind"
$executablePath = Join-Path $bundlePath "nexusmind.exe"
$smokeRuntimeRoot = Join-Path $bundlePath ".nexusmind"
$archivePath = Join-Path $distRoot "nexusmind-windows-portable.zip"
$smokeWorkingDirectory = Join-Path $repositoryRoot "build\smoke-cwd"
$releaseSmokeRoot = Join-Path $repositoryRoot "build\release-smoke"

function Invoke-PortableCommand {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )

    $output = & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Portable command failed: $($Arguments -join ' ')"
    }
    return ($output -join "`n")
}

Push-Location $repositoryRoot
try {
    python -m PyInstaller packaging/nexusmind.spec --noconfirm --clean
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }
    if (-not (Test-Path -Path $executablePath -PathType Leaf)) {
        throw "Portable executable was not generated: $executablePath"
    }

    New-Item -ItemType Directory -Path $smokeWorkingDirectory -Force | Out-Null
    if (Test-Path Env:NEXUSMIND_RUNTIME_DIR) {
        Remove-Item Env:NEXUSMIND_RUNTIME_DIR
    }
    Push-Location $smokeWorkingDirectory
    try {
        & $executablePath --help
    }
    finally {
        Pop-Location
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Portable executable smoke test failed with exit code $LASTEXITCODE"
    }
    $smokeLog = Join-Path $bundlePath ".nexusmind\logs\nexusmind.log"
    if (-not (Test-Path -Path $smokeLog -PathType Leaf)) {
        throw "Portable executable did not create its runtime log: $smokeLog"
    }
    Remove-Item -Path $smokeRuntimeRoot -Recurse -Force

    if (Test-Path -Path $archivePath) {
        Remove-Item -Path $archivePath -Force
    }
    Compress-Archive -Path $bundlePath -DestinationPath $archivePath -Force
    if (-not (Test-Path -Path $archivePath -PathType Leaf)) {
        throw "Portable archive was not generated: $archivePath"
    }

    if (Test-Path -Path $releaseSmokeRoot) {
        Remove-Item -Path $releaseSmokeRoot -Recurse -Force
    }
    $extractionRoot = Join-Path $releaseSmokeRoot "extracted"
    $fixtureRoot = Join-Path $releaseSmokeRoot "fixture"
    $knowledgeBasePath = Join-Path $releaseSmokeRoot "knowledge-base"
    New-Item -ItemType Directory -Path $extractionRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $fixtureRoot -Force | Out-Null
    $fixturePath = Join-Path $fixtureRoot "smoke-fixture.md"
    Set-Content -Path $fixturePath -Value "NexusMind release-smoke-token portable validation." -Encoding utf8

    Expand-Archive -Path $archivePath -DestinationPath $extractionRoot -Force
    $smokeExecutable = Join-Path $extractionRoot "nexusmind\nexusmind.exe"
    if (-not (Test-Path -Path $smokeExecutable -PathType Leaf)) {
        throw "Extracted portable executable was not found: $smokeExecutable"
    }

    Invoke-PortableCommand -Executable $smokeExecutable -Arguments @("create", $knowledgeBasePath, "--name", "Release Smoke") | Out-Null
    Invoke-PortableCommand -Executable $smokeExecutable -Arguments @("source", "add", $fixturePath, "--knowledge-base", $knowledgeBasePath) | Out-Null
    Invoke-PortableCommand -Executable $smokeExecutable -Arguments @("sync", "--knowledge-base", $knowledgeBasePath, "--json") | Out-Null
    $searchJson = Invoke-PortableCommand -Executable $smokeExecutable -Arguments @("search", "release-smoke-token", "--knowledge-base", $knowledgeBasePath, "--json")
    $search = $searchJson | ConvertFrom-Json
    $inspectionJson = Invoke-PortableCommand -Executable $smokeExecutable -Arguments @("inspect", "--knowledge-base", $knowledgeBasePath, "--json")
    $inspection = $inspectionJson | ConvertFrom-Json
    if (@($search).Count -lt 1 -or $searchJson -notmatch "release-smoke-token") {
        throw "Portable search did not return the synchronized fixture"
    }
    if ($inspection.status.registered_source_count -ne 1 -or
        $inspection.status.canonical_source_count -ne 1 -or
        $inspection.status.document_count -ne 1) {
        throw "Portable inspect did not reopen canonical state"
    }
    Write-Host "Portable package: $archivePath"
}
finally {
    if (Test-Path -Path $releaseSmokeRoot) {
        Remove-Item -Path $releaseSmokeRoot -Recurse -Force
    }
    Pop-Location
}
