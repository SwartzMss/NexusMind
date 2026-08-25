$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$distRoot = Join-Path $repositoryRoot "dist"
$bundlePath = Join-Path $distRoot "nexusmind"
$executablePath = Join-Path $bundlePath "nexusmind.exe"
$smokeRuntimeRoot = Join-Path $bundlePath ".nexusmind"
$archivePath = Join-Path $distRoot "nexusmind-windows-portable.zip"
$smokeWorkingDirectory = Join-Path $repositoryRoot "build\smoke-cwd"

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
    Write-Host "Portable package: $archivePath"
}
finally {
    Pop-Location
}
