$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$distRoot = Join-Path $repositoryRoot "dist"
$bundlePath = Join-Path $distRoot "nexusmind"
$executablePath = Join-Path $bundlePath "nexusmind.exe"
$archivePath = Join-Path $distRoot "nexusmind-windows-portable.zip"
$smokeRuntime = Join-Path $repositoryRoot "build\smoke-runtime"

Push-Location $repositoryRoot
try {
    python -m PyInstaller packaging/nexusmind.spec --noconfirm --clean
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }
    if (-not (Test-Path -Path $executablePath -PathType Leaf)) {
        throw "Portable executable was not generated: $executablePath"
    }

    $env:NEXUSMIND_RUNTIME_DIR = $smokeRuntime
    & $executablePath --help
    if ($LASTEXITCODE -ne 0) {
        throw "Portable executable smoke test failed with exit code $LASTEXITCODE"
    }
    $smokeLog = Join-Path $smokeRuntime "logs\nexusmind.log"
    if (-not (Test-Path -Path $smokeLog -PathType Leaf)) {
        throw "Portable executable did not create its runtime log: $smokeLog"
    }

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
