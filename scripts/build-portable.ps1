$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$distRoot = Join-Path $repositoryRoot "dist"
$bundlePath = Join-Path $distRoot "nexusmind"
$executablePath = Join-Path $bundlePath "nexusmind.exe"
$smokeRuntimeRoot = Join-Path $bundlePath ".nexusmind"
$archivePath = Join-Path $distRoot "nexusmind-windows-portable.zip"
$smokeWorkingDirectory = Join-Path $repositoryRoot "build\smoke-cwd"

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

function Test-PortableArchive {
    param(
        [string]$ArchivePath,
        [string]$RepositoryRoot
    )

    $tempRoot = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [System.IO.Path]::GetTempPath() }
    $tempRoot = (Resolve-Path -LiteralPath $tempRoot).Path
    $repositoryPath = (Resolve-Path -LiteralPath $RepositoryRoot).Path
    $separator = [System.IO.Path]::DirectorySeparatorChar
    $tempPrefix = $tempRoot.TrimEnd('\', '/') + $separator
    $repositoryPrefix = $repositoryPath.TrimEnd('\', '/') + $separator
    $releaseSmokeRoot = [System.IO.Path]::GetFullPath(
        (Join-Path $tempRoot ("nexusmind-release-smoke-" + [guid]::NewGuid().ToString("N")))
    )
    if (-not $releaseSmokeRoot.StartsWith($tempPrefix, [System.StringComparison]::OrdinalIgnoreCase) -or
        $releaseSmokeRoot.StartsWith($repositoryPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Portable smoke directory must be outside the source checkout and inside the temporary directory"
    }

    # Own a fresh directory; never reuse or delete a previous smoke run's data.
    New-Item -ItemType Directory -Path $releaseSmokeRoot | Out-Null
    try {
        $extractionRoot = Join-Path $releaseSmokeRoot "extracted"
        $fixtureRoot = Join-Path $releaseSmokeRoot "fixture"
        $knowledgeBasePath = Join-Path $releaseSmokeRoot "knowledge-base"
        $releaseWorkingDirectory = Join-Path $releaseSmokeRoot "cwd"
        New-Item -ItemType Directory -Path $extractionRoot, $fixtureRoot, $releaseWorkingDirectory | Out-Null
        $sourceFixtureRoot = Join-Path $repositoryPath "tests\fixtures\structured"
        Copy-Item -LiteralPath (Join-Path $sourceFixtureRoot "marker.docx") -Destination (Join-Path $fixtureRoot "b.docx")
        Copy-Item -LiteralPath (Join-Path $sourceFixtureRoot "marker.pdf") -Destination (Join-Path $fixtureRoot "c.pdf")
        Set-Content -LiteralPath (Join-Path $fixtureRoot "a.md") -Value "NEXUSMIND_TEXT_MARKER portable plain text." -Encoding utf8

        Expand-Archive -LiteralPath $ArchivePath -DestinationPath $extractionRoot
        $smokeExecutable = Join-Path $extractionRoot "nexusmind\nexusmind.exe"
        if (-not (Test-Path -LiteralPath $smokeExecutable -PathType Leaf)) {
            throw "Extracted portable executable was not found: $smokeExecutable"
        }

        Push-Location -LiteralPath $releaseWorkingDirectory
        try {
            Write-Host "Portable archive E2E working directory: $releaseWorkingDirectory"
            Invoke-PortableCommand -Executable $smokeExecutable -Arguments @("create", $knowledgeBasePath) | Out-Null
            Invoke-PortableCommand -Executable $smokeExecutable -Arguments @("source", "add", $fixtureRoot, "--knowledge-base", $knowledgeBasePath) | Out-Null
            Invoke-PortableCommand -Executable $smokeExecutable -Arguments @("sync", "--knowledge-base", $knowledgeBasePath, "--json") | Out-Null
            $expectations = @(
                @{ Query = "NEXUSMIND_TEXT_MARKER"; Marker = "NEXUSMIND_TEXT_MARKER"; Path = "a.md" },
                @{ Query = "NEXUSMIND_DOCX_MARKER"; Marker = "NEXUSMIND_DOCX_MARKER"; Path = "b.docx" },
                @{ Query = "NEXUSMIND_PDF_MARKER"; Marker = "NEXUSMIND_PDF_MARKER"; Path = "c.pdf" }
            )
            foreach ($expected in $expectations) {
                $searchJson = Invoke-PortableCommand -Executable $smokeExecutable -Arguments @("search", $expected.Query, "--knowledge-base", $knowledgeBasePath, "--json")
                $search = @($searchJson | ConvertFrom-Json)
                $matchingHits = @($search | Where-Object {
                    $_.document.logical_path -eq $expected.Path -and
                    $_.hit.chunk.content -match [regex]::Escape($expected.Marker)
                })
                if ($matchingHits.Count -lt 1) {
                    throw "Portable search did not return $($expected.Marker) from $($expected.Path)"
                }
            }
            $inspectionJson = Invoke-PortableCommand -Executable $smokeExecutable -Arguments @("inspect", "--knowledge-base", $knowledgeBasePath, "--json")
            $inspection = $inspectionJson | ConvertFrom-Json
            if ($inspection.status.registered_source_count -ne 1 -or
                $inspection.status.canonical_source_count -ne 1 -or
                $inspection.status.document_count -ne 3) {
                throw "Portable inspect did not reopen canonical state"
            }
        }
        finally {
            Pop-Location
        }
    }
    finally {
        # Resolve and verify the exact owned target before recursive cleanup.
        $cleanupPath = (Resolve-Path -LiteralPath $releaseSmokeRoot).Path
        if (-not $cleanupPath.Equals($releaseSmokeRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
            -not $cleanupPath.StartsWith($tempPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to clean an unexpected portable smoke directory: $cleanupPath"
        }
        Remove-Item -LiteralPath $cleanupPath -Recurse -Force
    }
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

    Test-PortableArchive -ArchivePath $archivePath -RepositoryRoot $repositoryRoot
    Write-Host "Portable package: $archivePath"
}
finally {
    Pop-Location
}
