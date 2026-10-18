[CmdletBinding()]
param(
    [switch]$WithAI,
    [switch]$SkipDependencyInstall,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

function Invoke-Python {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE`: $($Arguments -join ' ')"
    }
}

function Test-FrozenApplication {
    param([string]$Executable, [string]$Report)
    New-Item -ItemType Directory -Force (Split-Path -Parent $Report) | Out-Null
    if (Test-Path $Report) { Remove-Item -LiteralPath $Report -Force }
    $process = Start-Process -FilePath $Executable `
        -ArgumentList "--smoke-test --smoke-output `"$Report`"" `
        -WorkingDirectory (Split-Path -Parent $Executable) -PassThru
    if (-not $process.WaitForExit(120000)) {
        $process.Kill()
        throw "Frozen application smoke test exceeded 120 seconds: $Executable"
    }
    $process.Refresh()
    if ($process.ExitCode -ne 0) {
        if (Test-Path -LiteralPath $Report) {
            Write-Host (Get-Content -LiteralPath $Report -Raw -Encoding UTF8)
        }
        throw "Frozen application returned $($process.ExitCode): $Executable"
    }
    if (-not (Test-Path -LiteralPath $Report)) {
        throw "Frozen application did not write its smoke-test report: $Report"
    }
    $result = Get-Content -LiteralPath $Report -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($result.status -ne "ok") {
        throw "Frozen application reported a failure: $(Get-Content -LiteralPath $Report -Raw)"
    }
    Write-Host "Frozen application passed: $Executable"
}

Invoke-Python -Arguments @("-c", "import sys; assert sys.version_info[:2] == (3, 11), 'Use Python 3.11 x64'; assert sys.maxsize > 2**32, 'Use 64-bit Python'")
if (-not $SkipDependencyInstall) {
    Invoke-Python -Arguments @("-m", "pip", "install", "--disable-pip-version-check", "-r", "requirements-dev.txt")
    if ($WithAI) {
        Invoke-Python -Arguments @("-m", "pip", "install", "--disable-pip-version-check", "-r", "requirements-ai.txt")
    }
}
Invoke-Python -Arguments @("-m", "pip", "check")

$previousPlatform = $env:QT_QPA_PLATFORM
$previousAI = $env:IMAGEBOOST_INCLUDE_AI
try {
    $env:QT_QPA_PLATFORM = "offscreen"
    Invoke-Python -Arguments @("-m", "ruff", "check", ".")
    Invoke-Python -Arguments @("-m", "pytest", "-q", "--junitxml=reports/windows-tests.xml")
    $env:IMAGEBOOST_INCLUDE_AI = if ($WithAI) { "1" } else { "0" }
    Invoke-Python -Arguments @("-m", "PyInstaller", "--noconfirm", "--clean", "ImageBoostPro.spec")

    # Exercise the actual Windows Qt platform plugin in the frozen program.
    $env:QT_QPA_PLATFORM = "windows"
    $distribution = Join-Path $projectRoot "dist/ImageBoostPro"
    $executable = Join-Path $distribution "ImageBoostPro.exe"
    if (-not (Test-Path -LiteralPath $executable)) { throw "Missing distribution: $executable" }
    $reports = Join-Path $projectRoot "reports"
    Test-FrozenApplication -Executable $executable -Report (Join-Path $reports "windows-frozen-smoke.json")

    Copy-Item -LiteralPath (Join-Path $projectRoot "README.md") -Destination $distribution
    Copy-Item -LiteralPath (Join-Path $projectRoot "docs/BUILD.md") -Destination (Join-Path $distribution "BUILD.md")
    Copy-Item -LiteralPath (Join-Path $projectRoot "docs/THIRD_PARTY.md") -Destination (Join-Path $distribution "THIRD_PARTY.md")
    $archive = Join-Path $projectRoot "dist/ImageBoostPro-Windows.zip"
    Compress-Archive -LiteralPath $distribution -DestinationPath $archive -Force -CompressionLevel Optimal

    # Validate what the user downloads, including its directory layout.
    $extracted = Join-Path $projectRoot "build/distribution-check"
    if (Test-Path -LiteralPath $extracted) { Remove-Item -LiteralPath $extracted -Recurse -Force }
    Expand-Archive -LiteralPath $archive -DestinationPath $extracted -Force
    $extractedExe = Join-Path $extracted "ImageBoostPro/ImageBoostPro.exe"
    Test-FrozenApplication -Executable $extractedExe -Report (Join-Path $reports "windows-zip-smoke.json")
    $hash = Get-FileHash -LiteralPath $archive -Algorithm SHA256
    "$($hash.Hash.ToLowerInvariant())  ImageBoostPro-Windows.zip" |
        Set-Content -LiteralPath (Join-Path $projectRoot "dist/SHA256SUMS.txt") -Encoding ascii
    Write-Host "Created and tested: $archive"
}
finally {
    $env:QT_QPA_PLATFORM = $previousPlatform
    $env:IMAGEBOOST_INCLUDE_AI = $previousAI
}
