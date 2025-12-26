param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,
    [string]$OutputDir = (Join-Path $PSScriptRoot "..\\public"),
    [int]$Mp4Crf = 26,
    [int]$Av1Crf = 38,
    [string]$Mp4Preset = "slow",
    [int]$Av1Preset = 8
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    throw "ffmpeg not found in PATH."
}

if (-not (Test-Path $OutputDir)) {
    throw "Output directory does not exist: $OutputDir"
}

if (-not (Test-Path $InputPath)) {
    throw "Input file not found: $InputPath"
}

$baseName = [IO.Path]::GetFileNameWithoutExtension($InputPath)
$mp4Output = Join-Path $OutputDir "$baseName.encoded.mp4"
$av1Output = Join-Path $OutputDir "$baseName.av1.mp4"

Write-Host "Encoding MP4 for $baseName..."
& ffmpeg -y -i $InputPath -an -c:v libx264 -preset $Mp4Preset -crf $Mp4Crf -movflags +faststart $mp4Output
if ($LASTEXITCODE -ne 0) { throw "ffmpeg MP4 encode failed for $InputPath" }

Write-Host "Encoding AV1 for $baseName..."
& ffmpeg -y -i $InputPath -an -c:v libsvtav1 -preset $Av1Preset -crf $Av1Crf -pix_fmt yuv420p -movflags +faststart $av1Output
if ($LASTEXITCODE -ne 0) { throw "ffmpeg AV1 encode failed for $InputPath" }

Write-Host "Wrote $mp4Output"
Write-Host "Wrote $av1Output"
