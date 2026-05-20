param(
    [switch]$SkipUserPathPersist
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).ProviderPath
Set-Location $ProjectRoot

function Normalize-PathEntry {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue
    )

    $candidate = ($PathValue.Trim() -replace "/", "\").TrimEnd("\")
    try {
        return (Resolve-Path -LiteralPath $candidate -ErrorAction Stop).ProviderPath.TrimEnd("\")
    } catch {
        return $candidate
    }
}

function Split-PathEntries {
    param(
        [AllowEmptyString()]
        [string]$PathValue
    )

    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return @()
    }

    return @(
        $PathValue.Split(";", [System.StringSplitOptions]::RemoveEmptyEntries) |
            ForEach-Object { $_.Trim() } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
}

function Add-UniquePathEntry {
    param(
        [AllowEmptyString()]
        [string]$PathValue,
        [Parameter(Mandatory = $true)]
        [string]$Entry
    )

    $normalizedEntry = Normalize-PathEntry -PathValue $Entry
    $entries = New-Object System.Collections.Generic.List[string]
    $seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)

    foreach ($existingEntry in (Split-PathEntries -PathValue $PathValue)) {
        $normalizedExistingEntry = Normalize-PathEntry -PathValue $existingEntry
        if ($seen.Add($normalizedExistingEntry)) {
            $entries.Add($normalizedExistingEntry) | Out-Null
        }
    }

    $added = $false
    if ($seen.Add($normalizedEntry)) {
        $entries.Add($normalizedEntry) | Out-Null
        $added = $true
    }

    return [pscustomobject]@{
        Path  = ($entries -join ";")
        Entry = $normalizedEntry
        Added = $added
    }
}

function Prepend-UniquePathEntry {
    param(
        [AllowEmptyString()]
        [string]$PathValue,
        [Parameter(Mandatory = $true)]
        [string]$Entry
    )

    $normalizedEntry = Normalize-PathEntry -PathValue $Entry
    $entries = New-Object System.Collections.Generic.List[string]
    $seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)

    $entries.Add($normalizedEntry) | Out-Null
    $seen.Add($normalizedEntry) | Out-Null

    foreach ($existingEntry in (Split-PathEntries -PathValue $PathValue)) {
        $normalizedExistingEntry = Normalize-PathEntry -PathValue $existingEntry
        if ($seen.Add($normalizedExistingEntry)) {
            $entries.Add($normalizedExistingEntry) | Out-Null
        }
    }

    return [pscustomobject]@{
        Path  = ($entries -join ";")
        Entry = $normalizedEntry
        Added = $true
    }
}

function Remove-PathEntry {
    param(
        [AllowEmptyString()]
        [string]$PathValue,
        [Parameter(Mandatory = $true)]
        [string]$Entry
    )

    $normalizedEntry = Normalize-PathEntry -PathValue $Entry
    $entries = New-Object System.Collections.Generic.List[string]
    $seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    $removed = $false

    foreach ($existingEntry in (Split-PathEntries -PathValue $PathValue)) {
        $normalizedExistingEntry = Normalize-PathEntry -PathValue $existingEntry
        if ([string]::Equals($normalizedExistingEntry, $normalizedEntry, [System.StringComparison]::OrdinalIgnoreCase)) {
            $removed = $true
            continue
        }
        if ($seen.Add($normalizedExistingEntry)) {
            $entries.Add($normalizedExistingEntry) | Out-Null
        }
    }

    return [pscustomobject]@{
        Path    = ($entries -join ";")
        Removed = $removed
    }
}

function Remove-PathEntries {
    param(
        [AllowEmptyString()]
        [string]$PathValue,
        [string[]]$Entries
    )

    $updatedPath = $PathValue
    foreach ($entry in ($Entries | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })) {
        $updatedPath = (Remove-PathEntry -PathValue $updatedPath -Entry $entry).Path
    }
    return $updatedPath
}

function Get-FFmpegDiscovery {
    $srcPath = Join-Path $ProjectRoot "src"
    $previousPythonPath = $env:PYTHONPATH

    if ([string]::IsNullOrWhiteSpace($previousPythonPath)) {
        $env:PYTHONPATH = $srcPath
    } else {
        $env:PYTHONPATH = "$srcPath;$previousPythonPath"
    }

    try {
        $rawJson = & uv run python -m matteflow.ffmpeg_env
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to run matteflow.ffmpeg_env discovery helper."
        }
    } finally {
        if ([string]::IsNullOrWhiteSpace($previousPythonPath)) {
            Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
        } else {
            $env:PYTHONPATH = $previousPythonPath
        }
    }
    if ([string]::IsNullOrWhiteSpace($rawJson)) {
        throw "Media tool discovery helper returned empty output."
    }

    return $rawJson | ConvertFrom-Json
}

function Get-FFmpegToolkitCacheRoot {
    return Join-Path $ProjectRoot ".cache\ffmpeg"
}

function Find-FFmpegToolkitBinDir {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RootPath
    )

    $binDirs = Get-ChildItem -Path $RootPath -Filter ffmpeg.exe -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.DirectoryName -match "\\bin$" }
    $selected = $binDirs | Select-Object -First 1
    if ($null -eq $selected) {
        return $null
    }
    return Normalize-PathEntry -PathValue $selected.DirectoryName
}

function Ensure-CompleteFFmpegToolkit {
    $cacheRoot = Get-FFmpegToolkitCacheRoot
    $toolkitRoot = Join-Path $cacheRoot "toolkit"
    $downloadRoot = Join-Path $cacheRoot "downloads"
    $zipPath = Join-Path $downloadRoot "ffmpeg-release-essentials.zip"
    $downloadUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

    $existingBinDir = Find-FFmpegToolkitBinDir -RootPath $toolkitRoot
    if ($existingBinDir) {
        $ffmpegExe = Join-Path $existingBinDir "ffmpeg.exe"
        $ffprobeExe = Join-Path $existingBinDir "ffprobe.exe"
        if ((Test-Path -LiteralPath $ffmpegExe) -and (Test-Path -LiteralPath $ffprobeExe)) {
            return [pscustomobject]@{
                Source = "download_cache"
                Root   = (Normalize-PathEntry -PathValue $toolkitRoot)
                BinDir = $existingBinDir
            }
        }
    }

    New-Item -ItemType Directory -Path $downloadRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $toolkitRoot -Force | Out-Null

    Write-Host "Downloading complete FFmpeg toolkit..."
    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath

    if (Test-Path -LiteralPath $toolkitRoot) {
        Get-ChildItem -LiteralPath $toolkitRoot -Force -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
    }

    Write-Host "Extracting FFmpeg toolkit..."
    Expand-Archive -LiteralPath $zipPath -DestinationPath $toolkitRoot -Force

    $binDir = Find-FFmpegToolkitBinDir -RootPath $toolkitRoot
    if (-not $binDir) {
        throw "Downloaded FFmpeg toolkit did not contain a usable bin directory."
    }

    $ffmpegExe = Join-Path $binDir "ffmpeg.exe"
    $ffprobeExe = Join-Path $binDir "ffprobe.exe"
    if (-not ((Test-Path -LiteralPath $ffmpegExe) -and (Test-Path -LiteralPath $ffprobeExe))) {
        throw "Downloaded FFmpeg toolkit is missing ffmpeg.exe or ffprobe.exe."
    }

    return [pscustomobject]@{
        Source = "downloaded_toolkit"
        Root   = (Normalize-PathEntry -PathValue $toolkitRoot)
        BinDir = $binDir
    }
}

function Resolve-MediaToolCommandPaths {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FFmpegPath,
        [Parameter(Mandatory = $true)]
        [string]$FFprobePath,
        [Parameter(Mandatory = $true)]
        [string]$BinDir
    )

    $ffmpegFileName = [System.IO.Path]::GetFileName($FFmpegPath)
    $ffprobeFileName = [System.IO.Path]::GetFileName($FFprobePath)
    $hasDirectPair = [string]::Equals($ffmpegFileName, "ffmpeg.exe", [System.StringComparison]::OrdinalIgnoreCase) `
        -and [string]::Equals($ffprobeFileName, "ffprobe.exe", [System.StringComparison]::OrdinalIgnoreCase)
    if ($hasDirectPair) {
        return [pscustomobject]@{
            Strategy        = "direct"
            PathEntry       = $BinDir
            FFmpegShimPath  = $null
            FFprobeShimPath = $null
            PersistUserPath = $true
        }
    }

    $shimDir = Join-Path $env:LOCALAPPDATA "MatteFlow\ffmpeg\bin"
    New-Item -ItemType Directory -Path $shimDir -Force | Out-Null

    $normalizedShimDir = Normalize-PathEntry -PathValue $shimDir
    $ffmpegShimPath = Join-Path $normalizedShimDir "ffmpeg.cmd"
    $ffprobeShimPath = Join-Path $normalizedShimDir "ffprobe.cmd"
    $ffmpegShimContents = @(
        "@echo off",
        "`"$FFmpegPath`" %*"
    )
    $ffprobeShimContents = @(
        "@echo off",
        "`"$FFprobePath`" %*"
    )

    $reuseShim = $false
    if ((Test-Path -LiteralPath $ffmpegShimPath) -and (Test-Path -LiteralPath $ffprobeShimPath)) {
        try {
            $existingFFmpegShimContents = Get-Content -LiteralPath $ffmpegShimPath -ErrorAction Stop
            $existingFFprobeShimContents = Get-Content -LiteralPath $ffprobeShimPath -ErrorAction Stop
            if ((($existingFFmpegShimContents -join "`n") -eq ($ffmpegShimContents -join "`n")) -and
                (($existingFFprobeShimContents -join "`n") -eq ($ffprobeShimContents -join "`n"))) {
                $reuseShim = $true
            }
        } catch {
            $reuseShim = $false
        }
    }

    if ($reuseShim) {
        return [pscustomobject]@{
            Strategy        = "shim"
            PathEntry       = $normalizedShimDir
            FFmpegShimPath  = $ffmpegShimPath
            FFprobeShimPath = $ffprobeShimPath
            PersistUserPath = $true
        }
    }

    try {
        Set-Content -LiteralPath $ffmpegShimPath -Value $ffmpegShimContents -Encoding ASCII -ErrorAction Stop
        Set-Content -LiteralPath $ffprobeShimPath -Value $ffprobeShimContents -Encoding ASCII -ErrorAction Stop
        return [pscustomobject]@{
            Strategy        = "shim"
            PathEntry       = $normalizedShimDir
            FFmpegShimPath  = $ffmpegShimPath
            FFprobeShimPath = $ffprobeShimPath
            PersistUserPath = $true
        }
    } catch [System.UnauthorizedAccessException] {
        $fallbackShimDir = Join-Path $ProjectRoot ".cache\ffmpeg\bin"
        New-Item -ItemType Directory -Path $fallbackShimDir -Force | Out-Null

        $normalizedFallbackShimDir = Normalize-PathEntry -PathValue $fallbackShimDir
        $fallbackFFmpegShimPath = Join-Path $normalizedFallbackShimDir "ffmpeg.cmd"
        $fallbackFFprobeShimPath = Join-Path $normalizedFallbackShimDir "ffprobe.cmd"
        Set-Content -LiteralPath $fallbackFFmpegShimPath -Value $ffmpegShimContents -Encoding ASCII -ErrorAction Stop
        Set-Content -LiteralPath $fallbackFFprobeShimPath -Value $ffprobeShimContents -Encoding ASCII -ErrorAction Stop

        return [pscustomobject]@{
            Strategy        = "shim_fallback"
            PathEntry       = $normalizedFallbackShimDir
            FFmpegShimPath  = $fallbackFFmpegShimPath
            FFprobeShimPath = $fallbackFFprobeShimPath
            PersistUserPath = $false
        }
    }
}

Write-Host "Discovering FFmpeg and FFprobe..."
$discovery = Get-FFmpegDiscovery

if (-not $discovery.complete) {
    $toolkit = Ensure-CompleteFFmpegToolkit
    $discovery = [pscustomobject]@{
        ffmpeg_path        = Join-Path $toolkit.BinDir "ffmpeg.exe"
        ffprobe_path       = Join-Path $toolkit.BinDir "ffprobe.exe"
        bin_dir            = $toolkit.BinDir
        source             = $toolkit.Source
        complete           = $true
        download_required  = $false
    }
}

if (-not $discovery.complete) {
    throw "Unable to locate a complete FFmpeg toolkit containing both ffmpeg.exe and ffprobe.exe."
}

$ffmpegPath = Normalize-PathEntry -PathValue $discovery.ffmpeg_path
$ffprobePath = Normalize-PathEntry -PathValue $discovery.ffprobe_path
$ffmpegBinDir = Normalize-PathEntry -PathValue $discovery.bin_dir
$commandPath = Resolve-MediaToolCommandPaths -FFmpegPath $ffmpegPath -FFprobePath $ffprobePath -BinDir $ffmpegBinDir

if ($commandPath.Strategy -like "shim*") {
    Write-Host "Configured FFmpeg shim: $($commandPath.FFmpegShimPath)"
    Write-Host "Configured FFprobe shim: $($commandPath.FFprobeShimPath)"
}

$sessionPathValue = $env:Path
$sharedShimDir = Normalize-PathEntry -PathValue (Join-Path $env:LOCALAPPDATA "MatteFlow\ffmpeg\bin")
$projectShimDir = Normalize-PathEntry -PathValue (Join-Path $ProjectRoot ".cache\ffmpeg\bin")
$sessionPathValue = Remove-PathEntries -PathValue $sessionPathValue -Entries @(
    $ffmpegBinDir,
    $commandPath.PathEntry,
    $sharedShimDir,
    $projectShimDir
)

$sessionPathResult = Prepend-UniquePathEntry -PathValue $sessionPathValue -Entry $commandPath.PathEntry
$env:Path = $sessionPathResult.Path
Write-Host "Added FFmpeg to current session PATH: $($sessionPathResult.Entry)"

$userPathUpdated = $false
if ($SkipUserPathPersist) {
    Write-Host "Skipping user PATH persistence."
} elseif (-not $commandPath.PersistUserPath) {
    Write-Host "Skipping user PATH persistence for sandbox-local FFmpeg shim."
} else {
    $currentUserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $currentUserPath = Remove-PathEntries -PathValue $currentUserPath -Entries @(
        $ffmpegBinDir,
        $commandPath.PathEntry,
        $sharedShimDir,
        $projectShimDir
    )
    $userPathResult = Prepend-UniquePathEntry -PathValue $currentUserPath -Entry $commandPath.PathEntry
    [Environment]::SetEnvironmentVariable("Path", $userPathResult.Path, "User")
    Write-Host "Added FFmpeg to user PATH: $($userPathResult.Entry)"
    $userPathUpdated = $true
}

$ffmpegCommand = Get-Command ffmpeg -ErrorAction SilentlyContinue
$ffprobeCommand = Get-Command ffprobe -ErrorAction SilentlyContinue
if ($null -eq $ffmpegCommand) {
    throw "FFmpeg is still not resolvable on PATH after configuration."
}
if ($null -eq $ffprobeCommand) {
    throw "FFprobe is still not resolvable on PATH after configuration."
}

$ffmpegVersionOutput = & ffmpeg -version 2>$null
$ffmpegVersionLine = $ffmpegVersionOutput | Select-Object -First 1
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($ffmpegVersionLine)) {
    throw "ffmpeg -version failed after PATH configuration."
}

$ffprobeVersionOutput = & ffprobe -version 2>$null
$ffprobeVersionLine = $ffprobeVersionOutput | Select-Object -First 1
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($ffprobeVersionLine)) {
    throw "ffprobe -version failed after PATH configuration."
}

Write-Host "FFmpeg verification passed: $ffmpegVersionLine"
Write-Host "FFprobe verification passed: $ffprobeVersionLine"

[pscustomobject]@{
    Found              = $true
    Source             = $discovery.source
    FFmpegPath         = $ffmpegPath
    FFprobePath        = $ffprobePath
    BinDir             = $ffmpegBinDir
    PathEntry          = $commandPath.PathEntry
    FFmpegShimPath     = $commandPath.FFmpegShimPath
    FFprobeShimPath    = $commandPath.FFprobeShimPath
    SessionPathUpdated = [bool]$sessionPathResult.Added
    UserPathUpdated    = [bool]$userPathUpdated
    FFmpegVersion      = $ffmpegVersionLine
    FFprobeVersion     = $ffprobeVersionLine
}
