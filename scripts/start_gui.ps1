param(
    [int]$Port = 7860,
    [switch]$Share,
    [switch]$DebugMode
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

Write-Host "Checking port $Port..."
$connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
$processIds = $connections | Select-Object -ExpandProperty OwningProcess -Unique

foreach ($processId in $processIds) {
    if (-not $processId) {
        continue
    }

    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        continue
    }

    Write-Host "Stopping process $processId ($($process.ProcessName)) on port $Port..."
    Stop-Process -Id $processId -Force
}

$deadline = (Get-Date).AddSeconds(15)
do {
    Start-Sleep -Milliseconds 500
    $remaining = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
} while ($remaining -and (Get-Date) -lt $deadline)

if ($remaining) {
    $remainingProcessIds = $remaining | Select-Object -ExpandProperty OwningProcess -Unique
    throw "Port $Port is still occupied after waiting. Owning process id(s): $($remainingProcessIds -join ', ')"
}

$argsList = @("run", "python", "scripts/web_gui.py", "--port", "$Port")
if ($Share) {
    $argsList += "--share"
}
if ($DebugMode) {
    $argsList += "--debug"
}

Write-Host "Starting MatteFlow GUI on http://localhost:$Port ..."
& uv @argsList
