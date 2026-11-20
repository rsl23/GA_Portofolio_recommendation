param(
    [ValidateSet('start', 'stop', 'restart', 'status')]
    [string]$Action = 'start',
    [ValidateRange(1, 60)]
    [int]$GracefulTimeoutSeconds = 10
)

$ErrorActionPreference = 'Stop'
$backendProjectRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$backendRuntimeDir = Join-Path $backendProjectRoot 'outputs\backend-runtime'
$backendStateFile = Join-Path $backendRuntimeDir 'state.json'
$backendPython = Join-Path $backendProjectRoot '.venv\Scripts\python.exe'

function Get-ManagedBackend {
    if (-not (Test-Path -LiteralPath $backendStateFile)) { return $null }
    $backendState = Get-Content -LiteralPath $backendStateFile -Raw | ConvertFrom-Json
    if ([System.IO.Path]::GetFullPath($backendState.project_root) -ne $backendProjectRoot) {
        throw 'Identitas backend tidak cocok dengan proyek ini.'
    }
    $backendProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$backendState.pid)"
    if (-not $backendProcess) { return $null }
    if ($backendProcess.CommandLine -notlike '*-m src.backend.serve*' -or
        $backendProcess.CommandLine -notlike "*$backendStateFile*") {
        throw 'ID proses sudah berubah; proses lain tidak akan dihentikan.'
    }
    return $backendState
}

function Stop-ManagedBackend {
    $backendState = Get-ManagedBackend
    if (-not $backendState) { Write-Output 'Backend terkelola sudah berhenti.'; return }
    $backendStopFile = [System.IO.Path]::GetFullPath($backendState.stop_file)
    if ([System.IO.Path]::GetDirectoryName($backendStopFile) -ne $backendRuntimeDir -or
        [System.IO.Path]::GetFileName($backendStopFile) -ne "stop-$($backendState.pid).flag") {
        throw 'Lokasi permintaan berhenti tidak cocok.'
    }
    Set-Content -LiteralPath $backendStopFile -Value 'stop' -Encoding utf8
    for ($backendWait = 0; $backendWait -lt ($GracefulTimeoutSeconds * 5); $backendWait++) {
        if (-not (Get-Process -Id ([int]$backendState.pid) -ErrorAction SilentlyContinue)) {
            Write-Output 'Backend berhenti dengan baik.'
            return
        }
        Start-Sleep -Milliseconds 200
    }
    # A library blocked in native/network I/O may ignore cooperative shutdown.
    # Recheck both the state identity and a held process handle before killing.
    $backendStillRunning = Get-ManagedBackend
    if (-not $backendStillRunning) { Write-Output 'Backend sudah berhenti.'; return }
    if ($backendStillRunning.pid -ne $backendState.pid -or
        $backendStillRunning.created_at -ne $backendState.created_at) {
        throw 'Identitas backend berubah saat menunggu; proses pengganti tidak dihentikan.'
    }
    $backendForceProcess = Get-Process -Id ([int]$backendState.pid) -ErrorAction SilentlyContinue
    if ($backendForceProcess) {
        $backendStartedUtc = $backendForceProcess.StartTime.ToUniversalTime()
        $backendRecordedUtc = [DateTimeOffset]::Parse($backendState.created_at).UtcDateTime
        $backendBirthGap = ($backendRecordedUtc - $backendStartedUtc).TotalSeconds
        if ($backendBirthGap -lt -1 -or $backendBirthGap -gt 60) {
            throw 'Waktu mulai proses tidak cocok; proses lain tidak dihentikan.'
        }
        Write-Output 'Batas tunggu habis. Menghentikan proses backend proyek ini yang masih tersangkut.'
        $backendForceProcess.Kill()
        if (-not $backendForceProcess.WaitForExit(5000)) {
            throw 'Proses belum keluar setelah penghentian. Periksa proses sebelum memulai kembali.'
        }
    }
    if (Test-Path -LiteralPath $backendStateFile) {
        $backendRemainingState = Get-Content -LiteralPath $backendStateFile -Raw | ConvertFrom-Json
        if ($backendRemainingState.pid -eq $backendState.pid -and
            $backendRemainingState.created_at -eq $backendState.created_at) {
            Remove-Item -LiteralPath $backendStateFile
        }
    }
    if (Test-Path -LiteralPath $backendStopFile) { Remove-Item -LiteralPath $backendStopFile }
    Write-Output 'Backend berhenti. Jadwal yang sebelumnya selesai tetap tersimpan.'
}

if ($Action -eq 'stop' -or $Action -eq 'restart') {
    Stop-ManagedBackend
    if ($Action -eq 'stop') { return }
}
$backendExisting = Get-ManagedBackend
if ($Action -eq 'status') {
    if ($backendExisting) {
        try {
            Invoke-WebRequest 'http://127.0.0.1:8000/' -TimeoutSec 2 -UseBasicParsing | Out-Null
            Write-Output "Backend siap: http://127.0.0.1:8000 (PID $($backendExisting.pid))."
        } catch {
            Write-Output "Proses backend hidup (PID $($backendExisting.pid)), tetapi API belum merespons. Periksa log atau gunakan perintah stop."
        }
    }
    else { Write-Output 'Backend terkelola tidak sedang berjalan.' }
    Write-Output "Log: $backendRuntimeDir"
    return
}
if ($backendExisting) { Write-Output 'Backend sudah aktif.'; return }
if (-not (Test-Path -LiteralPath $backendPython)) { throw 'Python .venv proyek tidak ditemukan.' }
if (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Port 8000 masih digunakan. Hentikan backend lama sebelum memakai peluncur ini.'
}
New-Item -ItemType Directory -Path $backendRuntimeDir -Force | Out-Null
$backendLogStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$backendStdout = Join-Path $backendRuntimeDir "backend-$backendLogStamp.stdout.log"
$backendStderr = Join-Path $backendRuntimeDir "backend-$backendLogStamp.stderr.log"
$backendArguments = @('-X', 'utf8', '-u', '-m', 'src.backend.serve', '--state-file', ('"{0}"' -f $backendStateFile))
Start-Process -FilePath $backendPython -ArgumentList $backendArguments -WorkingDirectory $backendProjectRoot `
    -WindowStyle Hidden -RedirectStandardOutput $backendStdout -RedirectStandardError $backendStderr | Out-Null
for ($backendStartWait = 0; $backendStartWait -lt 50; $backendStartWait++) {
    $backendStarted = Get-ManagedBackend
    if ($backendStarted) { break }
    Start-Sleep -Milliseconds 100
}
if ($backendStarted) {
    Write-Output "Proses backend dimulai (PID $($backendStarted.pid)); kesiapan API tercatat di log."
} else {
    throw "Proses backend belum terdeteksi. Periksa log: $backendStderr"
}
Write-Output "Log: $backendStdout"
Write-Output "Log: $backendStderr"
Write-Output 'Setelah mengubah kode backend: .\start_backend.ps1 -Action restart'
Write-Output 'Untuk menghentikan backend: .\start_backend.ps1 -Action stop'
