<#
setup_cuda.ps1
PowerShell helper to detect NVIDIA GPU/driver, determine TensorFlow ↔ CUDA/cuDNN compatibility,
and open the appropriate download pages (or optionally download installers).

Usage:
  .\scripts\setup_cuda.ps1 [-VenvPath .\.venv-captcha] [-TFVersion 2.12.0] [-AutoOpen]

Notes:
 - This script performs detection and guided actions only. Full automatic install of CUDA/cuDNN
   is not performed because it requires admin privileges and user confirmation.
 - If TF version is not provided, the script will try to read it from the venv if present,
   otherwise you will be prompted.
#>

param(
    [string]$VenvPath = ".\.venv-captcha",
    [string]$TFVersion = $null,
    [switch]$AutoOpen
)

function Write-Info($s){ Write-Host $s -ForegroundColor Cyan }
function Write-Warn($s){ Write-Host $s -ForegroundColor Yellow }
function Write-Err($s){ Write-Host $s -ForegroundColor Red }

# Get nvidia-smi output
function Get-NvidiaSmiInfo() {
    try {
        $smi = & nvidia-smi -L 2>&1
        $full = & nvidia-smi 2>&1
        return @{ List = $smi; Full = $full }
    } catch {
        return $null
    }
}

# Try to read python and tensorflow version from venv
function Get-TensorFlowVersionFromVenv([string]$venvPath) {
    try {
        $venv = Get-Item -LiteralPath $venvPath -ErrorAction Stop
    } catch {
        return $null
    }

    if ($IsWindows) {
        $python = Join-Path $venvPath 'Scripts\python.exe'
    } else {
        $python = Join-Path $venvPath 'bin/python'
    }

    if (-not (Test-Path $python)) { return $null }

    $code = "import tensorflow as tf; import sys; print(tf.__version__); print(getattr(tf.test,'is_built_with_cuda', lambda:False)()); print(tf.config.list_physical_devices('GPU'))"
    try {
        # Call the python executable directly with the -c argument using the call operator
        $out = & "$python" -c $code 2>&1
        return $out -join "`n"
    } catch {
        return $null
    }
}

# Try to fetch TF GPU compatibility info from tensorflow website
function Get-TensorFlowGpuCompatibility([string]$tfVersion) {
    Write-Info "Attempting to fetch TensorFlow GPU compatibility information from tensorflow.org..."
    try {
        $url = 'https://www.tensorflow.org/install/source#gpu'
        $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -ErrorAction Stop
        $text = $resp.Content

        # Search for 'CUDA' and 'cuDNN' occurrences near each other
        $cudaMatches = [regex]::Matches($text, 'CUDA\s*([0-9]+\.[0-9]+)') | ForEach-Object { $_.Groups[1].Value } | Select-Object -Unique
        $cudnnMatches = [regex]::Matches($text, 'cuDNN\s*([0-9]+\.?[0-9]+)') | ForEach-Object { $_.Groups[1].Value } | Select-Object -Unique

        return @{ CUDA = $cudaMatches; cuDNN = $cudnnMatches; Source = $url }
    } catch {
        Write-Warn "Could not fetch tensorflow.org content: $_"
        return $null
    }
}

# Open download pages in default browser
function Open-Downloads([string]$cudaVer, [string]$cudnnVer) {
    Write-Info "Opening NVIDIA CUDA download page and cuDNN page in your browser..."
    Start-Process "https://developer.nvidia.com/cuda-downloads"
    Start-Process "https://developer.nvidia.com/rdp/cudnn-download"
    Write-Info "If you want, download the CUDA Toolkit matching: $cudaVer and cuDNN: $cudnnVer"
    Write-Info "Follow TensorFlow install docs for exact file placement (copy cuDNN DLLs into CUDA folder, etc.)."
}

# Execution starts here
Write-Info "Detecting NVIDIA GPU and driver via nvidia-smi..."
$ninfo = Get-NvidiaSmiInfo
if ($null -eq $ninfo) {
    Write-Warn "nvidia-smi not found or failed. Make sure NVIDIA driver is installed and nvidia-smi is on PATH."
} else {
    Write-Info "nvidia-smi -L output:"
    Write-Host $ninfo.List
    Write-Info "nvidia-smi full output:"
    Write-Host $ninfo.Full
}

# Check venv / TF version
if (-not $TFVersion) {
    Write-Info "Trying to detect TensorFlow in venv path: $VenvPath"
    $tfInfo = Get-TensorFlowVersionFromVenv -venvPath $VenvPath
    if ($tfInfo) {
        Write-Info "TensorFlow info from venv:"
        Write-Host $tfInfo
        $lines = $tfInfo -split "`n" | Where-Object { $_ -ne '' }
        if ($lines.Count -ge 1) { $TFVersion = $lines[0].Trim() }
    } else {
        Write-Warn "Could not detect TensorFlow in venv. You can pass -TFVersion explicitly or activate the venv and run this script from inside it."
    }
}

if (-not $TFVersion) {
    $TFVersion = Read-Host "Enter desired TensorFlow version to support (e.g. 2.12.0)"
}

Write-Info "Target TensorFlow version: $TFVersion"

# Fetch compatibility from tensorflow.org
$compat = Get-TensorFlowGpuCompatibility -tfVersion $TFVersion
if ($compat) {
    Write-Info "Found potential CUDA versions on tensorflow.org:"
    Write-Host ($compat.CUDA -join ', ')
    Write-Info "Found potential cuDNN versions on tensorflow.org:"
    Write-Host ($compat.cuDNN -join ', ')
    Write-Info "Source: $($compat.Source)"
}

# Try quick heuristic: if nvidia-smi returned 'CUDA Version: XX.Y', parse it
$detectedCuda = $null
if ($ninfo -and $ninfo.Full) {
    $m = [regex]::Match($ninfo.Full, 'CUDA Version:\s*([0-9]+\.[0-9]+)')
    if ($m.Success) { $detectedCuda = $m.Groups[1].Value }
}

if ($detectedCuda) { Write-Info "Driver reports CUDA Version: $detectedCuda" }

# Decide recommendation
if ($compat -and $compat.CUDA.Count -gt 0) {
    $recommendedCuda = $compat.CUDA[0]
    $recommendedCudnn = if ($compat.cuDNN.Count -gt 0) { $compat.cuDNN[0] } else { 'See tensorflow docs' }
    Write-Info "Recommended (first match) CUDA: $recommendedCuda, cuDNN: $recommendedCudnn"
    if ($AutoOpen) { Open-Downloads -cudaVer $recommendedCuda -cudnnVer $recommendedCudnn }
    Write-Info "Next steps: Download CUDA Toolkit $recommendedCuda and corresponding cuDNN; follow TF install guide to put cuDNN files into CUDA directories."
} else {
    Write-Warn "Could not determine a recommended CUDA/cuDNN combination automatically."
    Write-Info "Options:"
    Write-Host "  1) Open CUDA/cuDNN download pages now"
    Write-Host "  2) Consult https://www.tensorflow.org/install/gpu for the compatibility matrix"
    if ($AutoOpen) { Start-Process "https://www.tensorflow.org/install/gpu" }
}

Write-Info "Manual install checklist (summary):"
Write-Host "  - Install NVIDIA driver (you already have driver)"
Write-Host "  - Download + install CUDA Toolkit matching TF requirements"
Write-Host "  - Download cuDNN matching same CUDA (copy cuDNN DLLs/lib/include into CUDA folders)"
Write-Host "  - Reboot"
Write-Host "  - Activate venv and install GPU-enabled tensorflow wheel (pip install tensorflow==<version>)"

Write-Info "If you want, run this script again with -AutoOpen to open download pages automatically."

# End of script
