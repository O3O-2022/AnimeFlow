# AnimeFlow: download AI models and tools (called by setup.bat)
# Items that already exist are skipped. If a download fails, use the manual links in README.md.
param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is much faster without the progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root = Split-Path -Parent $PSScriptRoot
$TP = Join-Path $Root "third_party"
$Tmp = Join-Path $env:TEMP "animeflow_setup"
New-Item -ItemType Directory -Force $Tmp, $TP | Out-Null
$failed = @()

function Get-File([string]$Url, [string]$Out) {
    Write-Host "    downloading: $Url"
    Invoke-WebRequest -Uri $Url -OutFile $Out -UseBasicParsing
}

function Expand-To([string]$Zip, [string]$Name) {
    $dst = Join-Path $Tmp $Name
    if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
    Expand-Archive -Path $Zip -DestinationPath $dst -Force
    return $dst
}

function Step([string]$Name, [string]$Check, [scriptblock]$Action) {
    if (Test-Path $Check) { Write-Host "  [ok]  $Name (already present)"; return }
    Write-Host "  [get] $Name"
    try { & $Action; Write-Host "    done" }
    catch { Write-Host "    FAILED: $($_.Exception.Message)" -ForegroundColor Yellow; $script:failed += $Name }
}

# 1. FFmpeg 8.1 (BtbN stable build; the nightly build had an AAC encoder hang)
Step "FFmpeg 8.1" (Join-Path $Root "tools\ffmpeg\ffmpeg.exe") {
    $zip = Join-Path $Tmp "ffmpeg.zip"
    Get-File "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-gpl-8.1.zip" $zip
    $x = Expand-To $zip "ffmpeg"
    $bin = (Get-ChildItem $x -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1).DirectoryName
    $dst = Join-Path $Root "tools\ffmpeg"
    New-Item -ItemType Directory -Force $dst | Out-Null
    Copy-Item (Join-Path $bin "*.exe") $dst -Force
}

# 2. Practical-RIFE code (MIT)
Step "Practical-RIFE code" (Join-Path $TP "Practical-RIFE\model\warplayer.py") {
    $zip = Join-Path $Tmp "rife_code.zip"
    Get-File "https://github.com/hzwer/Practical-RIFE/archive/refs/heads/main.zip" $zip
    $x = Expand-To $zip "rife_code"
    $dst = Join-Path $TP "Practical-RIFE"
    New-Item -ItemType Directory -Force $dst | Out-Null
    Copy-Item (Join-Path $x "Practical-RIFE-main\*") $dst -Recurse -Force
}

# 3. RIFE 4.25 model (Google Drive, released by the RIFE author)
Step "RIFE 4.25 model" (Join-Path $TP "Practical-RIFE\train_log\flownet.pkl") {
    $zip = Join-Path $Tmp "rife425.zip"
    & $Python -m gdown "1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg" -O $zip -q
    if (-not (Test-Path $zip)) { throw "Google Drive download failed (try again later)" }
    $x = Expand-To $zip "rife425"
    $dst = Join-Path $TP "Practical-RIFE\train_log"
    New-Item -ItemType Directory -Force $dst | Out-Null
    foreach ($f in "flownet.pkl", "IFNet_HDv3.py", "RIFE_HDv3.py", "refine.py") {
        Copy-Item (Join-Path $x "train_log\$f") $dst -Force
    }
}

# 4. Depth Anything V2 code + Small model (Apache-2.0, used by the experimental relighting)
Step "Depth Anything V2 code" (Join-Path $TP "Depth-Anything-V2\depth_anything_v2\dpt.py") {
    $zip = Join-Path $Tmp "depth_code.zip"
    Get-File "https://github.com/DepthAnything/Depth-Anything-V2/archive/refs/heads/main.zip" $zip
    $x = Expand-To $zip "depth_code"
    $dst = Join-Path $TP "Depth-Anything-V2"
    New-Item -ItemType Directory -Force $dst | Out-Null
    Copy-Item (Join-Path $x "Depth-Anything-V2-main\*") $dst -Recurse -Force
}
Step "Depth Anything V2 Small model" (Join-Path $TP "Depth-Anything-V2\checkpoints\depth_anything_v2_vits.pth") {
    $dst = Join-Path $TP "Depth-Anything-V2\checkpoints"
    New-Item -ItemType Directory -Force $dst | Out-Null
    Get-File "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth" (Join-Path $dst "depth_anything_v2_vits.pth")
}

# 5. Real-ESRGAN AnimeVideo v3 (BSD-3, source restoration)
Step "Real-ESRGAN AnimeVideo v3" (Join-Path $TP "restore\realesr-animevideov3.pth") {
    $dst = Join-Path $TP "restore"
    New-Item -ItemType Directory -Force $dst | Out-Null
    Get-File "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth" (Join-Path $dst "realesr-animevideov3.pth")
}

Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue
if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "Not downloaded: $($failed -join ', ')" -ForegroundColor Yellow
    Write-Host "Run setup.bat again, or download them manually (see 'Manual install' in README.md)." -ForegroundColor Yellow
    exit 1
}
exit 0
