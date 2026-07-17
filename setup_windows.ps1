# AI-Native-Radio -- Windows setup: .venv + pyadi-iio with a WORKING libiio binding.
#
# On Windows, pyadi-iio's ctypes bindings need the OFFICIAL Analog Devices libiio.dll
# (installed system-wide by ADI's Windows installer) on PATH -- unlike Ubuntu, there is no
# custom-build workaround needed here; pip's pylibiio wheel is fine once that DLL exists.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File setup_windows.ps1
#   .venv\Scripts\Activate.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "== AI-Native-Radio Windows setup =="

python -m venv .venv
& .venv\Scripts\pip install --upgrade pip
& .venv\Scripts\pip install -r requirements.txt
& .venv\Scripts\pip install pyadi-iio

$importOk = $true
try {
    & .venv\Scripts\python -c "import adi" 2>$null
    if ($LASTEXITCODE -ne 0) { $importOk = $false }
} catch {
    $importOk = $false
}

if ($importOk) {
    Write-Host "== pyadi-iio/libiio import OK =="
} else {
    Write-Host ""
    Write-Host "!! 'import adi' failed -- you need the official ADI libiio Windows driver/DLL:"
    Write-Host "   1. Download + run the libiio Windows installer (matches your pyadi-iio version):"
    Write-Host "      https://github.com/analogdevicesinc/libiio/releases"
    Write-Host "   2. Also install the PlutoSDR USB drivers if this is your first ADALM-Pluto:"
    Write-Host "      https://wiki.analog.com/university/tools/pluto/drivers/windows"
    Write-Host "   3. Re-run this script (or just re-open a shell so PATH picks up the new DLL)."
    exit 1
}

Write-Host ""
Write-Host "== done. Each new shell: .venv\Scripts\Activate.ps1 =="
Write-Host "== offline sanity check: python test_offline.py =="
