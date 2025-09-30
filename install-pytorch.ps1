# install-pytorch.ps1

Write-Host "[INFO] Installing PyTorch inside virtual environment..."

# Ensure venv is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Error "[ERROR] Virtual environment not detected. Please activate your venv first in VSCode terminal."
    exit 1
}

# Run pip install inside the venv
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] PyTorch installation complete."
}
else {
    Write-Error "[ERROR] PyTorch installation failed."
}
