# ============================================================================
# LlamaIndex Text-to-SQL Chatbot - New Virtual Environment Setup Script
# ============================================================================
# Run this script in PowerShell from the project directory
# (c:\Eroolt\Chatbot\chatbot_with_masking_updated)
# ============================================================================

# Step 1: Navigate to project directory
Set-Location "c:\Eroolt\Chatbot\chatbot_with_masking_updated"
Write-Host "📁 Changed to project directory: $(Get-Location)" -ForegroundColor Green

# Step 2: Remove old venv if it exists (optional - uncomment if needed)
# if (Test-Path "venv_llamaindex11") {
#     Write-Host "🗑️ Removing old virtual environment..." -ForegroundColor Yellow
#     Remove-Item -Recurse -Force "venv_llamaindex11"
# }

# Step 3: Create new virtual environment
Write-Host "🔧 Creating new virtual environment..." -ForegroundColor Cyan
python -m venv venv_llamaindex11

# Step 4: Activate the virtual environment
Write-Host "⚡ Activating virtual environment..." -ForegroundColor Cyan
& ".\venv_llamaindex11\Scripts\Activate.ps1"

# Step 5: Upgrade pip
Write-Host "📦 Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Step 6: Install requirements
Write-Host "📥 Installing requirements from requirements_new.txt..." -ForegroundColor Cyan
pip install -r requirements_new.txt

# Step 7: Verify installation
Write-Host "`n✅ Installation complete! Verifying packages..." -ForegroundColor Green
Write-Host "-" * 60

# Check key packages
$packages = @(
    "llama-index-core",
    "llama-index-llms-openai",
    "llama-index-llms-ollama",
    "llama-index-embeddings-openai",
    "llama-index-embeddings-ollama",
    "pydantic",
    "sqlalchemy",
    "flask"
)

foreach ($pkg in $packages) {
    $version = pip show $pkg 2>$null | Select-String "^Version:"
    if ($version) {
        Write-Host "  ✓ $pkg - $($version -replace 'Version: ', '')" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $pkg - NOT INSTALLED" -ForegroundColor Red
    }
}

Write-Host "`n" + "=" * 60
Write-Host "🎉 Virtual environment setup complete!" -ForegroundColor Green
Write-Host "=" * 60
Write-Host "`nTo activate this environment in the future, run:"
Write-Host "  .\venv_llamaindex11\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "`nTo run the chatbot:"
Write-Host "  python run_web.py" -ForegroundColor Yellow
Write-Host ""
