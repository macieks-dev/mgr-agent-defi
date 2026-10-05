@echo off
echo Setting up the project environment...

echo.
echo [1/4] Checking prerequisites...
python --version
if ERRORLEVEL 1 (
    echo ERROR: Python not found. Install Python 3.10+.
    exit /b 1
)

echo.
echo [2/4] Setting up Python environment...
if not exist ".venv" (
    python -m venv .venv
    echo   Created virtual environment: .venv
)

call .venv\Scripts\activate.bat

pip install --upgrade pip setuptools wheel -q

echo.
echo [3/4] Installing Python dependencies...
pip install gymnasium "stable-baselines3[extra]" shimmy numpy pandas pyarrow ^
    web3 torch tensorboard matplotlib scipy python-dotenv requests tqdm ^
    pytest pytest-cov black isort mypy -q

echo   Python packages installed

echo.
echo [4/4] Setting up environment...
if not exist ".env" (
    copy .env.example .env
    echo   Created .env from .env.example
) else (
    echo   .env already exists
)

echo.
echo Setup complete.
echo.
echo Recommended next steps:
echo   1. Review .env only if you need custom RPC or external data access
echo   2. Run Python tests: pytest tests/ -v
echo   3. Run final V4 evaluation: python scripts/final_evaluation_v4.py
echo   4. Generate V4 figures: python scripts/model_explainability.py
echo   5. Generate thesis diagrams: python poprawki/generate_all_diagrams_v4.py
echo   6. Run Solidity tests if needed: forge test -vvv
