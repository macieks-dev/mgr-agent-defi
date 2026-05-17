#!/usr/bin/env bash
set -euo pipefail

echo "Setting up the project environment..."

# ── 1. Check prerequisites 
echo ""
echo "[1/5] Checking prerequisites..."

command -v python >/dev/null 2>&1 || { echo " Python not found. Install Python 3.10+."; exit 1; }
command -v pip >/dev/null 2>&1 || { echo " pip not found."; exit 1; }

PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python: $PYTHON_VERSION"

if command -v forge >/dev/null 2>&1; then
    echo "  Foundry: $(forge --version | head -1)"
else
    echo "  Foundry not found. Install it only if you need Solidity tests or deployment."
fi

# ── 2. Python environment ──
echo ""
echo "[2/5] Setting up Python environment..."

if [ ! -d ".venv" ]; then
    python -m venv .venv
    echo "  Created virtual environment: .venv"
fi

source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null

pip install --upgrade pip setuptools wheel -q
echo "  pip upgraded"

# ── 3. Install Python dependencies ──
echo ""
echo "[3/5] Installing Python dependencies..."

if command -v poetry >/dev/null 2>&1; then
    poetry install
else
    pip install gymnasium stable-baselines3[extra] shimmy numpy pandas \
        web3 torch tensorboard matplotlib scipy python-dotenv requests tqdm \
        pytest pytest-cov black isort mypy -q
fi

echo "  ✓ Python packages installed"

# ── 4. Foundry dependencies ──
echo ""
echo "[4/5] Installing Foundry dependencies..."

if command -v forge >/dev/null 2>&1; then
    forge install foundry-rs/forge-std --no-commit 2>/dev/null || true
    forge install Uniswap/v4-core --no-commit 2>/dev/null || true
    forge install Uniswap/v4-periphery --no-commit 2>/dev/null || true
    forge install OpenZeppelin/openzeppelin-contracts --no-commit 2>/dev/null || true
    echo "  ✓ Foundry dependencies installed"
else
    echo "    Skipping (Foundry not installed)"
fi

# ── 5. Environment file ──
echo ""
echo "[5/5] Setting up environment..."

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  Created .env from .env.example"
else
    echo "  .env already exists"
fi

echo ""
echo "Setup complete."
echo ""
echo "Recommended next steps:"
echo "  1. Review .env only if you need custom RPC or external data access"
echo "  2. Run Python tests: pytest tests/ -v"
echo "  3. Run final V4 evaluation: python scripts/final_evaluation_v4.py"
echo "  4. Generate V4 figures: python scripts/model_explainability.py"
echo "  5. Generate the final figure index: python poprawki/generate_spisy_docx_v2.py"
echo "  6. Run Solidity tests if needed: forge test -vvv"
