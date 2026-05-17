# ══════════════════════════════════════════════════════════════════════
#  Uniswap V4 RL Hook — Docker Image
#  Python 3.11 + Foundry + PyTorch (CUDA optional)
# ══════════════════════════════════════════════════════════════════════

# ── Stage 1: Base with Python + system deps ──
FROM python:3.11-slim-bookworm AS base

# Prevent Python from writing .pyc and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: Install Foundry ──
FROM base AS foundry

# Install Foundry (forge, cast, anvil)
RUN curl -L https://foundry.paradigm.xyz | bash \
    && /root/.foundry/bin/foundryup

ENV PATH="/root/.foundry/bin:${PATH}"

# Verify Foundry installation
RUN forge --version && anvil --version && cast --version

# ── Stage 3: Python dependencies ──
FROM foundry AS python-deps

WORKDIR /app

# Copy only dependency file first (for Docker cache optimization)
COPY pyproject.toml ./

# Install Python dependencies (without project itself)
# Using pip directly instead of Poetry for simplicity in Docker
RUN pip install --upgrade pip setuptools wheel && \
    pip install \
    "gymnasium>=0.29.1" \
    "stable-baselines3[extra]>=2.2.1" \
    "shimmy>=1.3.0" \
    "numpy>=1.26.0" \
    "pandas>=2.1.0" \
    "web3>=6.15.0" \
    "torch>=2.1.0" \
    "tensorboard>=2.15.0" \
    "matplotlib>=3.8.0" \
    "scipy>=1.11.0" \
    "python-dotenv>=1.0.0" \
    "requests>=2.31.0" \
    "tqdm>=4.66.0" \
    "pytest>=7.4.0" \
    "pytest-cov>=4.1.0"

# ── Stage 4: Final image with project code ──
FROM python-deps AS final

WORKDIR /app

# Copy entire project
COPY . .

# Install Foundry submodules (Uniswap v4-core, v4-periphery, forge-std, OpenZeppelin)
# Clone into contracts/lib/ to match remappings.txt
RUN mkdir -p contracts/lib && \
    git clone --depth 1 https://github.com/foundry-rs/forge-std.git contracts/lib/forge-std || true && \
    git clone --depth 1 https://github.com/Uniswap/v4-core.git contracts/lib/v4-core || true && \
    git clone --depth 1 https://github.com/Uniswap/v4-periphery.git contracts/lib/v4-periphery || true && \
    git clone --depth 1 https://github.com/OpenZeppelin/openzeppelin-contracts.git contracts/lib/openzeppelin-contracts || true && \
    git clone --depth 1 https://github.com/OpenZeppelin/uniswap-hooks.git contracts/lib/uniswap-hooks || true && \
    git clone --depth 1 https://github.com/transmissions11/solmate.git contracts/lib/solmate || true

# Create data directories
RUN mkdir -p data/raw data/processed models runs

# Build Solidity contracts (foundry.toml is at project root)
RUN forge build || echo "Warning: forge build had issues — will retry at runtime"

# Make entrypoint executable
RUN chmod +x scripts/docker-entrypoint.sh 2>/dev/null || true

# Default env file
RUN if [ ! -f .env ]; then cp .env.example .env 2>/dev/null || true; fi

# Expose ports
# 8545 = Anvil (local EVM)
# 6006 = TensorBoard
EXPOSE 8545 6006

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import gymnasium; import stable_baselines3; print('OK')" || exit 1

# Default command
CMD ["bash"]
