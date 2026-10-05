
FROM python:3.11-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

FROM base AS foundry

RUN curl -L https://foundry.paradigm.xyz | bash \
    && /root/.foundry/bin/foundryup

ENV PATH="/root/.foundry/bin:${PATH}"

RUN forge --version && anvil --version && cast --version

FROM foundry AS python-deps

WORKDIR /app

COPY pyproject.toml ./

RUN pip install --upgrade pip setuptools wheel && \
    pip install \
    "gymnasium>=0.29.1" \
    "stable-baselines3[extra]>=2.2.1" \
    "shimmy>=1.3.0" \
    "numpy>=1.26.0" \
    "pandas>=2.1.0" \
    "pyarrow>=14.0.0" \
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

FROM python-deps AS final

WORKDIR /app

COPY . .

RUN mkdir -p contracts/lib && \
    git clone --depth 1 https://github.com/foundry-rs/forge-std.git contracts/lib/forge-std && \
    git clone --depth 1 https://github.com/Uniswap/v4-core.git contracts/lib/v4-core && \
    git clone --depth 1 https://github.com/Uniswap/v4-periphery.git contracts/lib/v4-periphery && \
    git clone --depth 1 https://github.com/OpenZeppelin/openzeppelin-contracts.git contracts/lib/openzeppelin-contracts && \
    git clone --depth 1 https://github.com/OpenZeppelin/uniswap-hooks.git contracts/lib/uniswap-hooks && \
    git clone --depth 1 https://github.com/transmissions11/solmate.git contracts/lib/solmate

RUN mkdir -p data/raw data/processed models runs

RUN forge build

RUN chmod +x scripts/docker-entrypoint.sh

EXPOSE 8545 6006

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import gymnasium; import stable_baselines3; print('OK')" || exit 1

CMD ["bash"]
