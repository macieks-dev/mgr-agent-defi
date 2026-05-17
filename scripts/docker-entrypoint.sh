#!/usr/bin/env bash
set -euo pipefail

log() { echo "[hook] $*"; }
ok() { echo "[ok] $*"; }
err() { echo "[error] $*" >&2; }

# ─── Defaults ────────────────────────────────────────────────────────
RPC_URL="${ANVIL_RPC_URL:-http://anvil:8545}"
DEPLOYER_KEY="${DEPLOYER_PRIVATE_KEY:-0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80}"

# ─── Wait for Anvil ─────────────────────────────────────────────────
wait_for_anvil() {
    log "Waiting for Anvil at ${RPC_URL}..."
    for i in $(seq 1 30); do
        if cast block-number --rpc-url "$RPC_URL" &>/dev/null; then
            ok "Anvil is ready (block $(cast block-number --rpc-url "$RPC_URL"))"
            return 0
        fi
        sleep 1
    done
    err "Anvil not reachable after 30s"
    exit 1
}

# ─── Commands ────────────────────────────────────────────────────────
CMD="${1:-help}"

case "$CMD" in

  forge-build)
    log "Building Solidity contracts..."
    cd /app
    forge build --force
    ok "Forge build complete"
    ;;

  forge-test)
    log "Running Solidity tests..."
    cd /app
    forge test -vvv
    ok "Forge tests complete"
    ;;

  pytest)
    log "Running Python tests..."
    cd /app
    python -m pytest tests/ -v --tb=short
    ok "Python tests complete"
    ;;

  train)
    log "Starting RL training..."
    cd /app
    python -m src.training.train_ppo \
        --timesteps "${TOTAL_TIMESTEPS:-2000000}" \
        --n-envs "${N_ENVS:-16}" \
        --seed "${SEED:-42}" \
        --experiment "${EXPERIMENT:-ppo_uniswap_v4}"
    ok "Training complete"
    ;;

  fetch-data)
    log "Fetching CEX data from Binance Data Vision (12s block-level)..."
    cd /app
    python scripts/fetch_data.py --fetch-cex -v
    ok "Data pipeline complete"
    ;;

  deploy)
    wait_for_anvil
    log "Deploying VolatilityFeeHook to Anvil..."
    cd /app
    forge script contracts/script/DeployHook.s.sol:DeployHook \
        --rpc-url "$RPC_URL" \
        --private-key "$DEPLOYER_KEY" \
        --broadcast \
        -vvv
    ok "Deployment complete"
    ;;

  bridge)
    wait_for_anvil
    log "Starting agent bridge loop..."
    cd /app
    python -m src.bridge.web3_bridge
    ;;

  all-tests)
    log "Running ALL tests (Forge + Python)..."
    cd /app && forge test -vvv
    cd /app && python -m pytest tests/ -v --tb=short
    ok "All tests passed!"
    ;;

  shell|bash)
    exec bash
    ;;

  help|*)
    cat <<'EOF'
Available commands:
  forge-build  Build Solidity contracts
  forge-test   Run Forge test suite
  pytest       Run Python test suite
  fetch-data   Download CEX data and build features
  train        Start PPO training
  deploy       Deploy VolatilityFeeHook to local Anvil
  bridge       Start the RL agent to on-chain bridge loop
  all-tests    Run Forge and Python tests
  shell        Open an interactive shell
EOF
    ;;
esac
