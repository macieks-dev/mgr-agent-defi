"""
Configuration constants and .env loading.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Config:
    """Central configuration for the project."""

    # ── Paths ──
    project_root: Path = PROJECT_ROOT
    data_raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    data_processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    model_save_dir: Path = PROJECT_ROOT / "models"
    tensorboard_log_dir: Path = PROJECT_ROOT / "runs"

    # ── Blockchain ──
    anvil_rpc_url: str = field(
        default_factory=lambda: os.getenv("ANVIL_RPC_URL", "http://127.0.0.1:8545")
    )
    eth_rpc_url: str = field(
        default_factory=lambda: os.getenv("ETH_RPC_URL", "")
    )
    deployer_private_key: str = field(
        default_factory=lambda: os.getenv(
            "DEPLOYER_PRIVATE_KEY",
            "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        )
    )
    hook_address: str = field(
        default_factory=lambda: os.getenv("HOOK_ADDRESS", "")
    )
    pool_manager_address: str = field(
        default_factory=lambda: os.getenv("POOL_MANAGER_ADDRESS", "")
    )

    # ── Training ──
    seed: int = field(
        default_factory=lambda: int(os.getenv("SEED", "42"))
    )
    n_envs: int = 16
    total_timesteps: int = 10_000_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    policy_kwargs: dict = field(default_factory=lambda: {
        "net_arch": [128, 128],
    })

    # ── Environment ──
    episode_length: int = 1000
    fee_min_bps: float = 1.0       # 0.01%
    fee_max_bps: float = 200.0     # 2.00%
    fee_step_bps: float = 0.5
    initial_liquidity_usd: float = 1_000_000.0
    price_elasticity: float = -1.5
    elasticity_min: float = -3.0
    elasticity_max: float = -0.5
    domain_randomize: bool = True   # sample ε per episode
    action_delay_blocks: int = 1    # tx lands in N+1

    # ── LVR ──
    lvr_window: int = 100
    external_price_weight: float = 1.0

    # ── Gas ──
    gas_penalty_weight: float = 0.001

    # ── EMA (matching on-chain) ──
    ema_alpha: float = 0.1

    # ── Data Pipeline ──
    data_symbol: str = "ETHUSDT"
    data_interval: str = "12s"                # block-level aggregation
    block_time_seconds: int = 12              # Ethereum L1
    pool_address: str = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"  # Uni V3 ETH/USDC 0.05%
    n_synthetic: int = 1_000_000
    synthetic_volatility: float = 0.015       # calibrated to real ETH vol

    # ── Temporal Split (mid-2023 onwards — current MEV ecosystem) ──
    train_start: str = "2023-07-01"
    train_end: str = "2025-03-31"
    val_start: str = "2025-04-01"
    val_end: str = "2025-09-30"
    test_start: str = "2025-10-01"
    test_end: str = "2026-01-31"

    vol_windows: list = field(default_factory=lambda: [10, 50, 100, 500])
    ofi_windows: list = field(default_factory=lambda: [10, 50])
    obs_dim: int = 7  # 3 market + 4 runtime features

    # ── Curriculum Learning ──
    # Phase transitions at fraction of total_timesteps
    curriculum_phase2_frac: float = 0.30
    curriculum_phase3_frac: float = 0.70

    def __post_init__(self):
        for d in [self.data_raw_dir, self.data_processed_dir,
                  self.model_save_dir, self.tensorboard_log_dir]:
            d.mkdir(parents=True, exist_ok=True)


# Singleton config instance
config = Config()
