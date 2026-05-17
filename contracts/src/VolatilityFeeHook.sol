// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {BaseHook} from "@openzeppelin/uniswap-hooks/src/base/BaseHook.sol";
import {Hooks} from "@uniswap/v4-core/src/libraries/Hooks.sol";
import {IPoolManager} from "@uniswap/v4-core/src/interfaces/IPoolManager.sol";
import {PoolKey} from "@uniswap/v4-core/src/types/PoolKey.sol";
import {PoolId, PoolIdLibrary} from "@uniswap/v4-core/src/types/PoolId.sol";
import {BalanceDelta} from "@uniswap/v4-core/src/types/BalanceDelta.sol";
import {BeforeSwapDelta, BeforeSwapDeltaLibrary} from "@uniswap/v4-core/src/types/BeforeSwapDelta.sol";
import {SwapParams} from "@uniswap/v4-core/src/types/PoolOperation.sol";
import {LPFeeLibrary} from "@uniswap/v4-core/src/libraries/LPFeeLibrary.sol";
import {StateLibrary} from "@uniswap/v4-core/src/libraries/StateLibrary.sol";

/// @title VolatilityFeeHook
/// @notice Uniswap V4 Hook that dynamically adjusts LP fees based on market volatility.
///         An off-chain RL agent periodically updates fee parameters via authorized calls.
///         The Hook applies those parameters to every swap via beforeSwap.
/// @dev    Security model:
///         - Only `operator` (the RL bridge) can update fee parameters.
///         - Owner can change operator and set emergency mode.
///         - Fee is bounded by [MIN_FEE, MAX_FEE] — agent cannot set absurd values.
///         - Emergency mode freezes fee to EMERGENCY_FEE (safe fallback).
///         - Slippage Tax: when agent signal is stale, fee escalates exponentially
///           fee = agentTargetFee * (1 + SLIPPAGE_TAX_K/1e4)^(staleness - MAX_STALENESS_BLOCKS)
///           This makes arbitrage unprofitable while still allowing retail trades (expensive).
///         - On-chain EMA oracle provides secondary volatility signal.
contract VolatilityFeeHook is BaseHook {
    using PoolIdLibrary for PoolKey;
    using StateLibrary for IPoolManager;
    using LPFeeLibrary for uint24;

    // ══════════════════════════════════════════════════════════════════════
    //                           CONSTANTS
    // ══════════════════════════════════════════════════════════════════════

    /// @notice Minimum fee the agent can set (1 basis point = 0.01%)
    uint24 public constant MIN_FEE = 100;       // 0.01% in hundredths of a bip

    /// @notice Maximum fee the agent can set (200 basis points = 2.00%)
    uint24 public constant MAX_FEE = 20_000;    // 2.00%

    /// @notice Default fee if agent has never updated (30 basis points = 0.30%)
    uint24 public constant DEFAULT_FEE = 3_000; // 0.30%

    /// @notice Emergency mode fee (50 basis points = 0.50%)
    uint24 public constant EMERGENCY_FEE = 5_000;

    /// @notice EMA smoothing factor (alpha), scaled by 1e18. alpha=0.1 → fast reaction
    uint256 public constant EMA_ALPHA = 1e17; // 0.1 * 1e18

    /// @notice EMA scaling precision
    uint256 public constant EMA_PRECISION = 1e18;

    /// @notice Maximum staleness: if agent hasn't updated in this many blocks, apply slippage tax
    uint256 public constant MAX_STALENESS_BLOCKS = 50;

    /// @notice Slippage tax rate per stale block, in basis points of basis points (1e4 precision).
    ///         500 = 5% compounding per stale block → fee ≈ doubles every ~14 blocks past threshold.
    ///         Formula: fee = agentTargetFee * (1 + SLIPPAGE_TAX_K_BPS / 1e4) ^ staleExcess
    uint256 public constant SLIPPAGE_TAX_K_BPS = 500;

    /// @notice Maximum gas budget for beforeSwap logic (excluding base cost)
    uint256 public constant MAX_HOOK_GAS = 80_000;

    // ══════════════════════════════════════════════════════════════════════
    //                           STATE
    // ══════════════════════════════════════════════════════════════════════

    /// @notice Owner of the Hook (can set operator, toggle emergency mode)
    address public owner;

    /// @notice Authorized operator (the RL bridge wallet)
    address public operator;

    /// @notice Whether the Hook is in emergency mode (frozen fee)
    bool public emergencyMode;

    /// @notice Per-pool dynamic fee set by the RL agent
    struct PoolFeeState {
        uint24  currentFee;         // Current fee applied to swaps
        uint24  agentTargetFee;     // Fee target communicated by RL agent
        uint256 lastAgentUpdate;    // Block number of last agent update
        uint256 totalSwapCount;     // # swaps processed (for stats)
        int256  emaPriceDelta;      // EMA of price deltas (volatility proxy), scaled 1e18
        int24   lastTick;           // Last observed tick (for delta calculation)
        bool    initialized;        // Whether pool was seen before
    }

    /// @notice Pool ID → fee state mapping
    mapping(PoolId => PoolFeeState) public poolFeeStates;

    // ══════════════════════════════════════════════════════════════════════
    //                           EVENTS
    // ══════════════════════════════════════════════════════════════════════

    event FeeUpdatedByAgent(PoolId indexed poolId, uint24 oldFee, uint24 newFee, uint256 blockNumber);
    event FeeAppliedOnSwap(PoolId indexed poolId, uint24 fee, uint256 swapCount);
    event EmergencyModeToggled(bool active);
    event OperatorChanged(address indexed oldOperator, address indexed newOperator);
    event PoolInitializedByHook(PoolId indexed poolId, uint24 initialFee);
    event EMAUpdated(PoolId indexed poolId, int256 newEma, int24 tickDelta);
    event SlippageTaxApplied(PoolId indexed poolId, uint24 baseFee, uint24 taxedFee, uint256 staleBlocks);

    // ══════════════════════════════════════════════════════════════════════
    //                           ERRORS
    // ══════════════════════════════════════════════════════════════════════

    error OnlyOwner();
    error OnlyOperator();
    error FeeOutOfBounds(uint24 fee);
    error EmergencyActive();
    error PoolNotInitialized();
    error ZeroAddress();

    // ══════════════════════════════════════════════════════════════════════
    //                         CONSTRUCTOR
    // ══════════════════════════════════════════════════════════════════════

    constructor(
        IPoolManager _poolManager,
        address _operator
    ) BaseHook(_poolManager) {
        if (_operator == address(0)) revert ZeroAddress();
        owner = msg.sender;
        operator = _operator;
    }

    // ══════════════════════════════════════════════════════════════════════
    //                         MODIFIERS
    // ══════════════════════════════════════════════════════════════════════

    modifier onlyOwner() {
        if (msg.sender != owner) revert OnlyOwner();
        _;
    }

    modifier onlyOperator() {
        if (msg.sender != operator) revert OnlyOperator();
        _;
    }

    // ══════════════════════════════════════════════════════════════════════
    //                   HOOK PERMISSION FLAGS
    // ══════════════════════════════════════════════════════════════════════

    function getHookPermissions() public pure override returns (Hooks.Permissions memory) {
        return Hooks.Permissions({
            beforeInitialize: true,
            afterInitialize: false,
            beforeAddLiquidity: false,
            afterAddLiquidity: false,
            beforeRemoveLiquidity: false,
            afterRemoveLiquidity: false,
            beforeSwap: true,
            afterSwap: true,
            beforeDonate: false,
            afterDonate: false,
            beforeSwapReturnDelta: false,
            afterSwapReturnDelta: false,
            afterAddLiquidityReturnDelta: false,
            afterRemoveLiquidityReturnDelta: false
        });
    }

    // ══════════════════════════════════════════════════════════════════════
    //                   HOOK CALLBACKS
    // ══════════════════════════════════════════════════════════════════════

    /// @notice Called when a pool using this Hook is initialized.
    ///         Sets up default fee state.
    function _beforeInitialize(
        address,
        PoolKey calldata key,
        uint160
    ) internal override returns (bytes4) {
        PoolId poolId = key.toId();

        poolFeeStates[poolId] = PoolFeeState({
            currentFee: DEFAULT_FEE,
            agentTargetFee: DEFAULT_FEE,
            lastAgentUpdate: block.number,
            totalSwapCount: 0,
            emaPriceDelta: 0,
            lastTick: 0,
            initialized: true
        });

        emit PoolInitializedByHook(poolId, DEFAULT_FEE);
        return this.beforeInitialize.selector;
    }

    /// @notice Core logic: determines fee for this swap.
    /// @dev    Gas-optimized: minimal branching, no external calls.
    ///         Decision tree:
    ///         1. Emergency mode → EMERGENCY_FEE
    ///         2. Agent update is fresh (< MAX_STALENESS_BLOCKS) → agent fee
    ///         3. Agent update is stale → Slippage Tax: exponential fee escalation
    ///            fee = agentTargetFee * (1 + k)^(staleness - MAX_STALENESS_BLOCKS)
    ///            "Profesjonalny giełdziarz podnosi cenę, amator zamyka sklep."
    function _beforeSwap(
        address,
        PoolKey calldata key,
        SwapParams calldata,
        bytes calldata
    ) internal override returns (bytes4, BeforeSwapDelta, uint24) {
        PoolId poolId = key.toId();
        PoolFeeState storage state = poolFeeStates[poolId];

        uint24 feeToApply;

        if (emergencyMode) {
            // ─── Emergency: frozen safe fee ───
            feeToApply = EMERGENCY_FEE;
        } else if (!state.initialized) {
            // ─── Unknown pool: default fee ───
            feeToApply = DEFAULT_FEE;
        } else {
            uint256 staleness = block.number - state.lastAgentUpdate;

            if (staleness <= MAX_STALENESS_BLOCKS) {
                // ─── Fresh agent signal: trust the RL agent ───
                feeToApply = state.agentTargetFee;
            } else {
                // ─── Stale agent: Slippage Tax (exponential fee escalation) ───
                uint256 staleExcess = staleness - MAX_STALENESS_BLOCKS;
                feeToApply = _computeSlippageTax(state.agentTargetFee, staleExcess);
                emit SlippageTaxApplied(poolId, state.agentTargetFee, feeToApply, staleExcess);
            }
        }

        // Clamp fee within safety bounds
        feeToApply = _clampFee(feeToApply);
        state.currentFee = feeToApply;
        state.totalSwapCount++;

        emit FeeAppliedOnSwap(poolId, feeToApply, state.totalSwapCount);

        // Return the fee override using LPFeeLibrary
        // The fee is returned as the third return value with the OVERRIDE flag
        return (
            this.beforeSwap.selector,
            BeforeSwapDeltaLibrary.ZERO_DELTA,
            feeToApply | LPFeeLibrary.OVERRIDE_FEE_FLAG
        );
    }

    /// @notice After each swap, update the on-chain EMA volatility oracle.
    /// @dev    This is lightweight: just a tick delta → EMA update.
    function _afterSwap(
        address,
        PoolKey calldata key,
        SwapParams calldata,
        BalanceDelta,
        bytes calldata
    ) internal override returns (bytes4, int128) {
        PoolId poolId = key.toId();
        PoolFeeState storage state = poolFeeStates[poolId];

        if (state.initialized) {
            // Read current tick from pool
            (, int24 currentTick,,) = poolManager.getSlot0(poolId);
            
            if (state.lastTick != 0) {
                int256 tickDelta = int256(currentTick) - int256(state.lastTick);
                // Absolute tick delta as volatility proxy
                int256 absTickDelta = tickDelta >= 0 ? tickDelta : -tickDelta;

                // EMA update: ema = alpha * absTickDelta + (1 - alpha) * ema
                state.emaPriceDelta = int256(
                    (EMA_ALPHA * uint256(absTickDelta) + (EMA_PRECISION - EMA_ALPHA) * uint256(state.emaPriceDelta >= 0 ? state.emaPriceDelta : -state.emaPriceDelta))
                ) / int256(EMA_PRECISION);

                emit EMAUpdated(poolId, state.emaPriceDelta, int24(tickDelta));
            }

            state.lastTick = currentTick;
        }

        return (this.afterSwap.selector, 0);
    }

    // ══════════════════════════════════════════════════════════════════════
    //         AGENT INTERFACE (Called by RL Bridge / Operator)
    // ══════════════════════════════════════════════════════════════════════

    /// @notice RL agent updates the target fee for a specific pool.
    /// @param key The pool key identifying the pool.
    /// @param newFee The new fee in hundredths of a basis point.
    function updateFee(PoolKey calldata key, uint24 newFee) external onlyOperator {
        if (emergencyMode) revert EmergencyActive();
        if (newFee < MIN_FEE || newFee > MAX_FEE) revert FeeOutOfBounds(newFee);

        PoolId poolId = key.toId();
        PoolFeeState storage state = poolFeeStates[poolId];
        if (!state.initialized) revert PoolNotInitialized();

        uint24 oldFee = state.agentTargetFee;
        state.agentTargetFee = newFee;
        state.lastAgentUpdate = block.number;

        emit FeeUpdatedByAgent(poolId, oldFee, newFee, block.number);
    }

    /// @notice Batch update fees for multiple pools in one transaction (gas efficient).
    /// @param keys Array of pool keys.
    /// @param newFees Array of corresponding fees.
    function batchUpdateFees(PoolKey[] calldata keys, uint24[] calldata newFees) external onlyOperator {
        if (emergencyMode) revert EmergencyActive();
        require(keys.length == newFees.length, "Length mismatch");

        for (uint256 i = 0; i < keys.length; i++) {
            uint24 fee = newFees[i];
            if (fee < MIN_FEE || fee > MAX_FEE) revert FeeOutOfBounds(fee);

            PoolId poolId = keys[i].toId();
            PoolFeeState storage state = poolFeeStates[poolId];
            if (!state.initialized) revert PoolNotInitialized();

            uint24 oldFee = state.agentTargetFee;
            state.agentTargetFee = fee;
            state.lastAgentUpdate = block.number;

            emit FeeUpdatedByAgent(poolId, oldFee, fee, block.number);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    //                    VIEW FUNCTIONS (For Bridge)
    // ══════════════════════════════════════════════════════════════════════

    /// @notice Returns complete fee state for a pool (used by Python bridge).
    function getPoolState(PoolKey calldata key) external view returns (
        uint24 currentFee,
        uint24 agentTargetFee,
        uint256 lastAgentUpdate,
        uint256 totalSwapCount,
        int256 emaPriceDelta,
        int24 lastTick,
        bool initialized
    ) {
        PoolId poolId = key.toId();
        PoolFeeState storage state = poolFeeStates[poolId];
        return (
            state.currentFee,
            state.agentTargetFee,
            state.lastAgentUpdate,
            state.totalSwapCount,
            state.emaPriceDelta,
            state.lastTick,
            state.initialized
        );
    }

    /// @notice Returns just the current fee for a pool.
    function getCurrentFee(PoolKey calldata key) external view returns (uint24) {
        PoolId poolId = key.toId();
        return poolFeeStates[poolId].currentFee;
    }

    /// @notice Returns the on-chain EMA volatility estimate.
    function getEMAVolatility(PoolKey calldata key) external view returns (int256) {
        PoolId poolId = key.toId();
        return poolFeeStates[poolId].emaPriceDelta;
    }

    // ══════════════════════════════════════════════════════════════════════
    //                      ADMIN FUNCTIONS
    // ══════════════════════════════════════════════════════════════════════

    /// @notice Toggle emergency mode. When active, all pools use EMERGENCY_FEE.
    function setEmergencyMode(bool _active) external onlyOwner {
        emergencyMode = _active;
        emit EmergencyModeToggled(_active);
    }

    /// @notice Change the authorized operator (RL bridge wallet).
    function setOperator(address _newOperator) external onlyOwner {
        if (_newOperator == address(0)) revert ZeroAddress();
        address old = operator;
        operator = _newOperator;
        emit OperatorChanged(old, _newOperator);
    }

    /// @notice Transfer ownership.
    function transferOwnership(address _newOwner) external onlyOwner {
        if (_newOwner == address(0)) revert ZeroAddress();
        owner = _newOwner;
    }

    // ══════════════════════════════════════════════════════════════════════
    //                      INTERNAL HELPERS
    // ══════════════════════════════════════════════════════════════════════

    /// @dev Computes Slippage Tax: exponential fee escalation when agent is stale.
    ///      fee = agentTargetFee * (1 + SLIPPAGE_TAX_K_BPS / 1e4) ^ staleExcess
    ///      Implemented as iterative multiplication for gas efficiency and precision.
    ///      Capped at MAX_FEE to prevent overflow; early exit on saturation.
    ///
    ///      Example with K=500 (5% per block):
    ///        staleExcess=1:  fee * 1.05
    ///        staleExcess=14: fee * ~2.0  (doubles)
    ///        staleExcess=50: fee * ~11.5 (arb impossible)
    function _computeSlippageTax(
        uint24 baseFee,
        uint256 staleExcess
    ) internal pure returns (uint24) {
        // Start with base fee scaled up by 1e4 for precision
        uint256 fee = uint256(baseFee) * 1e4;
        uint256 multiplier = 1e4 + SLIPPAGE_TAX_K_BPS; // e.g. 10500 for 5%

        // Cap iterations to prevent excessive gas (staleExcess > 200 already hits MAX_FEE)
        uint256 iters = staleExcess > 200 ? 200 : staleExcess;

        for (uint256 i = 0; i < iters; i++) {
            fee = (fee * multiplier) / 1e4;
            // Early exit if already at max
            if (fee >= uint256(MAX_FEE) * 1e4) {
                return MAX_FEE;
            }
        }

        uint24 result = uint24(fee / 1e4);
        if (result > MAX_FEE) return MAX_FEE;
        if (result < MIN_FEE) return MIN_FEE;
        return result;
    }

    /// @dev Clamps fee within [MIN_FEE, MAX_FEE].
    function _clampFee(uint24 fee) internal pure returns (uint24) {
        if (fee < MIN_FEE) return MIN_FEE;
        if (fee > MAX_FEE) return MAX_FEE;
        return fee;
    }
}
