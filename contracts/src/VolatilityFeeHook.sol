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

contract VolatilityFeeHook is BaseHook {
    using PoolIdLibrary for PoolKey;
    using StateLibrary for IPoolManager;
    using LPFeeLibrary for uint24;

    uint24 public constant MIN_FEE = 100;

    uint24 public constant MAX_FEE = 20_000;

    uint24 public constant DEFAULT_FEE = 3_000;

    uint24 public constant EMERGENCY_FEE = 5_000;

    uint256 public constant EMA_ALPHA = 1e17;

    uint256 public constant EMA_PRECISION = 1e18;

    uint256 public constant MAX_STALENESS_BLOCKS = 50;

    uint256 public constant SLIPPAGE_TAX_K_BPS = 500;

    uint256 public constant MAX_HOOK_GAS = 80_000;

    address public owner;

    address public operator;

    bool public emergencyMode;

    struct PoolFeeState {
        uint24  currentFee;
        uint24  agentTargetFee;
        uint256 lastAgentUpdate;
        uint256 totalSwapCount;
        int256  emaPriceDelta;
        int24   lastTick;
        bool    initialized;
    }

    mapping(PoolId => PoolFeeState) public poolFeeStates;

    event FeeUpdatedByAgent(PoolId indexed poolId, uint24 oldFee, uint24 newFee, uint256 blockNumber);
    event FeeAppliedOnSwap(PoolId indexed poolId, uint24 fee, uint256 swapCount);
    event EmergencyModeToggled(bool active);
    event OperatorChanged(address indexed oldOperator, address indexed newOperator);
    event PoolInitializedByHook(PoolId indexed poolId, uint24 initialFee);
    event EMAUpdated(PoolId indexed poolId, int256 newEma, int24 tickDelta);
    event SlippageTaxApplied(PoolId indexed poolId, uint24 baseFee, uint24 taxedFee, uint256 staleBlocks);

    error OnlyOwner();
    error OnlyOperator();
    error FeeOutOfBounds(uint24 fee);
    error EmergencyActive();
    error PoolNotInitialized();
    error ZeroAddress();

    constructor(
        IPoolManager _poolManager,
        address _operator
    ) BaseHook(_poolManager) {
        if (_operator == address(0)) revert ZeroAddress();
        owner = msg.sender;
        operator = _operator;
    }

    modifier onlyOwner() {
        if (msg.sender != owner) revert OnlyOwner();
        _;
    }

    modifier onlyOperator() {
        if (msg.sender != operator) revert OnlyOperator();
        _;
    }

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
            feeToApply = EMERGENCY_FEE;
        } else if (!state.initialized) {
            feeToApply = DEFAULT_FEE;
        } else {
            uint256 staleness = block.number - state.lastAgentUpdate;

            if (staleness <= MAX_STALENESS_BLOCKS) {
                feeToApply = state.agentTargetFee;
            } else {
                uint256 staleExcess = staleness - MAX_STALENESS_BLOCKS;
                feeToApply = _computeSlippageTax(state.agentTargetFee, staleExcess);
                emit SlippageTaxApplied(poolId, state.agentTargetFee, feeToApply, staleExcess);
            }
        }

        feeToApply = _clampFee(feeToApply);
        state.currentFee = feeToApply;
        state.totalSwapCount++;

        emit FeeAppliedOnSwap(poolId, feeToApply, state.totalSwapCount);

        return (
            this.beforeSwap.selector,
            BeforeSwapDeltaLibrary.ZERO_DELTA,
            feeToApply | LPFeeLibrary.OVERRIDE_FEE_FLAG
        );
    }

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
            (, int24 currentTick,,) = poolManager.getSlot0(poolId);

            if (state.lastTick != 0) {
                int256 tickDelta = int256(currentTick) - int256(state.lastTick);
                int256 absTickDelta = tickDelta >= 0 ? tickDelta : -tickDelta;

                state.emaPriceDelta = int256(
                    (EMA_ALPHA * uint256(absTickDelta) + (EMA_PRECISION - EMA_ALPHA) * uint256(state.emaPriceDelta >= 0 ? state.emaPriceDelta : -state.emaPriceDelta))
                ) / int256(EMA_PRECISION);

                emit EMAUpdated(poolId, state.emaPriceDelta, int24(tickDelta));
            }

            state.lastTick = currentTick;
        }

        return (this.afterSwap.selector, 0);
    }

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

    function getCurrentFee(PoolKey calldata key) external view returns (uint24) {
        PoolId poolId = key.toId();
        return poolFeeStates[poolId].currentFee;
    }

    function getEMAVolatility(PoolKey calldata key) external view returns (int256) {
        PoolId poolId = key.toId();
        return poolFeeStates[poolId].emaPriceDelta;
    }

    function setEmergencyMode(bool _active) external onlyOwner {
        emergencyMode = _active;
        emit EmergencyModeToggled(_active);
    }

    function setOperator(address _newOperator) external onlyOwner {
        if (_newOperator == address(0)) revert ZeroAddress();
        address old = operator;
        operator = _newOperator;
        emit OperatorChanged(old, _newOperator);
    }

    function transferOwnership(address _newOwner) external onlyOwner {
        if (_newOwner == address(0)) revert ZeroAddress();
        owner = _newOwner;
    }

    function _computeSlippageTax(
        uint24 baseFee,
        uint256 staleExcess
    ) internal pure returns (uint24) {
        uint256 fee = uint256(baseFee) * 1e4;
        uint256 multiplier = 1e4 + SLIPPAGE_TAX_K_BPS;

        uint256 iters = staleExcess > 200 ? 200 : staleExcess;

        for (uint256 i = 0; i < iters; i++) {
            fee = (fee * multiplier) / 1e4;
            if (fee >= uint256(MAX_FEE) * 1e4) {
                return MAX_FEE;
            }
        }

        uint24 result = uint24(fee / 1e4);
        if (result > MAX_FEE) return MAX_FEE;
        if (result < MIN_FEE) return MIN_FEE;
        return result;
    }

    function _clampFee(uint24 fee) internal pure returns (uint24) {
        if (fee < MIN_FEE) return MIN_FEE;
        if (fee > MAX_FEE) return MAX_FEE;
        return fee;
    }
}
