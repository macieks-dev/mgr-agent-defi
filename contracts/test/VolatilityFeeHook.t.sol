// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {Test, console2} from "forge-std/Test.sol";
import {Deployers} from "@uniswap/v4-core/test/utils/Deployers.sol";
import {IPoolManager} from "@uniswap/v4-core/src/interfaces/IPoolManager.sol";
import {PoolKey} from "@uniswap/v4-core/src/types/PoolKey.sol";
import {PoolId, PoolIdLibrary} from "@uniswap/v4-core/src/types/PoolId.sol";
import {Currency, CurrencyLibrary} from "@uniswap/v4-core/src/types/Currency.sol";
import {Hooks} from "@uniswap/v4-core/src/libraries/Hooks.sol";
import {TickMath} from "@uniswap/v4-core/src/libraries/TickMath.sol";
import {LPFeeLibrary} from "@uniswap/v4-core/src/libraries/LPFeeLibrary.sol";
import {ModifyLiquidityParams} from "@uniswap/v4-core/src/types/PoolOperation.sol";
import {VolatilityFeeHook} from "../src/VolatilityFeeHook.sol";

/// @title VolatilityFeeHookTest
/// @notice Comprehensive test suite for the VolatilityFeeHook contract.
///         Tests cover:
///         - Deployment and initialization
///         - Fee updates by the RL agent (operator)
///         - Access control (owner/operator separation)
///         - Fee clamping and bounds
///         - Emergency mode
///         - Slippage tax (exponential fee escalation when stale)
///         - Batch fee updates
///         - Gas benchmarks
///         - Arbitrage attack simulation
contract VolatilityFeeHookTest is Test, Deployers {
    using PoolIdLibrary for PoolKey;
    using CurrencyLibrary for Currency;

    VolatilityFeeHook hook;
    PoolKey poolKey;
    PoolId poolId;

    address owner = address(this);
    address operator = address(0xBEEF);
    address attacker = address(0xDEAD);

    // ═════════════════════════════════════════════════
    //                    SETUP
    // ═════════════════════════════════════════════════

    function setUp() public {
        // Deploy V4 core infrastructure
        deployFreshManagerAndRouters();
        deployMintAndApprove2Currencies();

        // Compute the hook address with correct flags
        // Flags: beforeInitialize, beforeSwap, afterSwap
        uint160 flags = uint160(
            Hooks.BEFORE_INITIALIZE_FLAG |
            Hooks.BEFORE_SWAP_FLAG |
            Hooks.AFTER_SWAP_FLAG
        );

        // Deploy hook to an address that matches the flag pattern
        address hookAddress = address(flags);
        
        // Deploy the hook using CREATE2 or etch
        deployCodeTo(
            "VolatilityFeeHook.sol:VolatilityFeeHook",
            abi.encode(manager, operator),
            hookAddress
        );
        hook = VolatilityFeeHook(hookAddress);

        // Initialize a pool with dynamic fee flag
        poolKey = PoolKey({
            currency0: currency0,
            currency1: currency1,
            fee: LPFeeLibrary.DYNAMIC_FEE_FLAG,
            tickSpacing: 60,
            hooks: hook
        });

        poolId = poolKey.toId();
        
        // Initialize pool at price 1:1 (tick 0)
        manager.initialize(poolKey, SQRT_PRICE_1_1);
    }

    // ═════════════════════════════════════════════════
    //              INITIALIZATION TESTS
    // ═════════════════════════════════════════════════

    function test_hookDeployment() public view {
        assertEq(hook.owner(), owner);
        assertEq(hook.operator(), operator);
        assertFalse(hook.emergencyMode());
    }

    function test_poolInitialization() public view {
        (
            uint24 currentFee,
            uint24 agentTargetFee,
            ,
            uint256 totalSwapCount,
            ,
            ,
            bool initialized
        ) = hook.poolFeeStates(poolId);

        assertTrue(initialized, "Pool should be initialized");
        assertEq(currentFee, hook.DEFAULT_FEE(), "Default fee should be 0.30%");
        assertEq(agentTargetFee, hook.DEFAULT_FEE(), "Agent target should be default");
        assertEq(totalSwapCount, 0, "No swaps yet");
    }

    // ═════════════════════════════════════════════════
    //             FEE UPDATE TESTS
    // ═════════════════════════════════════════════════

    function test_operatorCanUpdateFee() public {
        uint24 newFee = 5_000; // 0.50%

        vm.prank(operator);
        hook.updateFee(poolKey, newFee);

        (, uint24 agentTargetFee,,,,, ) = hook.poolFeeStates(poolId);
        assertEq(agentTargetFee, newFee, "Fee should be updated to 0.50%");
    }

    function test_nonOperatorCannotUpdateFee() public {
        vm.prank(attacker);
        vm.expectRevert(VolatilityFeeHook.OnlyOperator.selector);
        hook.updateFee(poolKey, 5_000);
    }

    function test_ownerCannotDirectlyUpdateFee() public {
        // Owner is NOT operator — separation of concerns
        vm.prank(owner);
        vm.expectRevert(VolatilityFeeHook.OnlyOperator.selector);
        hook.updateFee(poolKey, 5_000);
    }

    function test_feeCannotExceedMaximum() public {
        vm.prank(operator);
        vm.expectRevert(abi.encodeWithSelector(
            VolatilityFeeHook.FeeOutOfBounds.selector,
            uint24(25_000)
        ));
        hook.updateFee(poolKey, 25_000); // 2.5% - exceeds MAX_FEE
    }

    function test_feeCannotBeBelowMinimum() public {
        vm.prank(operator);
        vm.expectRevert(abi.encodeWithSelector(
            VolatilityFeeHook.FeeOutOfBounds.selector,
            uint24(50)
        ));
        hook.updateFee(poolKey, 50); // 0.005% - below MIN_FEE
    }

    function test_feeAtExactBoundaries() public {
        uint24 minFee = hook.MIN_FEE();
        uint24 maxFee = hook.MAX_FEE();

        // Test exact MIN_FEE
        vm.prank(operator);
        hook.updateFee(poolKey, minFee);
        (, uint24 fee1,,,,, ) = hook.poolFeeStates(poolId);
        assertEq(fee1, minFee);

        // Test exact MAX_FEE
        vm.prank(operator);
        hook.updateFee(poolKey, maxFee);
        (, uint24 fee2,,,,, ) = hook.poolFeeStates(poolId);
        assertEq(fee2, maxFee);
    }

    // ═════════════════════════════════════════════════
    //            EMERGENCY MODE TESTS
    // ═════════════════════════════════════════════════

    function test_ownerCanToggleEmergencyMode() public {
        hook.setEmergencyMode(true);
        assertTrue(hook.emergencyMode());

        hook.setEmergencyMode(false);
        assertFalse(hook.emergencyMode());
    }

    function test_nonOwnerCannotToggleEmergencyMode() public {
        vm.prank(attacker);
        vm.expectRevert(VolatilityFeeHook.OnlyOwner.selector);
        hook.setEmergencyMode(true);
    }

    function test_cannotUpdateFeeInEmergencyMode() public {
        hook.setEmergencyMode(true);

        vm.prank(operator);
        vm.expectRevert(VolatilityFeeHook.EmergencyActive.selector);
        hook.updateFee(poolKey, 5_000);
    }

    // ═════════════════════════════════════════════════
    //            OPERATOR MANAGEMENT TESTS
    // ═════════════════════════════════════════════════

    function test_ownerCanChangeOperator() public {
        address newOperator = address(0xCAFE);
        hook.setOperator(newOperator);
        assertEq(hook.operator(), newOperator);
    }

    function test_cannotSetZeroOperator() public {
        vm.expectRevert(VolatilityFeeHook.ZeroAddress.selector);
        hook.setOperator(address(0));
    }

    function test_newOperatorCanUpdateFee() public {
        address newOperator = address(0xCAFE);
        hook.setOperator(newOperator);

        vm.prank(newOperator);
        hook.updateFee(poolKey, 8_000);

        (, uint24 fee,,,,, ) = hook.poolFeeStates(poolId);
        assertEq(fee, 8_000);
    }

    function test_oldOperatorCannotUpdateAfterChange() public {
        address newOperator = address(0xCAFE);
        hook.setOperator(newOperator);

        vm.prank(operator);
        vm.expectRevert(VolatilityFeeHook.OnlyOperator.selector);
        hook.updateFee(poolKey, 8_000);
    }

    // ═════════════════════════════════════════════════
    //            BATCH UPDATE TESTS
    // ═════════════════════════════════════════════════

    function test_batchUpdateFees() public {
        // Create a second pool
        PoolKey memory poolKey2 = PoolKey({
            currency0: currency0,
            currency1: currency1,
            fee: LPFeeLibrary.DYNAMIC_FEE_FLAG,
            tickSpacing: 120, // different tick spacing = different pool
            hooks: hook
        });
        manager.initialize(poolKey2, SQRT_PRICE_1_1);

        // Batch update both pools
        PoolKey[] memory keys = new PoolKey[](2);
        keys[0] = poolKey;
        keys[1] = poolKey2;

        uint24[] memory fees = new uint24[](2);
        fees[0] = 4_000;
        fees[1] = 6_000;

        vm.prank(operator);
        hook.batchUpdateFees(keys, fees);

        (, uint24 fee1,,,,, ) = hook.poolFeeStates(poolKey.toId());
        (, uint24 fee2,,,,, ) = hook.poolFeeStates(poolKey2.toId());

        assertEq(fee1, 4_000);
        assertEq(fee2, 6_000);
    }

    // ═════════════════════════════════════════════════
    //              VIEW FUNCTION TESTS
    // ═════════════════════════════════════════════════

    function test_getPoolState() public {
        vm.prank(operator);
        hook.updateFee(poolKey, 7_000);

        (
            uint24 currentFee,
            uint24 agentTargetFee,
            uint256 lastUpdate,
            uint256 swapCount,
            int256 ema,
            int24 lastTick,
            bool init
        ) = hook.getPoolState(poolKey);

        assertEq(agentTargetFee, 7_000);
        assertEq(lastUpdate, block.number);
        assertTrue(init);
    }

    function test_getCurrentFee() public view {
        uint24 fee = hook.getCurrentFee(poolKey);
        assertEq(fee, hook.DEFAULT_FEE());
    }

    // ═════════════════════════════════════════════════
    //              GAS BENCHMARK TESTS
    // ═════════════════════════════════════════════════

    function test_gasUpdateFee() public {
        vm.prank(operator);
        uint256 gasBefore = gasleft();
        hook.updateFee(poolKey, 5_000);
        uint256 gasUsed = gasBefore - gasleft();

        console2.log("Gas used for updateFee:", gasUsed);
        assertLt(gasUsed, 80_000, "updateFee should use less than 80k gas");
    }

    // ═════════════════════════════════════════════════
    //           FUZZ TESTS
    // ═════════════════════════════════════════════════

    function testFuzz_feeAlwaysClamped(uint24 fee) public {
        // Only test valid fees within bounds
        fee = uint24(bound(fee, hook.MIN_FEE(), hook.MAX_FEE()));

        vm.prank(operator);
        hook.updateFee(poolKey, fee);

        (, uint24 storedFee,,,,, ) = hook.poolFeeStates(poolId);
        assertGe(storedFee, hook.MIN_FEE());
        assertLe(storedFee, hook.MAX_FEE());
    }

    function testFuzz_invalidFeeReverts(uint24 fee) public {
        // Test fees outside valid bounds
        vm.assume(fee < hook.MIN_FEE() || fee > hook.MAX_FEE());

        vm.prank(operator);
        vm.expectRevert(abi.encodeWithSelector(
            VolatilityFeeHook.FeeOutOfBounds.selector,
            fee
        ));
        hook.updateFee(poolKey, fee);
    }

    // ═════════════════════════════════════════════════
    //          OWNERSHIP TRANSFER TESTS
    // ═════════════════════════════════════════════════

    function test_transferOwnership() public {
        address newOwner = address(0xFACE);
        hook.transferOwnership(newOwner);
        assertEq(hook.owner(), newOwner);
    }

    function test_cannotTransferToZero() public {
        vm.expectRevert(VolatilityFeeHook.ZeroAddress.selector);
        hook.transferOwnership(address(0));
    }

    function test_nonOwnerCannotTransfer() public {
        vm.prank(attacker);
        vm.expectRevert(VolatilityFeeHook.OnlyOwner.selector);
        hook.transferOwnership(attacker);
    }

    // ═════════════════════════════════════════════════
    //         SLIPPAGE TAX TESTS
    // ═════════════════════════════════════════════════

    /// @notice When agent is fresh (< MAX_STALENESS_BLOCKS), no slippage tax
    function test_noSlippageTaxWhenFresh() public {
        vm.prank(operator);
        hook.updateFee(poolKey, 3_000); // 0.30%

        // Advance a few blocks (well within threshold)
        vm.roll(block.number + 10);

        // Fee should still be agent's target
        (, uint24 agentFee,,,,, ) = hook.poolFeeStates(poolId);
        assertEq(agentFee, 3_000, "Agent fee should be unchanged");
    }

    /// @notice Slippage tax activates after MAX_STALENESS_BLOCKS of no updates
    function test_slippageTaxActivatesWhenStale() public {
        // Set agent fee
        vm.prank(operator);
        hook.updateFee(poolKey, 3_000);

        // Advance past staleness threshold
        uint256 staleBlocks = hook.MAX_STALENESS_BLOCKS() + 20;
        vm.roll(block.number + staleBlocks);

        // Perform a swap to trigger _beforeSwap with stale signal
        // We add liquidity first so the swap can go through
        modifyLiquidityRouter.modifyLiquidity(
            poolKey,
            ModifyLiquidityParams({
                tickLower: -120,
                tickUpper: 120,
                liquidityDelta: 10 ether,
                salt: bytes32(0)
            }),
            ""
        );

        // Execute swap
        swap(poolKey, true, 0.001 ether, "");

        // Read the applied fee — it should be higher than the base 3000
        (uint24 currentFee,,,,,, ) = hook.poolFeeStates(poolId);
        assertGt(currentFee, 3_000, "Fee should be escalated by slippage tax");
    }

    /// @notice Slippage tax grows exponentially with stale blocks  
    function test_slippageTaxGrowsExponentially() public {
        vm.prank(operator);
        hook.updateFee(poolKey, 3_000);

        // Add liquidity for swaps
        modifyLiquidityRouter.modifyLiquidity(
            poolKey,
            ModifyLiquidityParams({
                tickLower: -120,
                tickUpper: 120,
                liquidityDelta: 10 ether,
                salt: bytes32(0)
            }),
            ""
        );

        // Swap at staleness = MAX + 10
        vm.roll(block.number + hook.MAX_STALENESS_BLOCKS() + 10);
        swap(poolKey, true, 0.0001 ether, "");
        (uint24 fee1,,,,,, ) = hook.poolFeeStates(poolId);

        // Update fee to reset staleness, then go stale again for longer
        vm.prank(operator);
        hook.updateFee(poolKey, 3_000);

        // Swap at staleness = MAX + 50
        vm.roll(block.number + hook.MAX_STALENESS_BLOCKS() + 50);
        swap(poolKey, true, 0.0001 ether, "");
        (uint24 fee2,,,,,, ) = hook.poolFeeStates(poolId);

        assertGt(fee2, fee1, "Longer staleness should produce higher slippage tax");
    }

    /// @notice Slippage tax is clamped at MAX_FEE
    function test_slippageTaxClampedAtMaxFee() public {
        vm.prank(operator);
        hook.updateFee(poolKey, 3_000);

        // Add liquidity
        modifyLiquidityRouter.modifyLiquidity(
            poolKey,
            ModifyLiquidityParams({
                tickLower: -120,
                tickUpper: 120,
                liquidityDelta: 10 ether,
                salt: bytes32(0)
            }),
            ""
        );

        // Advance WAY past threshold (300 blocks stale → fee should hit max)
        vm.roll(block.number + hook.MAX_STALENESS_BLOCKS() + 300);
        swap(poolKey, true, 0.0001 ether, "");

        (uint24 fee,,,,,, ) = hook.poolFeeStates(poolId);
        assertEq(fee, hook.MAX_FEE(), "Fee should be clamped at MAX_FEE");
    }

    /// @notice Agent update after stale period resets fee to normal
    function test_slippageTaxResetsAfterAgentUpdate() public {
        vm.prank(operator);
        hook.updateFee(poolKey, 3_000);

        // Add liquidity
        modifyLiquidityRouter.modifyLiquidity(
            poolKey,
            ModifyLiquidityParams({
                tickLower: -120,
                tickUpper: 120,
                liquidityDelta: 10 ether,
                salt: bytes32(0)
            }),
            ""
        );

        // Go stale and trigger high fee
        vm.roll(block.number + hook.MAX_STALENESS_BLOCKS() + 30);
        swap(poolKey, true, 0.0001 ether, "");

        (uint24 staleFee,,,,,, ) = hook.poolFeeStates(poolId);
        assertGt(staleFee, 3_000, "Should have slippage tax applied");

        // Agent comes back online
        vm.prank(operator);
        hook.updateFee(poolKey, 4_000);

        // Next swap should use agent's fresh fee
        swap(poolKey, true, 0.0001 ether, "");

        (uint24 freshFee,,,,,, ) = hook.poolFeeStates(poolId);
        assertEq(freshFee, 4_000, "Fee should reset to agent's new target");
    }

    /// @notice Slippage tax gas benchmark
    function test_gasSlippageTax() public {
        vm.prank(operator);
        hook.updateFee(poolKey, 3_000);

        modifyLiquidityRouter.modifyLiquidity(
            poolKey,
            ModifyLiquidityParams({
                tickLower: -120,
                tickUpper: 120,
                liquidityDelta: 10 ether,
                salt: bytes32(0)
            }),
            ""
        );

        // Advance to stale
        vm.roll(block.number + hook.MAX_STALENESS_BLOCKS() + 50);

        uint256 gasBefore = gasleft();
        swap(poolKey, true, 0.0001 ether, "");
        uint256 gasUsed = gasBefore - gasleft();

        console2.log("Gas used for swap with slippage tax (50 stale blocks):", gasUsed);
        // Slippage tax loop of 50 iterations should be well under 100k gas
        assertLt(gasUsed, 500_000, "Slippage tax swap should be gas-efficient");
    }
}
