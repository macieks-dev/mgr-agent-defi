// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {Script, console2} from "forge-std/Script.sol";
import {IPoolManager} from "@uniswap/v4-core/src/interfaces/IPoolManager.sol";
import {VolatilityFeeHook} from "../src/VolatilityFeeHook.sol";
import {Hooks} from "@uniswap/v4-core/src/libraries/Hooks.sol";
import {HookMiner} from "@uniswap/v4-periphery/src/utils/HookMiner.sol";

contract DeployHook is Script {
    function run() external {
        require(block.chainid == 31337, "DeployHook only supports local Anvil (chain ID 31337)");

        address poolManager = vm.envAddress("POOL_MANAGER_ADDRESS");
        address operator = vm.envAddress("OPERATOR_ADDRESS");
        uint256 deployerKey = vm.envUint("DEPLOYER_PRIVATE_KEY");

        uint160 flags = uint160(
            Hooks.BEFORE_INITIALIZE_FLAG |
            Hooks.BEFORE_SWAP_FLAG |
            Hooks.AFTER_SWAP_FLAG
        );

        (address hookAddress, bytes32 salt) = HookMiner.find(
            CREATE2_FACTORY,
            flags,
            type(VolatilityFeeHook).creationCode,
            abi.encode(poolManager, operator)
        );

        console2.log("Deploying VolatilityFeeHook to:", hookAddress);
        console2.log("Salt:", uint256(salt));

        vm.startBroadcast(deployerKey);

        VolatilityFeeHook hook = new VolatilityFeeHook{salt: salt}(
            IPoolManager(poolManager),
            operator
        );

        require(address(hook) == hookAddress, "Hook address mismatch");

        console2.log("VolatilityFeeHook deployed at:", address(hook));
        console2.log("Owner:", hook.owner());
        console2.log("Operator:", hook.operator());

        vm.stopBroadcast();
    }
}
