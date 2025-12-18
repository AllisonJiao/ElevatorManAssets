# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to list all joints in the elevator USD file."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from cfg.elevator import ELEVATOR_CFG

def main():
    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    
    # Create elevator articulation
    elevator_cfg = ELEVATOR_CFG.copy()
    elevator_cfg.prim_path = "/World/Elevator/root"
    
    # Try to create articulation without joint configs first
    # Temporarily remove joint_pos and actuators to see what joints exist
    temp_cfg = elevator_cfg.copy()
    temp_cfg.init_state.joint_pos = {}
    temp_cfg.actuators = {}
    
    try:
        elevator = Articulation(cfg=temp_cfg)
        sim.reset()
        
        print("\n" + "="*60)
        print("ELEVATOR JOINT DISCOVERY")
        print("="*60)
        print(f"\nTotal joints found: {len(elevator.data.joint_names)}")
        print("\nJoint names in USD file:")
        for i, joint_name in enumerate(elevator.data.joint_names, 1):
            print(f"  {i}. {joint_name}")
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"\nError loading articulation: {e}")
        print("\nThis might mean:")
        print("  1. The USD file path is incorrect")
        print("  2. The USD file doesn't contain a valid articulation")
        print("  3. The USD file structure is different than expected")
    
    simulation_app.close()

if __name__ == "__main__":
    main()

