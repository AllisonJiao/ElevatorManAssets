# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom elevator to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
# import prims as prim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from agibot import AGIBOT_A2D_CFG

ELEVATOR_ASSET_PATH = "ElevatorManAssets/assets/elevator_standalone_bodies.usdc"

def design_scene() -> tuple[dict]:
    """Designs the scene."""

    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Light(s)
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Elevator
    cfg = sim_utils.UsdFileCfg(usd_path=ELEVATOR_ASSET_PATH)
    cfg.func("/World/Elevator", cfg)

    # Origin(s)
    # origins = [[0.0, 0.0, 0.0]]
    # Origin 1
    # prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])

    # Robot(s)
    agibot_cfg = AGIBOT_A2D_CFG.copy()
    agibot_cfg.prim_path = "/World/Robot"
    agibot = Articulation(cfg = agibot_cfg)

    scene_entities = {"agibot": agibot}
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation]):
    robot = entities["agibot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            # reset counters
            count = 0
            # reset the scene entities
            for index, robot in enumerate(entities.values()):
                # root state
                root_state = robot.data.default_root_state.clone()
                # root_state[:, :3] += origins[index]
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])
                # set joint positions with some noise
                joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
                joint_pos += torch.rand_like(joint_pos) * 0.1
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                # clear internal buffers
                robot.reset()
            print("[INFO]: Resetting robot state...")
        # apply random actions to the robot(s)
        for robot in entities.values():
            num_envs, num_dofs = robot.data.joint_pos.shape
            device = robot.data.joint_pos.device

            fixed_joint_names = ["joint_lift_body", "joint_body_pitch"]
            
            # resolve indices for the fixed joints (assumes robot.data.joint_names is iterable of strings)
            joint_names_raw = list(robot.data.joint_names)
            joint_names = [(n.decode("utf-8") if isinstance(n, (bytes, bytearray)) else str(n)) for n in joint_names_raw]
            
            fixed_idx_list = [i for i, name in enumerate(joint_names) if name in fixed_joint_names]
            fixed_idx_list = [i for i in fixed_idx_list if 0 <= i < num_dofs]
            
            fixed_mask = torch.zeros((num_dofs,), dtype=torch.bool, device=device)
            if fixed_idx_list:
                fixed_mask[torch.tensor(fixed_idx_list, dtype=torch.long, device=device)] = True

            period = 500
            t = count % period
            frac = float(t) / float(max(1, period - 1))  # fraction [0,1]

            # read joint limits (shape N x 2: [min, max])
            limits = robot.data.soft_joint_pos_limits.to(device)

            # normalize limits to shape (num_dofs, 2)
            if limits.dim() == 3:          # e.g. (num_envs, num_dofs, 2)
                limits_ = limits[0]
            else:                          # e.g. (num_dofs, 2)
                limits_ = limits

            lo = limits_[:, 0]             # (num_dofs,)
            hi = limits_[:, 1]             # (num_dofs,)

            joint_pos_target_1d = lo + (hi - lo) * float(frac)     # (num_dofs,)
            joint_pos_target = joint_pos_target_1d.unsqueeze(0).repeat(num_envs, 1)  # (num_envs, num_dofs)
            
            # --- keep fixed joints at current positions ---
            current_pos = robot.data.joint_pos.to(device)          # (num_envs, num_dofs)
            if fixed_mask.any():
                joint_pos_target[:, fixed_mask] = current_pos[:, fixed_mask]
            
            robot.set_joint_position_target(joint_pos_target)
            robot.write_data_to_sim()

            if count % 60 == 0:
                print("root pos:", robot.data.root_pos_w[0].cpu().numpy())

        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # update buffers
        for robot in entities.values():
            robot.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_entities = design_scene()
    # scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()