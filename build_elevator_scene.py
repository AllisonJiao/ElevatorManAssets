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

ELEVATOR_ASSET_PATH = "ElevatorManAssets/assets/elevator_rigged.usdc"

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

    animate_joint_names = [
        n for n in robot.data.joint_names
        if n.startswith("left_arm_joint") or n.startswith("right_arm_joint")
    ]

    animate_ids, _ = robot.find_joints(animate_joint_names)   # returns (ids, names)
    animate_ids = torch.as_tensor(animate_ids, device=robot.device, dtype=torch.long)

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
        period = 500
        t = count % period
        alpha = t / max(1, period - 1)
        # apply random actions to the robots
        for robot in entities.values():
            # generate random joint positions
            joint_pos_target = robot.data.default_joint_pos.clone()
            # Only animate chosen joints: current + 2 * pi (scaled by alpha)
            joint_pos_target[:, animate_ids] += alpha * (2 * torch.pi)

            joint_pos_target = joint_pos_target.clamp_(
                robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
            )
            # apply action to the robot
            robot.set_joint_position_target(joint_pos_target)
            # write data to sim
            robot.write_data_to_sim()
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