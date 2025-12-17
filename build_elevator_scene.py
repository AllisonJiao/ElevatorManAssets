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

from cfg.agibot import AGIBOT_A2D_CFG
from cfg.elevator import ELEVATOR_CFG

def design_scene() -> tuple[dict]:
    """Designs the scene."""

    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Light(s)
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Elevator
    elevator_cfg = ELEVATOR_CFG.copy()
    elevator_cfg.prim_path = "/World/Elevator/root"
    elevator = Articulation(cfg = elevator_cfg)

    # Origin(s)
    # origins = [[0.0, 0.0, 0.0]]
    # Origin 1
    # prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])

    # Robot(s)
    agibot_cfg = AGIBOT_A2D_CFG.copy()
    agibot_cfg.prim_path = "/World/Robot"
    agibot = Articulation(cfg = agibot_cfg)

    scene_entities = {"agibot": agibot, "elevator": elevator}
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation]):
    agibot = entities["agibot"]
    elevator = entities["elevator"]

    animate_joint_names = [
        n for n in agibot.data.joint_names
        if n.startswith("left_arm_joint") or n.startswith("right_arm_joint")
    ]
    animate_ids, _ = agibot.find_joints(animate_joint_names)
    animate_ids = torch.as_tensor(animate_ids, device=agibot.device, dtype=torch.long)

    sim_dt = sim.get_physics_dt()
    count = 0
    period = 500

    while simulation_app.is_running():
        if count % period == 0:
            count = 0
            for a in [agibot, elevator]:
                root_state = a.data.default_root_state.clone()
                a.write_root_pose_to_sim(root_state[:, :7])
                a.write_root_velocity_to_sim(root_state[:, 7:])

                joint_pos = a.data.default_joint_pos.clone()
                joint_vel = a.data.default_joint_vel.clone()
                joint_pos += torch.rand_like(joint_pos) * 0.1
                a.write_joint_state_to_sim(joint_pos, joint_vel)
                a.reset()
            print("[INFO]: Resetting state...")

        alpha = (count % period) / max(1, period - 1)

        # control agibot ONLY
        joint_pos_target = agibot.data.default_joint_pos.clone()
        joint_pos_target[:, animate_ids] += alpha * (2 * torch.pi)
        joint_pos_target = joint_pos_target.clamp_(
            agibot.data.soft_joint_pos_limits[..., 0], agibot.data.soft_joint_pos_limits[..., 1]
        )
        agibot.set_joint_position_target(joint_pos_target)

        # write to sim
        agibot.write_data_to_sim()
        elevator.write_data_to_sim()  # no control yet

        sim.step()
        count += 1

        agibot.update(sim_dt)
        elevator.update(sim_dt)

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