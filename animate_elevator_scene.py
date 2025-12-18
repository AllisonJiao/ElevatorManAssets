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
parser.add_argument("--robot", type=str, default="agibot", help="Name of the robot.")
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
from isaaclab.assets import AssetBaseCfg, Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from cfg.agibot import AGIBOT_A2D_CFG  # isort:skip
from cfg.elevator import ELEVATOR_CFG  # isort:skip

# -----------------------------------------------------------------------------
# Scene config
# -----------------------------------------------------------------------------
ELEVATOR_ASSET_PATH = "ElevatorManAssets/assets/Collected_elevator_asset_tmp/elevator_asset.usdc"


@configclass
class ElevatorSceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )

    elevator = ELEVATOR_CFG.replace(prim_path="/World/Elevator/root")

    robot = AGIBOT_A2D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# -----------------------------------------------------------------------------
# Simulator loop
# -----------------------------------------------------------------------------
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):

    robot = scene["robot"]
    device = robot.device

    # ---------------- IK controllers ----------------
    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    )

    left_ik = DifferentialIKController(ik_cfg, scene.num_envs, device)
    right_ik = DifferentialIKController(ik_cfg, scene.num_envs, device)

    # ---------------- Scene entities ----------------
    left_cfg = SceneEntityCfg(
        "robot",
        joint_names=["left_arm_joint[1-5]"],
        body_names=["Link4_l"],
    )
    # right_cfg = SceneEntityCfg(
    #     "robot",
    #     joint_names=["right_arm_joint[1-5]"],
    #     body_names=["Link4_r"],
    # )
    right_cfg = SceneEntityCfg(
        "robot",
        joint_names=["left_arm_joint[6-7]"],
        body_names=["Link6_l"],
    )

    left_cfg.resolve(scene)
    right_cfg.resolve(scene)

    if robot.is_fixed_base:
        left_ee_jac = left_cfg.body_ids[0] - 1
        right_ee_jac = right_cfg.body_ids[0] - 1
    else:
        left_ee_jac = left_cfg.body_ids[0]
        right_ee_jac = right_cfg.body_ids[0]

    # ---------------- Markers ----------------
    frame_cfg = FRAME_MARKER_CFG.copy()
    frame_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    left_ee_marker = VisualizationMarkers(frame_cfg.replace(prim_path="/Visuals/left_ee"))
    right_ee_marker = VisualizationMarkers(frame_cfg.replace(prim_path="/Visuals/right_ee"))
    left_goal_marker = VisualizationMarkers(frame_cfg.replace(prim_path="/Visuals/left_goal"))
    right_goal_marker = VisualizationMarkers(frame_cfg.replace(prim_path="/Visuals/right_goal"))

    # ---------------- Goals ----------------
    left_arm_goals = torch.tensor([
        [0.20,  0.20, 0.60, 0.0, 0.707, 0.0, 0.707],
        [0.45,  0.20, 0.60, 0.0, 0.707, 0.0, 0.707],
        [-0.20,  0.20, 0.60, 0.0, 0.707, 0.0, 0.707],
    ], device=device)

    # right_arm_goals = torch.tensor([
    #     [0.45, 0.25, 0.60, 0.0, 0.707, 0.0, 0.707],
    #     [0.50, 0.20, 0.50, 0.0, 0.707, 0.0, 0.707],
    #     [0.40, 0.30, 0.55, 0.0, 0.707, 0.0, 0.707],
    # ], device=device)
    right_arm_goals = torch.tensor([
        [0.20,  0.20, 0.60, 0.0, 0.707, 0.0, 0.707],
        [0.45,  0.20, 0.60, 0.0, 0.707, 0.0, 0.707],
        [-0.20,  0.20, 0.60, 0.0, 0.707, 0.0, 0.707],
    ], device=device)

    left_cmd = torch.zeros(scene.num_envs, 7, device=device)
    right_cmd = torch.zeros(scene.num_envs, 7, device=device)

    # ---------------- Timing ----------------
    sim_dt = sim.get_physics_dt()
    period = 150
    count = 0
    goal_idx = 0

    # ---------------- Loop ----------------
    while simulation_app.is_running():

        if count % period == 0:
            count = 0

            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos,
                robot.data.default_joint_vel,
            )
            robot.reset()

            goal_idx = (goal_idx + 1) % left_arm_goals.shape[0]
            left_cmd[:] = left_arm_goals[goal_idx]
            right_cmd[:] = right_arm_goals[goal_idx]

            left_ik.reset()
            left_ik.set_command(left_cmd)

            right_ik.reset()
            right_ik.set_command(right_cmd)

            print("[INFO]: Resetting state...")

        # ---------------- Left arm IK ----------------
        root_w = robot.data.root_pose_w

        left_ee_w = robot.data.body_pose_w[:, left_cfg.body_ids[0]]
        left_jac = robot.root_physx_view.get_jacobians()[:, left_ee_jac, :, left_cfg.joint_ids]
        left_q = robot.data.joint_pos[:, left_cfg.joint_ids]

        left_pos_b, left_quat_b = subtract_frame_transforms(
            root_w[:, :3], root_w[:, 3:7],
            left_ee_w[:, :3], left_ee_w[:, 3:7],
        )

        left_q_des = left_ik.compute(left_pos_b, left_quat_b, left_jac, left_q)

        # ---------------- Right arm IK ----------------
        right_ee_w = robot.data.body_pose_w[:, right_cfg.body_ids[0]]
        right_jac = robot.root_physx_view.get_jacobians()[:, right_ee_jac, :, right_cfg.joint_ids]
        right_q = robot.data.joint_pos[:, right_cfg.joint_ids]

        right_pos_b, right_quat_b = subtract_frame_transforms(
            root_w[:, :3], root_w[:, 3:7],
            right_ee_w[:, :3], right_ee_w[:, 3:7],
        )

        right_q_des = right_ik.compute(right_pos_b, right_quat_b, right_jac, right_q)

        # ---------------- Apply ----------------
        robot.set_joint_position_target(left_q_des, left_cfg.joint_ids)
        robot.set_joint_position_target(right_q_des, right_cfg.joint_ids)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        count += 1

        # ---------------- Visuals ----------------
        left_ee_marker.visualize(left_ee_w[:, :3], left_ee_w[:, 3:7])
        right_ee_marker.visualize(right_ee_w[:, :3], right_ee_w[:, 3:7])
        left_goal_marker.visualize(left_cmd[:, :3] + scene.env_origins, left_cmd[:, 3:7])
        right_goal_marker.visualize(right_cmd[:, :3] + scene.env_origins, right_cmd[:, 3:7])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = ElevatorSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()