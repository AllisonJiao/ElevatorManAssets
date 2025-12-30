# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import sys
from pathlib import Path

# Add parent directory to path so we can import from cfg
sys.path.insert(0, str(Path(__file__).parent.parent))

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
import math

import isaaclab.sim as sim_utils
# import prims as prim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, Articulation
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

@configclass
class ElevatorSceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )

    # elevator
    elevator: ArticulationCfg = ELEVATOR_CFG.replace(prim_path="/World/elevator")

    # robot
    agibot: ArticulationCfg = AGIBOT_A2D_CFG.replace(
        prim_path="/World/Agibot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=AGIBOT_A2D_CFG.init_state.joint_pos,  # preserve original joint positions
            pos=(-2.0, -0.2, 0.0),
            rot=(math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)), # (w,x,y,z)
        ),
    )

# -----------------------------------------------------------------------------
# Get EE Goals
# -----------------------------------------------------------------------------
def get_ee_goals(
    elevator: Articulation,
    robot: Articulation,
    button_body_names: list[str],
    default_quat: torch.Tensor = None,
) -> torch.Tensor:
    """Get elevator button positions in world frame and convert to robot local frame.
    
    Args:
        elevator: The elevator articulation
        robot: The robot articulation (for root pose reference)
        button_body_names: List of button body names (e.g., ["ElevatorButton_0_0", "ElevatorButton_0_1", ...])
        default_quat: Default quaternion for button orientation [qx, qy, qz, qw]. 
                      If None, uses identity quaternion [0, 0, 0, 1]
    
    Returns:
        goals: Tensor of shape (num_buttons, 7) with [x, y, z, qx, qy, qz, qw] in robot local frame
               Each row corresponds to one button goal
    """
    device = elevator.device
    
    # Find button body IDs by matching body names
    button_body_ids = []
    for body_name in button_body_names:
        if body_name in elevator.data.body_names:
            body_id = list(elevator.data.body_names).index(body_name)
            button_body_ids.append(body_id)
    
    num_buttons = len(button_body_ids)
    
    if num_buttons == 0:
        return torch.zeros((0, 7), device=device)
    
    button_body_ids_tensor = torch.tensor(button_body_ids, device=device, dtype=torch.long)
    
    # Get button body poses in world frame (use env 0)
    # body_pose_w shape: (num_envs, num_bodies, 7) where last dim is [x, y, z, qx, qy, qz, qw]
    button_poses_w = elevator.data.body_pose_w[0, button_body_ids_tensor, :]  # (num_buttons, 7)
    
    # Get robot root pose for frame transformation (use env 0)
    robot_root_pos_w = robot.data.root_pose_w[0:1, :3]  # (1, 3) - batched for subtract_frame_transforms
    robot_root_quat_w = robot.data.root_pose_w[0:1, 3:7]  # (1, 4) - batched for subtract_frame_transforms
    
    # Default quaternion for button orientation (identity if not provided)
    if default_quat is None:
        default_quat_w = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)  # [qx, qy, qz, qw]
    else:
        default_quat_w = default_quat
    
    # Extract button positions and orientations from world frame
    button_pos_w = button_poses_w[:, :3]  # (num_buttons, 3)
    button_quat_w = button_poses_w[:, 3:7] if button_poses_w.shape[1] >= 7 else default_quat_w.unsqueeze(0).expand(num_buttons, 4)  # (num_buttons, 4)
    
    # Convert each button pose from world frame to robot local frame
    goals_list = []
    for button_idx in range(num_buttons):
        # Extract single button pose (batched for subtract_frame_transforms)
        button_pos_w_i = button_pos_w[button_idx:button_idx+1, :]  # (1, 3)
        button_quat_w_i = button_quat_w[button_idx:button_idx+1, :]  # (1, 4)
        
        # Convert from world frame to robot local frame using subtract_frame_transforms
        # Expects (num_envs, 3) and (num_envs, 4) shapes
        button_pos_b, button_quat_b = subtract_frame_transforms(
            robot_root_pos_w, robot_root_quat_w,
            button_pos_w_i, button_quat_w_i,
        )
        
        # Extract result (first and only env)
        button_pos_b = button_pos_b[0, :]  # (3,)
        button_quat_b = button_quat_b[0, :]  # (4,)
        
        # Combine position and quaternion: [x, y, z, qx, qy, qz, qw]
        goal = torch.cat([button_pos_b, button_quat_b], dim=-1)  # (7,)
        goals_list.append(goal)
    
    # Stack all goals: (num_buttons, 7)
    goals = torch.stack(goals_list, dim=0)  # (num_buttons, 7)
    
    return goals

# -----------------------------------------------------------------------------
# Simulator loop
# -----------------------------------------------------------------------------
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):

    robot = scene["agibot"]
    device = robot.device

    # ---------------- IK controllers ----------------
    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    )

    left_ik = DifferentialIKController(ik_cfg, scene.num_envs, device)
    # right_ik = DifferentialIKController(ik_cfg, scene.num_envs, device)

    # ---------------- Scene entities ----------------
    left_cfg = SceneEntityCfg(
        "agibot",
        joint_names=["left_arm_joint[1-7]"],
        body_names=["Link6_l"],
    )
    # right_cfg = SceneEntityCfg(
    #     "robot",
    #     joint_names=["right_arm_joint[1-7]"],
    #     body_names=["Link6_r"],
    # )

    left_cfg.resolve(scene)
    # right_cfg.resolve(scene)

    if robot.is_fixed_base:
        left_ee_jac = left_cfg.body_ids[0] - 1
        # right_ee_jac = right_cfg.body_ids[0] - 1
    else:
        left_ee_jac = left_cfg.body_ids[0]
        # right_ee_jac = right_cfg.body_ids[0]

    # ---------------- Markers ----------------
    frame_cfg = FRAME_MARKER_CFG.copy()
    frame_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    left_ee_marker = VisualizationMarkers(frame_cfg.replace(prim_path="/Visuals/left_ee"))
    # right_ee_marker = VisualizationMarkers(frame_cfg.replace(prim_path="/Visuals/right_ee"))
    left_goal_marker = VisualizationMarkers(frame_cfg.replace(prim_path="/Visuals/left_goal"))
    # right_goal_marker = VisualizationMarkers(frame_cfg.replace(prim_path="/Visuals/right_goal"))

    # ---------------- Goals ----------------
    left_arm_goals = get_ee_goals(
        scene["elevator"], 
        robot, 
        ["button_0_0_link", "button_0_1_link", "button_1_0_link", "button_1_1_link", "button_2_0_link", "button_2_1_link", "button_3_0_link", "button_3_1_link"]
    )

    # right_arm_goals = torch.tensor([
    #     [0.25, -0.22, 0.18, 0.0, 0.7071, 0.0, 0.7071],
    #     [0.30, -0.20, 0.26, 0.0, 0.7071, 0.0, 0.7071],
    #     [0.25, -0.18, 0.34, 0.0, 0.7071, 0.0, 0.7071],
    # ], device=device)

    left_cmd = torch.zeros(scene.num_envs, 7, device=device)
    # right_cmd = torch.zeros(scene.num_envs, 7, device=device)

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
            # right_cmd[:] = right_arm_goals[goal_idx]

            left_ik.reset()
            left_ik.set_command(left_cmd)

            # right_ik.reset()
            # right_ik.set_command(right_cmd)

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
        # right_ee_w = robot.data.body_pose_w[:, right_cfg.body_ids[0]]
        # right_jac = robot.root_physx_view.get_jacobians()[:, right_ee_jac, :, right_cfg.joint_ids]
        # right_q = robot.data.joint_pos[:, right_cfg.joint_ids]

        # right_pos_b, right_quat_b = subtract_frame_transforms(
        #     root_w[:, :3], root_w[:, 3:7],
        #     right_ee_w[:, :3], right_ee_w[:, 3:7],
        # )

        # right_q_des = right_ik.compute(right_pos_b, right_quat_b, right_jac, right_q)

        # ---------------- Apply ----------------
        robot.set_joint_position_target(left_q_des, left_cfg.joint_ids)
        # robot.set_joint_position_target(right_q_des, right_cfg.joint_ids)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        count += 1

        # ---------------- Visuals ----------------
        left_ee_marker.visualize(left_ee_w[:, :3], left_ee_w[:, 3:7])
        # right_ee_marker.visualize(right_ee_w[:, :3], right_ee_w[:, 3:7])
        left_goal_marker.visualize(left_cmd[:, :3] + scene.env_origins, left_cmd[:, 3:7])
        # right_goal_marker.visualize(right_cmd[:, :3] + scene.env_origins, right_cmd[:, 3:7])


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