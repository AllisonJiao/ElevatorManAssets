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
parser.add_argument("--robot_animation_range", type=float, default=0.1, help="Range of robot arm animation (0.0-1.0, where 1.0 = full 2π rotation)")
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
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from cfg.agibot import AGIBOT_A2D_CFG
from cfg.elevator import ELEVATOR_CFG

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
        spawn=AGIBOT_A2D_CFG.spawn.replace(
            scale=(1.2, 1.2, 1.2),  # Scale factor (x, y, z)
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=AGIBOT_A2D_CFG.init_state.joint_pos,  # preserve original joint positions
            pos=(-2.0, -0.2, 0.0),
            rot=(math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)), # (w,x,y,z)
        ),
    )

# -----------------------------------------------------------------------------
# Get EE Goals (copied from ik_solver.py)
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
        button_body_names: List of button body names (e.g., ["button_0_0_link", "button_0_1_link", ...])
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


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    agibot: Articulation,
    elevator: Articulation,
    right_joint_groups: dict[str, torch.Tensor],
    right_arm_joint_ids: torch.Tensor,  # All right arm joint IDs [1-7]
    right_forearm_gripper_ids: torch.Tensor,  # Forearm and gripper joint IDs [4-7]
    elevator_door_ids: torch.Tensor,
    elevator_button_ids: torch.Tensor,
    right_ik: DifferentialIKController,
    right_cfg: SceneEntityCfg,
    right_ee_jac: int,
    right_arm_goals: torch.Tensor,
    fixed_shoulder_angles: torch.Tensor,
):
    """Run the simulation loop with robot and elevator animations."""
    # Animation parameters
    count = 0
    period = 500
    open_delta = -0.5  # 50 cm along chosen axis
    close_delta = 0.0

    print("[INFO] Done. Close the window to exit.")

    right_cmd = torch.zeros(scene.num_envs, 7, device=agibot.device)
    goal_idx = 0

    while simulation_app.is_running():
        # Reset robot and elevator to default positions at the start of each period
        if count % period == 0:
            # Reset robot joint positions to default
            agibot.write_joint_state_to_sim(
                agibot.data.default_joint_pos.clone(),
                agibot.data.default_joint_vel.clone()
            )
            agibot.reset()
            
            # Reset elevator door position and button positions to default (initial position)
            elevator.write_joint_state_to_sim(
                elevator.data.default_joint_pos.clone(),
                elevator.data.default_joint_vel.clone()
            )
            elevator.reset()
            
            # Move to next button goal
            goal_idx = (goal_idx + 1) % right_arm_goals.shape[0]
            # right_cmd shape: (num_envs, 7), right_arm_goals[goal_idx] shape: (7,)
            # Need to expand to match batch dimension
            right_cmd[:, :] = right_arm_goals[goal_idx].unsqueeze(0).expand(scene.num_envs, -1)
            right_ik.reset()
            right_ik.set_command(right_cmd)
            
            count = 0
        
        # Calculate phase for animations
        phase = count % period

        # Calculate door animation delta based on phase
        if phase < 100:        # opening (first 100 frames)
            t = phase / 99.0
            delta = close_delta + t * (open_delta - close_delta)
        elif phase < 400:      # hold open (frames 100-399)
            delta = open_delta
        else:                  # closing (frames 400-499)
            t = (phase - 400) / 99.0
            delta = open_delta + t * (close_delta - open_delta)

        # Update elevator joint positions (doors and buttons) using joint-based animation
        joint_pos_target = elevator.data.default_joint_pos.clone()
        
        # Update door position
        joint_pos_target[:, elevator_door_ids] += delta
        
        # Update button positions - press down gradually over the period
        # Button press animation: starts at 0, reaches max press at phase 0.5, stays pressed
        button_press_delta = min(phase / (period / 2.0), 1.0) * 0.05  # Max press distance of 0.05
        joint_pos_target[:, elevator_button_ids] += button_press_delta
        
        # Clamp all joints to their limits
        joint_pos_target = joint_pos_target.clamp_(
            elevator.data.soft_joint_pos_limits[..., 0], elevator.data.soft_joint_pos_limits[..., 1]
        )
        elevator.set_joint_position_target(joint_pos_target)
        elevator.write_data_to_sim()

        # Compute IK for right arm (IK computes all joints [1-7], but we'll only use forearm/gripper [4-7])
        root_w = agibot.data.root_pose_w
        right_ee_w = agibot.data.body_pose_w[:, right_cfg.body_ids[0]]
        right_jac = agibot.root_physx_view.get_jacobians()[:, right_ee_jac, :, right_cfg.joint_ids]
        right_q = agibot.data.joint_pos[:, right_cfg.joint_ids]
        
        right_pos_b, right_quat_b = subtract_frame_transforms(
            root_w[:, :3], root_w[:, 3:7],
            right_ee_w[:, :3], right_ee_w[:, 3:7],
        )
        
        right_q_des_all = right_ik.compute(right_pos_b, right_quat_b, right_jac, right_q)
        
        # Build joint position target: fixed shoulder + IK-computed forearm/gripper
        joint_pos_target = agibot.data.default_joint_pos.clone()
        
        # Set shoulder to fixed angles
        if fixed_shoulder_angles is not None:
            if fixed_shoulder_angles.dim() == 1:
                fixed_shoulder_angles_batched = fixed_shoulder_angles.unsqueeze(0)
            else:
                fixed_shoulder_angles_batched = fixed_shoulder_angles
            shoulder_ids = right_joint_groups["shoulder"]
            joint_pos_target[:, shoulder_ids] = fixed_shoulder_angles_batched
        
        # Set forearm and gripper to IK-computed values
        # right_q_des_all contains positions for joints in right_cfg.joint_ids (should be [1-7])
        # Find which positions in right_q_des_all correspond to forearm/gripper joints
        right_cfg_joint_ids_tensor = torch.tensor(right_cfg.joint_ids, device=agibot.device, dtype=torch.long)
        forearm_gripper_mask = torch.isin(right_cfg_joint_ids_tensor, right_forearm_gripper_ids)
        forearm_gripper_indices_in_ik = torch.where(forearm_gripper_mask)[0]
        
        if len(forearm_gripper_indices_in_ik) > 0:
            # Extract IK-computed positions for forearm/gripper
            forearm_gripper_ik_positions = right_q_des_all[:, forearm_gripper_indices_in_ik]
            joint_pos_target[:, right_forearm_gripper_ids] = forearm_gripper_ik_positions
        
        # Clamp to joint limits
        joint_pos_target = joint_pos_target.clamp_(
            agibot.data.soft_joint_pos_limits[..., 0], 
            agibot.data.soft_joint_pos_limits[..., 1]
        )
        agibot.set_joint_position_target(joint_pos_target)
        agibot.write_data_to_sim()

        sim.step()
        scene.update(sim.get_physics_dt())

        count += 1


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    scene_cfg = ElevatorSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    # Access the robot articulation (because we used ArticulationCfg)
    agibot: Articulation = scene["agibot"]
    elevator: Articulation = scene["elevator"]

    # Setup robot joint animation - organize right arm into groups: shoulder (1-3), forearm (4-5), gripper (6-7)
    right_shoulder_names = ["right_arm_joint1", "right_arm_joint2", "right_arm_joint3"]
    right_forearm_names = ["right_arm_joint4", "right_arm_joint5"]
    right_gripper_names = ["right_arm_joint6", "right_arm_joint7"]

    # Find joint indices for each group
    right_shoulder_ids, _ = agibot.find_joints(right_shoulder_names)
    right_forearm_ids, _ = agibot.find_joints(right_forearm_names)
    right_gripper_ids, _ = agibot.find_joints(right_gripper_names)

    # Organize into groups
    right_joint_groups = {
        "shoulder": torch.as_tensor(right_shoulder_ids, device=agibot.device, dtype=torch.long),
        "forearm": torch.as_tensor(right_forearm_ids, device=agibot.device, dtype=torch.long),
        "gripper": torch.as_tensor(right_gripper_ids, device=agibot.device, dtype=torch.long),
    }

    total_right = sum(len(ids) for ids in right_joint_groups.values())

    if total_right > 0:
        print(f"[INFO] Organized right arm joints into groups:")
        print(f"  Right arm - Shoulder: {len(right_joint_groups['shoulder'])}, Forearm: {len(right_joint_groups['forearm'])}, Gripper: {len(right_joint_groups['gripper'])}")

        # Ensure data is updated
        scene.update(sim.get_physics_dt())
    else:
        right_joint_groups = {}
        print("[WARN] No right arm joints found for animation. Robot will use default pose.")

    elevator_door_joint_names = ["door2_joint"]
    elevator_door_ids, _ = elevator.find_joints(elevator_door_joint_names)
    elevator_door_ids = torch.as_tensor(elevator_door_ids, device=elevator.device, dtype=torch.long)

    elevator_button_joint_names = ["button_0_0_joint", "button_0_1_joint", "button_1_0_joint", "button_1_1_joint", "button_2_0_joint", "button_2_1_joint", "button_3_0_joint", "button_3_1_joint"]
    elevator_button_ids, _ = elevator.find_joints(elevator_button_joint_names)
    elevator_button_ids = torch.as_tensor(elevator_button_ids, device=elevator.device, dtype=torch.long)

    # Set fixed shoulder angles for positioning arm in front of body
    # Adjust these to position the arm forward (e.g., rotate joint1 more forward)
    # Values: [joint1, joint2, joint3] in radians
    fixed_shoulder_angles = torch.tensor([1.5, 0, 2], device=agibot.device)

    # Setup IK controller for right arm
    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    )
    right_ik = DifferentialIKController(ik_cfg, scene.num_envs, agibot.device)

    # Setup right arm scene entity (for IK computation)
    right_cfg = SceneEntityCfg(
        "agibot",
        joint_names=["right_arm_joint[4-7]"],
        body_names=["right_Right_Pad_Link"],
    )
    right_cfg.resolve(scene)

    if agibot.is_fixed_base:
        right_ee_jac = right_cfg.body_ids[0] - 1
    else:
        right_ee_jac = right_cfg.body_ids[0]

    # Get all right arm joint IDs and forearm+gripper IDs
    right_arm_joint_names = ["right_arm_joint1", "right_arm_joint2", "right_arm_joint3", 
                              "right_arm_joint4", "right_arm_joint5", "right_arm_joint6", "right_arm_joint7"]
    right_arm_joint_ids, _ = agibot.find_joints(right_arm_joint_names)
    right_arm_joint_ids = torch.as_tensor(right_arm_joint_ids, device=agibot.device, dtype=torch.long)
    
    # Forearm and gripper joint IDs (4-7)
    right_forearm_gripper_ids = torch.cat([
        right_joint_groups["forearm"],
        right_joint_groups["gripper"]
    ])

    # Get button goals (button body names need to be checked - using link names from ik_solver.py)
    button_body_names = ["button_0_0_link", "button_0_1_link", "button_1_0_link", "button_1_1_link", 
                         "button_2_0_link", "button_2_1_link", "button_3_0_link", "button_3_1_link"]
    right_arm_goals = get_ee_goals(elevator, agibot, button_body_names)

    # Run the simulator
    run_simulator(
        sim, scene, agibot, elevator,
        right_joint_groups,
        right_arm_joint_ids,
        right_forearm_gripper_ids,
        elevator_door_ids, elevator_button_ids,
        right_ik, right_cfg, right_ee_jac,
        right_arm_goals,
        fixed_shoulder_angles
    )

    simulation_app.close()


if __name__ == "__main__":
    main()