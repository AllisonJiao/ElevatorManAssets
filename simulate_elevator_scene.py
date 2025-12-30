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
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

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
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=AGIBOT_A2D_CFG.init_state.joint_pos,  # preserve original joint positions
            pos=(-2.0, -0.2, 0.0),
            rot=(math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)), # (w,x,y,z)
        ),
    )

def set_robot_pose_demo(
    agibot: Articulation, 
    phase: float, 
    right_joint_groups: dict[str, torch.Tensor],
    robot_animation_range: float = 1.0,
    sequential_linkages: bool = True,
    fixed_shoulder_angles: torch.Tensor = None,
):
    """Set robot right arm joints based on phase for smooth animation.
    
    Args:
        agibot: The robot articulation
        phase: Normalized phase value [0, 1] for animation cycle
        right_joint_groups: Dict mapping group names to tensors of right arm joint indices (e.g., {"shoulder": [...], "forearm": [...], "gripper": [...]})
        robot_animation_range: Multiplier for animation range (default 1.0 = full 2π rotation)
        sequential_linkages: If True, animates joint groups (forearm, gripper) sequentially (shoulder is fixed)
        fixed_shoulder_angles: Optional tensor of fixed angles for shoulder joints [joint1, joint2, joint3]. 
                               If None, uses default positions. Shape: (3,) or (num_envs, 3)
    """
    # Get all right joint IDs from groups
    all_right_ids = torch.cat([ids for ids in right_joint_groups.values()]) if right_joint_groups else torch.tensor([], dtype=torch.long, device=agibot.device)
    
    if len(all_right_ids) == 0:
        return
    
    # Calculate joint positions based on phase (smooth rotation)
    joint_pos_target = agibot.data.default_joint_pos.clone()
    
    # Fix shoulder to specified fixed angles (don't animate it)
    if "shoulder" in right_joint_groups:
        shoulder_ids = right_joint_groups["shoulder"]
        if fixed_shoulder_angles is not None:
            # Use the provided fixed angles
            if fixed_shoulder_angles.dim() == 1:
                # (3,) -> (1, 3) for broadcasting
                fixed_shoulder_angles = fixed_shoulder_angles.unsqueeze(0)
            joint_pos_target[:, shoulder_ids] = fixed_shoulder_angles
        else:
            # Use default positions if no fixed angles provided
            joint_pos_target[:, shoulder_ids] = agibot.data.default_joint_pos[:, shoulder_ids]
    
    # Get animatable groups (forearm and gripper, excluding shoulder)
    animatable_groups = {name: ids for name, ids in right_joint_groups.items() if name != "shoulder"}
    animatable_group_names = list(animatable_groups.keys())
    num_animatable_groups = len(animatable_group_names)
    
    if sequential_linkages and num_animatable_groups > 0:
        # Animate only forearm and gripper groups sequentially (shoulder is fixed)
        group_phase_range = 1.0 / num_animatable_groups
        for i, group_name in enumerate(animatable_group_names):
            group_ids = animatable_groups[group_name]
            group_start = i * group_phase_range
            group_end = (i + 1) * group_phase_range
            
            if phase >= group_end:
                # This group is complete, set to final position
                animation_offset = (2 * torch.pi * robot_animation_range)
                joint_pos_target[:, group_ids] = agibot.data.default_joint_pos[:, group_ids] - animation_offset
            elif phase >= group_start:
                # This group is currently animating
                group_phase = (phase - group_start) / group_phase_range
                animation_offset = group_phase * (2 * torch.pi * robot_animation_range)
                joint_pos_target[:, group_ids] = agibot.data.default_joint_pos[:, group_ids] - animation_offset
            else:
                # This group hasn't started yet, keep at default position
                joint_pos_target[:, group_ids] = agibot.data.default_joint_pos[:, group_ids]
    else:
        # Animate all animatable joints together (shoulder is fixed)
        animatable_ids = torch.cat([ids for ids in animatable_groups.values()]) if animatable_groups else torch.tensor([], dtype=torch.long, device=agibot.device)
        if len(animatable_ids) > 0:
            animation_offset = phase * (2 * torch.pi * robot_animation_range)
            joint_pos_target[:, animatable_ids] = agibot.data.default_joint_pos[:, animatable_ids] - animation_offset
    
    # Clamp to joint limits
    joint_pos_target = joint_pos_target.clamp_(
        agibot.data.soft_joint_pos_limits[..., 0], 
        agibot.data.soft_joint_pos_limits[..., 1]
    )
    agibot.set_joint_position_target(joint_pos_target)
    agibot.write_data_to_sim()


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    agibot: Articulation,
    elevator: Articulation,
    right_joint_groups: dict[str, torch.Tensor],
    elevator_door_ids: torch.Tensor,
    elevator_button_ids: torch.Tensor,
    robot_animation_range: float,
    fixed_shoulder_angles: torch.Tensor = None,
):
    """Run the simulation loop with robot and elevator animations."""
    # Animation parameters
    count = 0
    period = 500
    open_delta = -0.5  # 50 cm along chosen axis
    close_delta = 0.0

    print("[INFO] Done. Close the window to exit.")

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
            
            count = 0
        
        # Calculate phase for animations
        phase = count % period
        alpha = phase / max(1, period - 1)  # Normalized phase [0, 1] for robot animation

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

        # Update robot pose using phase-based animation (with sequential linkage movement)
        set_robot_pose_demo(
            agibot, alpha, right_joint_groups, robot_animation_range,
            sequential_linkages=True,
            fixed_shoulder_angles=fixed_shoulder_angles
        )

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

    robot_animation_range = args_cli.robot_animation_range

    # Set fixed shoulder angles for positioning arm in front of body
    # Adjust these to position the arm forward (e.g., rotate joint1 more forward)
    # Values: [joint1, joint2, joint3] in radians
    fixed_shoulder_angles = torch.tensor([0.8, -0.5, 0.8], device=agibot.device)

    # Run the simulator
    run_simulator(
        sim, scene, agibot, elevator,
        right_joint_groups,
        elevator_door_ids, elevator_button_ids,
        robot_animation_range,
        fixed_shoulder_angles=fixed_shoulder_angles
    )

    simulation_app.close()


if __name__ == "__main__":
    main()