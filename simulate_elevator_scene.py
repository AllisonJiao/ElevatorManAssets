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
parser.add_argument("--period", type=int, default=1000, help="Number of simulation steps per animation period (longer = more time for IK to converge)")
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
    """Get elevator button positions in world frame (computed from Blender layout) and convert to robot local frame.
    
    Args:
        elevator: The elevator articulation (not used for positions, but for device reference)
        robot: The robot articulation (for root pose reference)
        button_body_names: List of button body names (e.g., ["button_0_0_link", "button_0_1_link", ...])
        default_quat: Default quaternion for button orientation [qx, qy, qz, qw]. 
                      If None, uses identity quaternion [0, 0, 0, 1]
    
    Returns:
        goals: Tensor of shape (num_buttons, 7) with [x, y, z, qx, qy, qz, qw] in robot local frame
               Each row corresponds to one button goal
    """
    device = elevator.device
    
    # Button layout parameters from Blender
    start_x, start_y, start_z = -1.75, -0.92, 1.625
    dx, dz = -0.22, -0.178
    rows, cols = 4, 2
    
    # Compute button positions in world frame based on Blender layout
    # Parse button indices from names (e.g., "button_0_0_link" -> row=0, col=0)
    button_positions_w = []
    for body_name in button_body_names:
        # Extract row and col from button name (format: "button_<row>_<col>_link")
        try:
            parts = body_name.split("_")
            if len(parts) >= 3 and parts[0] == "button":
                row = int(parts[1])
                col = int(parts[2])
                # Compute position: x = start_x + c * dx, y = start_y, z = start_z + r * dz
                x = start_x + col * dx
                y = start_y
                z = start_z + row * dz
                button_positions_w.append([x, y, z])
            else:
                # If name doesn't match expected format, use default position
                print(f"[WARN] Button name '{body_name}' doesn't match expected format 'button_<row>_<col>_link', using default position")
                button_positions_w.append([start_x, start_y, start_z])
        except (ValueError, IndexError) as e:
            print(f"[WARN] Failed to parse button name '{body_name}': {e}, using default position")
            button_positions_w.append([start_x, start_y, start_z])
    
    num_buttons = len(button_positions_w)
    if num_buttons == 0:
        return torch.zeros((0, 7), device=device)
    
    # Convert to tensor: (num_buttons, 3)
    button_pos_w = torch.tensor(button_positions_w, device=device, dtype=torch.float32)
    
    # Get robot root pose for frame transformation (use env 0)
    robot_root_pos_w = robot.data.root_pose_w[0:1, :3]  # (1, 3) - batched for subtract_frame_transforms
    robot_root_quat_w = robot.data.root_pose_w[0:1, 3:7]  # (1, 4) - batched for subtract_frame_transforms
    
    # Default quaternion for button orientation (identity if not provided)
    if default_quat is None:
        default_quat_w = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)  # [qx, qy, qz, qw]
    else:
        default_quat_w = default_quat
    
    # Expand quaternion to match number of buttons
    button_quat_w = default_quat_w.unsqueeze(0).expand(num_buttons, 4)  # (num_buttons, 4)
    
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
    elevator_door_ids: torch.Tensor,
    elevator_button_ids: torch.Tensor,
    right_ik: DifferentialIKController,
    right_cfg: SceneEntityCfg,
    right_ee_jac: int,
    right_arm_goals: torch.Tensor,
    button_body_names: list[str],
    goal_marker: VisualizationMarkers,
    period: int = 1000,  # Increased default period to allow more time for IK to converge
):
    """Run the simulation loop with robot and elevator animations.
    
    Args:
        period: Number of simulation steps per animation period. Longer periods give IK more time to converge.
        goal_marker: Visualization marker for displaying the IK goal position
    """
    # Animation parameters
    count = 0
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
        # Button press animation: starts at 0, reaches max press at 50% of period, stays pressed
        button_press_delta = min(phase / (period / 2.0), 1.0) * 0.05  # Max press distance of 0.05
        joint_pos_target[:, elevator_button_ids] += button_press_delta
        
        # Clamp all joints to their limits
        joint_pos_target = joint_pos_target.clamp_(
            elevator.data.soft_joint_pos_limits[..., 0], elevator.data.soft_joint_pos_limits[..., 1]
        )
        elevator.set_joint_position_target(joint_pos_target)
        elevator.write_data_to_sim()

        # Compute IK for right arm (all joints [1-7])
        root_w = agibot.data.root_pose_w
        right_ee_w = agibot.data.body_pose_w[:, right_cfg.body_ids[0]]
        right_jac = agibot.root_physx_view.get_jacobians()[:, right_ee_jac, :, right_cfg.joint_ids]
        right_q = agibot.data.joint_pos[:, right_cfg.joint_ids]
        
        right_pos_b, right_quat_b = subtract_frame_transforms(
            root_w[:, :3], root_w[:, 3:7],
            right_ee_w[:, :3], right_ee_w[:, 3:7],
        )
        
        right_q_des = right_ik.compute(right_pos_b, right_quat_b, right_jac, right_q)
        
        # Apply IK-computed joint positions directly to the robot
        # right_q_des contains positions for all joints in right_cfg.joint_ids [1-7]
        agibot.set_joint_position_target(right_q_des, right_cfg.joint_ids)
        agibot.write_data_to_sim()

        sim.step()
        scene.update(sim.get_physics_dt())

        # Update goal marker visualization
        # Compute goal position in world frame directly from Blender layout
        if goal_idx < len(button_body_names):
            # Blender layout parameters (same as in get_ee_goals)
            start_x, start_y, start_z = -1.75, -0.92, 1.625
            dx, dz = -0.22, -0.178
            
            # Parse row and col from button name to get world position
            try:
                body_name = button_body_names[goal_idx]
                parts = body_name.split("_")
                if len(parts) >= 3 and parts[0] == "button":
                    row = int(parts[1])
                    col = int(parts[2])
                    x = start_x + col * dx
                    y = start_y
                    z = start_z + row * dz
                    goal_pos_w = torch.tensor([[x, y, z]], device=agibot.device, dtype=torch.float32)  # (1, 3)
                    goal_quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=agibot.device, dtype=torch.float32)  # Identity quaternion (1, 4)
                    
                    # Expand to all envs for visualization
                    goal_pos_w_all = goal_pos_w.expand(scene.num_envs, -1)
                    goal_quat_w_all = goal_quat_w.expand(scene.num_envs, -1)
                    goal_marker.visualize(goal_pos_w_all, goal_quat_w_all)
            except (ValueError, IndexError):
                # Skip visualization if parsing fails
                pass

        # Debug output: Print EE position and button positions at the end of each period
        if count % period == period - 1:  # Last frame of the period
            # Get end effector position in world frame (after update)
            right_ee_w_final = agibot.data.body_pose_w[0, right_cfg.body_ids[0], :7]  # [x, y, z, qx, qy, qz, qw]
            right_ee_pos_w_np = right_ee_w_final[:3].cpu().numpy()
            
            print(f"\n[DEBUG] === Period End (goal_idx={goal_idx}, count={count}) ===")
            print(f"[DEBUG] Right End Effector Position (World Frame): [{right_ee_pos_w_np[0]:.4f}, {right_ee_pos_w_np[1]:.4f}, {right_ee_pos_w_np[2]:.4f}]")
            
            # Button positions based on Blender layout (same parameters as in get_ee_goals)
            start_x, start_y, start_z = -1.75, -0.92, 1.625
            dx, dz = -0.22, -0.178
            rows, cols = 4, 2
            
            # Compute button positions in world frame from Blender layout
            button_positions_w = []
            for body_name in button_body_names:
                # Parse row and col from button name (format: "button_<row>_<col>_link")
                try:
                    parts = body_name.split("_")
                    if len(parts) >= 3 and parts[0] == "button":
                        row = int(parts[1])
                        col = int(parts[2])
                        x = start_x + col * dx
                        y = start_y
                        z = start_z + row * dz
                        button_positions_w.append([x, y, z])
                    else:
                        button_positions_w.append([start_x, start_y, start_z])
                except (ValueError, IndexError):
                    button_positions_w.append([start_x, start_y, start_z])
            
            # Convert to numpy for printing
            button_positions_w_np = torch.tensor(button_positions_w, device=agibot.device, dtype=torch.float32).cpu().numpy()
            
            print(f"[DEBUG] Button Positions (World Frame - from Blender layout):")
            for i, body_name in enumerate(button_body_names):
                if i < len(button_positions_w_np):
                    pos = button_positions_w_np[i]
                    print(f"  {body_name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
            
            # Convert button positions to robot local frame (current robot pose)
            root_w = agibot.data.root_pose_w[0:1, :]  # (1, 7)
            robot_root_pos_w = root_w[:, :3]  # (1, 3)
            robot_root_quat_w = root_w[:, 3:7]  # (1, 4)
            
            button_positions_w_tensor = torch.tensor(button_positions_w, device=agibot.device, dtype=torch.float32)
            
            print(f"[DEBUG] Button Positions (Robot Local Frame):")
            for i, body_name in enumerate(button_body_names):
                if i < len(button_positions_w_tensor):
                    button_pos_w_i = button_positions_w_tensor[i:i+1, :]  # (1, 3)
                    button_quat_w_i = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=agibot.device)  # Identity quaternion
                    
                    button_pos_b, button_quat_b = subtract_frame_transforms(
                        robot_root_pos_w, robot_root_quat_w,
                        button_pos_w_i, button_quat_w_i,
                    )
                    pos_b = button_pos_b[0, :].cpu().numpy()
                    print(f"  {body_name}: [{pos_b[0]:.4f}, {pos_b[1]:.4f}, {pos_b[2]:.4f}]")
            
            # Get current EE position in robot local frame
            right_ee_pos_w = right_ee_w_final[:3].unsqueeze(0)  # (1, 3)
            right_ee_quat_w = right_ee_w_final[3:7].unsqueeze(0)  # (1, 4)
            
            right_ee_pos_b, right_ee_quat_b = subtract_frame_transforms(
                robot_root_pos_w, robot_root_quat_w,
                right_ee_pos_w, right_ee_quat_w,
            )
            right_ee_pos_b_np = right_ee_pos_b[0, :].cpu().numpy()
            
            # Print current goal position (robot local frame)
            if goal_idx < right_arm_goals.shape[0]:
                current_goal = right_arm_goals[goal_idx, :3].cpu().numpy()
                print(f"[DEBUG] Current IK Goal Position (Robot Local Frame): [{current_goal[0]:.4f}, {current_goal[1]:.4f}, {current_goal[2]:.4f}]")
                
                # Calculate error/distance between current EE and goal
                current_ee_pos_b_tensor = right_ee_pos_b[0, :]  # (3,)
                goal_pos_b_tensor = torch.tensor(current_goal, device=agibot.device, dtype=torch.float32)
                error = torch.norm(current_ee_pos_b_tensor - goal_pos_b_tensor).cpu().item()
                error_xyz = (current_ee_pos_b_tensor - goal_pos_b_tensor).cpu().numpy()
                
                print(f"[DEBUG] Current EE Position (Robot Local Frame): [{right_ee_pos_b_np[0]:.4f}, {right_ee_pos_b_np[1]:.4f}, {right_ee_pos_b_np[2]:.4f}]")
                print(f"[DEBUG] Position Error (Goal - Current): [{error_xyz[0]:.4f}, {error_xyz[1]:.4f}, {error_xyz[2]:.4f}]")
                print(f"[DEBUG] Distance to Goal: {error:.4f} m")
            
            print(f"[DEBUG] ===========================================\n")

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

    elevator_door_joint_names = ["door2_joint"]
    elevator_door_ids, _ = elevator.find_joints(elevator_door_joint_names)
    elevator_door_ids = torch.as_tensor(elevator_door_ids, device=elevator.device, dtype=torch.long)

    elevator_button_joint_names = ["button_0_0_joint", "button_0_1_joint", "button_1_0_joint", "button_1_1_joint", "button_2_0_joint", "button_2_1_joint", "button_3_0_joint", "button_3_1_joint"]
    elevator_button_ids, _ = elevator.find_joints(elevator_button_joint_names)
    elevator_button_ids = torch.as_tensor(elevator_button_ids, device=elevator.device, dtype=torch.long)

    # Setup IK controller for right arm
    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    )
    right_ik = DifferentialIKController(ik_cfg, scene.num_envs, agibot.device)

    # Setup right arm scene entity (for IK computation) - all joints [1-7]
    right_cfg = SceneEntityCfg(
        "agibot",
        joint_names=["right_arm_joint[1-7]"],
        body_names=["right_Right_Pad_Link"],
    )
    right_cfg.resolve(scene)

    if agibot.is_fixed_base:
        right_ee_jac = right_cfg.body_ids[0] - 1
    else:
        right_ee_jac = right_cfg.body_ids[0]

    # Get button goals (button body names need to be checked - using link names from ik_solver.py)
    button_body_names = ["button_0_0_link", "button_0_1_link", "button_1_0_link", "button_1_1_link", 
                         "button_2_0_link", "button_2_1_link", "button_3_0_link", "button_3_1_link"]
    right_arm_goals = get_ee_goals(elevator, agibot, button_body_names)

    # Setup marker for EE goal visualization
    frame_cfg = FRAME_MARKER_CFG.copy()
    frame_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)  # Make marker visible but not too large
    goal_marker = VisualizationMarkers(frame_cfg.replace(prim_path="/Visuals/right_goal"))

    # Ensure data is updated
    scene.update(sim.get_physics_dt())

    # Run the simulator
    run_simulator(
        sim, scene, agibot, elevator,
        elevator_door_ids, elevator_button_ids,
        right_ik, right_cfg, right_ee_jac,
        right_arm_goals,
        button_body_names,
        goal_marker,
        period=args_cli.period
    )

    simulation_app.close()


if __name__ == "__main__":
    main()