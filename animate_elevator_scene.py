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
parser.add_argument("--simultaneous_arms", action="store_true", help="Move both arms simultaneously. If not set, arms move sequentially (left first, then right)")
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
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, Articulation
from isaaclab.sim import SimulationContext
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from cfg.agibot import AGIBOT_A2D_CFG
from cfg.elevator import ELEVATOR_CFG

# NEW: USD access
import omni.usd
from pxr import UsdGeom, Gf, Usd

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
    elevator: ArticulationCfg = ELEVATOR_CFG.replace(prim_path="/World/Elevator/root")

    # robot
    agibot: ArticulationCfg = AGIBOT_A2D_CFG.replace(prim_path="/World/Agibot")

def set_robot_pose_demo(
    agibot: Articulation, 
    phase: float, 
    left_joint_groups: dict[str, torch.Tensor],
    right_joint_groups: dict[str, torch.Tensor],
    robot_animation_range: float = 1.0,
    symmetric_base: bool = True,
    sequential: bool = True,
    sequential_linkages: bool = True,
    cached_symmetric_refs: dict[str, torch.Tensor] = None,
    cached_symmetric_ref_all: torch.Tensor = None,
    lift_body_joint_ids: torch.Tensor = None,
    lift_body_target: torch.Tensor = None
):
    """Set robot joints based on phase for smooth animation.
    
    Args:
        agibot: The robot articulation
        phase: Normalized phase value [0, 1] for animation cycle
        left_joint_groups: Dict mapping group names to tensors of left arm joint indices (e.g., {"shoulder": [...], "forearm": [...], "gripper": [...]})
        right_joint_groups: Dict mapping group names to tensors of right arm joint indices
        robot_animation_range: Multiplier for animation range (default 1.0 = full 2π rotation)
        symmetric_base: If True, ensures symmetric starting positions for left and right arms
        sequential: If True, moves arms one at a time (left first, then right). If False, moves both simultaneously.
        sequential_linkages: If True, animates joint groups (shoulder, forearm, gripper) sequentially within each arm
    """
    # Get all joint IDs from groups for checking
    all_left_ids = torch.cat([ids for ids in left_joint_groups.values()]) if left_joint_groups else torch.tensor([], dtype=torch.long, device=agibot.device)
    all_right_ids = torch.cat([ids for ids in right_joint_groups.values()]) if right_joint_groups else torch.tensor([], dtype=torch.long, device=agibot.device)
    
    if len(all_left_ids) == 0 and len(all_right_ids) == 0:
        return
    
    # Calculate joint positions based on phase (smooth rotation)
    joint_pos_target = agibot.data.default_joint_pos.clone()
    
    # Get group names in order (shoulder, forearm, gripper)
    group_names = list(left_joint_groups.keys()) if left_joint_groups else list(right_joint_groups.keys())
    num_groups = len(group_names)
    
    def animate_joint_group(group_ids, group_phase, symmetric_ref_group=None, is_left=True):
        """Helper to animate a joint group based on phase"""
        if len(group_ids) == 0:
            return

        animation_offset = group_phase * (2 * torch.pi * robot_animation_range)
        if symmetric_ref_group is not None:
            if is_left:
                joint_pos_target[:, group_ids] = symmetric_ref_group + animation_offset
            else:
                joint_pos_target[:, group_ids] = symmetric_ref_group - animation_offset
        else:
            if is_left:
                joint_pos_target[:, group_ids] += animation_offset
            else:
                joint_pos_target[:, group_ids] -= animation_offset
    
    if symmetric_base and len(all_left_ids) > 0 and len(all_right_ids) > 0:
        # For symmetric motion: use average of left/right defaults as symmetric reference
        # Use cached references if provided, otherwise compute them
        if cached_symmetric_refs is not None and cached_symmetric_ref_all is not None:
            symmetric_refs = cached_symmetric_refs
            symmetric_ref_all = cached_symmetric_ref_all
        else:
            # Compute symmetric ref for each group separately
            symmetric_refs = {}
            for group_name in group_names:
                left_group_ids = left_joint_groups[group_name]
                right_group_ids = right_joint_groups[group_name]
                if len(left_group_ids) > 0 and len(right_group_ids) > 0:
                    left_default_group = agibot.data.default_joint_pos[:, left_group_ids]
                    right_default_group = agibot.data.default_joint_pos[:, right_group_ids]
                    symmetric_refs[group_name] = (left_default_group + right_default_group) / 2.0
            
            # Also compute for all joints together (for non-sequential-linkages case)
            left_default_all = agibot.data.default_joint_pos[:, all_left_ids]
            right_default_all = agibot.data.default_joint_pos[:, all_right_ids]
            symmetric_ref_all = (left_default_all + right_default_all) / 2.0
        
        if sequential:
            # Sequential movement: left arm moves first (phase 0-0.5), then right arm (phase 0.5-1.0)
            if phase < 0.5:
                left_phase = phase * 2.0  # Map [0, 0.5] to [0, 1]
                
                if sequential_linkages and num_groups > 0:
                    # Animate left arm groups sequentially
                    group_phase_range = 1.0 / num_groups
                    for i, group_name in enumerate(group_names):
                        group_start = i * group_phase_range
                        group_end = (i + 1) * group_phase_range
                        if left_phase >= group_end:
                            # This group is complete, set to final position
                            animate_joint_group(left_joint_groups[group_name], 1.0, symmetric_refs[group_name], is_left=True)
                        elif left_phase >= group_start:
                            # This group is currently animating
                            group_phase = (left_phase - group_start) / group_phase_range
                            animate_joint_group(left_joint_groups[group_name], group_phase, symmetric_refs[group_name], is_left=True)
                        else:
                            # This group hasn't started yet, keep at reference position
                            joint_pos_target[:, left_joint_groups[group_name]] = symmetric_refs[group_name]
                    
                    # Right arm stays at reference
                    for group_name in group_names:
                        joint_pos_target[:, right_joint_groups[group_name]] = symmetric_refs[group_name]
                else:
                    # Animate all left joints together
                    animation_offset = left_phase * (2 * torch.pi * robot_animation_range)
                    joint_pos_target[:, all_left_ids] = symmetric_ref_all + animation_offset
                    joint_pos_target[:, all_right_ids] = symmetric_ref_all
            else:
                right_phase = (phase - 0.5) * 2.0  # Map [0.5, 1.0] to [0, 1]
                
                # Left arm at final position
                left_final_offset = (2 * torch.pi * robot_animation_range)
                joint_pos_target[:, all_left_ids] = symmetric_ref_all + left_final_offset
                
                if sequential_linkages and num_groups > 0:
                    # Animate right arm groups sequentially
                    group_phase_range = 1.0 / num_groups
                    for i, group_name in enumerate(group_names):
                        group_start = i * group_phase_range
                        group_end = (i + 1) * group_phase_range
                        if right_phase >= group_end:
                            # This group is complete, set to final position
                            animate_joint_group(right_joint_groups[group_name], 1.0, symmetric_refs[group_name], is_left=False)
                        elif right_phase >= group_start:
                            # This group is currently animating
                            group_phase = (right_phase - group_start) / group_phase_range
                            animate_joint_group(right_joint_groups[group_name], group_phase, symmetric_refs[group_name], is_left=False)
                        else:
                            # This group hasn't started yet, keep at reference position
                            joint_pos_target[:, right_joint_groups[group_name]] = symmetric_refs[group_name]
                else:
                    # Animate all right joints together
                    animation_offset = right_phase * (2 * torch.pi * robot_animation_range)
                    joint_pos_target[:, all_right_ids] = symmetric_ref_all - animation_offset
        else:
            # Simultaneous movement: both arms move together
            if sequential_linkages and num_groups > 0:
                group_phase_range = 1.0 / num_groups
                for i, group_name in enumerate(group_names):
                    group_start = i * group_phase_range
                    group_end = (i + 1) * group_phase_range
                    if phase >= group_end:
                        # This group is complete, set to final position
                        animate_joint_group(left_joint_groups[group_name], 1.0, symmetric_refs[group_name], is_left=True)
                        animate_joint_group(right_joint_groups[group_name], 1.0, symmetric_refs[group_name], is_left=False)
                    elif phase >= group_start:
                        # This group is currently animating
                        group_phase = (phase - group_start) / group_phase_range
                        animate_joint_group(left_joint_groups[group_name], group_phase, symmetric_refs[group_name], is_left=True)
                        animate_joint_group(right_joint_groups[group_name], group_phase, symmetric_refs[group_name], is_left=False)
                    else:
                        # This group hasn't started yet, keep at reference position
                        joint_pos_target[:, left_joint_groups[group_name]] = symmetric_refs[group_name]
                        joint_pos_target[:, right_joint_groups[group_name]] = symmetric_refs[group_name]
            else:
                animation_offset = phase * (2 * torch.pi * robot_animation_range)
                joint_pos_target[:, all_left_ids] = symmetric_ref_all + animation_offset
                joint_pos_target[:, all_right_ids] = symmetric_ref_all - animation_offset
    else:
        # Non-symmetric base - simplified version (can be expanded if needed)
        animation_offset = phase * (2 * torch.pi * robot_animation_range) if not sequential else (phase * 2.0 if phase < 0.5 else (phase - 0.5) * 2.0) * (2 * torch.pi * robot_animation_range)
        if len(all_left_ids) > 0:
            joint_pos_target[:, all_left_ids] += animation_offset if phase < 0.5 or not sequential else (2 * torch.pi * robot_animation_range)
        if len(all_right_ids) > 0:
            joint_pos_target[:, all_right_ids] -= animation_offset if phase >= 0.5 or not sequential else 0
    
    # Apply lift_body animation if provided (this preserves arm joint targets)
    if lift_body_joint_ids is not None and lift_body_target is not None and len(lift_body_joint_ids) > 0:
        joint_pos_target[:, lift_body_joint_ids] = lift_body_target
    
    # Clamp to joint limits
    joint_pos_target = joint_pos_target.clamp_(
        agibot.data.soft_joint_pos_limits[..., 0], 
        agibot.data.soft_joint_pos_limits[..., 1]
    )
    agibot.set_joint_position_target(joint_pos_target)
    agibot.write_data_to_sim()

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    scene_cfg = ElevatorSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    # Access the robot articulation (because we used ArticulationCfg)
    agibot: Articulation = scene["agibot"]
    elevator: Articulation = scene["elevator"]

    # Setup robot joint animation - organize into groups: shoulder (1-3), forearm (4-5), gripper (6-7)
    left_shoulder_names = ["left_arm_joint1", "left_arm_joint2", "left_arm_joint3"]
    left_forearm_names = ["left_arm_joint4", "left_arm_joint5"]
    left_gripper_names = ["left_arm_joint6", "left_arm_joint7"]
    
    right_shoulder_names = ["right_arm_joint1", "right_arm_joint2", "right_arm_joint3"]
    right_forearm_names = ["right_arm_joint4", "right_arm_joint5"]
    right_gripper_names = ["right_arm_joint6", "right_arm_joint7"]
    
    # Find joint indices for each group
    left_shoulder_ids, _ = agibot.find_joints(left_shoulder_names)
    left_forearm_ids, _ = agibot.find_joints(left_forearm_names)
    left_gripper_ids, _ = agibot.find_joints(left_gripper_names)
    
    right_shoulder_ids, _ = agibot.find_joints(right_shoulder_names)
    right_forearm_ids, _ = agibot.find_joints(right_forearm_names)
    right_gripper_ids, _ = agibot.find_joints(right_gripper_names)
    
    # Organize into groups
    left_joint_groups = {
        "shoulder": torch.as_tensor(left_shoulder_ids, device=agibot.device, dtype=torch.long),
        "forearm": torch.as_tensor(left_forearm_ids, device=agibot.device, dtype=torch.long),
        "gripper": torch.as_tensor(left_gripper_ids, device=agibot.device, dtype=torch.long),
    }
    right_joint_groups = {
        "shoulder": torch.as_tensor(right_shoulder_ids, device=agibot.device, dtype=torch.long),
        "forearm": torch.as_tensor(right_forearm_ids, device=agibot.device, dtype=torch.long),
        "gripper": torch.as_tensor(right_gripper_ids, device=agibot.device, dtype=torch.long),
    }
    
    total_left = sum(len(ids) for ids in left_joint_groups.values())
    total_right = sum(len(ids) for ids in right_joint_groups.values())
    
    if total_left > 0 or total_right > 0:
        print(f"[INFO] Organized arm joints into groups:")
        print(f"  Left arm - Shoulder: {len(left_joint_groups['shoulder'])}, Forearm: {len(left_joint_groups['forearm'])}, Gripper: {len(left_joint_groups['gripper'])}")
        print(f"  Right arm - Shoulder: {len(right_joint_groups['shoulder'])}, Forearm: {len(right_joint_groups['forearm'])}, Gripper: {len(right_joint_groups['gripper'])}")
        
        # Debug: Check default positions for symmetry
        scene.update(sim.get_physics_dt())  # Ensure data is updated
        
        # Cache symmetric reference positions once (computed from default positions)
        # This ensures consistent symmetry across all animation cycles
        group_names = list(left_joint_groups.keys())
        cached_symmetric_refs = {}
        all_left_ids = torch.cat([ids for ids in left_joint_groups.values()])
        all_right_ids = torch.cat([ids for ids in right_joint_groups.values()])
        for group_name in group_names:
            left_group_ids = left_joint_groups[group_name]
            right_group_ids = right_joint_groups[group_name]
            if len(left_group_ids) > 0 and len(right_group_ids) > 0:
                left_default_group = agibot.data.default_joint_pos[:, left_group_ids]
                right_default_group = agibot.data.default_joint_pos[:, right_group_ids]
                cached_symmetric_refs[group_name] = (left_default_group + right_default_group) / 2.0
        cached_symmetric_ref_all = (agibot.data.default_joint_pos[:, all_left_ids] + agibot.data.default_joint_pos[:, all_right_ids]) / 2.0
    else:
        left_joint_groups = {}
        right_joint_groups = {}
        cached_symmetric_refs = {}
        cached_symmetric_ref_all = None
        all_left_ids = torch.tensor([], device=agibot.device, dtype=torch.long)
        all_right_ids = torch.tensor([], device=agibot.device, dtype=torch.long)
        print("[WARN] No arm joints found for animation. Robot will use default pose.")

    # Setup door2 mesh transform animation
    # DOOR2_PRIM_PATH = "/World/Elevator/root/Elevator/ElevatorRig/Door2/Cube_025"
    # stage = omni.usd.get_context().get_stage()
    # door2_prim = stage.GetPrimAtPath(DOOR2_PRIM_PATH)
    animate_elevator_joint_names = [ "door2_joint" ]
    animate_elevator_ids, _ = elevator.find_joints(animate_elevator_joint_names)
    animate_elevator_ids = torch.as_tensor(animate_elevator_ids, device=elevator.device, dtype=torch.long)
    if len(animate_elevator_ids) > 0:
        print(f"[INFO] Found door2_joint at index {animate_elevator_ids[0].item()}")
        print(f"[INFO] door2_joint default position: {elevator.data.default_joint_pos[0, animate_elevator_ids[0]].item():.6f}")
        print(f"[INFO] door2_joint joint type: {elevator.data.joint_types[animate_elevator_ids[0]].item() if hasattr(elevator.data, 'joint_types') else 'N/A'}")
    else:
        print("[ERROR] door2_joint not found!")
    
    # Setup robot's joint_lift_body prismatic joint animation (for testing/reference)
    lift_body_joint_names = ["joint_lift_body"]
    lift_body_joint_ids, _ = agibot.find_joints(lift_body_joint_names)
    lift_body_joint_ids = torch.as_tensor(lift_body_joint_ids, device=agibot.device, dtype=torch.long)
    if len(lift_body_joint_ids) > 0:
        print(f"[INFO] Found joint_lift_body joint for testing prismatic joint movement")
    
    # if not door2_prim.IsValid():
    #     raise RuntimeError(f"Door2 prim not found at: {DOOR2_PRIM_PATH}")

    # door2_xform = UsdGeom.XformCommonAPI(door2_prim)

    # Cache initial transform (we'll only offset translation)
    # tc = Usd.TimeCode.Default()
    # init_t, init_r, init_s, init_pivot, _ = door2_xform.GetXformVectors(tc)
    # init_t = Gf.Vec3d(init_t)  # make sure it's a Vec3d
    # print("[INFO] Door2 initial translate:", init_t)

    # Animation parameters
    count = 0
    period = 500
    open_delta = -0.5  # 50 cm along chosen axis
    close_delta = 0.0
    robot_animation_range = args_cli.robot_animation_range
    lift_body_range = 0.2  # Range for joint_lift_body animation (meters)

    print("[INFO] Done. Close the window to exit.")
    while simulation_app.is_running():
        # Reset robot to default positions at the start of each period to maintain symmetry
        if count % period == 0:
            # Reset joint positions to default
            agibot.write_joint_state_to_sim(
                agibot.data.default_joint_pos.clone(),
                agibot.data.default_joint_vel.clone()
            )
            agibot.reset()
            count = 0
        
        # Calculate phase for animations
        phase = count % period
        alpha = phase / max(1, period - 1)  # Normalized phase [0, 1]
        
        # Calculate door animation delta based on phase
        if phase < 100:        # opening
            t = phase / 99.0
            delta = close_delta + t * (open_delta - close_delta)
        elif phase < 400:      # hold open
            delta = open_delta
        else:                  # closing
            t = (phase - 400) / 99.0
            delta = open_delta + t * (close_delta - open_delta)

        # Update door position using joint-based animation
        joint_pos_target = elevator.data.default_joint_pos.clone()
        joint_pos_target[:, animate_elevator_ids] += delta
        # Temporarily removed clamping to diagnose movement issues
        # joint_pos_target = joint_pos_target.clamp_(
        #     elevator.data.soft_joint_pos_limits[..., 0], elevator.data.soft_joint_pos_limits[..., 1]
        # )
        elevator.set_joint_position_target(joint_pos_target)
        elevator.write_data_to_sim()
        
        # Calculate lift_body animation target (same phase-based pattern as door)
        lift_body_target = None
        if len(lift_body_joint_ids) > 0:
            # Apply lift_body animation: default position + scaled delta (same pattern as door)
            lift_body_target = agibot.data.default_joint_pos[:, lift_body_joint_ids] + (delta * lift_body_range)
        
        # Update robot pose using phase-based animation (with sequential linkage movement)
        # This includes both arm animation and lift_body animation, set together to avoid overwriting
        set_robot_pose_demo(
            agibot, alpha, left_joint_groups, right_joint_groups, robot_animation_range, 
            sequential=not args_cli.simultaneous_arms, sequential_linkages=True,
            cached_symmetric_refs=cached_symmetric_refs,
            cached_symmetric_ref_all=cached_symmetric_ref_all,
            lift_body_joint_ids=lift_body_joint_ids if len(lift_body_joint_ids) > 0 else None,
            lift_body_target=lift_body_target
        )
        
        if len(lift_body_joint_ids) > 0 and count % 20 == 0:
            print(f"[lift_body] target: {lift_body_target[0, 0].item():.4f}")

        sim.step()
        scene.update(sim.get_physics_dt())
        
        # Debug door2 joint: compare target vs actual
        if count % 20 == 0:
            # Get actual joint data after physics update
            actual_pos = elevator.data.joint_pos[:, animate_elevator_ids]
            actual_vel = elevator.data.joint_vel[:, animate_elevator_ids]
            target_pos = joint_pos_target[:, animate_elevator_ids]
            
            print(f"[door2] target: {target_pos[0, 0].item():.6f}, actual: {actual_pos[0, 0].item():.6f}, vel: {actual_vel[0, 0].item():.6f}, diff: {(target_pos[0, 0] - actual_pos[0, 0]).item():.6f}")
            
            # Also print joint limits to see if we're hitting them
            joint_limits_lower = elevator.data.soft_joint_pos_limits[0, animate_elevator_ids, 0]
            joint_limits_upper = elevator.data.soft_joint_pos_limits[0, animate_elevator_ids, 1]
            print(f"[door2] limits: [{joint_limits_lower[0].item():.6f}, {joint_limits_upper[0].item():.6f}]")
        
        count += 1

    simulation_app.close()


if __name__ == "__main__":
    main()