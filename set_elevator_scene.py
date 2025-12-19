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
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, Articulation
from isaaclab.sim import SimulationContext
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from cfg.agibot import AGIBOT_A2D_CFG

# NEW: USD access
import omni.usd
from pxr import UsdGeom, Gf, Usd

ELEVATOR_ASSET_PATH = "ElevatorManAssets/assets/Collected_elevator_asset_tmp/elevator_asset.usdc"

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
    elevator = AssetBaseCfg(
        prim_path="/World/Elevator/root",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ELEVATOR_ASSET_PATH
        ),
    )

    # robot
    agibot: ArticulationCfg = AGIBOT_A2D_CFG.replace(prim_path="/World/Agibot")

def set_articulation_joints_by_name(
    art: Articulation,
    joint_targets: dict[str, float],
    env_ids: torch.Tensor | None = None,
):
    """Directly overwrite joint positions (q) in the simulator for selected joints.

    joint_targets: {"joint_name": target_position_in_radians_or_meters, ...}
    """
    if env_ids is None:
        env_ids = torch.arange(art.num_envs, device=art.device)

    # Resolve joint indices
    joint_names = list(joint_targets.keys())
    joint_ids, _ = art.find_joints(joint_names)
    if len(joint_ids) != len(joint_names):
        missing = set(joint_names) - set([art.joint_names[i] for i in joint_ids])
        raise RuntimeError(f"Some joints were not found: {missing}")

    # Read current full joint state
    q = art.data.joint_pos.clone()
    qd = art.data.joint_vel.clone()

    # Overwrite only the selected joints for selected envs
    target_vals = torch.tensor(
        [joint_targets[n] for n in joint_names],
        device=art.device,
        dtype=q.dtype,
    )
    q[env_ids[:, None], joint_ids[None, :]] = target_vals[None, :]
    qd[env_ids[:, None], joint_ids[None, :]] = 0.0

    # Write back to simulator immediately (this is the "no simulation" teleport)
    art.write_joint_state_to_sim(q, qd, env_ids=env_ids)

def set_robot_pose_demo(
    agibot: Articulation, 
    phase: float, 
    left_joint_ids: torch.Tensor,
    right_joint_ids: torch.Tensor,
    robot_animation_range: float = 1.0,
    symmetric_base: bool = True
):
    """Set robot joints based on phase for smooth animation.
    
    Args:
        agibot: The robot articulation
        phase: Normalized phase value [0, 1] for animation cycle
        left_joint_ids: Tensor of left arm joint indices to animate (rotates opposite direction)
        right_joint_ids: Tensor of right arm joint indices to animate
        robot_animation_range: Multiplier for animation range (default 1.0 = full 2π rotation)
        symmetric_base: If True, ensures symmetric starting positions for left and right arms
    """
    if len(left_joint_ids) == 0 and len(right_joint_ids) == 0:
        return
    
    # Calculate joint positions based on phase (smooth rotation)
    joint_pos_target = agibot.data.default_joint_pos.clone()
    
    # Calculate animation offset
    animation_offset = phase * (2 * torch.pi * robot_animation_range)
    
    if symmetric_base and len(left_joint_ids) == len(right_joint_ids) and len(left_joint_ids) > 0:
        # For symmetric motion: use average of left/right defaults as symmetric reference
        left_default = agibot.data.default_joint_pos[:, left_joint_ids]
        right_default = agibot.data.default_joint_pos[:, right_joint_ids]
        symmetric_ref = (left_default + right_default) / 2.0
        
        # Apply symmetric animation: left moves negative, right moves positive from reference
        joint_pos_target[:, left_joint_ids] = symmetric_ref - animation_offset
        joint_pos_target[:, right_joint_ids] = symmetric_ref + animation_offset
    else:
        # Apply animation to left joints (negative direction)
        if len(left_joint_ids) > 0:
            joint_pos_target[:, left_joint_ids] -= animation_offset
        
        # Apply animation to right joints (positive direction)
        if len(right_joint_ids) > 0:
            joint_pos_target[:, right_joint_ids] += animation_offset
    
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

    # Setup robot joint animation - find left and right arm joints
    left_joint_names = [
        "left_arm_joint[1-3]"
    ]
    right_joint_names = [
        "right_arm_joint[1-3]"
    ]
    
    left_joint_ids, _ = agibot.find_joints(left_joint_names)
    right_joint_ids, _ = agibot.find_joints(right_joint_names)
    
    if len(left_joint_ids) > 0 or len(right_joint_ids) > 0:
        left_joint_ids = torch.as_tensor(left_joint_ids, device=agibot.device, dtype=torch.long)
        right_joint_ids = torch.as_tensor(right_joint_ids, device=agibot.device, dtype=torch.long)
        print(f"[INFO] Animating {len(left_joint_ids)} left arm joints and {len(right_joint_ids)} right arm joints")
        print(f"  Left joints: {left_joint_names}")
        print(f"  Right joints: {right_joint_names}")
        
        # Debug: Check default positions for symmetry
        scene.update(sim.get_physics_dt())  # Ensure data is updated
        if len(left_joint_ids) == len(right_joint_ids) and len(left_joint_ids) > 0:
            left_default = agibot.data.default_joint_pos[0, left_joint_ids]
            right_default = agibot.data.default_joint_pos[0, right_joint_ids]
            print(f"[INFO] Default positions - Left: {left_default.cpu().numpy()}")
            print(f"[INFO] Default positions - Right: {right_default.cpu().numpy()}")
    else:
        left_joint_ids = torch.tensor([], device=agibot.device, dtype=torch.long)
        right_joint_ids = torch.tensor([], device=agibot.device, dtype=torch.long)
        print("[WARN] No arm joints found for animation. Robot will use default pose.")

    # Setup door2 mesh transform animation
    DOOR2_PRIM_PATH = "/World/Elevator/root/Elevator/ElevatorRig/Door2/Cube_025"
    stage = omni.usd.get_context().get_stage()
    door2_prim = stage.GetPrimAtPath(DOOR2_PRIM_PATH)
    
    if not door2_prim.IsValid():
        raise RuntimeError(f"Door2 prim not found at: {DOOR2_PRIM_PATH}")

    door2_xform = UsdGeom.XformCommonAPI(door2_prim)

    # Cache initial transform (we'll only offset translation)
    tc = Usd.TimeCode.Default()
    init_t, init_r, init_s, init_pivot, _ = door2_xform.GetXformVectors(tc)
    init_t = Gf.Vec3d(init_t)  # make sure it's a Vec3d
    print("[INFO] Door2 initial translate:", init_t)

    # Animation parameters
    count = 0
    period = 500
    open_delta = -0.5  # 5 cm along chosen axis
    close_delta = 0.0
    robot_animation_range = 0.5  # Reduce this value to make robot animation smaller (0.5 = half range, 1.0 = full 2π)

    print("[INFO] Done. Close the window to exit.")
    while simulation_app.is_running():
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

        # Update door position
        new_t = Gf.Vec3d(init_t[0] + delta, init_t[1], init_t[2])
        door2_xform.SetTranslate(new_t, Usd.TimeCode.Default())

        # Update robot pose using phase-based animation (left and right together)
        set_robot_pose_demo(agibot, alpha, left_joint_ids, right_joint_ids, robot_animation_range)

        if count % 20 == 0:
            print(f"[door2-mesh] delta={delta:+.4f} translate={new_t}")

        sim.step()
        scene.update(sim.get_physics_dt())
        count += 1

    simulation_app.close()


if __name__ == "__main__":
    main()