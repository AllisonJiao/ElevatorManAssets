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

def set_robot_pose_demo(agibot: Articulation):
    """Example: set some robot joints to fixed values instantly."""
    # TODO: replace these names with YOUR robot's joint names
    # Print available joints once:
    print("\n[INFO] Robot joint names:")
    for i, n in enumerate(agibot.joint_names):
        print(f"  {i:02d}: {n}")

    # Example targets (radians)
    joint_targets = {
        # "left_shoulder_pitch": 0.6,
        # "left_elbow": -1.0,
        # "right_shoulder_pitch": 0.6,
        # "right_elbow": -1.0,
    }

    if len(joint_targets) == 0:
        print("\n[WARN] joint_targets is empty. Fill in joint names from the printed list.")
        return

    set_articulation_joints_by_name(agibot, joint_targets)
    print("[INFO] Wrote robot joint state directly.")

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    scene_cfg = ElevatorSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    # Access the robot articulation (because we used ArticulationCfg)
    agibot: Articulation = scene["agibot"]

    # Make sure buffers are valid
    scene.update(sim.get_physics_dt())

    # 1) Instantly set joint positions once
    set_robot_pose_demo(agibot)

    # 2) Step a few frames only to refresh viewport (not "simulating" motion)
    for _ in range(5):
        sim.step()
        scene.update(sim.get_physics_dt())

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

    print("[INFO] Done. Close the window to exit.")
    while simulation_app.is_running():
        # Calculate door animation delta based on phase
        phase = count % period
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

        if count % 20 == 0:
            print(f"[door2-mesh] delta={delta:+.4f} translate={new_t}")

        sim.step()
        scene.update(sim.get_physics_dt())
        count += 1

    simulation_app.close()


if __name__ == "__main__":
    main()