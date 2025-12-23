# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import torch

from isaaclab.app import AppLauncher
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from cfg.agibot import AGIBOT_A2D_CFG
from cfg.elevator import ELEVATOR_CFG


# --------------------------------------------------
# App launcher
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# --------------------------------------------------
# Scene setup
# --------------------------------------------------
def design_scene():
    # Ground
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", None)

    # Light
    sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75)
    ).func("/World/Light", None)

    # ---------------- Elevator ----------------
    elevator_cfg = ELEVATOR_CFG.copy()
    elevator_cfg.prim_path = "/World/elevator"   # articulation root
    elevator = Articulation(cfg=elevator_cfg)

    # ---------------- Robot -------------------
    agibot_cfg = AGIBOT_A2D_CFG.copy()
    agibot_cfg.prim_path = "/World/Robot"
    agibot = Articulation(cfg=agibot_cfg)

    return {
        "elevator": elevator,
        "agibot": agibot,
    }


# --------------------------------------------------
# Simulation loop
# --------------------------------------------------
def run_simulator(sim: SimulationContext, entities: dict[str, Articulation]):
    elevator = entities["elevator"]
    agibot = entities["agibot"]

    sim_dt = sim.get_physics_dt()

    # ==================================================
    # DEBUG: articulation + joint sanity checks
    # ==================================================
    print("\n========== DEBUG: Elevator ==========")
    print("Joint names:", elevator.data.joint_names)
    print("Default joint pos:", elevator.data.default_joint_pos)
    print("Soft joint limits:", elevator.data.soft_joint_pos_limits)
    print("=====================================\n")

    assert "door2_joint" in elevator.data.joint_names, \
        "door2_joint NOT found in elevator articulation!"

    door2_id = elevator.data.joint_names.index("door2_joint")
    door2_id = torch.tensor([door2_id], device=elevator.device)

    # --------------------------------------------------
    # Motion parameters
    # --------------------------------------------------
    open_pos = -0.5     # meters
    close_pos = 0.0
    period = 500
    count = 0

    while simulation_app.is_running():

        # --------------------------------------------------
        # Reset robot occasionally (keep it quiet)
        # --------------------------------------------------
        if count % period == 0:
            root = agibot.data.default_root_state.clone()
            agibot.write_root_pose_to_sim(root[:, :7])
            agibot.write_root_velocity_to_sim(root[:, 7:])
            agibot.reset()

            print("\n[INFO] Reset robot")

        # --------------------------------------------------
        # Compute door target (triangle wave)
        # --------------------------------------------------
        phase = count % period

        if phase < 100:
            t = phase / 99.0
            door_target = close_pos + t * (open_pos - close_pos)
        elif phase < 400:
            door_target = open_pos
        else:
            t = (phase - 400) / 99.0
            door_target = open_pos + t * (close_pos - open_pos)

        # --------------------------------------------------
        # Drive door — NO clamping (debug mode)
        # --------------------------------------------------
        joint_pos_target = elevator.data.default_joint_pos.clone()
        joint_pos_target[:, door2_id] = door_target

        print(f"[DEBUG] door2 target = {door_target:.3f}")

        elevator.set_joint_position_target(joint_pos_target)

        # --------------------------------------------------
        # Write & step
        # --------------------------------------------------
        elevator.write_data_to_sim()
        agibot.write_data_to_sim()

        sim.step()

        elevator.update(sim_dt)
        agibot.update(sim_dt)

        count += 1


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view(
        [2.5, 0.0, 4.0],
        [0.0, 0.0, 2.0]
    )

    entities = design_scene()
    sim.reset()

    print("\n[INFO] Simulation ready\n")
    run_simulator(sim, entities)


if __name__ == "__main__":
    main()
    simulation_app.close()
