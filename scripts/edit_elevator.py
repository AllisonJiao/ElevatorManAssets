import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

ELEVATOR_ASSET_PATH = "ElevatorManAssets/assets/elevator/elevator.usd"
ROBOT_ASSET_PATH = None  # optional: if you want to reference robot USD directly too

def design_scene():
    # Ground
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Light
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Elevator: just add the USD to the stage
    cfg = sim_utils.UsdFileCfg(usd_path=ELEVATOR_ASSET_PATH)
    cfg.func("/World/Elevator", cfg)

def main():
    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    design_scene()

    # Important: reset initializes the stage and lets the viewport show it.
    sim.reset()
    print("[INFO] Stage loaded. You can now author rigid bodies/joints in the GUI.")

    # Keep app alive
    while simulation_app.is_running():
        sim.render()

if __name__ == "__main__":
    main()
    simulation_app.close()
