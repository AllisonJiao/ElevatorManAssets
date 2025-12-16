# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Elevator.

The following configurations are available:

* :obj:`ELEVATOR`: Elevator


"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

ELEVATOR_ASSET_PATH = "ElevatorManAssets/assets/elevator_rigged.usdc"

##
# Configuration
##

ELEVATOR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ELEVATOR_ASSET_PATH}",
        activate_contact_sensors=False, # Temp set to False
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint_ElevatorButton_0_0": 0.0,
            "joint_ElevatorButton_0_1": 0.0,
            "joint_ElevatorButton_1_0": 0.0,
            "joint_ElevatorButton_1_1": 0.0,
            "joint_ElevatorButton_2_0": 0.0,
            "joint_ElevatorButton_2_1": 0.0,
            "joint_ElevatorButton_3_0": 0.0,
            "joint_ElevatorButton_3_1": 0.0,
            "joint_Door1": 0.0,
            "joint_Door2": 0.0,
        },
        pos=(0.0, 0.0, 0.0),  # init pos of the articulation for teleop
    ),
    actuators={
        # Elevator buttons
        "elevator_buttons": ImplicitActuatorCfg(
            joint_names_expr=["joint_ElevatorButton_[0-3]_[0-1]"],
            effort_limit_sim=10.0,
            velocity_limit_sim=1.0,
            stiffness=1000.0,
            damping=10.0,
        ),
        # Elevator doors
        "elevator_doors": ImplicitActuatorCfg(
            joint_names_expr=["joint_Door[1-2]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=1.0,
            stiffness=5000.0,
            damping=50.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)