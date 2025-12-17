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

ELEVATOR_ASSET_PATH = "ElevatorManAssets/assets/Collected_rigid_body_flattened/rigid_body_flattened.usdc"

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
            "button_0_0_joint": 0.0,
            "button_0_1_joint": 0.0,
            "button_1_0_joint": 0.0,
            "button_1_1_joint": 0.0,
            "button_2_0_joint": 0.0,
            "button_2_1_joint": 0.0,
            "button_3_0_joint": 0.0,
            "button_3_1_joint": 0.0,
            "door1_joint": 0.0,
            "door2_joint": 0.0,
        },
        pos=(0.0, 0.0, 0.0),  # init pos of the articulation for teleop
    ),
    actuators={
        # Elevator buttons
        "elevator_buttons": ImplicitActuatorCfg(
            joint_names_expr=["button_[0-3]_[0-1]_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=1.0,
            stiffness=1000.0,
            damping=10.0,
        ),
        # Elevator doors
        "elevator_doors": ImplicitActuatorCfg(
            joint_names_expr=["door[1-2]_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=1.0,
            stiffness=5000.0,
            damping=50.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)