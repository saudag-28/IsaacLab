# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Spawning two quadrupeds in interactive scene with walls

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/01_assets/run_rigid_object.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a rigid object.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)
from omni.isaac.lab.actuators import DCMotorCfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets.anymal import ANYMAL_B_CFG, ANYMAL_C_CFG, ANYMAL_D_CFG  # isort:skip
from omni.isaac.lab_assets.spot import SPOT_CFG  # isort:skip
from omni.isaac.lab_assets.unitree import UNITREE_A1_CFG, UNITREE_GO1_CFG, UNITREE_GO2_CFG  # isort:skip


@configclass
class QuadrupedSceneCfg(InteractiveSceneCfg):

    """Configuration for a quadruped scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # walls
    # object collection
    object_collection: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            "object_A": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Object_A",
                spawn=sim_utils.CuboidCfg(
                    size=(5.0, 0.1, 0.4),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=True,
                        kinematic_enabled=True,  # Make it a kinematic object
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.5, 0.2)),
            ),
            "object_B": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Object_B",
                spawn=sim_utils.CuboidCfg(
                    size=(5.0, 0.1, 0.4),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=True,
                        kinematic_enabled=True,  # Make it a kinematic object
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.5, 0.2)),
            ),
        }
    )

    # articulation
    unitree_go1_robot1 = ArticulationCfg(
                        prim_path="{ENV_REGEX_NS}/Robot1",
                        spawn=sim_utils.UsdFileCfg(
                            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go1/go1.usd",
                            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                    disable_gravity=False,
                                    retain_accelerations=False,
                                    linear_damping=0.0,
                                    angular_damping=0.0,
                                    max_linear_velocity=1000.0,
                                    max_angular_velocity=1000.0,
                                    max_depenetration_velocity=1.0,
                                ),
                                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                                    enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
                                ),
                        ),
                        init_state=ArticulationCfg.InitialStateCfg(
                        pos=(-2.0, 0.0, 0.6),
                        joint_pos={
                            ".*L_hip_joint": 0.1,
                            ".*R_hip_joint": -0.1,
                            "F[L,R]_thigh_joint": 0.8,
                            "R[L,R]_thigh_joint": 1.0,
                            ".*_calf_joint": -1.5,
                        },
                        joint_vel={".*": 0.0},
                    ),
                    soft_joint_pos_limit_factor=0.9,
                        actuators={
                            "base_legs": DCMotorCfg(
                                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                                effort_limit=33.5,
                                saturation_effort=33.5,
                                velocity_limit=21.0,
                                stiffness=25.0,
                                damping=0.5,
                                friction=0.0,
                            ),
                        },
        )
    unitree_go1_robot2 = ArticulationCfg(
                        prim_path="{ENV_REGEX_NS}/Robot2",
                        spawn=sim_utils.UsdFileCfg(
                            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go1/go1.usd",
                            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                    disable_gravity=False,
                                    retain_accelerations=False,
                                    linear_damping=0.0,
                                    angular_damping=0.0,
                                    max_linear_velocity=1000.0,
                                    max_angular_velocity=1000.0,
                                    max_depenetration_velocity=1.0,
                                ),
                                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                                    enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
                                ),
                        ),
                        init_state=ArticulationCfg.InitialStateCfg(
                        pos=(2.0, 0.0, 0.6),
                        rot = (0.0, 0.0, 0.0, 1.0),
                        joint_pos={
                            ".*L_hip_joint": 0.1,
                            ".*R_hip_joint": -0.1,
                            "F[L,R]_thigh_joint": 0.8,
                            "R[L,R]_thigh_joint": 1.0,
                            ".*_calf_joint": -1.5,
                        },
                        joint_vel={".*": 0.0},
                    ),
                    soft_joint_pos_limit_factor=0.9,
                        actuators={
                            "base_legs": DCMotorCfg(
                                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                                effort_limit=33.5,
                                saturation_effort=33.5,
                                velocity_limit=21.0,
                                stiffness=25.0,
                                damping=0.5,
                                friction=0.0,
                            ),
                        },
        )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.

    unitree_go1_robot1 = scene["unitree_go1_robot1"]
    unitree_go1_robot2 = scene["unitree_go1_robot2"]

    rigid_object_collection: RigidObjectCollection = scene["object_collection"]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world

            # robot
            root_state1 = unitree_go1_robot1.data.default_root_state.clone()
            root_state2 = unitree_go1_robot2.data.default_root_state.clone()

            # object collection
            object_state = rigid_object_collection.data.default_object_state.clone()
            object_state[..., :3] += scene.env_origins.unsqueeze(1)
            rigid_object_collection.write_object_state_to_sim(object_state)



            root_state1[:, :3] += scene.env_origins
            root_state2[:, :3] += scene.env_origins

            unitree_go1_robot1.write_root_state_to_sim(root_state1)
            unitree_go1_robot2.write_root_state_to_sim(root_state2)

            # set joint positions with some noise

            joint_pos1, joint_vel1 = unitree_go1_robot1.data.default_joint_pos.clone(), unitree_go1_robot1.data.default_joint_vel.clone()
            # joint_pos1 += torch.rand_like(joint_pos1) * 0.1
            unitree_go1_robot1.write_joint_state_to_sim(joint_pos1, joint_vel1)

            joint_pos2, joint_vel2 = unitree_go1_robot2.data.default_joint_pos.clone(), unitree_go1_robot2.data.default_joint_vel.clone()
            # joint_pos2 += torch.rand_like(joint_pos2) * 0.1
            unitree_go1_robot2.write_joint_state_to_sim(joint_pos2, joint_vel2)

            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply default action
        # joint positions
        joint_pos_target_1 = unitree_go1_robot1.data.default_joint_pos
        joint_pos_target_2 = unitree_go1_robot2.data.default_joint_pos

        # -- apply action to the robot
        unitree_go1_robot1.set_joint_position_target(joint_pos_target_1)
        unitree_go1_robot2.set_joint_position_target(joint_pos_target_2)

        # -- write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = QuadrupedSceneCfg(num_envs=args_cli.num_envs, env_spacing=8.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

