# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Spawning two quadrupeds

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/01_assets/run_rigid_object.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a rigid object.")
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
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext

##
# Pre-defined configs
##
from omni.isaac.lab_assets.anymal import ANYMAL_B_CFG, ANYMAL_C_CFG, ANYMAL_D_CFG  # isort:skip
from omni.isaac.lab_assets.spot import SPOT_CFG  # isort:skip
from omni.isaac.lab_assets.unitree import UNITREE_A1_CFG, UNITREE_GO1_CFG, UNITREE_GO2_CFG  # isort:skip


# adding all the prims to the scene
def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Create a group called "Origin0"
    origins = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]

    # This function basically sets the origin locations for the same prim to spawn multiple times, in this case the prim consists of two rigid objects
    # prim_utils.create_prim("/World/Origin1", "Xform", translation=origin)

    # We are adding two rigid objects to the scene
    # Articulation class contains information about the asset's spawning strategy, default inital state and other meta-information
    # Robot 1
    prim_utils.create_prim("/World/Origin0", "Xform", translation=origins[0])
    unitree_go1 = Articulation(UNITREE_GO1_CFG.replace(prim_path="/World/Origin0/Robot1"))

    # Robot 2
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[1])
    unitree_go2 = Articulation(UNITREE_GO1_CFG.replace(prim_path="/World/Origin1/Robot2"))

    # Return the scene information (entity) to the simulator to interact with the rigid object
    scene_entities = {"unitree_go1": unitree_go1, "unitree_go2": unitree_go2}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    unitree_go1_robot = entities["unitree_go1"]
    unitree_go2_robot = entities["unitree_go2"]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Simulate physics
    while simulation_app.is_running():
        # Reset every 250 steps
        if count % 250 == 0:
            sim_time = 0.0
            count = 0

            # Reset root states for both objects
            root_state1 = unitree_go1_robot.data.default_root_state.clone()
            root_state2 = unitree_go2_robot.data.default_root_state.clone()

            # Sample random positions and apply them
            root_state1[:, :3] += origins[0]
            root_state2[:, :3] += origins[1]
            
            unitree_go1_robot.write_root_state_to_sim(root_state1)
            unitree_go2_robot.write_root_state_to_sim(root_state2)

            # set joint positions with some noise
            joint_pos1, joint_vel1 = unitree_go1_robot.data.default_joint_pos.clone(), unitree_go2_robot.data.default_joint_vel.clone()
            joint_pos1 += torch.rand_like(joint_pos1) * 0.1
            unitree_go1_robot.write_joint_state_to_sim(joint_pos1, joint_vel1)

            joint_pos2, joint_vel2 = unitree_go2_robot.data.default_joint_pos.clone(), unitree_go2_robot.data.default_joint_vel.clone()
            joint_pos2 += torch.rand_like(joint_pos2) * 0.1
            unitree_go2_robot.write_joint_state_to_sim(joint_pos2, joint_vel2)

            # Reset buffers
            unitree_go1_robot.reset()
            unitree_go2_robot.reset()
            print("----------------------------------------")
            print("[INFO]: Resetting robot states...")

        # Apply default actions
        # -- generate random joint positions
        joint_pos_target = unitree_go1_robot.data.default_joint_pos + torch.randn_like(unitree_go1_robot.data.joint_pos) * 0.1
        joint_pos_target = unitree_go2_robot.data.default_joint_pos + torch.randn_like(unitree_go2_robot.data.joint_pos) * 0.1

        # -- apply action to the robot
        unitree_go1_robot.set_joint_position_target(joint_pos_target)
        unitree_go2_robot.set_joint_position_target(joint_pos_target)

        # Apply simulation data
        unitree_go1_robot.write_data_to_sim()
        unitree_go2_robot.write_data_to_sim()

        # Perform simulation step
        sim.step()

        # Update simulation time
        count += 1

        # Update buffers
        unitree_go1_robot.update(sim_dt)
        unitree_go2_robot.update(sim_dt)



def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[1.5, 0.0, 1.0], target=[0.0, 0.0, 0.0])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
