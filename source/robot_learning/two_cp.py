# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Spawning two cartpoles

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
from omni.isaac.lab_assets import CARTPOLE_CFG  # isort:skip


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
    origin = [0.0, 0.0, 0.0]

    # This function basically sets the origin locations for the same prim to spawn multiple times, in this case the prim consists of two rigid objects
    prim_utils.create_prim("/World/Origin0", "Xform", translation=origin)

    # We are adding two rigid objects to the scene
    # RigitObjecCfg class contains information about the asset's spawning strategy, default inital state and other meta-information
    # Rigid Object 1
    cartpole1_cfg = CARTPOLE_CFG.copy()
    cartpole1_cfg.prim_path = "/World/Origin0/Cartpole1"
    cartpole1 = Articulation(cfg=cartpole1_cfg)

    # Rigid Object 2
    cartpole2_cfg = CARTPOLE_CFG.copy()
    cartpole2_cfg.prim_path = "/World/Origin0/Cartpole2"
    cartpole2 = Articulation(cfg=cartpole2_cfg)

    # Return the scene information (entity) to the simulator to interact with the rigid object
    scene_entities = {"cartpole1": cartpole1, "cartpole2": cartpole2}
    return scene_entities, origin



def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    cartpole1_robot = entities["cartpole1"]
    cartpole2_robot = entities["cartpole2"]

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
            root_state1 = cartpole1_robot.data.default_root_state.clone()
            root_state2 = cartpole2_robot.data.default_root_state.clone()

            # Sample random positions and apply them
            root_state1[:, :3] += origins
            root_state2[:, :3] += origins
            
            cartpole1_robot.write_root_state_to_sim(root_state1)
            cartpole2_robot.write_root_state_to_sim(root_state2)

            # set joint positions with some noise
            joint_pos1, joint_vel1 = cartpole1_robot.data.default_joint_pos.clone(), cartpole1_robot.data.default_joint_vel.clone()
            joint_pos1 += torch.rand_like(joint_pos1) * 0.1
            cartpole1_robot.write_joint_state_to_sim(joint_pos1, joint_vel1)

            joint_pos2, joint_vel2 = cartpole2_robot.data.default_joint_pos.clone(), cartpole2_robot.data.default_joint_vel.clone()
            joint_pos2 += torch.rand_like(joint_pos2) * 0.1
            cartpole2_robot.write_joint_state_to_sim(joint_pos2, joint_vel2)

            # Reset buffers
            cartpole1_robot.reset()
            cartpole2_robot.reset()
            print("----------------------------------------")
            print("[INFO]: Resetting robot states...")

        # Apply random action
        # -- generate random joint efforts
        efforts1 = torch.randn_like(cartpole1_robot.data.joint_pos) * 5.0
        efforts2 = torch.randn_like(cartpole2_robot.data.joint_pos) * 5.0
        # -- apply action to the robot
        cartpole1_robot.set_joint_effort_target(efforts1)
        cartpole2_robot.set_joint_effort_target(efforts2)

        # Apply simulation data
        cartpole1_robot.write_data_to_sim()
        cartpole2_robot.write_data_to_sim()

        # Perform simulation step
        sim.step()

        # Update simulation time
        count += 1

        # Update buffers
        cartpole1_robot.update(sim_dt)
        cartpole2_robot.update(sim_dt)



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
