# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Spawning two cones

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
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.sim import SimulationContext


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
    cone1_cfg = RigidObjectCfg(
        prim_path="/World/Origin0/Cone1",
        spawn=sim_utils.ConeCfg(
            radius=0.1,
            height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    cone1 = RigidObject(cfg=cone1_cfg)

    # Rigid Object 2
    cone2_cfg = RigidObjectCfg(
        prim_path="/World/Origin0/Cone2",  # Different path but same location
        spawn=sim_utils.ConeCfg(
            radius=0.1,
            height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.5),  # Different color for clarity
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    cone2 = RigidObject(cfg=cone2_cfg)

    # Return the scene information (entity) to the simulator to interact with the rigid object
    scene_entities = {"cone1": cone1, "cone2": cone2}
    return scene_entities, origin



def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    cone1_object = entities["cone1"]
    cone2_object = entities["cone2"]

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
            root_state1 = cone1_object.data.default_root_state.clone()
            root_state2 = cone2_object.data.default_root_state.clone()

            # Sample random positions and apply them
            root_state1[:, :3] += origins
            root_state2[:, :3] += origins
            root_state1[:, :3] += math_utils.sample_cylinder(radius=0.1, h_range=(0.25, 0.5), size=cone1_object.num_instances, device=cone1_object.device)
            root_state2[:, :3] += math_utils.sample_cylinder(radius=0.1, h_range=(0.25, 0.5), size=cone2_object.num_instances, device=cone2_object.device)

            # Write root states to simulation
            cone1_object.write_root_state_to_sim(root_state1)
            cone2_object.write_root_state_to_sim(root_state2)

            # Reset buffers
            cone1_object.reset()
            cone2_object.reset()
            print("----------------------------------------")
            print("[INFO]: Resetting object states...")

        # Apply simulation data
        cone1_object.write_data_to_sim()
        cone2_object.write_data_to_sim()

        # Perform simulation step
        sim.step()

        # Update simulation time
        sim_time += sim_dt
        count += 1

        # Update buffers
        cone1_object.update(sim_dt)
        cone2_object.update(sim_dt)

        # Print root positions every 50 steps
        if count % 50 == 0:
            print(f"Cone1 Root position (in world): {cone1_object.data.root_state_w[:, :3]}")
            print(f"Cone2 Root position (in world): {cone2_object.data.root_state_w[:, :3]}")



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
