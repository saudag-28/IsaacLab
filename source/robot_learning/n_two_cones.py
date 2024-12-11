# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Spawning multiple 2 cones

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
    """Designs the scene with multiple rigid object instances."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Define multiple spawn origins
    origins = [[0.25, 0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]]

    # Initialize dictionary to store entities
    scene_entities = {}

    for i, origin in enumerate(origins):
        # Create a group for each origin
        group_path = f"/World/Origin{i}"
        prim_utils.create_prim(group_path, "Xform", translation=origin)

        # Cone1
        cone1_cfg = RigidObjectCfg(
            prim_path=f"{group_path}/Cone1",
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

        # Cone2
        cone2_cfg = RigidObjectCfg(
            prim_path=f"{group_path}/Cone2",
            spawn=sim_utils.ConeCfg(
                radius=0.1,
                height=0.2,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.5),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(),
        )
        cone2 = RigidObject(cfg=cone2_cfg)

        # Store objects in the entities dictionary
        scene_entities[f"cone1_{i}"] = cone1
        scene_entities[f"cone2_{i}"] = cone2

    # Return the scene information
    return scene_entities, origins




def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject], origins: torch.Tensor):
    """Runs the simulation loop."""
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

            # Iterate through all entities
            for key, obj in entities.items():
                # Reset root state
                root_state = obj.data.default_root_state.clone()

                # Apply positions based on origins
                idx = int(key.split("_")[-1])  # Extract origin index from the key
                root_state[:, :3] += origins[idx]
                root_state[:, :3] += math_utils.sample_cylinder(
                    radius=0.1, h_range=(0.25, 0.5), size=obj.num_instances, device=obj.device
                )

                # Write root state to simulation
                obj.write_root_state_to_sim(root_state)

                # Reset buffers
                obj.reset()

            print("----------------------------------------")
            print("[INFO]: Resetting object states...")

        # Apply simulation data and step for each object
        for obj in entities.values():
            obj.write_data_to_sim()
            obj.update(sim_dt)

        # Perform simulation step
        sim.step()
        sim_time += sim_dt
        count += 1

        # Print positions every 50 steps
        if count % 50 == 0:
            for key, obj in entities.items():
                print(f"{key} Root position (in world): {obj.data.root_state_w[:, :3]}")



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
