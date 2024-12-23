# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from omni.isaac.lab_assets.cartpole import CARTPOLE_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform


# holds settings for the simulation parameters, the scene, the actors and the task. This class
# also defines the number of actions and observations for the environment.
@configclass
class DoubleCartpoleEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    action_scale = 100.0  # [N]
    action_space = 1*2
    observation_space = 4*2
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.init_state.pos = (-0.3, 0, 1.5)
    robot_cfg2: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot2")
    robot_cfg2.init_state.pos = (0.3, 0, 2.5)
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # reset
    max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    initial_pole_angle_range = [-0.25, 0.25]  # the range in which the pole angle is sampled from on reset [rad]

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.1 # originally -0.01
    rew_scale_pole_vel = -0.005

class SingleAgent:
    def __init__(self, cfg: DoubleCartpoleEnvCfg, robot_cfg):
        self.cfg = cfg
        self.robot_cfg = robot_cfg

    def setup_scene(self):
        self.cartpole = Articulation(self.robot_cfg)

    def post_setup_scene(self):
        self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def apply_action(self) -> None:
        self.cartpole.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)

    def get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def get_rewards(self, reset_terminated) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            reset_terminated,
        )
        return total_reward

    def get_dones(self, episode_length_buf, max_episode_length) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        time_out = episode_length_buf >= max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        return out_of_bounds, time_out

    def reset_idx(self, env_ids: Sequence[int] | None, scene: InteractiveScene):
        joint_pos = self.cartpole.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.cartpole.data.default_joint_vel[env_ids]

        default_root_state = self.cartpole.data.default_root_state[env_ids]
        default_root_state[:, :3] += scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


class DoubleCartpoleEnv(DirectRLEnv):
    cfg: DoubleCartpoleEnvCfg

    def __init__(self, cfg: DoubleCartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        self.r1 = SingleAgent(cfg, cfg.robot_cfg)
        self.r2 = SingleAgent(cfg, cfg.robot_cfg2)
        super().__init__(cfg, render_mode, **kwargs)

        self.r1.post_setup_scene()
        self.r2.post_setup_scene()


    def _setup_scene(self):
        self.r1.setup_scene()
        self.r2.setup_scene()
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
        self.scene.articulations["cartpole1"] = self.r1.cartpole
        self.scene.articulations["cartpole2"] = self.r2.cartpole
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.r1.pre_physics_step(actions[:, :self.cfg.action_space//2])
        self.r2.pre_physics_step(actions[:, self.cfg.action_space//2:])

    def _apply_action(self) -> None:
        self.r1.apply_action()
        self.r2.apply_action()

    def _get_observations(self) -> dict:
        obs1 = self.r1.get_observations()
        obs2 = self.r2.get_observations()
        # observations = {"policy1": obs1, "policy2": obs2}
        observations = {"policy": torch.cat((obs1["policy"], obs2["policy"]), dim=1)}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward1 = self.r1.get_rewards(self.reset_terminated)
        total_reward2 = self.r2.get_rewards(self.reset_terminated)
        return total_reward1 + total_reward2

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        out_of_bounds1, time_out1 = self.r1.get_dones(self.episode_length_buf, self.max_episode_length)
        out_of_bounds2, time_out2 = self.r2.get_dones(self.episode_length_buf, self.max_episode_length)
        out_of_bounds = out_of_bounds1 | out_of_bounds2
        time_out = time_out1 | time_out2
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.r1.cartpole._ALL_INDICES
        super()._reset_idx(env_ids)

        self.r1.reset_idx(env_ids, self.scene)
        self.r2.reset_idx(env_ids, self.scene)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward
