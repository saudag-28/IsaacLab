##### RVMod! #######
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.sensors import ContactSensor, RayCaster
from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene

from .double_anymal_c_env_cfg import DoubleAnymalCFlatEnvCfg

class SingleAgent:
    def __init__(self, cfg: DoubleAnymalCFlatEnvCfg, robot_cfg, contact_cfg, num_envs: int):
        self.cfg = cfg
        self.robot_cfg = robot_cfg
        self.contact_cfg = contact_cfg
        self.num_envs = num_envs

    def setup_scene(self):
        self._robot = Articulation(self.robot_cfg)
        self._contact_sensor = ContactSensor(self.contact_cfg)

    def post_setup_scene(self, device: torch.device, action_dim, step_dt: float):
        self.device = device
        # self.single_action_space = single_action_space
        self.action_dim = action_dim
        self.step_dt = step_dt

        # Joint position command (deviation from default joint positions)
        # self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        # self._previous_actions = torch.zeros(
        #     self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        # )
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, action_dim, device=self.device)

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*FOOT")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*THIGH")
        pass

    def pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        height_data = None
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    height_data,
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids] # Ignore this red squiggly
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        # undersired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        ) # Ignore this red squiggly
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undersired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward
    
    def get_dones(self) -> torch.Tensor:
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1) # Ignore this red squiggly
        return died

    def reset_idx(self, env_ids: torch.Tensor | None, scene: InteractiveScene):
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, :3] += scene.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids) # Ignore this red squiggly
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids) # Ignore this red squiggly
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids) # Ignore this red squiggly

    def get_robot(self):
        return self._robot


class DoubleAnymalCFlatEnv(DirectRLEnv):
    cfg: DoubleAnymalCFlatEnvCfg

    def __init__(self, cfg: DoubleAnymalCFlatEnvCfg, render_mode: str | None = None, **kwargs):
        # num_envs = 4096
        num_envs = cfg.scene.num_envs
        self._r1 = SingleAgent(cfg, cfg.robot1, cfg.contact_sensor1, num_envs)
        self._r2 = SingleAgent(cfg, cfg.robot2, cfg.contact_sensor2, num_envs)
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        # self._previous_actions = torch.zeros(
        #     self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        # )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
            ]
        }

        # Get specific body indices
        self._r1.post_setup_scene(self.device, self.cfg.action_space // 2, self.step_dt) # Ignore this red squiggly
        self._r2.post_setup_scene(self.device, self.cfg.action_space // 2, self.step_dt) # Ignore this red squiggly
        # self._base_id, _ = self._contact_sensor.find_bodies("base")
        # self._feet_ids, _ = self._contact_sensor.find_bodies(".*FOOT")
        # self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*THIGH")

    def _setup_scene(self):
        self._r1.setup_scene()
        self._r2.setup_scene()

        self.scene.articulations["robot1"] = self._r1.get_robot()
        self.scene.articulations["robot2"] = self._r2.get_robot()
        self.scene.sensors["contact_sensor1"] = self._r1._contact_sensor
        self.scene.sensors["contact_sensor2"] = self._r2._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._r1.pre_physics_step(actions[:, : self.cfg.action_space // 2]) # Ignore this red squiggly
        self._r2.pre_physics_step(actions[:, self.cfg.action_space // 2 :]) # Ignore this red squiggly

    def _apply_action(self):
        self._r1.apply_action()
        self._r2.apply_action()

    def _get_observations(self) -> dict:
        obs1 = self._r1.get_observations()
        obs2 = self._r2.get_observations()
        # self._previous_actions = self._actions.clone()

        observations = {"policy": torch.cat((obs1["policy"], obs2["policy"]), dim=1)}
        tmp = observations["policy"]
        # print(f"Combined shape: {tmp.shape}")
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward1 = self._r1.get_rewards()
        total_reward2 = self._r2.get_rewards()
        return total_reward1 + total_reward2

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died1 = self._r1.get_dones()
        died2 = self._r2.get_dones()
        died = died1 | died2
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._r1._robot._ALL_INDICES
        super()._reset_idx(env_ids) # Ignore this red squiggly

        self._r1.reset_idx(env_ids, self.scene)
        self._r2.reset_idx(env_ids, self.scene)
       
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
