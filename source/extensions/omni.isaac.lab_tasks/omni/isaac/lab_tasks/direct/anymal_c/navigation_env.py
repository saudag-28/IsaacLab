# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectMARLEnv
from omni.isaac.lab.sensors import ContactSensor, RayCaster

from .anymal_c_env_cfg import AnymalCFlatEnvCfg, AnymalCRoughEnvCfg


class AnymalCEnv(DirectMARLEnv):
    cfg: AnymalCFlatEnvCfg

    def __init__(self, cfg: AnymalCFlatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        print("Hello init")

        # Joint position command (deviation from default joint positions)
        # robot1
        self._actions_robot1 = torch.zeros(self.num_envs, gym.spaces.flatdim(self.action_spaces['robot1']), device=self.device)
        print(f"Action space: {self.action_spaces['robot1']}")
        print(f"Flatdim: {gym.spaces.flatdim(self.action_spaces['robot1'])}")
        self._previous_actions_robot1 = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.action_spaces['robot1']), device=self.device
        )
        # robot2
        self._actions_robot2 = torch.zeros(self.num_envs, gym.spaces.flatdim(self.action_spaces['robot2']), device=self.device)
        self._previous_actions_robot2 = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.action_spaces['robot2']), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        # robot1
        self._commands_robot1 = torch.zeros(self.num_envs, 3, device=self.device)
        # robot2
        self._commands_robot2 = torch.zeros(self.num_envs, 3, device=self.device)

        print("Hello init2")

        # Get specific body indices
        # robot1
        self._base_id_robot1, _ = self._contact_sensor_robot1.find_bodies("base")
        self._feet_ids_robot1, _ = self._contact_sensor_robot1.find_bodies(".*FOOT")
        self._undesired_contact_body_ids_robot1, _ = self._contact_sensor_robot1.find_bodies(".*THIGH")

        print("Hello init 3")
        # robot2
        self._base_id_robot2, _ = self._contact_sensor_robot2.find_bodies("base")
        self._feet_ids_robot2, _ = self._contact_sensor_robot2.find_bodies(".*FOOT")
        self._undesired_contact_body_ids_robot2, _ = self._contact_sensor_robot2.find_bodies(".*THIGH")

        print("Hello init 4")

    def _setup_scene(self):
        print("Hello scene")
        self._robot1 = Articulation(self.cfg.robot1)
        self._robot2 = Articulation(self.cfg.robot2)

        self.scene.articulations["quad_robot1"] = self._robot1
        self.scene.articulations["quad_robot2"] = self._robot2

        self._contact_sensor_robot1 = ContactSensor(self.cfg.contact_sensor1)
        self._contact_sensor_robot2 = ContactSensor(self.cfg.contact_sensor2)

        self.scene.sensors["contact_sensor1"] = self._contact_sensor_robot1
        self.scene.sensors["contact_sensor2"] = self._contact_sensor_robot2

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # TODO: IDK IF THIS IS CORRECT
    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        print("Hello pre")
        self.actions= actions
        # self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self) -> None:
        print("Hello action")
        self._robot1.set_joint_position_target(self.actions["_robot1"])
        self._robot2.set_joint_position_target(self.actions["_robot2"])

    def _get_observations(self) -> dict[str, torch.Tensor]:
        print("Hello obs")

        self._previous_actions_robot1 = self._actions_robot1
        self._previous_actions_robot2 = self._actions_robot2
        height_data = None

        obs = {
            "_robot1": torch.cat(
                [
                    self._robot1.data.root_lin_vel_b,
                    self._robot1.data.root_ang_vel_b,
                    self._robot1.data.projected_gravity_b,
                    self._commands_robot1,
                    self._robot1.data.joint_pos - self._robot1.data.default_joint_pos,
                    self._robot1.data.joint_vel,
                    height_data,
                    self.actions["_robot1"],
                ],
                dim = -1,
            ),
            "_robot2": torch.cat(
                [
                    self._robot2.data.root_lin_vel_b,
                    self._robot2.data.root_ang_vel_b,
                    self._robot2.data.projected_gravity_b,
                    self._commands_robot2,
                    self._robot2.data.joint_pos - self._robot2.data.default_joint_pos,
                    self._robot2.data.joint_vel,
                    height_data,
                    self.actions["_robot2"],
                ],
                dim = -1,
            ),
        }

        # observations = {"policy": obs}
        return obs

        # self._previous_actions = self._actions.clone()
        # height_data = None
        
        # obs = torch.cat(
        #     [
        #         tensor
        #         for tensor in (
        #             self._robot.data.root_lin_vel_b,
        #             self._robot.data.root_ang_vel_b,
        #             self._robot.data.projected_gravity_b,
        #             self._commands,
        #             self._robot.data.joint_pos - self._robot.data.default_joint_pos,
        #             self._robot.data.joint_vel,
        #             height_data,
        #             self._actions,
        #         )
        #         if tensor is not None
        #     ],
        #     dim=-1,
        # )
        # observations = {"policy": obs}
        # return observations

    def _get_rewards(self) -> torch.Tensor:
        print("Hello rewards")
        # linear velocity tracking
        # robot1
        lin_vel_error_robot1 = torch.sum(torch.square(self._commands_robot1[:, :2] - self._robot1.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped_robot1 = torch.exp(-lin_vel_error_robot1 / 0.25)
        # robot2
        lin_vel_error_robot2 = torch.sum(torch.square(self._commands_robot2[:, :2] - self._robot2.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped_robot2 = torch.exp(-lin_vel_error_robot2 / 0.25)

        # yaw rate tracking
        # robot1
        yaw_rate_error_robot1 = torch.square(self._commands_robot1[:, 2] - self._robot1.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped_robot1 = torch.exp(-yaw_rate_error_robot1 / 0.25)
        # robot2
        yaw_rate_error_robot2 = torch.square(self._commands_robot2[:, 2] - self._robot2.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped_robot2 = torch.exp(-yaw_rate_error_robot2 / 0.25)

        # z velocity tracking
        # robot1
        z_vel_error_robot1 = torch.square(self._robot1.data.root_lin_vel_b[:, 2])
        # robot2
        z_vel_error_robot2 = torch.square(self._robot2.data.root_lin_vel_b[:, 2])

        # angular velocity x/y
        # robot1
        ang_vel_error_robot1 = torch.sum(torch.square(self._robot1.data.root_ang_vel_b[:, :2]), dim=1)
        # robot2
        ang_vel_error_robot2 = torch.sum(torch.square(self._robot2.data.root_ang_vel_b[:, :2]), dim=1)

        # joint torques
        # robot1
        joint_torques_robot1 = torch.sum(torch.square(self._robot1.data.applied_torque), dim=1)
        # robot2
        joint_torques_robot2 = torch.sum(torch.square(self._robot2.data.applied_torque), dim=1)

        # joint acceleration
        # robot1
        joint_accel_robot1 = torch.sum(torch.square(self._robot1.data.joint_acc), dim=1)
        # robot2
        joint_accel_robot2 = torch.sum(torch.square(self._robot2.data.joint_acc), dim=1)

        # action rate
        # robot1
        action_rate_robot1 = torch.sum(torch.square(self._actions_robot1 - self._previous_actions_robot1), dim=1)
        # robot2
        action_rate_robot2 = torch.sum(torch.square(self._actions_robot2 - self._previous_actions_robot2), dim=1)

        # feet air time
        # robot1
        first_contact_robot1 = self._contact_sensor_robot1.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time_robot1 = self._contact_sensor_robot1.data.last_air_time[:, self._feet_ids]
        air_time_robot1 = torch.sum((last_air_time_robot1 - 0.5) * first_contact_robot1, dim=1) * (
            torch.norm(self._commands_robot1[:, :2], dim=1) > 0.1
        )
        # robot2
        first_contact_robot2 = self._contact_sensor_robot2.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time_robot2 = self._contact_sensor_robot2.data.last_air_time[:, self._feet_ids]
        air_time_robot2 = torch.sum((last_air_time_robot2 - 0.5) * first_contact_robot2, dim=1) * (
            torch.norm(self._commands_robot2[:, :2], dim=1) > 0.1
        )

        # undersired contacts
        # robot1
        net_contact_forces_robot1 = self._contact_sensor_robot1.data.net_forces_w_history
        is_contact_robot1 = (
            torch.max(torch.norm(net_contact_forces_robot1[:, :, self._undesired_contact_body_ids_robot1], dim=-1), dim=1)[0] > 1.0
        )
        contacts_robot1 = torch.sum(is_contact_robot1, dim=1)
        # robot2
        net_contact_forces_robot2 = self._contact_sensor_robot2.data.net_forces_w_history
        is_contact_robot2 = (
            torch.max(torch.norm(net_contact_forces_robot2[:, :, self._undesired_contact_body_ids_robot2], dim=-1), dim=1)[0] > 1.0
        )
        contacts_robot2 = torch.sum(is_contact_robot2, dim=1)

        # flat orientation
        # robot1
        flat_orientation_robot1 = torch.sum(torch.square(self._robot1.data.projected_gravity_b[:, :2]), dim=1)
        # robot2
        flat_orientation_robot2 = torch.sum(torch.square(self._robot2.data.projected_gravity_b[:, :2]), dim=1)

        rewards = {
            "track_lin_vel_xy_exp": (lin_vel_error_mapped_robot1 + lin_vel_error_mapped_robot2) * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": (yaw_rate_error_mapped_robot1 + yaw_rate_error_mapped_robot2) * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": (z_vel_error_robot1 + z_vel_error_robot2) * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": (ang_vel_error_robot1 + ang_vel_error_robot2) * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": (joint_torques_robot1 + joint_torques_robot2) * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": (joint_accel_robot1 + joint_accel_robot2) * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": (action_rate_robot1 + joint_accel_robot2) * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": (air_time_robot1 + air_time_robot2) * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": (contacts_robot1 + air_time_robot2) * self.cfg.undersired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": (flat_orientation_robot1 + flat_orientation_robot1) * self.cfg.flat_orientation_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # # Logging
        # for key, value in rewards.items():
        #     self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        print("Hello dones")
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces_robot1 = self._contact_sensor_robot1.data.net_forces_w_history
        net_contact_forces_robot2 = self._contact_sensor_robot2.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces_robot1[:, :, self._base_id_robot1] + net_contact_forces_robot2[:, :, self._base_id_robot2], dim=-1), dim=1)[0] > 1.0, dim=1)

        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return died, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        print("Hello reset")
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot1._ALL_INDICES
        self._robot1.reset(env_ids)
        self._robot2.reset(env_ids)

        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions_robot1[env_ids] = 0.0
        self._previous_actions_robot1[env_ids] = 0.0
        self._actions_robot2[env_ids] = 0.0
        self._previous_actions_robot2[env_ids] = 0.0

        # Sample new commands
        # robot1
        self._commands_robot1[env_ids] = torch.zeros_like(self._commands_robot1[env_ids]).uniform_(-1.0, 1.0)
        # robot2
        self._commands_robot2[env_ids] = torch.zeros_like(self._commands_robot2[env_ids]).uniform_(-1.0, 1.0)

        # Reset robot state
        joint_pos = self._robot1.data.default_joint_pos[env_ids]
        joint_vel = self._robot1.data.default_joint_vel[env_ids]
        default_root_state = self._robot1.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        # robot1
        self._robot1.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot1.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot1.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # robot2
        self._robot2.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot2.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot2.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # # Logging
        # extras = dict()
        # for key in self._episode_sums.keys():
        #     episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
        #     extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
        #     self._episode_sums[key][env_ids] = 0.0
        # self.extras["log"] = dict()
        # self.extras["log"].update(extras)
        # extras = dict()
        # extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        # extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        # self.extras["log"].update(extras)
