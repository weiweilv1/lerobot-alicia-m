#!/usr/bin/env python3
# Copyright (c) 2025 Synria Robotics Co., Ltd.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai


from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("alicia_d_leader")
@dataclass
class AliciaDLeaderConfig(TeleoperatorConfig):
    """Configuration for an Alicia-D leader arm."""

    port: str = ""

    # Arguments forwarded to alicia_d_sdk.create_robot().
    version: str = "v5_6"
    variant: str | None = None
    model_format: str = "urdf"
    base_link: str = "base_link"
    end_link: str = "tool0"
    backend: str | None = None
    device: str = "cpu"
    gripper_type: str | None = None
    debug_mode: bool = False

    # Whether the leader arm directly controls the follower arm via hardware wire.
    # If True, actions don't need to be sent through the computer (default: True).
    # If False, actions will be sent through the computer from teleoperator to robot.
    directly_controls_robot: bool = True

    # When directly controlling the follower via hardware, choose whether the
    # leader state should overwrite robot joint observations during recording.
    # True: teleop-only joint observations (case 1).
    # False: robot observations + teleop actions (case 2).
    use_action_as_observation: bool = True

    # Delay (in frames) between observation (leader state) and action (leader command).
    # A value of 1 means observation is the previous frame's leader state.
    action_observation_delay_frames: int = 1

    # Target follower kinematic convention used for output joint values.
    # - "alicia_d": no remapping (native Alicia-D leader joint ranges)
    # - "alicia_m": remap Alicia-D leader joints to Alicia-M follower joint ranges
    target_follower_type: str = "alicia_d"

    read_retry_count: int = 3
    read_retry_interval_s: float = 0.001
    max_consecutive_read_failures: int = 10

    def __post_init__(self) -> None:
        if self.action_observation_delay_frames < 0:
            raise ValueError("action_observation_delay_frames must be non-negative.")
        if self.target_follower_type not in {"alicia_d", "alicia_m"}:
            raise ValueError("target_follower_type must be 'alicia_d' or 'alicia_m'.")
        if self.read_retry_count < 1:
            raise ValueError("read_retry_count must be at least 1.")
        if self.read_retry_interval_s < 0:
            raise ValueError("read_retry_interval_s must be non-negative.")
        if self.max_consecutive_read_failures < 0:
            raise ValueError("max_consecutive_read_failures must be non-negative.")
