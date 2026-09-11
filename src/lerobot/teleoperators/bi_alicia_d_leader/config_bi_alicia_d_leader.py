#!/usr/bin/env python
"""Bimanual Alicia-D Leader Configuration

Configuration class for Bimanual Alicia-D Leader teleoperator integration with LeRobot.

Copyright (c) 2025 Synria Robotics Co., Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

import math
from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_alicia_d_leader")
@dataclass
class BiAliciaDLeaderConfig(TeleoperatorConfig):
    """Configuration for a pair of Alicia-D leader arms."""

    left_arm_port: str = ""
    right_arm_port: str = ""

    # SDK model settings shared by both arms.
    version: str = "v5_6"
    variant: str | None = None
    model_format: str = "urdf"
    base_link: str = "base_link"
    end_link: str = "tool0"
    backend: str | None = None
    device: str = "cpu"

    # Gripper types for Alicia-D (e.g., "50mm" or "100mm")
    left_arm_gripper_type: str | None = None
    right_arm_gripper_type: str | None = None

    # Debug mode for SDK
    left_arm_debug_mode: bool = False
    right_arm_debug_mode: bool = False

    # Whether the leader arms directly control the follower arms via hardware wire.
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

    target_follower_type: str = "alicia_d"

    read_retry_count: int = 3
    read_retry_interval_s: float = 0.001
    max_consecutive_read_failures: int = 10

    def __post_init__(self) -> None:
        if not self.left_arm_port or not self.right_arm_port:
            raise ValueError("Both left_arm_port and right_arm_port are required for a bimanual leader.")
        if self.left_arm_port.casefold() == self.right_arm_port.casefold():
            raise ValueError("left_arm_port and right_arm_port must identify different serial ports.")
        if self.action_observation_delay_frames < 0:
            raise ValueError("action_observation_delay_frames must be non-negative.")
        if self.target_follower_type not in {"alicia_d", "alicia_m"}:
            raise ValueError("target_follower_type must be 'alicia_d' or 'alicia_m'.")
        if self.read_retry_count < 1:
            raise ValueError("read_retry_count must be at least 1.")
        if not math.isfinite(self.read_retry_interval_s) or self.read_retry_interval_s < 0:
            raise ValueError("read_retry_interval_s must be finite and non-negative.")
        if self.max_consecutive_read_failures < 0:
            raise ValueError("max_consecutive_read_failures must be non-negative.")
