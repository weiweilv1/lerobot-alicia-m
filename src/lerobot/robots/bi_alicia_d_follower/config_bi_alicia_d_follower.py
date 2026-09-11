#!/usr/bin/env python
"""Bimanual Alicia-D Follower Configuration

Configuration class for Bimanual Alicia-D Follower robot integration with LeRobot.

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
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("bi_alicia_d_follower")
@dataclass
class BiAliciaDFollowerConfig(RobotConfig):
    """Configuration for a pair of Alicia-D follower arms."""

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

    # Optional left arm settings
    left_arm_connect: bool | None = None
    left_arm_disable_torque_on_disconnect: bool = False
    left_arm_max_relative_target: float | dict[str, float] | None = None
    left_arm_gripper_type: str | None = None
    left_arm_debug_mode: bool = False
    left_arm_speed_deg_s: float = 20.0
    left_arm_gripper_speed_deg_s: float = 483.4

    # Optional right arm settings
    right_arm_connect: bool | None = None
    right_arm_disable_torque_on_disconnect: bool = False
    right_arm_max_relative_target: float | dict[str, float] | None = None
    right_arm_gripper_type: str | None = None
    right_arm_debug_mode: bool = False
    right_arm_speed_deg_s: float = 20.0
    right_arm_gripper_speed_deg_s: float = 483.4

    read_retry_count: int = 3
    read_retry_interval_s: float = 0.02

    # cameras (shared between both arms)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Allow teleop state to overwrite joint observations when the leader is
    # directly wired to the follower.
    use_teleop_state_for_observation: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        speeds = {
            "left_arm_speed_deg_s": self.left_arm_speed_deg_s,
            "left_arm_gripper_speed_deg_s": self.left_arm_gripper_speed_deg_s,
            "right_arm_speed_deg_s": self.right_arm_speed_deg_s,
            "right_arm_gripper_speed_deg_s": self.right_arm_gripper_speed_deg_s,
        }
        for name, value in speeds.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.read_retry_count < 1:
            raise ValueError("read_retry_count must be at least 1.")
        if not math.isfinite(self.read_retry_interval_s) or self.read_retry_interval_s < 0:
            raise ValueError("read_retry_interval_s must be finite and non-negative.")

        if self.left_arm_connect is True and not self.left_arm_port:
            raise ValueError("left_arm_port is required when left_arm_connect=true.")
        if self.right_arm_connect is True and not self.right_arm_port:
            raise ValueError("right_arm_port is required when right_arm_connect=true.")

        left_connects = self.left_arm_connect is True or (
            self.left_arm_connect is None and bool(self.left_arm_port)
        )
        right_connects = self.right_arm_connect is True or (
            self.right_arm_connect is None and bool(self.right_arm_port)
        )
        if (
            left_connects
            and right_connects
            and self.left_arm_port.casefold() == self.right_arm_port.casefold()
        ):
            raise ValueError("left_arm_port and right_arm_port must identify different serial ports.")
