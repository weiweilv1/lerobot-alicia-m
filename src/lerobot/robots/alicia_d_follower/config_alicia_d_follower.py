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

import math
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("alicia_d_follower")
@dataclass
class AliciaDFollowerConfig(RobotConfig):
    """Configuration for an Alicia-D follower arm and its optional cameras."""

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

    disable_torque_on_disconnect: bool = False

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Command speeds used by alicia_d_sdk.set_robot_state().
    speed_deg_s: float = 100.0
    gripper_speed_deg_s: float = 483.4

    # Retry transient SDK serial read failures before aborting the control loop.
    read_retry_count: int = 3
    read_retry_interval_s: float = 0.02

    # Allow teleop state to overwrite joint observations when the leader is
    # directly wired to the follower.
    use_teleop_state_for_observation: bool = True

    # If None, connect the arm only when a non-empty port is configured. This
    # allows a directly wired leader/follower pair to use LeRobot for cameras.
    connect_arm: bool | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not math.isfinite(self.speed_deg_s)
            or not math.isfinite(self.gripper_speed_deg_s)
            or self.speed_deg_s <= 0
            or self.gripper_speed_deg_s <= 0
        ):
            raise ValueError("Alicia-D command speeds must be positive.")
        if self.read_retry_count < 1:
            raise ValueError("read_retry_count must be at least 1.")
        if not math.isfinite(self.read_retry_interval_s) or self.read_retry_interval_s < 0:
            raise ValueError("read_retry_interval_s must be finite and non-negative.")
