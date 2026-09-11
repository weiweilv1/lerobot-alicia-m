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


@RobotConfig.register_subclass("alicia_m_follower")
@dataclass
class AliciaMFollowerConfig(RobotConfig):
    """Configuration for an Alicia-M follower arm and its optional cameras."""

    port: str = ""

    # Arguments supported by Alicia-M-SDK 1.1.1rc2 and newer.
    version: str = "auto"
    variant: str | None = None
    baudrate: int = 1_000_000
    control_aim: str | None = "follower"
    control_mode: str | None = "mit"
    sync_control_mode: bool = True
    backend: str = "cpp"
    extended_polling: bool = False
    debug_mode: bool = False

    # Effective in MIT mode. PV mode always uses firmware interpolation.
    use_interpolation: bool = False

    disable_torque_on_disconnect: bool = False

    # Maximum relative per-step target for safety clipping.
    max_relative_target: float | dict[str, float] | None = None

    speed: float = 100.0
    gripper_speed: float = 100.0

    # Before a computer-mediated teleoperation loop starts, smoothly move to
    # the leader's mapped pose when the initial error exceeds these thresholds.
    startup_alignment: bool = True
    startup_alignment_threshold_deg: float = 2.0
    startup_gripper_alignment_threshold: float = 50.0

    # Retry transient SDK state-read failures before aborting the control loop.
    read_retry_count: int = 3
    read_retry_interval_s: float = 0.02

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Allow teleop state to overwrite joint observations when wired leader->follower.
    use_teleop_state_for_observation: bool = True

    # If None, connect the arm only when a non-empty port is configured.
    connect_arm: bool | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.control_aim not in (None, "leader", "follower"):
            raise ValueError("Alicia-M control_aim must be 'leader', 'follower', or None.")
        if self.control_mode not in (None, "pv", "mit"):
            raise ValueError("Alicia-M control_mode must be 'pv', 'mit', or None.")
        if (
            not math.isfinite(self.speed)
            or not math.isfinite(self.gripper_speed)
            or not 0 < self.speed <= 400
            or not 0 < self.gripper_speed <= 400
        ):
            raise ValueError("Alicia-M speeds must be in the range (0, 400].")
        if (
            not math.isfinite(self.startup_alignment_threshold_deg)
            or self.startup_alignment_threshold_deg < 0
        ):
            raise ValueError("startup_alignment_threshold_deg must be finite and non-negative.")
        if (
            not math.isfinite(self.startup_gripper_alignment_threshold)
            or self.startup_gripper_alignment_threshold < 0
        ):
            raise ValueError("startup_gripper_alignment_threshold must be finite and non-negative.")
        if self.read_retry_count < 1:
            raise ValueError("read_retry_count must be at least 1.")
        if not math.isfinite(self.read_retry_interval_s) or self.read_retry_interval_s < 0:
            raise ValueError("read_retry_interval_s must be finite and non-negative.")
