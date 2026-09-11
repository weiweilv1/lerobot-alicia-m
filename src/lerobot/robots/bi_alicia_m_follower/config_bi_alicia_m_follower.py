#!/usr/bin/env python

# Copyright (c) 2025 Synria Robotics Co., Ltd.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import math
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("bi_alicia_m_follower")
@dataclass
class BiAliciaMFollowerConfig(RobotConfig):
    """Configuration for two independently connected Alicia-M follower arms."""

    left_arm_port: str = ""
    right_arm_port: str = ""

    # SDK settings shared by both arms.
    version: str = "auto"
    variant: str | None = None
    baudrate: int = 1_000_000
    control_aim: str | None = "follower"
    control_mode: str | None = "mit"
    sync_control_mode: bool = True
    backend: str = "cpp"
    extended_polling: bool = False
    use_interpolation: bool = False
    startup_alignment: bool = True
    startup_alignment_threshold_deg: float = 2.0
    startup_gripper_alignment_threshold: float = 50.0
    read_retry_count: int = 3
    read_retry_interval_s: float = 0.02

    # Per-arm connection, safety, and motion settings.
    left_arm_connect: bool | None = None
    left_arm_disable_torque_on_disconnect: bool = False
    left_arm_max_relative_target: float | dict[str, float] | None = None
    left_arm_speed: float = 100.0
    left_arm_gripper_speed: float = 100.0
    left_arm_debug_mode: bool = False

    right_arm_connect: bool | None = None
    right_arm_disable_torque_on_disconnect: bool = False
    right_arm_max_relative_target: float | dict[str, float] | None = None
    right_arm_speed: float = 100.0
    right_arm_gripper_speed: float = 100.0
    right_arm_debug_mode: bool = False

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Used only by direct-wired recording. Computer-mediated D->M control keeps
    # this enabled but reads the real Alicia-M observations because the leader's
    # directly_controls_robot flag is false.
    use_teleop_state_for_observation: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.control_aim not in (None, "leader", "follower"):
            raise ValueError("Bi-Alicia-M control_aim must be 'leader', 'follower', or None.")
        if self.control_mode not in (None, "pv", "mit"):
            raise ValueError("Bi-Alicia-M control_mode must be 'pv', 'mit', or None.")

        speeds = {
            "left_arm_speed": self.left_arm_speed,
            "left_arm_gripper_speed": self.left_arm_gripper_speed,
            "right_arm_speed": self.right_arm_speed,
            "right_arm_gripper_speed": self.right_arm_gripper_speed,
        }
        for name, value in speeds.items():
            if not math.isfinite(value) or not 0 < value <= 400:
                raise ValueError(f"{name} must be in the range (0, 400].")
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
