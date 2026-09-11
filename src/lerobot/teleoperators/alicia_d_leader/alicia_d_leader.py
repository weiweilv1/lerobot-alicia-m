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

import logging
import math
import time
from contextlib import suppress
from functools import cached_property
from typing import Any

from lerobot.lerobot_types import RobotAction
from lerobot.utils.console_utils import safe_console_output
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import require_package

from ..teleoperator import Teleoperator
from .config_alicia_d_leader import AliciaDLeaderConfig

logger = logging.getLogger(__name__)

URDF_LIMIT = {
    "ALICIA_D": [
        {"jointName": "Joint1", "lower": -157.5, "upper": 157.5},
        {"jointName": "Joint2", "lower": -100.2, "upper": 100.2},
        {"jointName": "Joint3", "lower": -34.3, "upper": 126.0},
        {"jointName": "Joint4", "lower": -159.8, "upper": 159.8},
        {"jointName": "Joint5", "lower": -89.9, "upper": 89.9},
        {"jointName": "Joint6", "lower": -179.9, "upper": 179.9},
    ],
    "ALICIA_M": [
        {"jointName": "Joint1", "lower": -157.5, "upper": 157.5},
        {"jointName": "Joint2", "lower": -179.9, "upper": 0},
        {"jointName": "Joint3", "lower": -179.9, "upper": 0},
        {"jointName": "Joint4", "lower": -89.9, "upper": 89.9},
        {"jointName": "Joint5", "lower": -89.9, "upper": 89.9},
        {"jointName": "Joint6", "lower": -157.5, "upper": 157.5},
    ],
}

REVERSED_JOINT_INDEXES = {3, 5}
PROPORTIONAL_JOINT_INDEXES = {2}
NEGATED_INPUT_JOINT_INDEXES = {2}


def clamp(value: float, lower: float, upper: float) -> float:
    """
    Clamp a value to the specified range.
    """
    return max(lower, min(value, upper))


def map_joint_value(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    """
    Linearly map a joint value from the source range to the target range.

    :param value: Joint value
    :param src_min: Source joint lower bound
    :param src_max: Source joint upper bound
    :param dst_min: Target joint lower bound
    :param dst_max: Target joint upper bound
    :return: Mapped joint value
    """
    return ((value - src_min) / (src_max - src_min)) * (dst_max - dst_min) + dst_min


def map_joint_value_with_m_limit(
    value: float,
    dst_min: float,
    dst_max: float,
    reverse: bool = False,
    align_to_center: bool = True,
) -> float:
    """
    Keep a 1:1 D-to-M angle relationship while clamping the result to M limits.

    Rules:
    - If align_to_center=True, D's 0 degree maps to the midpoint of M's range
    - If align_to_center=False, D's 0 degree maps to M's 0 degree
    - 1 degree in M corresponds to 1 degree in D
    - If reverse=True, the target direction is flipped
    - The final output will not exceed M's joint limits
    """
    dst_origin = (dst_min + dst_max) / 2 if align_to_center else 0.0
    direction = -1.0 if reverse else 1.0
    mapped = direction * value + dst_origin
    return clamp(mapped, min(dst_min, dst_max), max(dst_min, dst_max))


def convert_joints_deg_from_alicia_d_to_alicia_m(joints_deg: list[float]) -> list[float]:
    """
    Convert Alicia-D joint values to Alicia-M joint values.

    Notes:
    - This function returns a new list
    - D's 0 degree is aligned to the midpoint of M's joint range
    - 1 degree in M corresponds to 1 degree in D, without proportional scaling
    - Python indexes 3 and 5 (i.e. joints 4 and 6) use reversed mapping
    - Joint 3 (index 2) is handled specially:
      first negate the input, then apply proportional mapping
      from D[-126.0, 34.3] to M[-179.9, 0]

    :param joints_deg: Alicia-D joint values
    :return: Alicia-M joint values
    """
    if len(joints_deg) != 6:
        raise ValueError(f"Expected 6 Alicia-D joint angles, got {len(joints_deg)}.")

    result = []

    for i, joint in enumerate(joints_deg):
        if i in PROPORTIONAL_JOINT_INDEXES:
            if i in NEGATED_INPUT_JOINT_INDEXES:
                joint = -joint
                src_min = -URDF_LIMIT["ALICIA_D"][i]["upper"]
                src_max = -URDF_LIMIT["ALICIA_D"][i]["lower"]
            else:
                src_min = URDF_LIMIT["ALICIA_D"][i]["lower"]
                src_max = URDF_LIMIT["ALICIA_D"][i]["upper"]
            dst_min = URDF_LIMIT["ALICIA_M"][i]["lower"]
            dst_max = URDF_LIMIT["ALICIA_M"][i]["upper"]
            mapped = map_joint_value(
                clamp(joint, src_min, src_max),
                src_min,
                src_max,
                dst_min,
                dst_max,
            )
            mapped = clamp(mapped, min(dst_min, dst_max), max(dst_min, dst_max))
        else:
            mapped = map_joint_value_with_m_limit(
                joint,
                URDF_LIMIT["ALICIA_M"][i]["lower"],
                URDF_LIMIT["ALICIA_M"][i]["upper"],
                reverse=i in REVERSED_JOINT_INDEXES,
            )
        result.append(mapped)

    return result


class AliciaDLeader(Teleoperator):
    """
    Alicia-D Leader Arm - LeRobot teleoperator integration using Alicia-D SDK.

    This teleoperator reads joint positions and button status from the leader arm
    and provides them as actions for follower arms. Leader arms use the same API
    as follower arms but don't have pose data.
    """

    config_class = AliciaDLeaderConfig
    name = "alicia_d_leader"

    def __init__(self, config: AliciaDLeaderConfig):
        require_package("alicia_d_sdk", extra="alicia", import_name="alicia_d_sdk")
        import alicia_d_sdk

        super().__init__(config)
        self.config = config

        with safe_console_output():
            self.robot_api: Any = alicia_d_sdk.create_robot(
                port=self.config.port,
                version=self.config.version,
                variant=self.config.variant,
                model_format=self.config.model_format,
                debug_mode=self.config.debug_mode,
                auto_connect=False,
                base_link=self.config.base_link,
                end_link=self.config.end_link,
                backend=self.config.backend,
                device=self.config.device,
                gripper_type=self.config.gripper_type,
            )

        self._joint_names = [f"joint{i}" for i in range(1, 7)]
        self._gripper_name = "gripper"
        self._last_action: RobotAction | None = None
        self._consecutive_read_failures = 0

    @cached_property
    def action_features(self) -> dict[str, type]:
        ft = {f"{name}.pos": float for name in self._joint_names}
        ft[f"{self._gripper_name}.pos"] = float
        return ft

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def directly_controls_robot(self) -> bool:
        """
        Whether the leader arm directly controls the follower arm via hardware wire.

        Returns the value from configuration. If True, actions don't need to be sent
        through the computer. If False, actions will be sent through the computer.
        """
        return self.config.directly_controls_robot

    @property
    def uses_action_as_observation(self) -> bool:
        """Whether to use leader actions as robot observations during recording."""
        return self.config.use_action_as_observation

    @property
    def action_observation_delay_frames(self) -> int:
        """Frame delay between observation and action when using leader state."""
        return max(0, int(self.config.action_observation_delay_frames))

    @property
    def target_follower_type(self) -> str:
        return self.config.target_follower_type

    @property
    def is_connected(self) -> bool:
        return bool(self.robot_api.is_connected())

    @property
    def has_connected_resources(self) -> bool:
        with suppress(Exception):
            return bool(self.robot_api.is_connected())
        return False

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        try:
            if not self.robot_api.connect():
                target = self.config.port or "auto-discovered port"
                raise ConnectionError(f"Failed to connect to Alicia-D leader on {target}")
            if calibrate and not self.is_calibrated:
                self.calibrate()
            self.configure()
        except Exception:
            with suppress(Exception):
                if self.robot_api.is_connected():
                    self.robot_api.disconnect()
            raise
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """
        Check if leader arm is calibrated.

        Note: Alicia-D robots typically don't require calibration in the same way
        as Feetech motors. This always returns True.
        """
        return True

    def calibrate(self) -> None:
        logger.info("Alicia-D uses the calibration stored by its SDK; no LeRobot calibration is required.")

    def configure(self) -> None:
        """Apply configuration to the leader arm."""
        # Alicia-D SDK handles most configuration automatically
        logger.debug(f"{self} configured.")

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """
        Get current action from the leader arm.

        Reads joint positions and gripper value from the leader arm.
        Leader arms don't have pose data, only joint positions.

        Returns:
            Dictionary with joint positions and gripper value in the format:
            {'joint1.pos': 0.5, 'joint2.pos': 0.3, ..., 'gripper.pos': 0.5}
        """
        start = time.perf_counter()

        # Retry transient serial timeouts from SDK (its logger.error raises Exception).
        state = None
        last_error: Exception | None = None
        for _ in range(self.config.read_retry_count):
            try:
                state = self.robot_api.get_robot_state("joint_gripper")
                if state is not None:
                    break
            except Exception as e:  # noqa: BLE001
                last_error = e
            time.sleep(self.config.read_retry_interval_s)

        if state is None:
            self._consecutive_read_failures += 1
            if (
                self._last_action is not None
                and self._consecutive_read_failures <= self.config.max_consecutive_read_failures
            ):
                logger.warning(
                    f"{self} read timeout x{self._consecutive_read_failures}, "
                    "reusing last valid leader action."
                )
                return dict(self._last_action)
            if last_error is not None:
                raise ConnectionError(f"Failed to read robot state from {self}") from last_error
            raise ConnectionError(f"Failed to read robot state from {self}")

        if isinstance(state, dict):
            joint_angles_rad = state.get("angles", state.get("joints"))
            gripper_value = state.get("gripper")
            button_status = state.get("run_status_text", "unknown")
        else:
            joint_angles_rad = getattr(state, "angles", None)
            gripper_value = getattr(state, "gripper", None)
            button_status = getattr(state, "run_status_text", "unknown")
        if joint_angles_rad is None:
            raise ValueError(f"{self} joint state has no angles: {state!r}")
        if len(joint_angles_rad) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(joint_angles_rad)}")
        joint_angles_deg = [math.degrees(float(angle)) for angle in joint_angles_rad]
        gripper_value = 0.0 if gripper_value is None else float(gripper_value)
        if not all(math.isfinite(value) for value in (*joint_angles_deg, gripper_value)):
            raise ValueError(f"{self} returned non-finite joint or gripper state.")

        if self.target_follower_type == "alicia_m":
            joint_angles_deg = convert_joints_deg_from_alicia_d_to_alicia_m(joint_angles_deg)

        action: RobotAction = {
            f"{joint_name}.pos": float(joint_angles_deg[i]) for i, joint_name in enumerate(self._joint_names)
        }
        action[f"{self._gripper_name}.pos"] = float(gripper_value)
        self._last_action = dict(action)
        self._consecutive_read_failures = 0

        logger.debug(f"{self} button status: {button_status}")

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """
        Send feedback to the leader arm.

        Note: Force feedback is not currently implemented for Alicia-D leader arms.
        """
        # TODO: Implement force feedback if supported by hardware
        logger.debug(f"Feedback not implemented for {self}")

    def disconnect(self) -> None:
        if self.has_connected_resources:
            self.robot_api.disconnect()
        logger.info(f"{self} disconnected.")
