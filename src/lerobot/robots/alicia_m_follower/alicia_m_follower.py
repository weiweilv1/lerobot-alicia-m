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

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.console_utils import safe_console_output
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import require_package

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_alicia_m_follower import AliciaMFollowerConfig

logger = logging.getLogger(__name__)


class AliciaMFollower(Robot):
    """Alicia-M follower arm implementation for LeRobot."""

    config_class = AliciaMFollowerConfig
    name = "alicia_m_follower"

    def __init__(self, config: AliciaMFollowerConfig):
        require_package("alicia_m_sdk", extra="alicia", import_name="alicia_m_sdk")
        import alicia_m_sdk

        super().__init__(config)
        self.config = config
        self._arm_connected = False
        self._arm_connection_required = True

        with safe_console_output():
            self.robot_api: Any = alicia_m_sdk.create_robot(
                port=self.config.port,
                version=self.config.version,
                variant=self.config.variant,
                control_aim=self.config.control_aim,
                control_mode=self.config.control_mode,
                sync_control_mode=self.config.sync_control_mode,
                baudrate=self.config.baudrate,
                backend=self.config.backend,
                debug_mode=self.config.debug_mode,
                auto_connect=False,
                extended_polling=self.config.extended_polling,
            )

        self._joint_names = [f"joint{i}" for i in range(1, 7)]
        self._gripper_name = "gripper"
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        ft = {f"{name}.pos": float for name in self._joint_names}
        ft[f"{self._gripper_name}.pos"] = float
        return ft

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        features: dict[str, tuple] = {}
        for cam_key in self.cameras:
            cfg = self.config.cameras[cam_key]
            if getattr(cfg, "use_rgb", True):
                features[cam_key] = (cfg.height, cfg.width, 3)
            if getattr(cfg, "use_depth", False):
                features[f"{cam_key}_depth"] = (cfg.height, cfg.width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        arm_ok = True
        if self._arm_connection_required:
            arm_ok = self._arm_connected and bool(self.robot_api.is_connected())
        return arm_ok and all(cam.is_connected for cam in self.cameras.values())

    @property
    def has_connected_resources(self) -> bool:
        api_connected = False
        with suppress(Exception):
            api_connected = bool(self.robot_api.is_connected())
        return self._arm_connected or api_connected or any(cam.is_connected for cam in self.cameras.values())

    @property
    def uses_teleop_state_for_observation(self) -> bool:
        return self.config.use_teleop_state_for_observation

    @property
    def requires_teleop_startup_alignment(self) -> bool:
        return self.config.startup_alignment and self._arm_connection_required

    def _should_connect_arm(self) -> bool:
        if self.config.connect_arm is None:
            return bool(self.config.port)
        return self.config.connect_arm

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._arm_connection_required = self._should_connect_arm()
        try:
            if self._arm_connection_required:
                if not self.robot_api.connect():
                    target = self.config.port or "auto-discovered port"
                    raise ConnectionError(f"Failed to connect to Alicia-M on {target}")
                self._arm_connected = True
            else:
                logger.info(f"{self} is using cameras only; follower serial connection is disabled.")

            if calibrate and not self.is_calibrated:
                self.calibrate()
            for cam in self.cameras.values():
                cam.connect()
            self.configure()
        except Exception:
            self._disconnect_resources()
            raise
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("Alicia-M uses the calibration stored by its SDK; no LeRobot calibration is required.")

    def configure(self) -> None:
        control_mode = (self.config.control_mode or "").lower()
        if control_mode == "mit":
            logger.info(
                "%s using MIT %s mode.",
                self,
                "interpolation" if self.config.use_interpolation else "direct PD",
            )
        elif not self.config.use_interpolation:
            logger.warning(
                "%s use_interpolation is ignored when control_mode=%s.",
                self,
                self.config.control_mode,
            )
        logger.debug(f"{self} configured.")

    def _get_joint_state_with_retry(self):
        """Read joint_gripper state with retry for transient SDK read failures."""
        last_error: Exception | None = None
        for attempt in range(self.config.read_retry_count):
            try:
                state = self.robot_api.get_robot_state("joint_gripper")
                if state is not None:
                    return state
            except Exception as e:  # noqa: BLE001
                last_error = e
            if attempt + 1 < self.config.read_retry_count:
                time.sleep(self.config.read_retry_interval_s)
        if last_error is not None:
            raise last_error
        return None

    def _extract_joint_gripper_state(self, joint_state: Any) -> tuple[list[float], float]:
        """Support both dict-style and object-style SDK state payloads."""
        if isinstance(joint_state, dict):
            joint_angles_rad = joint_state.get("angles")
            gripper_raw = joint_state.get("gripper")
        else:
            joint_angles_rad = getattr(joint_state, "angles", None)
            gripper_raw = getattr(joint_state, "gripper", None)

        if joint_angles_rad is None:
            raise ValueError(f"{self} joint_gripper state is missing 'angles': {joint_state!r}")
        if len(joint_angles_rad) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(joint_angles_rad)}")

        joint_angles_rad = [float(angle) for angle in joint_angles_rad]
        gripper_value = 0.0 if gripper_raw is None else float(gripper_raw)
        if not all(math.isfinite(value) for value in (*joint_angles_rad, gripper_value)):
            raise ValueError(f"{self} returned non-finite joint or gripper state.")
        return joint_angles_rad, gripper_value

    @check_if_not_connected
    def prepare_for_teleoperation(self, action: RobotAction) -> bool:
        """Smoothly align to the first leader pose before teleoperation starts."""
        if not self.requires_teleop_startup_alignment:
            return False

        joint_state = self._get_joint_state_with_retry()
        if joint_state is None:
            raise ConnectionError(f"Failed to read current Alicia-M state from {self}.")
        current_joints_rad, current_gripper = self._extract_joint_gripper_state(joint_state)
        current_joints_deg = [math.degrees(angle) for angle in current_joints_rad]

        joint_keys = [f"{joint_name}.pos" for joint_name in self._joint_names]
        target_joints_deg = [
            float(action.get(key, current_joints_deg[index])) for index, key in enumerate(joint_keys)
        ]
        gripper_key = f"{self._gripper_name}.pos"
        target_gripper = max(0.0, min(1000.0, float(action.get(gripper_key, current_gripper))))
        if not all(math.isfinite(value) for value in (*target_joints_deg, target_gripper)):
            raise ValueError(f"Refusing to align {self} to a non-finite action.")

        joint_errors = [
            abs(target - current)
            for target, current in zip(target_joints_deg, current_joints_deg, strict=True)
        ]
        gripper_error = abs(target_gripper - current_gripper)
        needs_alignment = (
            max(joint_errors, default=0.0) > self.config.startup_alignment_threshold_deg
            or gripper_error > self.config.startup_gripper_alignment_threshold
        )
        if not needs_alignment:
            logger.info("%s startup pose is already aligned.", self)
            return False

        logger.info(
            "%s aligning to the teleoperator pose at speed %.1f "
            "(max joint error %.1f deg, gripper error %.1f).",
            self,
            self.config.speed,
            max(joint_errors, default=0.0),
            gripper_error,
        )
        success = self.robot_api.set_robot_state(
            target_joints=target_joints_deg,
            gripper_value=target_gripper,
            joint_format="deg",
            speed=self.config.speed,
            gripper_speed=self.config.gripper_speed,
            wait_for_completion=True,
            # In MIT mode speed is effective only with firmware interpolation.
            # PV mode ignores this flag and still respects speed.
            use_interpolation=True,
        )
        if not success:
            raise RuntimeError(f"{self} failed to reach the teleoperation startup pose.")
        logger.info("%s startup alignment complete.", self)
        return True

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        if self._arm_connection_required:
            start = time.perf_counter()
            joint_state = self._get_joint_state_with_retry()
            if joint_state is None:
                raise ConnectionError(f"Failed to read robot state from {self}.")

            joint_angles_rad, gripper_value = self._extract_joint_gripper_state(joint_state)
            joint_angles_deg = [angle * 180.0 / math.pi for angle in joint_angles_rad]
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read state: {dt_ms:.1f}ms")
        else:
            joint_angles_deg = [0.0 for _ in self._joint_names]
            gripper_value = 0.0

        obs_dict: RobotObservation = {
            f"{joint_name}.pos": float(joint_angles_deg[i]) for i, joint_name in enumerate(self._joint_names)
        }
        obs_dict[f"{self._gripper_name}.pos"] = float(gripper_value)

        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                start = time.perf_counter()
                obs_dict[cam_key] = cam.read_latest()
                logger.debug(f"{self} read {cam_key}: {(time.perf_counter() - start) * 1e3:.1f}ms")
            if getattr(cam, "use_depth", False):
                start = time.perf_counter()
                obs_dict[f"{cam_key}_depth"] = cam.read_latest_depth()
                logger.debug(f"{self} read {cam_key} depth: {(time.perf_counter() - start) * 1e3:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if not self._arm_connection_required:
            return {key: float(value) for key, value in action.items() if key.endswith(".pos")}

        joint_keys = [f"{joint_name}.pos" for joint_name in self._joint_names]
        gripper_key = f"{self._gripper_name}.pos"
        needs_current_state = (
            self.config.max_relative_target is not None
            or any(key not in action for key in joint_keys)
            or gripper_key not in action
        )
        if needs_current_state:
            joint_state = self._get_joint_state_with_retry()
            if joint_state is None:
                raise ConnectionError(f"Failed to read current Alicia-M state from {self}.")
            current_joints_rad, current_gripper = self._extract_joint_gripper_state(joint_state)
            current_joints_deg = [math.degrees(angle) for angle in current_joints_rad]
        else:
            current_joints_deg = [0.0] * len(self._joint_names)
            current_gripper = 0.0
        goal_pos_deg = {
            joint_name: float(action.get(f"{joint_name}.pos", current_joints_deg[index]))
            for index, joint_name in enumerate(self._joint_names)
        }
        gripper_value = float(action.get(gripper_key, current_gripper))
        gripper_value = max(0.0, min(1000.0, gripper_value))

        if not all(math.isfinite(value) for value in (*goal_pos_deg.values(), gripper_value)):
            raise ValueError(f"Refusing to send a non-finite action to {self}.")

        if self.config.max_relative_target is not None:
            goal_present_pos = {
                joint_name: (goal_pos_deg[joint_name], current_joints_deg[index])
                for index, joint_name in enumerate(self._joint_names)
            }
            goal_pos_deg = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        goal_joints_deg = [goal_pos_deg[name] for name in self._joint_names]
        success = self.robot_api.set_robot_state(
            target_joints=goal_joints_deg,
            gripper_value=gripper_value,
            joint_format="deg",
            speed=self.config.speed,
            gripper_speed=self.config.gripper_speed,
            wait_for_completion=False,
            use_interpolation=self.config.use_interpolation,
        )
        if not success:
            raise RuntimeError(f"Failed to send action to {self}.")

        sent_action: RobotAction = {
            f"{joint_name}.pos": float(goal_joints_deg[i]) for i, joint_name in enumerate(self._joint_names)
        }
        sent_action[f"{self._gripper_name}.pos"] = gripper_value
        return sent_action

    def _disconnect_resources(self) -> None:
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()
        api_connected = False
        with suppress(Exception):
            api_connected = bool(self.robot_api.is_connected())
        if self._arm_connected or api_connected:
            if self.config.disable_torque_on_disconnect:
                try:
                    self.robot_api.torque_control("off")
                except Exception as exc:
                    logger.warning(f"Failed to disable Alicia-M torque: {exc}")
            try:
                self.robot_api.disconnect()
            finally:
                self._arm_connected = False

    def disconnect(self) -> None:
        self._disconnect_resources()
        logger.info(f"{self} disconnected.")
