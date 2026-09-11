#!/usr/bin/env python
"""Bimanual Alicia-D Follower Arms - LeRobot Integration

This module provides LeRobot-compatible interface for bimanual Alicia-D robot arms,
using the SynriaRobotAPI from Alicia-D SDK.

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

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots.alicia_d_follower import AliciaDFollower
from lerobot.robots.alicia_d_follower.config_alicia_d_follower import AliciaDFollowerConfig
from lerobot.utils.bimanual import BimanualMixin
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_bi_alicia_d_follower import BiAliciaDFollowerConfig

logger = logging.getLogger(__name__)


class BiAliciaDFollower(BimanualMixin, Robot):
    """
    Bimanual Alicia-D Follower Arms - LeRobot integration using SynriaRobotAPI from Alicia-D SDK.

    This bimanual robot uses two Alicia-D follower arms (left and right) and provides
    a unified interface for controlling both arms simultaneously.
    """

    config_class = BiAliciaDFollowerConfig
    name = "bi_alicia_d_follower"

    def __init__(self, config: BiAliciaDFollowerConfig):
        super().__init__(config)
        self.config = config
        self._executor: ThreadPoolExecutor | None = None

        # Create left arm configuration
        left_arm_config = AliciaDFollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_port,
            version=config.version,
            variant=config.variant,
            model_format=config.model_format,
            base_link=config.base_link,
            end_link=config.end_link,
            backend=config.backend,
            device=config.device,
            connect_arm=config.left_arm_connect,
            disable_torque_on_disconnect=config.left_arm_disable_torque_on_disconnect,
            max_relative_target=config.left_arm_max_relative_target,
            gripper_type=config.left_arm_gripper_type,
            debug_mode=config.left_arm_debug_mode,
            speed_deg_s=config.left_arm_speed_deg_s,
            gripper_speed_deg_s=config.left_arm_gripper_speed_deg_s,
            read_retry_count=config.read_retry_count,
            read_retry_interval_s=config.read_retry_interval_s,
            cameras={},
        )

        # Create right arm configuration
        right_arm_config = AliciaDFollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_port,
            version=config.version,
            variant=config.variant,
            model_format=config.model_format,
            base_link=config.base_link,
            end_link=config.end_link,
            backend=config.backend,
            device=config.device,
            connect_arm=config.right_arm_connect,
            disable_torque_on_disconnect=config.right_arm_disable_torque_on_disconnect,
            max_relative_target=config.right_arm_max_relative_target,
            gripper_type=config.right_arm_gripper_type,
            debug_mode=config.right_arm_debug_mode,
            speed_deg_s=config.right_arm_speed_deg_s,
            gripper_speed_deg_s=config.right_arm_gripper_speed_deg_s,
            read_retry_count=config.read_retry_count,
            read_retry_interval_s=config.read_retry_interval_s,
            cameras={},
        )

        # Create left and right arm instances
        self.left_arm = AliciaDFollower(left_arm_config)
        self.right_arm = AliciaDFollower(right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bi-alicia-d")
        return self._executor

    def _run_parallel(self, left_call: Callable[[], Any], right_call: Callable[[], Any]) -> tuple[Any, Any]:
        futures = [self._ensure_executor().submit(left_call), self._ensure_executor().submit(right_call)]
        wait(futures)
        errors = [error for future in futures if (error := future.exception()) is not None]
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("Both Alicia-D arms failed", errors)
        return futures[0].result(), futures[1].result()

    @property
    def _motors_ft(self) -> dict[str, type]:
        left_ft = {f"left_{key}": value for key, value in self.left_arm.action_features.items()}
        right_ft = {f"right_{key}": value for key, value in self.right_arm.action_features.items()}
        return {**left_ft, **right_ft}

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
        """Observation features combining motors and cameras."""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action features (same as motor features)."""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.left_arm.is_connected
            and self.right_arm.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def has_connected_resources(self) -> bool:
        return (
            self.left_arm.has_connected_resources
            or self.right_arm.has_connected_resources
            or any(cam.is_connected for cam in self.cameras.values())
            or self._executor is not None
        )

    @property
    def uses_teleop_state_for_observation(self) -> bool:
        """Whether the recorder should replace joint observations with teleop state."""
        return self.config.use_teleop_state_for_observation

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        try:
            self._run_parallel(
                lambda: self.left_arm.connect(calibrate),
                lambda: self.right_arm.connect(calibrate),
            )
            for cam in self.cameras.values():
                cam.connect()
            self.configure()
        except Exception:
            self._disconnect_resources()
            raise
        logger.info(f"{self} connected.")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        left_obs, right_obs = self._run_parallel(
            self.left_arm.get_observation,
            self.right_arm.get_observation,
        )
        obs_dict: RobotObservation = {
            **{f"left_{key}": value for key, value in left_obs.items()},
            **{f"right_{key}": value for key, value in right_obs.items()},
        }

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
        """Send action to both arms.

        Args:
            action: Dictionary with "left_" and "right_" prefixed joint positions and gripper values.

        Returns:
            The action actually sent to both arms, potentially clipped.
        """
        # Remove "left_" prefix for left arm
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }

        # Remove "right_" prefix for right arm
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        # Independent serial links can be commanded concurrently, reducing
        # left/right skew and keeping bimanual control inside the frame budget.
        send_action_left, send_action_right = self._run_parallel(
            lambda: self.left_arm.send_action(left_action),
            lambda: self.right_arm.send_action(right_action),
        )

        # Add prefixes back to returned actions
        prefixed_send_action_left = {f"left_{key}": value for key, value in send_action_left.items()}
        prefixed_send_action_right = {f"right_{key}": value for key, value in send_action_right.items()}

        return {**prefixed_send_action_left, **prefixed_send_action_right}

    def _disconnect_resources(self) -> None:
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()
        if self._executor is not None:
            calls = [arm.disconnect for arm in (self.left_arm, self.right_arm) if arm.has_connected_resources]
            futures = [self._executor.submit(call) for call in calls]
            wait(futures)
            for future in futures:
                if (error := future.exception()) is not None:
                    logger.warning("Failed to disconnect an Alicia-D arm: %s", error)
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def disconnect(self) -> None:
        self._disconnect_resources()
        logger.info(f"{self} disconnected.")
