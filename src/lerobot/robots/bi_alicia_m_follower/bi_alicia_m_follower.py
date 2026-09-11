#!/usr/bin/env python

# Copyright (c) 2025 Synria Robotics Co., Ltd.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots.alicia_m_follower import AliciaMFollower
from lerobot.robots.alicia_m_follower.config_alicia_m_follower import AliciaMFollowerConfig
from lerobot.utils.bimanual import BimanualMixin
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_bi_alicia_m_follower import BiAliciaMFollowerConfig

logger = logging.getLogger(__name__)


class BiAliciaMFollower(BimanualMixin, Robot):
    """Two Alicia-M followers exposed as one LeRobot bimanual robot."""

    config_class = BiAliciaMFollowerConfig
    name = "bi_alicia_m_follower"

    def __init__(self, config: BiAliciaMFollowerConfig):
        super().__init__(config)
        self.config = config
        self._executor: ThreadPoolExecutor | None = None

        common = {
            "calibration_dir": config.calibration_dir,
            "version": config.version,
            "variant": config.variant,
            "baudrate": config.baudrate,
            "control_aim": config.control_aim,
            "control_mode": config.control_mode,
            "sync_control_mode": config.sync_control_mode,
            "backend": config.backend,
            "extended_polling": config.extended_polling,
            "use_interpolation": config.use_interpolation,
            "startup_alignment": config.startup_alignment,
            "startup_alignment_threshold_deg": config.startup_alignment_threshold_deg,
            "startup_gripper_alignment_threshold": config.startup_gripper_alignment_threshold,
            "read_retry_count": config.read_retry_count,
            "read_retry_interval_s": config.read_retry_interval_s,
            "cameras": {},
        }
        left_config = AliciaMFollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            port=config.left_arm_port,
            connect_arm=config.left_arm_connect,
            disable_torque_on_disconnect=config.left_arm_disable_torque_on_disconnect,
            max_relative_target=config.left_arm_max_relative_target,
            speed=config.left_arm_speed,
            gripper_speed=config.left_arm_gripper_speed,
            debug_mode=config.left_arm_debug_mode,
            **common,
        )
        right_config = AliciaMFollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            port=config.right_arm_port,
            connect_arm=config.right_arm_connect,
            disable_torque_on_disconnect=config.right_arm_disable_torque_on_disconnect,
            max_relative_target=config.right_arm_max_relative_target,
            speed=config.right_arm_speed,
            gripper_speed=config.right_arm_gripper_speed,
            debug_mode=config.right_arm_debug_mode,
            **common,
        )

        self.left_arm = AliciaMFollower(left_config)
        self.right_arm = AliciaMFollower(right_config)
        self.cameras = make_cameras_from_configs(config.cameras)

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bi-alicia-m")
        return self._executor

    def _run_parallel(self, left_call: Callable[[], Any], right_call: Callable[[], Any]) -> tuple[Any, Any]:
        executor = self._ensure_executor()
        futures = [executor.submit(left_call), executor.submit(right_call)]
        wait(futures)
        errors = [error for future in futures if (error := future.exception()) is not None]
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("Both Alicia-M arms failed", errors)
        return futures[0].result(), futures[1].result()

    @property
    def _motors_ft(self) -> dict[str, type]:
        left = {f"left_{key}": value for key, value in self.left_arm.action_features.items()}
        right = {f"right_{key}": value for key, value in self.right_arm.action_features.items()}
        return {**left, **right}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        features: dict[str, tuple] = {}
        for camera_name, camera in self.cameras.items():
            if getattr(camera, "use_rgb", True):
                config = self.config.cameras[camera_name]
                features[camera_name] = (config.height, config.width, 3)
            if getattr(camera, "use_depth", False):
                config = self.config.cameras[camera_name]
                features[f"{camera_name}_depth"] = (config.height, config.width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.left_arm.is_connected
            and self.right_arm.is_connected
            and all(camera.is_connected for camera in self.cameras.values())
        )

    @property
    def has_connected_resources(self) -> bool:
        return (
            self.left_arm.has_connected_resources
            or self.right_arm.has_connected_resources
            or any(camera.is_connected for camera in self.cameras.values())
            or self._executor is not None
        )

    @property
    def uses_teleop_state_for_observation(self) -> bool:
        return self.config.use_teleop_state_for_observation

    @property
    def requires_teleop_startup_alignment(self) -> bool:
        return (
            self.left_arm.requires_teleop_startup_alignment
            or self.right_arm.requires_teleop_startup_alignment
        )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        try:
            self._run_parallel(
                lambda: self.left_arm.connect(calibrate),
                lambda: self.right_arm.connect(calibrate),
            )
            for camera in self.cameras.values():
                camera.connect()
        except Exception:
            self._disconnect_resources()
            raise
        logger.info("%s connected.", self)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        left_observation, right_observation = self._run_parallel(
            self.left_arm.get_observation,
            self.right_arm.get_observation,
        )
        observation: RobotObservation = {
            **{f"left_{key}": value for key, value in left_observation.items()},
            **{f"right_{key}": value for key, value in right_observation.items()},
        }

        for camera_name, camera in self.cameras.items():
            if getattr(camera, "use_rgb", True):
                start = time.perf_counter()
                observation[camera_name] = camera.read_latest()
                logger.debug("%s read %s: %.1fms", self, camera_name, (time.perf_counter() - start) * 1e3)
            if getattr(camera, "use_depth", False):
                start = time.perf_counter()
                observation[f"{camera_name}_depth"] = camera.read_latest_depth()
                logger.debug(
                    "%s read %s depth: %.1fms",
                    self,
                    camera_name,
                    (time.perf_counter() - start) * 1e3,
                )
        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }
        sent_left, sent_right = self._run_parallel(
            lambda: self.left_arm.send_action(left_action),
            lambda: self.right_arm.send_action(right_action),
        )
        return {
            **{f"left_{key}": value for key, value in sent_left.items()},
            **{f"right_{key}": value for key, value in sent_right.items()},
        }

    @check_if_not_connected
    def prepare_for_teleoperation(self, action: RobotAction) -> bool:
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }
        left_moved, right_moved = self._run_parallel(
            lambda: self.left_arm.prepare_for_teleoperation(left_action),
            lambda: self.right_arm.prepare_for_teleoperation(right_action),
        )
        return bool(left_moved or right_moved)

    def _disconnect_resources(self) -> None:
        if self._executor is not None:
            calls: list[Callable[[], Any]] = []
            for arm in (self.left_arm, self.right_arm):
                if arm.has_connected_resources:
                    calls.append(arm.disconnect)
            futures = [self._executor.submit(call) for call in calls]
            wait(futures)
            for future in futures:
                if (error := future.exception()) is not None:
                    logger.warning("Failed to disconnect an Alicia-M arm: %s", error)

        for camera in self.cameras.values():
            if camera.is_connected:
                try:
                    camera.disconnect()
                except Exception as error:  # noqa: BLE001
                    logger.warning("Failed to disconnect an Alicia-M camera: %s", error)

        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def disconnect(self) -> None:
        self._disconnect_resources()
        logger.info("%s disconnected.", self)
