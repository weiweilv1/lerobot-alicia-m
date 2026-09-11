#!/usr/bin/env python
"""Bimanual Alicia-D Leader Arms - LeRobot Teleoperator Integration

This module provides LeRobot-compatible teleoperator interface for bimanual Alicia-D leader arms,
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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from functools import cached_property
from typing import Any

from lerobot.lerobot_types import RobotAction
from lerobot.teleoperators.alicia_d_leader import AliciaDLeader
from lerobot.teleoperators.alicia_d_leader.config_alicia_d_leader import AliciaDLeaderConfig
from lerobot.utils.bimanual import BimanualMixin
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_bi_alicia_d_leader import BiAliciaDLeaderConfig

logger = logging.getLogger(__name__)


class BiAliciaDLeader(BimanualMixin, Teleoperator):
    """
    Bimanual Alicia-D Leader Arms - LeRobot teleoperator integration using Alicia-D SDK.

    This bimanual teleoperator uses two Alicia-D leader arms (left and right) and provides
    a unified interface for reading joint positions and button status from both arms.
    Leader arms use the same API as follower arms but don't have pose data.
    """

    config_class = BiAliciaDLeaderConfig
    name = "bi_alicia_d_leader"

    def __init__(self, config: BiAliciaDLeaderConfig):
        super().__init__(config)
        self.config = config
        self._executor: ThreadPoolExecutor | None = None

        # Create left arm configuration
        left_arm_config = AliciaDLeaderConfig(
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
            gripper_type=config.left_arm_gripper_type,
            debug_mode=config.left_arm_debug_mode,
            directly_controls_robot=config.directly_controls_robot,
            use_action_as_observation=config.use_action_as_observation,
            action_observation_delay_frames=config.action_observation_delay_frames,
            target_follower_type=config.target_follower_type,
            read_retry_count=config.read_retry_count,
            read_retry_interval_s=config.read_retry_interval_s,
            max_consecutive_read_failures=config.max_consecutive_read_failures,
        )

        # Create right arm configuration
        right_arm_config = AliciaDLeaderConfig(
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
            gripper_type=config.right_arm_gripper_type,
            debug_mode=config.right_arm_debug_mode,
            directly_controls_robot=config.directly_controls_robot,
            use_action_as_observation=config.use_action_as_observation,
            action_observation_delay_frames=config.action_observation_delay_frames,
            target_follower_type=config.target_follower_type,
            read_retry_count=config.read_retry_count,
            read_retry_interval_s=config.read_retry_interval_s,
            max_consecutive_read_failures=config.max_consecutive_read_failures,
        )

        # Create left and right arm instances
        self.left_arm = AliciaDLeader(left_arm_config)
        self.right_arm = AliciaDLeader(right_arm_config)

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bi-alicia-d-leader")
        return self._executor

    def _run_parallel(self, left_call: Callable[[], Any], right_call: Callable[[], Any]) -> tuple[Any, Any]:
        executor = self._ensure_executor()
        futures = [executor.submit(left_call), executor.submit(right_call)]
        wait(futures)
        errors = [error for future in futures if (error := future.exception()) is not None]
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("Both Alicia-D leaders failed", errors)
        return futures[0].result(), futures[1].result()

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action features dictionary with left_ and right_ prefixes."""
        # Remove ".pos" suffix from keys, then add prefix
        left_ft = {f"left_{key}": value for key, value in self.left_arm.action_features.items()}
        right_ft = {f"right_{key}": value for key, value in self.right_arm.action_features.items()}
        return {**left_ft, **right_ft}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        """Feedback features dictionary (not used for leader arms)."""
        return {}

    @property
    def directly_controls_robot(self) -> bool:
        """
        Whether the leader arms directly control the follower arms via hardware wire.

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
    def has_connected_resources(self) -> bool:
        return (
            self.left_arm.has_connected_resources
            or self.right_arm.has_connected_resources
            or self._executor is not None
        )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        try:
            self._run_parallel(
                lambda: self.left_arm.connect(calibrate),
                lambda: self.right_arm.connect(calibrate),
            )
            self.configure()
        except Exception:
            self._disconnect_resources()
            raise
        logger.info(f"{self} connected.")

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """
        Get current action from both leader arms.

        Reads joint positions and gripper values from both leader arms.
        Leader arms don't have pose data, only joint positions.

        Returns:
            Dictionary with left_ and right_ prefixed joint positions and gripper values.
        """
        left_action, right_action = self._run_parallel(
            self.left_arm.get_action,
            self.right_arm.get_action,
        )
        return {
            **{f"left_{key}": value for key, value in left_action.items()},
            **{f"right_{key}": value for key, value in right_action.items()},
        }

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, float]) -> None:
        """
        Send feedback to both leader arms.

        Note: Force feedback is not currently implemented for Alicia-D leader arms.
        """
        # Remove "left_" prefix
        left_feedback = {
            key.removeprefix("left_"): value for key, value in feedback.items() if key.startswith("left_")
        }
        # Remove "right_" prefix
        right_feedback = {
            key.removeprefix("right_"): value for key, value in feedback.items() if key.startswith("right_")
        }

        if left_feedback:
            self.left_arm.send_feedback(left_feedback)
        if right_feedback:
            self.right_arm.send_feedback(right_feedback)

    def _disconnect_resources(self) -> None:
        if self._executor is not None:
            calls = [arm.disconnect for arm in (self.left_arm, self.right_arm) if arm.has_connected_resources]
            futures = [self._executor.submit(call) for call in calls]
            wait(futures)
            for future in futures:
                if (error := future.exception()) is not None:
                    logger.warning("Failed to disconnect an Alicia-D leader: %s", error)
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def disconnect(self) -> None:
        self._disconnect_resources()
        logger.info(f"{self} disconnected.")
