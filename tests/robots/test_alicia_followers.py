#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.alicia_d_follower import AliciaDFollower, AliciaDFollowerConfig
from lerobot.robots.alicia_m_follower import AliciaMFollower, AliciaMFollowerConfig
from lerobot.robots.bi_alicia_d_follower import BiAliciaDFollower, BiAliciaDFollowerConfig
from lerobot.robots.bi_alicia_m_follower import BiAliciaMFollower, BiAliciaMFollowerConfig
from lerobot.robots.utils import prepare_robot_for_teleoperation

_D_MODULE = "lerobot.robots.alicia_d_follower.alicia_d_follower"
_M_MODULE = "lerobot.robots.alicia_m_follower.alicia_m_follower"


class FakeAliciaApi:
    def __init__(self, connect_result: bool = True, set_result: bool = True):
        self.connected = False
        self.connect_result = connect_result
        self.set_result = set_result
        self.state = {
            "angles": [math.radians(value) for value in (1, 2, 3, 4, 5, 6)],
            "gripper": 250.0,
        }
        self.sent_commands: list[dict] = []
        self.torque_commands: list[str] = []

    def connect(self):
        self.connected = self.connect_result
        return self.connect_result

    def disconnect(self):
        self.connected = False

    def is_connected(self):
        return self.connected

    def get_robot_state(self, info_type="joint_gripper"):
        if info_type == "joint":
            return list(self.state["angles"])
        return dict(self.state)

    def set_robot_state(self, **kwargs):
        self.sent_commands.append(kwargs)
        return self.set_result

    def torque_control(self, command):
        self.torque_commands.append(command)


def _fake_sdk(name: str, apis: list[FakeAliciaApi]) -> ModuleType:
    module = ModuleType(name)
    module.create_robot = MagicMock(side_effect=apis)
    return module


def test_alicia_d_observation_and_partial_action_hold_current_position(tmp_path):
    api = FakeAliciaApi()
    sdk = _fake_sdk("alicia_d_sdk", [api])
    with patch(f"{_D_MODULE}.require_package"), patch.dict(sys.modules, {"alicia_d_sdk": sdk}):
        robot = AliciaDFollower(
            AliciaDFollowerConfig(port="COM1", calibration_dir=tmp_path, max_relative_target=None)
        )

    robot.connect(calibrate=False)
    observation = robot.get_observation()
    assert observation["joint1.pos"] == pytest.approx(1.0)
    assert observation["joint6.pos"] == pytest.approx(6.0)
    assert observation["gripper.pos"] == 250.0

    sent = robot.send_action({"joint1.pos": 30.0, "gripper.pos": 1200.0})
    command = api.sent_commands[-1]
    assert command["target_joints"] == pytest.approx([30.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert command["gripper_value"] == 1000.0
    assert command["joint_format"] == "deg"
    assert command["wait_for_completion"] is False
    assert sent["joint2.pos"] == pytest.approx(2.0)
    assert sent["gripper.pos"] == 1000.0

    robot.disconnect()
    assert not robot.is_connected


def test_alicia_d_camera_only_mode_does_not_open_or_command_arm(tmp_path):
    api = FakeAliciaApi()
    sdk = _fake_sdk("alicia_d_sdk", [api])
    with patch(f"{_D_MODULE}.require_package"), patch.dict(sys.modules, {"alicia_d_sdk": sdk}):
        robot = AliciaDFollower(AliciaDFollowerConfig(connect_arm=False, calibration_dir=tmp_path))

    robot.connect(calibrate=False)
    assert robot.is_connected
    assert not api.connected
    assert robot.get_observation()["joint1.pos"] == 0.0
    assert robot.send_action({"joint1.pos": 12.0}) == {"joint1.pos": 12.0}
    assert api.sent_commands == []
    robot.disconnect()


def test_alicia_d_retries_transient_reads_and_rejects_failed_or_nan_commands(tmp_path):
    api = FakeAliciaApi()
    sdk = _fake_sdk("alicia_d_sdk", [api])
    with patch(f"{_D_MODULE}.require_package"), patch.dict(sys.modules, {"alicia_d_sdk": sdk}):
        robot = AliciaDFollower(
            AliciaDFollowerConfig(
                port="COM1",
                calibration_dir=tmp_path,
                read_retry_interval_s=0.0,
            )
        )

    robot.connect(calibrate=False)
    valid_state = dict(api.state)
    api.get_robot_state = MagicMock(side_effect=[RuntimeError("timeout"), valid_state])
    assert robot.get_observation()["joint1.pos"] == pytest.approx(1.0)
    assert api.get_robot_state.call_count == 2
    api.get_robot_state = MagicMock(return_value=valid_state)

    with pytest.raises(ValueError, match="non-finite action"):
        robot.send_action({"joint1.pos": math.nan})
    api.set_result = False
    with pytest.raises(RuntimeError, match="Failed to send action"):
        robot.send_action({"joint1.pos": 1.0})
    robot.disconnect()


def test_alicia_m_uses_current_sdk_arguments_and_action_api(tmp_path):
    api = FakeAliciaApi()
    sdk = _fake_sdk("alicia_m_sdk", [api])
    with patch(f"{_M_MODULE}.require_package"), patch.dict(sys.modules, {"alicia_m_sdk": sdk}):
        robot = AliciaMFollower(AliciaMFollowerConfig(port="COM2", calibration_dir=tmp_path))

    create_kwargs = sdk.create_robot.call_args.kwargs
    assert create_kwargs["version"] == "auto"
    assert create_kwargs["control_aim"] == "follower"
    assert create_kwargs["sync_control_mode"] is True
    assert create_kwargs["backend"] == "cpp"
    assert "base_link" not in create_kwargs
    assert "end_link" not in create_kwargs
    assert "skip_mit_init" not in create_kwargs

    robot.connect(calibrate=False)
    sent = robot.send_action({"joint6.pos": -20.0})
    command = api.sent_commands[-1]
    assert command["target_joints"] == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, -20.0])
    assert command["gripper_value"] == 250.0
    assert command["gripper_speed"] == robot.config.gripper_speed
    assert command["use_interpolation"] is False
    assert sent["gripper.pos"] == 250.0
    robot.disconnect()


def test_alicia_m_smoothly_aligns_before_teleoperation(tmp_path):
    api = FakeAliciaApi()
    sdk = _fake_sdk("alicia_m_sdk", [api])
    with patch(f"{_M_MODULE}.require_package"), patch.dict(sys.modules, {"alicia_m_sdk": sdk}):
        robot = AliciaMFollower(
            AliciaMFollowerConfig(
                port="COM2",
                calibration_dir=tmp_path,
                speed=25.0,
                gripper_speed=30.0,
            )
        )

    robot.connect(calibrate=False)
    moved = robot.prepare_for_teleoperation(
        {
            "joint1.pos": 40.0,
            "joint2.pos": 2.0,
            "joint3.pos": 3.0,
            "joint4.pos": 4.0,
            "joint5.pos": 5.0,
            "joint6.pos": 6.0,
            "gripper.pos": 400.0,
        }
    )

    assert moved is True
    command = api.sent_commands[-1]
    assert command["target_joints"] == pytest.approx([40.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert command["speed"] == 25.0
    assert command["gripper_speed"] == 30.0
    assert command["wait_for_completion"] is True
    assert command["use_interpolation"] is True
    robot.disconnect()


def test_alicia_m_skips_startup_move_when_pose_is_within_threshold(tmp_path):
    api = FakeAliciaApi()
    sdk = _fake_sdk("alicia_m_sdk", [api])
    with patch(f"{_M_MODULE}.require_package"), patch.dict(sys.modules, {"alicia_m_sdk": sdk}):
        robot = AliciaMFollower(AliciaMFollowerConfig(port="COM2", calibration_dir=tmp_path))

    robot.connect(calibrate=False)
    moved = robot.prepare_for_teleoperation(
        {
            "joint1.pos": 2.5,
            "joint2.pos": 2.0,
            "joint3.pos": 3.0,
            "joint4.pos": 4.0,
            "joint5.pos": 5.0,
            "joint6.pos": 6.0,
            "gripper.pos": 275.0,
        }
    )

    assert moved is False
    assert api.sent_commands == []
    robot.disconnect()


def test_alicia_m_rejects_failed_or_non_finite_commands(tmp_path):
    api = FakeAliciaApi()
    sdk = _fake_sdk("alicia_m_sdk", [api])
    with patch(f"{_M_MODULE}.require_package"), patch.dict(sys.modules, {"alicia_m_sdk": sdk}):
        robot = AliciaMFollower(AliciaMFollowerConfig(port="COM2", calibration_dir=tmp_path))

    robot.connect(calibrate=False)
    with pytest.raises(ValueError, match="non-finite action"):
        robot.send_action({"joint1.pos": math.inf})
    api.set_result = False
    with pytest.raises(RuntimeError, match="Failed to send action"):
        robot.send_action({"joint1.pos": 1.0})
    robot.disconnect()


def test_teleoperation_preparation_uses_the_normal_processor_path():
    robot = MagicMock()
    robot.requires_teleop_startup_alignment = True
    robot.get_observation.return_value = {"joint1.pos": 1.0}
    robot.prepare_for_teleoperation.return_value = True
    teleop = MagicMock()
    teleop.directly_controls_robot = False
    teleop.get_action.return_value = {"joint1.pos": 10.0}

    moved = prepare_robot_for_teleoperation(
        robot,
        teleop,
        lambda transition: {"joint1.pos": transition[0]["joint1.pos"] + 1.0},
        lambda transition: {"joint1.pos": transition[0]["joint1.pos"] + 2.0},
    )

    assert moved is True
    robot.prepare_for_teleoperation.assert_called_once_with({"joint1.pos": 13.0})


def test_bimanual_alicia_features_are_prefixed(tmp_path):
    apis = [FakeAliciaApi(), FakeAliciaApi()]
    sdk = _fake_sdk("alicia_d_sdk", apis)
    with patch(f"{_D_MODULE}.require_package"), patch.dict(sys.modules, {"alicia_d_sdk": sdk}):
        robot = BiAliciaDFollower(BiAliciaDFollowerConfig(calibration_dir=tmp_path))

    assert "left_joint1.pos" in robot.action_features
    assert "left_gripper.pos" in robot.action_features
    assert "right_joint6.pos" in robot.action_features
    assert "right_gripper.pos" in robot.observation_features


def test_bimanual_alicia_m_routes_observations_and_actions(tmp_path):
    left_api, right_api = FakeAliciaApi(), FakeAliciaApi()
    sdk = _fake_sdk("alicia_m_sdk", [left_api, right_api])
    with patch(f"{_M_MODULE}.require_package"), patch.dict(sys.modules, {"alicia_m_sdk": sdk}):
        robot = BiAliciaMFollower(
            BiAliciaMFollowerConfig(
                left_arm_port="COM2",
                right_arm_port="COM3",
                calibration_dir=tmp_path,
                left_arm_max_relative_target=None,
                right_arm_max_relative_target=None,
            )
        )

    assert sdk.create_robot.call_args_list[0].kwargs["port"] == "COM2"
    assert sdk.create_robot.call_args_list[1].kwargs["port"] == "COM3"
    assert "left_joint1.pos" in robot.action_features
    assert "right_gripper.pos" in robot.observation_features

    robot.connect(calibrate=False)
    observation = robot.get_observation()
    assert observation["left_joint1.pos"] == pytest.approx(1.0)
    assert observation["right_joint6.pos"] == pytest.approx(6.0)

    aligned = robot.prepare_for_teleoperation(
        {
            "left_joint1.pos": 20.0,
            "right_joint6.pos": -20.0,
        }
    )
    assert aligned is True
    assert left_api.sent_commands[-1]["wait_for_completion"] is True
    assert right_api.sent_commands[-1]["wait_for_completion"] is True
    assert left_api.sent_commands[-1]["use_interpolation"] is True
    assert right_api.sent_commands[-1]["use_interpolation"] is True

    sent = robot.send_action(
        {
            "left_joint1.pos": 11.0,
            "left_gripper.pos": 300.0,
            "right_joint6.pos": -12.0,
            "right_gripper.pos": 400.0,
        }
    )
    assert left_api.sent_commands[-1]["target_joints"] == pytest.approx([11.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert right_api.sent_commands[-1]["target_joints"] == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, -12.0])
    assert sent["left_gripper.pos"] == 300.0
    assert sent["right_gripper.pos"] == 400.0

    robot.disconnect()
    assert not left_api.connected
    assert not right_api.connected


def test_bimanual_alicia_m_rolls_back_partial_connection(tmp_path):
    left_api, right_api = FakeAliciaApi(), FakeAliciaApi(connect_result=False)
    sdk = _fake_sdk("alicia_m_sdk", [left_api, right_api])
    with patch(f"{_M_MODULE}.require_package"), patch.dict(sys.modules, {"alicia_m_sdk": sdk}):
        robot = BiAliciaMFollower(
            BiAliciaMFollowerConfig(
                left_arm_port="COM2",
                right_arm_port="COM3",
                calibration_dir=tmp_path,
            )
        )

    with pytest.raises(ConnectionError, match="Failed to connect to Alicia-M"):
        robot.connect(calibrate=False)

    assert not left_api.connected
    assert not right_api.connected
    assert robot._executor is None


def test_bimanual_alicia_m_cleans_remaining_resources_after_one_side_drops(tmp_path):
    left_api, right_api = FakeAliciaApi(), FakeAliciaApi()
    sdk = _fake_sdk("alicia_m_sdk", [left_api, right_api])
    with patch(f"{_M_MODULE}.require_package"), patch.dict(sys.modules, {"alicia_m_sdk": sdk}):
        robot = BiAliciaMFollower(
            BiAliciaMFollowerConfig(
                left_arm_port="COM2",
                right_arm_port="COM3",
                calibration_dir=tmp_path,
            )
        )

    robot.connect(calibrate=False)
    left_api.connected = False
    assert not robot.is_connected
    assert robot.has_connected_resources

    robot.disconnect()
    assert not right_api.connected
    assert robot._executor is None


def test_bimanual_alicia_m_rejects_ambiguous_ports(tmp_path):
    with pytest.raises(ValueError, match="different serial ports"):
        BiAliciaMFollowerConfig(
            left_arm_port="COM2",
            right_arm_port="com2",
            calibration_dir=tmp_path,
        )

    with pytest.raises(ValueError, match="left_arm_port is required"):
        BiAliciaMFollowerConfig(
            left_arm_connect=True,
            right_arm_port="COM3",
            calibration_dir=tmp_path,
        )


def test_bimanual_alicia_d_rejects_ambiguous_active_ports(tmp_path):
    with pytest.raises(ValueError, match="different serial ports"):
        BiAliciaDFollowerConfig(
            left_arm_port="COM2",
            right_arm_port="com2",
            calibration_dir=tmp_path,
        )

    with pytest.raises(ValueError, match="left_arm_port is required"):
        BiAliciaDFollowerConfig(
            left_arm_connect=True,
            right_arm_port="COM3",
            calibration_dir=tmp_path,
        )
