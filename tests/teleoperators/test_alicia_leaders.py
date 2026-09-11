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
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from lerobot.teleoperators.alicia_d_leader import AliciaDLeader, AliciaDLeaderConfig
from lerobot.teleoperators.alicia_d_leader.alicia_d_leader import URDF_LIMIT
from lerobot.teleoperators.bi_alicia_d_leader import BiAliciaDLeader, BiAliciaDLeaderConfig

_D_MODULE = "lerobot.teleoperators.alicia_d_leader.alicia_d_leader"


class FakeLeaderApi:
    def __init__(self):
        self.connected = False
        self.state = SimpleNamespace(
            angles=[math.radians(value) for value in (10, 20, 30, 40, 50, 60)],
            gripper=400.0,
            run_status_text="idle",
        )

    def connect(self):
        self.connected = True
        return True

    def disconnect(self):
        self.connected = False

    def is_connected(self):
        return self.connected

    def get_robot_state(self, _info_type):
        return self.state


def _fake_sdk(apis: list[FakeLeaderApi]) -> ModuleType:
    module = ModuleType("alicia_d_sdk")
    module.create_robot = MagicMock(side_effect=apis)
    return module


def test_alicia_d_leader_reads_degrees_and_exposes_direct_wire_settings(tmp_path):
    api = FakeLeaderApi()
    sdk = _fake_sdk([api])
    config = AliciaDLeaderConfig(
        port="COM3",
        calibration_dir=tmp_path,
        directly_controls_robot=True,
        use_action_as_observation=True,
        action_observation_delay_frames=2,
    )
    with patch(f"{_D_MODULE}.require_package"), patch.dict(sys.modules, {"alicia_d_sdk": sdk}):
        leader = AliciaDLeader(config)

    leader.connect(calibrate=False)
    action = leader.get_action()
    assert action["joint1.pos"] == pytest.approx(10.0)
    assert action["joint6.pos"] == pytest.approx(60.0)
    assert action["gripper.pos"] == 400.0
    assert leader.directly_controls_robot
    assert leader.uses_action_as_observation
    assert leader.action_observation_delay_frames == 2
    leader.disconnect()


def test_alicia_d_to_m_mapping_stays_within_follower_limits(tmp_path):
    api = FakeLeaderApi()
    sdk = _fake_sdk([api])
    with patch(f"{_D_MODULE}.require_package"), patch.dict(sys.modules, {"alicia_d_sdk": sdk}):
        leader = AliciaDLeader(
            AliciaDLeaderConfig(
                port="COM3",
                calibration_dir=tmp_path,
                target_follower_type="alicia_m",
            )
        )

    leader.connect(calibrate=False)
    action = leader.get_action()
    for index in range(6):
        limits = URDF_LIMIT["ALICIA_M"][index]
        assert limits["lower"] <= action[f"joint{index + 1}.pos"] <= limits["upper"]
    leader.disconnect()


def test_bimanual_alicia_leader_prefixes_actions(tmp_path):
    sdk = _fake_sdk([FakeLeaderApi(), FakeLeaderApi()])
    with patch(f"{_D_MODULE}.require_package"), patch.dict(sys.modules, {"alicia_d_sdk": sdk}):
        leader = BiAliciaDLeader(
            BiAliciaDLeaderConfig(
                left_arm_port="COM3",
                right_arm_port="COM4",
                calibration_dir=tmp_path,
            )
        )

    leader.connect(calibrate=False)
    action = leader.get_action()
    assert action["left_joint1.pos"] == pytest.approx(10.0)
    assert action["right_joint6.pos"] == pytest.approx(60.0)
    assert "left_gripper.pos" in leader.action_features
    assert "right_gripper.pos" in leader.action_features
    leader.disconnect()


def test_bimanual_alicia_leader_cleans_other_side_after_partial_drop(tmp_path):
    left_api, right_api = FakeLeaderApi(), FakeLeaderApi()
    sdk = _fake_sdk([left_api, right_api])
    with patch(f"{_D_MODULE}.require_package"), patch.dict(sys.modules, {"alicia_d_sdk": sdk}):
        leader = BiAliciaDLeader(
            BiAliciaDLeaderConfig(
                left_arm_port="COM3",
                right_arm_port="COM4",
                calibration_dir=tmp_path,
            )
        )

    leader.connect(calibrate=False)
    left_api.connected = False
    assert not leader.is_connected
    assert leader.has_connected_resources

    leader.disconnect()
    assert not right_api.connected
    assert leader._executor is None


def test_bimanual_alicia_leader_requires_two_distinct_ports(tmp_path):
    with pytest.raises(ValueError, match="Both left_arm_port and right_arm_port"):
        BiAliciaDLeaderConfig(left_arm_port="COM3", calibration_dir=tmp_path)

    with pytest.raises(ValueError, match="different serial ports"):
        BiAliciaDLeaderConfig(
            left_arm_port="COM3",
            right_arm_port="com3",
            calibration_dir=tmp_path,
        )


def test_alicia_d_leader_rejects_non_finite_state(tmp_path):
    api = FakeLeaderApi()
    api.state.angles[0] = math.nan
    sdk = _fake_sdk([api])
    with patch(f"{_D_MODULE}.require_package"), patch.dict(sys.modules, {"alicia_d_sdk": sdk}):
        leader = AliciaDLeader(AliciaDLeaderConfig(port="COM3", calibration_dir=tmp_path))

    leader.connect(calibrate=False)
    with pytest.raises(ValueError, match="non-finite"):
        leader.get_action()
    leader.disconnect()
