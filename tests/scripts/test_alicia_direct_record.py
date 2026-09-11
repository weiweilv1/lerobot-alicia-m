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

from contextlib import contextmanager

import pytest

from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators import Teleoperator


class FakeTimer:
    def tick(self):
        pass

    @contextmanager
    def section(self, _name):
        yield

    def wait(self):
        pass


class FakeRobot:
    name = "fake_robot"
    uses_teleop_state_for_observation = True

    def __init__(self):
        self.send_calls = []

    def get_observation(self):
        return {"joint1.pos": 0.0, "camera": "preserved"}

    def send_action(self, action):
        self.send_calls.append(action)
        return action


class FakeTeleoperator(Teleoperator):
    config_class = object
    name = "fake_direct_teleop"

    def __init__(self, actions, direct=True, delay_frames=0):
        self.actions = iter(actions)
        self.direct = direct
        self.delay_frames = delay_frames

    @property
    def action_features(self):
        return {"joint1.pos": float}

    @property
    def feedback_features(self):
        return {}

    @property
    def directly_controls_robot(self):
        return self.direct

    @property
    def uses_action_as_observation(self):
        return True

    @property
    def action_observation_delay_frames(self):
        return self.delay_frames

    @property
    def is_connected(self):
        return True

    def connect(self, calibrate=True):
        pass

    @property
    def is_calibrated(self):
        return True

    def calibrate(self):
        pass

    def configure(self):
        pass

    def get_action(self):
        return next(self.actions)

    def send_feedback(self, feedback):
        pass

    def disconnect(self):
        pass


class FakeDataset:
    fps = 30
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["joint1.pos"],
        },
        "action": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["joint1.pos"],
        },
    }

    def __init__(self, events, stop_after=1):
        self.events = events
        self.stop_after = stop_after
        self.frames = []

    def add_frame(self, frame):
        self.frames.append(frame)
        if len(self.frames) >= self.stop_after:
            self.events["exit_early"] = True


def _identity_tuple_processor(data):
    return data[0]


def _identity_processor(data):
    return data


def _run_loop(robot, teleop, dataset, events):
    record_loop(
        robot=robot,
        events=events,
        fps=30,
        teleop_action_processor=_identity_tuple_processor,
        robot_action_processor=_identity_tuple_processor,
        robot_observation_processor=_identity_processor,
        teleop=teleop,
        dataset=dataset,
        control_time_s=10,
        single_task="test",
        timer=FakeTimer(),
    )


def test_direct_wire_recording_uses_delayed_leader_state_and_skips_send():
    events = {"exit_early": False}
    dataset = FakeDataset(events, stop_after=2)
    robot = FakeRobot()
    teleop = FakeTeleoperator([{"joint1.pos": 10.0}, {"joint1.pos": 20.0}], direct=True, delay_frames=1)

    _run_loop(robot, teleop, dataset, events)

    assert robot.send_calls == []
    assert dataset.frames[0]["observation.state"][0] == pytest.approx(10.0)
    assert dataset.frames[1]["observation.state"][0] == pytest.approx(10.0)
    assert dataset.frames[1]["action"][0] == pytest.approx(20.0)


def test_computer_mediated_recording_keeps_robot_observation_and_sends_action():
    events = {"exit_early": False}
    dataset = FakeDataset(events)
    robot = FakeRobot()
    teleop = FakeTeleoperator([{"joint1.pos": 10.0}], direct=False)

    _run_loop(robot, teleop, dataset, events)

    assert robot.send_calls == [{"joint1.pos": 10.0}]
    assert dataset.frames[0]["observation.state"][0] == pytest.approx(0.0)
    assert dataset.frames[0]["action"][0] == pytest.approx(10.0)


def test_computer_mediated_recording_stores_action_returned_by_robot():
    class ClippingRobot(FakeRobot):
        def send_action(self, action):
            self.send_calls.append(action)
            return {"joint1.pos": 5.0}

    events = {"exit_early": False}
    dataset = FakeDataset(events)
    robot = ClippingRobot()
    teleop = FakeTeleoperator([{"joint1.pos": 10.0}], direct=False)

    _run_loop(robot, teleop, dataset, events)

    assert robot.send_calls == [{"joint1.pos": 10.0}]
    assert dataset.frames[0]["action"][0] == pytest.approx(5.0)
