#!/usr/bin/env python

from contextlib import nullcontext
from unittest.mock import MagicMock

from lerobot.scripts import lerobot_replay
from lerobot.scripts.lerobot_replay import DatasetReplayConfig, ReplayConfig
from lerobot.utils.constants import ACTION


def test_replay_aligns_alicia_m_before_sending_first_action(monkeypatch):
    robot = MagicMock()
    robot.requires_teleop_startup_alignment = True
    robot.get_observation.return_value = {"joint1.pos": 0.0}
    robot.is_connected = True

    dataset = MagicMock()
    dataset.fps = 30
    dataset.num_frames = 1
    dataset.features = {ACTION: {"names": ["joint1.pos"]}}
    dataset.select_columns.return_value = [{ACTION: [12.0]}]

    timer = MagicMock()
    timer.section.side_effect = lambda _name: nullcontext()

    monkeypatch.setattr(lerobot_replay, "make_robot_from_config", lambda _cfg: robot)
    monkeypatch.setattr(lerobot_replay, "LeRobotDataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(lerobot_replay, "make_default_robot_action_processor", lambda: lambda item: item[0])
    monkeypatch.setattr(lerobot_replay, "CycleTimer", lambda *args, **kwargs: timer)
    monkeypatch.setattr(lerobot_replay, "log_say", lambda *args, **kwargs: None)

    replay_cfg = ReplayConfig(
        robot=MagicMock(),
        dataset=DatasetReplayConfig(repo_id="user/alicia", episode=0),
        play_sounds=False,
    )
    lerobot_replay.replay(replay_cfg)

    robot.prepare_for_teleoperation.assert_called_once_with({"joint1.pos": 12.0})
    robot.send_action.assert_called_once_with({"joint1.pos": 12.0})
    robot.disconnect.assert_called_once()
