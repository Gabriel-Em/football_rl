"""Unit tests for the preallocated replay recorder (`recording.py`) and its env integration.

Verifies:
1. Frames materialise to the documented JSON frame schema with hand-computed rounding.
2. Per-step payloads are shared by the frames of one RL step; events are stored sparsely.
3. Capacity growth, reset, snapshot independence and degenerate inputs.
4. `FootballParallelEnv.replay_frames` exposes the recorded frames as an idempotent read-only view.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from football_rl.config import DEFAULT_CONFIG
from football_rl.env import FootballParallelEnv
from football_rl.recording import ReplayBuffer

PAYLOAD = {
    "player_rewards": [0.1, -0.2],
    "cum_rewards": [1.5, 2.5],
    "v_pred": None,
    "rew_breakdown": [{"r": 1.0}, {"r": 2.0}],
    "role_state": ["GK", "BALL"],
    "role_targets": [[1.0, 2.0], [3.0, 4.0]],
    "actions": [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
}


def _record(buf: ReplayBuffer, frame: int = 7, step_index: int = 1, events=None, ball_x: float = 100.5549) -> None:
    buf.record_frame(
        frame=frame,
        time=frame / 60.0,
        half=1,
        team_side=np.array([0, 1], dtype=np.int32),
        box_radii=np.array([12.34, 15.0]),
        player_pos=np.array([[12.3456, 7.891], [0.0, -3.14159]]),
        player_angle=np.array([0.123456, -3.14159265]),
        player_flags=np.array([5, 0]),
        player_vel=np.array([[1.239, -1.5], [0.0, 2.0]]),
        ball_pos=np.array([ball_x, 50.0]),
        ball_team=1,
        ball_vel=np.array([3.333, -4.444]),
        step_index=step_index,
        score=[1, 0],
        possession_frames=[3, 1],
        stationary=2,
        passes=[4, 5],
        tackles=[0, 1],
        saves=[2, 0],
        shots=[1, 1],
        events=events,
    )


EXPECTED_FRAME = {
    "frame": 7,
    "time": 0.117,
    "half": 1,
    "team_side": [0, 1],
    "box_radii": [12.3, 15.0],
    "players": [
        [12.35, 7.89, 0.1235, 5, 1.24, -1.5],
        [0.0, -3.14, -3.1416, 0, 0.0, 2.0],
    ],
    "ball": [100.55, 50.0, 1, 3.33, -4.44],
    "player_rewards": [0.1, -0.2],
    "cum_rewards": [1.5, 2.5],
    "v_pred": None,
    "rew_breakdown": [{"r": 1.0}, {"r": 2.0}],
    "role_state": ["GK", "BALL"],
    "role_targets": [[1.0, 2.0], [3.0, 4.0]],
    "actions": [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
    "stats": {
        "score": [1, 0],
        "possession_pct": [75.0, 25.0],
        "true_possession_pct": [42.9, 14.3, 42.9],
        "stationary_ball_frames": 2,
        "passes": [4, 5],
        "tackles": [0, 1],
        "saves": [2, 0],
        "shots": [1, 1],
    },
    "events": [],
}


def _buffer(max_frames: int = 8) -> ReplayBuffer:
    buf = ReplayBuffer(max_frames=max_frames, num_players=2)
    buf.set_step_payload(1, dict(PAYLOAD))
    return buf


def test_materialized_frame_matches_hand_computed_schema():
    buf = _buffer()
    _record(buf)
    frames = buf.materialize()
    assert frames == [EXPECTED_FRAME]
    assert list(frames[0].keys()) == list(EXPECTED_FRAME.keys())
    assert list(frames[0]["stats"].keys()) == list(EXPECTED_FRAME["stats"].keys())


def test_materialized_values_have_python_types():
    buf = _buffer()
    _record(buf)
    frame = buf.materialize()[0]
    assert type(frame["frame"]) is int
    assert type(frame["half"]) is int
    assert all(type(v) is int for v in frame["team_side"])
    assert type(frame["players"][0][3]) is int
    assert type(frame["players"][0][0]) is float
    assert type(frame["ball"][2]) is int
    assert type(frame["stats"]["stationary_ball_frames"]) is int
    assert all(type(v) is int for v in frame["stats"]["score"])


def test_empty_buffer_materializes_to_empty_list():
    buf = ReplayBuffer(max_frames=4, num_players=2)
    assert len(buf) == 0
    assert buf.materialize() == []


def test_frames_of_one_step_share_that_steps_payload():
    buf = ReplayBuffer(max_frames=8, num_players=2)
    buf.set_step_payload(0, {**PAYLOAD, "player_rewards": [0.0, 0.0]})
    buf.set_step_payload(1, {**PAYLOAD, "player_rewards": [1.0, 1.0]})
    buf.set_step_payload(2, {**PAYLOAD, "player_rewards": [2.0, 2.0]})
    for frame, step in ((0, 0), (1, 1), (2, 1), (3, 2)):
        _record(buf, frame=frame, step_index=step)
    rewards = [f["player_rewards"] for f in buf.materialize()]
    assert rewards == [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]]


def test_payload_set_after_frames_is_reflected_in_materialization():
    buf = ReplayBuffer(max_frames=4, num_players=2)
    buf.set_step_payload(1, {**PAYLOAD, "player_rewards": [9.0, 9.0]})
    _record(buf)
    assert buf.materialize()[0]["player_rewards"] == [9.0, 9.0]
    buf.set_step_payload(1, {**PAYLOAD, "player_rewards": [8.0, 8.0]})
    assert buf.materialize()[0]["player_rewards"] == [8.0, 8.0]


def test_events_are_stored_per_frame_and_appendable_to_last_frame():
    buf = _buffer()
    goal = {"type": "GOAL!", "x": 1.0, "y": 2.0, "team": 0, "agent": "a", "reward": None}
    _record(buf, frame=0, events=[goal])
    _record(buf, frame=1)
    full_time = {"type": "FULLTIME!", "x": 3.0, "y": 4.0, "team": -1, "agent": "referee", "reward": None}
    buf.append_event_to_last_frame(full_time)
    events = [f["events"] for f in buf.materialize()]
    assert events == [[goal], [full_time]]


def test_event_list_passed_to_record_is_copied():
    buf = _buffer()
    events = [{"type": "A"}]
    _record(buf, events=events)
    events.clear()
    assert buf.materialize()[0]["events"] == [{"type": "A"}]


def test_append_event_on_empty_buffer_is_rejected():
    buf = ReplayBuffer(max_frames=2, num_players=2)
    with pytest.raises(IndexError):
        buf.append_event_to_last_frame({"type": "FULLTIME!"})


def test_buffer_grows_past_initial_capacity_without_losing_frames():
    buf = _buffer(max_frames=1)
    for k in range(5):
        _record(buf, frame=k, ball_x=float(k))
    frames = buf.materialize()
    assert len(buf) == 5
    assert [f["frame"] for f in frames] == [0, 1, 2, 3, 4]
    assert [f["ball"][0] for f in frames] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_zero_capacity_is_usable():
    buf = ReplayBuffer(max_frames=0, num_players=2)
    buf.set_step_payload(1, dict(PAYLOAD))
    _record(buf)
    assert len(buf) == 1


def test_reset_clears_frames_events_and_payloads():
    buf = _buffer()
    _record(buf, events=[{"type": "A"}])
    buf.reset()
    assert len(buf) == 0
    assert buf.materialize() == []
    buf.set_step_payload(1, dict(PAYLOAD))
    _record(buf)
    assert buf.materialize() == [EXPECTED_FRAME]


def test_snapshot_is_independent_of_later_recording_and_reset():
    buf = _buffer()
    _record(buf, frame=7, ball_x=100.5549, events=[{"type": "A"}])
    snap = buf.snapshot()
    _record(buf, frame=8, ball_x=1.0)
    buf.append_event_to_last_frame({"type": "B"})
    buf.reset()
    _record(buf, frame=9, ball_x=2.0)
    frames = snap.materialize()
    assert len(frames) == 1
    assert frames[0]["ball"][0] == 100.55
    assert frames[0]["events"] == [{"type": "A"}]


def test_snapshot_of_empty_buffer_is_empty():
    assert ReplayBuffer(max_frames=3, num_players=2).snapshot().materialize() == []


def test_non_finite_values_materialize_without_raising():
    buf = _buffer()
    buf.record_frame(
        frame=1,
        time=0.0,
        half=1,
        team_side=np.array([0, 1], dtype=np.int32),
        box_radii=np.array([np.nan, np.inf]),
        player_pos=np.full((2, 2), np.nan),
        player_angle=np.array([np.inf, -np.inf]),
        player_flags=np.zeros(2),
        player_vel=np.zeros((2, 2)),
        ball_pos=np.zeros(2),
        ball_team=-1,
        ball_vel=np.zeros(2),
        step_index=1,
        score=[0, 0],
        possession_frames=[0, 0],
        stationary=0,
        passes=[0, 0],
        tackles=[0, 0],
        saves=[0, 0],
        shots=[0, 0],
    )
    frame = buf.materialize()[0]
    assert np.isnan(frame["players"][0][0])
    assert frame["players"][0][2] == float("inf")
    assert frame["stats"]["true_possession_pct"] == [0.0, 0.0, 100.0]


def test_materialize_is_idempotent_and_does_not_mutate_buffer():
    buf = _buffer()
    _record(buf)
    assert buf.materialize() == buf.materialize()
    assert len(buf) == 1


def _recording_env(steps: int, seed: int = 3) -> FootballParallelEnv:
    cfg = replace(
        DEFAULT_CONFIG,
        replay=replace(DEFAULT_CONFIG.replay, enable_recording=True, interest_threshold=float("inf")),
    )
    env = FootballParallelEnv(config=cfg)
    env.reset(seed=seed)
    acts = np.random.default_rng(seed).uniform(-1, 1, size=(steps, 12, 4))
    for k in range(steps):
        env.step_array(acts[k])
    return env


def test_env_frame_count_is_physics_frames_plus_one_and_property_is_idempotent():
    env = _recording_env(steps=5)
    frames = env.replay_frames
    assert len(frames) == env.physics_frame_count + 1 == 5 * DEFAULT_CONFIG.physics.frame_skip + 1
    assert [f["frame"] for f in frames] == list(range(len(frames)))
    assert env.replay_frames == frames
    assert env.replay_frames is env.replay_frames


def test_env_replay_frames_is_read_only_and_reset_starts_a_new_recording():
    env = _recording_env(steps=3)
    with pytest.raises(AttributeError):
        env.replay_frames = []
    env.reset(seed=4)
    assert len(env.replay_frames) == 1
    assert env.replay_frames[0]["frame"] == 0


def test_env_without_recording_exposes_no_frames():
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(config=cfg)
    env.reset(seed=1)
    env.step_array(np.zeros((12, 4)))
    assert env.replay_frames == []


def test_env_frames_of_one_rl_step_share_step_telemetry_and_reset_frame_has_initial_values():
    env = _recording_env(steps=2)
    frames = env.replay_frames
    skip = DEFAULT_CONFIG.physics.frame_skip
    assert frames[0]["player_rewards"] == [0.0] * 12
    assert frames[0]["cum_rewards"] == [0.0] * 12
    assert frames[0]["v_pred"] is None
    first_step = frames[1 : 1 + skip]
    second_step = frames[1 + skip : 1 + 2 * skip]
    assert all(f["player_rewards"] == first_step[0]["player_rewards"] for f in first_step)
    assert all(f["actions"] == second_step[0]["actions"] for f in second_step)
    assert first_step[0]["actions"] != second_step[0]["actions"]
    assert all(f["cum_rewards"] == second_step[0]["cum_rewards"] for f in second_step)


def test_env_full_time_event_is_on_the_last_frame_only():
    env = _recording_env(steps=DEFAULT_CONFIG.physics.max_rl_steps)
    frames = env.replay_frames
    assert len(frames) == DEFAULT_CONFIG.physics.max_physics_frames + 1
    full_time = [i for i, f in enumerate(frames) if any(e["type"] == "FULLTIME!" for e in f["events"])]
    assert full_time == [len(frames) - 1]


def test_env_deferred_replay_survives_the_next_reset():
    cfg = replace(
        DEFAULT_CONFIG,
        replay=replace(
            DEFAULT_CONFIG.replay,
            enable_recording=True,
            interest_threshold=0.0,
            save_best_only=True,
            save_dir="/nonexistent-should-not-be-written",
        ),
    )
    env = FootballParallelEnv(config=cfg)
    env.reset(seed=5)
    acts = np.random.default_rng(5).uniform(-1, 1, size=(DEFAULT_CONFIG.physics.max_rl_steps, 12, 4))
    for k in range(DEFAULT_CONFIG.physics.max_rl_steps):
        env.step_array(acts[k])
    assert env._deferred_replay is not None
    deferred_frames = env._deferred_replay["frames"].materialize()
    assert len(deferred_frames) == DEFAULT_CONFIG.physics.max_physics_frames + 1
    last_before = deferred_frames[-1]
    env.reset(seed=6)
    for k in range(10):
        env.step_array(acts[k])
    assert env._deferred_replay["frames"].materialize()[-1] == last_before


def test_frames_recorded_with_flight_state_carry_inflight_and_rounded_post_kick_timers():
    buf = ReplayBuffer(max_frames=2, num_players=2)
    buf.set_step_payload(1, PAYLOAD)
    kwargs = dict(
        frame=7, time=7 / 60.0, half=1, team_side=np.array([0, 1]), box_radii=np.array([12.34, 15.0]),
        player_pos=np.zeros((2, 2)), player_angle=np.zeros(2), player_flags=np.zeros(2),
        player_vel=np.zeros((2, 2)), ball_pos=np.zeros(2), ball_team=0, ball_vel=np.zeros(2), step_index=1,
        score=[0, 0], possession_frames=[0, 0], stationary=0, passes=[0, 0], tackles=[0, 0], saves=[0, 0],
        shots=[0, 0],
    )
    buf.record_frame(**kwargs, inflight=True, post_kick=np.array([0.12345, 0.5]))
    buf.record_frame(**kwargs, inflight=False, post_kick=np.array([0.0, 0.0004]))
    frames = buf.materialize()
    assert frames[0]["inflight"] is True and frames[0]["pk"] == [0.123, 0.5]
    assert frames[1]["inflight"] is False and frames[1]["pk"] == [0.0, 0.0]


def test_materialized_frames_carry_no_stored_gradient_arrows():
    buf = _buffer()
    _record(buf)
    assert "grad_phi" not in buf.materialize()[0]


def test_frames_recorded_without_flight_state_materialise_without_inflight_and_pk():
    buf = ReplayBuffer(max_frames=1, num_players=2)
    buf.set_step_payload(1, PAYLOAD)
    _record(buf)
    frame = buf.materialize()[0]
    assert "inflight" not in frame and "pk" not in frame


def test_flight_state_survives_growth_and_snapshot():
    buf = ReplayBuffer(max_frames=1, num_players=2)
    buf.set_step_payload(1, PAYLOAD)
    for k in range(3):
        buf.record_frame(
            frame=k, time=0.0, half=1, team_side=np.array([0, 1]), box_radii=np.zeros(2),
            player_pos=np.zeros((2, 2)), player_angle=np.zeros(2), player_flags=np.zeros(2),
            player_vel=np.zeros((2, 2)), ball_pos=np.zeros(2), ball_team=0, ball_vel=np.zeros(2), step_index=1,
            score=[0, 0], possession_frames=[0, 0], stationary=0, passes=[0, 0], tackles=[0, 0], saves=[0, 0],
            shots=[0, 0], inflight=(k == 1), post_kick=np.array([0.25 * k, 0.0]),
        )
    for frames in (buf.materialize(), buf.snapshot().materialize()):
        assert [f["inflight"] for f in frames] == [False, True, False]
        assert [f["pk"][0] for f in frames] == [0.0, 0.25, 0.5]


def test_env_frames_carry_flight_state_and_post_kick_timers():
    env = _recording_env(steps=40)
    frames = env.replay_frames
    assert all(isinstance(f["inflight"], bool) and len(f["pk"]) == 12 for f in frames)
    assert all(0.0 <= t <= 0.5 for f in frames for t in f["pk"])
    last = frames[-1]
    assert last["inflight"] == env.ball_kicked_in_flight
    assert last["pk"] == [round(float(t), 3) for t in env.post_kick_pbrs_timer]


# ==============================================================================
# Policy mean actions
# ==============================================================================
def _mean_action_table() -> np.ndarray:
    """(12, 4) table whose entry [p, j] is (4p + j) / 100, so every cell is individually recognisable."""
    return (np.arange(48, dtype=np.float64).reshape(12, 4)) / 100.0


def _env_with_means(steps: int, with_means_from: int = 0) -> FootballParallelEnv:
    cfg = replace(
        DEFAULT_CONFIG,
        replay=replace(DEFAULT_CONFIG.replay, enable_recording=True, interest_threshold=float("inf")),
    )
    env = FootballParallelEnv(config=cfg)
    env.reset(seed=11)
    acts = np.random.default_rng(11).uniform(-1, 1, size=(steps, 12, 4))
    for k in range(steps):
        env.step_array(acts[k], action_means=_mean_action_table() if k >= with_means_from else None)
    return env


def test_buffer_frame_carries_action_mean_from_the_step_payload():
    buf = ReplayBuffer(max_frames=4, num_players=2)
    buf.set_step_payload(1, {**PAYLOAD, "action_mean": [[0.1, 0.2, 0.3, 0.4], [-0.1, -0.2, -0.3, -0.4]]})
    _record(buf)
    assert buf.materialize()[0]["action_mean"] == [[0.1, 0.2, 0.3, 0.4], [-0.1, -0.2, -0.3, -0.4]]


def test_buffer_frame_without_action_mean_has_no_such_key():
    buf = _buffer()
    _record(buf)
    assert "action_mean" not in buf.materialize()[0]


def test_env_replay_frames_carry_a_12_by_4_action_mean_next_to_the_sampled_actions():
    env = _env_with_means(steps=2)
    skip = DEFAULT_CONFIG.physics.frame_skip
    frames = env.replay_frames
    assert len(frames) == 1 + 2 * skip
    for f in frames[1:]:
        mean = f["action_mean"]
        assert len(mean) == 12 and all(len(row) == 4 for row in mean)
        assert mean[0] == [0.0, 0.01, 0.02, 0.03]
        assert mean[11] == [0.44, 0.45, 0.46, 0.47]
        assert f["actions"] != mean


def test_env_steps_without_means_record_no_action_mean_and_do_not_inherit_stale_ones():
    env = _env_with_means(steps=3, with_means_from=2)
    skip = DEFAULT_CONFIG.physics.frame_skip
    frames = env.replay_frames
    assert all("action_mean" not in f for f in frames[: 1 + 2 * skip])
    assert all("action_mean" in f for f in frames[1 + 2 * skip :])
    env2 = _env_with_means(steps=2)
    env2.step_array(np.zeros((12, 4)))
    assert "action_mean" not in env2.replay_frames[-1]


def test_env_rejects_action_means_of_the_wrong_shape():
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=True))
    env = FootballParallelEnv(config=cfg)
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step_array(np.zeros((12, 4)), action_means=np.zeros((6, 4)))
