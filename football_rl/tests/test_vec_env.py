"""Unit and integration tests for vectorized football environments."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch

from football_rl.config import ALL_AGENTS, DEFAULT_CONFIG, ACTION_DIM, OBS_DIM
from football_rl.env import FootballParallelEnv, _fast_clip
from football_rl.train_ppo import ParameterSharingPPOTrainer
from football_rl.vec_env import (
    DummyVecFootballEnv,
    SubprocVecFootballEnv,
    make_vec_football_env,
)


def _zero_actions_for_all() -> dict[str, np.ndarray]:
    return {
        agent: np.array([0.0, 0.0, -1.0, -1.0], dtype=np.float32)
        for agent in ALL_AGENTS
    }


def test_fast_clip_behavior() -> None:
    """Verify _fast_clip behaves identically to np.clip for float bounds."""
    test_cases = [
        (0.5, 0.0, 1.0),
        (-0.5, 0.0, 1.0),
        (1.5, 0.0, 1.0),
        (0.0, -1.0, 1.0),
        (-1.2, -1.0, 1.0),
        (2.3, -1.0, 1.0),
    ]
    for val, low, high in test_cases:
        expected = float(np.clip(val, low, high))
        actual = _fast_clip(val, low, high)
        assert actual == pytest.approx(expected)


def test_dummy_vec_env_lifecycle() -> None:
    """Verify DummyVecFootballEnv lifecycle: make, reset, step, set_gen, and close."""
    vec = make_vec_football_env(DEFAULT_CONFIG, num_envs=1)
    assert isinstance(vec, DummyVecFootballEnv)
    assert vec.num_envs == 1

    obs_list, info_list = vec.reset(
        seeds=[42],
        options_list=[{"generation": 5, "match_id": 10}],
    )
    assert len(obs_list) == 1
    assert len(info_list) == 1
    assert len(obs_list[0]) == len(ALL_AGENTS)
    for agent in ALL_AGENTS:
        assert obs_list[0][agent].shape == (OBS_DIM,)

    actions = [_zero_actions_for_all()]
    obs_l, rew_l, term_l, trunc_l, info_l, done_l = vec.step(actions)
    assert len(obs_l) == 1
    assert len(rew_l) == 1
    assert len(done_l) == 1
    assert done_l[0] is False

    vec.set_generation(7)
    assert vec.envs[0].current_generation == 7

    vec.close()
    assert vec.closed is True


def test_subproc_vec_env_lifecycle() -> None:
    """Verify SubprocVecFootballEnv handles multiprocessing reset, step, partial reset, and close."""
    cfg = replace(
        DEFAULT_CONFIG,
        replay=replace(DEFAULT_CONFIG.replay, enable_recording=False),
    )
    vec = make_vec_football_env(cfg, num_envs=2)
    assert isinstance(vec, SubprocVecFootballEnv)
    assert vec.num_envs == 2

    # 1. Reset all environments with explicit options
    obs_list, info_list = vec.reset(
        seeds=[101, 102],
        options_list=[
            {"generation": 1, "match_id": 1},
            {"generation": 1, "match_id": 2},
        ],
    )
    assert len(obs_list) == 2
    for env_i in range(2):
        for agent in ALL_AGENTS:
            assert obs_list[env_i][agent].shape == (OBS_DIM,)

    # 2. Step both environments in parallel
    actions = [_zero_actions_for_all(), _zero_actions_for_all()]
    obs_l, rew_l, term_l, trunc_l, info_l, done_l = vec.step(actions)
    assert len(obs_l) == 2
    assert len(rew_l) == 2
    assert len(done_l) == 2
    assert all(not d for d in done_l)

    # 3. Partial reset of environment 1 only
    partial_obs, partial_info = vec.reset(
        seeds=[202],
        options_list=[{"generation": 2, "match_id": 3}],
        env_indices=[1],
    )
    assert len(partial_obs) == 1
    assert len(partial_obs[0]) == len(ALL_AGENTS)

    # 4. Set generation across workers
    vec.set_generation(3)

    # 5. Clean teardown
    vec.close()
    assert vec.closed is True


def test_vectorized_observation_integrity() -> None:
    """Verify that observations from FootballParallelEnv maintain shape and valid normalized ranges."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    obs, _ = env.reset(seed=42)
    assert len(obs) == len(ALL_AGENTS)

    for agent, agent_obs in obs.items():
        assert agent_obs.shape == (OBS_DIM,)
        assert not np.isnan(agent_obs).any()
        assert not np.isinf(agent_obs).any()
        # Positional and angle features are normalized
        assert np.all(agent_obs >= -1.5)
        assert np.all(agent_obs <= 1.5)

    # Step once and check observations again
    obs, _, _, _, _ = env.step(_zero_actions_for_all())
    for agent, agent_obs in obs.items():
        assert agent_obs.shape == (OBS_DIM,)
        assert not np.isnan(agent_obs).any()
        assert not np.isinf(agent_obs).any()
    env.close()


def test_subproc_trainer_rollout_integration(tmp_path: Path) -> None:
    """Verify ParameterSharingPPOTrainer correctly collects rollouts and updates with SubprocVec."""
    cfg = replace(
        DEFAULT_CONFIG,
        replay=replace(DEFAULT_CONFIG.replay, enable_recording=False),
    )
    trainer = ParameterSharingPPOTrainer(
        config=cfg,
        num_envs=2,
        rollout_steps=32,
        device="cpu",
        checkpoint_dir=str(tmp_path / "ckpt"),
        seed=42,
    )
    try:
        assert isinstance(trainer.vec_env, SubprocVecFootballEnv)
        metrics = trainer.train(total_updates=1)
        assert metrics["generation"] == 1
        assert metrics["total_matches"] >= 0
    finally:
        trainer.close()


def test_trainer_monotonic_match_id_and_every_gen_checkpoint(tmp_path: Path) -> None:
    """Verify trainer allocates monotonic match IDs across resumes and persists checkpoints every generation."""
    ckpt_dir = tmp_path / "ckpt"
    rep_dir = tmp_path / "replays"
    os.makedirs(rep_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Pre-create a dummy replay file: match_0380_gen0048_...
    dummy_replay = rep_dir / "match_0380_gen0048_score1_0_20261105_000000.json"
    dummy_replay.write_text('{"metadata": {"match_id": 380, "generation": 48}}', encoding="utf-8")

    cfg = replace(
        DEFAULT_CONFIG,
        replay=replace(DEFAULT_CONFIG.replay, save_dir=str(rep_dir), enable_recording=False),
    )
    trainer = ParameterSharingPPOTrainer(
        config=cfg,
        num_envs=1,
        rollout_steps=16,
        device="cpu",
        checkpoint_dir=str(ckpt_dir),
        seed=42,
    )
    try:
        # Match ID must start strictly higher than 380
        assert trainer.next_match_id >= 381
        m1 = trainer._allocate_match_id()
        assert m1 >= 381
        assert trainer.next_match_id == m1 + 1

        # Run 1 generation and verify checkpoint was saved immediately
        trainer.train(total_updates=1)
        state_file = ckpt_dir / "training_state.pt"
        assert state_file.exists()
        state = torch.load(state_file, map_location="cpu", weights_only=False)
        assert state["current_generation"] == 1
    finally:
        trainer.close()


def test_step_array_and_single_worker_replay_export(tmp_path: Path) -> None:
    """Verify fast step_array IPC and that save_best_only writes only the winning worker replay to disk."""
    rep_dir = tmp_path / "replays"
    ckpt_dir = tmp_path / "ckpt"
    cfg = replace(
        DEFAULT_CONFIG,
        physics=replace(DEFAULT_CONFIG.physics, max_rl_steps=4),
        replay=replace(
            DEFAULT_CONFIG.replay,
            save_dir=str(rep_dir),
            enable_recording=True,
            save_best_only=True,
            interest_threshold=0.0,
        ),
    )

    vec = make_vec_football_env(cfg, num_envs=2)
    try:
        obs_b, cen_b = vec.reset_array(
            seeds=[11, 22],
            options_list=[
                {"generation": 1, "match_id": 1},
                {"generation": 1, "match_id": 2},
            ],
        )
        assert obs_b.shape == (2, 12, OBS_DIM)
        assert cen_b.shape == (2, 12, 54)

        act_arr = np.zeros((2, 12, ACTION_DIM), dtype=np.float32)
        obs_next, cen_next, rew_next, done_next, summaries = vec.step_array(act_arr)
        assert obs_next.shape == (2, 12, OBS_DIM)
        assert cen_next.shape == (2, 12, 54)
        assert rew_next.shape == (2, 12)
        assert done_next.shape == (2,)
        assert not np.any(done_next)
        assert summaries == [None, None]
    finally:
        vec.close()

    trainer = ParameterSharingPPOTrainer(
        config=cfg,
        num_envs=2,
        rollout_steps=4,
        device="cpu",
        checkpoint_dir=str(ckpt_dir),
        seed=42,
        save_best_only=True,
    )
    try:
        rollout = trainer.collect_rollouts(active_generation=1)
        assert rollout["completed_episodes"] == 2
        assert len(rollout["exported_replays"]) == 1
        saved_files = [f for f in os.listdir(rep_dir) if f.endswith(".json")]
        assert len(saved_files) == 1
    finally:
        trainer.close()




@pytest.mark.parametrize("num_envs", [1, 2])
def test_action_means_and_uniform_capture_reach_only_the_selected_worker(tmp_path: Path, num_envs: int) -> None:
    """Means travel through `step_array`; only the worker selected for capture keeps and exports its match."""
    cfg = replace(
        DEFAULT_CONFIG,
        physics=replace(DEFAULT_CONFIG.physics, max_rl_steps=4),
        replay=replace(
            DEFAULT_CONFIG.replay,
            save_dir=str(tmp_path / "rep"),
            enable_recording=True,
            interest_threshold=float("inf"),
        ),
    )
    chosen = num_envs - 1
    vec = make_vec_football_env(cfg, num_envs=num_envs)
    try:
        vec.reset_array(
            seeds=list(range(num_envs)),
            options_list=[{"generation": 3, "match_id": 10 + i} for i in range(num_envs)],
        )
        vec.set_generation(3)
        vec.set_uniform_capture(chosen)
        means = np.full((num_envs, 12, ACTION_DIM), 0.25, dtype=np.float32)
        done = np.zeros(num_envs, dtype=bool)
        for _ in range(4):
            _, _, _, done, _ = vec.step_array(np.zeros((num_envs, 12, ACTION_DIM), dtype=np.float32), action_means=means)
        assert done.all()
        for idx in range(num_envs):
            path = vec.export_uniform_replay(idx)
            if idx != chosen:
                assert path is None
                continue
            assert path is not None and "_uniform_" in Path(path).name and f"match_{10 + idx:04d}_gen0003" in Path(path).name
            frames = json.loads(Path(path).read_text(encoding="utf-8"))["frames"]
            assert len(frames) == 4 * DEFAULT_CONFIG.physics.frame_skip + 1
            assert frames[-1]["action_mean"] == [[0.25] * ACTION_DIM] * 12
        assert sorted(p.name for p in (tmp_path / "rep").glob("match_*.json")) == [Path(path).name]
        vec.set_uniform_capture(None)
        vec.set_generation(4)
        assert vec.export_uniform_replay(chosen) is None
    finally:
        vec.close()
