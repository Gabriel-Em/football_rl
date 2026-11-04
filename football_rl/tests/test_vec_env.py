"""Unit and integration tests for vectorized football environments."""

from __future__ import annotations

from dataclasses import replace
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
