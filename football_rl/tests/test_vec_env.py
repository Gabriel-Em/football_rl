"""Unit and integration tests for vectorized football environments."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from typing import Optional, Tuple
import numpy as np
import pytest
import torch

from football_rl.config import ALL_AGENTS, CENTRAL_STATE_DIM, DEFAULT_CONFIG, ACTION_DIM, OBS_DIM
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


# ---------------------------------------------------------------------------
# Rollout bit-identity oracle
# ---------------------------------------------------------------------------
ROLLOUT_ORACLE_STEPS = 270
ROLLOUT_ORACLE_BLOCK = 30
ROLLOUT_ORACLE_BUFFERS = ("obs", "central_state", "actions", "logprobs", "values", "rewards", "dones")
ROLLOUT_DIGEST_FILE = Path(__file__).parent / "golden" / "rollout_digests.json"


def _oracle_config(variant: str, replay_dir: Path):
    """`full_match`: default episode length (no episode ends in 270 steps); `episodes`: 100-step matches (two resets per env)."""
    physics = DEFAULT_CONFIG.physics
    if variant == "episodes":
        physics = replace(physics, max_rl_steps=100)
    return replace(
        DEFAULT_CONFIG,
        physics=physics,
        replay=replace(DEFAULT_CONFIG.replay, save_dir=str(replay_dir), enable_recording=True),
    )


def collect_oracle_rollout(variant: str, workdir: Path) -> dict[str, np.ndarray]:
    """Seeded 8-env, 270-step `collect_rollouts` on CPU; returns the seven rollout buffers (dones as `(T, N, 12)`)."""
    trainer = ParameterSharingPPOTrainer(
        config=_oracle_config(variant, workdir / "replays"),
        num_envs=8,
        rollout_steps=ROLLOUT_ORACLE_STEPS,
        device="cpu",
        checkpoint_dir=str(workdir / "ckpt"),
        seed=7,
    )
    dones: list[np.ndarray] = []
    inner = trainer.vec_env.step_array

    def recording_step_array(*args, **kwargs):
        result = inner(*args, **kwargs)
        dones.append(np.array(result[3], dtype=np.float32))
        return result

    trainer.vec_env.step_array = recording_step_array
    try:
        rollout = trainer.collect_rollouts(active_generation=1)
    finally:
        trainer.close()
    buffers = {name: np.array(rollout[name]) for name in ROLLOUT_ORACLE_BUFFERS if name != "dones"}
    buffers["dones"] = np.repeat(np.stack(dones)[:, :, None], 12, axis=2)
    return buffers


def rollout_digests(buffers: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    """SHA-256 of every buffer plus one per 30-step block, so a divergence can be localised to a block."""
    out: dict[str, dict[str, object]] = {}
    for name in ROLLOUT_ORACLE_BUFFERS:
        arr = np.ascontiguousarray(buffers[name])
        out[name] = {
            "shape": list(arr.shape),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
            "blocks": [
                hashlib.sha256(arr[i : i + ROLLOUT_ORACLE_BLOCK].tobytes()).hexdigest()
                for i in range(0, arr.shape[0], ROLLOUT_ORACLE_BLOCK)
            ],
        }
    return out


def test_rollout_matches_stored_digests_of_the_pre_batching_pipeline(tmp_path: Path) -> None:
    """Seeded 8-env rollouts reproduce, bit for bit, the buffers recorded from the per-pipe, per-group pipeline."""
    golden = json.loads(ROLLOUT_DIGEST_FILE.read_text(encoding="utf-8"))
    if golden["torch"] != torch.__version__ or golden["torch_threads"] != torch.get_num_threads():
        pytest.skip(
            f"digests recorded with torch {golden['torch']} x{golden['torch_threads']} threads; "
            f"running torch {torch.__version__} x{torch.get_num_threads()}"
        )
    assert golden["steps"] == ROLLOUT_ORACLE_STEPS
    for variant in ("full_match", "episodes"):
        buffers = collect_oracle_rollout(variant, tmp_path / variant)
        actual = rollout_digests(buffers)
        for name in ROLLOUT_ORACLE_BUFFERS:
            want = golden["variants"][variant][name]
            assert list(buffers[name].shape) == want["shape"], f"{variant}/{name}: shape"
            if actual[name]["sha256"] != want["sha256"]:
                first = next(i for i, (a, b) in enumerate(zip(actual[name]["blocks"], want["blocks"])) if a != b)
                pytest.fail(
                    f"{variant}/{name} diverges from the stored rollout in steps "
                    f"{first * ROLLOUT_ORACLE_BLOCK}..{(first + 1) * ROLLOUT_ORACLE_BLOCK - 1}"
                )


def test_episode_variant_of_the_oracle_exercises_done_and_reset(tmp_path: Path) -> None:
    """The `episodes` oracle must contain episode ends (100-step matches, 270 steps -> 2 ends per env)."""
    golden = json.loads(ROLLOUT_DIGEST_FILE.read_text(encoding="utf-8"))
    assert golden["variants"]["episodes"]["dones"]["sha256"] != golden["variants"]["full_match"]["dones"]["sha256"]
    buffers = collect_oracle_rollout("episodes", tmp_path)
    assert buffers["dones"][:, :, 0].sum(axis=0).tolist() == [2.0] * 8
    assert buffers["dones"][99, :, 0].tolist() == [1.0] * 8 and buffers["dones"][98, :, 0].sum() == 0.0
    assert buffers["dones"][199, :, 0].tolist() == [1.0] * 8


@torch.no_grad()
def _reference_select_actions(
    trainer: ParameterSharingPPOTrainer,
    obs_batch: np.ndarray,
    central_batch: Optional[np.ndarray] = None,
    deterministic: bool = False,
    with_means: bool = False,
) -> Tuple[np.ndarray, ...]:
    """Per-group action selection through the four role policies, one forward pass and one RNG draw per group.

    Reference oracle for the batched `ParameterSharingPPOTrainer._select_actions`.
    """
    obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=trainer.inference_device)
    cen_t = (
        torch.as_tensor(central_batch, dtype=torch.float32, device=trainer.inference_device)
        if central_batch is not None
        else None
    )
    actions = torch.zeros((trainer.num_envs, trainer.num_agents, ACTION_DIM), dtype=torch.float32, device=trainer.inference_device)
    logprobs = torch.zeros((trainer.num_envs, trainer.num_agents), dtype=torch.float32, device=trainer.inference_device)
    values = torch.zeros((trainer.num_envs, trainer.num_agents), dtype=torch.float32, device=trainer.inference_device)
    means = (
        torch.zeros((trainer.num_envs, trainer.num_agents, ACTION_DIM), dtype=torch.float32, device=trainer.inference_device)
        if with_means
        else None
    )

    if not trainer.separate_teams:
        k_obs = obs_t[:, trainer.keeper_indices, :].reshape(-1, OBS_DIM)
        k_cen = (
            cen_t[:, trainer.keeper_indices, :].reshape(-1, CENTRAL_STATE_DIM)
            if cen_t is not None
            else None
        )
        k_act, k_logp, _, k_val, *k_mean = trainer.cpu_blue_keeper_policy.get_action_and_value(
            k_obs, deterministic=deterministic, central_state=k_cen, return_mean=with_means
        )
        if with_means:
            means[:, trainer.keeper_indices, :] = k_mean[0].view(trainer.num_envs, len(trainer.keeper_indices), ACTION_DIM)
        actions[:, trainer.keeper_indices, :] = k_act.view(trainer.num_envs, len(trainer.keeper_indices), ACTION_DIM)
        logprobs[:, trainer.keeper_indices] = k_logp.view(trainer.num_envs, len(trainer.keeper_indices))
        values[:, trainer.keeper_indices] = k_val.view(trainer.num_envs, len(trainer.keeper_indices))

        f_obs = obs_t[:, trainer.field_indices, :].reshape(-1, OBS_DIM)
        f_cen = (
            cen_t[:, trainer.field_indices, :].reshape(-1, CENTRAL_STATE_DIM)
            if cen_t is not None
            else None
        )
        f_act, f_logp, _, f_val, *f_mean = trainer.cpu_blue_field_policy.get_action_and_value(
            f_obs, deterministic=deterministic, central_state=f_cen, return_mean=with_means
        )
        if with_means:
            means[:, trainer.field_indices, :] = f_mean[0].view(trainer.num_envs, len(trainer.field_indices), ACTION_DIM)
        actions[:, trainer.field_indices, :] = f_act.view(trainer.num_envs, len(trainer.field_indices), ACTION_DIM)
        logprobs[:, trainer.field_indices] = f_logp.view(trainer.num_envs, len(trainer.field_indices))
        values[:, trainer.field_indices] = f_val.view(trainer.num_envs, len(trainer.field_indices))
    else:
        groups = [
            (trainer.blue_keeper_indices, trainer.cpu_blue_keeper_policy),
            (trainer.blue_field_indices, trainer.cpu_blue_field_policy),
            (trainer.red_keeper_indices, trainer.cpu_red_keeper_policy),
            (trainer.red_field_indices, trainer.cpu_red_field_policy),
        ]
        for indices, policy in groups:
            cnt = len(indices)
            g_obs = obs_t[:, indices, :].reshape(-1, OBS_DIM)
            g_cen = (
                cen_t[:, indices, :].reshape(-1, CENTRAL_STATE_DIM)
                if cen_t is not None
                else None
            )
            g_act, g_logp, _, g_val, *g_mean = policy.get_action_and_value(
                g_obs, deterministic=deterministic, central_state=g_cen, return_mean=with_means
            )
            if with_means:
                means[:, indices, :] = g_mean[0].view(trainer.num_envs, cnt, ACTION_DIM)
            actions[:, indices, :] = g_act.view(trainer.num_envs, cnt, ACTION_DIM)
            logprobs[:, indices] = g_logp.view(trainer.num_envs, cnt)
            values[:, indices] = g_val.view(trainer.num_envs, cnt)

    result = (
        actions.cpu().numpy(),
        logprobs.cpu().numpy(),
        values.cpu().numpy(),
    )
    if with_means:
        return result + (means.cpu().numpy(),)
    return result

def _randomize_policies(trainer: ParameterSharingPPOTrainer, seed: int) -> None:
    """Replace the near-zero initial output layers with O(1) random weights so every output bit matters."""
    gen = torch.Generator().manual_seed(seed)
    policies = {
        id(p): p
        for p in (
            trainer.blue_field_policy, trainer.blue_keeper_policy, trainer.red_field_policy, trainer.red_keeper_policy,
        )
    }
    with torch.no_grad():
        for policy in policies.values():
            for param in policy.parameters():
                param.copy_(torch.randn(param.shape, generator=gen) * 0.3)
    trainer._sync_cpu_policies()


@pytest.mark.parametrize("separate_teams", [True, False])
@pytest.mark.parametrize("with_means", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_select_actions_equals_per_group_reference_and_consumes_the_same_rng(
    tmp_path: Path, separate_teams: bool, with_means: bool, deterministic: bool
) -> None:
    """Batched selection returns the reference actions, log-probs, values and means exactly, and leaves the RNG alike."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    trainer = ParameterSharingPPOTrainer(
        config=cfg, num_envs=2, rollout_steps=1, device="cpu",
        checkpoint_dir=str(tmp_path / "ckpt"), seed=3, separate_teams=separate_teams,
    )
    try:
        _randomize_policies(trainer, seed=11)
        rng = np.random.default_rng(5)
        obs = rng.uniform(-1.0, 1.0, (2, 12, OBS_DIM)).astype(np.float32)
        cen = rng.uniform(-1.0, 1.0, (2, 12, CENTRAL_STATE_DIM)).astype(np.float32)
        for call in range(3):
            torch.manual_seed(100 + call)
            got = trainer._select_actions(obs, cen, deterministic=deterministic, with_means=with_means)
            rng_after_got = torch.get_rng_state()
            torch.manual_seed(100 + call)
            want = _reference_select_actions(trainer, obs, cen, deterministic=deterministic, with_means=with_means)
            rng_after_want = torch.get_rng_state()
            assert len(got) == len(want) == (4 if with_means else 3)
            for g, w in zip(got, want):
                assert g.dtype == w.dtype and g.shape == w.shape
                assert g.tobytes() == w.tobytes()
            assert torch.equal(rng_after_got, rng_after_want)
            obs = rng.uniform(-1.0, 1.0, (2, 12, OBS_DIM)).astype(np.float32)
    finally:
        trainer.close()


@pytest.mark.parametrize("threads", [1, 2, 3])
def test_select_actions_bits_do_not_depend_on_the_intra_op_thread_count(tmp_path: Path, threads: int) -> None:
    """Rollouts run with fewer CPU threads than training; the sampled actions must not change with the count."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    trainer = ParameterSharingPPOTrainer(
        config=cfg, num_envs=8, rollout_steps=1, device="cpu", checkpoint_dir=str(tmp_path / "ckpt"), seed=3,
    )
    previous = torch.get_num_threads()
    try:
        _randomize_policies(trainer, seed=13)
        rng = np.random.default_rng(8)
        obs = rng.uniform(-1.0, 1.0, (8, 12, OBS_DIM)).astype(np.float32)
        cen = rng.uniform(-1.0, 1.0, (8, 12, CENTRAL_STATE_DIM)).astype(np.float32)
        torch.set_num_threads(previous)
        torch.manual_seed(21)
        want = _reference_select_actions(trainer, obs, cen, with_means=True)
        torch.set_num_threads(threads)
        torch.manual_seed(21)
        got = trainer._select_actions(obs, cen, with_means=True)
        for g, w in zip(got, want):
            assert g.tobytes() == w.tobytes()
    finally:
        torch.set_num_threads(previous)
        trainer.close()


@pytest.mark.parametrize("cpus,expected", [(6, 3), (2, 1), (64, 6)])
def test_collect_rollouts_reserves_cores_for_workers_and_restores_the_thread_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cpus: int, expected: int
) -> None:
    """Two workers on `cpus` CPUs leave `cpus - 2 - 1` threads for inference, within [1, current]; reset afterwards."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    trainer = ParameterSharingPPOTrainer(
        config=cfg, num_envs=2, rollout_steps=2, device="cpu", checkpoint_dir=str(tmp_path / "ckpt"), seed=3,
    )
    previous = torch.get_num_threads()
    torch.set_num_threads(6)
    monkeypatch.setattr(os, "cpu_count", lambda: cpus)
    seen: list[int] = []
    inner = trainer._select_actions

    def spy(*args, **kwargs):
        seen.append(torch.get_num_threads())
        return inner(*args, **kwargs)

    trainer._select_actions = spy
    try:
        trainer.collect_rollouts(active_generation=1)
        assert seen == [expected, expected]
        assert torch.get_num_threads() == 6
    finally:
        torch.set_num_threads(previous)
        trainer.close()


def test_compute_values_equals_per_group_reference(tmp_path: Path) -> None:
    """Bootstrap values through the stacked path equal a plain per-policy `get_value`."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    trainer = ParameterSharingPPOTrainer(
        config=cfg, num_envs=2, rollout_steps=1, device="cpu", checkpoint_dir=str(tmp_path / "ckpt"), seed=3,
    )
    try:
        _randomize_policies(trainer, seed=12)
        rng = np.random.default_rng(6)
        obs = rng.uniform(-1.0, 1.0, (2, 12, OBS_DIM)).astype(np.float32)
        cen = rng.uniform(-1.0, 1.0, (2, 12, CENTRAL_STATE_DIM)).astype(np.float32)
        got = trainer._compute_values(obs, cen)
        want = np.zeros((2, 12), dtype=np.float32)
        groups = [
            ([0], trainer.cpu_blue_keeper_policy), ([1, 2, 3, 4, 5], trainer.cpu_blue_field_policy),
            ([6], trainer.cpu_red_keeper_policy), ([7, 8, 9, 10, 11], trainer.cpu_red_field_policy),
        ]
        with torch.no_grad():
            for idx, policy in groups:
                o = torch.as_tensor(obs[:, idx]).reshape(-1, OBS_DIM)
                c = torch.as_tensor(cen[:, idx]).reshape(-1, CENTRAL_STATE_DIM)
                want[:, idx] = policy.get_value(o, central_state=c).view(2, len(idx)).numpy()
        assert got.tobytes() == want.tobytes()
    finally:
        trainer.close()


# ---------------------------------------------------------------------------
# Shared-memory transport: double buffers, crashes, cleanup
# ---------------------------------------------------------------------------
def _shm_segments() -> set[str]:
    return set(os.listdir("/dev/shm")) if os.path.isdir("/dev/shm") else set()


def _no_recording_config(max_rl_steps: Optional[int] = None):
    physics = DEFAULT_CONFIG.physics if max_rl_steps is None else replace(DEFAULT_CONFIG.physics, max_rl_steps=max_rl_steps)
    return replace(DEFAULT_CONFIG, physics=physics, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))


def _reset_vec(vec: SubprocVecFootballEnv) -> None:
    vec.reset_array(
        seeds=list(range(vec.num_envs)),
        options_list=[{"generation": 1, "match_id": i + 1} for i in range(vec.num_envs)],
    )


def test_step_results_alternate_between_two_shared_buffers() -> None:
    """Results of step t stay intact while step t+1 is produced; step t+2 reuses the memory of step t."""
    vec = make_vec_football_env(_no_recording_config(), num_envs=2)
    try:
        _reset_vec(vec)
        rng = np.random.default_rng(0)
        results = []
        for _ in range(3):
            act = rng.uniform(-1.0, 1.0, (2, 12, ACTION_DIM)).astype(np.float32)
            obs, cen, rew, done, _ = vec.step_array(act)
            results.append((obs, cen, rew, done, obs.copy(), cen.copy(), rew.copy()))
        (obs0, cen0, rew0, _, obs0_copy, cen0_copy, rew0_copy) = results[0]
        (obs1, _, _, _, obs1_copy, _, _) = results[1]
        (obs2, _, _, _, obs2_copy, _, _) = results[2]
        assert not np.shares_memory(obs0, obs1)
        assert np.shares_memory(obs0, obs2)
        assert np.array_equal(obs1, obs1_copy)
        assert np.array_equal(obs2, obs2_copy)
        assert not np.array_equal(obs0_copy, obs1_copy)
        assert obs0.shape == (2, 12, OBS_DIM) and cen0.shape == (2, 12, CENTRAL_STATE_DIM) and rew0.shape == (2, 12)
        assert obs0.dtype == np.float32 and rew0.dtype == np.float32
        assert cen0_copy.shape == cen0.shape and rew0_copy.shape == rew0.shape
    finally:
        vec.close()


def test_shared_memory_exists_while_open_and_is_released_on_close_and_double_close() -> None:
    before = _shm_segments()
    vec = make_vec_football_env(_no_recording_config(), num_envs=2)
    during = _shm_segments()
    assert len(during - before) >= 1
    _reset_vec(vec)
    vec.step_array(np.zeros((2, 12, ACTION_DIM), dtype=np.float32))
    vec.close()
    vec.close()
    assert _shm_segments() == before


def test_shared_memory_is_released_after_a_worker_crash() -> None:
    before = _shm_segments()
    vec = make_vec_football_env(_no_recording_config(), num_envs=2)
    try:
        _reset_vec(vec)
        os.kill(vec.processes[0].pid, signal.SIGKILL)
        with pytest.raises(RuntimeError):
            vec.step_array(np.zeros((2, 12, ACTION_DIM), dtype=np.float32))
    finally:
        vec.close()
    assert _shm_segments() == before


@pytest.mark.parametrize("victim", [0, 1])
def test_killed_worker_raises_runtime_error_naming_it_within_five_seconds(victim: int) -> None:
    vec = make_vec_football_env(_no_recording_config(), num_envs=2)
    try:
        _reset_vec(vec)
        vec.step_array(np.zeros((2, 12, ACTION_DIM), dtype=np.float32))
        os.kill(vec.processes[victim].pid, signal.SIGKILL)
        start = time.monotonic()
        with pytest.raises(RuntimeError, match=rf"worker {victim}\b"):
            vec.step_array(np.zeros((2, 12, ACTION_DIM), dtype=np.float32))
        assert time.monotonic() - start < 5.0
    finally:
        vec.close()


def test_worker_exception_is_reported_with_its_traceback_within_five_seconds() -> None:
    """A worker that raises (here: an unknown command) reports its traceback; the next step names that worker."""
    vec = make_vec_football_env(_no_recording_config(), num_envs=2)
    try:
        _reset_vec(vec)
        vec.remotes[1].send(("no-such-command", None))
        start = time.monotonic()
        with pytest.raises(RuntimeError, match=r"worker 1\b[\s\S]*NotImplementedError"):
            vec.step_array(np.zeros((2, 12, ACTION_DIM), dtype=np.float32))
        assert time.monotonic() - start < 5.0
    finally:
        vec.close()


def test_stalled_worker_times_out_instead_of_hanging() -> None:
    vec = SubprocVecFootballEnv(
        [__import__("football_rl.vec_env", fromlist=["_EnvWorkerThunk"])._EnvWorkerThunk(_no_recording_config())] * 2,
        context="spawn",
        step_timeout=1.0,
    )
    try:
        _reset_vec(vec)
        os.kill(vec.processes[1].pid, signal.SIGSTOP)
        start = time.monotonic()
        with pytest.raises(RuntimeError, match=r"worker 1 timed out"):
            vec.step_array(np.zeros((2, 12, ACTION_DIM), dtype=np.float32))
        assert time.monotonic() - start < 5.0
    finally:
        os.kill(vec.processes[1].pid, signal.SIGCONT)
        vec.close()


def test_vec_env_state_after_a_worker_failure_is_closed_cleanly() -> None:
    """After a failed step, `close()` returns promptly even though a worker is gone."""
    vec = make_vec_football_env(_no_recording_config(), num_envs=2)
    _reset_vec(vec)
    os.kill(vec.processes[0].pid, signal.SIGKILL)
    with pytest.raises(RuntimeError):
        vec.step_array(np.zeros((2, 12, ACTION_DIM), dtype=np.float32))
    start = time.monotonic()
    vec.close()
    assert time.monotonic() - start < 5.0
    assert vec.closed
