"""Unit and integration tests for Centralized Training with Decentralized Execution (CTDE) PPO."""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import replace
import json
import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from football_rl.config import (
    ACTION_DIM,
    ALL_AGENTS,
    CENTRAL_STATE_DIM,
    CRITIC_STATE_DIM,
    DEFAULT_CONFIG,
    OBS_DIM,
)
from football_rl.env import FootballParallelEnv
from football_rl.train_ppo import (
    ActorCritic,
    ParameterSharingPPOTrainer,
    archive_batch,
    migrate_team_color_keys,
    print_active_status,
    select_archive_replays,
    uniform_env_index,
)


def test_ctde_ppo_rollout_and_update() -> None:
    """Verify CTDE PPO rollout and update with centralized critic state (136-dim) and decentralized actor (82-dim)."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    trainer = ParameterSharingPPOTrainer(
        config=cfg,
        num_envs=1,
        rollout_steps=10,
        device="cpu",
    )

    # Verify network architectures: actor (82), centralized critic (136 = 82 + 54)
    assert (OBS_DIM, CRITIC_STATE_DIM) == (82, 136)
    assert trainer.blue_field_policy.critic[0].in_features == CRITIC_STATE_DIM
    assert trainer.blue_field_policy.actor_mean[0].in_features == OBS_DIM
    assert CRITIC_STATE_DIM == OBS_DIM + CENTRAL_STATE_DIM

    # Test single-agent forward with central_state
    dummy_obs = torch.zeros((4, OBS_DIM), dtype=torch.float32)
    dummy_cen = torch.zeros((4, CENTRAL_STATE_DIM), dtype=torch.float32)
    val = trainer.blue_field_policy.get_value(dummy_obs, central_state=dummy_cen)
    assert val.shape == (4,)

    # Verify backward-compatibility when central_state is None (zero-padded)
    val_pad = trainer.blue_field_policy.get_value(dummy_obs)
    assert val_pad.shape == (4,)

    # Collect mini rollout
    rollout = trainer.collect_rollouts(active_generation=1)
    assert "central_state" in rollout
    assert rollout["central_state"].shape == (10, 1, 12, CENTRAL_STATE_DIM)
    assert rollout["obs"].shape == (10, 1, 12, OBS_DIM)

    # Perform PPO policy update
    metrics = trainer.update_policies(rollout)
    assert "field" in metrics
    assert "keeper" in metrics
    assert "v_loss" in metrics["field"]
    assert "pg_loss" in metrics["field"]


def test_adapted_weights_loader_zero_pads_legacy_critic() -> None:
    """Verify _load_adapted_weights zero-pads the central-state columns of an observation-only critic."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    trainer = ParameterSharingPPOTrainer(config=cfg, num_envs=1, rollout_steps=10, device="cpu")

    # Create dummy legacy state dict with critic.0.weight shape (hidden, OBS_DIM)
    target_policy = trainer.blue_field_policy
    legacy_weights = {}
    for k, v in target_policy.state_dict().items():
        if k == "critic.0.weight":
            # observation-only legacy input features
            legacy_weights[k] = torch.ones((v.shape[0], OBS_DIM), dtype=torch.float32)
        else:
            legacy_weights[k] = v.clone()

    # Load adapted weights into the OBS_DIM + CENTRAL_STATE_DIM critic
    trainer._load_adapted_weights(target_policy, legacy_weights)

    loaded_critic_w = target_policy.critic[0].weight.data
    # First OBS_DIM cols should be ones (from legacy checkpoint)
    assert torch.allclose(loaded_critic_w[:, :OBS_DIM], torch.ones((loaded_critic_w.shape[0], OBS_DIM)))
    # Remaining CENTRAL_STATE_DIM cols should be zero-padded
    assert torch.allclose(loaded_critic_w[:, OBS_DIM:], torch.zeros((loaded_critic_w.shape[0], CENTRAL_STATE_DIM)))


LEGACY_OBS_DIM = 76  # observation width of checkpoints written before the match-state columns 76..81


def _recorded_observations(n_rows: int = 1000, seed: int = 17) -> tuple[torch.Tensor, torch.Tensor]:
    """First `n_rows` (observation, central state) rows of a seeded random-action rollout, 12 rows per step."""
    env = FootballParallelEnv(config=_quiet_cfg())
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    obs, cen = env._get_observations_array(), env.get_centralized_state()
    obs_rows, cen_rows = [], []
    for step_i in range(-(-n_rows // 12)):
        if step_i == 10:
            # Trigger a real loose-ball turnover so column 81 (turnover cooldown) also carries signal.
            env.ball_holder = None
            env.ball_kicked_in_flight = False
            env.ball_vel[:] = 0.0
            env.last_ball_team = 0
            env.ball_pos[:] = env.player_pos[11]
            env.pickup_cooldown[11] = 0.0
        obs_rows.append(obs.copy())
        cen_rows.append(cen.copy())
        obs, cen, _, _, _ = env.step_array(rng.uniform(-1.0, 1.0, size=(12, 4)))
    return (
        torch.as_tensor(np.concatenate(obs_rows)[:n_rows]),
        torch.as_tensor(np.concatenate(cen_rows)[:n_rows]),
    )


def _perturbed_policy(obs_dim: int, seed: int) -> ActorCritic:
    """An ActorCritic of input width `obs_dim` with every parameter moved off its initialisation."""
    torch.manual_seed(seed)
    net = ActorCritic(obs_dim=obs_dim)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return net


def _column_ids(width: int, rows: int = 256) -> torch.Tensor:
    """A (rows, width) weight whose column j holds j + 1, so every column's origin is readable after a load."""
    return (torch.arange(width, dtype=torch.float32) + 1.0).repeat(rows, 1)


@pytest.fixture(scope="module")
def loader_trainer(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("loader")
    trainer = ParameterSharingPPOTrainer(
        config=_quiet_cfg(save_dir=str(tmp / "rep")), num_envs=1, rollout_steps=10, device="cpu",
        checkpoint_dir=str(tmp / "ckpt"), seed=3,
    )
    yield trainer
    trainer.close()


@pytest.mark.parametrize("old_obs_dim", [76, 81])
def test_adapted_loader_keeps_a_76_input_checkpoint_bit_identical(loader_trainer, old_obs_dim) -> None:
    old = _perturbed_policy(old_obs_dim, seed=11)
    new = ActorCritic(obs_dim=82)
    loader_trainer._load_adapted_weights(new, old.state_dict())

    obs, cen = _recorded_observations()
    assert obs.shape == (1000, 82) and cen.shape == (1000, 54)
    assert (obs[:, old_obs_dim:] != 0.0).any(), "the appended columns must carry signal"
    with torch.no_grad():
        assert torch.equal(new.actor_mean(obs), old.actor_mean(obs[:, :old_obs_dim]))
        assert torch.equal(new.get_value(obs, cen), old.get_value(obs[:, :old_obs_dim], cen))
    assert torch.equal(new.actor_logstd, old.actor_logstd)


@pytest.mark.parametrize(
    "old_obs_dim, old_critic_dim, actor_ids, critic_ids",
    [
        # actor 76, critic 76 + 54: the six new columns sit between the observation and the central state
        (76, 130, list(range(1, 77)) + [0] * 6, list(range(1, 77)) + [0] * 6 + list(range(77, 131))),
        # actor 76, observation-only critic
        (76, 76, list(range(1, 77)) + [0] * 6, list(range(1, 77)) + [0] * 60),
        # actor 81, critic 81 + 54: the turnover-cooldown column sits between obs 81 and central state
        (81, 135, list(range(1, 82)) + [0], list(range(1, 82)) + [0] + list(range(82, 136))),
        (81, 81, list(range(1, 82)) + [0], list(range(1, 82)) + [0] * 55),
        # current widths load unchanged
        (82, 136, list(range(1, 83)), list(range(1, 137))),
        (82, 82, list(range(1, 83)), list(range(1, 83)) + [0] * 54),
    ],
)
def test_adapted_loader_places_every_old_input_column(
    loader_trainer, old_obs_dim, old_critic_dim, actor_ids, critic_ids
) -> None:
    target = ActorCritic(obs_dim=82)
    weights = {k: v.clone() for k, v in target.state_dict().items()}
    weights["actor_mean.0.weight"] = _column_ids(old_obs_dim)
    weights["critic.0.weight"] = _column_ids(old_critic_dim)
    loader_trainer._load_adapted_weights(target, weights)
    assert torch.equal(target.actor_mean[0].weight, torch.tensor(actor_ids, dtype=torch.float32).repeat(256, 1))
    assert torch.equal(target.critic[0].weight, torch.tensor(critic_ids, dtype=torch.float32).repeat(256, 1))


@pytest.mark.parametrize(
    "actor_width, critic_width",
    [
        (87, 141),  # wider than the network: columns cannot be dropped
        (76, 100),  # critic is neither observation-only nor observation + central state
        (76, 136),  # critic built for a different observation width than the actor
        (81, 136),  # critic built for 82-dim obs while actor is 81-dim
    ],
)
def test_adapted_loader_rejects_checkpoints_it_cannot_place(loader_trainer, actor_width, critic_width) -> None:
    target = ActorCritic(obs_dim=82)
    before = {k: v.clone() for k, v in target.state_dict().items()}
    weights = {k: v.clone() for k, v in before.items()}
    weights["actor_mean.0.weight"] = torch.ones(256, actor_width)
    weights["critic.0.weight"] = torch.ones(256, critic_width)
    with pytest.raises(ValueError):
        loader_trainer._load_adapted_weights(target, weights)
    assert all(torch.equal(before[k], v) for k, v in target.state_dict().items())


@pytest.mark.parametrize("old_obs_dim", [76, 81])
def test_resume_from_a_76_input_training_state_pads_adam_moments_and_trains(tmp_path: Path, old_obs_dim: int) -> None:
    ckpt = tmp_path / "ckpt"
    writer = ParameterSharingPPOTrainer(
        config=_quiet_cfg(save_dir=str(tmp_path / "rep")), num_envs=1, rollout_steps=10, device="cpu",
        checkpoint_dir=str(ckpt), seed=3,
    )
    writer.save_checkpoints()
    writer.close()

    state = torch.load(ckpt / "training_state.pt", map_location="cpu", weights_only=False)
    legacy = {}
    for i, name in enumerate(("blue_field", "blue_keeper", "red_field", "red_keeper")):
        net = _perturbed_policy(old_obs_dim, seed=20 + i)
        opt = torch.optim.Adam(net.parameters(), lr=3e-4, eps=1e-5)
        x, c = torch.randn(16, old_obs_dim), torch.randn(16, 54)
        (net.actor_mean(x).square().sum() + net.get_value(x, c).square().sum()).backward()
        opt.step()
        state[f"{name}_policy"], state[f"{name}_optimizer"] = net.state_dict(), opt.state_dict()
        legacy[name] = (net, opt)
    for kind in ("field", "keeper"):
        state[f"{kind}_policy"] = state[f"blue_{kind}_policy"]
        state[f"{kind}_optimizer"] = state[f"blue_{kind}_optimizer"]
    torch.save(state, ckpt / "training_state.pt")

    trainer = ParameterSharingPPOTrainer(
        config=_quiet_cfg(save_dir=str(tmp_path / "rep")), num_envs=1, rollout_steps=10, device="cpu",
        checkpoint_dir=str(ckpt), seed=4,
    )
    assert trainer.load_checkpoints()
    policy, optimizer = trainer.blue_field_policy, trainer.blue_field_optimizer
    old_net, old_opt = legacy["blue_field"]
    assert policy.actor_mean[0].in_features == 82 and policy.critic[0].in_features == 136

    new_actor = optimizer.state[policy.actor_mean[0].weight]
    old_actor = old_opt.state[old_net.actor_mean[0].weight]
    new_critic = optimizer.state[policy.critic[0].weight]
    old_critic = old_opt.state[old_net.critic[0].weight]
    for moment in ("exp_avg", "exp_avg_sq"):
        assert torch.equal(new_actor[moment][:, :old_obs_dim], old_actor[moment])
        assert not new_actor[moment][:, old_obs_dim:].any()
        assert torch.equal(new_critic[moment][:, :old_obs_dim], old_critic[moment][:, :old_obs_dim])
        assert not new_critic[moment][:, old_obs_dim:82].any()
        assert torch.equal(new_critic[moment][:, 82:], old_critic[moment][:, old_obs_dim:])
    assert float(new_actor["step"]) == 1.0

    metrics = trainer.update_policies(trainer.collect_rollouts(active_generation=1))
    assert "field" in metrics and "keeper" in metrics
    trainer.close()


def test_per_role_advantage_normalization() -> None:
    """Verify _normalize_advantages_per_role normalizes across the pooled team unit without intra-role zero-sum washing."""
    obs = torch.zeros((12, OBS_DIM), dtype=torch.float32)
    # 4 DEF [1,0,0], 4 MID [0,1,0], 4 ATK [0,0,1]
    obs[0:4, 13:16] = torch.tensor([1.0, 0.0, 0.0])
    obs[4:8, 13:16] = torch.tensor([0.0, 1.0, 0.0])
    obs[8:12, 13:16] = torch.tensor([0.0, 0.0, 1.0])

    # Give ATK 100x larger raw advantage scale than DEF
    adv = torch.tensor(
        [
            -0.02, -0.01, 0.01, 0.02,     # DEF
            -0.50, -0.20, 0.20, 0.50,     # MID
            -20.0, -10.0, 10.0, 20.0,     # ATK
        ],
        dtype=torch.float32,
    )

    norm_adv = ParameterSharingPPOTrainer._normalize_advantages_per_role(obs, adv)
    assert abs(float(norm_adv.mean().item())) < 1e-6
    assert abs(float(norm_adv.std().item()) - 1.0) < 1e-5
    # Verify relative scale is preserved (ATK advantages remain larger than DEF advantages)
    assert float(norm_adv[11].item()) > float(norm_adv[7].item()) > float(norm_adv[3].item())


def test_select_archive_replays() -> None:
    """Verify archive selection retains exactly 15 matches (first 5, middle 5, last 5)."""
    # 1. Total matches <= 15: keep all
    short_list = [f"match_{i:04d}_gen{i:04d}_score10_0.json" for i in range(1, 10)]
    assert select_archive_replays(short_list) == short_list

    # 2. Total matches > 15 (e.g. 50 generations)
    files_50 = [f"match_{i:04d}_gen{i:04d}_score10_0.json" for i in range(1, 51)]
    kept = select_archive_replays(files_50, seed=123)
    assert len(kept) == 15

    gens_kept = []
    for f in kept:
        m = re.search(r"gen(\d+)", f)
        assert m is not None
        gens_kept.append(int(m.group(1)))

    # First 5 generations
    assert gens_kept[:5] == [1, 2, 3, 4, 5]
    # Last 5 generations
    assert gens_kept[-5:] == [46, 47, 48, 49, 50]
    # Middle 5 generations spread out
    mid_gens = gens_kept[5:-5]
    assert len(mid_gens) == 5
    for g in mid_gens:
        assert 6 <= g <= 45
    # Strict monotonically increasing
    assert mid_gens == sorted(mid_gens)
    assert len(set(mid_gens)) == 5


def test_select_archive_replays_prefers_uniform_over_higher_scoring_best() -> None:
    """[RED] When both best and `_uniform_` exist in a generation, `_uniform_` wins regardless of score."""
    files = []
    for g in range(1, 21):
        files.append(f"match_{2 * g - 1:04d}_gen{g:04d}_score99_0_20261119_120000.json")
        files.append(f"match_{2 * g:04d}_gen{g:04d}_score1_0_uniform_20261119_120000.json")

    kept = select_archive_replays(files, seed=7)
    expected_gens = [1, 2, 3, 4, 5, 7, 8, 11, 12, 14, 16, 17, 18, 19, 20]
    expected = [f"match_{2 * g:04d}_gen{g:04d}_score1_0_uniform_20261119_120000.json" for g in expected_gens]
    assert kept == expected
    assert all("_uniform_" in f for f in kept)


def test_select_archive_replays_best_only_and_mixed_fallback_unchanged() -> None:
    """[RED] Best-only generations still select the highest-scoring best file, matching legacy order."""
    expected_gens = [1, 2, 3, 4, 5, 7, 8, 11, 12, 14, 16, 17, 18, 19, 20]
    best_only_files = []
    for g in range(1, 21):
        best_only_files.append(f"match_{2 * g - 1:04d}_gen{g:04d}_score5_0_20261119_120000.json")
        best_only_files.append(f"match_{2 * g:04d}_gen{g:04d}_score12_0_20261119_120000.json")
    expected_best_only = [f"match_{2 * g:04d}_gen{g:04d}_score12_0_20261119_120000.json" for g in expected_gens]
    assert select_archive_replays(best_only_files, seed=7) == expected_best_only

    # Mixed: gen 3 has only best, other chosen gens have both; generation list is unchanged.
    mixed_files = []
    for g in range(1, 21):
        mixed_files.append(f"match_{2 * g - 1:04d}_gen{g:04d}_score99_0_20261119_120000.json")
        if g != 3:
            mixed_files.append(f"match_{2 * g:04d}_gen{g:04d}_score1_0_uniform_20261119_120000.json")
    kept_mixed = select_archive_replays(mixed_files, seed=7)
    kept_gens = [int(re.search(r"gen(\d+)", f).group(1)) for f in kept_mixed]
    assert kept_gens == expected_gens
    assert kept_mixed[2] == "match_0005_gen0003_score99_0_20261119_120000.json"
    assert all("_uniform_" in f for idx, f in enumerate(kept_mixed) if idx != 2)


def test_select_archive_replays_orders_multiple_uniform_files_by_score_then_match_id() -> None:
    """[RED] Among multiple `_uniform_` files in one generation, higher `(score, match_id)` wins."""
    files = []
    for g in range(1, 21):
        if g == 1:
            files.append("match_0010_gen0001_score2_0_uniform_20261119_120000.json")
            files.append("match_0020_gen0001_score8_0_uniform_20261119_120000.json")
            files.append("match_0099_gen0001_score50_0_20261119_120000.json")
        elif g == 2:
            files.append("match_0011_gen0002_score5_0_uniform_20261119_120000.json")
            files.append("match_0025_gen0002_score5_0_uniform_20261119_120000.json")
        else:
            files.append(f"match_{100 + g:04d}_gen{g:04d}_score1_0_uniform_20261119_120000.json")
    kept = select_archive_replays(files, seed=7)
    assert kept[0] == "match_0020_gen0001_score8_0_uniform_20261119_120000.json"
    assert kept[1] == "match_0025_gen0002_score5_0_uniform_20261119_120000.json"


def test_select_archive_replays_small_gen_count_fillup_prefers_uniform_first_no_duplicates() -> None:
    """[RED] When N <= 15 with >15 files of both kinds, first pass takes every uniform file once, fill-up takes remaining bests."""
    files = []
    for g in range(1, 11):
        files.append(f"match_{2 * g - 1:04d}_gen{g:04d}_score{10 + g}_0_20261119_120000.json")
        files.append(f"match_{2 * g:04d}_gen{g:04d}_score1_0_uniform_20261119_120000.json")

    kept = select_archive_replays(files, seed=7)
    expected_first_10 = [f"match_{2 * g:04d}_gen{g:04d}_score1_0_uniform_20261119_120000.json" for g in range(1, 11)]
    expected_fillup_5 = [f"match_{2 * g - 1:04d}_gen{g:04d}_score{10 + g}_0_20261119_120000.json" for g in (10, 9, 8, 7, 6)]
    assert kept == expected_first_10 + expected_fillup_5
    assert len(set(kept)) == 15


def test_select_archive_replays_ignores_uniform_in_directory_name() -> None:
    """[RED] Only `_uniform_` in the file basename counts; directory names containing `_uniform_` do not."""
    files = []
    for g in range(1, 21):
        if g == 1:
            files.append("runs_uniform_/match_0001_gen0001_score90_0_20261119_120000.json")
            files.append("plain_dir/match_0002_gen0001_score1_0_uniform_20261119_120000.json")
        elif g == 2:
            files.append("runs_uniform_/match_0003_gen0002_score1_0_20261119_120000.json")
            files.append("plain_dir/match_0004_gen0002_score90_0_20261119_120000.json")
        else:
            files.append(f"plain_dir/match_{100 + g:04d}_gen{g:04d}_score1_0_uniform_20261119_120000.json")
    kept = select_archive_replays(files, seed=7)
    assert kept[0] == "plain_dir/match_0002_gen0001_score1_0_uniform_20261119_120000.json"
    assert kept[1] == "plain_dir/match_0004_gen0002_score90_0_20261119_120000.json"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_select_archive_replays_mixed_property_invariants(seed: int) -> None:
    """[RED] Property: random mixed file lists return unique input members of length min(15, len(files))."""
    import random

    rng = random.Random(seed + 100)
    files = []
    n_gens = rng.randint(6, 25)
    for g in range(1, n_gens + 1):
        if rng.random() < 0.8:
            files.append(f"match_{rng.randint(1, 9999):04d}_gen{g:04d}_score{rng.randint(1, 99)}_0_20261119_120000.json")
        if rng.random() < 0.7:
            files.append(f"match_{rng.randint(1, 9999):04d}_gen{g:04d}_score{rng.randint(0, 20)}_0_uniform_20261119_120000.json")
    rng.shuffle(files)

    kept = select_archive_replays(files, seed=seed)
    assert len(kept) == min(15, len(files))
    assert len(set(kept)) == len(kept)
    assert all(f in files for f in kept)


def test_archive_and_restore_both_kinds_keeps_uniform_replays(tmp_path: Path) -> None:
    """[RED] `archive_batch` + `load_archived_batch` on a `--replay-kind both` run keeps only `_uniform_` replays."""
    from football_rl.train_ppo import load_archived_batch

    trainer = _replay_cap_trainer(tmp_path, name="both_arch", num_envs=1, max_rl_steps=5, rollout_steps=5, replay_kind="both")
    try:
        trainer.train(total_updates=16)
    finally:
        trainer.close()

    rep_dir = tmp_path / "rep_both_arch"
    ckpt_dir = tmp_path / "ckpt_both_arch"
    arch_root = tmp_path / "archives"
    all_files = sorted(p.name for p in rep_dir.glob("match_*.json"))
    assert len(all_files) == 32

    dest = Path(archive_batch("both_run", str(ckpt_dir), str(rep_dir), str(arch_root)))
    archived_names = sorted(p.name for p in (dest / "replays").glob("match_*.json"))
    assert len(archived_names) == 15
    assert all("_uniform_" in name for name in archived_names)

    restored_ckpt = tmp_path / "restored_both_ckpt"
    restored_rep = tmp_path / "restored_both_rep"
    load_archived_batch("both_run", str(restored_ckpt), str(restored_rep), str(arch_root))
    assert sorted(p.name for p in restored_rep.glob("match_*.json")) == archived_names
    assert (restored_ckpt / "training_state.pt").exists()



def test_archive_batch_keeps_fifteen_matches(tmp_path: Path) -> None:
    """Verify archive_batch preserves only 15 matches and non-match files in archives."""
    ckpt_dir = tmp_path / "checkpoints"
    rep_dir = tmp_path / "replays"
    arch_dir = tmp_path / "archives"
    ckpt_dir.mkdir(parents=True)
    rep_dir.mkdir(parents=True)

    # Create dummy checkpoint
    (ckpt_dir / "training_state.pt").write_text("state", encoding="utf-8")
    (ckpt_dir / "training_history.json").write_text("{}", encoding="utf-8")

    # Create 40 replay files and 1 summary markdown
    (rep_dir / "matches_summary.md").write_text("# Summary", encoding="utf-8")
    for i in range(1, 41):
        (rep_dir / f"match_{i:04d}_gen{i:04d}_score5_0_20261107_120000.json").write_text(
            f'{{"metadata": {{"match_id": {i}, "generation": {i}}}}}',
            encoding="utf-8",
        )

    # Archive batch
    dest = archive_batch("test_exp", str(ckpt_dir), str(rep_dir), str(arch_dir))
    arch_replays = Path(dest) / "replays"

    # Verify summary markdown preserved
    assert (arch_replays / "matches_summary.md").exists()

    # Verify exactly 15 JSON replays in archive
    arch_json_files = list(arch_replays.glob("*.json"))
    assert len(arch_json_files) == 15

    # Verify active replays remain untouched (all 40 replays still in active replays)
    assert len(list(rep_dir.glob("*.json"))) == 40


def test_archive_batch_default_timestamp_naming(tmp_path: Path) -> None:
    """Verify archive_batch generates a timestamp-formatted folder name when name is omitted or empty."""
    ckpt_dir = tmp_path / "checkpoints"
    rep_dir = tmp_path / "replays"
    arch_dir = tmp_path / "archives"
    ckpt_dir.mkdir(parents=True)
    rep_dir.mkdir(parents=True)

    (ckpt_dir / "training_state.pt").write_text("state", encoding="utf-8")
    (rep_dir / "matches_summary.md").write_text("# Summary", encoding="utf-8")
    (rep_dir / "match_0001_gen0001_score1_0_20261107_120000.json").write_text(
        '{"metadata": {"match_id": 1, "generation": 1}}',
        encoding="utf-8",
    )

    # 1. Name is None
    dest_default = archive_batch(None, str(ckpt_dir), str(rep_dir), str(arch_dir))
    archive_name = Path(dest_default).name
    assert re.match(r"^\d{8}_\d{6}$", archive_name) is not None
    assert (Path(dest_default) / "checkpoints" / "training_state.pt").exists()
    assert (Path(dest_default) / "replays" / "matches_summary.md").exists()

    # 2. Name is empty or whitespace
    dest_empty = archive_batch("   ", str(ckpt_dir), str(rep_dir), str(arch_dir))
    archive_empty_name = Path(dest_empty).name
    assert re.match(r"^\d{8}_\d{6}$", archive_empty_name) is not None
    assert (Path(dest_empty) / "checkpoints" / "training_state.pt").exists()





def _quiet_cfg(save_dir: str = "replays", recording: bool = False):
    return replace(
        DEFAULT_CONFIG,
        replay=replace(DEFAULT_CONFIG.replay, enable_recording=recording, save_dir=save_dir),
    )


def _full_match(score=(3, 1), match_id=1, gen=1) -> dict:
    return {
        "match_id": match_id,
        "generation": gen,
        "score": list(score),
        "interest_score": 12.0,
        "possession_pct": [60.0, 40.0],
        "shots": [5, 2],
        "team_passes": [30, 20],
        "team_tackles": [4, 6],
        "team_saves": [1, 3],
        "field_5s_violations": 0,
        "keeper_5s_violations": 0,
        "replay_path": None,
    }


def test_migrate_team_color_keys_swaps_reward_keys_and_marks_schema() -> None:
    out = migrate_team_color_keys({"generations": [{"blue_mean_reward": 1.0, "red_mean_reward": 2.0}]})
    gen = out["generations"][0]
    assert gen["red_mean_reward"] == 1.0
    assert gen["blue_mean_reward"] == 2.0
    assert out["team_color_schema"] == 2


def test_migrate_team_color_keys_is_idempotent_and_does_not_mutate_input() -> None:
    legacy = {"generations": [{"generation": 1, "blue_mean_reward": 1.0, "red_mean_reward": 2.0}]}
    snapshot = copy.deepcopy(legacy)
    once = migrate_team_color_keys(legacy)
    assert legacy == snapshot
    twice = migrate_team_color_keys(copy.deepcopy(once))
    assert twice == once
    assert once["generations"][0]["generation"] == 1


def test_migrate_team_color_keys_degenerate_histories() -> None:
    assert migrate_team_color_keys({"generations": []}) == {"generations": [], "team_color_schema": 2}
    entry = {"generation": 7, "mean_field_reward": 0.5}
    out = migrate_team_color_keys({"generations": [dict(entry)]})
    assert out["generations"] == [entry]
    assert out["team_color_schema"] == 2
    # A history without a generations list only receives the marker.
    assert migrate_team_color_keys({}) == {"team_color_schema": 2}
    # Only one of the two keys present: the value moves to the opposite key, nothing is invented.
    lone = migrate_team_color_keys({"generations": [{"blue_mean_reward": 4.0}]})["generations"][0]
    assert lone == {"red_mean_reward": 4.0}


def test_series_summary_team_a_is_red() -> None:
    ns = SimpleNamespace(match_history=[{"score": [3, 1]}])
    series = ParameterSharingPPOTrainer._compute_series_summary(ns)
    assert series["red_wins"] == 1
    assert series["blue_wins"] == 0
    assert series["red_goals"] == 3
    assert series["blue_goals"] == 1
    assert series["draws"] == 0


def test_series_summary_counts_draws_and_blue_wins() -> None:
    ns = SimpleNamespace(match_history=[{"score": [0, 2]}, {"score": [1, 1]}, {"score": [2, 0]}])
    series = ParameterSharingPPOTrainer._compute_series_summary(ns)
    assert (series["red_wins"], series["blue_wins"], series["draws"]) == (1, 1, 1)
    assert (series["red_goals"], series["blue_goals"]) == (3, 3)
    empty = ParameterSharingPPOTrainer._compute_series_summary(SimpleNamespace(match_history=[]))
    assert empty == {"red_wins": 0, "blue_wins": 0, "draws": 0, "red_goals": 0, "blue_goals": 0}


def test_generation_record_stores_team_a_reward_under_red_key(tmp_path: Path) -> None:
    trainer = ParameterSharingPPOTrainer(
        config=_quiet_cfg(), num_envs=1, rollout_steps=10, device="cpu",
        checkpoint_dir=str(tmp_path / "ckpt"), seed=3,
    )
    captured = {}
    original = trainer.collect_rollouts

    def spy(*args, **kwargs):
        captured["rollout"] = original(*args, **kwargs)
        return captured["rollout"]

    trainer.collect_rollouts = spy
    trainer.train(total_updates=1)
    rewards = captured["rollout"]["rewards"]
    team_a = round(float(rewards[:, :, [0, 1, 2, 3, 4, 5]].mean()), 5)
    team_b = round(float(rewards[:, :, [6, 7, 8, 9, 10, 11]].mean()), 5)
    assert team_a != team_b
    rec = trainer.generation_history[0]
    assert rec["red_mean_reward"] == team_a
    assert rec["blue_mean_reward"] == team_b


def test_save_backs_up_legacy_history_once_and_stamps_schema(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    legacy_text = json.dumps({"generations": [{"blue_mean_reward": 1.0, "red_mean_reward": 2.0}]})
    (ckpt / "training_history.json").write_text(legacy_text, encoding="utf-8")
    trainer = ParameterSharingPPOTrainer(
        config=_quiet_cfg(), num_envs=1, rollout_steps=10, device="cpu",
        checkpoint_dir=str(ckpt), seed=3,
    )
    trainer.save_checkpoints()
    backup = ckpt / "training_history.pre_color_schema.json"
    assert backup.read_text(encoding="utf-8") == legacy_text
    saved = json.loads((ckpt / "training_history.json").read_text(encoding="utf-8"))
    assert saved["team_color_schema"] == 2

    trainer.save_checkpoints()
    assert backup.read_text(encoding="utf-8") == legacy_text


def test_load_migrates_legacy_generation_history_exactly_once(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    kwargs = dict(config=_quiet_cfg(), num_envs=1, rollout_steps=10, device="cpu",
                  checkpoint_dir=str(ckpt), seed=3)
    writer = ParameterSharingPPOTrainer(**kwargs)
    writer.generation_history = [{"generation": 1, "blue_mean_reward": 1.0, "red_mean_reward": 2.0}]
    writer.save_checkpoints()
    state_path = ckpt / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    state.pop("team_color_schema", None)
    torch.save(state, state_path)

    reader = ParameterSharingPPOTrainer(**kwargs)
    assert reader.load_checkpoints()
    rec = reader.generation_history[0]
    assert (rec["red_mean_reward"], rec["blue_mean_reward"]) == (1.0, 2.0)

    reader.save_checkpoints()
    again = ParameterSharingPPOTrainer(**kwargs)
    assert again.load_checkpoints()
    rec = again.generation_history[0]
    assert (rec["red_mean_reward"], rec["blue_mean_reward"]) == (1.0, 2.0)


def test_status_and_summary_label_team_a_as_red(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "training_history.json").write_text(
        json.dumps({"current_generation": 1, "matches": [_full_match((3, 1))]}), encoding="utf-8"
    )
    print_active_status(str(ckpt), str(tmp_path / "rep"))
    out = capsys.readouterr().out
    assert "Series (Red - Blue) : 1 Wins - 0 Draws - 0 Wins" in out
    assert "Goals: Red 3 - 1 Blue" in out
    assert "Score (R-B)" in out

    trainer = ParameterSharingPPOTrainer(
        config=_quiet_cfg(save_dir=str(tmp_path / "rep"), recording=True), num_envs=1,
        rollout_steps=10, device="cpu", checkpoint_dir=str(ckpt), seed=3,
    )
    trainer.match_history = [_full_match((3, 1))]
    path = trainer._write_matches_summary_markdown()
    text = Path(path).read_text(encoding="utf-8")
    assert "Score (Red - Blue)" in text
    assert "Poss% (R/B)" in text
    assert "Head-to-Head Series (Red - Draws - Blue):** `1 W - 0 D - 0 W`" in text
    assert "Goals: **Red `3` - `1` Blue**" in text
    assert "Blue - Red" not in text


def _recording_trainer(tmp_path: Path, **kwargs) -> ParameterSharingPPOTrainer:
    return ParameterSharingPPOTrainer(
        config=_quiet_cfg(save_dir=str(tmp_path / "rep"), recording=True), num_envs=1,
        rollout_steps=10, device="cpu", checkpoint_dir=str(tmp_path / "ckpt"), seed=3, **kwargs,
    )


def test_summary_not_written_to_cwd(tmp_path: Path, monkeypatch) -> None:
    decoy = tmp_path / "decoy.md"
    decoy.write_bytes(b"decoy-bytes\n")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "matches_summary.md").symlink_to(decoy)
    monkeypatch.chdir(cwd)

    trainer = _recording_trainer(tmp_path)
    trainer.train(total_updates=1)

    assert decoy.read_bytes() == b"decoy-bytes\n"
    assert (tmp_path / "rep" / "matches_summary.md").exists()


def test_root_summary_copy_written_only_when_enabled_and_never_through_symlink(
    tmp_path: Path, monkeypatch
) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    trainer = _recording_trainer(tmp_path, write_root_summary=True)
    trainer.current_generation = 7
    trainer._write_matches_summary_markdown()
    root = cwd / "matches_summary.md"
    assert root.exists() and not root.is_symlink()
    assert "`7`" in root.read_text(encoding="utf-8")

    root.unlink()
    decoy = tmp_path / "decoy.md"
    decoy.write_bytes(b"keep")
    root.symlink_to(decoy)
    trainer._write_matches_summary_markdown()
    assert decoy.read_bytes() == b"keep"

    root.unlink()
    off = _recording_trainer(tmp_path)
    off._write_matches_summary_markdown()
    assert not root.exists()


def test_interrupt_saves_completed_generation_count(tmp_path: Path) -> None:
    trainer = _recording_trainer(tmp_path)
    original = trainer.collect_rollouts

    def interrupting(active_generation: int):
        if active_generation == 2:
            raise KeyboardInterrupt
        return original(active_generation=active_generation)

    trainer.collect_rollouts = interrupting
    trainer.train(total_updates=3)

    state = torch.load(tmp_path / "ckpt" / "training_state.pt", map_location="cpu", weights_only=False)
    assert len(state["generation_history"]) == 1
    assert state["current_generation"] == 1
    history = json.loads((tmp_path / "ckpt" / "training_history.json").read_text(encoding="utf-8"))
    assert history["current_generation"] == 1


def test_interrupt_before_first_generation_completes_saves_zero(tmp_path: Path) -> None:
    trainer = _recording_trainer(tmp_path)

    def interrupting(active_generation: int):
        raise KeyboardInterrupt

    trainer.collect_rollouts = interrupting
    trainer.train(total_updates=2)
    state = torch.load(tmp_path / "ckpt" / "training_state.pt", map_location="cpu", weights_only=False)
    assert state["current_generation"] == 0
    assert state["generation_history"] == []


# ==============================================================================
# CLI defaults, generation provenance and resume mismatch warning
# ==============================================================================
def test_cli_defaults_run_eight_envs_for_full_matches() -> None:
    import football_rl.train_ppo as tp

    args = tp.build_arg_parser().parse_args([])
    assert args.num_envs == 8
    assert args.rollout_steps == 2700
    assert args.total_updates == 100
    assert args.seed == 42
    assert args.save_best_only is False
    assert args.resume is False


def test_cli_num_envs_override_and_help_documents_default() -> None:
    import football_rl.train_ppo as tp

    parser = tp.build_arg_parser()
    assert parser.parse_args(["--num-envs", "1"]).num_envs == 1
    help_text = parser.format_help()
    assert "--num-envs" in help_text
    assert "default: 8" in help_text.replace("\n", " ").replace("  ", " ")


def test_generation_record_carries_num_envs_and_rollout_steps(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    trainer = ParameterSharingPPOTrainer(
        config=_quiet_cfg(), num_envs=2, rollout_steps=10, device="cpu",
        checkpoint_dir=str(ckpt), seed=3,
    )
    trainer.train(total_updates=1)
    rec = trainer.generation_history[-1]
    assert rec["num_envs"] == 2
    assert rec["rollout_steps"] == 10

    history = json.loads((ckpt / "training_history.json").read_text(encoding="utf-8"))
    assert history["generations"][-1]["num_envs"] == 2
    assert history["generations"][-1]["rollout_steps"] == 10
    state = torch.load(ckpt / "training_state.pt", map_location="cpu", weights_only=False)
    assert state["generation_history"][-1]["num_envs"] == 2
    assert state["generation_history"][-1]["rollout_steps"] == 10


def _write_history(ckpt: Path, generations) -> None:
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "training_history.json").write_text(json.dumps({"generations": generations}), encoding="utf-8")


def test_resume_warning_fires_when_last_generation_num_envs_differs(tmp_path: Path) -> None:
    import football_rl.train_ppo as tp

    ckpt = tmp_path / "ckpt"
    _write_history(ckpt, [{"generation": 1, "num_envs": 8}, {"generation": 2, "num_envs": 8}])
    msg = tp.resume_num_envs_warning(str(ckpt), 1)
    assert msg is not None
    assert msg.startswith("WARNING")
    assert "num_envs=8" in msg and "num_envs=1" in msg


def test_resume_warning_silent_on_match_missing_key_or_missing_history(tmp_path: Path) -> None:
    import football_rl.train_ppo as tp

    ckpt = tmp_path / "ckpt"
    _write_history(ckpt, [{"generation": 1, "num_envs": 4}, {"generation": 2, "num_envs": 8}])
    assert tp.resume_num_envs_warning(str(ckpt), 8) is None  # only the LAST record counts

    _write_history(ckpt, [{"generation": 1, "num_envs": 8}, {"generation": 2}])
    assert tp.resume_num_envs_warning(str(ckpt), 1) is None  # legacy last record without the key

    _write_history(ckpt, [])
    assert tp.resume_num_envs_warning(str(ckpt), 1) is None
    assert tp.resume_num_envs_warning(str(tmp_path / "absent"), 1) is None

    (ckpt / "training_history.json").write_text("{not json", encoding="utf-8")
    assert tp.resume_num_envs_warning(str(ckpt), 1) is None


# ==============================================================================
# Co-evolution soft-sync over a rolling window of completed matches
# ==============================================================================
def _matches_with_totals(goals_a: int, goals_b: int, n: int = 8, gen: int = 1) -> list:
    """`n` completed-match records whose goals sum to the given team totals (spread round-robin)."""
    recs = []
    for i in range(n):
        a = goals_a // n + (1 if i < goals_a % n else 0)
        b = goals_b // n + (1 if i < goals_b % n else 0)
        recs.append({"generation": gen, "score": [a, b]})
    return recs


def _coevo_trainer(tmp_path: Path, num_envs: int = 1) -> ParameterSharingPPOTrainer:
    trainer = ParameterSharingPPOTrainer(
        config=_quiet_cfg(), num_envs=num_envs, rollout_steps=10, device="cpu",
        checkpoint_dir=str(tmp_path / "ckpt"), seed=3,
    )
    with torch.no_grad():
        # Team A (stored as blue_*) = 1.0, team B (stored as red_*) = 0.0 everywhere.
        for p in list(trainer.blue_field_policy.parameters()) + list(trainer.blue_keeper_policy.parameters()):
            p.fill_(1.0)
        for p in list(trainer.red_field_policy.parameters()) + list(trainer.red_keeper_policy.parameters()):
            p.fill_(0.0)
    return trainer


def _team_b_weight(trainer: ParameterSharingPPOTrainer) -> float:
    return float(next(trainer.red_field_policy.parameters()).flatten()[0])


def _team_a_weight(trainer: ParameterSharingPPOTrainer) -> float:
    return float(next(trainer.blue_field_policy.parameters()).flatten()[0])


def test_coevo_window_constant_is_twenty_four_matches() -> None:
    import football_rl.train_ppo as tp

    assert tp.COEVO_WINDOW_MATCHES == 24


def test_coevo_eight_envs_reproduces_three_generation_decisions(tmp_path: Path) -> None:
    """8 matches per generation: window of 24 matches == old 3-generation window.

    Hand-computed per-generation (A, B) goal totals and rolling-window sums:
      gen3: A=4+3+2=9,  B=1+1+1=3   -> 9 >= 3*3 and total 12  -> sync Team B (clear)
      gen6: A=1+0+1=2,  B=5+3+2=10  -> 10 >= 3*2 and total 12 -> sync Team A (clear)
      gen7-9: A=B=6 (no side dominates), gen10: window gens 8-10 A=12, B=4 -> 12 >= 12 -> sync Team B
      gen11-12: only 8 / 16 matches in the freshly cleared window -> never evaluated
      gen13: gens 11-13 A=12, B=1 -> sync Team B
    """
    trainer = _coevo_trainer(tmp_path, num_envs=8)
    per_gen = {
        1: (4, 1), 2: (3, 1), 3: (2, 1),
        4: (1, 5), 5: (0, 3), 6: (1, 2),
        7: (2, 2), 8: (2, 2), 9: (2, 2),
        10: (8, 0),
        11: (5, 1), 12: (4, 0), 13: (3, 0),
    }
    synced = []
    for gen, (a, b) in per_gen.items():
        if trainer._coevolution_step(_matches_with_totals(a, b, gen=gen)):
            synced.append(gen)
    assert synced == [3, 6, 10, 13]
    assert len(trainer.recent_match_scores) == 0


def test_coevo_single_env_three_generations_of_three_nil_does_not_sync(tmp_path: Path) -> None:
    trainer = _coevo_trainer(tmp_path, num_envs=1)
    for gen in (1, 2, 3):
        assert not trainer._coevolution_step([{"generation": gen, "score": [3, 0]}])
    assert list(trainer.recent_match_scores) == [(3, 0)] * 3
    assert _team_b_weight(trainer) == 0.0 and _team_a_weight(trainer) == 1.0


def test_coevo_boundary_exactly_three_to_one_syncs_team_b_towards_a(tmp_path: Path) -> None:
    trainer = _coevo_trainer(tmp_path)
    recs = _matches_with_totals(6, 2, n=24)  # 24 matches: A=6, B=2 -> total 8, 6 >= 3*2
    assert trainer._coevolution_step(recs) is True
    assert _team_b_weight(trainer) == pytest.approx(0.35)  # 0.65*0 + 0.35*1
    assert _team_a_weight(trainer) == 1.0  # the dominant team is untouched
    assert len(trainer.recent_match_scores) == 0


def test_coevo_boundary_five_to_two_does_not_sync(tmp_path: Path) -> None:
    trainer = _coevo_trainer(tmp_path)
    assert trainer._coevolution_step(_matches_with_totals(5, 2, n=24)) is False  # 5 < 3*2
    assert _team_b_weight(trainer) == 0.0
    assert len(trainer.recent_match_scores) == 24


def test_coevo_total_goals_below_eight_does_not_sync(tmp_path: Path) -> None:
    trainer = _coevo_trainer(tmp_path)
    assert trainer._coevolution_step(_matches_with_totals(6, 1, n=24)) is False  # total 7
    assert _team_b_weight(trainer) == 0.0
    assert len(trainer.recent_match_scores) == 24


def test_coevo_sync_log_names_both_teams_by_their_rendered_colour(tmp_path: Path, capsys) -> None:
    """[GUARD] The sync line follows the team colour schema: team A (seats 0-5, `blue_*` policy files, `score[0]`) is drawn
    red and is "Red" in every console line; team B (seats 6-11, `red_*` files, `score[1]`) is drawn blue and is "Blue".
    """
    from football_rl.replay import COLOR_TEAM_A, COLOR_TEAM_B

    assert COLOR_TEAM_A[0] > COLOR_TEAM_A[2] and COLOR_TEAM_B[2] > COLOR_TEAM_B[0]  # A drawn red, B drawn blue

    trainer = _coevo_trainer(tmp_path)
    assert trainer.blue_keeper_indices == [0] and trainer.blue_field_indices == [1, 2, 3, 4, 5]
    assert trainer.red_keeper_indices == [6] and trainer.red_field_indices == [7, 8, 9, 10, 11]

    # Team A scores 6, team B scores 2: A (Red) dominated, so B (Blue, stored as red_*) is pulled towards A.
    assert trainer._coevolution_step(_matches_with_totals(6, 2, n=24)) is True
    line = capsys.readouterr().out
    assert "[Co-Evolution Sync] Red dominated (6-2 over 24 matches). Soft-synced Blue policy (tau=0.35)" in line
    assert _team_b_weight(trainer) == pytest.approx(0.35) and _team_a_weight(trainer) == 1.0

    # Team B scores 7, team A scores 1: B (Blue) dominated, so A (Red, stored as blue_*) is pulled towards B.
    trainer2 = _coevo_trainer(tmp_path / "second")
    assert trainer2._coevolution_step(_matches_with_totals(1, 7, n=24)) is True
    line2 = capsys.readouterr().out
    assert "[Co-Evolution Sync] Blue dominated (7-1 over 24 matches). Soft-synced Red policy (tau=0.35)" in line2
    assert _team_a_weight(trainer2) == pytest.approx(0.65) and _team_b_weight(trainer2) == 0.0


def test_coevo_zero_goal_opponent_uses_floor_of_one(tmp_path: Path) -> None:
    trainer = _coevo_trainer(tmp_path)
    assert trainer._coevolution_step(_matches_with_totals(8, 0, n=24)) is True  # 8 >= 3*max(1,0)=3
    assert _team_b_weight(trainer) == pytest.approx(0.35)


def test_coevo_window_is_a_sliding_deque_of_last_24_matches(tmp_path: Path) -> None:
    trainer = _coevo_trainer(tmp_path)
    # 23 scoreless matches then one 1-0: window not full -> no evaluation even though ordering is irrelevant.
    trainer._coevolution_step([{"generation": 1, "score": [0, 0]}] * 23)
    assert len(trainer.recent_match_scores) == 23
    trainer._coevolution_step([{"generation": 2, "score": [1, 0]}])
    assert len(trainer.recent_match_scores) == 24
    # Oldest entry (0, 0) is evicted, newest is retained, in completion order.
    trainer._coevolution_step([{"generation": 3, "score": [2, 0]}])
    scores = list(trainer.recent_match_scores)
    assert len(scores) == 24
    assert scores[-2:] == [(1, 0), (2, 0)]
    assert scores[0] == (0, 0)


def test_coevo_empty_generation_is_a_noop(tmp_path: Path) -> None:
    trainer = _coevo_trainer(tmp_path)
    assert trainer._coevolution_step([]) is False
    assert len(trainer.recent_match_scores) == 0


def test_coevo_is_disabled_for_shared_teams(tmp_path: Path) -> None:
    trainer = ParameterSharingPPOTrainer(
        config=_quiet_cfg(), num_envs=1, rollout_steps=10, device="cpu",
        checkpoint_dir=str(tmp_path / "ckpt"), seed=3, separate_teams=False,
    )
    assert trainer._coevolution_step(_matches_with_totals(24, 0, n=24)) is False
    assert len(trainer.recent_match_scores) == 0


def test_coevo_window_persists_as_pairs_and_round_trips(tmp_path: Path) -> None:
    trainer = _coevo_trainer(tmp_path)
    trainer._coevolution_step([{"generation": 1, "score": [2, 1]}, {"generation": 1, "score": [0, 3]}])
    trainer.save_checkpoints()
    state = torch.load(tmp_path / "ckpt" / "training_state.pt", map_location="cpu", weights_only=False)
    assert state["recent_match_scores"] == [[2, 1], [0, 3]] or state["recent_match_scores"] == [(2, 1), (0, 3)]
    assert "recent_team_goals" not in state

    reader = ParameterSharingPPOTrainer(
        config=_quiet_cfg(), num_envs=1, rollout_steps=10, device="cpu",
        checkpoint_dir=str(tmp_path / "ckpt"), seed=3,
    )
    assert reader.load_checkpoints()
    assert list(reader.recent_match_scores) == [(2, 1), (0, 3)]
    assert reader.recent_match_scores.maxlen == 24


def test_coevo_legacy_state_with_only_recent_team_goals_starts_empty(tmp_path: Path) -> None:
    trainer = _coevo_trainer(tmp_path)
    trainer.save_checkpoints()
    state_path = tmp_path / "ckpt" / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    state.pop("recent_match_scores", None)
    state["recent_team_goals"] = [(9, 0), (9, 0), (9, 0)]
    torch.save(state, state_path)

    reader = ParameterSharingPPOTrainer(
        config=_quiet_cfg(), num_envs=1, rollout_steps=10, device="cpu",
        checkpoint_dir=str(tmp_path / "ckpt"), seed=3,
    )
    assert reader.load_checkpoints()
    assert len(reader.recent_match_scores) == 0
    assert reader.recent_match_scores.maxlen == 24


def test_train_feeds_completed_matches_of_each_generation_into_window(tmp_path: Path, capsys) -> None:
    """A generation that completes one 3-0 match (num_envs=1) never reaches the 24-match window in 3 gens."""
    trainer = _coevo_trainer(tmp_path, num_envs=1)
    original = trainer.collect_rollouts

    def collect_with_one_match(active_generation: int):
        out = original(active_generation=active_generation)
        trainer.match_history.append({"generation": active_generation, "score": [3, 0], "match_id": active_generation})
        return out

    trainer.collect_rollouts = collect_with_one_match
    trainer.train(total_updates=3)
    assert list(trainer.recent_match_scores) == [(3, 0)] * 3
    assert "Co-Evolution Sync" not in capsys.readouterr().out


# ==============================================================================
# Uniform replay selection, generation snapshots and per-match kick KPIs
# ==============================================================================
SNAPSHOT_FILES = (
    "field_policy.pt",
    "keeper_policy.pt",
    "blue_field_policy.pt",
    "blue_keeper_policy.pt",
    "red_field_policy.pt",
    "red_keeper_policy.pt",
    "training_state.pt",
)


def test_uniform_env_index_is_uniform_over_200_seeded_generations() -> None:
    num_envs, gens = 8, 200
    counts = [0] * num_envs
    for g in range(1, gens + 1):
        counts[uniform_env_index(seed=42, generation=g, num_envs=num_envs)] += 1
    expected = gens / num_envs
    tolerance = 3.0 * math.sqrt(gens * (1 / num_envs) * (1 - 1 / num_envs))
    assert sum(counts) == gens
    assert all(abs(c - expected) <= tolerance for c in counts), counts


def test_uniform_env_index_is_seeded_bounded_and_trivial_for_one_env() -> None:
    picks = [uniform_env_index(seed=7, generation=g, num_envs=8) for g in range(1, 60)]
    assert picks == [uniform_env_index(seed=7, generation=g, num_envs=8) for g in range(1, 60)]
    assert picks != [uniform_env_index(seed=8, generation=g, num_envs=8) for g in range(1, 60)]
    assert all(0 <= p < 8 for p in picks)
    assert {uniform_env_index(seed=7, generation=g, num_envs=1) for g in range(1, 20)} == {0}
    assert type(picks[0]) is int


def _snapshot_dirs(ckpt: Path) -> list:
    root = ckpt / "snapshots"
    return sorted(p.name for p in root.iterdir()) if root.exists() else []


def test_snapshot_written_at_generation_50_and_100_but_not_49(tmp_path: Path) -> None:
    trainer = _recording_trainer(tmp_path)
    ckpt = tmp_path / "ckpt"
    trainer.current_generation = 48
    trainer.train(total_updates=1)  # completes generation 49
    assert trainer.current_generation == 49
    assert _snapshot_dirs(ckpt) == []

    trainer.train(total_updates=1)  # completes generation 50
    assert _snapshot_dirs(ckpt) == ["gen0050"]
    snap = ckpt / "snapshots" / "gen0050"
    assert sorted(p.name for p in snap.iterdir()) == sorted(SNAPSHOT_FILES)
    state = torch.load(snap / "training_state.pt", map_location="cpu", weights_only=False)
    assert state["current_generation"] == 50

    trainer.current_generation = 99
    trainer.train(total_updates=1)
    assert _snapshot_dirs(ckpt) == ["gen0050", "gen0100"]


def test_snapshot_is_frozen_while_live_checkpoint_moves_on(tmp_path: Path) -> None:
    trainer = _recording_trainer(tmp_path)
    ckpt = tmp_path / "ckpt"
    trainer.current_generation = 49
    trainer.train(total_updates=2)  # generations 50 and 51
    live = torch.load(ckpt / "training_state.pt", map_location="cpu", weights_only=False)
    snap = torch.load(ckpt / "snapshots" / "gen0050" / "training_state.pt", map_location="cpu", weights_only=False)
    assert live["current_generation"] == 51 and snap["current_generation"] == 50
    assert _snapshot_dirs(ckpt) == ["gen0050"]


def test_resume_and_repeated_save_do_not_duplicate_or_rewrite_a_snapshot(tmp_path: Path) -> None:
    trainer = _recording_trainer(tmp_path)
    ckpt = tmp_path / "ckpt"
    trainer.current_generation = 49
    trainer.train(total_updates=1)
    snap = ckpt / "snapshots" / "gen0050"
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snap.iterdir()}

    trainer.save_checkpoints()  # e.g. an interrupt landing exactly on the snapshot generation
    resumed = _recording_trainer(tmp_path)
    assert resumed.load_checkpoints() and resumed.current_generation == 50
    for p in resumed.cpu_blue_field_policy.parameters():
        p.data.add_(1.0)  # the live weights now differ from the snapshot
    resumed.blue_field_policy.load_state_dict(resumed.cpu_blue_field_policy.state_dict())
    resumed.save_checkpoints()

    assert _snapshot_dirs(ckpt) == ["gen0050"]
    assert sorted(p.name for p in (ckpt / "snapshots").iterdir()) == ["gen0050"]
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snap.iterdir()}
    assert after == before


def test_no_snapshot_at_generation_zero(tmp_path: Path) -> None:
    trainer = _recording_trainer(tmp_path)
    trainer.save_checkpoints()
    assert _snapshot_dirs(tmp_path / "ckpt") == []


@pytest.fixture(scope="module")
def one_full_generation(tmp_path_factory):
    """One 2,700-step generation on one recording env; only the uniform capture can export a replay."""
    root = tmp_path_factory.mktemp("full_gen")
    cfg = replace(
        DEFAULT_CONFIG,
        replay=replace(
            DEFAULT_CONFIG.replay,
            enable_recording=True,
            interest_threshold=float("inf"),
            save_dir=str(root / "rep"),
        ),
    )
    trainer = ParameterSharingPPOTrainer(
        config=cfg, num_envs=1, rollout_steps=2700, device="cpu",
        checkpoint_dir=str(root / "ckpt"), seed=3,
    )
    trainer.train(total_updates=1)
    return root, trainer


def test_each_generation_exports_one_uniform_replay_from_the_pre_drawn_env(one_full_generation) -> None:
    root, trainer = one_full_generation
    files = sorted(p.name for p in (root / "rep").glob("match_*.json"))
    assert len(files) == 1 and "_uniform_" in files[0] and "_gen0001_" in files[0]
    rec = trainer.generation_history[0]
    assert rec["uniform_env"] == uniform_env_index(seed=3, generation=1, num_envs=1) == 0
    assert Path(rec["uniform_replay"]).name == files[0]
    assert trainer.match_history[0]["replay_path"] == "" or "uniform" not in trainer.match_history[0]["replay_path"]


def test_uniform_replay_frames_carry_action_means_of_shape_12_by_4(one_full_generation) -> None:
    root, trainer = one_full_generation
    path = Path(trainer.generation_history[0]["uniform_replay"])
    frames = json.loads(path.read_text(encoding="utf-8"))["frames"]
    assert len(frames) == 10801
    assert "action_mean" not in frames[0]
    for f in (frames[1], frames[5400], frames[-1]):
        mean = f["action_mean"]
        assert len(mean) == 12 and all(len(row) == ACTION_DIM for row in mean)
        assert len(f["actions"]) == 12 and f["actions"] != mean


def test_match_record_carries_balanced_kick_and_keeper_hold_counters(one_full_generation) -> None:
    _, trainer = one_full_generation
    m = trainer.match_history[0]
    for key in ("gk_holds", "kicks", "kick_to_teammate", "kick_to_opponent", "kick_other"):
        assert isinstance(m[key], int) and m[key] >= 0, key
    assert m["kicks"] > 0
    assert m["kicks"] == m["kick_to_teammate"] + m["kick_to_opponent"] + m["kick_other"]
    persisted = json.loads((Path(trainer.checkpoint_dir) / "training_history.json").read_text(encoding="utf-8"))
    assert persisted["matches"][0]["kicks"] == m["kicks"]


def test_collect_rollouts_alone_exports_no_uniform_replay(tmp_path: Path) -> None:
    """Only `train()` owns the uniform replay; evaluation tools that call `collect_rollouts` see no extra files."""
    cfg = replace(
        DEFAULT_CONFIG,
        physics=replace(DEFAULT_CONFIG.physics, max_rl_steps=4),
        replay=replace(
            DEFAULT_CONFIG.replay,
            enable_recording=True,
            interest_threshold=float("inf"),
            save_dir=str(tmp_path / "rep"),
        ),
    )
    trainer = ParameterSharingPPOTrainer(
        config=cfg, num_envs=1, rollout_steps=4, device="cpu", checkpoint_dir=str(tmp_path / "ckpt"), seed=3
    )
    rollout = trainer.collect_rollouts(active_generation=1)
    assert rollout["completed_episodes"] == 1
    assert not (tmp_path / "rep").exists() or list((tmp_path / "rep").glob("*.json")) == []
    trainer.train(total_updates=1)
    assert len(list((tmp_path / "rep").glob("match_*_uniform_*.json"))) == 1


# ==============================================================================
# Critic-only warm-up, critic-fit correlations, seed provenance & update digest
# ==============================================================================
UPDATE_DIGEST_FILE = Path(__file__).parent / "golden" / "update_digests.json"


def _hash_tensors(tensors) -> str:
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _policy_groups(trainer: ParameterSharingPPOTrainer) -> list[tuple[str, torch.nn.Module, torch.optim.Optimizer]]:
    return [
        ("blue_field", trainer.blue_field_policy, trainer.blue_field_optimizer),
        ("blue_keeper", trainer.blue_keeper_policy, trainer.blue_keeper_optimizer),
        ("red_field", trainer.red_field_policy, trainer.red_field_optimizer),
        ("red_keeper", trainer.red_keeper_policy, trainer.red_keeper_optimizer),
    ]


def _snapshot_actor_and_adam(
    trainer: ParameterSharingPPOTrainer,
) -> tuple[dict[tuple[str, str], bytes], dict[tuple[str, str, str], bytes], dict[tuple[str, str], bytes]]:
    actor_bytes: dict[tuple[str, str], bytes] = {}
    adam_bytes: dict[tuple[str, str, str], bytes] = {}
    critic_bytes: dict[tuple[str, str], bytes] = {}
    for role, pol, opt in _policy_groups(trainer):
        for name, param in pol.named_parameters():
            raw = param.detach().cpu().contiguous().numpy().tobytes()
            if name.startswith("actor_"):
                actor_bytes[(role, name)] = raw
                st = opt.state.get(param, {})
                for k in ("exp_avg", "exp_avg_sq", "step"):
                    if k in st:
                        val = st[k]
                        adam_bytes[(role, name, k)] = (
                            val.detach().cpu().contiguous().numpy().tobytes()
                            if isinstance(val, torch.Tensor)
                            else repr(val).encode("utf-8")
                        )
            elif name.startswith("critic."):
                critic_bytes[(role, name)] = raw
    return actor_bytes, adam_bytes, critic_bytes


def test_update_policies_matches_golden_digest(tmp_path: Path) -> None:
    """[GUARD] With N=0, weights after one seeded update_policies match the pre-edit SHA-256."""
    golden = json.loads(UPDATE_DIGEST_FILE.read_text(encoding="utf-8"))
    if golden["torch"] != torch.__version__ or golden["torch_threads"] != torch.get_num_threads():
        pytest.skip(
            f"digests recorded with torch {golden['torch']} x{golden['torch_threads']} threads; "
            f"running torch {torch.__version__} x{torch.get_num_threads()}"
        )
    trainer = ParameterSharingPPOTrainer(
        config=replace(_quiet_cfg(), rewards=replace(DEFAULT_CONFIG.rewards, shaping_mode="legacy")),
        num_envs=golden["num_envs"],
        rollout_steps=golden["rollout_steps"],
        device="cpu",
        checkpoint_dir=str(tmp_path / "ckpt"),
        seed=golden["seed"],
    )
    try:
        rollout = trainer.collect_rollouts(active_generation=1)
        trainer.update_policies(rollout)
        for role, pol, _ in _policy_groups(trainer):
            sd = pol.state_dict()
            actor_keys = [k for k in sd if k.startswith("actor_")]
            critic_keys = [k for k in sd if k.startswith("critic.")]
            assert _hash_tensors([sd[k] for k in actor_keys]) == golden["policies"][role]["actor"]
            assert _hash_tensors([sd[k] for k in critic_keys]) == golden["policies"][role]["critic"]
            assert _hash_tensors(list(sd.values())) == golden["policies"][role]["all"]
    finally:
        trainer.close()


def test_critic_only_generation_freezes_actors_and_adam_state_and_updates_critics(tmp_path: Path) -> None:
    """[RED] A critic-only generation keeps all 4 actors and their Adam state byte-identical while updating critics."""
    trainer = ParameterSharingPPOTrainer(
        config=_quiet_cfg(),
        num_envs=1,
        rollout_steps=16,
        device="cpu",
        checkpoint_dir=str(tmp_path / "ckpt"),
        seed=5,
        critic_only_until_gen=2,
    )
    try:
        # Run one direct update first so Adam state (exp_avg, exp_avg_sq, step) is populated and non-zero.
        trainer.update_policies(trainer.collect_rollouts(active_generation=1))
        with torch.no_grad():
            # Set logstd outside [-1.25, -0.35] to verify critic-only skips logstd clamping too.
            trainer.blue_field_policy.actor_logstd.fill_(-2.0)

        actor_before, adam_before, critic_before = _snapshot_actor_and_adam(trainer)
        assert len(adam_before) > 0

        trainer.train(total_updates=1)  # gen 1 <= 2
        actor_after, adam_after, critic_after = _snapshot_actor_and_adam(trainer)

        assert actor_after == actor_before
        assert adam_after == adam_before
        for role in ("blue_field", "blue_keeper", "red_field", "red_keeper"):
            assert any(
                critic_after[(role, name)] != critic_before[(role, name)]
                for (r, name) in critic_before
                if r == role
            ), f"{role} critic did not update"
        assert trainer.generation_history[-1]["critic_only"] is True
    finally:
        trainer.close()


def test_generation_after_critic_only_warmup_updates_actors(tmp_path: Path) -> None:
    """[RED] Generation N + 1 updates actor weights for all four policies."""
    trainer = ParameterSharingPPOTrainer(
        config=_quiet_cfg(),
        num_envs=1,
        rollout_steps=16,
        device="cpu",
        checkpoint_dir=str(tmp_path / "ckpt"),
        seed=5,
        critic_only_until_gen=1,
    )
    try:
        trainer.train(total_updates=1)  # gen 1: critic-only
        actor_after_1, _, _ = _snapshot_actor_and_adam(trainer)

        trainer.train(total_updates=1)  # gen 2 = N + 1: normal update
        actor_after_2, _, _ = _snapshot_actor_and_adam(trainer)

        for role in ("blue_field", "blue_keeper", "red_field", "red_keeper"):
            assert any(
                actor_after_2[(role, name)] != actor_after_1[(role, name)]
                for (r, name) in actor_after_1
                if r == role
            ), f"{role} actor did not update at gen N+1"
        assert trainer.generation_history[-1]["critic_only"] is False
    finally:
        trainer.close()


def test_resume_inside_critic_only_warmup_completes_remaining_warmup_then_unfreezes(tmp_path: Path) -> None:
    """[RED] With N=3 interrupted after gen 2, resumed gen 3 is critic-only and gen 4 updates actors."""
    ckpt = tmp_path / "ckpt"
    trainer = ParameterSharingPPOTrainer(
        config=_quiet_cfg(),
        num_envs=1,
        rollout_steps=16,
        device="cpu",
        checkpoint_dir=str(ckpt),
        seed=5,
        critic_only_until_gen=3,
    )
    orig_collect = trainer.collect_rollouts

    def interrupt_on_gen3(active_generation: int):
        if active_generation == 3:
            raise KeyboardInterrupt
        return orig_collect(active_generation=active_generation)

    trainer.collect_rollouts = interrupt_on_gen3
    trainer.train(total_updates=4)
    assert trainer.current_generation == 2
    trainer.close()

    resumed = ParameterSharingPPOTrainer(
        config=_quiet_cfg(),
        num_envs=1,
        rollout_steps=16,
        device="cpu",
        checkpoint_dir=str(ckpt),
        seed=5,
        critic_only_until_gen=3,
    )
    try:
        assert resumed.load_checkpoints()
        assert resumed.current_generation == 2
        actor_before_g3, adam_before_g3, critic_before_g3 = _snapshot_actor_and_adam(resumed)

        resumed.train(total_updates=1)  # gen 3 <= 3
        actor_after_g3, adam_after_g3, critic_after_g3 = _snapshot_actor_and_adam(resumed)
        assert actor_after_g3 == actor_before_g3
        assert adam_after_g3 == adam_before_g3
        assert critic_after_g3 != critic_before_g3
        assert resumed.generation_history[-1]["generation"] == 3
        assert resumed.generation_history[-1]["critic_only"] is True

        resumed.train(total_updates=1)  # gen 4 > 3
        actor_after_g4, _, _ = _snapshot_actor_and_adam(resumed)
        for role in ("blue_field", "blue_keeper", "red_field", "red_keeper"):
            assert any(
                actor_after_g4[(role, name)] != actor_after_g3[(role, name)]
                for (r, name) in actor_after_g3
                if r == role
            )
        assert resumed.generation_history[-1]["generation"] == 4
        assert resumed.generation_history[-1]["critic_only"] is False
    finally:
        resumed.close()


def test_critic_only_generations_skip_coevolution_sync_and_window(tmp_path: Path, capsys) -> None:
    """[RED] Dominating scores during critic-only generations cause no sync and never enter later windows."""
    trainer = ParameterSharingPPOTrainer(
        config=_quiet_cfg(),
        num_envs=8,
        rollout_steps=8,
        device="cpu",
        checkpoint_dir=str(tmp_path / "ckpt"),
        seed=3,
        critic_only_until_gen=3,
    )
    try:
        with torch.no_grad():
            for p in list(trainer.blue_field_policy.parameters()) + list(trainer.blue_keeper_policy.parameters()):
                p.fill_(1.0)
            for p in list(trainer.red_field_policy.parameters()) + list(trainer.red_keeper_policy.parameters()):
                p.fill_(0.0)

        orig_collect = trainer.collect_rollouts
        gen_scores = {
            1: (3, 0), 2: (3, 0), 3: (3, 0),  # 24 matches of 3-0 during warm-up
            4: (0, 0),
            5: (0, 3), 6: (0, 3),            # with gen 4: 0-48 over 24 matches -> sync Team A on gen 6
        }

        def spy_collect(active_generation: int):
            out = orig_collect(active_generation=active_generation)
            a, b = gen_scores[active_generation]
            for idx in range(8):
                trainer.match_history.append(
                    {"generation": active_generation, "score": [a, b], "match_id": active_generation * 10 + idx}
                )
            return out

        trainer.collect_rollouts = spy_collect

        trainer.train(total_updates=3)  # gens 1..3
        assert "Co-Evolution Sync" not in capsys.readouterr().out
        assert len(trainer.recent_match_scores) == 0
        # Team B actor weights remain exact 0.0 (no soft-blend occurred).
        assert float(trainer.red_field_policy.actor_mean[0].weight.abs().max()) == 0.0

        trainer.train(total_updates=1)  # gen 4
        assert list(trainer.recent_match_scores) == [(0, 0)] * 8

        trainer.train(total_updates=2)  # gens 5..6 -> triggers sync using only gens 4..6
        assert "Co-Evolution Sync" in capsys.readouterr().out
        assert len(trainer.recent_match_scores) == 0
    finally:
        trainer.close()


def test_v_corr_mc_hand_values_and_segments(tmp_path: Path) -> None:
    """[RED] Monte-Carlo returns and correlations match hand-solved values across segments and degenerate inputs."""
    from football_rl.train_ppo import compute_mc_returns, pearson_correlation

    # Single segment: rewards [1, 0, 0, 2], gamma = 0.5 -> G = [1.25, 0.5, 1.0, 2.0]
    rewards_1 = np.array([1.0, 0.0, 0.0, 2.0], dtype=np.float32)
    dones_1 = np.zeros(4, dtype=np.float32)
    g1 = compute_mc_returns(rewards_1, dones_1, gamma=0.5)
    assert np.allclose(g1, np.array([1.25, 0.5, 1.0, 2.0]))
    assert pearson_correlation(2.0 * g1 + 3.0, g1) == pytest.approx(1.0)
    assert pearson_correlation(-g1, g1) == pytest.approx(-1.0)

    # Two segments: [1, 1 | 1, 1] with done at t = 1, gamma = 0.5 -> G = [1.5, 1.0, 1.5, 1.0]
    rewards_2 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    dones_2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    g2 = compute_mc_returns(rewards_2, dones_2, gamma=0.5)
    assert np.allclose(g2, np.array([1.5, 1.0, 1.5, 1.0]))

    # Degenerate zero-variance and single-element inputs
    assert pearson_correlation(np.ones(4), g1) == 0.0
    assert pearson_correlation(np.array([1.0]), np.array([2.0])) == 0.0

    # End-to-end through update_policies with cfg.ppo.gamma = 0.5
    cfg = replace(
        DEFAULT_CONFIG,
        ppo=replace(DEFAULT_CONFIG.ppo, gamma=0.5),
        replay=replace(DEFAULT_CONFIG.replay, enable_recording=False),
    )
    trainer = ParameterSharingPPOTrainer(
        config=cfg, num_envs=1, rollout_steps=4, device="cpu", checkpoint_dir=str(tmp_path / "ckpt"), seed=3
    )
    try:
        rollout = trainer.collect_rollouts(active_generation=1)
        rollout["dones"] = np.zeros((4, 1, 12), dtype=np.float32)
        rollout["rewards"][:, 0, 0] = rewards_1
        rollout["values"][:, 0, 0] = 2.0 * g1 + 3.0
        rollout["rewards"][:, 0, 6] = rewards_1
        rollout["values"][:, 0, 6] = -g1
        metrics = trainer.update_policies(rollout)
        assert metrics["blue_keeper"]["v_corr_mc"] == pytest.approx(1.0)
        assert metrics["red_keeper"]["v_corr_mc"] == pytest.approx(-1.0)
    finally:
        trainer.close()


def test_generation_record_and_state_persist_v_corr_critic_only_and_seed(tmp_path: Path, capsys) -> None:
    """[RED] v_corr_*, critic_only, and seed persist to JSON and PT; resuming with a different seed logs once."""
    import football_rl.train_ppo as tp

    parser = tp.build_arg_parser()
    assert parser.parse_args([]).critic_only_until_gen == 0
    assert parser.parse_args(["--critic-only-until-gen", "1070"]).critic_only_until_gen == 1070

    ckpt = tmp_path / "ckpt"
    trainer = ParameterSharingPPOTrainer(
        config=_quiet_cfg(),
        num_envs=1,
        rollout_steps=16,
        device="cpu",
        checkpoint_dir=str(ckpt),
        seed=42,
        critic_only_until_gen=1,
    )
    try:
        trainer.train(total_updates=1)
    finally:
        trainer.close()

    history = json.loads((ckpt / "training_history.json").read_text(encoding="utf-8"))
    rec = history["generations"][0]
    assert rec["critic_only"] is True
    assert rec["seed"] == 42
    for key in ("v_corr_lambda", "v_corr_mc"):
        assert key in rec
        assert set(rec[key].keys()) == {"blue_field", "blue_keeper", "red_field", "red_keeper"}
        for val in rec[key].values():
            assert isinstance(val, float) and math.isfinite(val) and -1.0 <= val <= 1.0

    state = torch.load(ckpt / "training_state.pt", map_location="cpu", weights_only=False)
    assert state["seed"] == 42

    resumed = ParameterSharingPPOTrainer(
        config=_quiet_cfg(),
        num_envs=1,
        rollout_steps=16,
        device="cpu",
        checkpoint_dir=str(ckpt),
        seed=43,
        critic_only_until_gen=1,
    )
    try:
        capsys.readouterr()
        assert resumed.load_checkpoints() is True
        out = capsys.readouterr().out.strip().splitlines()
        assert len(out) == 1
        assert "42" in out[0] and "43" in out[0]
        assert resumed.seed == 43

        resumed.train(total_updates=1)
    finally:
        resumed.close()

    history2 = json.loads((ckpt / "training_history.json").read_text(encoding="utf-8"))
    rec2 = history2["generations"][1]
    assert rec2["seed"] == 43
    assert rec2["critic_only"] is False
    state2 = torch.load(ckpt / "training_state.pt", map_location="cpu", weights_only=False)
    assert state2["seed"] == 43


# --------------------------------------------------------------------------------------
# Run-config overrides and provenance
# --------------------------------------------------------------------------------------


def _override_trainer(tmp_path: Path, specs, name: str = "ckpt", num_envs: int = 1, **kwargs):
    import football_rl.train_ppo as tp

    cfg, applied = tp.parse_config_overrides(specs, _quiet_cfg(save_dir=str(tmp_path / f"rep_{name}")))
    return ParameterSharingPPOTrainer(
        config=cfg,
        num_envs=num_envs,
        rollout_steps=8,
        device="cpu",
        checkpoint_dir=str(tmp_path / name),
        seed=3,
        config_overrides=applied,
        **kwargs,
    )


def _phi_total_next(env: FootballParallelEnv) -> np.ndarray:
    w = env.cfg.rewards.team_advance_pbrs_weight
    ball = np.array(
        [w * env._compute_ball_potential(0)] * 6 + [w * env._compute_ball_potential(1)] * 6, dtype=np.float64
    )
    return ball + env._phi_curr


def test_cli_accepts_repeated_config_override_and_accept_flag() -> None:
    import football_rl.train_ppo as tp

    parser = tp.build_arg_parser()
    defaults = parser.parse_args([])
    assert defaults.config_overrides == [] and defaults.accept_config_change is False
    args = parser.parse_args(
        ["--config-override", "rewards.shaping_mode=legacy", "--config-override", "ppo.gamma=0.99",
         "--accept-config-change"]
    )
    assert args.config_overrides == ["rewards.shaping_mode=legacy", "ppo.gamma=0.99"]
    assert args.accept_config_change is True


def test_shaping_override_reaches_trainer_config_and_worker_envs(tmp_path: Path) -> None:
    """[RED] `rewards.shaping_mode=legacy` reaches the trainer and every worker env.

    Hand value: no teleport on the first step, so legacy pays Phi(s') - Phi(s) and `all` pays
    0.995 * Phi(s') - Phi(s); the difference all - legacy is -0.005 * Phi(s').
    """
    acts = np.random.default_rng(17).uniform(-1.0, 1.0, size=(12, 4)).astype(np.float32)
    t_leg = _override_trainer(tmp_path, ["rewards.shaping_mode=legacy"], "leg")
    t_all = _override_trainer(tmp_path, ["rewards.shaping_mode=all"], "all")
    try:
        assert t_leg.cfg.rewards.shaping_mode == "legacy"
        assert t_leg.vec_env.envs[0].cfg.rewards.shaping_mode == "legacy"
        assert t_all.vec_env.envs[0].cfg.rewards.shaping_mode == "all"
        t_leg.vec_env.reset_array(seeds=[17])
        t_all.vec_env.reset_array(seeds=[17])
        _, _, rew_leg, _, _ = t_leg.vec_env.step_array(acts[None])
        _, _, rew_all, _, _ = t_all.vec_env.step_array(acts[None])
        env_leg, env_all = t_leg.vec_env.envs[0], t_all.vec_env.envs[0]
        phi = _phi_total_next(env_all)
        assert np.max(np.abs(phi)) > 0.05
        np.testing.assert_allclose(
            env_all.last_shaping_rewards - env_leg.last_shaping_rewards, -0.005 * phi, rtol=0.0, atol=1e-12
        )
        assert np.max(np.abs(rew_all[0] - rew_leg[0])) > 1e-4
    finally:
        t_leg.close()
        t_all.close()

    t_sub = _override_trainer(tmp_path, ["rewards.shaping_mode=legacy"], "sub", num_envs=2)
    t_sub_all = _override_trainer(tmp_path, ["rewards.shaping_mode=all"], "suball", num_envs=2)
    try:
        for t in (t_sub, t_sub_all):
            t.vec_env.reset_array(seeds=[17, 18])
        batch = np.stack([acts, acts], axis=0)
        _, _, r_sub, _, _ = t_sub.vec_env.step_array(batch)
        _, _, r_all, _, _ = t_sub_all.vec_env.step_array(batch)
        assert np.max(np.abs(r_sub[0] - r_all[0])) > 1e-4
    finally:
        t_sub.close()
        t_sub_all.close()


def test_override_values_are_coerced_to_the_field_type() -> None:
    """[RED] float field given `0` becomes 0.0; int field stays int; str stays str; keys come back sorted."""
    import football_rl.train_ppo as tp

    cfg, applied = tp.parse_config_overrides(
        ["rewards.successful_tackle_team=0", "ppo.update_epochs=2", "rewards.shaping_mode=ball", "ppo.gamma=0.99"]
    )
    assert cfg.rewards.successful_tackle_team == 0.0 and type(cfg.rewards.successful_tackle_team) is float
    assert cfg.ppo.update_epochs == 2 and type(cfg.ppo.update_epochs) is int
    assert cfg.rewards.shaping_mode == "ball"
    assert cfg.ppo.gamma == 0.99
    assert applied == {
        "ppo.gamma": 0.99,
        "ppo.update_epochs": 2,
        "rewards.shaping_mode": "ball",
        "rewards.successful_tackle_team": 0.0,
    }
    assert list(applied) == sorted(applied)
    assert type(applied["rewards.successful_tackle_team"]) is float
    assert type(applied["ppo.update_epochs"]) is int

    base_cfg, base_applied = tp.parse_config_overrides([])
    assert base_cfg == DEFAULT_CONFIG and base_applied == {}


@pytest.mark.parametrize(
    "raw,annotation,expected",
    [("true", "bool", True), ("false", "bool", False), ("True", "bool", True), ("FALSE", "bool", False),
     ("7", "int", 7), ("-3", "int", -3), ("1e-3", "float", 0.001), ("x=y", "str", "x=y")],
)
def test_coerce_override_value_hand_values(raw, annotation, expected) -> None:
    import football_rl.train_ppo as tp

    got = tp.coerce_override_value(raw, annotation)
    assert got == expected and type(got) is type(expected)


@pytest.mark.parametrize(
    "raw,annotation", [("yes", "bool"), ("1", "bool"), ("", "bool"), ("2.5", "int"), ("", "int"),
                       ("abc", "float"), ("", "float"), ("nan", "float"), ("inf", "float"), ("1", "Tuple[int, int]")]
)
def test_coerce_override_value_rejects_unparseable_values(raw, annotation) -> None:
    import football_rl.train_ppo as tp

    with pytest.raises(ValueError):
        tp.coerce_override_value(raw, annotation)


def test_gamma_override_shifts_shaping_by_the_hand_value() -> None:
    """[RED] `ppo.gamma=0.99` vs 0.995 on one seeded step: shaping differs by 0.005 * Phi(s') per player."""
    import football_rl.train_ppo as tp

    def stepped(gamma_spec: str) -> FootballParallelEnv:
        cfg, _ = tp.parse_config_overrides(
            [gamma_spec, "rewards.shaping_mode=all"],
            replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False)),
        )
        env = FootballParallelEnv(cfg)
        env.reset(seed=17)
        env.step_array(np.random.default_rng(17).uniform(-1.0, 1.0, size=(12, 4)).astype(np.float32))
        return env

    env_99, env_995 = stepped("ppo.gamma=0.99"), stepped("ppo.gamma=0.995")
    phi = _phi_total_next(env_995)
    assert np.max(np.abs(phi)) > 0.05
    np.testing.assert_allclose(
        env_995.last_shaping_rewards - env_99.last_shaping_rewards, 0.005 * phi, rtol=0.0, atol=1e-12
    )


@pytest.mark.parametrize(
    "spec,named",
    [
        ("physics.dt=0.1", "physics"),
        ("rewards.nope=1", "rewards.nope"),
        ("ppo.gamma=abc", "ppo.gamma=abc"),
        ("rewards.shaping_mode=bogus", "bogus"),
        ("ppo.update_epochs=2.5", "ppo.update_epochs=2.5"),
        ("ppo.hidden_dims=64", "ppo.hidden_dims"),
        ("ppo.gamma", "ppo.gamma"),
        ("gamma=0.9", "gamma=0.9"),
        ("=1", "=1"),
        (".gamma=1", ".gamma=1"),
        ("rewards.=1", "rewards.=1"),
        ("replay.save_dir=/tmp/x", "replay"),
    ],
)
def test_invalid_override_is_rejected_and_names_the_item(spec: str, named: str) -> None:
    """[RED] Unknown/disallowed section, unknown key, unparseable value and failed validation each raise ValueError."""
    import football_rl.train_ppo as tp

    with pytest.raises(ValueError) as exc:
        tp.parse_config_overrides(["ppo.update_epochs=2", spec])
    assert named in str(exc.value)


def test_duplicate_override_key_is_rejected() -> None:
    import football_rl.train_ppo as tp

    with pytest.raises(ValueError, match="ppo.gamma"):
        tp.parse_config_overrides(["ppo.gamma=0.9", "ppo.gamma=0.8"])


@pytest.mark.parametrize("from_scratch", [False, True])
def test_main_rejects_bad_override_before_touching_disk(tmp_path: Path, monkeypatch, from_scratch: bool) -> None:
    """[RED] A bad override aborts `main()` before checkpoints are created or wiped (even with --from-scratch)."""
    import football_rl.train_ppo as tp

    monkeypatch.chdir(tmp_path)
    marker = tmp_path / "checkpoints" / "marker.txt"
    if from_scratch:
        marker.parent.mkdir()
        marker.write_text("keep", encoding="utf-8")
    argv = ["train_ppo", "--config-override", "rewards.nope=1", "--total-updates", "1"]
    if from_scratch:
        argv.append("--from-scratch")
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(ValueError, match="rewards.nope"):
        tp.main()
    if from_scratch:
        assert marker.read_text(encoding="utf-8") == "keep"
        assert sorted(p.name for p in (tmp_path / "checkpoints").iterdir()) == ["marker.txt"]
        assert not (tmp_path / "replays").exists()
    else:
        assert not (tmp_path / "checkpoints").exists() and not (tmp_path / "replays").exists()


def _train_one_generation(trainer: ParameterSharingPPOTrainer) -> None:
    try:
        trainer.train(total_updates=1)
    finally:
        trainer.close()


def test_overrides_persist_in_state_and_every_generation_record(tmp_path: Path) -> None:
    """[RED] The sorted dict appears in training_state.pt and each generation record; none means `{}`."""
    specs = ["rewards.shaping_mode=legacy", "ppo.update_epochs=1"]
    trainer = _override_trainer(tmp_path, specs, "ckpt")
    try:
        trainer.train(total_updates=2)
    finally:
        trainer.close()
    expected = {"ppo.update_epochs": 1, "rewards.shaping_mode": "legacy"}
    state = torch.load(tmp_path / "ckpt" / "training_state.pt", map_location="cpu", weights_only=False)
    assert state["config_overrides"] == expected and list(state["config_overrides"]) == sorted(expected)
    history = json.loads((tmp_path / "ckpt" / "training_history.json").read_text(encoding="utf-8"))
    assert len(history["generations"]) == 2
    assert all(g["config_overrides"] == expected for g in history["generations"])

    plain = _override_trainer(tmp_path, [], "plain")
    _train_one_generation(plain)
    state = torch.load(tmp_path / "plain" / "training_state.pt", map_location="cpu", weights_only=False)
    assert state["config_overrides"] == {}
    history = json.loads((tmp_path / "plain" / "training_history.json").read_text(encoding="utf-8"))
    assert history["generations"][0]["config_overrides"] == {}


def test_resume_with_different_overrides_raises_and_names_both_sides(tmp_path: Path) -> None:
    """[RED] Stored vs requested overrides must match on resume."""
    first = _override_trainer(tmp_path, ["ppo.update_epochs=1"], "ckpt")
    _train_one_generation(first)

    for specs in (["ppo.update_epochs=2"], [], ["ppo.update_epochs=1", "rewards.shaping_mode=legacy"]):
        resumed = _override_trainer(tmp_path, specs, "ckpt")
        try:
            with pytest.raises(ValueError) as exc:
                resumed.load_checkpoints()
            msg = str(exc.value)
            assert "ppo.update_epochs" in msg and "stored" in msg and "requested" in msg
            assert resumed.current_generation == 0
        finally:
            resumed.close()

    same = _override_trainer(tmp_path, ["ppo.update_epochs=1"], "ckpt")
    try:
        assert same.load_checkpoints() is True
        assert same.current_generation == 1
    finally:
        same.close()


def test_accept_config_change_prints_one_line_and_proceeds(tmp_path: Path, capsys) -> None:
    """[RED] `--accept-config-change` downgrades the mismatch to one printed line; the new dict is then stored."""
    first = _override_trainer(tmp_path, ["ppo.update_epochs=1"], "ckpt")
    _train_one_generation(first)

    resumed = _override_trainer(tmp_path, ["ppo.update_epochs=2"], "ckpt", accept_config_change=True)
    try:
        capsys.readouterr()
        assert resumed.load_checkpoints() is True
        out = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
        assert len(out) == 1 and "ppo.update_epochs" in out[0]
        assert resumed.current_generation == 1
        resumed.train(total_updates=1)
    finally:
        resumed.close()
    state = torch.load(tmp_path / "ckpt" / "training_state.pt", map_location="cpu", weights_only=False)
    assert state["config_overrides"] == {"ppo.update_epochs": 2}
    history = json.loads((tmp_path / "ckpt" / "training_history.json").read_text(encoding="utf-8"))
    assert [g["config_overrides"] for g in history["generations"]] == [{"ppo.update_epochs": 1}, {"ppo.update_epochs": 2}]


def test_state_without_overrides_key_resumes_with_no_overrides(tmp_path: Path, capsys) -> None:
    """[RED] A state saved before the key existed reads as `{}`: resumes silently with no overrides, raises with some."""
    first = _override_trainer(tmp_path, [], "ckpt")
    _train_one_generation(first)
    state_path = tmp_path / "ckpt" / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    del state["config_overrides"]
    torch.save(state, state_path)

    resumed = _override_trainer(tmp_path, [], "ckpt")
    try:
        capsys.readouterr()
        assert resumed.load_checkpoints() is True
        assert capsys.readouterr().out.strip() == ""
        assert resumed.current_generation == 1
    finally:
        resumed.close()

    mismatched = _override_trainer(tmp_path, ["ppo.update_epochs=1"], "ckpt")
    try:
        with pytest.raises(ValueError, match="ppo.update_epochs"):
            mismatched.load_checkpoints()
    finally:
        mismatched.close()


def _fake_git(monkeypatch, rev, status, calls=None):
    import football_rl.train_ppo as tp

    def fake(args):
        if calls is not None:
            calls.append(tuple(args))
        if tuple(args) == ("rev-parse", "--short", "HEAD"):
            return rev
        if tuple(args) == ("status", "--porcelain"):
            return status
        raise AssertionError(f"unexpected git invocation {args}")

    monkeypatch.setattr(tp, "_run_git", fake)
    return tp


@pytest.mark.parametrize(
    "rev,status,expected",
    [("abc1234\n", "", "abc1234"), ("abc1234\n", " M football_rl/env.py\n", "abc1234-dirty"),
     ("abc1234\n", "?? new.txt\n", "abc1234-dirty"), (None, None, "unknown"), (None, "", "unknown"),
     ("", "", "unknown"), ("abc1234\n", None, "unknown")],
)
def test_current_git_commit_clean_dirty_and_unavailable(monkeypatch, rev, status, expected) -> None:
    """[RED] `git rev-parse --short HEAD` plus `-dirty` when `git status --porcelain` is non-empty; else `unknown`."""
    tp = _fake_git(monkeypatch, rev, status)
    assert tp.current_git_commit() == expected


def test_current_git_commit_real_process_returns_a_short_hash_or_unknown() -> None:
    import football_rl.train_ppo as tp

    assert re.fullmatch(r"([0-9a-f]{7,}(-dirty)?|unknown)", tp.current_git_commit())


def test_git_commit_in_state_and_every_generation_record_with_one_call_per_process(tmp_path: Path, monkeypatch) -> None:
    """[RED] `git_commit` is stored in both files; git runs once at trainer construction, not per generation."""
    calls: list = []
    _fake_git(monkeypatch, "abc1234\n", " M x.py\n", calls)
    trainer = _override_trainer(tmp_path, [], "ckpt")
    assert len(calls) == 2
    try:
        trainer.train(total_updates=2)
    finally:
        trainer.close()
    assert len(calls) == 2
    state = torch.load(tmp_path / "ckpt" / "training_state.pt", map_location="cpu", weights_only=False)
    assert state["git_commit"] == "abc1234-dirty"
    history = json.loads((tmp_path / "ckpt" / "training_history.json").read_text(encoding="utf-8"))
    assert [g["git_commit"] for g in history["generations"]] == ["abc1234-dirty"] * 2

    _fake_git(monkeypatch, None, None)
    unavailable = _override_trainer(tmp_path, [], "ckpt2")
    _train_one_generation(unavailable)
    state = torch.load(tmp_path / "ckpt2" / "training_state.pt", map_location="cpu", weights_only=False)
    assert state["git_commit"] == "unknown"


def test_main_applies_overrides_stores_them_and_enforces_them_on_resume(tmp_path: Path, monkeypatch) -> None:
    """[RED] End to end through `main()`: override reaches the stored state, and resume without it is refused."""
    import football_rl.train_ppo as tp

    monkeypatch.chdir(tmp_path)
    base = ["train_ppo", "--num-envs", "1", "--rollout-steps", "8", "--total-updates", "1", "--device", "cpu"]
    monkeypatch.setattr("sys.argv", base + ["--config-override", "rewards.shaping_mode=legacy"])
    tp.main()
    state = torch.load(tmp_path / "checkpoints" / "training_state.pt", map_location="cpu", weights_only=False)
    assert state["config_overrides"] == {"rewards.shaping_mode": "legacy"}
    assert state["current_generation"] == 1

    monkeypatch.setattr("sys.argv", base)
    with pytest.raises(ValueError, match="rewards.shaping_mode"):
        tp.main()
    state = torch.load(tmp_path / "checkpoints" / "training_state.pt", map_location="cpu", weights_only=False)
    assert state["current_generation"] == 1

    monkeypatch.setattr("sys.argv", base + ["--accept-config-change"])
    tp.main()
    state = torch.load(tmp_path / "checkpoints" / "training_state.pt", map_location="cpu", weights_only=False)
    assert state["config_overrides"] == {} and state["current_generation"] == 2


# ==============================================================================
# Decoupled action heads (`ppo.policy_head = "hybrid5"`)
# ==============================================================================
from football_rl import train_ppo as tp  # noqa: E402

# z with floor 0 and logit bound 6 such that P(fire) = 0.25 exactly: 6 * atanh(ln(1/3) / 6).
Z_FOR_QUARTER = -1.1111428236846903
GAUSS = -0.9189385332046727  # ln N(0 | 0, 1)
ROLE_KEYS = ("blue_field", "blue_keeper", "red_field", "red_keeper")


def _hybrid_cfg(recording: bool = False, save_dir: str = "replays", max_rl_steps: int | None = None, **ppo):
    cfg = _quiet_cfg(save_dir=save_dir, recording=recording)
    cfg = replace(cfg, ppo=replace(cfg.ppo, policy_head="hybrid5", **ppo))
    if max_rl_steps is not None:
        cfg = replace(cfg, physics=replace(cfg.physics, max_rl_steps=max_rl_steps))
    return cfg


def _head_trainer(tmp_path: Path, head: str = "hybrid5", name: str = "ckpt", num_envs: int = 1,
                  rollout_steps: int = 8, seed: int = 3, **ppo) -> ParameterSharingPPOTrainer:
    cfg = _quiet_cfg(save_dir=str(tmp_path / f"rep_{name}"))
    cfg = replace(cfg, ppo=replace(cfg.ppo, policy_head=head, **ppo))
    return ParameterSharingPPOTrainer(
        config=cfg, num_envs=num_envs, rollout_steps=rollout_steps, device="cpu",
        checkpoint_dir=str(tmp_path / name), seed=seed,
    )


def _constant_hybrid_net(bias, logstd=(0.0, 0.0, 0.0), floor: float = 0.0, bound: float = 6.0) -> ActorCritic:
    """Hybrid network whose outputs ignore the observation: final-layer weights are zero, biases are `bias`."""
    net = ActorCritic(policy_head="hybrid5", fire_prob_floor=floor, fire_logit_bound=bound)
    with torch.no_grad():
        net.actor_mean[4].weight.zero_()
        net.actor_mean[4].bias.copy_(torch.tensor(bias, dtype=torch.float32))
        net.actor_logstd.copy_(torch.tensor([list(logstd)], dtype=torch.float32))
    return net


# ------------------------------------------------------------------ configuration
def test_policy_head_config_defaults_are_todays_policy_and_literals() -> None:
    """[GUARD] The defaults select the existing Gaussian policy and today's clamp literals."""
    ppo = DEFAULT_CONFIG.ppo
    assert (ppo.policy_head, ppo.logstd_min, ppo.logstd_max) == ("gaussian4", -1.25, -0.35)
    assert (ppo.fire_prob_floor, ppo.fire_logit_bound) == (0.005, 6.0)


@pytest.mark.parametrize(
    "changes",
    [
        {"policy_head": "hybrid6"},
        {"policy_head": ""},
        {"logstd_min": -0.35},                       # equal bounds
        {"logstd_min": -0.3},                        # inverted bounds
        {"logstd_min": float("nan")},
        {"logstd_max": float("inf")},
        {"logstd_min": float("-inf")},
        {"fire_prob_floor": -0.001},
        {"fire_prob_floor": 0.5},
        {"fire_prob_floor": float("nan")},
        {"fire_logit_bound": 0.0},
        {"fire_logit_bound": -6.0},
        {"fire_logit_bound": float("inf")},
        {"fire_logit_bound": float("nan")},
    ],
)
def test_policy_head_config_rejects_invalid_values(changes) -> None:
    """[RED] Each invalid head setting is refused by the dataclass."""
    with pytest.raises(ValueError):
        replace(DEFAULT_CONFIG.ppo, **changes)


def test_policy_head_config_accepts_boundary_values() -> None:
    """[RED] Floor 0, floor just under 0.5, a tiny positive bound and a tight but ordered logstd range are valid."""
    for changes in (
        {"fire_prob_floor": 0.0},
        {"fire_prob_floor": 0.4999},
        {"fire_logit_bound": 1e-3},
        {"logstd_min": -1.0, "logstd_max": -0.999},
        {"policy_head": "hybrid5"},
    ):
        replace(DEFAULT_CONFIG.ppo, **changes)


def test_head_settings_are_reachable_through_config_overrides() -> None:
    """[RED] `--config-override ppo.<key>=<value>` sets every head field with the field's type."""
    cfg, applied = tp.parse_config_overrides(
        [
            "ppo.policy_head=hybrid5", "ppo.logstd_min=-1.6", "ppo.logstd_max=-0.9",
            "ppo.fire_prob_floor=0.01", "ppo.fire_logit_bound=8",
        ]
    )
    ppo = cfg.ppo
    assert (ppo.policy_head, ppo.logstd_min, ppo.logstd_max, ppo.fire_prob_floor, ppo.fire_logit_bound) == (
        "hybrid5", -1.6, -0.9, 0.01, 8.0,
    )
    assert isinstance(ppo.fire_logit_bound, float) and applied["ppo.fire_logit_bound"] == 8.0
    with pytest.raises(ValueError, match="ppo.policy_head=bogus"):
        tp.parse_config_overrides(["ppo.policy_head=bogus"])
    with pytest.raises(ValueError, match="logstd_min"):
        tp.parse_config_overrides(["ppo.logstd_min=-0.2"])


# ------------------------------------------------------------------ env-action decode
def test_to_env_actions_hand_values() -> None:
    """[RED] Kick = thr + (1 - thr) * max(power, 0.001) when K = 1 else -1; special = +1 / -1; thr = 0.35."""
    # Columns: throttle, steer, u, K, S.
    pol = np.array(
        [
            [0.2, -0.3, 0.0, 1.0, 1.0],     # power 0.5  -> kick 0.675, special +1
            [0.0, 0.0, 1.0, 1.0, 0.0],      # power 1.0  -> kick 1.0
            [0.0, 0.0, -1.0, 1.0, 0.0],     # power 0    -> floor 0.001 -> 0.35065
            [0.5, 0.5, 1.0, 0.0, 0.0],      # no kick for any u
            [0.5, 0.5, -1.0, 0.0, 1.0],
            [0.5, 0.5, 7.0, 1.0, 0.0],      # u far above range -> power clipped to 1
            [0.5, 0.5, -7.0, 1.0, 0.0],     # u far below range -> power clipped to 0
        ],
        dtype=np.float32,
    )
    env = tp.to_env_actions(pol, 0.35)
    assert env.shape == (7, 4) and env.dtype == np.float32
    np.testing.assert_allclose(env[:, 2], [0.675, 1.0, 0.35065, -1.0, -1.0, 1.0, 0.35065], atol=1e-6)
    np.testing.assert_allclose(env[:, 3], [1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0])
    np.testing.assert_array_equal(env[:, :2], pol[:, :2])


def test_to_env_actions_every_fired_kick_exceeds_the_env_threshold() -> None:
    """[RED] For any u, K = 1 gives a value strictly above 0.35 and K = 0 gives exactly -1."""
    u = np.linspace(-5.0, 5.0, 101, dtype=np.float32)
    fired = np.stack([np.zeros_like(u), np.zeros_like(u), u, np.ones_like(u), np.zeros_like(u)], axis=1)
    silent = fired.copy()
    silent[:, 3] = 0.0
    assert (tp.to_env_actions(fired, 0.35)[:, 2] > 0.35).all()
    assert (tp.to_env_actions(silent, 0.35)[:, 2] == -1.0).all()
    assert (tp.to_env_actions(fired, 0.35)[:, 2] <= 1.0).all()


def test_to_env_actions_contract_edges() -> None:
    """[RED] Leading dimensions survive, inputs are not mutated, throttle/steer are untouched, threshold is a parameter."""
    pol = np.zeros((2, 12, 5), dtype=np.float32)
    pol[..., 0] = 1.7            # outside [-1, 1]: the env clips, the decode leaves it alone
    pol[..., 1] = -2.5
    pol[..., 3] = 1.0
    before = pol.copy()
    env = tp.to_env_actions(pol, 0.35)
    assert env.shape == (2, 12, 4)
    np.testing.assert_array_equal(pol, before)
    assert (env[..., 0] == np.float32(1.7)).all() and (env[..., 1] == np.float32(-2.5)).all()
    # Threshold 0.5, power 0.5: 0.5 + 0.5 * 0.5 = 0.75.
    assert tp.to_env_actions(pol[:1, :1], 0.5)[0, 0, 2] == pytest.approx(0.75)
    with pytest.raises(ValueError):
        tp.to_env_actions(np.zeros((3, 4), dtype=np.float32), 0.35)
    assert tp.to_env_actions(np.zeros((0, 5), dtype=np.float32), 0.35).shape == (0, 4)


# ------------------------------------------------------------------ fire probability
def test_fire_probability_bounds_and_midpoint_hand_values() -> None:
    """[RED] Floor 0.005, L = 6: z -> -inf gives 0.005 + 0.99 * sigmoid(-6) = 0.007448; +inf gives 0.992552; 0 gives 0.5."""
    p = tp.fire_probability(torch.tensor([-1e6, 1e6, 0.0], dtype=torch.float64), 0.005, 6.0)
    np.testing.assert_allclose(p.numpy(), [0.007448, 0.992552, 0.5], atol=1e-5)
    # Floor 0: sigmoid(+-6) = 0.0024726 / 0.9975274.
    p0 = tp.fire_probability(torch.tensor([-1e6, 1e6], dtype=torch.float64), 0.0, 6.0)
    np.testing.assert_allclose(p0.numpy(), [0.0024726, 0.9975274], atol=1e-6)
    # Floor 0 and z solved for 0.25.
    assert float(tp.fire_probability(torch.tensor([Z_FOR_QUARTER], dtype=torch.float64), 0.0, 6.0)) == pytest.approx(
        0.25, abs=1e-9
    )


def test_fire_probability_is_monotone_symmetric_and_finite_at_extremes() -> None:
    """[RED] Strictly increasing on a grid, p(z) + p(-z) = 1, finite and inside the bounds for float32 extremes."""
    z = torch.linspace(-30.0, 30.0, 121, dtype=torch.float64)
    p = tp.fire_probability(z, 0.005, 6.0)
    assert (p[1:] > p[:-1]).all()
    np.testing.assert_allclose((p + tp.fire_probability(-z, 0.005, 6.0)).numpy(), 1.0, atol=1e-12)
    extreme = tp.fire_probability(torch.tensor([-3e38, -1e30, 1e30, 3e38], dtype=torch.float32), 0.005, 6.0)
    assert torch.isfinite(extreme).all()
    assert (extreme >= 0.00744).all() and (extreme <= 0.99256).all()


# ------------------------------------------------------------------ log-probability and entropy
def _hand_inputs(k: float, s: float, u: float = 0.0):
    mean = torch.zeros(1, 3, dtype=torch.float64)
    logstd = torch.zeros(1, 3, dtype=torch.float64)           # sigma = 1 on all three Gaussian heads
    p_kick = torch.tensor([0.25], dtype=torch.float64)
    p_spec = torch.tensor([0.5], dtype=torch.float64)
    action = torch.tensor([[0.0, 0.0, u, k, s]], dtype=torch.float64)
    return mean, logstd, p_kick, p_spec, action


def test_hybrid_log_prob_and_entropy_hand_values() -> None:
    """[RED] K=1,S=0: 3*ln N(0) + ln 0.25 + ln 0.5 = -4.836257. K=0,S=0: 2*ln N(0) + ln 0.75 + ln 0.5 = -2.818706.
    Entropy: 2*1.418939 + H_B(0.25) + H_B(0.5) + 0.25 * 1.418939 = 4.448094."""
    lp1, ent = tp.hybrid_log_prob_entropy(*_hand_inputs(1.0, 0.0))
    lp0, ent0 = tp.hybrid_log_prob_entropy(*_hand_inputs(0.0, 0.0))
    assert float(lp1) == pytest.approx(-4.836257, abs=1e-5)
    assert float(lp0) == pytest.approx(-2.818706, abs=1e-5)
    assert float(ent) == pytest.approx(4.448094, abs=1e-5)
    assert float(ent0) == pytest.approx(4.448094, abs=1e-5)       # entropy does not depend on the sampled K


def test_hybrid_power_term_counts_only_when_a_kick_fired() -> None:
    """[RED] u = +1 adds -0.5 to ln N when K = 1 (-5.336257) and changes nothing when K = 0."""
    assert float(tp.hybrid_log_prob_entropy(*_hand_inputs(1.0, 0.0, u=1.0))[0]) == pytest.approx(-5.336257, abs=1e-5)
    for u in (-3.0, 0.0, 2.5):
        assert float(tp.hybrid_log_prob_entropy(*_hand_inputs(0.0, 0.0, u=u))[0]) == pytest.approx(-2.818706, abs=1e-5)
    # Special fired: ln 0.5 either way; S=1,K=0 equals S=0,K=0.
    assert float(tp.hybrid_log_prob_entropy(*_hand_inputs(0.0, 1.0))[0]) == pytest.approx(-2.818706, abs=1e-5)


def test_network_log_prob_and_entropy_match_the_hand_values() -> None:
    """[RED] A network with p_kick = 0.25, p_spec = 0.5, sigma = 1 and zero means reproduces the hand values from raw actions."""
    net = _constant_hybrid_net([0.0, 0.0, 0.0, Z_FOR_QUARTER, 0.0])
    obs = torch.randn(3, OBS_DIM)
    fire = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]] * 3)
    silent = torch.zeros(3, 5)
    _, lp1, ent, _ = net.get_action_and_value(obs, fire)
    _, lp0, _, _ = net.get_action_and_value(obs, silent)
    np.testing.assert_allclose(lp1.detach().numpy(), -4.836257, atol=1e-5)
    np.testing.assert_allclose(lp0.detach().numpy(), -2.818706, atol=1e-5)
    np.testing.assert_allclose(ent.detach().numpy(), 4.448094, atol=1e-5)


def test_stored_log_prob_equals_the_update_recomputation_from_the_five_float_action() -> None:
    """[RED] Sampling log-probs match `get_action_and_value` on the stored action, for every row and both K values."""
    torch.manual_seed(0)
    net = ActorCritic(policy_head="hybrid5")
    with torch.no_grad():
        net.actor_mean[4].weight.normal_(0.0, 0.5)      # make p_kick / p_spec / means depend on the observation
    obs = torch.randn(512, OBS_DIM)
    action, logp, _ = net.sample_action_and_value(obs)
    assert action.shape == (512, 5)
    assert 0 < int(action[:, 3].sum()) < 512             # both K = 0 and K = 1 occur
    _, recomputed, _, _ = net.get_action_and_value(obs, action)
    torch.testing.assert_close(recomputed, logp, rtol=0.0, atol=1e-5)


def test_sampling_log_prob_of_unfired_kicks_does_not_depend_on_the_power_sample() -> None:
    """[RED] With constant heads, every K = 0 row has the same log-prob although u differs row to row."""
    net = _constant_hybrid_net([0.0, 0.0, 0.0, Z_FOR_QUARTER, 0.0])
    torch.manual_seed(1)
    action, logp, _ = net.sample_action_and_value(torch.zeros(2000, OBS_DIM))
    no_kick = action[:, 3] == 0
    assert 1000 < int(no_kick.sum()) < 1900
    assert float(action[no_kick, 2].std()) > 0.5                 # u really varies
    # S=0: ln0.5 ; S=1: ln0.5 ; K=0: ln0.75 ; two Gaussian heads at the sampled values.
    thr_steer = -0.5 * (action[:, 0] ** 2 + action[:, 1] ** 2) + 2 * GAUSS
    expected = thr_steer + math.log(0.75) + math.log(0.5)
    torch.testing.assert_close(logp[no_kick], expected[no_kick], rtol=0.0, atol=1e-5)


def test_deterministic_action_uses_the_means_and_fires_above_one_half() -> None:
    """[RED] Deterministic mode returns [tanh(z) x3, p_kick > 0.5, p_spec > 0.5]."""
    net = _constant_hybrid_net([0.5, -0.5, 0.25, 2.0, -2.0], floor=0.005)
    action, _, _, _ = net.get_action_and_value(torch.zeros(2, OBS_DIM), deterministic=True)
    np.testing.assert_allclose(action.detach().numpy(), [[math.tanh(0.5), math.tanh(-0.5), math.tanh(0.25), 1.0, 0.0]] * 2, atol=1e-6)
    sampled, _, _ = net.sample_action_and_value(torch.zeros(2, OBS_DIM), deterministic=True)
    torch.testing.assert_close(sampled, action)


def test_hybrid_outputs_are_finite_for_extreme_observations() -> None:
    """[RED] Observations of +-1e6 saturate the heads; log-prob, entropy and the bounded probabilities stay finite."""
    torch.manual_seed(2)
    net = ActorCritic(policy_head="hybrid5")
    obs = torch.cat([torch.full((4, OBS_DIM), 1e6), torch.full((4, OBS_DIM), -1e6), torch.zeros(1, OBS_DIM)])
    action, logp, value = net.sample_action_and_value(obs)
    _, lp2, ent, _ = net.get_action_and_value(obs, action)
    for t in (action, logp, value, lp2, ent):
        assert torch.isfinite(t).all()
    _, p_kick, p_spec = net.hybrid_heads(obs)
    assert ((p_kick >= 0.00744) & (p_kick <= 0.99256) & (p_spec >= 0.00744) & (p_spec <= 0.99256)).all()


# ------------------------------------------------------------------ initialisation
def test_hybrid_initial_rates_and_biases_reproduce_todays_policy() -> None:
    """[RED] Zero observation: P(kick) 0.1289, P(special) 0.0275, power mean 0.4666, throttle bias 0.35, steer bias 0."""
    torch.manual_seed(0)
    net = ActorCritic(policy_head="hybrid5")
    mean3, p_kick, p_spec = net.hybrid_heads(torch.zeros(1, OBS_DIM))
    assert float(p_kick) == pytest.approx(0.129, abs=1e-3)
    assert float(p_spec) == pytest.approx(0.0275, abs=5e-4)
    assert float((1.0 + mean3[0, 2]) / 2.0) == pytest.approx(0.4666, abs=1e-3)
    bias = net.actor_mean[4].bias
    assert float(bias[0]) == pytest.approx(0.35) and float(bias[1]) == 0.0 and float(bias[2]) == pytest.approx(-0.067)
    assert net.actor_mean[4].weight.shape == (5, 256)
    np.testing.assert_allclose(torch.linalg.norm(net.actor_mean[4].weight, dim=1).detach().numpy(), 0.01, atol=1e-5)
    assert net.actor_logstd.shape == (1, 3) and bool((net.actor_logstd == -0.5).all())


def test_hybrid_sampled_initial_fire_rates_match_the_analytic_values() -> None:
    """[RED] 200k seeded draws at a zero observation fire kicks 12.9% and specials 2.75% of the time."""
    torch.manual_seed(5)
    net = ActorCritic(policy_head="hybrid5")
    action, _, _ = net.sample_action_and_value(torch.zeros(200_000, OBS_DIM))
    assert float(action[:, 3].mean()) == pytest.approx(0.1289, abs=0.004)
    assert float(action[:, 4].mean()) == pytest.approx(0.0275, abs=0.002)


@pytest.mark.parametrize("floor,bound", [(0.0, 4.0), (0.0, 6.0), (0.0, 12.0), (0.01, 6.0), (0.01, 12.0)])
def test_hybrid_initial_rates_hold_for_other_floors_and_bounds(floor: float, bound: float) -> None:
    """[RED] The biases are solved from the configured floor and bound, so the initial rates do not move."""
    net = ActorCritic(policy_head="hybrid5", fire_prob_floor=floor, fire_logit_bound=bound)
    _, p_kick, p_spec = net.hybrid_heads(torch.zeros(1, OBS_DIM))
    assert float(p_kick) == pytest.approx(0.1289, abs=1e-3)
    assert float(p_spec) == pytest.approx(0.0275, abs=5e-4)


def test_hybrid_rejects_settings_that_cannot_reproduce_the_initial_rates() -> None:
    """[RED] A floor at or above today's initial special rate (0.0275), or a logit bound whose smallest probability
    exceeds it (floor 0 and bound 3: sigmoid(-3) = 0.047; floor 0.01 and bound 4: 0.01 + 0.98 * sigmoid(-4) = 0.0277),
    cannot start at the gaussian4 rates and is refused with the field named."""
    with pytest.raises(ValueError, match="fire_prob_floor"):
        ActorCritic(policy_head="hybrid5", fire_prob_floor=0.03)
    with pytest.raises(ValueError, match="fire_logit_bound"):
        ActorCritic(policy_head="hybrid5", fire_logit_bound=3.0)
    with pytest.raises(ValueError, match="fire_logit_bound"):
        ActorCritic(policy_head="hybrid5", fire_prob_floor=0.01, fire_logit_bound=4.0)


@pytest.mark.parametrize(
    "overrides, field",
    [
        (["ppo.fire_prob_floor=0.05"], "fire_prob_floor"),
        (["ppo.fire_prob_floor=0.03"], "fire_prob_floor"),
        (["ppo.fire_logit_bound=3"], "fire_logit_bound"),
        (["ppo.fire_prob_floor=0.01", "ppo.fire_logit_bound=4"], "fire_logit_bound"),
    ],
)
def test_main_rejects_unreachable_fire_priors_before_wiping_the_batch(
    tmp_path: Path, monkeypatch, overrides, field
) -> None:
    """[RED] A hybrid5 floor / bound pair that cannot reach the initial fire rates aborts `main(--from-scratch)`
    with the field named, and the active `checkpoints/` marker file survives."""
    import football_rl.train_ppo as tp

    monkeypatch.chdir(tmp_path)
    marker = tmp_path / "checkpoints" / "marker.txt"
    marker.parent.mkdir()
    marker.write_text("keep", encoding="utf-8")
    argv = ["train_ppo", "--from-scratch", "--total-updates", "1", "--config-override", "ppo.policy_head=hybrid5"]
    for item in overrides:
        argv += ["--config-override", item]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(ValueError, match=field):
        tp.main()
    assert marker.read_text(encoding="utf-8") == "keep"
    assert sorted(p.name for p in (tmp_path / "checkpoints").iterdir()) == ["marker.txt"]
    assert not (tmp_path / "replays").exists()


@pytest.mark.parametrize(
    "changes, field",
    [
        ({"fire_prob_floor": 0.0275}, "fire_prob_floor"),  # equals the initial special rate: no room
        ({"fire_prob_floor": 0.4999}, "fire_prob_floor"),
        ({"fire_logit_bound": 3.0}, "fire_logit_bound"),
        ({"fire_prob_floor": 0.01, "fire_logit_bound": 4.0}, "fire_logit_bound"),
    ],
)
def test_hybrid_config_rejects_unreachable_fire_priors(changes, field) -> None:
    """[RED] `PPOHyperparams` refuses an unreachable hybrid5 pair itself; the same values under gaussian4 stay valid."""
    with pytest.raises(ValueError, match=field):
        replace(DEFAULT_CONFIG.ppo, policy_head="hybrid5", **changes)
    replace(DEFAULT_CONFIG.ppo, policy_head="gaussian4", **changes)


def test_hybrid_config_accepts_the_reachable_edge_of_the_fire_priors() -> None:
    """[GUARD] Floor 0.02 with the default bound, and floor 0 with bound 4 (sigmoid(-4) = 0.018 < 0.0275), are valid."""
    replace(DEFAULT_CONFIG.ppo, policy_head="hybrid5", fire_prob_floor=0.02)
    replace(DEFAULT_CONFIG.ppo, policy_head="hybrid5", fire_prob_floor=0.0, fire_logit_bound=4.0)


def test_state_dict_layouts() -> None:
    """[GUARD] gaussian4 keeps today's keys and shapes; hybrid5 adds its two bounds as buffers and a 5-wide head."""
    g = ActorCritic()
    assert set(g.state_dict()) == {
        "actor_logstd",
        *(f"actor_mean.{i}.{p}" for i in (0, 2, 4) for p in ("weight", "bias")),
        *(f"critic.{i}.{p}" for i in (0, 2, 4) for p in ("weight", "bias")),
    }
    assert g.actor_logstd.shape == (1, 4) and g.actor_mean[4].weight.shape == (4, 256)
    h = ActorCritic(policy_head="hybrid5", fire_prob_floor=0.01, fire_logit_bound=8.0)
    assert set(h.state_dict()) - set(g.state_dict()) == {"fire_prob_floor", "fire_logit_bound"}
    assert float(h.state_dict()["fire_prob_floor"]) == pytest.approx(0.01)
    assert float(h.state_dict()["fire_logit_bound"]) == pytest.approx(8.0)
    assert h.actor_logstd.shape == (1, 3) and h.actor_mean[4].weight.shape == (5, 256)
    assert tp.policy_head_of_weights(g.state_dict()) == "gaussian4"
    assert tp.policy_head_of_weights(h.state_dict()) == "hybrid5"
    assert tp.policy_head_of_weights({"critic.0.weight": torch.zeros(1, 1)}) is None


# ------------------------------------------------------------------ gradients
def _surrogate_grads(net: ActorCritic, action: torch.Tensor, ent_coef: float):
    obs = torch.randn(32, OBS_DIM)
    adv = torch.randn(32)
    _, logp, entropy, _ = net.get_action_and_value(obs, action)
    loss = -(adv * logp).mean() - ent_coef * entropy.mean()
    net.zero_grad()
    loss.backward()


def test_power_head_receives_no_gradient_from_a_batch_without_kicks() -> None:
    """[RED] With K = 0 everywhere the power row, bias and sigma get exactly zero policy-gradient; the other heads do not."""
    torch.manual_seed(4)
    net = ActorCritic(policy_head="hybrid5")
    with torch.no_grad():
        net.actor_mean[4].weight.normal_(0.0, 0.3)
    action = torch.randn(32, 5)
    action[:, 3] = 0.0
    action[:, 4] = torch.bernoulli(torch.full((32,), 0.5))
    _surrogate_grads(net, action, ent_coef=0.0)
    w, b = net.actor_mean[4].weight.grad, net.actor_mean[4].bias.grad
    assert not w[2].any() and float(b[2]) == 0.0 and float(net.actor_logstd.grad[0, 2]) == 0.0
    for row in (0, 1, 3, 4):
        assert w[row].abs().max() > 0 and float(b[row]) != 0.0
    assert float(net.actor_logstd.grad[0, 0]) != 0.0 and float(net.actor_logstd.grad[0, 1]) != 0.0
    # With any kick present the power head learns.
    action[0, 3] = 1.0
    _surrogate_grads(net, action, ent_coef=0.0)
    assert net.actor_mean[4].weight.grad[2].abs().max() > 0 and float(net.actor_logstd.grad[0, 2]) != 0.0


def test_entropy_bonus_reaches_the_power_sigma_only_through_p_kick() -> None:
    """[RED] d(-ent_coef * p_kick * H_N)/d logstd_power = -ent_coef * p_kick = -0.005 * 0.25; the power mean row stays at zero."""
    net = _constant_hybrid_net([0.0, 0.0, 0.0, Z_FOR_QUARTER, 0.0])
    action = torch.zeros(32, 5)
    _surrogate_grads(net, action, ent_coef=0.005)
    assert float(net.actor_logstd.grad[0, 2]) == pytest.approx(-0.005 * 0.25, abs=1e-7)
    assert not net.actor_mean[4].weight.grad[2].any() and float(net.actor_mean[4].bias.grad[2]) == 0.0


# ------------------------------------------------------------------ one-state bandit
BANDIT_UPDATE_CALLS = 20


def _bandit_policy(trainer: ParameterSharingPPOTrainer):
    obs = torch.linspace(-1.0, 1.0, OBS_DIM).unsqueeze(0)
    return trainer.blue_field_policy, trainer.blue_field_optimizer, obs


def _p_kick(policy: ActorCritic, obs: torch.Tensor) -> float:
    with torch.no_grad():
        return float(policy.hybrid_heads(obs)[1])


def _bandit_calls(trainer, policy, opt, obs, sign: float, calls: int, stop) -> int:
    """Run update calls of 512 sampled decisions each; reward = sign when K = 1 else 0. Return the calls used."""
    batch = obs.repeat(512, 1)
    for used in range(1, calls + 1):
        with torch.no_grad():
            action, logp, value = policy.sample_action_and_value(batch)
        reward = sign * action[:, 3]
        trainer._ppo_update_role(policy, opt, batch, action, logp, reward - reward.mean(), reward, value)
        if stop(_p_kick(policy, obs)):
            return used
    return calls


def test_hybrid_bandit_learns_to_fire_and_to_stop_firing(tmp_path: Path) -> None:
    """[RED] Reward +1 for K=1: P(kick) > 0.9 inside the budget. Then reward -1 for K=1: P(kick) < 0.05 inside the same budget."""
    torch.manual_seed(0)
    trainer = _head_trainer(tmp_path)
    try:
        policy, opt, obs = _bandit_policy(trainer)
        up = _bandit_calls(trainer, policy, opt, obs, +1.0, BANDIT_UPDATE_CALLS, lambda p: p > 0.9)
        assert _p_kick(policy, obs) > 0.9, f"P(kick)={_p_kick(policy, obs):.3f} after {up} calls"
        down = _bandit_calls(trainer, policy, opt, obs, -1.0, BANDIT_UPDATE_CALLS, lambda p: p < 0.05)
        assert _p_kick(policy, obs) < 0.05, f"P(kick)={_p_kick(policy, obs):.3f} after {down} calls"
        print(f"bandit update calls needed: rise {up}, fall {down}, budget {BANDIT_UPDATE_CALLS}")
    finally:
        trainer.close()


# ------------------------------------------------------------------ rollout wiring
def test_gaussian4_trainer_keeps_four_wide_actions(tmp_path: Path) -> None:
    """[GUARD] The default trainer stores and returns 4-wide actions."""
    trainer = _head_trainer(tmp_path, head="gaussian4", num_envs=2)
    try:
        acts, lps, vals, means = trainer._select_actions(
            trainer.current_obs, trainer.current_central_state, with_means=True
        )
        assert acts.shape == (2, 12, 4) and means.shape == (2, 12, 4) and lps.shape == vals.shape == (2, 12)
    finally:
        trainer.close()


def test_hybrid_select_actions_shapes_and_binary_fires(tmp_path: Path) -> None:
    """[RED] Policy actions are 5 wide with K and S in {0, 1}; the replay means stay 4 wide and in env space."""
    trainer = _head_trainer(tmp_path, num_envs=2)
    try:
        acts, lps, vals, means = trainer._select_actions(
            trainer.current_obs, trainer.current_central_state, with_means=True
        )
        assert acts.shape == (2, 12, 5) and means.shape == (2, 12, 4) and lps.shape == vals.shape == (2, 12)
        assert set(np.unique(acts[..., 3:])) <= {0.0, 1.0}
        assert (np.abs(means[..., 2:]) <= 1.0).all() and np.isfinite(lps).all()
    finally:
        trainer.close()


def test_hybrid_rollout_feeds_the_env_decoded_actions_and_stores_policy_actions(tmp_path: Path) -> None:
    """[RED] The env receives `to_env_actions` of the stored 5-float actions; the buffer keeps the 5 floats."""
    trainer = _head_trainer(tmp_path, num_envs=2, rollout_steps=6)
    seen = []
    real_step = trainer.vec_env.step_array

    def spy(actions, values, action_means=None):
        seen.append(np.array(actions, copy=True))
        return real_step(actions, values, action_means=action_means)

    trainer.vec_env.step_array = spy
    try:
        rollout = trainer.collect_rollouts(active_generation=1)
        stored = rollout["actions"]
        assert stored.shape == (6, 2, 12, 5) and len(seen) == 6
        decoded = np.clip(tp.to_env_actions(stored, 0.35), -1.0, 1.0)
        np.testing.assert_array_equal(np.stack(seen), decoded)
        assert ((np.stack(seen)[..., 2] > 0.35) == (stored[..., 3] == 1.0)).all()
        assert ((np.stack(seen)[..., 3] > 0.5) == (stored[..., 4] == 1.0)).all()
    finally:
        trainer.close()


def test_hybrid_rollout_log_probs_equal_the_update_recomputation(tmp_path: Path) -> None:
    """[RED] For every role group, the log-probs stored during the rollout equal the update-time recomputation (ratio 1)."""
    trainer = _head_trainer(tmp_path, num_envs=2, rollout_steps=8)
    try:
        rollout = trainer.collect_rollouts(active_generation=1)
        for indices, policy in trainer._rollout_groups():
            obs = torch.as_tensor(rollout["obs"][:, :, indices, :].reshape(-1, OBS_DIM))
            act = torch.as_tensor(rollout["actions"][:, :, indices, :].reshape(-1, 5))
            cen = torch.as_tensor(rollout["central_state"][:, :, indices, :].reshape(-1, CENTRAL_STATE_DIM))
            _, newlogp, _, _ = policy.get_action_and_value(obs, act, central_state=cen)
            stored = torch.as_tensor(rollout["logprobs"][:, :, indices].reshape(-1))
            torch.testing.assert_close(newlogp, stored, rtol=0.0, atol=1e-5)
    finally:
        trainer.close()


def test_hybrid_update_changes_the_actor_and_stays_finite(tmp_path: Path) -> None:
    """[RED] One seeded generation updates all four policies without NaNs."""
    trainer = _head_trainer(tmp_path, num_envs=2, rollout_steps=16)
    try:
        before = {n: p.detach().clone() for n, p in trainer.blue_field_policy.named_parameters()}
        rollout = trainer.collect_rollouts(active_generation=1)
        metrics = trainer.update_policies(rollout)
        after = dict(trainer.blue_field_policy.named_parameters())
        assert any(not torch.equal(before[n], after[n]) for n in before if n.startswith("actor_"))
        for role in ROLE_KEYS:
            assert all(math.isfinite(v) for v in metrics[role].values()), role
        for p in trainer.blue_field_policy.parameters():
            assert torch.isfinite(p).all()
    finally:
        trainer.close()


def test_hybrid_seeded_rollouts_are_reproducible(tmp_path: Path) -> None:
    """[RED] Two trainers with the same seed draw identical actions and log-probs."""
    outs = []
    for name in ("a", "b"):
        trainer = _head_trainer(tmp_path, name=name, num_envs=2, rollout_steps=6, seed=11)
        try:
            outs.append(trainer.collect_rollouts(active_generation=1))
        finally:
            trainer.close()
    for key in ("actions", "logprobs", "rewards"):
        np.testing.assert_array_equal(outs[0][key], outs[1][key])


def test_hybrid_replay_frames_record_four_wide_env_space_actions_and_means(tmp_path: Path) -> None:
    """[RED] Uniform replay frames hold (12, 4) `actions` and `action_mean`; fired kicks are > 0.35, others -1."""
    cfg = _hybrid_cfg(recording=True, save_dir=str(tmp_path / "rep"), max_rl_steps=6)
    cfg = replace(cfg, replay=replace(cfg.replay, interest_threshold=float("inf")))
    trainer = ParameterSharingPPOTrainer(
        config=cfg, num_envs=1, rollout_steps=6, device="cpu", checkpoint_dir=str(tmp_path / "ckpt"), seed=3
    )
    try:
        trainer.train(total_updates=1)
        frames = json.loads(Path(trainer.generation_history[0]["uniform_replay"]).read_text(encoding="utf-8"))["frames"]
        for f in frames[1:]:
            assert len(f["actions"]) == 12 and all(len(r) == 4 for r in f["actions"])
            assert len(f["action_mean"]) == 12 and all(len(r) == 4 for r in f["action_mean"])
            assert all(r[2] == -1.0 or r[2] > 0.35 for r in f["actions"])
            assert all(r[3] in (-1.0, 1.0) for r in f["actions"])
    finally:
        trainer.close()


# ------------------------------------------------------------------ logstd clamp and telemetry
@pytest.mark.parametrize("head", ["gaussian4", "hybrid5"])
@pytest.mark.parametrize("bounds", [(-1.25, -0.35), (-1.0, -0.8)])
def test_logstd_clamp_reads_the_configured_range(tmp_path: Path, head: str, bounds) -> None:
    """[GUARD] After a generation every actor sigma sits inside [logstd_min, logstd_max]; far-out values land on the edges."""
    trainer = _head_trainer(tmp_path, head=head, logstd_min=bounds[0], logstd_max=bounds[1])
    try:
        pols = (trainer.blue_field_policy, trainer.blue_keeper_policy, trainer.red_field_policy, trainer.red_keeper_policy)
        for i, pol in enumerate(pols):
            with torch.no_grad():
                pol.actor_logstd.fill_(2.0 if i % 2 == 0 else -4.0)
        trainer.train(total_updates=1)
        width = 4 if head == "gaussian4" else 3
        for i, pol in enumerate(pols):
            assert pol.actor_logstd.shape == (1, width)
            edge = bounds[1] if i % 2 == 0 else bounds[0]
            assert bool((pol.actor_logstd == torch.tensor(edge, dtype=torch.float32)).all())
    finally:
        trainer.close()


def test_generation_record_carries_fire_telemetry_for_hybrid5_only(tmp_path: Path) -> None:
    """[RED] hybrid5: per role `fire_sat` {kick, spec}, `p_kick`, `p_spec` of the rollout policy; gaussian4: none of them.
    Constant heads z_kick = +100, z_spec = 0 give saturation 1 / 0 and probabilities 0.9925521 / 0.5 before the update."""
    trainer = _head_trainer(tmp_path)
    try:
        for pol in (trainer.blue_field_policy, trainer.blue_keeper_policy, trainer.red_field_policy, trainer.red_keeper_policy):
            with torch.no_grad():
                pol.actor_mean[4].weight.zero_()
                pol.actor_mean[4].bias.copy_(torch.tensor([0.0, 0.0, 0.0, 100.0, 0.0]))
        trainer.train(total_updates=1)
        rec = trainer.generation_history[0]
        for role in ROLE_KEYS:
            assert rec["fire_sat"][role]["kick"] == pytest.approx(1.0) and rec["fire_sat"][role]["spec"] == pytest.approx(0.0)
            assert rec["p_kick"][role] == pytest.approx(0.9925521, abs=1e-5)
            assert rec["p_spec"][role] == pytest.approx(0.5, abs=1e-6)
        persisted = json.loads((tmp_path / "ckpt" / "training_history.json").read_text(encoding="utf-8"))
        assert persisted["generations"][0]["fire_sat"] == rec["fire_sat"]
    finally:
        trainer.close()
    plain = _head_trainer(tmp_path, head="gaussian4", name="plain")
    try:
        plain.train(total_updates=1)
        rec = plain.generation_history[0]
        assert not {"fire_sat", "p_kick", "p_spec"} & set(rec)
    finally:
        plain.close()


# ------------------------------------------------------------------ persistence
def test_hybrid_state_round_trips_and_stores_the_head(tmp_path: Path) -> None:
    """[RED] `training_state.pt` stores `policy_head`; a hybrid5 trainer reloads weights, bounds and Adam state exactly."""
    a = _head_trainer(tmp_path, name="a", num_envs=1, rollout_steps=8, fire_prob_floor=0.01, fire_logit_bound=8.0)
    try:
        a.train(total_updates=1)
        state = torch.load(tmp_path / "a" / "training_state.pt", map_location="cpu", weights_only=False)
        assert state["policy_head"] == "hybrid5"
    finally:
        a.close()
    b = _head_trainer(tmp_path, name="a", num_envs=1, rollout_steps=8, fire_prob_floor=0.01, fire_logit_bound=8.0)
    try:
        assert b.load_checkpoints() is True
        assert b.current_generation == 1
        for name, tensor in a.blue_field_policy.state_dict().items():
            assert torch.equal(tensor, b.blue_field_policy.state_dict()[name]), name
        assert float(b.blue_field_policy.fire_prob_floor) == pytest.approx(0.01)
    finally:
        b.close()
    plain = _head_trainer(tmp_path, head="gaussian4", name="g")
    try:
        plain.train(total_updates=1)
        assert torch.load(tmp_path / "g" / "training_state.pt", map_location="cpu", weights_only=False)["policy_head"] == "gaussian4"
    finally:
        plain.close()


@pytest.mark.parametrize("saved,configured", [("hybrid5", "gaussian4"), ("gaussian4", "hybrid5")])
def test_loading_a_checkpoint_of_the_other_head_raises_and_names_both(tmp_path: Path, saved: str, configured: str) -> None:
    """[RED] No conversion between heads: the error names both, through the state file and through bare policy files."""
    src = _head_trainer(tmp_path, head=saved, name="src")
    try:
        src.train(total_updates=1)
    finally:
        src.close()
    for bare in (False, True):
        if bare:
            (tmp_path / "src" / "training_state.pt").unlink()
        dst = _head_trainer(tmp_path, head=configured, name="src")
        try:
            with pytest.raises(ValueError, match=rf"(?s)(?=.*{saved})(?=.*{configured})"):
                dst.load_checkpoints()
            assert dst.current_generation == 0 and dst.match_history == [] and dst.generation_history == []
        finally:
            dst.close()


def test_state_without_a_head_key_is_a_gaussian4_checkpoint(tmp_path: Path) -> None:
    """[GUARD] Older states have no `policy_head`: they load into gaussian4 trainers and are refused by hybrid5 ones."""
    src = _head_trainer(tmp_path, head="gaussian4", name="old")
    try:
        src.train(total_updates=1)
    finally:
        src.close()
    path = tmp_path / "old" / "training_state.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    del state["policy_head"]
    torch.save(state, path)
    ok = _head_trainer(tmp_path, head="gaussian4", name="old")
    try:
        assert ok.load_checkpoints() is True and ok.current_generation == 1
    finally:
        ok.close()
    refused = _head_trainer(tmp_path, head="hybrid5", name="old")
    try:
        with pytest.raises(ValueError, match="gaussian4"):
            refused.load_checkpoints()
    finally:
        refused.close()


def test_hybrid_batched_selection_matches_the_per_group_networks(tmp_path: Path) -> None:
    """[RED] Deterministic selection returns, per role group, what each network's own `sample_action_and_value` returns."""
    trainer = _head_trainer(tmp_path, num_envs=2)
    try:
        torch.manual_seed(7)
        for pol in (trainer.blue_field_policy, trainer.blue_keeper_policy, trainer.red_field_policy, trainer.red_keeper_policy):
            with torch.no_grad():
                pol.actor_mean[4].weight.normal_(0.0, 0.4)
        obs = np.random.default_rng(0).normal(size=(2, 12, OBS_DIM)).astype(np.float32)
        cen = np.random.default_rng(1).normal(size=(2, 12, CENTRAL_STATE_DIM)).astype(np.float32)
        acts, lps, vals, means = trainer._select_actions(obs, cen, deterministic=True, with_means=True)
        for indices, policy in trainer._rollout_groups():
            g_obs = torch.as_tensor(obs[:, indices, :].reshape(-1, OBS_DIM))
            g_cen = torch.as_tensor(cen[:, indices, :].reshape(-1, CENTRAL_STATE_DIM))
            a, lp, v, m = policy.sample_action_and_value(g_obs, central_state=g_cen, deterministic=True, return_mean=True)
            np.testing.assert_allclose(acts[:, indices].reshape(-1, 5), a.numpy(), atol=1e-6)
            np.testing.assert_allclose(lps[:, indices].reshape(-1), lp.numpy(), atol=1e-5)
            np.testing.assert_allclose(vals[:, indices].reshape(-1), v.numpy(), atol=1e-5)
            np.testing.assert_allclose(means[:, indices].reshape(-1, 4), m.numpy(), atol=1e-6)
    finally:
        trainer.close()


def test_hybrid_batched_sampling_uses_each_groups_own_sigma(tmp_path: Path) -> None:
    """[RED] With a different sigma per role policy, sampled log-probs still equal the update recomputation of every group."""
    trainer = _head_trainer(tmp_path, num_envs=2, rollout_steps=6)
    try:
        pols = (trainer.blue_keeper_policy, trainer.blue_field_policy, trainer.red_keeper_policy, trainer.red_field_policy)
        for i, pol in enumerate(pols):
            with torch.no_grad():
                pol.actor_logstd.copy_(torch.tensor([[-0.3 - 0.3 * i, -0.4 - 0.2 * i, -0.5 + 0.1 * i]]))
        rollout = trainer.collect_rollouts(active_generation=1)
        for indices, policy in trainer._rollout_groups():
            obs = torch.as_tensor(rollout["obs"][:, :, indices, :].reshape(-1, OBS_DIM))
            act = torch.as_tensor(rollout["actions"][:, :, indices, :].reshape(-1, 5))
            cen = torch.as_tensor(rollout["central_state"][:, :, indices, :].reshape(-1, CENTRAL_STATE_DIM))
            _, newlogp, _, _ = policy.get_action_and_value(obs, act, central_state=cen)
            torch.testing.assert_close(newlogp, torch.as_tensor(rollout["logprobs"][:, :, indices].reshape(-1)), rtol=0.0, atol=1e-5)
        # The sampled noise really differs by group: deviations of the throttle sample scale with each group's sigma.
        sd = [float((rollout["actions"][:, :, idx, 0]).std()) for idx, _ in trainer._rollout_groups()]
        assert len(set(round(x, 3) for x in sd)) > 1
    finally:
        trainer.close()


# ==============================================================================
# Opponent snapshot pool (`pool_envs`)
# ==============================================================================
POOL_ROLES = ("blue_field", "blue_keeper", "red_field", "red_keeper")
TANH_HALF = 0.46211715726000974  # tanh(0.5)
TANH_QUARTER = 0.24491866240370913  # tanh(0.25)


def _pool_trainer(tmp_path: Path, head: str = "gaussian4", name: str = "ckpt", num_envs: int = 2,
                  pool_envs: int = 1, pool_size: int = 10, rollout_steps: int = 8, seed: int = 3,
                  max_rl_steps: int | None = None, recording: bool = False, ppo: dict | None = None,
                  **kwargs) -> ParameterSharingPPOTrainer:
    cfg = _quiet_cfg(save_dir=str(tmp_path / f"rep_{name}"), recording=recording)
    cfg = replace(cfg, ppo=replace(cfg.ppo, policy_head=head, **(ppo or {})))
    if max_rl_steps is not None:
        cfg = replace(cfg, physics=replace(cfg.physics, max_rl_steps=max_rl_steps))
    return ParameterSharingPPOTrainer(
        config=cfg, num_envs=num_envs, rollout_steps=rollout_steps, device="cpu",
        checkpoint_dir=str(tmp_path / name), seed=seed, pool_envs=pool_envs, pool_size=pool_size, **kwargs,
    )


def _seeded_net(head: str, seed: int) -> ActorCritic:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ActorCritic(policy_head=head)


def _constant_net(head: str, bias, logstd: float) -> ActorCritic:
    """Network whose action distribution ignores the observation: zero output weights, output bias `bias`."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        net = ActorCritic(policy_head=head, fire_prob_floor=0.0, fire_logit_bound=6.0) if head == "hybrid5" else ActorCritic()
    with torch.no_grad():
        net.actor_mean[4].weight.zero_()
        net.actor_mean[4].bias.copy_(torch.tensor(bias, dtype=torch.float32))
        net.actor_logstd.fill_(logstd)
    return net


def _write_member(ckpt: Path, gen: int, head: str = "gaussian4", seed: int = 100, net: ActorCritic | None = None) -> Path:
    """A snapshot directory `snapshots/genNNNN` holding the four role policies (one `net` for all, else seeded ones)."""
    directory = ckpt / "snapshots" / f"gen{gen:04d}"
    directory.mkdir(parents=True)
    for i, role in enumerate(POOL_ROLES):
        torch.save((net if net is not None else _seeded_net(head, seed + i)).state_dict(), directory / f"{role}_policy.pt")
    return directory


def _spy_env_actions(trainer: ParameterSharingPPOTrainer) -> dict:
    seen: dict = {"actions": [], "values": [], "means": []}
    orig = trainer.vec_env.step_array

    def spy(actions, values, action_means=None):
        seen["actions"].append(np.array(actions, copy=True))
        seen["values"].append(np.array(values, copy=True))
        seen["means"].append(None if action_means is None else np.array(action_means, copy=True))
        return orig(actions, values, action_means=action_means)

    trainer.vec_env.step_array = spy
    return seen


def _seeded_collect(trainer: ParameterSharingPPOTrainer, generation: int, torch_seed: int = 1234) -> dict:
    """`collect_rollouts` from a freshly seeded global torch stream, so two trainers can be compared draw for draw."""
    torch.manual_seed(torch_seed)
    return trainer.collect_rollouts(active_generation=generation)


def _train_to_generation(trainer: ParameterSharingPPOTrainer, completed: int) -> None:
    """Make `completed` the last completed generation by playing the next one (it writes the snapshot when due)."""
    trainer.current_generation = completed - 1
    trainer.train(total_updates=1)
    assert trainer.current_generation == completed


# ------------------------------------------------------------------ listing and schedule (pure functions)
def _make_snapshot_dirs(ckpt: Path, names) -> None:
    for name in names:
        (ckpt / "snapshots" / name).mkdir(parents=True)


@pytest.mark.parametrize(
    "last_completed, pool_size, expected",
    [
        (500, 3, [400, 450, 500]),   # newest three of gen0050..gen0600, bounded above by the last completed generation
        (499, 3, [350, 400, 450]),   # the bound is inclusive at 500 and exclusive at 499 -> 450 is the newest
        (120, 10, [50, 100]),        # fewer snapshots than the pool size
        (50, 10, [50]),              # inclusive bound
        (49, 10, []),                # nothing completed yet
        (0, 10, []),
        (600, 1, [600]),
        (600, 12, list(range(50, 650, 50))),
    ],
)
def test_list_pool_generations_hand_values(tmp_path: Path, last_completed: int, pool_size: int, expected) -> None:
    """[RED] The pool is the newest `pool_size` snapshot generations that are <= the last completed generation."""
    _make_snapshot_dirs(tmp_path, [f"gen{g:04d}" for g in range(50, 650, 50)])
    assert tp.list_pool_generations(str(tmp_path), last_completed, pool_size) == expected


def test_list_pool_generations_ignores_staging_and_foreign_entries_and_a_missing_root(tmp_path: Path) -> None:
    """[RED] `genNNNN.tmp` staging directories, unrelated names and plain files are not members; no root -> empty pool."""
    assert tp.list_pool_generations(str(tmp_path / "nowhere"), 500, 10) == []
    _make_snapshot_dirs(tmp_path, ["gen0050", "gen0100.tmp", "notes", "gen12x", "gen"])
    (tmp_path / "snapshots" / "gen0150").write_text("a file, not a snapshot", encoding="utf-8")
    assert tp.list_pool_generations(str(tmp_path), 500, 10) == [50]
    _make_snapshot_dirs(tmp_path, ["gen10000"])  # five-digit generations are listed
    assert tp.list_pool_generations(str(tmp_path), 20000, 10) == [50, 10000]


def test_pool_schedule_degenerate_inputs_give_an_empty_schedule() -> None:
    """[RED] No pool envs, or an empty pool listing, schedules nothing."""
    assert tp.pool_schedule(7, 60, 8, 0, [50, 100]) == []
    assert tp.pool_schedule(7, 60, 8, 2, []) == []


def test_pool_schedule_rejects_more_pool_envs_than_envs() -> None:
    with pytest.raises(ValueError, match="pool_envs"):
        tp.pool_schedule(7, 60, 4, 5, [50])
    assert len(tp.pool_schedule(7, 60, 4, 4, [50])) == 4  # the boundary: every env is a pool env


@pytest.mark.parametrize("generation, expected_sides", [(51, ["A", "B", "A", "B"]), (52, ["B", "A", "B", "A"])])
def test_pool_schedule_sides_alternate_with_generation_and_position(generation: int, expected_sides) -> None:
    """[RED] The k-th pool env freezes B when (g + k) is even and A otherwise (hand-listed for g = 51 and 52)."""
    sched = tp.pool_schedule(7, generation, 8, 4, [50, 100, 150])
    assert [e["frozen_team"] for e in sched] == expected_sides
    envs = [e["env"] for e in sched]
    assert len(set(envs)) == 4 and all(isinstance(i, int) and 0 <= i < 8 for i in envs)
    assert all(e["member_gen"] in (50, 100, 150) for e in sched)
    assert all(set(e) == {"env", "frozen_team", "member_gen"} for e in sched)


def test_pool_schedule_with_two_pool_envs_freezes_each_side_once_in_every_generation() -> None:
    """[RED] N = 2: one env freezes A and the other B, in each of 100 generations."""
    for g in range(51, 151):
        sched = tp.pool_schedule(5, g, 8, 2, [50])
        assert sorted(e["frozen_team"] for e in sched) == ["A", "B"], g


def test_pool_schedule_is_a_pure_function_that_touches_no_global_random_stream() -> None:
    """[RED] Same arguments, same schedule; calling it leaves numpy's and torch's global generators where they were."""
    np.random.seed(123)
    torch.manual_seed(456)
    np_state, torch_state = np.random.get_state(), torch.get_rng_state()
    first = tp.pool_schedule(9, 77, 8, 3, [50, 100, 150, 200])
    second = tp.pool_schedule(9, 77, 8, 3, [50, 100, 150, 200])
    assert first == second
    after = np.random.get_state()
    assert np_state[0] == after[0] and np.array_equal(np_state[1], after[1]) and np_state[2:] == after[2:]
    assert torch.equal(torch_state, torch.get_rng_state())
    # A different listing changes only which members can be drawn.
    assert [e["env"] for e in tp.pool_schedule(9, 77, 8, 3, [50])] == [e["env"] for e in first]


def test_pool_schedule_differs_between_seeds_over_fifty_generations() -> None:
    """[RED] Seeds 1 and 2 give different schedules in more than half of generations 51..100."""
    differing = sum(
        tp.pool_schedule(1, g, 8, 2, [50, 100]) != tp.pool_schedule(2, g, 8, 2, [50, 100]) for g in range(51, 101)
    )
    assert differing > 25


def test_pool_schedule_env_and_member_frequencies_are_uniform_within_three_binomial_sds() -> None:
    """[RED] Over 400 generations with N = 2 of 8 envs each env is chosen 100 +- 3 sd times (sd = sqrt(400 * 1/4 * 3/4)
    = 8.66); with four members, each member is drawn 200 +- 3 sd of 800 draws (sd = sqrt(800 * 1/4 * 3/4) = 12.25)."""
    env_counts = np.zeros(8)
    member_counts = {50: 0, 100: 0, 150: 0, 200: 0}
    for g in range(51, 451):
        for e in tp.pool_schedule(42, g, 8, 2, [50, 100, 150, 200]):
            env_counts[e["env"]] += 1
            member_counts[e["member_gen"]] += 1
    assert np.all(np.abs(env_counts - 100.0) <= 3 * 8.66), env_counts
    assert all(abs(c - 200.0) <= 3 * 12.25 for c in member_counts.values()), member_counts


# ------------------------------------------------------------------ settings, CLI, persistence
def test_pool_cli_defaults_and_flags() -> None:
    """[RED] `--pool-envs` defaults to 0 (today's behaviour) and `--pool-size` to 10."""
    parser = tp.build_arg_parser()
    args = parser.parse_args([])
    assert (args.pool_envs, args.pool_size) == (0, 10)
    args = parser.parse_args(["--pool-envs", "2", "--pool-size", "5"])
    assert (args.pool_envs, args.pool_size) == (2, 5)


def test_trainer_pool_settings_default_off_and_are_validated(tmp_path: Path) -> None:
    """[RED] Defaults: pool off, size 10. Rejected: negative or oversized pool_envs, size < 1 with a pool, a shared-team
    trainer with a pool, a negative seed with a pool; boundaries pool_envs = num_envs and pool_size = 1 are accepted."""
    off = ParameterSharingPPOTrainer(
        config=_quiet_cfg(save_dir=str(tmp_path / "r0")), num_envs=1, rollout_steps=4, device="cpu",
        checkpoint_dir=str(tmp_path / "c0"), seed=3,
    )
    try:
        assert (off.pool_envs, off.pool_size) == (0, 10)
    finally:
        off.close()
    for bad, match in (
        (dict(pool_envs=-1), "pool_envs"),
        (dict(pool_envs=3), "pool_envs"),
        (dict(pool_envs=1, pool_size=0), "pool_size"),
        (dict(pool_envs=1, separate_teams=False), "shared"),
        (dict(pool_envs=1, seed=-1), "seed"),
    ):
        with pytest.raises(ValueError, match=match):
            _pool_trainer(tmp_path, name="bad", num_envs=2, **bad)
    ok = _pool_trainer(tmp_path, name="edge", num_envs=2, pool_envs=2, pool_size=1)
    try:
        assert (ok.pool_envs, ok.pool_size) == (2, 1)
    finally:
        ok.close()
    zero = _pool_trainer(tmp_path, name="zero", pool_envs=0, pool_size=0)
    zero.close()


@pytest.mark.parametrize("extra", [["--pool-envs", "9"], ["--pool-envs", "1", "--pool-size", "0"], ["--pool-envs", "-1"]])
def test_main_rejects_bad_pool_settings_before_wiping_the_batch(tmp_path: Path, monkeypatch, extra) -> None:
    """[RED] Bad pool settings abort `main(--from-scratch)` with the field named; the active batch survives."""
    monkeypatch.chdir(tmp_path)
    marker = tmp_path / "checkpoints" / "marker.txt"
    marker.parent.mkdir()
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["train_ppo", "--from-scratch", "--total-updates", "1"] + extra)
    with pytest.raises(ValueError, match="pool_"):
        tp.main()
    assert marker.read_text(encoding="utf-8") == "keep"
    assert sorted(p.name for p in (tmp_path / "checkpoints").iterdir()) == ["marker.txt"]
    assert not (tmp_path / "replays").exists()


def test_pool_settings_persist_in_state_and_every_generation_record(tmp_path: Path) -> None:
    """[RED] `training_state.pt` and each generation record carry pool_envs, pool_size and the (here empty) schedule."""
    trainer = _pool_trainer(tmp_path, pool_envs=1, pool_size=4)
    try:
        trainer.train(total_updates=2)  # generations 1 and 2: no snapshot yet -> empty pool
    finally:
        trainer.close()
    state = torch.load(tmp_path / "ckpt" / "training_state.pt", map_location="cpu", weights_only=False)
    assert (state["pool_envs"], state["pool_size"]) == (1, 4)
    history = json.loads((tmp_path / "ckpt" / "training_history.json").read_text(encoding="utf-8"))
    for rec in history["generations"]:
        assert (rec["pool_envs"], rec["pool_size"], rec["pool_schedule"]) == (1, 4, [])
    assert all(m["pool"] is None for m in history["matches"])


def test_resume_with_different_pool_settings_raises_and_names_both_sides(tmp_path: Path) -> None:
    """[RED] Stored vs requested pool settings must match on resume, as config overrides must."""
    first = _pool_trainer(tmp_path, pool_envs=1, pool_size=4)
    try:
        first.train(total_updates=1)
    finally:
        first.close()
    for changes in (dict(pool_envs=2, pool_size=4), dict(pool_envs=1, pool_size=5), dict(pool_envs=0, pool_size=4)):
        resumed = _pool_trainer(tmp_path, **changes)
        try:
            with pytest.raises(ValueError) as exc:
                resumed.load_checkpoints()
            msg = str(exc.value)
            assert "pool_envs" in msg and "stored" in msg and "requested" in msg and "--accept-config-change" in msg
            assert resumed.current_generation == 0
        finally:
            resumed.close()
    same = _pool_trainer(tmp_path, pool_envs=1, pool_size=4)
    try:
        assert same.load_checkpoints() is True and same.current_generation == 1
    finally:
        same.close()


def test_accept_config_change_downgrades_a_pool_mismatch_to_one_line(tmp_path: Path, capsys) -> None:
    """[RED] With `accept_config_change` a pool mismatch prints one line and the requested values are stored next."""
    first = _pool_trainer(tmp_path, pool_envs=1, pool_size=4)
    try:
        first.train(total_updates=1)
    finally:
        first.close()
    resumed = _pool_trainer(tmp_path, pool_envs=2, pool_size=6, accept_config_change=True)
    try:
        capsys.readouterr()
        assert resumed.load_checkpoints() is True
        out = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
        assert len(out) == 1 and "pool_envs" in out[0]
        resumed.train(total_updates=1)
    finally:
        resumed.close()
    state = torch.load(tmp_path / "ckpt" / "training_state.pt", map_location="cpu", weights_only=False)
    assert (state["pool_envs"], state["pool_size"]) == (2, 6)


def test_state_without_pool_keys_reads_as_pool_off_and_pool_size_is_moot_when_off(tmp_path: Path, capsys) -> None:
    """[RED] A state saved before the keys existed resumes silently with pool_envs = 0 (any pool_size) and raises with a pool."""
    first = _pool_trainer(tmp_path, pool_envs=0)
    try:
        first.train(total_updates=1)
    finally:
        first.close()
    state_path = tmp_path / "ckpt" / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    assert (state["pool_envs"], state["pool_size"]) == (0, 10)
    del state["pool_envs"], state["pool_size"]
    torch.save(state, state_path)

    for size in (10, 3):
        resumed = _pool_trainer(tmp_path, pool_envs=0, pool_size=size)
        try:
            capsys.readouterr()
            assert resumed.load_checkpoints() is True
            assert capsys.readouterr().out.strip() == ""
        finally:
            resumed.close()
    mismatched = _pool_trainer(tmp_path, pool_envs=1)
    try:
        with pytest.raises(ValueError, match="pool_envs"):
            mismatched.load_checkpoints()
    finally:
        mismatched.close()


# ------------------------------------------------------------------ rollout behaviour
ROLLOUT_ARRAYS = ("obs", "central_state", "actions", "logprobs", "advantages", "returns", "values", "rewards", "dones")


def test_trainer_schedule_uses_its_pool_size_and_listing_up_to_the_previous_generation(tmp_path: Path) -> None:
    """[RED] The schedule for generation g draws from the newest `pool_size` snapshots completed before g."""
    trainer = _pool_trainer(tmp_path, pool_envs=2, pool_size=2)
    try:
        for g in (50, 100, 150):
            _write_member(tmp_path / "ckpt", g)
        assert trainer.pool_schedule_for(50) == []                       # gen0050 is not completed before generation 50
        assert {e["member_gen"] for e in trainer.pool_schedule_for(51)} == {50}
        members = {e["member_gen"] for g in range(152, 252) for e in trainer.pool_schedule_for(g)}
        assert members == {100, 150}                                     # pool_size = 2 keeps the newest two
        assert {e["member_gen"] for e in trainer.pool_schedule_for(101)} <= {50, 100}
        assert {e["member_gen"] for e in trainer.pool_schedule_for(151)} <= {100, 150}
    finally:
        trainer.close()


def test_schedule_is_the_same_in_two_trainers_and_after_save_and_load(tmp_path: Path) -> None:
    """[RED] Two trainer instances on the same checkpoint directory, and a resumed copy, reproduce the schedule,
    the generation record's schedule and the match tags."""
    first = _pool_trainer(tmp_path, name="a", pool_envs=1, max_rl_steps=4)
    try:
        _train_to_generation(first, 50)
        shutil.copytree(tmp_path / "a", tmp_path / "b")
        twin = _pool_trainer(tmp_path, name="a", pool_envs=1, max_rl_steps=4)
        resumed = _pool_trainer(tmp_path, name="b", pool_envs=1, max_rl_steps=4)
        try:
            assert resumed.load_checkpoints() and resumed.current_generation == 50
            for g in range(51, 71):
                assert first.pool_schedule_for(g) == twin.pool_schedule_for(g) == resumed.pool_schedule_for(g)
            first.train(total_updates=1)
            resumed.train(total_updates=1)
        finally:
            twin.close()
            resumed.close()
        rec_first, rec_resumed = first.generation_history[-1], resumed.generation_history[-1]
        assert rec_first["pool_schedule"] == rec_resumed["pool_schedule"] != []
        assert [m["pool"] for m in first.match_history[-4:]] == [m["pool"] for m in resumed.match_history[-4:]]
    finally:
        first.close()


def test_a_generation_with_an_empty_pool_equals_the_n0_rollout(tmp_path: Path) -> None:
    """[RED] With no snapshot yet, a pool trainer's rollout buffers are bit-identical to a pool_envs = 0 trainer's."""
    plain = _pool_trainer(tmp_path, name="plain", pool_envs=0)
    pooled = _pool_trainer(tmp_path, name="pooled", pool_envs=2)
    try:
        a = _seeded_collect(plain, 1)
        b = _seeded_collect(pooled, 1)
        for key in ROLLOUT_ARRAYS:
            assert np.array_equal(a[key], b[key]), key
        assert b["pool_schedule"] == [] and (b["frozen_seats"] is None or not np.any(b["frozen_seats"]))
        assert a["pool_schedule"] == [] and a["frozen_seats"] is None
    finally:
        plain.close()
        pooled.close()


@pytest.mark.parametrize("head", ["gaussian4", "hybrid5"])
def test_pool_sampling_never_touches_the_trainers_own_random_stream(tmp_path: Path, head: str) -> None:
    """[RED] With a pool env active (and a member loaded, and frozen seats sampled), every non-pool env's buffers are
    bit-identical to a pool_envs = 0 trainer's: loading and sampling use no draw of the trainer's generators."""
    plain = _pool_trainer(tmp_path, head=head, name="plain", pool_envs=0)
    pooled = _pool_trainer(tmp_path, head=head, name="pooled", pool_envs=1)
    try:
        _write_member(tmp_path / "pooled", 50, head=head)
        sched = pooled.pool_schedule_for(52)
        pool_env = sched[0]["env"]
        other = 1 - pool_env
        a = _seeded_collect(plain, 52)
        b = _seeded_collect(pooled, 52)
        for key in ROLLOUT_ARRAYS:
            assert np.array_equal(a[key][:, other], b[key][:, other]), key
        assert not np.array_equal(a["actions"][:, pool_env, 6:], b["actions"][:, pool_env, 6:])  # pool env really differs
    finally:
        plain.close()
        pooled.close()


@pytest.mark.parametrize("generation, frozen_seats, free_seats", [(51, range(0, 6), range(6, 12)), (52, range(6, 12), range(0, 6))])
def test_frozen_seats_play_the_pool_members_policy_gaussian4(tmp_path: Path, generation, frozen_seats, free_seats) -> None:
    """[RED] A constant member (action mean tanh([.5, -.5, .25, -.25]), sigma e^-20) fixes the frozen team's env actions
    to the hand values; g = 51 freezes team A (seats 0-5), g = 52 team B (seats 6-11); the other team is not affected."""
    trainer = _pool_trainer(tmp_path, pool_envs=1)
    try:
        _write_member(tmp_path / "ckpt", 50, net=_constant_net("gaussian4", [0.5, -0.5, 0.25, -0.25], -20.0))
        pool_env = trainer.pool_schedule_for(generation)[0]["env"]
        seen = _spy_env_actions(trainer)
        rollout = trainer.collect_rollouts(active_generation=generation)
        acts = np.stack(seen["actions"])  # (T, 2, 12, 4) as sent to the environments
        want = np.array([TANH_HALF, -TANH_HALF, TANH_QUARTER, -TANH_QUARTER])
        for seat in frozen_seats:
            np.testing.assert_allclose(acts[:, pool_env, seat], np.broadcast_to(want, (8, 4)), atol=1e-5)
        assert np.abs(acts[:, pool_env, list(free_seats)] - want).max() > 0.05
        assert np.abs(acts[:, 1 - pool_env] - want).max() > 0.05
        expected_mask = np.zeros((2, 12), dtype=bool)
        expected_mask[pool_env, list(frozen_seats)] = True
        assert np.array_equal(rollout["frozen_seats"], expected_mask)
        assert rollout["pool_schedule"] == [{"env": pool_env, "frozen_team": "A" if generation == 51 else "B", "member_gen": 50}]
    finally:
        trainer.close()


def _frozen_hybrid_run(tmp_path: Path, name: str, seed: int):
    trainer = _pool_trainer(tmp_path, head="hybrid5", name=name, pool_envs=1, seed=seed)
    try:
        _write_member(tmp_path / name, 50, net=_constant_net("hybrid5", [0.5, -0.5, 0.0, 0.0, 0.0], -20.0))
        pool_env = trainer.pool_schedule_for(52)[0]["env"]
        seen = _spy_env_actions(trainer)
        trainer.collect_rollouts(active_generation=52)
        return pool_env, np.stack(seen["actions"])[:, pool_env, 6:]  # team B is frozen at g = 52
    finally:
        trainer.close()


def test_hybrid5_frozen_seats_play_env_space_actions_and_repeat_for_the_same_seed(tmp_path: Path) -> None:
    """[RED] A hybrid5 trainer with N = 1 and a hybrid5 member: the frozen seats' env actions are in env space (kick is -1
    or the decoded power 0.35 + 0.65 * 0.5 = 0.675, special +-1, throttle / steer tanh(+-0.5)); two runs with the same seed
    give identical frozen actions and another seed does not."""
    _, run_a = _frozen_hybrid_run(tmp_path, "a", seed=3)
    _, run_b = _frozen_hybrid_run(tmp_path, "b", seed=3)
    _, run_c = _frozen_hybrid_run(tmp_path, "c", seed=4)
    assert np.array_equal(run_a, run_b)
    assert not np.array_equal(run_a, run_c)
    kick, special = run_a[..., 2], run_a[..., 3]
    assert run_a.shape == (8, 6, 4)
    assert np.all(np.isclose(kick, -1.0, atol=1e-6) | np.isclose(kick, 0.675, atol=1e-5))
    assert np.all((kick == -1.0) | (kick > 0.35))
    assert np.any(np.isclose(kick, -1.0)) and np.any(np.isclose(kick, 0.675, atol=1e-5))  # p = 1/2 over 48 draws
    assert set(np.unique(special)) == {-1.0, 1.0}
    assert np.isclose(run_a[..., 0], TANH_HALF, atol=1e-6).all() and np.isclose(run_a[..., 1], -TANH_HALF, atol=1e-6).all()


def test_pool_generations_with_recording_pass_the_members_means_and_values(tmp_path: Path) -> None:
    """[RED] With recording on, the frozen seats' replay means are the member's env-space means (tanh(bias) for a constant
    gaussian4 member) and their values come from the member's critic, not the trainer's."""
    trainer = _pool_trainer(tmp_path, pool_envs=1, recording=True)
    try:
        member = _constant_net("gaussian4", [0.5, -0.5, 0.25, -0.25], -20.0)
        _write_member(tmp_path / "ckpt", 50, net=member)
        pool_env = trainer.pool_schedule_for(51)[0]["env"]
        obs0, cen0 = trainer.current_obs.copy(), trainer.current_central_state.copy()
        own_values = trainer._compute_values(obs0, cen0)[pool_env, :6]
        seen = _spy_env_actions(trainer)
        trainer.collect_rollouts(active_generation=51)  # team A frozen
        means = np.stack(seen["means"])[:, pool_env, :6]
        np.testing.assert_allclose(means, np.broadcast_to([TANH_HALF, -TANH_HALF, TANH_QUARTER, -TANH_QUARTER], (8, 6, 4)), atol=1e-6)
        member_values = member.get_value(torch.as_tensor(obs0[pool_env, :6]), central_state=torch.as_tensor(cen0[pool_env, :6])).detach().numpy()
        first_step = np.stack(seen["values"])[0, pool_env, :6]
        np.testing.assert_allclose(first_step, member_values, atol=1e-5)
        assert not np.allclose(first_step, own_values, atol=1e-5)
    finally:
        trainer.close()


def _optimizer_state_bytes(policy: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[tuple[str, str], bytes]:
    """Every Adam state entry of every parameter of `policy` (actor and critic), keyed by (parameter name, state key)."""
    out: dict[tuple[str, str], bytes] = {}
    for name, param in policy.named_parameters():
        for key, val in optimizer.state.get(param, {}).items():
            out[(name, key)] = (
                val.detach().cpu().contiguous().numpy().tobytes() if isinstance(val, torch.Tensor) else repr(val).encode("utf-8")
            )
    return out


@pytest.mark.parametrize("generation, frozen_roles, learner_roles", [
    (52, ("red_field", "red_keeper"), ("blue_field", "blue_keeper")),   # team B frozen
    (51, ("blue_field", "blue_keeper"), ("red_field", "red_keeper")),   # team A frozen
])
def test_frozen_seats_do_not_learn(tmp_path: Path, generation, frozen_roles, learner_roles) -> None:
    """[RED] 1 env, N = 1: after `update_policies` the frozen team's two policies and their Adam states are byte-identical
    and the other team's two policies changed. The Adam comparison covers every parameter, critic included."""
    trainer = _pool_trainer(tmp_path, num_envs=1, pool_envs=1)
    try:
        trainer.update_policies(trainer.collect_rollouts(active_generation=1))  # populate every Adam state
        _write_member(tmp_path / "ckpt", 50)
        rollout = trainer.collect_rollouts(active_generation=generation)
        assert rollout["frozen_seats"] is not None and rollout["frozen_seats"].shape == (1, 12)
        before = _snapshot_actor_and_adam(trainer)
        weights_before = {role: _hash_tensors(list(pol.state_dict().values())) for role, pol, _ in _policy_groups(trainer)}
        adam_before = {role: _optimizer_state_bytes(pol, opt) for role, pol, opt in _policy_groups(trainer)}
        metrics = trainer.update_policies(rollout)
        after = _snapshot_actor_and_adam(trainer)
        weights_after = {role: _hash_tensors(list(pol.state_dict().values())) for role, pol, _ in _policy_groups(trainer)}
        adam_after = {role: _optimizer_state_bytes(pol, opt) for role, pol, opt in _policy_groups(trainer)}
        for role in frozen_roles:
            assert weights_after[role] == weights_before[role], role
            pol = dict((r, p) for r, p, _ in _policy_groups(trainer))[role]
            critic_names = {n for n, _ in pol.named_parameters() if n.startswith("critic.")}
            assert critic_names and critic_names <= {n for n, _ in adam_before[role]}, role  # critic Adam state is populated
            assert adam_after[role] == adam_before[role], role
            for part in (0, 1, 2):  # actor bytes, Adam state, critic bytes
                assert {k: v for k, v in after[part].items() if k[0] == role} == {k: v for k, v in before[part].items() if k[0] == role}
            assert math.isnan(metrics[role]["pg_loss"]) and metrics[role]["v_corr_lambda"] == 0.0
        for role in learner_roles:
            assert weights_after[role] != weights_before[role], role
            assert math.isfinite(metrics[role]["pg_loss"])
    finally:
        trainer.close()


@pytest.mark.parametrize("frozen_team, frozen_cols, roles", [("B", slice(6, 12), ("red_field", "red_keeper")), ("A", slice(0, 6), ("blue_field", "blue_keeper"))])
def test_frozen_rows_never_reach_a_batch_the_fire_statistics_or_the_value_correlations(tmp_path: Path, frozen_team, frozen_cols, roles) -> None:
    """[RED] On a 2-env hybrid5 rollout with env 1's team frozen and every frozen row poisoned with NaN, that team's update
    (weights, losses, v_corr_*, fire_sat_*, p_kick, p_spec) equals the update on the env-0-only slice of the rollout."""
    ppo = {"num_minibatches": 1}
    full = _pool_trainer(tmp_path, head="hybrid5", name="full", num_envs=2, pool_envs=0, ppo=ppo)
    ref = _pool_trainer(tmp_path, head="hybrid5", name="ref", num_envs=2, pool_envs=0, ppo=ppo)
    try:
        rollout = full.collect_rollouts(active_generation=1)
        sliced = {k: (v[:, :1].copy() if k in ROLLOUT_ARRAYS else v) for k, v in rollout.items()}
        poisoned = {k: (v.copy() if k in ROLLOUT_ARRAYS else v) for k, v in rollout.items()}
        frozen = np.zeros((2, 12), dtype=bool)
        frozen[1, frozen_cols] = True
        for key in ROLLOUT_ARRAYS:
            poisoned[key][:, 1, frozen_cols] = np.nan
        poisoned["frozen_seats"] = frozen
        m_full = full.update_policies(poisoned)
        m_ref = ref.update_policies(sliced)
        by_role_full = {role: pol for role, pol, _ in _policy_groups(full)}
        by_role_ref = {role: pol for role, pol, _ in _policy_groups(ref)}
        for role in roles:
            for (name, a), (_, b) in zip(by_role_full[role].state_dict().items(), by_role_ref[role].state_dict().items()):
                assert torch.isfinite(a).all(), (role, name)
                assert torch.equal(a, b), (role, name)
            assert set(m_full[role]) >= {"p_kick", "p_spec", "fire_sat_kick", "fire_sat_spec", "v_corr_lambda", "v_corr_mc"}
            assert m_full[role] == m_ref[role], role
            assert all(math.isfinite(v) for v in m_full[role].values())
    finally:
        full.close()
        ref.close()


def test_the_pool_members_weights_are_unchanged_on_disk_and_in_memory_after_a_generation(tmp_path: Path) -> None:
    """[RED] After a full generation (rollout, update, checkpoint) with the member in play, the snapshot files are
    byte-identical and the loaded member networks still equal the files and are not the trainer's networks."""
    trainer = _pool_trainer(tmp_path, pool_envs=2, max_rl_steps=4)
    try:
        _train_to_generation(trainer, 50)
        snap = tmp_path / "ckpt" / "snapshots" / "gen0050"
        digest = lambda: {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(snap.iterdir())}  # noqa: E731
        before = digest()
        trainer.train(total_updates=2)  # generations 51 and 52, both with the member in play
        assert digest() == before
        loaded = trainer._pool_members
        assert loaded, "the member of the last generation is retained for inspection"
        trainer_nets = {id(p) for _, p, _ in _policy_groups(trainer)}
        for (member_gen, team), (keeper, field) in loaded.items():
            assert member_gen == 50
            prefix = "blue" if team == "A" else "red"
            for net, role in ((keeper, "keeper"), (field, "field")):
                assert id(net) not in trainer_nets
                disk = torch.load(snap / f"{prefix}_{role}_policy.pt", map_location="cpu", weights_only=True)
                assert _hash_tensors(list(net.state_dict().values())) == _hash_tensors(list(disk.values()))
                assert not any(p.requires_grad for p in net.parameters()) and not net.training
        live = torch.load(tmp_path / "ckpt" / "blue_field_policy.pt", map_location="cpu", weights_only=True)
        snapshot = torch.load(snap / "blue_field_policy.pt", map_location="cpu", weights_only=True)
        assert _hash_tensors(list(live.values())) != _hash_tensors(list(snapshot.values()))  # training moved on
    finally:
        trainer.close()


# ------------------------------------------------------------------ records and co-evolution
def test_pool_matches_do_not_enter_the_coevolution_window(tmp_path: Path) -> None:
    """[RED] 24 matches of 1-0 for team A sync team B when untagged; the same 24 tagged as pool matches leave
    `recent_match_scores` empty and both teams' weights alone; a half-tagged set leaves 12 scores and no sync."""
    pool_tag = {"frozen_team": "B", "member_gen": 50}
    records = _matches_with_totals(24, 0, n=24)

    control = _coevo_trainer(tmp_path / "c")
    try:
        assert control._coevolution_step([dict(r) for r in records]) is True
        assert _team_b_weight(control) == pytest.approx(0.35)
    finally:
        control.close()

    pooled = _coevo_trainer(tmp_path / "p")
    try:
        assert pooled._coevolution_step([dict(r, pool=pool_tag) for r in records]) is False
        assert len(pooled.recent_match_scores) == 0
        assert (_team_a_weight(pooled), _team_b_weight(pooled)) == (1.0, 0.0)
    finally:
        pooled.close()

    half = _coevo_trainer(tmp_path / "h")
    try:
        tagged = [dict(r, pool=pool_tag) if i % 2 else dict(r, pool=None) for i, r in enumerate(records)]
        assert half._coevolution_step(tagged) is False
        assert list(half.recent_match_scores) == [(1, 0)] * 12
        assert _team_b_weight(half) == 0.0
    finally:
        half.close()


def test_a_training_generation_of_pool_matches_leaves_the_window_unchanged(tmp_path: Path) -> None:
    """[RED] 2 envs, N = 2, 4-step matches, 8 steps: gen 50 (no pool) adds its 4 matches to the window, gen 51 (every env
    a pool env) adds none."""
    trainer = _pool_trainer(tmp_path, pool_envs=2, max_rl_steps=4)
    try:
        _train_to_generation(trainer, 50)
        assert len(trainer.recent_match_scores) == 4
        trainer.train(total_updates=1)
        assert trainer.current_generation == 51 and len(trainer.match_history) == 8
        assert len(trainer.recent_match_scores) == 4
        assert [m["pool"] is not None for m in trainer.match_history[-4:]] == [True] * 4
        assert [m["pool"] for m in trainer.match_history[:4]] == [None] * 4
    finally:
        trainer.close()


def test_match_records_and_the_generation_record_carry_the_pool_tags(tmp_path: Path) -> None:
    """[RED] Gen 51, N = 1 of 2 envs, 4-step matches: two of four matches carry {"frozen_team": "A", "member_gen": 50},
    the others `None`; the generation record lists the schedule; a pool_envs = 0 trainer records empty schedules and `None` tags."""
    trainer = _pool_trainer(tmp_path, pool_envs=1, max_rl_steps=4)
    plain = _pool_trainer(tmp_path, name="plain", pool_envs=0, max_rl_steps=4)
    try:
        _train_to_generation(trainer, 50)
        trainer.train(total_updates=1)
        tags = [m["pool"] for m in trainer.match_history[-4:]]
        assert sorted(tags, key=lambda t: t is None) == [{"frozen_team": "A", "member_gen": 50}] * 2 + [None] * 2
        rec = trainer.generation_history[-1]
        pool_env = trainer.pool_schedule_for(51)[0]["env"]
        assert rec["pool_schedule"] == [{"env": pool_env, "frozen_team": "A", "member_gen": 50}]
        assert (rec["pool_envs"], rec["pool_size"]) == (1, 10)

        plain.train(total_updates=1)
        assert plain.generation_history[-1]["pool_schedule"] == [] and plain.generation_history[-1]["pool_envs"] == 0
        assert all(m["pool"] is None for m in plain.match_history)
    finally:
        trainer.close()
        plain.close()


def test_a_match_that_began_in_a_pool_generation_stays_tagged_when_it_ends_in_another(tmp_path: Path) -> None:
    """[RED] 6-step generations and 5-step matches make matches straddle generations. Seed 1 picks env 1 in gen 51 and env 0
    in gen 52 (asserted). Gen 51's tick-10 matches: env 1's began in gen 50 and ends under the pool, env 0's is free. Gen
    52's tick-15 matches: env 0's began in gen 51 under the pool and ends free, env 1's began free and ends under the pool;
    both are tagged, with the generation that froze them (A in gen 51, B in gen 52)."""
    trainer = _pool_trainer(tmp_path, pool_envs=1, seed=1, rollout_steps=6, max_rl_steps=5)
    try:
        _train_to_generation(trainer, 50)
        assert [trainer.pool_schedule_for(g)[0]["env"] for g in (51, 52)] == [1, 0]
        mark = len(trainer.match_history)
        trainer.train(total_updates=1)  # gen 51
        gen51 = trainer.match_history[mark:]
        assert len(gen51) == 2
        assert sorted(m["pool"] is None for m in gen51) == [False, True]
        assert [m["pool"] for m in gen51 if m["pool"]] == [{"frozen_team": "A", "member_gen": 50}]
        mark = len(trainer.match_history)
        trainer.train(total_updates=1)  # gen 52
        gen52 = trainer.match_history[mark:]
        assert len(gen52) == 2
        assert sorted((m["pool"] for m in gen52), key=lambda t: t["frozen_team"]) == [
            {"frozen_team": "A", "member_gen": 50}, {"frozen_team": "B", "member_gen": 50}]
    finally:
        trainer.close()


@pytest.mark.parametrize("trainer_head, member_head", [("hybrid5", "gaussian4"), ("gaussian4", "hybrid5")])
def test_a_member_with_the_other_head_is_skipped_with_one_line_and_its_envs_play_self_play(tmp_path: Path, capsys, trainer_head, member_head) -> None:
    """[RED] Both pool envs draw the only member, which has the other head: exactly one printed line names it and both heads,
    the schedule is empty, no seat is frozen, and the buffers equal a pool_envs = 0 trainer's."""
    plain = _pool_trainer(tmp_path, head=trainer_head, name="plain", pool_envs=0)
    pooled = _pool_trainer(tmp_path, head=trainer_head, name="pooled", pool_envs=2)
    try:
        _write_member(tmp_path / "pooled", 50, head=member_head)
        a = _seeded_collect(plain, 51)
        capsys.readouterr()
        b = _seeded_collect(pooled, 51)
        lines = [ln for ln in capsys.readouterr().out.splitlines() if "gen0050" in ln]
        assert len(lines) == 1 and member_head in lines[0] and trainer_head in lines[0]
        assert b["pool_schedule"] == [] and (b["frozen_seats"] is None or not np.any(b["frozen_seats"]))
        for key in ROLLOUT_ARRAYS:
            assert np.array_equal(a[key], b[key]), key
    finally:
        plain.close()
        pooled.close()


def test_hybrid5_pool_generation_keeps_the_fire_telemetry_finite_and_per_learner_batch(tmp_path: Path) -> None:
    """[RED] A hybrid5 pool generation records finite fire_sat / p_kick / p_spec for all four roles."""
    trainer = _pool_trainer(tmp_path, head="hybrid5", pool_envs=1, max_rl_steps=4)
    try:
        _train_to_generation(trainer, 50)
        trainer.train(total_updates=1)
        rec = trainer.generation_history[-1]
        assert rec["pool_schedule"] != []
        for role in ROLE_KEYS:
            assert all(math.isfinite(v) for v in (rec["fire_sat"][role]["kick"], rec["fire_sat"][role]["spec"], rec["p_kick"][role], rec["p_spec"][role]))
    finally:
        trainer.close()


def test_main_stores_the_pool_settings_and_enforces_them_on_resume(tmp_path: Path, monkeypatch) -> None:
    """[RED] End to end through `main()`: `--pool-envs` / `--pool-size` reach the stored state and the generation record,
    and resuming without them is refused."""
    monkeypatch.chdir(tmp_path)
    base = ["train_ppo", "--num-envs", "2", "--rollout-steps", "8", "--total-updates", "1", "--device", "cpu"]
    monkeypatch.setattr("sys.argv", base + ["--pool-envs", "1", "--pool-size", "4"])
    tp.main()
    state = torch.load(tmp_path / "checkpoints" / "training_state.pt", map_location="cpu", weights_only=False)
    assert (state["pool_envs"], state["pool_size"], state["current_generation"]) == (1, 4, 1)
    assert [(g["pool_envs"], g["pool_size"]) for g in state["generation_history"]] == [(1, 4)]

    monkeypatch.setattr("sys.argv", base)
    with pytest.raises(ValueError, match="pool_envs"):
        tp.main()
    assert torch.load(tmp_path / "checkpoints" / "training_state.pt", map_location="cpu", weights_only=False)["current_generation"] == 1


@pytest.mark.parametrize("head", ["gaussian4", "hybrid5"])
def test_frozen_seat_batch_matches_every_networks_own_forward_pass(head: str) -> None:
    """[RED] Four groups with four different networks (keeper / field seats of two teams in different envs): the batched
    values, replay means and (sigma e^-20) sampled Gaussian actions equal each network's own forward pass on its own
    rows; stored log-probabilities are 0; every seat outside the groups keeps its sentinel in every output."""
    rng = np.random.default_rng(5)
    nets = []
    for seed in range(4):
        net = _seeded_net(head, 40 + seed)
        with torch.no_grad():
            net.actor_mean[4].weight.mul_(60.0)  # outputs that depend on the observation
            net.actor_logstd.fill_(-20.0)
        nets.append(net)
    groups = [
        (np.array([0, 2]), [0], nets[0]),
        (np.array([0, 2]), [1, 2, 3, 4, 5], nets[1]),
        (np.array([1]), [6], nets[2]),
        (np.array([1]), [7, 8, 9, 10, 11], nets[3]),
    ]
    obs = rng.normal(size=(3, 12, OBS_DIM)).astype(np.float32)
    cen = rng.normal(size=(3, 12, CENTRAL_STATE_DIM)).astype(np.float32)
    width = 5 if head == "hybrid5" else 4
    actions = np.full((3, 12, width), np.nan, dtype=np.float32)
    logprobs = np.full((3, 12), np.nan, dtype=np.float32)
    values = np.full((3, 12), np.nan, dtype=np.float32)
    means = np.full((3, 12, 4), np.nan, dtype=np.float32)
    tp.FrozenSeatBatch(groups).apply(obs, cen, tp.pool_generator(1, 2), actions, logprobs, values, means)

    touched = np.zeros((3, 12), dtype=bool)
    for envs, seats, net in groups:
        touched[np.ix_(envs, seats)] = True
        ix = np.ix_(envs, seats)
        o = torch.as_tensor(obs[ix].reshape(-1, OBS_DIM))
        c = torch.as_tensor(cen[ix].reshape(-1, CENTRAL_STATE_DIM))
        _, _, _, value, mean = net.get_action_and_value(o, deterministic=True, central_state=c, return_mean=True)
        shape = (len(envs), len(seats))
        np.testing.assert_allclose(values[ix], value.detach().numpy().reshape(shape), atol=1e-4)
        np.testing.assert_allclose(means[ix], mean.detach().numpy().reshape(*shape, 4), atol=1e-5)
        assert (logprobs[ix] == 0.0).all()
        if head == "gaussian4":
            np.testing.assert_allclose(actions[ix], mean.detach().numpy().reshape(*shape, 4), atol=1e-5)
        else:
            mean3, _, _ = net.hybrid_heads(o)
            np.testing.assert_allclose(actions[ix][..., :3], mean3.detach().numpy().reshape(*shape, 3), atol=1e-5)
            assert set(np.unique(actions[ix][..., 3:])) <= {0.0, 1.0}
    for out in (actions, logprobs, values, means):
        assert np.isnan(out[~touched]).all()
    assert np.abs(values[touched]).max() > 0.01 and np.abs(means[touched]).max() > 0.05  # the check is not vacuous
    # Without replay means the same call must still fill actions, values and log-probabilities (and not fail).
    actions2, logprobs2, values2 = (np.full_like(a, np.nan) for a in (actions, logprobs, values))
    tp.FrozenSeatBatch(groups).apply(obs, cen, tp.pool_generator(1, 2), actions2, logprobs2, values2, None)
    np.testing.assert_allclose(values2[touched], values[touched], atol=1e-6)
    np.testing.assert_allclose(actions2[touched], actions[touched], atol=1e-6)
    assert np.isnan(actions2[~touched]).all()


# ------------------------------------------------------------------ hybrid5 update digest
HYBRID_UPDATE_DIGEST_ENV = ("2.6.0+cu124", 6)  # torch version and thread count the digests were recorded with
HYBRID_UPDATE_DIGESTS = {
    "blue_field": "15b1c6c9f379e1ae403afa91a667bb2afb088b89ea4306e6733a8577b5d40b14",
    "blue_keeper": "209ec30b5f4757e7b981ca6279313e1c1368d029bf657949bb2fdf73d99b2515",
    "red_field": "3792f0f5ceb6453dd1616abca979a93eea24a220cda22d57d46a74853e26826a",
    "red_keeper": "49a2f9256fa9113f6aa87b8de67290ad9204e65e1d21f6a4d56a6615bc3af819",
}


def _hybrid_update_digests(tmp_path: Path, **ppo) -> dict[str, str]:
    """SHA-256 per role over every actor and critic parameter and every Adam state tensor after one seeded update."""
    trainer = _head_trainer(tmp_path, num_envs=1, rollout_steps=16, seed=7, **ppo)
    try:
        trainer.update_policies(trainer.collect_rollouts(active_generation=1))
        digests = {}
        for role, pol, opt in _policy_groups(trainer):
            h = hashlib.sha256()
            for name, param in pol.named_parameters():
                h.update(name.encode("utf-8"))
                h.update(param.detach().cpu().contiguous().numpy().tobytes())
                for key in ("exp_avg", "exp_avg_sq", "step"):
                    h.update(opt.state[param][key].detach().cpu().contiguous().numpy().tobytes())
            digests[role] = h.hexdigest()
        return digests
    finally:
        trainer.close()


def test_hybrid_update_matches_golden_digest(tmp_path: Path) -> None:
    """[GUARD] One seeded hybrid5 update leaves every role's parameters and Adam state at the recorded SHA-256."""
    if HYBRID_UPDATE_DIGEST_ENV != (torch.__version__, torch.get_num_threads()):
        pytest.skip(f"digests recorded with torch {HYBRID_UPDATE_DIGEST_ENV}; running {(torch.__version__, torch.get_num_threads())}")
    assert _hybrid_update_digests(tmp_path) == HYBRID_UPDATE_DIGESTS


# ------------------------------------------------------------------ fire-head entropy switch
from football_rl.config import PPOHyperparams  # noqa: E402

# logstd = 0 on all three Gaussians, p_kick = p_spec = 0.25, ent_coef = 0.005.
# Columns: fire_entropy, fire_ent_coef, bonus, d/dp_kick, d/dp_spec (d/dlogstd is the same in every row).
BONUS_TABLE = [
    ("joint", 0.0, 0.02158641, 0.01258775, 0.00549306),
    ("separate", 0.0, 0.01596306, 0.0, 0.0),
    ("separate", 0.002, 0.01821240, 0.00219722, 0.00219722),
    ("separate", 0.005, 0.02158641, 0.00549306, 0.00549306),
]
BONUS_DLOGSTD = (0.005, 0.005, 0.00125)
JOINT_KICK_MAXIMISER = 0.744395  # sigmoid(H_N(power)) at logstd_power = -0.35


def _bonus_and_grads(logstd_values, p_values, ent_coef, fire_entropy, fire_ent_coef):
    """Per-sample bonus (float64) and the gradients of its sum with respect to logstd and p (zeros when unused)."""
    logstd = torch.tensor([list(logstd_values)], dtype=torch.float64, requires_grad=True)
    p = torch.tensor(p_values, dtype=torch.float64, requires_grad=True)
    bonus = tp.hybrid_entropy_bonus(logstd, p, ent_coef, fire_entropy, fire_ent_coef)
    g_logstd, g_p = torch.autograd.grad(bonus.sum(), (logstd, p), allow_unused=True)
    g_logstd = torch.zeros_like(logstd) if g_logstd is None else g_logstd
    g_p = torch.zeros_like(p) if g_p is None else g_p
    return bonus.detach(), g_logstd, g_p


def test_fire_entropy_config_defaults_are_todays_loss() -> None:
    """[RED] The defaults select the joint entropy with no separate fire coefficient."""
    ppo = DEFAULT_CONFIG.ppo
    assert (ppo.fire_entropy, ppo.fire_ent_coef) == ("joint", 0.0)


@pytest.mark.parametrize("bad", ["", "Joint", "both", "separate "])
def test_fire_entropy_rejects_unknown_modes(bad: str) -> None:
    """[RED] fire_entropy must be exactly 'joint' or 'separate'; the error names the field."""
    with pytest.raises(ValueError, match="fire_entropy"):
        PPOHyperparams(policy_head="hybrid5", fire_entropy=bad)


@pytest.mark.parametrize("bad", [-1e-9, -0.002, float("nan"), float("inf"), float("-inf")])
def test_fire_ent_coef_must_be_finite_and_non_negative(bad: float) -> None:
    """[RED] fire_ent_coef rejects negative and non-finite values; the error names the field."""
    with pytest.raises(ValueError, match="fire_ent_coef"):
        PPOHyperparams(policy_head="hybrid5", fire_entropy="separate", fire_ent_coef=bad)


def test_separate_fire_entropy_requires_the_hybrid_head() -> None:
    """[RED] 'separate' with the gaussian4 head is an error naming fire_entropy; with hybrid5 it is accepted."""
    with pytest.raises(ValueError, match="fire_entropy"):
        PPOHyperparams(policy_head="gaussian4", fire_entropy="separate")
    ok = PPOHyperparams(policy_head="hybrid5", fire_entropy="separate", fire_ent_coef=0.0)
    assert (ok.fire_entropy, ok.fire_ent_coef) == ("separate", 0.0)


@pytest.mark.parametrize("head", ["gaussian4", "hybrid5"])
def test_joint_fire_entropy_rejects_a_fire_coefficient(head: str) -> None:
    """[RED] A coefficient that 'joint' would ignore is an error naming fire_ent_coef, under either head."""
    with pytest.raises(ValueError, match="fire_ent_coef"):
        PPOHyperparams(policy_head=head, fire_entropy="joint", fire_ent_coef=0.001)


def test_fire_entropy_is_reachable_through_config_overrides() -> None:
    """[RED] The command-line override path sets the three fields together and returns the coerced values."""
    cfg, applied = tp.parse_config_overrides(
        ["ppo.policy_head=hybrid5", "ppo.fire_entropy=separate", "ppo.fire_ent_coef=0"]
    )
    assert (cfg.ppo.policy_head, cfg.ppo.fire_entropy, cfg.ppo.fire_ent_coef) == ("hybrid5", "separate", 0.0)
    assert isinstance(cfg.ppo.fire_ent_coef, float)
    assert applied == {"ppo.fire_ent_coef": 0.0, "ppo.fire_entropy": "separate", "ppo.policy_head": "hybrid5"}
    with pytest.raises(ValueError, match="fire_entropy"):
        tp.parse_config_overrides(["ppo.fire_entropy=separate"])  # head left at gaussian4


@pytest.mark.parametrize(("mode", "fire_coef", "value", "d_kick", "d_spec"), BONUS_TABLE)
def test_hybrid_entropy_bonus_hand_values(mode, fire_coef, value, d_kick, d_spec) -> None:
    """[RED] Value and gradients at logstd 0, p = 0.25, ent_coef 0.005 (hand-solved table; 1e-7).

    joint: 0.005 (2.25 H_N + 2 H_B(0.25)) with H_N = 1.418939, H_B(0.25) = 0.562335. d/dp_kick = 0.005 (ln 3 + H_N),
    d/dp_spec = 0.005 ln 3. separate: p_kick is detached in the power term, so only fire_ent_coef * ln 3 remains.
    """
    bonus, g_logstd, g_p = _bonus_and_grads((0.0, 0.0, 0.0), [[0.25, 0.25]], 0.005, mode, fire_coef)
    assert bonus.shape == (1,)
    assert float(bonus[0]) == pytest.approx(value, abs=1e-7)
    assert float(g_p[0, 0]) == pytest.approx(d_kick, abs=1e-7)
    assert float(g_p[0, 1]) == pytest.approx(d_spec, abs=1e-7)
    np.testing.assert_allclose(g_logstd.numpy().reshape(3), BONUS_DLOGSTD, atol=1e-7)


def test_hybrid_entropy_bonus_is_per_sample_and_row_independent() -> None:
    """[RED] A two-row batch returns two values; the p = 0.5 row has the joint value 0.005 (2.5 H_N + 2 ln 2) = 0.02466820."""
    bonus, _, _ = _bonus_and_grads((0.0, 0.0, 0.0), [[0.25, 0.25], [0.5, 0.5]], 0.005, "joint", 0.0)
    assert bonus.shape == (2,)
    np.testing.assert_allclose(bonus.numpy(), [0.02158641, 0.02466820], atol=1e-7)
    sep, _, _ = _bonus_and_grads((0.0, 0.0, 0.0), [[0.25, 0.25], [0.5, 0.5]], 0.005, "separate", 0.0)
    np.testing.assert_allclose(sep.numpy(), [0.01596306, 0.01773673], atol=1e-7)


def test_separate_bonus_with_zero_ent_coef_keeps_only_the_fire_term() -> None:
    """[RED] ent_coef 0, fire_ent_coef 0.002 at p = 0.25: value 0.002 * 2 * H_B(0.25) = 0.00224934; no logstd gradient."""
    bonus, g_logstd, g_p = _bonus_and_grads((0.0, 0.0, 0.0), [[0.25, 0.25]], 0.0, "separate", 0.002)
    assert float(bonus[0]) == pytest.approx(0.00224934, abs=1e-7)
    assert not g_logstd.any()
    np.testing.assert_allclose(g_p.numpy(), [[0.00219722, 0.00219722]], atol=1e-7)


def test_hybrid_entropy_bonus_rejects_an_unknown_mode() -> None:
    """[RED] An unrecognised mode raises instead of silently picking one."""
    with pytest.raises(ValueError, match="fire_entropy"):
        tp.hybrid_entropy_bonus(torch.zeros(1, 3), torch.full((1, 2), 0.25), 0.005, "both", 0.0)


def test_joint_entropy_pays_the_kick_head_for_firing_up_to_the_maximiser() -> None:
    """[RED] Joint, logstd_power = -0.35: d bonus / d p_kick is positive below p = sigmoid(1.0689385) = 0.744395 and zero at it.

    Separate with a zero fire coefficient has no such pull at any p_kick.
    """
    logstd = (0.0, 0.0, -0.35)
    grad_kick = lambda p, mode, coef: float(_bonus_and_grads(logstd, [[p, 0.25]], 0.005, mode, coef)[2][0, 0])
    for p in (0.01, 0.129, 0.5, 0.74):
        assert grad_kick(p, "joint", 0.0) > 0.0
    assert grad_kick(JOINT_KICK_MAXIMISER - 1e-6, "joint", 0.0) > 0.0
    assert grad_kick(JOINT_KICK_MAXIMISER + 1e-6, "joint", 0.0) < 0.0
    assert grad_kick(0.9, "joint", 0.0) < 0.0
    for p in (0.01, 0.129, 0.5, JOINT_KICK_MAXIMISER, 0.9):
        assert grad_kick(p, "separate", 0.0) == 0.0


def test_get_action_and_value_can_return_the_head_parameters_it_used() -> None:
    """[RED] `return_heads` appends `(logstd, p)` of the same forward pass; gaussian4 has no such heads and refuses."""
    net = _constant_hybrid_net([0.0, 0.0, 0.0, Z_FOR_QUARTER, 0.0], logstd=(0.1, -0.2, -0.35))
    obs = torch.randn(3, OBS_DIM)
    out = net.get_action_and_value(obs, torch.zeros(3, 5), return_heads=True)
    assert len(out) == 5
    logstd, p = out[4]
    np.testing.assert_allclose(logstd.detach().numpy(), [[0.1, -0.2, -0.35]], atol=1e-7)
    np.testing.assert_allclose(p.detach().numpy(), [[0.25, 0.5]] * 3, atol=1e-6)
    with_mean = net.get_action_and_value(obs, torch.zeros(3, 5), return_mean=True, return_heads=True)
    assert len(with_mean) == 6 and with_mean[5][1].shape == (3, 2)
    with pytest.raises(ValueError, match="hybrid5"):
        ActorCritic().get_action_and_value(obs, return_heads=True)


# ------------------------------------------------------------------ update behaviour
def _zero_advantage_update(tmp_path: Path, **ppo):
    """Run `_ppo_update_role` on a seeded batch with advantage 0 and returns equal to the values; return before/after."""
    torch.manual_seed(11)
    trainer = _head_trainer(tmp_path, **ppo)
    try:
        policy, opt, _ = _bandit_policy(trainer)
        obs = torch.randn(32, OBS_DIM)
        with torch.no_grad():
            action, logp, value = policy.sample_action_and_value(obs)
        before = {n: p.detach().clone() for n, p in policy.named_parameters()}
        with torch.no_grad():
            p_before = torch.stack(policy.hybrid_heads(obs)[1:], dim=-1).clone()
        trainer._ppo_update_role(policy, opt, obs, action, logp, torch.zeros(32), value.clone(), value)
        with torch.no_grad():
            p_after = torch.stack(policy.hybrid_heads(obs)[1:], dim=-1)
        after = {n: p.detach().clone() for n, p in policy.named_parameters()}
        return before, after, p_before, p_after
    finally:
        trainer.close()


def test_zero_advantage_gives_an_exactly_zero_policy_gradient_term(tmp_path: Path) -> None:
    """[GUARD] With advantage 0, ent_coef 0 and returns = values, an update leaves every actor parameter byte-identical.

    This establishes that the zero-advantage cases below isolate the entropy term: normalisation maps zeros to zeros.
    """
    before, after, _, _ = _zero_advantage_update(tmp_path, ent_coef=0.0)
    for name in before:
        if name.startswith("actor_"):
            assert torch.equal(before[name], after[name]), name


def test_separate_fire_entropy_leaves_the_mean_head_untouched_without_advantage(tmp_path: Path) -> None:
    """[RED] separate, 0, zero advantage: every `actor_mean` parameter is byte-identical and `actor_logstd` moves."""
    before, after, _, _ = _zero_advantage_update(tmp_path, fire_entropy="separate", fire_ent_coef=0.0)
    for name in before:
        if name.startswith("actor_mean"):
            assert torch.equal(before[name], after[name]), name
    assert not torch.equal(before["actor_logstd"], after["actor_logstd"])


def test_joint_fire_entropy_raises_both_fire_probabilities_without_advantage(tmp_path: Path) -> None:
    """[GUARD] joint, zero advantage: `actor_mean` changes and p_kick and p_spec at a fixed observation both rise.

    At the initial p_kick = 0.129 and p_spec = 0.0275 both entropy gradients point upward (ln((1-p)/p) > 0).
    """
    before, after, p_before, p_after = _zero_advantage_update(tmp_path)
    assert any(not torch.equal(before[n], after[n]) for n in before if n.startswith("actor_mean"))
    assert (p_after[0] > p_before[0]).all(), (p_before[0], p_after[0])
    assert (p_after.mean(dim=0) > p_before.mean(dim=0)).all()


def test_separate_fire_entropy_with_a_coefficient_raises_both_fire_probabilities(tmp_path: Path) -> None:
    """[RED] separate with fire_ent_coef 0.005 and zero advantage: the Bernoulli bonus alone lifts p_kick and p_spec."""
    before, after, p_before, p_after = _zero_advantage_update(tmp_path, fire_entropy="separate", fire_ent_coef=0.005)
    assert any(not torch.equal(before[n], after[n]) for n in before if n.startswith("actor_mean"))
    assert (p_after[0] > p_before[0]).all(), (p_before[0], p_after[0])


def test_hybrid_bandit_learns_to_fire_and_to_stop_firing_without_fire_entropy(tmp_path: Path) -> None:
    """[RED] separate, 0: the one-state bandit still rises above 0.9 and falls below 0.05 inside the same update budget."""
    torch.manual_seed(0)
    trainer = _head_trainer(tmp_path, fire_entropy="separate", fire_ent_coef=0.0)
    try:
        policy, opt, obs = _bandit_policy(trainer)
        up = _bandit_calls(trainer, policy, opt, obs, +1.0, BANDIT_UPDATE_CALLS, lambda p: p > 0.9)
        assert _p_kick(policy, obs) > 0.9, f"P(kick)={_p_kick(policy, obs):.3f} after {up} calls"
        down = _bandit_calls(trainer, policy, opt, obs, -1.0, BANDIT_UPDATE_CALLS, lambda p: p < 0.05)
        assert _p_kick(policy, obs) < 0.05, f"P(kick)={_p_kick(policy, obs):.3f} after {down} calls"
        print(f"separate bandit update calls needed: rise {up}, fall {down}, budget {BANDIT_UPDATE_CALLS}")
    finally:
        trainer.close()


@pytest.mark.parametrize(("mode", "fire_coef"), [("joint", 0.0), ("separate", 0.0), ("separate", 0.005)])
def test_reported_entropy_is_the_full_entropy_in_every_mode(tmp_path: Path, mode: str, fire_coef: float) -> None:
    """[RED] One minibatch, one epoch: the `entropy` statistic is the full hybrid entropy, 4.448094 at p = (0.25, 0.5), sigma 1."""
    torch.manual_seed(5)
    trainer = _head_trainer(tmp_path, fire_entropy=mode, fire_ent_coef=fire_coef, num_minibatches=1, update_epochs=1)
    try:
        policy, opt, _ = _bandit_policy(trainer)
        policy.load_state_dict(_constant_hybrid_net([0.0, 0.0, 0.0, Z_FOR_QUARTER, 0.0]).state_dict())
        obs = torch.randn(16, OBS_DIM)
        with torch.no_grad():
            action, logp, value = policy.sample_action_and_value(obs)
        stats = trainer._ppo_update_role(policy, opt, obs, action, logp, torch.zeros(16), value.clone(), value)
        assert stats["entropy"] == pytest.approx(4.448094, abs=1e-5)
    finally:
        trainer.close()


@pytest.mark.parametrize("mode", ["joint", "separate"])
def test_reported_entropy_equals_the_batch_mean_of_the_policy_entropy(tmp_path: Path, mode: str) -> None:
    """[RED] For an observation-dependent policy the statistic equals the mean `hybrid_log_prob_entropy` entropy of the batch."""
    torch.manual_seed(6)
    trainer = _head_trainer(tmp_path, fire_entropy=mode, fire_ent_coef=0.0, num_minibatches=1, update_epochs=1)
    try:
        policy, opt, _ = _bandit_policy(trainer)
        with torch.no_grad():
            policy.actor_mean[4].weight.normal_(0.0, 0.3)
        obs = torch.randn(24, OBS_DIM)
        with torch.no_grad():
            action, logp, value = policy.sample_action_and_value(obs)
            mean3, p_kick, p_spec = policy.hybrid_heads(obs)
            expected = float(tp.hybrid_log_prob_entropy(mean3, policy.actor_logstd, p_kick, p_spec, action)[1].mean())
        stats = trainer._ppo_update_role(policy, opt, obs, action, logp, torch.zeros(24), value.clone(), value)
        assert stats["entropy"] == pytest.approx(expected, abs=1e-6)
        assert stats["entropy"] > 2.0  # the throttle and steer Gaussians alone give 2 * 0.918939 = 1.837877
    finally:
        trainer.close()


def test_separate_fire_entropy_adds_no_actor_forward_pass_to_the_update(tmp_path: Path) -> None:
    """[RED] The actor runs once per minibatch step plus the single fire-statistics pass, identically in both modes."""
    counts = {}
    for mode in ("joint", "separate"):
        torch.manual_seed(7)
        trainer = _head_trainer(tmp_path, name=mode, fire_entropy=mode, num_minibatches=4, update_epochs=3)
        try:
            policy, opt, _ = _bandit_policy(trainer)
            obs = torch.randn(32, OBS_DIM)
            with torch.no_grad():
                action, logp, value = policy.sample_action_and_value(obs)
            calls = []
            handle = policy.actor_mean.register_forward_hook(lambda *_: calls.append(1))
            trainer._ppo_update_role(policy, opt, obs, action, logp, torch.randn(32), value.clone(), value)
            handle.remove()
            counts[mode] = len(calls)
        finally:
            trainer.close()
    assert counts == {"joint": 1 + 3 * 4, "separate": 1 + 3 * 4}


def test_separate_power_term_is_weighted_by_the_kick_probability_not_the_special_one() -> None:
    """[RED] separate, 0, logstd 0: p = (0.25, 0.5) gives 0.005 * 2.25 * H_N = 0.01596306; p = (0.5, 0.25) gives 0.005 * 2.5 * H_N = 0.01773673."""
    kick_low, _, _ = _bonus_and_grads((0.0, 0.0, 0.0), [[0.25, 0.5]], 0.005, "separate", 0.0)
    kick_high, _, _ = _bonus_and_grads((0.0, 0.0, 0.0), [[0.5, 0.25]], 0.005, "separate", 0.0)
    assert float(kick_low[0]) == pytest.approx(0.01596306, abs=1e-7)
    assert float(kick_high[0]) == pytest.approx(0.01773673, abs=1e-7)


@pytest.mark.parametrize(
    ("mode", "fire_coef", "d_bias_kick"),
    [("joint", 0.0, 0.00228107), ("separate", 0.0, 0.0), ("separate", 0.002, 0.00039817), ("separate", 0.005, 0.00099542)],
)
def test_update_step_moves_the_head_by_the_hand_solved_entropy_gradient(tmp_path: Path, mode, fire_coef, d_bias_kick) -> None:
    """[RED] One plain SGD step (lr 1) at zero advantage moves each parameter by the gradient of the entropy bonus.

    Constant policy: sigma 1, p_kick 0.25, p_spec 0.5, sigmoid slope dp/dz = p (1 - p) sech^2(z / 6) = 0.18121381 for
    the kick head. d logstd = ent_coef * (1, 1, p_kick) = (0.005, 0.005, 0.00125) in every mode. d bias_kick is
    joint: 0.005 (ln 3 + H_N) * 0.18121381; separate: fire_ent_coef * ln 3 * 0.18121381. The special bias stays put
    because ln((1 - 0.5) / 0.5) = 0.
    """
    torch.manual_seed(8)
    trainer = _head_trainer(tmp_path, fire_entropy=mode, fire_ent_coef=fire_coef, num_minibatches=1, update_epochs=1)
    try:
        policy, _, _ = _bandit_policy(trainer)
        policy.load_state_dict(_constant_hybrid_net([0.0, 0.0, 0.0, Z_FOR_QUARTER, 0.0]).state_dict())
        sgd = torch.optim.SGD(policy.parameters(), lr=1.0)
        obs = torch.randn(16, OBS_DIM)
        with torch.no_grad():
            action, logp, value = policy.sample_action_and_value(obs)
        logstd0, bias0 = policy.actor_logstd.detach().clone(), policy.actor_mean[4].bias.detach().clone()
        trainer._ppo_update_role(policy, sgd, obs, action, logp, torch.zeros(16), value.clone(), value)
        d_logstd = (policy.actor_logstd.detach() - logstd0).numpy().reshape(3)
        d_bias = (policy.actor_mean[4].bias.detach() - bias0).numpy()
        np.testing.assert_allclose(d_logstd, BONUS_DLOGSTD, atol=1e-7)
        np.testing.assert_allclose(d_bias, [0.0, 0.0, 0.0, d_bias_kick, 0.0], atol=1e-7)
    finally:
        trainer.close()


# ------------------------------------------------------------------ critic switches: value clip and return normalisation
def _critic_trainer(tmp_path: Path, name: str = "ckpt", head: str = "gaussian4", num_envs: int = 1,
                    rollout_steps: int = 24, seed: int = 5, separate_teams: bool = True,
                    full_batch: bool = False, **ppo) -> ParameterSharingPPOTrainer:
    """Trainer with the given PPO overrides; `full_batch` uses one minibatch and one epoch, so an update reports its pre-step loss."""
    cfg = _quiet_cfg(save_dir=str(tmp_path / f"rep_{name}"))
    if full_batch:
        ppo = {"num_minibatches": 1, "update_epochs": 1, **ppo}
    cfg = replace(cfg, ppo=replace(cfg.ppo, policy_head=head, **ppo))
    return ParameterSharingPPOTrainer(
        config=cfg, num_envs=num_envs, rollout_steps=rollout_steps, device="cpu",
        checkpoint_dir=str(tmp_path / name), seed=seed, separate_teams=separate_teams,
    )


def _set_return_stats(net: ActorCritic, mean: float, std: float) -> None:
    with torch.no_grad():
        net.ret_mean.fill_(mean)
        net.ret_std.fill_(std)


def _constant_head_critic(net: ActorCritic, head_value: float) -> None:
    """Critic whose raw head output is `head_value` for every input: zero last-layer weights, bias `head_value`."""
    with torch.no_grad():
        net.critic[4].weight.zero_()
        net.critic[4].bias.fill_(head_value)


def _critic_staircase(net: ActorCritic) -> torch.Tensor:
    """Make the critic output 0, 1, 2 for observations whose first column is the returned values.

    Hidden unit 0 carries `tanh(tanh(x))` (the other units are silent) and the output is `2 * h + 1`, so the outputs
    0 / 1 / 2 need `tanh(tanh(x)) = -0.5 / 0 / +0.5`, i.e. `x = atanh(atanh(+-0.5))`.
    """
    with torch.no_grad():
        for layer in (net.critic[0], net.critic[2], net.critic[4]):
            layer.weight.zero_()
            layer.bias.zero_()
        net.critic[0].weight[0, 0] = 1.0
        net.critic[2].weight[0, 0] = 1.0
        net.critic[4].weight[0, 0] = 2.0
        net.critic[4].bias.fill_(1.0)
    x = math.atanh(math.atanh(0.5))
    obs = torch.zeros(3, OBS_DIM)
    obs[:, 0] = torch.tensor([-x, 0.0, x])
    return obs


def _critic_update(trainer, policy, obs, old_values, returns, critic_only: bool) -> float:
    """Run one `_ppo_update_role` with zero advantages and a zero learning rate; return the reported value loss."""
    n = obs.shape[0]
    with torch.no_grad():
        action, logp, _ = policy.sample_action_and_value(obs)
    sgd = torch.optim.SGD(policy.parameters(), lr=0.0)
    metrics = trainer._ppo_update_role(
        policy, sgd, obs, action, logp, torch.zeros(n), torch.as_tensor(returns, dtype=torch.float32),
        torch.as_tensor(old_values, dtype=torch.float32), critic_only=critic_only,
    )
    return metrics["v_loss"]


def test_critic_switch_defaults_are_todays_loss() -> None:
    """[RED] The defaults keep the clipped value loss and raw targets; the momentum default is 0.95."""
    ppo = DEFAULT_CONFIG.ppo
    assert (ppo.value_clip, ppo.value_norm, ppo.value_norm_momentum) == (True, False, 0.95)


def test_value_norm_momentum_rejects_zero() -> None:
    """[RED] Momentum 0 would discard all history; the error names the field."""
    with pytest.raises(ValueError, match="value_norm_momentum"):
        PPOHyperparams(value_norm_momentum=0.0)


def test_value_norm_momentum_rejects_one() -> None:
    """[RED] Momentum 1 would never learn a statistic; the error names the field."""
    with pytest.raises(ValueError, match="value_norm_momentum"):
        PPOHyperparams(value_norm_momentum=1.0)


def test_value_norm_momentum_rejects_nan() -> None:
    """[RED] NaN momentum is refused with an error naming the field."""
    with pytest.raises(ValueError, match="value_norm_momentum"):
        PPOHyperparams(value_norm_momentum=float("nan"))


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("inf"), float("-inf")])
def test_value_norm_momentum_rejects_values_outside_the_open_unit_interval(bad: float) -> None:
    """[RED] Negative, above-one and infinite momenta are refused."""
    with pytest.raises(ValueError, match="value_norm_momentum"):
        PPOHyperparams(value_norm_momentum=bad)


@pytest.mark.parametrize("good", [0.01, 0.5, 0.99])
def test_value_norm_momentum_accepts_interior_values(good: float) -> None:
    """[RED] Any momentum strictly inside (0, 1) is accepted."""
    assert PPOHyperparams(value_norm_momentum=good).value_norm_momentum == good


def test_critic_switches_are_reachable_through_config_overrides() -> None:
    """[RED] `ppo.value_clip=false` and `ppo.value_norm=true` parse to False / True; the momentum parses as a float."""
    cfg, applied = tp.parse_config_overrides(
        ["ppo.value_clip=false", "ppo.value_norm=true", "ppo.value_norm_momentum=0.9"]
    )
    assert (cfg.ppo.value_clip, cfg.ppo.value_norm, cfg.ppo.value_norm_momentum) == (False, True, 0.9)
    assert applied == {"ppo.value_clip": False, "ppo.value_norm": True, "ppo.value_norm_momentum": 0.9}
    with pytest.raises(ValueError, match="value_norm_momentum"):
        tp.parse_config_overrides(["ppo.value_norm_momentum=1.0"])


def test_running_stat_first_batch_sets_mean_and_population_variance() -> None:
    """[RED] Batch [1, 2, 3]: mean 2, population variance (1 + 0 + 1) / 3 = 2/3, std sqrt(2/3) = 0.8164966."""
    stat = tp.RunningStat(0.5)
    assert (stat.mean, stat.std, stat.count) == (0.0, 1.0, 0)
    stat.update([1.0, 2.0, 3.0])
    assert stat.mean == pytest.approx(2.0, abs=1e-7)
    assert stat.var == pytest.approx(2.0 / 3.0, abs=1e-7)
    assert stat.std == pytest.approx(0.8164966, abs=1e-7)
    assert stat.count == 1


def test_running_stat_blends_first_and_second_moments_with_weight_one_minus_momentum() -> None:
    """[RED] After [1, 2, 3], batch [4] at m = 0.5: mean 0.5*2 + 0.5*4 = 3; second moment 0.5*(2/3 + 4) + 0.5*16 = 10.3333; var 10.3333 - 9 = 1.3333."""
    stat = tp.RunningStat(0.5)
    stat.update([1.0, 2.0, 3.0])
    stat.update([4.0])
    assert stat.mean == pytest.approx(3.0, abs=1e-7)
    assert stat.var == pytest.approx(4.0 / 3.0, abs=1e-7)
    assert stat.count == 2


def test_running_stat_weights_are_not_symmetric() -> None:
    """[RED] After [1, 2, 3], batch [4] at m = 0.75: mean 0.75*2 + 0.25*4 = 2.5; second moment 0.75*(2/3 + 4) + 0.25*16 = 7.5; var 7.5 - 6.25 = 1.25.

    At m = 0.5 swapping m and 1 - m changes nothing, so this case is the one that tells them apart.
    """
    stat = tp.RunningStat(0.75)
    stat.update([1.0, 2.0, 3.0])
    stat.update([4.0])
    assert stat.mean == pytest.approx(2.5, abs=1e-7)
    assert stat.var == pytest.approx(1.25, abs=1e-7)


def test_running_stat_ignores_empty_batches() -> None:
    """[RED] An empty batch changes nothing, fresh or not."""
    stat = tp.RunningStat(0.5)
    stat.update([])
    assert (stat.mean, stat.var, stat.count) == (0.0, 1.0, 0)
    stat.update([1.0, 2.0, 3.0])
    stat.update(np.zeros(0))
    assert stat.mean == pytest.approx(2.0, abs=1e-7)
    assert stat.count == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_running_stat_rejects_non_finite_batches_and_keeps_its_state(bad: float) -> None:
    """[RED] A batch holding NaN or an infinity raises ValueError and leaves the statistic untouched."""
    stat = tp.RunningStat(0.5)
    stat.update([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        stat.update([1.0, bad])
    assert stat.mean == pytest.approx(2.0, abs=1e-7) and stat.count == 1


def test_running_stat_std_never_drops_below_one_thousandth() -> None:
    """[RED] A constant batch [5, 5, 5] has variance 25 - 25 = 0, so std is the 1e-3 floor and 5 normalises to 0."""
    stat = tp.RunningStat(0.5)
    stat.update([5.0, 5.0, 5.0])
    assert stat.var == 0.0 and stat.std == 1e-3
    assert float(stat.normalize(torch.tensor(5.0))) == 0.0
    assert float(stat.normalize(torch.tensor(5.001))) == pytest.approx(1.0, abs=1e-4)


def test_running_stat_normalise_and_denormalise_invert_each_other() -> None:
    """[RED] After [1, 2, 3] (mean 2, std 0.8164966): 3 -> 1.2247449, and 1.2247449 -> 3."""
    stat = tp.RunningStat(0.5)
    stat.update([1.0, 2.0, 3.0])
    assert float(stat.normalize(torch.tensor(3.0, dtype=torch.float64))) == pytest.approx(1.2247449, abs=1e-7)
    assert float(stat.denormalize(torch.tensor(1.2247449, dtype=torch.float64))) == pytest.approx(3.0, abs=1e-6)


@pytest.mark.parametrize("critic_only", [False, True])
@pytest.mark.parametrize(("value_clip", "expected"), [(True, 0.44), (False, 1.0 / 3.0)])
def test_value_loss_on_the_three_sample_hand_case(tmp_path: Path, critic_only: bool, value_clip: bool, expected: float) -> None:
    """[RED] v = (0, 1, 2), old v = (0, 0, 0), R = (1, 1, 1), clip 0.2, in both paths.

    Clipped: 0.5 * mean(max(1, 1), max(0, 0.64), max(1, 0.64)) = 0.5 * 2.64 / 3 = 0.44 (the clipped value is
    0, 0.2, 0.2 for the three samples).  Unclipped: 0.5 * (1 + 0 + 1) / 3 = 1/3.
    """
    trainer = _critic_trainer(tmp_path, rollout_steps=8, seed=3, full_batch=True, value_clip=value_clip)
    try:
        policy = trainer.blue_field_policy
        obs = _critic_staircase(policy)
        with torch.no_grad():
            assert policy.get_value(obs).tolist() == pytest.approx([0.0, 1.0, 2.0], abs=1e-6)
        loss = _critic_update(trainer, policy, obs, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], critic_only)
        assert loss == pytest.approx(expected, abs=1e-6)
    finally:
        trainer.close()


@pytest.mark.parametrize("critic_only", [False, True])
@pytest.mark.parametrize(("value_clip", "expected"), [(True, 0.7225), (False, 0.625)])
def test_value_norm_loss_is_measured_in_normalised_units_with_the_batch_statistics(
    tmp_path: Path, critic_only: bool, value_clip: bool, expected: float
) -> None:
    """[RED] Raw returns (0, 3), raw old values (1.5, 1.5), head output 0.5 for both samples, clip 0.2.

    The batch sets mean 1.5 and variance (0 + 9) / 2 - 2.25 = 2.25, so std 1.5 and the normalised returns are
    (-1, +1) and the normalised old values (0, 0). Unclipped: 0.5 * mean(1.5^2, 0.5^2) = 0.5 * 2.5 / 2 = 0.625.
    Clipped: the clipped value is 0 + 0.2 = 0.2, so the clipped squares are 1.2^2 = 1.44 and 0.8^2 = 0.64 and the
    loss is 0.5 * mean(max(2.25, 1.44), max(0.25, 0.64)) = 0.5 * (2.25 + 0.64) / 2 = 0.7225. The statistics are
    those of the batch itself, so they must be in place before the loss is taken.
    """
    trainer = _critic_trainer(tmp_path, rollout_steps=8, seed=3, full_batch=True, value_norm=True, value_clip=value_clip)
    try:
        policy = trainer.blue_field_policy
        _constant_head_critic(policy, 0.5)
        loss = _critic_update(trainer, policy, torch.zeros(2, OBS_DIM), [1.5, 1.5], [0.0, 3.0], critic_only)
        assert loss == pytest.approx(expected, abs=1e-6)
        assert policy.return_stats() == pytest.approx((1.5, 1.5), abs=1e-7)
    finally:
        trainer.close()


def test_value_norm_statistics_blend_across_updates_with_the_configured_momentum(tmp_path: Path) -> None:
    """[RED] Returns (0, 3) then (4, 4) at m = 0.75: mean 0.75*1.5 + 0.25*4 = 2.125; second moment 0.75*(2.25 + 2.25) + 0.25*16 = 7.375; var 7.375 - 4.515625 = 2.859375; std 1.6909.

    Swapping m and 1 - m would give mean 0.25*1.5 + 0.75*4 = 3.375 instead.
    """
    trainer = _critic_trainer(tmp_path, rollout_steps=8, seed=3, full_batch=True, value_norm=True, value_norm_momentum=0.75)
    try:
        policy = trainer.blue_field_policy
        _critic_update(trainer, policy, torch.zeros(2, OBS_DIM), [1.5, 1.5], [0.0, 3.0], False)
        _critic_update(trainer, policy, torch.zeros(2, OBS_DIM), [2.0, 2.0], [4.0, 4.0], False)
        mean, std = policy.return_stats()
        assert mean == pytest.approx(2.125, abs=1e-6)
        # sqrt(2.859375) = 1.6909686573085854. The std reaches this value through the float32 `ret_std` buffer, whose
        # rounding error is at most half an ulp (6e-8 in [1, 2)), so the bound is one float32 ulp, 1.2e-7, not the 1e-7 of float64.
        assert std == pytest.approx(1.6909686573085854, abs=1.2e-7)
    finally:
        trainer.close()


def test_value_norm_off_keeps_the_critic_free_of_statistics_and_the_loss_in_raw_units(tmp_path: Path) -> None:
    """[GUARD] value_norm off: no `ret_*` entries in any state dict, and with head 1 against returns (0, 3) the unclipped loss is 0.5 * mean(1, 4) = 1.25 in raw units."""
    trainer = _critic_trainer(tmp_path, rollout_steps=8, seed=3, full_batch=True, value_clip=False)
    try:
        for _, policy, _ in _policy_groups(trainer):
            assert not any(k.startswith("ret_") for k in policy.state_dict())
            assert policy.value_norm is False and policy.return_stats() == (0.0, 1.0)
        policy = trainer.blue_field_policy
        _constant_head_critic(policy, 1.0)
        loss = _critic_update(trainer, policy, torch.zeros(2, OBS_DIM), [0.0, 0.0], [0.0, 3.0], False)
        assert loss == pytest.approx(1.25, abs=1e-6)
    finally:
        trainer.close()


@pytest.mark.parametrize("head", ["gaussian4", "hybrid5"])
def test_every_value_output_of_the_network_is_in_raw_units(head: str) -> None:
    """[RED] Head output 0.5, ret_mean 7, ret_std 2.5: every raw-unit output is 0.5 * 2.5 + 7 = 8.25; the normalised view is 0.5."""
    net = ActorCritic(policy_head=head, value_norm=True)
    _constant_head_critic(net, 0.5)
    _set_return_stats(net, 7.0, 2.5)
    obs = torch.randn(4, OBS_DIM, generator=torch.Generator().manual_seed(1))
    cen = torch.randn(4, CENTRAL_STATE_DIM, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        assert net.get_value(obs, cen).tolist() == [8.25] * 4
        assert net.get_value(obs, cen, normalized_value=True).tolist() == [0.5] * 4
        assert net.get_action_and_value(obs, central_state=cen)[3].tolist() == [8.25] * 4
        assert net.get_action_and_value(obs, central_state=cen, normalized_value=True)[3].tolist() == [0.5] * 4
        assert net.sample_action_and_value(obs, central_state=cen)[2].tolist() == [8.25] * 4
        assert net.sample_action_and_value(obs, central_state=cen, return_mean=True)[2].tolist() == [8.25] * 4
        if head == "hybrid5":
            assert net.hybrid_logits_and_value(obs, central_state=cen)[1].tolist() == [8.25] * 4


@pytest.mark.parametrize("head", ["gaussian4", "hybrid5"])
def test_default_network_has_no_statistics_and_values_are_the_raw_critic_output(head: str) -> None:
    """[GUARD] Without value_norm: no `ret_*` entries, and `get_value` is bit-identical to the critic's output."""
    net = _seeded_net(head, 4)
    assert not any(k.startswith("ret_") for k in net.state_dict())
    obs = torch.randn(5, OBS_DIM, generator=torch.Generator().manual_seed(1))
    cen = torch.randn(5, CENTRAL_STATE_DIM, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        assert torch.equal(net.get_value(obs, cen), net.critic(torch.cat([obs, cen], dim=-1)).squeeze(-1))


def test_a_network_loading_statistics_it_was_not_built_with_picks_them_up() -> None:
    """[RED] A plain `ActorCritic` loaded (strictly) from a state dict with `ret_mean` 7 and `ret_std` 2.5 returns raw values 8.25, as the harness loads."""
    saved = ActorCritic(value_norm=True)
    _constant_head_critic(saved, 0.5)
    _set_return_stats(saved, 7.0, 2.5)
    loaded = ActorCritic()
    loaded.load_state_dict(tp.adapt_policy_weights(saved.state_dict(), loaded.obs_dim, loaded.central_state_dim))
    assert loaded.value_norm is True and loaded.return_stats() == (7.0, 2.5)
    assert loaded.get_value(torch.zeros(3, OBS_DIM)).tolist() == [8.25] * 3


def test_adapt_policy_weights_passes_the_statistics_through() -> None:
    """[RED] The adapted state dict holds the very same `ret_mean` / `ret_std` tensors."""
    saved = ActorCritic(value_norm=True)
    _set_return_stats(saved, 7.0, 2.5)
    weights = saved.state_dict()
    adapted = tp.adapt_policy_weights(weights, OBS_DIM, CENTRAL_STATE_DIM)
    assert adapted["ret_mean"] is weights["ret_mean"] and adapted["ret_std"] is weights["ret_std"]


@pytest.mark.parametrize("head", ["gaussian4", "hybrid5"])
def test_value_norm_trainer_reaches_the_constant_return_in_raw_units(tmp_path: Path, head: str) -> None:
    """[RED] A constant reward of 1 per step that never ends, at gamma 0.9, has the fixed point V = 1 + 0.9 V, so V = 10.

    After 30 updates the critic's raw `get_value` is within 5% of 10 on every observation of the last rollout while
    its normalised head output is that raw value in units of the running spread and the last batch's normalised targets obey the
    mixture bound derived in the body, the saved policy files carry the statistics, and a
    network loaded the way the harness loads it returns the trainer's values to 1e-6. A rollout or bootstrap that
    read the normalised head would see values near 0 and never get there.
    """
    # Two minibatches per epoch (default 8): the same 30 updates in half the time; the critic still lands in the band.
    trainer = _critic_trainer(
        tmp_path, head=head, rollout_steps=64, seed=11, value_norm=True, value_norm_momentum=0.5, gamma=0.9,
        num_minibatches=2,
    )
    try:
        real_step = trainer.vec_env.step_array

        def constant_reward_step(actions, values=None, action_means=None):
            obs, cen, rew, done, summaries = real_step(actions, values, action_means=action_means)
            return obs, cen, np.full_like(rew, 1.0), done, summaries

        trainer.vec_env.step_array = constant_reward_step
        for gen in range(1, 31):
            rollout = trainer.collect_rollouts(active_generation=gen)
            trainer.update_policies(rollout)
        trainer.current_generation = 30
        trainer.save_checkpoints()

        for role, policy, indices in (
            ("blue_field", trainer.blue_field_policy, trainer.blue_field_indices),
            ("blue_keeper", trainer.blue_keeper_policy, trainer.blue_keeper_indices),
        ):
            obs = torch.as_tensor(rollout["obs"][:, :, indices].reshape(-1, OBS_DIM))
            cen = torch.as_tensor(rollout["central_state"][:, :, indices].reshape(-1, CENTRAL_STATE_DIM))
            with torch.no_grad():
                raw = policy.get_value(obs, cen)
                head_out = policy.get_value(obs, cen, normalized_value=True)
            assert float(raw.min()) > 9.5 and float(raw.max()) < 10.5, (role, float(raw.min()), float(raw.max()))
            # What the normalised head must look like, derived from the definitions and not from a run.
            # (1) It is the raw value in units of the running spread: head = (raw - ret_mean) / ret_std, to float32 precision.
            #     A head that was left in raw units (~10) or a denormalisation with the wrong statistics breaks this.
            #     Tolerance: raw is a float32 near 10 (ulp 9.5e-7), recovered through a division by ret_std, so
            #     |error| <= 4 * ulp / ret_std = 4e-6 / ret_std.
            mean, std = policy.return_stats()
            np.testing.assert_allclose(head_out.numpy(), ((raw - mean) / std).numpy(), rtol=0, atol=4e-6 / std)
            # (2) The normalised targets of the last batch are bounded. After the update the statistics are the mixture
            #     m * (old batches) + (1 - m) * (this batch) with m = 0.5, whose mean and variance are (ret_mean, ret_std^2).
            #     This batch carries weight (1 - m) of that mixture, so E_batch[(R - mean)^2] <= Var_mixture / (1 - m) and
            #     the root mean square of (R - mean) / std is at most 1 / sqrt(1 - m) = sqrt(2) = 1.41421356 (the 1e-3 std
            #     floor only lowers it). Float32 rounding of the buffers: relative 1e-5.
            returns = torch.as_tensor(rollout["returns"][:, :, indices].reshape(-1)).float()
            normalised_targets = (returns - mean) / std
            assert float(normalised_targets.pow(2).mean().sqrt()) <= math.sqrt(2.0) * (1 + 1e-5), role
            # (3) The head is NOT near 0 in general. An exact critic (V = 10 everywhere) reads (10 - ret_mean) / ret_std,
            #     and ret_std (the spread of the targets over recent batches, 1e-1 .. 1e-3 here) is far smaller than the
            #     lag of ret_mean behind 10, so the head sits a few ret_std from 0. Its size is therefore not asserted;
            #     it is the raw band above and the identity in (1) that pin it.
            for suffix in ("policy.pt",):
                weights = torch.load(tmp_path / "ckpt" / f"{role}_{suffix}", weights_only=True)
                assert float(weights["ret_mean"]) == pytest.approx(policy.return_stats()[0], abs=1e-6)
                assert float(weights["ret_std"]) == pytest.approx(policy.return_stats()[1], abs=1e-6)
                loaded = ActorCritic(policy_head=head)
                loaded.load_state_dict(tp.adapt_policy_weights(weights, loaded.obs_dim, loaded.central_state_dim))
                with torch.no_grad():
                    np.testing.assert_allclose(loaded.get_value(obs, cen).numpy(), raw.numpy(), atol=1e-6)
    finally:
        trainer.close()


def _assert_same_training_state(a: ParameterSharingPPOTrainer, b: ParameterSharingPPOTrainer) -> None:
    """Every parameter, buffer and Adam tensor of all four roles is bit-identical between two trainers."""
    for (role, pa, oa), (_, pb, ob) in zip(_policy_groups(a), _policy_groups(b)):
        sa, sb = pa.state_dict(), pb.state_dict()
        assert list(sa) == list(sb), role
        for key in sa:
            assert torch.equal(sa[key], sb[key]), (role, key)
        for (_, qa), (_, qb) in zip(pa.named_parameters(), pb.named_parameters()):
            for key in ("exp_avg", "exp_avg_sq", "step"):
                assert torch.equal(oa.state[qa][key], ob.state[qb][key]), (role, key)


def test_save_load_then_one_more_update_equals_the_uninterrupted_run(tmp_path: Path) -> None:
    """[RED] Two rollouts r1, r2 from the same start; update(r1) -> save -> load -> update(r2) matches update(r1), update(r2) bit for bit.

    The buffers take part: a checkpoint that dropped them would restart the second update from identity statistics.
    """
    ppo = dict(value_norm=True, value_clip=False)
    straight = _critic_trainer(tmp_path, "straight", **ppo)
    first = _critic_trainer(tmp_path, "resumed", **ppo)
    second = None
    try:
        r1, r2 = straight.collect_rollouts(1), straight.collect_rollouts(2)
        torch.manual_seed(100)
        straight.update_policies(r1)
        torch.manual_seed(101)
        straight.update_policies(r2)

        torch.manual_seed(100)
        first.update_policies(r1)
        first.current_generation = 1
        first.save_checkpoints()
        second = _critic_trainer(tmp_path, "resumed", **ppo)
        assert second.load_checkpoints()
        torch.manual_seed(101)
        second.update_policies(r2)

        assert not torch.equal(straight.blue_field_policy.ret_mean, torch.tensor(0.0))  # the statistics did move
        _assert_same_training_state(straight, second)
    finally:
        for t in (straight, first, second):
            if t is not None:
                t.close()


def test_a_checkpoint_without_statistics_loads_with_identity_statistics(tmp_path: Path) -> None:
    """[RED] A state-file checkpoint and a bare policy-file checkpoint saved with value_norm off load into a value_norm trainer as ret_mean 0, ret_std 1."""
    legacy = _critic_trainer(tmp_path, "legacy")
    try:
        legacy.update_policies(legacy.collect_rollouts(1))
        legacy.current_generation = 1
        legacy.save_checkpoints()
        weights = legacy.blue_field_policy.state_dict()
        assert not any(k.startswith("ret_") for k in weights)
    finally:
        legacy.close()

    for remove_state_file in (False, True):
        folder = tmp_path / ("bare" if remove_state_file else "legacy")
        if remove_state_file:
            shutil.copytree(tmp_path / "legacy", folder)
            (folder / "training_state.pt").unlink()
        fresh = _critic_trainer(tmp_path, folder.name, value_norm=True)
        try:
            assert fresh.load_checkpoints()
            for _, policy, _ in _policy_groups(fresh):
                assert policy.value_norm is True
                assert policy.return_stats() == (0.0, 1.0)
            for key, value in weights.items():
                assert torch.equal(fresh.blue_field_policy.state_dict()[key], value), key
        finally:
            fresh.close()


def test_a_checkpoint_with_statistics_is_refused_when_value_norm_is_off(tmp_path: Path) -> None:
    """[RED] Loading return statistics into a trainer that does not normalise raises, naming value_norm; the weights would be read in the wrong units."""
    saved = _critic_trainer(tmp_path, "norm", value_norm=True)
    try:
        saved.update_policies(saved.collect_rollouts(1))
        saved.current_generation = 1
        saved.save_checkpoints()
    finally:
        saved.close()
    plain = _critic_trainer(tmp_path, "norm")
    try:
        with pytest.raises(ValueError, match="value_norm"):
            plain.load_checkpoints()
    finally:
        plain.close()


def test_saved_policy_files_carry_the_statistics_of_their_role(tmp_path: Path) -> None:
    """[RED] All six policy files hold `ret_mean` / `ret_std` equal to the live buffers; the team files match the shared-name files."""
    trainer = _critic_trainer(tmp_path, value_norm=True)
    try:
        trainer.update_policies(trainer.collect_rollouts(1))
        trainer.current_generation = 1
        trainer.save_checkpoints()
        live = {
            "blue_field": trainer.blue_field_policy, "blue_keeper": trainer.blue_keeper_policy,
            "red_field": trainer.red_field_policy, "red_keeper": trainer.red_keeper_policy,
            "field": trainer.blue_field_policy, "keeper": trainer.blue_keeper_policy,
        }
        for name, policy in live.items():
            weights = torch.load(tmp_path / "ckpt" / f"{name}_policy.pt", weights_only=True)
            assert torch.equal(weights["ret_mean"], policy.ret_mean) and torch.equal(weights["ret_std"], policy.ret_std), name
        assert trainer.blue_field_policy.return_stats() != trainer.blue_keeper_policy.return_stats()
    finally:
        trainer.close()


def test_generation_records_carry_the_statistics_per_role_only_with_value_norm(tmp_path: Path) -> None:
    """[RED] With value_norm the record gains `ret_mean` and `ret_std` for the four roles, equal to the live buffers; without it neither key exists."""
    on = _critic_trainer(tmp_path, "on", rollout_steps=10, value_norm=True)
    off = _critic_trainer(tmp_path, "off", rollout_steps=10)
    try:
        on.train(total_updates=1)
        off.train(total_updates=1)
        rec_on, rec_off = on.generation_history[0], off.generation_history[0]
        roles = {"blue_field": on.blue_field_policy, "blue_keeper": on.blue_keeper_policy,
                 "red_field": on.red_field_policy, "red_keeper": on.red_keeper_policy}
        assert set(rec_on["ret_mean"]) == set(rec_on["ret_std"]) == set(roles)
        for role, policy in roles.items():
            assert rec_on["ret_mean"][role] == pytest.approx(policy.return_stats()[0], abs=1e-6)
            assert rec_on["ret_std"][role] == pytest.approx(policy.return_stats()[1], abs=1e-6)
        assert rec_on["ret_std"]["blue_field"] != 1.0
        assert "ret_mean" not in rec_off and "ret_std" not in rec_off
        assert set(rec_on) - {"ret_mean", "ret_std"} == set(rec_off)
    finally:
        on.close()
        off.close()


def test_shared_teams_share_the_statistics(tmp_path: Path) -> None:
    """[RED] With one policy per role for both teams, blue and red read the same buffers and the record repeats them."""
    trainer = _critic_trainer(tmp_path, rollout_steps=10, separate_teams=False, value_norm=True)
    try:
        trainer.train(total_updates=1)
        assert trainer.red_field_policy is trainer.blue_field_policy
        rec = trainer.generation_history[0]
        assert rec["ret_mean"]["red_field"] == rec["ret_mean"]["blue_field"]
        assert rec["ret_std"]["red_keeper"] == rec["ret_std"]["blue_keeper"] != 1.0
    finally:
        trainer.close()


def test_frozen_seat_values_are_in_raw_units() -> None:
    """[RED] A frozen network with ret_mean 7, ret_std 2.5 and head output 0.5 reports 8.25 for every seat of its group."""
    net = _seeded_net("gaussian4", 3)
    net.enable_value_norm()
    _constant_head_critic(net, 0.5)
    _set_return_stats(net, 7.0, 2.5)
    groups = [(np.array([0, 2]), [1, 2, 3, 4, 5], net)]
    rng = np.random.default_rng(5)
    obs = rng.normal(size=(3, 12, OBS_DIM)).astype(np.float32)
    cen = rng.normal(size=(3, 12, CENTRAL_STATE_DIM)).astype(np.float32)
    actions = np.full((3, 12, 4), np.nan, dtype=np.float32)
    logprobs = np.full((3, 12), np.nan, dtype=np.float32)
    values = np.full((3, 12), np.nan, dtype=np.float32)
    tp.FrozenSeatBatch(groups).apply(obs, cen, tp.pool_generator(1, 2), actions, logprobs, values, None)
    np.testing.assert_allclose(values[np.ix_([0, 2], [1, 2, 3, 4, 5])], 8.25, atol=1e-5)
    assert np.isnan(values[1]).all()


def test_pool_member_with_statistics_is_loaded_in_raw_units_by_a_plain_trainer(tmp_path: Path) -> None:
    """[RED] A snapshot saved with return statistics loads as a pool member into a trainer with value_norm off, keeping them; a snapshot without them gets none."""
    trainer = _critic_trainer(tmp_path, "pool", num_envs=2)
    try:
        snap = Path(trainer.checkpoint_dir) / "snapshots" / "gen0050"
        snap.mkdir(parents=True)
        for team in ("blue", "red"):
            for role in ("keeper", "field"):
                net = ActorCritic(value_norm=(team == "blue"))
                if team == "blue":
                    _set_return_stats(net, 7.0, 2.5)
                torch.save(net.state_dict(), snap / f"{team}_{role}_policy.pt")
        keeper, field = trainer._load_pool_member(50, "A")
        assert keeper.return_stats() == (7.0, 2.5) and field.return_stats() == (7.0, 2.5)
        keeper, field = trainer._load_pool_member(50, "B")
        assert keeper.value_norm is False and keeper.return_stats() == (0.0, 1.0)
    finally:
        trainer.close()


def test_the_harness_returns_raw_values_for_a_checkpoint_with_statistics(tmp_path: Path) -> None:
    """[RED] Policy files saved with ret_mean 7, ret_std 2.5 and head output 0.5: `TeamPolicy.act` reports value 8.25 for all six slots, bit-identical to `get_value`."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.audit import harness

    for prefix in ("blue", "red"):
        for role in ("keeper", "field"):
            net = _seeded_net("gaussian4", 9)
            net.enable_value_norm()
            _constant_head_critic(net, 0.5)
            _set_return_stats(net, 7.0, 2.5)
            torch.save(net.state_dict(), tmp_path / f"{prefix}_{role}_policy.pt")
    team = harness.TeamPolicy(str(tmp_path), "blue")
    obs = np.random.default_rng(1).normal(size=(6, OBS_DIM)).astype(np.float32)
    cen = np.random.default_rng(2).normal(size=(6, CENTRAL_STATE_DIM)).astype(np.float32)
    _, values, _ = team.act(obs, cen, "greedy", 1.0, torch.Generator().manual_seed(0))
    assert values.tolist() == [8.25] * 6

    hybrid_dir = tmp_path / "hybrid"
    hybrid_dir.mkdir()
    for prefix in ("blue", "red"):
        for role in ("keeper", "field"):
            net = _seeded_net("hybrid5", 9)
            net.enable_value_norm()
            _constant_head_critic(net, 0.5)
            _set_return_stats(net, 7.0, 2.5)
            torch.save(net.state_dict(), hybrid_dir / f"{prefix}_{role}_policy.pt")
    _, values, _ = harness.TeamPolicy(str(hybrid_dir), "blue").act(obs, cen, "greedy", 1.0, torch.Generator().manual_seed(0))
    assert values.tolist() == [8.25] * 6


# ------------------------------------------------------------------ critic switches: sync, statistics bookkeeping
_HEAD_PROBE_HIDDEN = math.atanh(0.5)  # critic hidden unit 0 outputs tanh(atanh(0.5)) = 0.5 when everything before it is silent


def _head_probe_pair(head: str):
    """Two value-normalising networks with identical hidden layers and different last critic layers and statistics.

    Every critic weight is zero except unit 0 of the second hidden layer, whose bias makes it output 0.5 for any input.
    The head output is then `W * 0.5 + b`, with
      dst: W = 2.0,  b = 1.0, ret_mean 10, ret_std 4  -> head 2.0, raw value 2.0 * 4 + 10 = 18;
      src: W = -1.0, b = 1.5, ret_mean -3, ret_std 2  -> head 1.0, raw value 1.0 * 2 - 3 = -1.
    """
    dst = _seeded_net(head, 21)
    dst.enable_value_norm()
    with torch.no_grad():
        for layer in (dst.critic[0], dst.critic[2], dst.critic[4]):
            layer.weight.zero_()
            layer.bias.zero_()
        dst.critic[2].bias[0] = _HEAD_PROBE_HIDDEN
    src = _seeded_net(head, 22)
    src.load_state_dict(dst.state_dict())
    with torch.no_grad():
        dst.critic[4].weight[0, 0], dst.critic[4].bias[0] = 2.0, 1.0
        src.critic[4].weight[0, 0], src.critic[4].bias[0] = -1.0, 1.5
    _set_return_stats(dst, 10.0, 4.0)
    _set_return_stats(src, -3.0, 2.0)
    return dst, src


def _raw_value(net: ActorCritic) -> float:
    with torch.no_grad():
        return float(net.get_value(torch.zeros(1, OBS_DIM))[0])


@pytest.mark.parametrize("head", ["gaussian4", "hybrid5"])
def test_soft_blend_with_tau_one_makes_dst_read_the_raw_values_of_src(head: str) -> None:
    """[RED] Blending critic heads that sit in different normalised units must not change what the blend means.

    Hand values (see `_head_probe_pair`): raw dst 18, raw src -1. Re-expressing src in dst's units gives
      W' = W_src * 2 / 4 = -0.5 and b' = (1.5 * 2 + (-3) - 10) / 4 = -2.5, so head' = -0.5 * 0.5 - 2.5 = -2.75 and
      raw = -2.75 * 4 + 10 = -1, as src itself reads. With tau = 1 dst reads -1. (Blending the heads as they stand gives
      W = -1, b = 1.5 -> head 1.0 -> raw 1.0 * 4 + 10 = 14, which is wrong.) dst keeps its statistics (10, 4).
    Tolerance 1e-5: the hidden unit is tanh(atanh(0.5)) in float32 (error ~6e-8) and the raw value is a few float32
    operations at magnitude ~20 (one ulp there is 1.9e-6).
    """
    dst, src = _head_probe_pair(head)
    assert _raw_value(dst) == pytest.approx(18.0, abs=1e-5) and _raw_value(src) == pytest.approx(-1.0, abs=1e-5)
    src_before = copy.deepcopy(src.state_dict())
    ParameterSharingPPOTrainer._soft_blend(dst, src, 1.0)
    assert _raw_value(dst) == pytest.approx(-1.0, abs=1e-5)
    assert dst.return_stats() == (10.0, 4.0)
    for key, value in src.state_dict().items():
        assert torch.equal(value, src_before[key]), key


@pytest.mark.parametrize("head", ["gaussian4", "hybrid5"])
def test_soft_blend_mixes_raw_values_by_tau_and_keeps_the_statistics_of_dst(head: str) -> None:
    """[RED] tau = 0.35: dst reads 0.65 * raw_dst + 0.35 * raw_src = 0.65 * 18 + 0.35 * (-1) = 11.7 - 0.35 = 11.35.

    In dst's units: head = 0.65 * 2.0 + 0.35 * (-2.75) = 1.3 - 0.9625 = 0.3375 and raw = 0.3375 * 4 + 10 = 11.35.
    Blending the heads as they stand would give head 0.65 * 2.0 + 0.35 * 1.0 ... = 1.65 -> raw 16.6. Tolerance 1e-5 as above.
    """
    dst, src = _head_probe_pair(head)
    ParameterSharingPPOTrainer._soft_blend(dst, src, 0.35)
    assert _raw_value(dst) == pytest.approx(11.35, abs=1e-5)
    assert dst.return_stats() == (10.0, 4.0)
    assert src.return_stats() == (-3.0, 2.0)


def test_soft_blend_without_value_norm_is_bit_identical_to_the_plain_parameter_blend() -> None:
    """[GUARD] Default networks: every parameter equals `(1 - tau) * dst + tau * src`, the blend as it was before value_norm existed."""
    for head in ("gaussian4", "hybrid5"):
        dst, src = _seeded_net(head, 31), _seeded_net(head, 32)
        expected = [((1.0 - 0.35) * p_d.detach() + 0.35 * p_s.detach()) for p_d, p_s in zip(dst.parameters(), src.parameters())]
        ParameterSharingPPOTrainer._soft_blend(dst, src, 0.35)
        for got, want in zip(dst.parameters(), expected):
            assert torch.equal(got.detach(), want)


def test_coevolution_sync_keeps_the_raw_critic_values_of_the_trailing_team_blended(tmp_path: Path) -> None:
    """[RED] 6-2 over 24 matches: Team B (red_*) is synced towards Team A (blue_*), field and keeper critics both.

    Hidden layers are made identical, last layers and statistics as in `_head_probe_pair` (blue = src, red = dst), so
    red reads 0.65 * 18 + 0.35 * (-1) = 11.35 afterwards and blue still reads -1. Tolerance 1e-5 as in the unit tests.
    """
    trainer = _critic_trainer(tmp_path, value_norm=True)
    try:
        for red, blue in ((trainer.red_field_policy, trainer.blue_field_policy),
                          (trainer.red_keeper_policy, trainer.blue_keeper_policy)):
            dst, src = _head_probe_pair("gaussian4")
            red.load_state_dict(dst.state_dict())
            blue.load_state_dict(src.state_dict())
        assert trainer._coevolution_step(_matches_with_totals(6, 2, n=24)) is True
        for red, blue in ((trainer.red_field_policy, trainer.blue_field_policy),
                          (trainer.red_keeper_policy, trainer.blue_keeper_policy)):
            assert _raw_value(red) == pytest.approx(11.35, abs=1e-5)
            assert _raw_value(blue) == pytest.approx(-1.0, abs=1e-5)
            assert red.return_stats() == (10.0, 4.0) and blue.return_stats() == (-3.0, 2.0)
    finally:
        trainer.close()


# ------------------------------------------------------------------ critic switches: initialisation is stored, not inferred
def test_a_first_batch_with_mean_zero_and_std_one_is_still_a_first_batch(tmp_path: Path) -> None:
    """[RED] First returns (-1, 1): mean 0, population variance 1, i.e. exactly the identity statistics. Then returns (4, 4) at m = 0.5.

    The second batch must blend, not overwrite: mean = 0.5 * 0 + 0.5 * 4 = 2; second moment = 0.5 * (1 + 0) + 0.5 * 16 = 8.5;
    var = 8.5 - 2^2 = 4.5, std = sqrt(4.5) = 2.1213203. Treating (0, 1) as "never updated" would overwrite to mean 4, std 1e-3.
    Tolerance: the std goes through a float32 buffer (half an ulp at 2.12 is 1.2e-7), so 2.5e-7.
    """
    trainer = _critic_trainer(tmp_path, rollout_steps=8, seed=3, full_batch=True, value_norm=True, value_norm_momentum=0.5)
    try:
        policy = trainer.blue_field_policy
        _critic_update(trainer, policy, torch.zeros(2, OBS_DIM), [0.0, 0.0], [-1.0, 1.0], False)
        assert policy.return_stats() == (0.0, 1.0)
        _critic_update(trainer, policy, torch.zeros(2, OBS_DIM), [0.0, 0.0], [4.0, 4.0], False)
        mean, std = policy.return_stats()
        assert mean == pytest.approx(2.0, abs=1e-6)
        assert std == pytest.approx(2.1213203435596424, abs=2.5e-7)
    finally:
        trainer.close()


def test_the_update_count_is_a_saved_buffer_and_survives_a_resume(tmp_path: Path) -> None:
    """[RED] After one update the buffer `ret_count` is 1 in the network and in the saved policy file (0 before any update).

    A resumed trainer whose first batch was (-1, 1) must then blend the next batch (4, 4) at m = 0.5 to mean 2.0
    (hand values in the test above); losing the count would restart it as a first batch and give mean 4.0.
    """
    first = _critic_trainer(tmp_path, "resume_count", rollout_steps=8, seed=3, full_batch=True, value_norm=True, value_norm_momentum=0.5)
    second = None
    try:
        policy = first.blue_field_policy
        assert int(policy.ret_count) == 0 and "ret_count" in policy.state_dict()
        _critic_update(first, policy, torch.zeros(2, OBS_DIM), [0.0, 0.0], [-1.0, 1.0], False)
        assert int(policy.ret_count) == 1
        first.current_generation = 1
        first.save_checkpoints()
        weights = torch.load(tmp_path / "resume_count" / "blue_field_policy.pt", weights_only=True)
        assert int(weights["ret_count"]) == 1
        second = _critic_trainer(tmp_path, "resume_count", rollout_steps=8, seed=3, full_batch=True, value_norm=True, value_norm_momentum=0.5)
        assert second.load_checkpoints()
        assert int(second.blue_field_policy.ret_count) == 1
        _critic_update(second, second.blue_field_policy, torch.zeros(2, OBS_DIM), [0.0, 0.0], [4.0, 4.0], False)
        assert second.blue_field_policy.return_stats()[0] == pytest.approx(2.0, abs=1e-6)
        assert int(second.blue_field_policy.ret_count) == 2
    finally:
        first.close()
        if second is not None:
            second.close()


# ------------------------------------------------------------------ critic switches: the statistics load as a complete set
_STAT_KEYS = ("ret_mean", "ret_std", "ret_count")


def _partial_statistics(missing: str):
    saved = ActorCritic(value_norm=True)
    _set_return_stats(saved, 7.0, 2.5)
    weights = dict(saved.state_dict())
    del weights[missing]
    return weights


@pytest.mark.parametrize("missing", _STAT_KEYS)
def test_a_network_refuses_a_state_dict_with_only_some_of_the_statistics(missing: str) -> None:
    """[RED] A plain `ActorCritic` loading a state dict that holds two of the three statistics raises ValueError naming the missing key."""
    weights = _partial_statistics(missing)
    with pytest.raises(ValueError, match=missing):
        ActorCritic().load_state_dict(weights)


@pytest.mark.parametrize("missing", _STAT_KEYS)
@pytest.mark.parametrize("value_norm", [False, True])
def test_the_trainer_refuses_a_checkpoint_with_only_some_of_the_statistics(tmp_path: Path, missing: str, value_norm: bool) -> None:
    """[RED] `_load_adapted_weights` raises ValueError naming the missing key, whether or not the trainer normalises values."""
    trainer = _critic_trainer(tmp_path, value_norm=value_norm)
    try:
        with pytest.raises(ValueError, match=missing):
            trainer._load_adapted_weights(trainer.blue_field_policy, _partial_statistics(missing))
    finally:
        trainer.close()


@pytest.mark.parametrize("missing", _STAT_KEYS)
def test_a_pool_snapshot_with_only_some_of_the_statistics_is_refused(tmp_path: Path, missing: str) -> None:
    """[RED] `_load_pool_member` raises ValueError naming the missing key instead of loading a half-described critic."""
    trainer = _critic_trainer(tmp_path, "pool_partial", num_envs=2)
    try:
        snap = Path(trainer.checkpoint_dir) / "snapshots" / "gen0050"
        snap.mkdir(parents=True)
        for team in ("blue", "red"):
            for role in ("keeper", "field"):
                torch.save(_partial_statistics(missing), snap / f"{team}_{role}_policy.pt")
        with pytest.raises(ValueError, match=missing):
            trainer._load_pool_member(50, "A")
    finally:
        trainer.close()


# --------------------------------------------------------------------------------------
# Replay cap (--replay-every, --replay-kind)
# --------------------------------------------------------------------------------------


def _replay_cap_cfg(save_dir: Path, max_rl_steps: int = 10, save_best_only: bool = True):
    return replace(
        DEFAULT_CONFIG,
        physics=replace(DEFAULT_CONFIG.physics, max_rl_steps=max_rl_steps),
        replay=replace(
            DEFAULT_CONFIG.replay,
            save_dir=str(save_dir),
            enable_recording=True,
            interest_threshold=0.0,
            save_best_only=save_best_only,
        ),
    )


def _replay_cap_trainer(
    tmp_path: Path,
    name: str = "cap",
    num_envs: int = 2,
    max_rl_steps: int = 10,
    rollout_steps: int = 10,
    save_best_only: bool = True,
    seed: int = 7,
    **kwargs,
) -> ParameterSharingPPOTrainer:
    rep_dir = tmp_path / f"rep_{name}"
    ckpt_dir = tmp_path / f"ckpt_{name}"
    return ParameterSharingPPOTrainer(
        config=_replay_cap_cfg(rep_dir, max_rl_steps=max_rl_steps, save_best_only=save_best_only),
        num_envs=num_envs,
        rollout_steps=rollout_steps,
        device="cpu",
        checkpoint_dir=str(ckpt_dir),
        seed=seed,
        save_best_only=save_best_only,
        **kwargs,
    )


def _replay_names_by_gen(rep_dir: Path) -> dict[int, list[str]]:
    by_gen: dict[int, list[str]] = {}
    for p in sorted(rep_dir.glob("match_*.json")):
        m = re.search(r"_gen(\d+)_", p.name)
        assert m is not None, p.name
        by_gen.setdefault(int(m.group(1)), []).append(p.name)
    return by_gen


def test_replay_cap_defaults_write_both_kinds_every_generation(tmp_path: Path) -> None:
    """[GUARD] Defaults: a 3-generation run writes both best and uniform replays in every generation."""
    trainer = _replay_cap_trainer(tmp_path, name="guard", num_envs=2)
    try:
        trainer.train(total_updates=3)
    finally:
        trainer.close()

    by_gen = _replay_names_by_gen(tmp_path / "rep_guard")
    assert sorted(by_gen.keys()) == [1, 2, 3]
    for gen in (1, 2, 3):
        names = by_gen[gen]
        uniform_names = [n for n in names if "_uniform_" in n]
        best_names = [n for n in names if "_uniform_" not in n]
        assert len(uniform_names) == 1 and len(best_names) == 1
        assert re.match(rf"^match_\d{{4}}_gen{gen:04d}_score\d+_\d+_uniform_\d{{8}}_\d{{6}}\.json$", uniform_names[0])
        assert re.match(rf"^match_\d{{4}}_gen{gen:04d}_score\d+_\d+_\d{{8}}_\d{{6}}\.json$", best_names[0])

    assert len(trainer.generation_history) == 3
    for rec in trainer.generation_history:
        assert rec["uniform_replay"] is not None and os.path.exists(rec["uniform_replay"])
    assert len(trainer.match_history) == 6
    saved_paths = [m["replay_path"] for m in trainer.match_history if m["replay_path"]]
    assert len(saved_paths) == 3 and all(os.path.exists(p) for p in saved_paths)


def test_replay_every_five_over_seven_generations_writes_only_five_and_seven(tmp_path: Path) -> None:
    """[RED] `--replay-every 5` over 7 generations writes replays only for generations 5 and 7."""
    trainer = _replay_cap_trainer(tmp_path, name="every5", num_envs=2, replay_every=5)
    try:
        trainer.train(total_updates=7)
    finally:
        trainer.close()

    by_gen = _replay_names_by_gen(tmp_path / "rep_every5")
    assert sorted(by_gen.keys()) == [5, 7]
    for gen in (5, 7):
        assert len([n for n in by_gen[gen] if "_uniform_" in n]) == 1
        assert len([n for n in by_gen[gen] if "_uniform_" not in n]) == 1
    for rec in trainer.generation_history:
        if rec["generation"] in (5, 7):
            assert rec["uniform_replay"] is not None and os.path.exists(rec["uniform_replay"])
        else:
            assert rec["uniform_replay"] is None


@pytest.mark.parametrize("save_best_only", [True, False])
def test_replay_kind_filters_uniform_and_best(tmp_path: Path, save_best_only: bool) -> None:
    """[RED] `uniform` writes only `*_uniform_*` files; `best` writes no `*_uniform_*` files."""
    t_uni = _replay_cap_trainer(
        tmp_path, name=f"uni_{save_best_only}", num_envs=2, save_best_only=save_best_only, replay_kind="uniform"
    )
    try:
        t_uni.train(total_updates=3)
    finally:
        t_uni.close()

    uni_files = sorted(p.name for p in (tmp_path / f"rep_uni_{save_best_only}").glob("match_*.json"))
    assert len(uni_files) == 3
    assert all("_uniform_" in name for name in uni_files)
    assert all(m["replay_path"] is None for m in t_uni.match_history)
    assert all(rec["uniform_replay"] is not None for rec in t_uni.generation_history)

    t_best = _replay_cap_trainer(
        tmp_path, name=f"best_{save_best_only}", num_envs=2, save_best_only=save_best_only, replay_kind="best"
    )
    try:
        t_best.train(total_updates=3)
    finally:
        t_best.close()

    best_files = sorted(p.name for p in (tmp_path / f"rep_best_{save_best_only}").glob("match_*.json"))
    expected_count = 3 if save_best_only else 6
    assert len(best_files) == expected_count
    assert all("_uniform_" not in name for name in best_files)
    assert all(rec["uniform_replay"] is None for rec in t_best.generation_history)


@pytest.mark.parametrize("save_best_only", [True, False])
def test_skipped_generation_calls_replay_writer_zero_times_and_keeps_no_deferred_export(
    tmp_path: Path, monkeypatch, save_best_only: bool
) -> None:
    """[RED] A skipped generation invokes `_write_replay_to_disk` zero times and leaves no deferred/uniform payload."""
    calls: list[tuple[int, int, str]] = []
    orig_write = FootballParallelEnv._write_replay_to_disk

    def spy_write(self, match_id, generation, interest_score, stats, frames, tag=""):
        calls.append((int(generation), int(match_id), tag))
        return orig_write(self, match_id, generation, interest_score, stats, frames, tag=tag)

    monkeypatch.setattr(FootballParallelEnv, "_write_replay_to_disk", spy_write)

    trainer = _replay_cap_trainer(
        tmp_path, name=f"spy_{save_best_only}", num_envs=1, save_best_only=save_best_only, replay_every=5
    )
    try:
        # Standalone skipped generation (gen 1, not last of train): zero writes and nothing deferred.
        rollout_1 = trainer.collect_rollouts(active_generation=1)
        assert calls == []
        assert rollout_1["exported_replays"] == []
        env0 = trainer.vec_env.envs[0]
        assert env0._deferred_replay is None
        assert env0._uniform_replay is None

        # Full 2-generation train() with replay_every=5: gen 1 is skipped (0 writes), gen 2 is last (writes).
        trainer.match_history.clear()
        trainer.train(total_updates=2)
        assert [c for c in calls if c[0] == 1] == []
        assert len([c for c in calls if c[0] == 2]) == 2
        assert env0._deferred_replay is None
        assert env0._uniform_replay is None
    finally:
        trainer.close()


def test_skipped_matches_stay_in_history_and_summary_with_none_replay_path(tmp_path: Path) -> None:
    """[RED] Skipped matches enter `match_history` (replay_path=None) and `matches_summary.md`; lookup returns None."""
    from football_rl.replay import find_replay_by_match_id

    trainer = _replay_cap_trainer(tmp_path, name="hist", num_envs=2, replay_every=5)
    try:
        trainer.train(total_updates=5)
    finally:
        trainer.close()

    rep_dir = tmp_path / "rep_hist"
    ckpt_dir = tmp_path / "ckpt_hist"
    assert len(trainer.match_history) == 10
    for m in trainer.match_history:
        if m["generation"] < 5:
            assert m["replay_path"] is None
            assert find_replay_by_match_id(m["match_id"], str(rep_dir), str(ckpt_dir)) is None

    gen5_paths = [m["replay_path"] for m in trainer.match_history if m["generation"] == 5 and m["replay_path"]]
    assert len(gen5_paths) == 1 and os.path.exists(gen5_paths[0])
    gen5_best_id = next(m["match_id"] for m in trainer.match_history if m["replay_path"] == gen5_paths[0])
    assert find_replay_by_match_id(gen5_best_id, str(rep_dir), str(ckpt_dir)) == gen5_paths[0]

    summary_text = (rep_dir / "matches_summary.md").read_text(encoding="utf-8")
    assert "Total Completed Matches:** `10`" in summary_text
    assert "**Match 1**" in summary_text and "_(not saved)_" in summary_text


def test_archive_and_restore_sparse_replay_run(tmp_path: Path) -> None:
    """[RED] Archiving a sparse run keeps existing files (fewer than 15 or >15 sparse gens) and restores cleanly."""
    from football_rl.train_ppo import load_archived_batch

    # 1. Fewer than 15 files from a real capped run (gens 5 and 7 out of 7)
    trainer = _replay_cap_trainer(tmp_path, name="sparse", num_envs=2, replay_every=5)
    try:
        trainer.train(total_updates=7)
    finally:
        trainer.close()

    rep_dir = tmp_path / "rep_sparse"
    ckpt_dir = tmp_path / "ckpt_sparse"
    arch_root = tmp_path / "archives"
    expected_files = sorted(p.name for p in rep_dir.glob("match_*.json"))
    assert len(expected_files) == 4

    dest = Path(archive_batch("sparse_run", str(ckpt_dir), str(rep_dir), str(arch_root)))
    assert sorted(p.name for p in (dest / "replays").glob("match_*.json")) == expected_files
    assert (dest / "replays" / "matches_summary.md").exists()

    restored_ckpt = tmp_path / "restored_ckpt"
    restored_rep = tmp_path / "restored_rep"
    load_archived_batch("sparse_run", str(restored_ckpt), str(restored_rep), str(arch_root))
    assert sorted(p.name for p in restored_rep.glob("match_*.json")) == expected_files
    assert (restored_ckpt / "training_state.pt").exists()

    # 2. More than 15 sparse generations (20 files at gens 5, 10, ..., 100)
    sparse_20 = [f"match_{g:04d}_gen{g:04d}_score10_0_uniform_20261119_120000.json" for g in range(5, 105, 5)]
    kept = select_archive_replays(sparse_20, seed=11)
    assert len(kept) == 15
    assert all(f in sparse_20 for f in kept)
    kept_gens = [int(re.search(r"gen(\d+)", f).group(1)) for f in kept]
    assert kept_gens[:5] == [5, 10, 15, 20, 25]
    assert kept_gens[-5:] == [80, 85, 90, 95, 100]


def test_replay_cap_persists_resumes_without_accept_flag_and_defaults_on_legacy_state(
    tmp_path: Path, capsys
) -> None:
    """[RED] `replay_every` and `replay_kind` persist; resume with different values prints one line and continues; legacy state defaults."""
    first = _replay_cap_trainer(tmp_path, name="res", num_envs=1, replay_every=5, replay_kind="uniform")
    try:
        first.train(total_updates=1)
    finally:
        first.close()

    state_path = tmp_path / "ckpt_res" / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    assert (state["replay_every"], state["replay_kind"]) == (5, "uniform")
    assert (first.generation_history[0]["replay_every"], first.generation_history[0]["replay_kind"]) == (5, "uniform")

    resumed = _replay_cap_trainer(
        tmp_path, name="res", num_envs=1, replay_every=2, replay_kind="best", accept_config_change=False
    )
    try:
        capsys.readouterr()
        assert resumed.load_checkpoints() is True
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
        assert len(lines) == 1
        assert (
            "replay_every=5" in lines[0]
            and "replay_every=2" in lines[0]
            and "replay_kind='uniform'" in lines[0]
            and "replay_kind='best'" in lines[0]
        )
        resumed.train(total_updates=1)
    finally:
        resumed.close()

    state2 = torch.load(state_path, map_location="cpu", weights_only=False)
    assert (state2["replay_every"], state2["replay_kind"]) == (2, "best")
    assert (resumed.generation_history[-1]["replay_every"], resumed.generation_history[-1]["replay_kind"]) == (2, "best")

    # Legacy checkpoint without the two keys reads as (1, "both")
    del state2["replay_every"], state2["replay_kind"]
    torch.save(state2, state_path)

    legacy_default = _replay_cap_trainer(tmp_path, name="res", num_envs=1, replay_every=1, replay_kind="both")
    try:
        capsys.readouterr()
        assert legacy_default.load_checkpoints() is True
        assert capsys.readouterr().out.strip() == ""
    finally:
        legacy_default.close()

    legacy_diff = _replay_cap_trainer(tmp_path, name="res", num_envs=1, replay_every=3, replay_kind="uniform")
    try:
        capsys.readouterr()
        assert legacy_diff.load_checkpoints() is True
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
        assert (
            len(lines) == 1
            and "replay_every=1" in lines[0]
            and "replay_kind='both'" in lines[0]
            and "replay_every=3" in lines[0]
            and "replay_kind='uniform'" in lines[0]
        )
    finally:
        legacy_diff.close()


def test_replay_cap_cli_defaults_and_flags() -> None:
    """[RED] CLI defaults: `--replay-every 1`, `--replay-kind both`; explicit flags parse cleanly."""
    import football_rl.train_ppo as tp

    parser = tp.build_arg_parser()
    defaults = parser.parse_args([])
    assert (defaults.replay_every, defaults.replay_kind) == (1, "both")
    parsed = parser.parse_args(["--replay-every", "5", "--replay-kind", "uniform"])
    assert (parsed.replay_every, parsed.replay_kind) == (5, "uniform")


@pytest.mark.parametrize(
    "argv_extra,flag_name",
    [
        (["--replay-every", "0"], "--replay-every"),
        (["--replay-every", "-1"], "--replay-every"),
        (["--replay-kind", "bogus"], "--replay-kind"),
    ],
)
def test_replay_cap_cli_rejects_invalid_values(tmp_path: Path, monkeypatch, capsys, argv_extra, flag_name) -> None:
    """[RED] Invalid `--replay-every` (0, -1) and unknown `--replay-kind` are rejected before touching disk."""
    import football_rl.train_ppo as tp

    monkeypatch.chdir(tmp_path)
    marker = tmp_path / "checkpoints" / "marker.txt"
    marker.parent.mkdir()
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["train_ppo", "--from-scratch", "--total-updates", "1"] + argv_extra)
    with pytest.raises((SystemExit, ValueError)) as exc:
        tp.main()
    captured = capsys.readouterr()
    assert "unrecognized arguments" not in captured.err
    error_lines = [ln for ln in captured.err.splitlines() if "error:" in ln]
    if isinstance(exc.value, ValueError):
        assert flag_name in str(exc.value)
    else:
        assert error_lines and any(flag_name in ln for ln in error_lines)
    assert marker.read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "replays").exists()


@pytest.mark.parametrize(
    "kwargs,flag_name",
    [
        ({"replay_every": 0}, "replay_every"),
        ({"replay_every": -1}, "replay_every"),
        ({"replay_kind": "bogus"}, "replay_kind"),
    ],
)
def test_replay_cap_trainer_rejects_invalid_values(tmp_path: Path, kwargs, flag_name) -> None:
    """[RED] Trainer validation rejects invalid `replay_every` (0, -1) and `replay_kind` naming the flag."""
    with pytest.raises(ValueError, match=flag_name):
        _replay_cap_trainer(tmp_path, name="invalid", num_envs=1, **kwargs)

