"""Unit and integration tests for Centralized Training with Decentralized Execution (CTDE) PPO."""

from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

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
from football_rl.train_ppo import (
    ActorCritic,
    ParameterSharingPPOTrainer,
    archive_batch,
    migrate_team_color_keys,
    print_active_status,
    select_archive_replays,
)


def test_ctde_ppo_rollout_and_update() -> None:
    """Verify CTDE PPO rollout and update with centralized critic state (130-dim) and decentralized actor (76-dim)."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    trainer = ParameterSharingPPOTrainer(
        config=cfg,
        num_envs=1,
        rollout_steps=10,
        device="cpu",
    )

    # Verify network architectures: actor (76), centralized critic (130 = 76 + 54)
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
    """Verify _load_adapted_weights cleanly adapts legacy 76-dim critic weights into 130-dim model."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    trainer = ParameterSharingPPOTrainer(config=cfg, num_envs=1, rollout_steps=10, device="cpu")

    # Create dummy legacy state dict with critic.0.weight shape (hidden, 76)
    target_policy = trainer.blue_field_policy
    legacy_weights = {}
    for k, v in target_policy.state_dict().items():
        if k == "critic.0.weight":
            # 76-dim legacy input features
            legacy_weights[k] = torch.ones((v.shape[0], OBS_DIM), dtype=torch.float32)
        else:
            legacy_weights[k] = v.clone()

    # Load adapted weights into 130-dim critic
    trainer._load_adapted_weights(target_policy, legacy_weights)

    loaded_critic_w = target_policy.critic[0].weight.data
    # First 76 cols should be ones (from legacy checkpoint)
    assert torch.allclose(loaded_critic_w[:, :OBS_DIM], torch.ones((loaded_critic_w.shape[0], OBS_DIM)))
    # Remaining 54 cols should be zero-padded
    assert torch.allclose(loaded_critic_w[:, OBS_DIM:], torch.zeros((loaded_critic_w.shape[0], CENTRAL_STATE_DIM)))


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
