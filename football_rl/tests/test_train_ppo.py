"""Unit and integration tests for Centralized Training with Decentralized Execution (CTDE) PPO."""

from __future__ import annotations

import copy
import hashlib
import math
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
    uniform_env_index,
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


def test_coevo_team_b_dominance_syncs_team_a_towards_b(tmp_path: Path) -> None:
    trainer = _coevo_trainer(tmp_path)
    assert trainer._coevolution_step(_matches_with_totals(1, 7, n=24)) is True  # 7 >= 3*max(1,1)
    assert _team_a_weight(trainer) == pytest.approx(0.65)  # 0.65*1 + 0.35*0
    assert _team_b_weight(trainer) == 0.0


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
