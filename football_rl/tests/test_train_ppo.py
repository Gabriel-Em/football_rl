"""Unit and integration tests for Centralized Training with Decentralized Execution (CTDE) PPO."""

from __future__ import annotations

from dataclasses import replace
import os
import re
from pathlib import Path
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


