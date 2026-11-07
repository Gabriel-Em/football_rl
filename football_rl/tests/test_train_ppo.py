"""Unit and integration tests for Centralized Training with Decentralized Execution (CTDE) PPO."""

from __future__ import annotations

from dataclasses import replace
import torch

from football_rl.config import (
    ACTION_DIM,
    ALL_AGENTS,
    CENTRAL_STATE_DIM,
    CRITIC_STATE_DIM,
    DEFAULT_CONFIG,
    OBS_DIM,
)
from football_rl.train_ppo import ActorCritic, ParameterSharingPPOTrainer


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
    """Verify _normalize_advantages_per_role normalizes DEF, MID, and ATK advantages independently to unit variance."""
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
    for sl in (slice(0, 4), slice(4, 8), slice(8, 12)):
        role_slice = norm_adv[sl]
        assert abs(float(role_slice.mean().item())) < 1e-6
        assert abs(float(role_slice.std().item()) - 1.0) < 1e-5

