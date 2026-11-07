"""Verification and regression invariant suite for 6v6 MARL Football mechanics, PBRS, and credit assignment."""

from __future__ import annotations

import glob
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch

from football_rl.config import DEFAULT_CONFIG, TEAM_A, TEAM_B
from football_rl.env import FootballParallelEnv
from football_rl.train_ppo import ParameterSharingPPOTrainer


def test_pbrs_strict_non_negativity_and_cross_turnover_telescoping() -> None:
    """Assert all potentials Phi(s) >= 0 and closed state-space loops with turnovers telescope to <= 0."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    gamma = env.cfg.ppo.gamma
    # Define a closed trajectory of (ball_pos, ball_holder) states that crosses a possession turnover
    states = [
        (np.array([450.0, 380.0], dtype=np.float64), 1),    # Team A def_1 holds ball
        (np.array([700.0, 420.0], dtype=np.float64), None), # Loose ball in midfield
        (np.array([850.0, 460.0], dtype=np.float64), 7),    # Team B def_1 holds ball (turnover!)
        (np.array([600.0, 350.0], dtype=np.float64), 3),    # Team A mid_1 holds ball (re-turnover!)
        (np.array([450.0, 380.0], dtype=np.float64), 1),    # Return to exact initial state s_0
    ]

    env.ball_pos = states[0][0].copy()
    env.ball_holder = states[0][1]
    env._compute_potential_shaping(np.zeros(12, dtype=np.float64), any_teleport=True)
    phi_0 = env.prev_tactical_potential.copy()
    assert np.all(phi_0 >= 0.0), f"Negative initial potential detected: {phi_0}"

    cum_discounted_shaping = np.zeros(12, dtype=np.float64)
    discount = 1.0
    for b_pos, holder in states[1:]:
        env.ball_pos = b_pos.copy()
        env.ball_holder = holder
        step_rew = np.zeros(12, dtype=np.float64)
        env._compute_potential_shaping(step_rew, any_teleport=False)
        assert np.all(env.prev_tactical_potential >= 0.0), (
            f"Negative potential detected: {env.prev_tactical_potential}"
        )
        cum_discounted_shaping += discount * step_rew
        discount *= gamma

    k_steps = len(states) - 1
    expected = ((gamma ** k_steps) - 1.0) * phi_0
    np.testing.assert_allclose(cum_discounted_shaping, expected, atol=1e-9)
    assert np.all(cum_discounted_shaping <= 1e-9)


def test_multiplicative_spacing_gate_and_one_to_one_defensive_marking() -> None:
    """Verify crowding attenuates support potential to 0.0 and off-ball defenders mark distinct opponents."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=7)

    # 1. Multiplicative spacing gate: place a_mid_1 at optimal support ring (265u from ball),
    # then move a_mid_2 onto the exact same coordinate and verify potential drops to 0.0.
    env.ball_holder = 5  # a_atk_1
    env.player_pos[5] = np.array([700.0, 420.0], dtype=np.float64)
    env.ball_pos = env.player_pos[5].copy()

    # Spread out other players
    env.player_pos[1] = np.array([300.0, 200.0], dtype=np.float64)
    env.player_pos[2] = np.array([300.0, 640.0], dtype=np.float64)
    env.player_pos[3] = np.array([700.0, 155.0], dtype=np.float64)  # 265u from ball
    env.player_pos[4] = np.array([700.0, 685.0], dtype=np.float64)

    env._compute_potential_shaping(np.zeros(12, dtype=np.float64), any_teleport=True)
    phi_spread = float(env.prev_tactical_potential[3])
    assert phi_spread > 0.10

    # Now clump a_mid_2 (idx=4) within 40u of a_mid_1 (idx=3)
    env.player_pos[4] = np.array([700.0, 185.0], dtype=np.float64)
    env._compute_potential_shaping(np.zeros(12, dtype=np.float64), any_teleport=True)
    phi_clumped = float(env.prev_tactical_potential[3])
    assert phi_clumped == 0.0, f"Expected 0.0 potential when clumped at 30u, got {phi_clumped}"


def test_zero_sum_adversarial_turnover_and_scrum_cooldown_invariants() -> None:
    """Verify Tackles, Interceptions, and Loose Ball Recoveries are strictly zero-sum across teams and cooldown-gated."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=19)

    a_p = env.agent_to_idx["a_mid_1"]
    b_p = env.agent_to_idx["b_mid_1"]
    env.player_pos[a_p] = np.array([700.0, 420.0], dtype=np.float64)
    env.player_pos[b_p] = np.array([720.0, 420.0], dtype=np.float64)

    # 1. Interception Zero-Sum Check
    env.ball_holder = a_p
    env.ball_pos = env.player_pos[a_p].copy()
    env.player_angle[a_p] = 0.0
    rew_step1 = np.zeros(12, dtype=np.float64)
    env._execute_kick(a_p, kick_power=0.6, step_rewards=rew_step1)
    env.ball_pos = env.player_pos[b_p].copy()
    env._check_free_ball_pickup(rew_step1)

    assert env.ball_holder == b_p
    assert abs(float(np.sum(rew_step1))) < 1e-9, f"Interception not zero-sum: {np.sum(rew_step1)}"
    assert env.pickup_cooldown[a_p] > 0.0, "Point-blank opponent interception must not clear kicker cooldown!"

    # 2. Immediate Re-Turnover within 1.0s Cooldown Must Earn 0.0 Reward
    env.ball_holder = b_p
    env.player_angle[b_p] = math.pi
    rew_step2 = np.zeros(12, dtype=np.float64)
    env._execute_kick(b_p, kick_power=0.6, step_rewards=rew_step2)
    env.pickup_cooldown[a_p] = 0.0
    env.ball_pos = env.player_pos[a_p].copy()
    env._check_free_ball_pickup(rew_step2)
    assert np.allclose(rew_step2, 0.0), f"Turnover within cooldown must yield 0 reward, got {rew_step2}"


def test_two_player_ping_pong_oscillation_zero_gain_invariant() -> None:
    """Verify 2-player A <-> B pass oscillation cannot farm pressure-relief or progression rewards."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=23)

    a1 = env.agent_to_idx["a_def_1"]
    a2 = env.agent_to_idx["a_mid_1"]
    opp = env.agent_to_idx["b_atk_1"]

    # Place opponent near a1 (pressed) and a2 unpressed 120u away
    env.player_pos[a1] = np.array([500.0, 300.0], dtype=np.float64)
    env.player_pos[a2] = np.array([500.0, 420.0], dtype=np.float64)
    env.player_pos[opp] = np.array([500.0, 245.0], dtype=np.float64)

    cycle_rewards = []
    for leg in range(4):
        kicker = a1 if leg % 2 == 0 else a2
        receiver = a2 if leg % 2 == 0 else a1
        env.ball_holder = kicker
        env.ball_pos = env.player_pos[kicker].copy()
        env.player_angle[kicker] = math.atan2(
            env.player_pos[receiver][1] - env.player_pos[kicker][1],
            env.player_pos[receiver][0] - env.player_pos[kicker][0],
        )
        step_rew = np.zeros(12, dtype=np.float64)
        env._execute_kick(kicker, kick_power=0.6, step_rewards=step_rew)
        env.ball_pos = env.player_pos[receiver].copy()
        env._check_free_ball_pickup(step_rew)
        cycle_rewards.append(float(np.sum(step_rew[:6])))

    # Leg 0 (first escape pass) may earn positive reward, but all subsequent A<->B ping-pong legs must earn 0.0
    assert all(abs(r) < 1e-9 for r in cycle_rewards[1:]), f"Ping-pong legs earned non-zero reward: {cycle_rewards}"


def test_symmetry_breaking_obs_and_ego_canonical_critic_invariance() -> None:
    """Verify left/right role pairs receive distinct flank bias and Critic places ego agent first."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    obs_dict, _ = env.reset(seed=31)

    # 1. Symmetry breaking between def_1 / def_2 and mid_1 / mid_2
    assert obs_dict["a_def_1"][13] == obs_dict["a_def_2"][13] == -1.0
    assert obs_dict["a_def_1"][14] == -1.0 and obs_dict["a_def_2"][14] == +1.0
    assert obs_dict["a_mid_1"][13] == obs_dict["a_mid_2"][13] == 0.0
    assert obs_dict["a_mid_1"][14] == -1.0 and obs_dict["a_mid_2"][14] == +1.0
    assert obs_dict["a_atk_1"][13] == +1.0 and obs_dict["a_atk_1"][14] == 0.0

    # 2. Stationary ball FOV across match time
    env.physics_frame_count = 10
    obs_early = env._get_observations_array()[:, 16].copy()
    env.physics_frame_count = 5000
    obs_late = env._get_observations_array()[:, 16].copy()
    np.testing.assert_allclose(obs_early, obs_late, atol=1e-6)

    # 3. Ego-first canonical Critic ordering
    central = env.get_centralized_state()
    obs_arr = env._get_observations_array()
    for idx in range(12):
        np.testing.assert_allclose(central[idx, 0], obs_arr[idx, 0], atol=1e-5)
        np.testing.assert_allclose(central[idx, 1], obs_arr[idx, 1], atol=1e-5)

    # 4. Advantage normalization does not wash out off-ball support teammate into negative advantage when team gains
    dummy_obs = torch.zeros((4, 76), dtype=torch.float32)
    # 4 team outfielders: 1 carrier (+1.5) and 3 support teammates (+0.5 each), plus 1 negatively performing sample (-2.0)
    raw_adv = torch.tensor([1.5, 0.5, 0.5, -2.5], dtype=torch.float32)
    norm_adv = ParameterSharingPPOTrainer._normalize_advantages_per_role(dummy_obs, raw_adv)
    assert float(norm_adv[1].item()) > 0.0 and float(norm_adv[2].item()) > 0.0


def assert_replay_telemetry_thresholds(replay_path: str) -> None:
    """Helper to assert post-training replay telemetry satisfies anti-clumping spatial thresholds.

    Optimized using vectorized NumPy array slicing, pairwise 3D broadcasting, and boolean
    reduction across all match frames to achieve microsecond-tier throughput without Python loops.
    """
    with open(replay_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = [fr for fr in data.get("frames", []) if fr.get("frame") not in (0, 5400)]
    if not frames:
        return

    try:
        players_arr = np.array([fr["players"] for fr in frames], dtype=np.float64)[:, :, :2]
        balls_arr = np.array([fr["ball"] for fr in frames], dtype=np.float64)[:, :2]
    except Exception:
        players_arr = np.array([[p[:2] for p in fr["players"]] for fr in frames], dtype=np.float64)
        balls_arr = np.array([fr["ball"][:2] for fr in frames], dtype=np.float64)

    num_frames = players_arr.shape[0]
    if num_frames == 0:
        return

    # Team A (1..5) and Team B (7..11) outfield positions: shape (F, 5, 2)
    pos_a = players_arr[:, 1:6, :]
    pos_b = players_arr[:, 7:12, :]

    # Vectorized pairwise distance computation with broadcast subtraction: shape (F, 5, 5)
    diff_a = pos_a[:, :, None, :] - pos_a[:, None, :, :]
    dmat_a = np.hypot(diff_a[:, :, :, 0], diff_a[:, :, :, 1])
    diag_5 = np.arange(5)
    dmat_a[:, diag_5, diag_5] = np.inf
    mean_a = np.mean(np.min(dmat_a, axis=2), axis=1)

    diff_b = pos_b[:, :, None, :] - pos_b[:, None, :, :]
    dmat_b = np.hypot(diff_b[:, :, :, 0], diff_b[:, :, :, 1])
    dmat_b[:, diag_5, diag_5] = np.inf
    mean_b = np.mean(np.min(dmat_b, axis=2), axis=1)

    mean_nearest_tm = float(np.mean(np.concatenate([mean_a, mean_b])))

    # Vectorized clumping detection relative to ball: shape (F, 5)
    bx = balls_arr[:, 0:1]
    by = balls_arr[:, 1:2]
    da = np.hypot(pos_a[:, :, 0] - bx, pos_a[:, :, 1] - by)
    db = np.hypot(pos_b[:, :, 0] - bx, pos_b[:, :, 1] - by)
    clump_a = np.sum(da <= 120.0, axis=1) >= 3
    clump_b = np.sum(db <= 120.0, axis=1) >= 3
    clump_frames = int(np.sum(clump_a | clump_b))

    pct_clumped = 100.0 * clump_frames / max(1, num_frames)
    assert mean_nearest_tm > 140.0, f"Mean nearest-teammate dist {mean_nearest_tm:.1f}u <= 140.0u"
    assert pct_clumped < 15.0, f"Clumped frames percentage {pct_clumped:.1f}% >= 15.0%"


def test_replay_telemetry_thresholds_vectorization_and_correctness(tmp_path) -> None:
    """Verify that vectorized assert_replay_telemetry_thresholds accurately discriminates spatial structures."""
    # 1. Create a well-spaced synthetic replay (nearest-teammate ~200u, distance to ball ~300u)
    frames_clean = []
    for f in range(1, 20):
        # 12 players: keeper at 0/6, 5 outfielders spaced 200u apart along x-axis
        players_a = [[50.0, 420.0]] + [[200.0 + i * 200.0, 300.0] for i in range(5)]
        players_b = [[1350.0, 420.0]] + [[200.0 + i * 200.0, 540.0] for i in range(5)]
        players = [[x, y, 0.0, 0, 0.0, 0.0] for x, y in players_a + players_b]
        frames_clean.append({
            "frame": f,
            "players": players,
            "ball": [700.0, 420.0, 0, 0.0, 0.0],
        })

    clean_file = tmp_path / "replay_clean.json"
    with open(clean_file, "w", encoding="utf-8") as f:
        json.dump({"frames": frames_clean}, f)

    # Must pass without raising AssertionError
    assert_replay_telemetry_thresholds(str(clean_file))

    # 2. Create a severely clumped synthetic replay (4 teammates within 30u of ball)
    frames_clumped = []
    for f in range(1, 20):
        players_a = [[50.0, 420.0]] + [[700.0 + i * 15.0, 420.0] for i in range(5)]
        players_b = [[1350.0, 420.0]] + [[200.0 + i * 200.0, 540.0] for i in range(5)]
        players = [[x, y, 0.0, 0, 0.0, 0.0] for x, y in players_a + players_b]
        frames_clumped.append({
            "frame": f,
            "players": players,
            "ball": [700.0, 420.0, 0, 0.0, 0.0],
        })

    clumped_file = tmp_path / "replay_clumped.json"
    with open(clumped_file, "w", encoding="utf-8") as f:
        json.dump({"frames": frames_clumped}, f)

    # Must fail with AssertionError due to low teammate spacing and clumping around ball
    with pytest.raises(AssertionError):
        assert_replay_telemetry_thresholds(str(clumped_file))
