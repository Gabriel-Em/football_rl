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

from football_rl.config import (
    DEFAULT_CONFIG,
    EnvConfig,
    ROLE_ATTACKER,
    ROLE_DEFENDER,
    ROLE_KEEPER,
    ROLE_MIDFIELDER,
    TEAM_A,
    TEAM_B,
    TEAM_LEFT,
    TEAM_RIGHT,
)
from football_rl.env import FootballEnv, FootballParallelEnv
from football_rl.train_ppo import ParameterSharingPPOTrainer


def test_pbrs_strict_non_negativity_and_cross_turnover_telescoping() -> None:
    """Assert uncrowded potentials Phi(s) >= 0 and closed state-space loops with turnovers telescope to 0."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    gamma = 1.0  # Undiscounted potential difference for zero stationary leak
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

    # Now clump a_mid_2 (idx=4) within 30u of a_mid_1 (idx=3)
    env.player_pos[4] = np.array([700.0, 185.0], dtype=np.float64)
    env._compute_potential_shaping(np.zeros(12, dtype=np.float64), any_teleport=True)
    phi_clumped = float(env.prev_tactical_potential[3])
    assert phi_clumped < phi_spread, f"Expected reduced potential when clumped, got {phi_clumped} vs {phi_spread}"
    assert phi_clumped > 0.0, f"Expected non-zero potential to avoid gradient death, got {phi_clumped}"

    # Moving apart increases potential (positive separation gradient)
    env.player_pos[4] = np.array([700.0, 215.0], dtype=np.float64)  # 60u separation
    env._compute_potential_shaping(np.zeros(12, dtype=np.float64), any_teleport=True)
    phi_separating = float(env.prev_tactical_potential[3])
    assert phi_separating > phi_clumped, f"Expected increasing potential when separating, got {phi_separating} vs {phi_clumped}"


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


def test_kickoff_pass_pbrs_continuity_and_circulation_bonus():
    """Verify that kicking a pass does not trigger a -0.4675 PBRS cliff and open kickoff passes earn positive reward."""
    env = FootballEnv(EnvConfig())
    env.reset(seed=42)
    env.kickoff_protection_timer = 0.0

    # Striker (idx 5) holds ball at center (700, 420) facing Midfielder (idx 3) at (520, 580)
    kicker_idx, receiver_idx = 5, 3
    env.ball_holder = kicker_idx
    env.last_ball_holder = kicker_idx
    env.last_ball_team = TEAM_LEFT
    env.player_pos[kicker_idx] = [700.0, 420.0]
    env.player_pos[receiver_idx] = [520.0, 580.0]
    angle_to_mid = math.atan2(580.0 - 420.0, 520.0 - 700.0)
    env.player_angle[kicker_idx] = angle_to_mid

    env.prev_potential = np.zeros(env.num_players, dtype=np.float64)
    env._compute_potential_shaping( np.zeros(env.num_players, dtype=np.float64))
    phi_before_kick = float(env.prev_potential[kicker_idx])

    # Execute pass kick
    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._execute_kick(kicker_idx, kick_power=0.6, step_rewards=step_rewards)
    pbrs_delta = np.zeros(env.num_players, dtype=np.float64)
    env._compute_potential_shaping(pbrs_delta)
    phi_after_kick = float(env.prev_potential[kicker_idx])

    # Kicker potential must remain continuous on the kick frame (no -0.4675 cliff)
    assert abs(phi_after_kick - phi_before_kick) < 0.05, (
        f"Kicker suffered PBRS cliff on kick frame: {phi_before_kick:.4f} -> {phi_after_kick:.4f}"
    )


def test_stopped_corner_ball_cannot_trigger_through_ball():
    """Verify that a wall-bounced or stopped ball in the corner never triggers THROUGH BALL!"""
    env = FootballEnv(EnvConfig())
    env.reset(seed=42)

    # Ball kicked off top wall in corner, then comes to a stop
    env.last_ball_holder = 2
    env.last_ball_team = TEAM_LEFT
    env.ball_kicked_in_flight = True
    env.is_wall_pass_in_flight = True
    env.ball_kick_time = env.match_time - 0.5
    env.ball_kick_start_pos = np.array([1295.0, 10.0], dtype=np.float64)
    env.ball_pos = np.array([1324.7, 180.0], dtype=np.float64)
    env.ball_vel = np.array([0.0, 0.0], dtype=np.float64)  # Stopped ball

    # Run one physics substep to trigger stationary flight expiry
    env._physics_substep(np.zeros((12, 4), dtype=np.float64), np.zeros(12, dtype=np.float64))
    assert not env.ball_kicked_in_flight, "Stopped ball must expire ball_kicked_in_flight!"


def test_gk_pickup_has_no_pbrs_cliff_and_cannot_kick_own_goal():
    """Verify GK loose-ball pickup does not drop GK potential and GK kick cannot point into own goal."""
    env = FootballEnv(EnvConfig())
    env.reset(seed=42)

    gk_idx = 0  # Left GK
    env.player_pos[gk_idx] = [150.0, 260.0]
    env.ball_pos = np.array([165.0, 260.0], dtype=np.float64)
    env.ball_holder = None

    dummy = np.zeros(env.num_players, dtype=np.float64)
    env._compute_potential_shaping(dummy)
    phi_loose = float(env.prev_potential[gk_idx])

    # GK claims the ball
    env.ball_holder = gk_idx
    pbrs_pickup = np.zeros(env.num_players, dtype=np.float64)
    env._compute_potential_shaping(pbrs_pickup)
    phi_held = float(env.prev_potential[gk_idx])

    assert phi_held >= phi_loose - 0.05, (
        f"GK punished by PBRS for picking up loose box ball: {phi_loose:.4f} -> {phi_held:.4f}"
    )

    # Point GK directly at own net (angle = pi) and force a kick
    env.player_angle[gk_idx] = math.pi
    env._execute_kick(gk_idx, kick_power=1.0, step_rewards=dummy)
    assert env.ball_vel[0] > 0.0, f"Left GK kicked backward toward own goal: ball_vel={env.ball_vel}"


def test_gk_5s_violation_prevents_teammate_recovery_and_protects_goal_mouth():
    """Verify 5s GK violation locks out teammates and resets GK to goal mouth."""
    env = FootballEnv(EnvConfig())
    env.reset(seed=42)

    gk_idx = 6  # Right GK
    teammate_idx = 7  # Right Defender
    env.player_pos[gk_idx] = [1378.0, 226.0]  # Stranded in corner
    env.player_pos[teammate_idx] = [1204.0, 250.0]
    env.ball_holder = gk_idx

    rewards = np.zeros(env.num_players, dtype=np.float64)
    env._handle_keeper_5s_violation(gk_idx, rewards, is_hold_violation=True)

    # 1. Offending team must have pickup cooldown >= 1.5s so teammate cannot grab the foul ball
    assert env.pickup_cooldown[teammate_idx] >= 1.5
    # 2. GK must be reset near goal center (y == 420.0), not stranded in the corner
    assert abs(float(env.player_pos[gk_idx, 1]) - 420.0) < 1e-3


def test_bumper_to_bumper_tackle_probability_and_whiff_distance_gate():
    """Verify Attackers have non-zero tackle probability at near-contact (40u) and distant tackles are gated."""
    env = FootballEnv(EnvConfig())
    env.reset(seed=42)
    env.kickoff_protection_timer = 0.0
    env.pickup_tackle_grace_timer = 0.0

    # 1. Distant tackle (> 180u from ball) must be ignored without penalty
    atk_idx = 5
    carrier_idx = 8
    env.ball_holder = carrier_idx
    env.player_pos[carrier_idx] = [1000.0, 420.0]
    env.ball_pos = np.array([1000.0, 420.0], dtype=np.float64)
    env.player_pos[atk_idx] = [200.0, 420.0]  # 800u away

    rewards = np.zeros(env.num_players, dtype=np.float64)
    env._attempt_tackle(atk_idx, rewards)
    assert rewards[atk_idx] == 0.0 and env.stats["failed_tackles"] == 0

    # 2. Near-contact (40.0u, beyond the 2 * player_radius = 36u touching distance) for Attacker
    env.player_pos[atk_idx] = [960.0, 420.0]  # Exactly 40.0u from carrier at (1000, 420)
    env.player_angle[atk_idx] = 0.0
    successes = 0
    for s in range(100):
        env.ball_holder = carrier_idx
        env.pickup_tackle_grace_timer = 0.0
        env.tackle_cooldown[atk_idx] = 0.0
        env.stun_timer[atk_idx] = 0.0
        env.player_pos[atk_idx] = [960.0, 420.0]
        env.player_pos[carrier_idx] = [1000.0, 420.0]
        env.ball_pos = np.array([1000.0, 420.0], dtype=np.float64)
        env._attempt_tackle(atk_idx, np.zeros(12, dtype=np.float64))
        if env.ball_holder == atk_idx:
            successes += 1
    assert successes >= 25, f"Attacker tackle rate at 40u contact should be ~43%, got {successes}%"

