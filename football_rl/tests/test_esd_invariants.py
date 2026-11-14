"""Unit test suite for core physical, reward, and invariant properties of the football environment."""

import copy
import math
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch

from football_rl.config import (
    DEFAULT_CONFIG,
    ROLE_ATK,
    ROLE_DEF,
    ROLE_KEEPER,
    ROLE_MID,
    TEAM_A,
    TEAM_B,
)
from football_rl.env import FootballParallelEnv, parallel_env
from football_rl.potentials import PotentialState, compute_potentials
from football_rl.train_ppo import ParameterSharingPPOTrainer


def _zero_actions(env: FootballParallelEnv) -> dict:
    return {
        agent: np.zeros(env.action_space(agent).shape, dtype=np.float32)
        for agent in env.possible_agents
    }


def test_closed_loop_pbrs_zero_gain_invariant() -> None:
    """Moving the ball along a closed spatial trajectory without scoring must yield net non-positive cumulative discounted shaping reward."""
    env = FootballParallelEnv(
        replace(DEFAULT_CONFIG, rewards=replace(DEFAULT_CONFIG.rewards, shaping_mode="legacy"))
    )
    env.reset(seed=42)

    # Disable other influences by placing all players at fixed positions
    for i in range(12):
        env.player_pos[i] = np.array([700.0, 100.0 + i * 40.0], dtype=np.float64)
        env.player_vel[i] = np.zeros(2, dtype=np.float64)
    env.ball_holder = None

    gamma = 1.0  # Undiscounted potential difference for zero stationary leak
    rew_w = env.cfg.rewards.team_advance_pbrs_weight
    # Define a closed spatial trajectory for the ball: s0 -> s1 -> s2 -> s3 -> s0
    waypoints = [
        np.array([700.0, 420.0], dtype=np.float64),
        np.array([900.0, 500.0], dtype=np.float64),
        np.array([600.0, 300.0], dtype=np.float64),
        np.array([500.0, 400.0], dtype=np.float64),
        np.array([700.0, 420.0], dtype=np.float64),
    ]

    total_discounted_pbrs_team_a = 0.0
    total_discounted_tactical_pbrs = np.zeros(12, dtype=np.float64)
    discount = 1.0

    # Initialize at waypoints[0]
    env.ball_pos = waypoints[0].copy()
    env.ball_vel = np.zeros(2, dtype=np.float64)
    phi_0 = env._compute_ball_potential(TEAM_A)
    phi_prev = phi_0
    env._compute_potential_shaping(np.zeros(12, dtype=np.float64), any_teleport=True)
    phi_tactical_0 = env.prev_tactical_potential.copy()

    for wp in waypoints[1:]:
        env.ball_pos = wp.copy()
        phi_curr = env._compute_ball_potential(TEAM_A)
        f_pbrs = rew_w * (gamma * phi_curr - phi_prev)
        total_discounted_pbrs_team_a += discount * f_pbrs

        step_tactical = np.zeros(12, dtype=np.float64)
        env._compute_potential_shaping(step_tactical, any_teleport=False)
        total_discounted_tactical_pbrs += discount * step_tactical

        discount *= gamma
        phi_prev = phi_curr

    # Net discounted PBRS over any closed loop telescopes exactly to (gamma^K - 1) * Phi(s_0)
    k_steps = len(waypoints) - 1
    assert total_discounted_pbrs_team_a <= 1e-6
    assert abs(total_discounted_pbrs_team_a - rew_w * ((gamma ** k_steps) - 1.0) * phi_0) < 1e-9
    np.testing.assert_allclose(
        total_discounted_tactical_pbrs,
        ((gamma ** k_steps) - 1.0) * phi_tactical_0,
        atol=1e-9,
    )


def test_stationary_triangle_convergence_invariant() -> None:
    """Stationary multi-player passing loops earn zero passer exploitation reward on non-progressive/unpressed legs and non-positive net PBRS over closed cycles."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=101)

    p1 = env.agent_to_idx["a_def_1"]
    p2 = env.agent_to_idx["a_def_2"]
    # Equidistant from enemy goal (1400, 420) with opponents far away -> d_phi_threat == 0, d_sigma_press == 0
    env.player_pos[p1] = np.array([400.0, 270.0], dtype=np.float64)
    env.player_pos[p2] = np.array([400.0, 570.0], dtype=np.float64)
    for idx in range(6, 12):
        env.player_pos[idx] = np.array([1200.0, 120.0 + (idx - 6) * 100.0], dtype=np.float64)

    passer_rewards = []
    for k in range(6):
        kicker = p1 if k % 2 == 0 else p2
        rx = p2 if k % 2 == 0 else p1
        env.ball_holder = kicker
        env.ball_pos = env.player_pos[kicker].copy()
        env.player_angle[kicker] = math.atan2(
            env.player_pos[rx][1] - env.player_pos[kicker][1],
            env.player_pos[rx][0] - env.player_pos[kicker][0],
        )
        step_rew = np.zeros(12, dtype=np.float64)
        env._execute_kick(kicker, kick_power=0.6, step_rewards=step_rew)

        env.ball_kicked_in_flight = True
        env.ball_kick_flight_timer = 0.5
        env.last_ball_holder = kicker
        env.last_ball_team = TEAM_A
        env.ball_pos = env.player_pos[rx].copy()
        env._check_free_ball_pickup(step_rew)
        passer_rewards.append(float(step_rew[kicker]))

    assert all(abs(r) <= 1e-6 for r in passer_rewards)


def test_midfield_blind_shot_negative_ev_invariant() -> None:
    """Firing a shot from X = 690 directly at Goalkeeper must yield negative net reward after catch."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=77)

    shooter = env.agent_to_idx["a_atk_1"]
    gk = env.agent_to_idx["b_keeper"]

    # In canonical coordinates for Team A, goal is at (1400, 420).
    # Team B's keeper is stationed in front of Team B's goal at (1380, 420).
    env.player_pos[gk] = np.array([1380.0, 420.0], dtype=np.float64)
    env.player_pos[shooter] = np.array([690.0, 420.0], dtype=np.float64)
    env.player_angle[shooter] = 0.0  # Facing enemy goal
    env.ball_holder = shooter
    env.ball_pos = env.player_pos[shooter].copy()

    # Artificially inflate pass chain count to test gating
    env.pass_chain_count[TEAM_A] = 10

    kick_rew = np.zeros(12, dtype=np.float64)
    env._execute_kick(shooter, kick_power=1.0, step_rewards=kick_rew)

    # Move ball to GK and trigger catch/save
    env.ball_pos = env.player_pos[gk].copy()
    env.ball_vel = np.array([300.0, 0.0], dtype=np.float64)
    catch_rew = np.zeros(12, dtype=np.float64)
    env._check_free_ball_pickup(catch_rew)

    net_shooter_reward = kick_rew[shooter] + catch_rew[shooter]
    assert net_shooter_reward < 0.0


def test_gk_box_micro_tap_shrink_monotonicity_invariant() -> None:
    """Goalkeeper micro-tapping inside own box earns 0 recovery reward and does not reset loose ball timer."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=99)

    gk = env.agent_to_idx["a_keeper"]
    # Team A keeper inside own box (< 205u from goal [0, 420])
    env.player_pos[gk] = np.array([100.0, 420.0], dtype=np.float64)
    env.ball_holder = gk
    env.ball_pos = env.player_pos[gk].copy()

    # Step timer forward
    env.box_loose_ball_timer[TEAM_A] = 2.0
    prev_timer = env.box_loose_ball_timer[TEAM_A]

    # GK releases ball slightly
    env.ball_holder = None
    env.ball_pos = np.array([105.0, 420.0], dtype=np.float64)
    env.ball_vel = np.array([5.0, 0.0], dtype=np.float64)

    # Run physics substep to advance loose ball timer
    act = np.zeros((env.num_players, 4), dtype=np.float32)
    env._physics_substep(act, np.zeros(env.num_players))
    assert env.box_loose_ball_timer[TEAM_A] > prev_timer

    # GK re-claims ball in box
    step_rew = np.zeros(12, dtype=np.float64)
    env.player_pos[gk] = env.ball_pos.copy()
    env._check_free_ball_pickup(step_rew)

    # Must earn 0.0 recovery reward
    assert step_rew[gk] == 0.0


def test_ghost_mode_pickup_prohibition_invariant() -> None:
    """No player with ghost_timer > 0.0 or stun_timer > 0.0 may ever hold or pick up the ball."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=88)

    # Move all players far away
    for i in range(12):
        env.player_pos[i] = np.array([100.0, 100.0 + i * 50.0], dtype=np.float64)

    player = env.agent_to_idx["a_mid_1"]
    env.ball_holder = None
    env.ball_pos = np.array([700.0, 420.0], dtype=np.float64)
    env.ball_vel = np.zeros(2, dtype=np.float64)
    env.player_pos[player] = env.ball_pos.copy()

    # Case 1: Ghost timer active
    env.ghost_timer[player] = 1.0
    env.is_ghost[player] = True
    step_rew = np.zeros(12, dtype=np.float64)
    env._check_free_ball_pickup(step_rew)
    assert env.ball_holder is None

    # Case 2: Stun timer active
    env.ghost_timer[player] = 0.0
    env.is_ghost[player] = False
    env.stun_timer[player] = 1.0
    env._check_free_ball_pickup(step_rew)
    assert env.ball_holder is None


def test_5s_violation_teammate_touch_cooldown_integrity_invariant() -> None:
    """A teammate picking up a ball dropped by a 5-second hold violation must not clear the violator's pickup cooldown."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=12)

    carrier = env.agent_to_idx["a_atk_1"]
    teammate = env.agent_to_idx["a_def_1"]

    # Trigger 5s violation manually
    env.ball_holder = carrier
    env.ball_hold_timer = 5.0
    act = np.zeros((env.num_players, 4), dtype=np.float32)
    env._physics_substep(act, np.zeros(env.num_players))

    assert env.pickup_cooldown[carrier] > 0.0

    # Teammate touches/picks up ball
    env.player_pos[teammate] = env.ball_pos.copy()
    step_rew = np.zeros(12, dtype=np.float64)
    env._check_free_ball_pickup(step_rew)

    assert env.ball_holder == teammate
    # Violator's cooldown must remain intact
    assert env.pickup_cooldown[carrier] > 0.0


def test_carried_ball_pitch_boundary_invariant() -> None:
    """Ball center must never exit [10, 1390] x [10, 830] except when crossing goal mouth [320, 520]."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=34)

    carrier = env.agent_to_idx["a_atk_1"]
    env.ball_holder = carrier

    # Place carrier outside top-left corner
    env.player_pos[carrier] = np.array([5.0, 5.0], dtype=np.float64)
    env.player_vel[carrier] = np.array([-50.0, -50.0], dtype=np.float64)
    env.ball_pos = env.player_pos[carrier].copy()

    # Step physics
    act = np.zeros((env.num_players, 4), dtype=np.float32)
    for _ in range(5):
        env._physics_substep(act, np.zeros(env.num_players))

    assert env.ball_pos[0] >= 10.0
    assert env.ball_pos[1] >= 10.0
    assert env.ball_pos[0] <= 1390.0
    assert env.ball_pos[1] <= 830.0


def test_active_lunge_solid_collision_invariant() -> None:
    """A player actively coasting on a tackle lunge impulse must remain physically solid until lunge timer expires."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=50)

    tackler = env.agent_to_idx["a_def_1"]
    env.lunge_timer[tackler] = env.cfg.mechanics.field_lunge_duration
    env.stun_timer[tackler] = env.cfg.mechanics.tackle_miss_stun_time
    env.is_ghost[tackler] = False
    env.ghost_timer[tackler] = 0.0

    assert not env._is_ghost(tackler)


def test_full_trajectory_gk_distribution_interception_gate_invariant() -> None:
    """A Goalkeeper kicking toward an opponent who intercepts the ball trajectory must receive 0.0 release bonus."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=65)

    gk = env.agent_to_idx["a_keeper"]
    opp = env.agent_to_idx["b_atk_1"]

    # Move other players far away
    for i in range(12):
        if i not in (gk, opp):
            env.player_pos[i] = np.array([700.0, 100.0 + i * 50.0], dtype=np.float64)

    env.ball_holder = gk
    env.player_pos[gk] = np.array([100.0, 420.0], dtype=np.float64)
    env.player_angle[gk] = 0.0  # Facing directly along +X ray

    # Place opponent directly on the ball ray at distance 420u
    env.player_pos[opp] = np.array([520.0, 420.0], dtype=np.float64)

    step_rew = np.zeros(12, dtype=np.float64)
    env._execute_kick(gk, kick_power=1.0, step_rewards=step_rew)

    # Intercepted release bonus must be 0.0
    assert step_rew[gk] == 0.0


def test_observation_and_critic_bounds_invariant() -> None:
    """Actor observations and centralized critic states must remain strictly bounded in [-1.0, 1.0] without NaN/Inf."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    obs_dict, _ = env.reset(seed=92)

    for agent, obs in obs_dict.items():
        assert obs.shape == (82,)
        assert not np.isnan(obs).any()
        assert not np.isinf(obs).any()
        assert np.all(obs >= -1.0)
        assert np.all(obs <= 1.0)

    central_state = env.get_centralized_state()
    assert central_state.shape == (12, 54)
    assert not np.isnan(central_state).any()
    assert not np.isinf(central_state).any()
    assert np.all(central_state >= -1.0)


def _short_all_mode_config(tmp_path: Path, shaping_mode: str = "all", gamma: float = 0.995):
    return replace(
        DEFAULT_CONFIG,
        physics=replace(
            DEFAULT_CONFIG.physics,
            half_physics_frames=140,
            max_rl_steps=50,
            max_physics_frames=200,
        ),
        rewards=replace(DEFAULT_CONFIG.rewards, shaping_mode=shaping_mode),
        ppo=replace(DEFAULT_CONFIG.ppo, gamma=gamma),
        replay=replace(
            DEFAULT_CONFIG.replay,
            enable_recording=True,
            save_dir=str(tmp_path / "replays"),
            interest_threshold=float("inf"),
        ),
    )


def _scripted_goal_halftime_actions(step_idx: int) -> np.ndarray:
    acts = np.zeros((12, 4), dtype=np.float32)
    # Team B keeper and striker turn 90 deg and run off the y=420 lane; a_atk_1 turns from kickoff angle to 0 and shoots
    if step_idx < 5:
        acts[6] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        acts[11] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    else:
        acts[6] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        acts[11] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if step_idx < 8:
        acts[5] = np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float32)
    else:
        acts[5] = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    return acts


def test_pbrs_t5_episode_identity_with_goal_halftime_and_fulltime(tmp_path: Path) -> None:
    """T5: In all mode, discounted sum of last_shaping_rewards over goal, halftime and fulltime equals -Phi_total(s_0)."""
    cfg = _short_all_mode_config(tmp_path, shaping_mode="all", gamma=0.995)
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)

    w_adv = cfg.rewards.team_advance_pbrs_weight
    phi_ball_0 = np.array(
        [w_adv * env._compute_ball_potential(TEAM_A)] * 6
        + [w_adv * env._compute_ball_potential(TEAM_B)] * 6,
        dtype=np.float64,
    )
    phi_total_0 = phi_ball_0 + env.prev_tactical_potential.copy()

    gamma = cfg.ppo.gamma
    total_shaping = np.zeros(12, dtype=np.float64)
    goal_step = None
    halftime_seen = False
    done = False

    for t in range(cfg.physics.max_rl_steps):
        _, _, _, done, _ = env.step_array(_scripted_goal_halftime_actions(t))
        assert env.last_shaping_rewards.dtype == np.float64
        assert env.last_shaping_rewards.shape == (12,)
        total_shaping += (gamma ** t) * env.last_shaping_rewards
        if goal_step is None and env.stats["goals"] >= 1:
            goal_step = t
        if env.current_half == 2:
            halftime_seen = True
        if done:
            break

    assert goal_step is not None and goal_step < 30
    assert halftime_seen
    assert done
    assert any(ev.get("type") == "FULLTIME!" for ev in env.replay_frames[-1]["events"])
    np.testing.assert_allclose(total_shaping, -phi_total_0, rtol=0.0, atol=1e-9)

    # Verify replay breakdown pot_delta + r_ball_adv reflects paid shaping
    last_bd = env.replay_frames[-1]["rew_breakdown"]
    paid_from_bd = np.array([b["r_ball_adv"] + b["r_pot_i"] for b in last_bd], dtype=np.float64)
    np.testing.assert_allclose(paid_from_bd, env.last_shaping_rewards, rtol=0.0, atol=5e-4)


def test_pbrs_t6_no_standing_wage_in_env(tmp_path: Path) -> None:
    """T6: Fixed inert state for 50 steps yields legacy - all == (1 - gamma^50) * Phi_total(s_0)."""
    def _setup_inert(shaping_mode: str):
        cfg = replace(
            DEFAULT_CONFIG,
            physics=replace(DEFAULT_CONFIG.physics, half_physics_frames=1000, max_rl_steps=200, max_physics_frames=800),
            rewards=replace(DEFAULT_CONFIG.rewards, shaping_mode=shaping_mode),
            replay=replace(DEFAULT_CONFIG.replay, enable_recording=False, save_dir=str(tmp_path / "r")),
        )
        env = FootballParallelEnv(cfg)
        env.reset(seed=7)
        # Stationary loose ball in open midfield (no wall/box/5s rule triggers within 50 steps)
        env.ball_holder = None
        env.ball_kicked_in_flight = False
        env.ball_pos = np.array([900.0, 420.0], dtype=np.float64)
        env.ball_vel = np.zeros(2, dtype=np.float64)
        env.kickoff_protection_timer = 0.0
        env.stun_timer[:] = 0.0
        env.ghost_timer[:] = 0.0
        env.is_ghost[:] = False
        env.pickup_cooldown[:] = 0.0
        # Place all players >= 100u from ball and >= 150u apart, zero velocity
        coords = [
            (45.0, 420.0), (250.0, 200.0), (250.0, 640.0), (500.0, 200.0), (500.0, 640.0), (800.0, 420.0),
            (1355.0, 420.0), (1150.0, 200.0), (1150.0, 640.0), (1050.0, 120.0), (1050.0, 720.0), (700.0, 120.0),
        ]
        for i, xy in enumerate(coords):
            env.player_pos[i] = np.array(xy, dtype=np.float64)
            env.player_vel[i] = 0.0
        env._compute_potential_shaping(np.zeros(12, dtype=np.float64), pay=False)
        w = cfg.rewards.team_advance_pbrs_weight
        phi_ball_0 = np.array(
            [w * env._compute_ball_potential(TEAM_A)] * 6 + [w * env._compute_ball_potential(TEAM_B)] * 6,
            dtype=np.float64,
        )
        phi_total_0 = phi_ball_0 + env.prev_tactical_potential.copy()
        gamma = cfg.ppo.gamma
        zeros = np.zeros((12, 4), dtype=np.float32)
        discounted = np.zeros(12, dtype=np.float64)
        for t in range(50):
            env.step_array(zeros)
            discounted += (gamma ** t) * env.last_shaping_rewards
        return discounted, phi_total_0, gamma

    disc_legacy, phi_0_leg, gamma = _setup_inert("legacy")
    disc_all, phi_0_all, _ = _setup_inert("all")
    np.testing.assert_allclose(phi_0_leg, phi_0_all, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(disc_legacy, 0.0, rtol=0.0, atol=1e-12)
    expected_excess = (1.0 - (gamma ** 50)) * phi_0_all
    assert expected_excess[5] > 0.0
    np.testing.assert_allclose(disc_legacy - disc_all, expected_excess, rtol=0.0, atol=1e-10)


def test_pbrs_t7_gamma_source_and_trainer_worker_propagation(tmp_path: Path) -> None:
    """T7: Stepping identical states with gamma=0.99 vs 0.995 differs by 0.005 * Phi(s'); trainer workers inherit gamma."""
    def _cfg(gamma: float) -> object:
        return replace(
            DEFAULT_CONFIG,
            ppo=replace(DEFAULT_CONFIG.ppo, gamma=gamma),
            replay=replace(DEFAULT_CONFIG.replay, enable_recording=False, save_dir=str(tmp_path / f"r_{gamma}")),
        )

    env_99 = FootballParallelEnv(_cfg(0.99))
    env_995 = FootballParallelEnv(_cfg(0.995))
    env_99.reset(seed=17)
    env_995.reset(seed=17)

    acts = np.random.default_rng(17).uniform(-1.0, 1.0, size=(12, 4)).astype(np.float32)
    env_99.step_array(acts)
    env_995.step_array(acts)

    w = DEFAULT_CONFIG.rewards.team_advance_pbrs_weight
    phi_ball_next = np.array(
        [w * env_995._compute_ball_potential(TEAM_A)] * 6
        + [w * env_995._compute_ball_potential(TEAM_B)] * 6,
        dtype=np.float64,
    )
    phi_total_next = phi_ball_next + env_995._phi_curr
    diff = env_995.last_shaping_rewards - env_99.last_shaping_rewards
    assert np.any(np.abs(diff) > 1e-4)
    np.testing.assert_allclose(diff, 0.005 * phi_total_next, rtol=0.0, atol=1e-12)

    # Trainer worker env propagation check (DummyVecFootballEnv and SubprocVecFootballEnv)
    t_dummy = ParameterSharingPPOTrainer(
        config=_cfg(0.99), num_envs=1, rollout_steps=8, device="cpu", checkpoint_dir=str(tmp_path / "c1"), seed=3
    )
    try:
        assert t_dummy.vec_env.envs[0].cfg.ppo.gamma == 0.99
    finally:
        t_dummy.close()

    t_99 = ParameterSharingPPOTrainer(
        config=_cfg(0.99), num_envs=2, rollout_steps=8, device="cpu", checkpoint_dir=str(tmp_path / "c2"), seed=3
    )
    t_995 = ParameterSharingPPOTrainer(
        config=_cfg(0.995), num_envs=2, rollout_steps=8, device="cpu", checkpoint_dir=str(tmp_path / "c3"), seed=3
    )
    try:
        t_99.vec_env.reset_array(seeds=[21, 22])
        t_995.vec_env.reset_array(seeds=[21, 22])
        batch_acts = np.stack([acts, acts], axis=0)
        _, _, rew_99, _, _ = t_99.vec_env.step_array(batch_acts)
        _, _, rew_995, _, _ = t_995.vec_env.step_array(batch_acts)
        worker_diff = rew_995[0].astype(np.float64) - rew_99[0].astype(np.float64)
        # Note: env_995 used seed 17 while worker 0 used seed 21; check with same seed 17:
        t_99.vec_env.reset_array(seeds=[17, 17])
        t_995.vec_env.reset_array(seeds=[17, 17])
        _, _, rew_99_s, _, _ = t_99.vec_env.step_array(batch_acts)
        _, _, rew_995_s, _, _ = t_995.vec_env.step_array(batch_acts)
        worker_diff = rew_995_s[0].astype(np.float64) - rew_99_s[0].astype(np.float64)
        np.testing.assert_allclose(worker_diff, 0.005 * phi_total_next, rtol=0.0, atol=1e-5)
    finally:
        t_99.close()
        t_995.close()


@pytest.mark.parametrize("shaping_mode", ["ball", "all"])
def test_ball_shaping_zero_sum_guard_across_teleports_and_terminal(tmp_path: Path, shaping_mode: str) -> None:
    """[GUARD] Ball-territory shaping summed across all 12 players is 0 to 1e-12 on every step."""
    cfg = _short_all_mode_config(tmp_path, shaping_mode=shaping_mode, gamma=0.995)
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)

    goal_seen = False
    halftime_seen = False
    done = False
    for t in range(cfg.physics.max_rl_steps):
        _, _, _, done, _ = env.step_array(_scripted_goal_halftime_actions(t))
        assert abs(float(env._tele_breakdown[0].sum())) <= 1e-12
        if env.stats["goals"] >= 1:
            goal_seen = True
        if env.current_half == 2:
            halftime_seen = True
        if done:
            break
    assert goal_seen and halftime_seen and done


def test_potential_state_markov_witness_guard() -> None:
    """[GUARD] Any PotentialState field change that alters Phi must also alter observations."""
    def _eval(env: FootballParallelEnv):
        stunned = env.stun_timer > 0.0
        state = PotentialState(
            pos=env.player_pos,
            angle=env.player_angle,
            ball=env.ball_pos,
            holder=-1 if env.ball_holder is None else env.ball_holder,
            ball_in_flight=env.ball_kicked_in_flight,
            last_ball_team=-1 if env.last_ball_team is None else env.last_ball_team,
            team_side=env.team_side,
            disabled=stunned | env.is_ghost,
            stunned=stunned,
            post_kick_timer=env.post_kick_pbrs_timer,
            depth_bias=env._tactical_depth_bias,
            flank_bias=env._tactical_flank_bias,
            active_box_radius=env.active_box_radius,
        )
        res = compute_potentials(state, env.cfg.rewards, env.cfg.physics)
        obs = env._get_observations_array()
        return res.phi, obs

    def _base_env() -> FootballParallelEnv:
        env = FootballParallelEnv(replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False)))
        env.reset(seed=11)
        env.ball_holder = None
        env.ball_kicked_in_flight = False
        env.last_ball_team = TEAM_A
        env.ball_pos = np.array([700.0, 420.0], dtype=np.float64)
        env.stun_timer[:] = 0.0
        env.ghost_timer[:] = 0.0
        env.is_ghost[:] = False
        env.post_kick_pbrs_timer[:] = 0.0
        return env

    failures = []
    cases = []

    # 1. holder
    b = _base_env()
    t = copy.deepcopy(b)
    t.ball_holder = 5
    cases.append(("holder", b, t))

    # 2. ball_in_flight
    b = _base_env()
    t = copy.deepcopy(b)
    t.ball_kicked_in_flight = True
    cases.append(("ball_in_flight", b, t))

    # 3. last_ball_team (with ball_in_flight=True)
    b = _base_env()
    b.ball_kicked_in_flight = True
    b.last_ball_team = TEAM_A
    t = copy.deepcopy(b)
    t.last_ball_team = TEAM_B
    cases.append(("last_ball_team", b, t))

    # 4. disabled (via ghosting without stunned; player 3 near ball and player 4 nearby)
    b = _base_env()
    b.player_pos[3] = np.array([650.0, 420.0], dtype=np.float64)
    b.player_pos[4] = np.array([640.0, 420.0], dtype=np.float64)
    t = copy.deepcopy(b)
    t.is_ghost[3] = True
    t.ghost_timer[3] = 1.0
    cases.append(("disabled", b, t))

    # 5. stunned (on non-ghosted player)
    t = copy.deepcopy(b)
    t.stun_timer[3] = 0.5
    cases.append(("stunned", b, t))

    # 6. post_kick_timer (with ball_in_flight=True, last_ball_team=TEAM_A; non-chaser teammate)
    b = _base_env()
    b.ball_kicked_in_flight = True
    b.last_ball_team = TEAM_A
    b.player_pos[5] = np.array([650.0, 420.0], dtype=np.float64)
    b.player_pos[1] = np.array([200.0, 200.0], dtype=np.float64)
    t = copy.deepcopy(b)
    t.post_kick_pbrs_timer[1] = 0.4
    cases.append(("post_kick_timer", b, t))

    # 7. active_box_radius (loose ball inside 195u left box, outside 100u box; keeper closer than outfielders)
    b = _base_env()
    b.ball_pos = np.array([150.0, 420.0], dtype=np.float64)
    b.player_pos[0] = np.array([110.0, 420.0], dtype=np.float64)
    b.player_pos[1] = np.array([250.0, 420.0], dtype=np.float64)
    b.active_box_radius = np.array([195.0, 195.0], dtype=np.float64)
    t = copy.deepcopy(b)
    t.active_box_radius = np.array([100.0, 195.0], dtype=np.float64)
    cases.append(("active_box_radius", b, t))

    # 8. angle (keeper angle when keeper bisector applies)
    b = _base_env()
    b.ball_pos = np.array([300.0, 300.0], dtype=np.float64)
    t = copy.deepcopy(b)
    t.player_angle[0] = math.pi / 2.0
    cases.append(("angle", b, t))

    for field_name, base_e, twin_e in cases:
        phi0, obs0 = _eval(base_e)
        phi1, obs1 = _eval(twin_e)
        phi_changed = np.where(np.abs(phi0 - phi1) > 1e-12)[0]
        assert len(phi_changed) > 0, f"Test setup for {field_name} did not alter Phi"
        for idx in phi_changed:
            if np.allclose(obs0[idx], obs1[idx], atol=1e-12):
                failures.append(f"{field_name}[agent={idx}]")

    assert failures == [], f"Markov witness failed for fields: {failures}"
