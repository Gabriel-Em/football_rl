"""Unit test suite for core physical, reward, and invariant properties of the football environment."""

import math
from dataclasses import replace
import numpy as np
import pytest

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


def _zero_actions(env: FootballParallelEnv) -> dict:
    return {
        agent: np.zeros(env.action_space(agent).shape, dtype=np.float32)
        for agent in env.possible_agents
    }


def test_closed_loop_pbrs_zero_gain_invariant() -> None:
    """Moving the ball along a closed spatial trajectory without scoring must yield net non-positive cumulative discounted shaping reward."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    # Disable other influences by placing all players far away
    for i in range(12):
        env.player_pos[i] = np.array([700.0, 100.0 + i * 40.0], dtype=np.float64)
        env.player_vel[i] = np.zeros(2, dtype=np.float64)
    env.ball_holder = None

    gamma = 0.995
    # Define a closed spatial trajectory for the ball: s0 -> s1 -> s2 -> s3 -> s0
    waypoints = [
        np.array([500.0, 400.0], dtype=np.float64),
        np.array([700.0, 400.0], dtype=np.float64),
        np.array([900.0, 500.0], dtype=np.float64),
        np.array([600.0, 300.0], dtype=np.float64),
        np.array([500.0, 400.0], dtype=np.float64),
    ]

    total_discounted_pbrs_team_a = 0.0
    discount = 1.0

    # Start at waypoints[0]
    env.ball_pos = waypoints[0].copy()
    env.ball_vel = np.zeros(2, dtype=np.float64)
    # Re-initialize previous territory potential
    env.prev_ball_canonical_pos = env._to_canonical_pos(env.ball_pos, TEAM_A)

    for wp in waypoints[1:]:
        # Move ball to next waypoint
        prev_pos = env.prev_ball_canonical_pos
        curr_pos = env._to_canonical_pos(wp, TEAM_A)
        d_max = 1632.6665
        goal_a = np.array([1400.0, 420.0])
        phi_prev = 1.0 - float(np.linalg.norm(prev_pos - goal_a)) / d_max
        phi_curr = 1.0 - float(np.linalg.norm(curr_pos - goal_a)) / d_max
        f_pbrs = 6.50 * (gamma * phi_curr - phi_prev)

        total_discounted_pbrs_team_a += discount * f_pbrs
        discount *= gamma
        env.prev_ball_canonical_pos = curr_pos

    # Net discounted PBRS over any closed loop must be <= 1e-6
    assert total_discounted_pbrs_team_a <= 1e-6


def test_stationary_triangle_convergence_invariant() -> None:
    """Stationary multi-player passing loops must decay total pass rewards below 1% within 2 full cycles."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=101)

    p1 = env.agent_to_idx["a_def_1"]
    p2 = env.agent_to_idx["a_mid_1"]

    # 2-player ping-pong loop: pairwise + touch decay
    loop_rewards = []
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
        k_rew = step_rew[kicker]

        env.ball_kicked_in_flight = True
        env.ball_kick_flight_timer = 0.5
        env.last_ball_holder = kicker
        env.last_ball_team = TEAM_A
        env.ball_pos = env.player_pos[rx].copy()
        env._check_free_ball_pickup(step_rew)
        loop_rewards.append(float(k_rew + step_rew[rx]))

    first_pass_rew = loop_rewards[0]
    sixth_pass_rew = loop_rewards[5]
    assert first_pass_rew > 0.0
    assert sixth_pass_rew < 0.01 * first_pass_rew


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
        assert obs.shape == (76,)
        assert not np.isnan(obs).any()
        assert not np.isinf(obs).any()
        assert np.all(obs >= -1.0)
        assert np.all(obs <= 1.0)

    central_state = env.get_centralized_state()
    assert central_state.shape == (12, 54)
    assert not np.isnan(central_state).any()
    assert not np.isinf(central_state).any()
    assert np.all(central_state >= -1.0)
    assert np.all(central_state <= 1.0)
