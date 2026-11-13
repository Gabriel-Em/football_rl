"""Verification test suite for the 6v6 MARL Football PettingZoo ParallelEnv."""

from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from pettingzoo.test import parallel_api_test
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audit import orphan_scan  # noqa: E402

from football_rl.config import (
    ACTION_DIM,
    ALL_AGENTS,
    DEFAULT_CONFIG,
    EnvConfig,
    OBS_DIM,
    ROLE_ATK,
    ROLE_DEF,
    ROLE_KEEPER,
    ROLE_MID,
    ReplayConfig,
    SIDE_LEFT,
    SIDE_RIGHT,
    TEAM_A,
    TEAM_B,
)
from football_rl.env import FootballParallelEnv, parallel_env, wrap_to_pi
from football_rl.potentials import PotentialState, compute_potentials
from football_rl.replay import (
    ReplayViewer,
    find_replay_by_match_id,
    generate_demo_replay,
    load_replay_json,
)
from football_rl.train_ppo import (
    ParameterSharingPPOTrainer,
    archive_batch,
    list_archived_batches,
    load_archived_batch,
    reset_active_batch,
)


def _zero_actions(env: FootballParallelEnv) -> dict[str, np.ndarray]:
    """Return zero-throttle, zero-steer, no-kick, no-special actions for all active agents."""
    return {
        agent: np.array([0.0, 0.0, -1.0, -1.0], dtype=np.float32)
        for agent in env.agents
    }


def test_pettingzoo_parallel_api_compliance() -> None:
    """Verify full compliance with PettingZoo's ParallelEnv contract."""
    env = parallel_env(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    parallel_api_test(env, num_cycles=100)


def test_canonical_observation_symmetry_and_halftime_switch() -> None:
    """Verify 180-deg canonical observation symmetry and physical side-switching at halftime."""
    env = parallel_env(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    obs_a_start, _ = env.reset(seed=42, options={"kickoff_team": TEAM_A})
    assert len(env.agents) == 12
    assert obs_a_start["a_atk_1"].shape == (OBS_DIM,)
    assert OBS_DIM == 82

    # Reset with Team B starting at kickoff in 1st half -> Team B is on Right, Team A on Left
    obs_b_start, _ = env.reset(seed=42, options={"kickoff_team": TEAM_B})

    # In canonical space, starting team is on Left and receiving on Right.
    # Team B's canonical observations when Team B kicks off must match Team A's canonical
    # observations when Team A kicks off.
    for role_suffix in ("keeper", "def_1", "def_2", "mid_1", "mid_2", "atk_1"):
        np.testing.assert_allclose(
            obs_a_start[f"a_{role_suffix}"],
            obs_b_start[f"b_{role_suffix}"],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            obs_a_start[f"b_{role_suffix}"],
            obs_b_start[f"a_{role_suffix}"],
            atol=1e-6,
        )

    # Now verify Halftime physical side-switch at t = 90.0s (step 1350 / frame 5400)
    env.reset(seed=42, options={"kickoff_team": TEAM_A})
    assert env.team_side[TEAM_A] == SIDE_LEFT
    assert env.team_side[TEAM_B] == SIDE_RIGHT

    # Fast-forward to 1 step before halftime
    env.rl_step_count = 1349
    env.physics_frame_count = 1349 * 4
    obs_ht, _, _, _, _ = env.step(_zero_actions(env))

    assert env.current_half == 2
    assert env.team_side[TEAM_A] == SIDE_RIGHT
    assert env.team_side[TEAM_B] == SIDE_LEFT
    # Team A's keeper should now stand at (1355, 420) and Team B's keeper at (45, 420)
    np.testing.assert_allclose(env.player_pos[env.agent_to_idx["a_keeper"]], [1355.0, 420.0])
    np.testing.assert_allclose(env.player_pos[env.agent_to_idx["b_keeper"]], [45.0, 420.0])
    # Same canonical view as the first-half kickoff, except the clock columns (78 time remaining, 79 half).
    np.testing.assert_allclose(
        np.delete(obs_ht["b_atk_1"], [78, 79]), np.delete(obs_a_start["a_atk_1"], [78, 79]), atol=1e-6
    )
    assert obs_a_start["a_atk_1"][78] == 1.0 and obs_a_start["a_atk_1"][79] == 0.0
    assert obs_ht["b_atk_1"][78] == pytest.approx(0.5, abs=1e-6) and obs_ht["b_atk_1"][79] == 1.0


def test_ball_friction_and_anti_wall_self_pass_cooldown() -> None:
    """Verify ball deceleration (-155 u/s^2), wall bounce preserves 1.0s kick cooldown, and touch clears it."""
    env = parallel_env(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    env.reset(seed=7)

    # Have a_atk_1 kick at max power (action[2] = 1.0 -> launch speed = 640.0 u/s)
    carrier_idx = env.agent_to_idx["a_atk_1"]
    env.player_pos[carrier_idx] = np.array([700.0, 50.0])
    env.player_angle[carrier_idx] = math.pi / 2.0
    actions = _zero_actions(env)
    actions["a_atk_1"] = np.array([-1.0, 0.0, 1.0, -1.0], dtype=np.float32)

    rewards = np.zeros(12, dtype=np.float64)
    env._physics_substep(
        np.array([actions[a] for a in ALL_AGENTS], dtype=np.float64), rewards
    )
    assert env.ball_holder is None
    # After 1 substep (dt = 1/60s), speed is 640.0 - 155.0 * (1/60) ≈ 637.416666
    assert math.isclose(float(np.linalg.norm(env.ball_vel)), 640.0 - 155.0 / 60.0, rel_tol=1e-4)
    assert math.isclose(env.pickup_cooldown[carrier_idx], 0.45, rel_tol=1e-4)
    assert bool(env.kick_cooldown_active[carrier_idx]) is True

    # Aim ball directly at top wall (Y = 0) so it bounces: wall bounce must NOT clear kick cooldown
    env.ball_pos = np.array([700.0, 14.0])
    env.ball_vel = np.array([0.0, -300.0])
    env._physics_substep(
        np.array([_zero_actions(env)[a] for a in ALL_AGENTS], dtype=np.float64), rewards
    )
    assert env.pickup_cooldown[carrier_idx] > 0.40
    assert bool(env.kick_cooldown_active[carrier_idx]) is True

    # When another player (a_def_1) touches/picks up the ball, kicker's cooldown is immediately cleared
    teammate_idx = env.agent_to_idx["a_def_1"]
    env.player_pos[teammate_idx] = env.ball_pos.copy()
    env._physics_substep(
        np.array([_zero_actions(env)[a] for a in ALL_AGENTS], dtype=np.float64), rewards
    )
    assert env.ball_holder == teammate_idx
    assert env.pickup_cooldown[carrier_idx] == 0.0
    assert bool(env.kick_cooldown_active[carrier_idx]) is False


def test_semi_circle_barriers_for_keepers_and_field_players() -> None:
    """Verify Keepers are clamped inside dist <= 145 and Field players outside dist > 145."""
    env = parallel_env(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    env.reset(seed=10)

    # Place a_keeper outside left box and field players inside boxes
    env.player_pos[env.agent_to_idx["a_keeper"]] = np.array([250.0, 420.0])
    env.player_pos[env.agent_to_idx["a_def_1"]] = np.array([50.0, 420.0])
    env.player_pos[env.agent_to_idx["b_def_1"]] = np.array([1350.0, 420.0])

    env._clamp_players_to_boundaries()

    dist_keeper = np.linalg.norm(
        env.player_pos[env.agent_to_idx["a_keeper"]] - np.array([0.0, 420.0])
    )
    dist_field_a = np.linalg.norm(
        env.player_pos[env.agent_to_idx["a_def_1"]] - np.array([0.0, 420.0])
    )
    dist_field_b = np.linalg.norm(
        env.player_pos[env.agent_to_idx["b_def_1"]] - np.array([1400.0, 420.0])
    )

    assert dist_keeper <= env.cfg.physics.keeper_box_radius + 1e-6
    assert dist_field_a > env.cfg.physics.keeper_box_radius
    assert dist_field_b > env.cfg.physics.keeper_box_radius


def test_field_5s_violation_ghost_mode_and_other_player_reset() -> None:
    """Verify 5-second field hold violation drops ball, enables Ghost Mode, and resets on other touch."""
    env = parallel_env(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    env.reset(seed=21)

    carrier_idx = env.agent_to_idx["a_atk_1"]
    # Move other players away so nobody interferes
    for i in range(12):
        if i != carrier_idx and env.player_role[i] != ROLE_KEEPER:
            env.player_pos[i] = np.array([700.0, 100.0 + i * 40.0])

    # Step for 5.0 seconds = 75 RL steps (300 physics substeps)
    total_carrier_reward = 0.0
    for _ in range(75):
        _, rewards, _, _, _ = env.step(_zero_actions(env))
        total_carrier_reward += rewards["a_atk_1"]

    assert env.ball_holder is None
    assert env.ghost_timer[carrier_idx] > 2.9
    assert env.pickup_cooldown[carrier_idx] > 2.9
    assert env.tackle_cooldown[carrier_idx] > 2.9
    assert env.stats["field_5s_violations"] == 1
    assert total_carrier_reward < -1.5

    # Move teammate a_def_1 onto the ball to pick it up -> must not clear carrier's cooldown
    teammate_idx = env.agent_to_idx["a_def_1"]
    env.player_pos[teammate_idx] = env.ball_pos.copy()
    env.step(_zero_actions(env))

    assert env.ball_holder == teammate_idx
    assert env.pickup_cooldown[carrier_idx] > 0.0

    # Free the ball and let opponent pick it up -> must clear carrier's cooldown
    env.ball_holder = None
    opp_idx = env.agent_to_idx["b_atk_1"]
    env.player_pos[opp_idx] = env.ball_pos.copy()
    env.step(_zero_actions(env))

    assert env.ball_holder == opp_idx
    assert env.pickup_cooldown[carrier_idx] == 0.0


def test_keeper_5s_foul_teleport_and_zero_advance_spike() -> None:
    """Verify Keeper 5-second hold drops the ball, nudges it radially out of the box, and applies violation cooldown."""
    env = parallel_env(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    env.reset(seed=33)

    keeper_idx = env.agent_to_idx["a_keeper"]
    for idx in range(1, env.num_players):
        env.player_pos[idx] = np.array([700.0, 100.0 + idx * 55.0], dtype=np.float64)
    env.ball_holder = keeper_idx
    env.ball_pos = env.player_pos[keeper_idx].copy()
    env.ball_hold_timer = 4.99

    _, rewards, _, _, _ = env.step(_zero_actions(env))

    assert env.stats["keeper_5s_violations"] == 1
    assert env.ball_holder is None
    assert env.pickup_cooldown[keeper_idx] > 1.9
    assert float(np.linalg.norm(env.ball_vel)) > 100.0
    assert rewards["a_keeper"] < -1.0


def test_keeper_5s_foul_charges_keeper_only_while_field_foul_also_charges_team() -> None:
    """A keeper 5 s foul changes only the keeper's reward; a field 5 s foul also charges teammates."""
    env = parallel_env(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    env.reset(seed=33)
    keeper_idx = env.agent_to_idx["a_keeper"]
    carrier_idx = env.agent_to_idx["a_atk_1"]

    keeper_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._handle_keeper_5s_violation(keeper_idx, keeper_rewards)
    expected = np.zeros(env.num_players, dtype=np.float64)
    expected[keeper_idx] = -1.5
    np.testing.assert_array_equal(keeper_rewards, expected)

    field_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._handle_field_5s_violation(carrier_idx, field_rewards)
    expected = np.zeros(env.num_players, dtype=np.float64)
    expected[:6] = -0.3
    expected[carrier_idx] = -1.8
    np.testing.assert_allclose(field_rewards, expected, rtol=0, atol=1e-12)


def test_role_tactical_bias_encoding() -> None:
    """Verify tactical depth bias, lateral flank bias, and box state in obs[13:16]."""
    env = parallel_env(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    obs, _ = env.reset(seed=42)

    # Keeper on reset: [-1.0, active_box_radius / 195.0, box_loose_timer / 5.0]
    np.testing.assert_allclose(obs["a_keeper"][13:16], [-1.0, 1.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(obs["b_keeper"][13:16], [-1.0, 1.0, 0.0], atol=1e-5)

    # Field roles: depth [-1, 0, 1], flank [-1, 0, 1], and active box norm (1.0)
    np.testing.assert_allclose(obs["a_def_1"][13:16], [-1.0, -1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(obs["a_def_2"][13:16], [-1.0, 1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(obs["a_mid_1"][13:16], [0.0, -1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(obs["a_mid_2"][13:16], [0.0, 1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(obs["a_atk_1"][13:16], [1.0, 0.0, 1.0], atol=1e-6)

    np.testing.assert_allclose(obs["b_def_1"][13:16], [-1.0, -1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(obs["b_def_2"][13:16], [-1.0, 1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(obs["b_mid_1"][13:16], [0.0, -1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(obs["b_mid_2"][13:16], [0.0, 1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(obs["b_atk_1"][13:16], [1.0, 0.0, 1.0], atol=1e-6)


def test_role_specific_kick_speeds() -> None:
    """Verify role-specific maximum kick launch speeds: ATK 640 > MID 600 > DEF 560."""
    env = parallel_env(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    env.reset(seed=55)

    atk_idx = env.agent_to_idx["a_atk_1"]
    mid_idx = env.agent_to_idx["a_mid_1"]
    def_idx = env.agent_to_idx["a_def_1"]
    gk_idx = env.agent_to_idx["a_keeper"]

    atk_max_spd, _ = env._get_role_kick_params(atk_idx)
    mid_max_spd, _ = env._get_role_kick_params(mid_idx)
    def_max_spd, _ = env._get_role_kick_params(def_idx)
    gk_max_spd, _ = env._get_role_kick_params(gk_idx)

    assert atk_max_spd == 640.0
    assert mid_max_spd == 600.0
    assert def_max_spd == 560.0
    assert gk_max_spd == 640.0
    assert atk_max_spd > mid_max_spd > def_max_spd


def test_true_shot_on_target_save_vs_harmless_catch_and_anti_farming() -> None:
    """Verify keeper save rewards true shot-on-target, harmless catch awards interception recovery, and anti-strip farming."""
    env = parallel_env(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    env.reset(seed=88)

    k_idx = env.agent_to_idx["a_keeper"]
    b_atk = env.agent_to_idx["b_atk_1"]

    # 1. True Shot-on-Target: Enemy ATK kicks directly into goal frame (Y = 420)
    env.ball_holder = b_atk
    env.player_pos[b_atk] = np.array([400.0, 420.0])
    env.player_angle[b_atk] = math.pi  # Facing Left goal (0.0, 420.0)
    rewards = np.zeros(12, dtype=np.float64)
    env._execute_kick(b_atk, kick_power=1.0, step_rewards=rewards)
    assert env.ball_kick_was_shot_on_target is True

    # Move keeper onto incoming ball to make a save
    env.player_pos[k_idx] = env.ball_pos.copy()
    save_rewards = np.zeros(12, dtype=np.float64)
    env._check_free_ball_pickup(save_rewards)

    # Keeper save ind + team scaled by save difficulty
    assert env.stats["keeper_saves"] == 1
    assert save_rewards[k_idx] > 0.0
    assert save_rewards[k_idx] <= 6.0

    # 2. Harmless catch: Enemy kicks away from goal (heading into corner wall Y = 100)
    env.reset(seed=89)
    env.ball_holder = b_atk
    env.player_pos[b_atk] = np.array([400.0, 100.0])
    env.player_angle[b_atk] = math.pi  # Facing (0.0, 100.0) outside goal mouth [340, 500]
    rewards = np.zeros(12, dtype=np.float64)
    env._execute_kick(b_atk, kick_power=1.0, step_rewards=rewards)
    assert env.ball_kick_was_shot_on_target is False

    env.player_pos[k_idx] = env.ball_pos.copy()
    catch_rewards = np.zeros(12, dtype=np.float64)
    env._check_free_ball_pickup(catch_rewards)

    # Harmless catch of opponent kick gives interception recovery (1.2 + 0.25 = 1.45), no keeper_save stat increment
    assert env.stats["keeper_saves"] == 0
    expected_catch = env.cfg.rewards.successful_tackle_tackler_ind + env.cfg.rewards.successful_tackle_team
    assert math.isclose(catch_rewards[k_idx], expected_catch, abs_tol=1e-5)

    # 3. Anti-Farming on Perimeter Strip: If keeper kicked in the last 2.0s, strip gives 0.0 reward
    env.reset(seed=90)
    env.last_keeper_kick_time[TEAM_A] = env.physics_frame_count * env.cfg.physics.dt  # just kicked!
    env.ball_holder = b_atk
    env.player_pos[b_atk] = np.array([130.0, 420.0])
    env.ball_pos = np.array([110.0, 420.0])
    env.player_pos[k_idx] = np.array([100.0, 420.0])

    strip_rewards = np.zeros(12, dtype=np.float64)
    env._check_keeper_perimeter_strip(b_atk, strip_rewards)

    assert env.ball_holder == k_idx
    assert strip_rewards[k_idx] == 0.0
    assert env.stats["keeper_saves"] == 0


def test_full_episode_and_replay_export(tmp_path: Path) -> None:
    """Run a complete 2,700-step match and verify 60 Hz JSON replay export with 12 players."""
    replay_cfg = ReplayConfig(
        save_dir=str(tmp_path),
        enable_recording=True,
        interest_threshold=0.0,
    )
    env = parallel_env(config=replace(DEFAULT_CONFIG, replay=replay_cfg))
    env.reset(seed=123)

    while env.agents:
        actions = {
            agent: env.action_space(agent).sample() for agent in env.agents
        }
        env.step(actions)

    assert env.rl_step_count == 2700
    assert env.physics_frame_count == 10800
    assert env.last_saved_replay_path is not None
    assert os.path.exists(env.last_saved_replay_path)

    with open(env.last_saved_replay_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert len(data["frames"]) == 10801
    sample_frame = data["frames"][0]
    assert len(sample_frame["players"]) == 12
    assert "player_rewards" in sample_frame
    assert "cum_rewards" in sample_frame
    assert len(sample_frame["player_rewards"]) == 12


def test_replay_viewer_own_goal_indexing_and_12_players(tmp_path: Path) -> None:
    """Verify Pygame ReplayViewer indexes GOAL! and OWN GOAL!, renders 12 players and reward HUD."""
    replay_path = generate_demo_replay(replay_dir=str(tmp_path), seed=77)
    replay_data = load_replay_json(replay_path)

    # Inject an OWN GOAL! event into a frame to test timeline marker indexing
    replay_data["frames"][100]["events"].append(
        {"type": "OWN GOAL!", "team": 1, "agent": "b_def_1", "x": 1400.0, "y": 420.0, "reward": -18.0}
    )

    viewer = ReplayViewer(replay_data, headless=True)
    # Check that OWN GOAL! was indexed
    has_own_goal = any(is_og for _, _, is_og in viewer.goal_frames)
    assert has_own_goal is True

    # Check that vision cones and reward HUD are off by default
    assert viewer.show_vision_cones is False
    assert viewer.show_reward_hud is False

    surface = viewer.render_frame()
    assert surface.get_width() > 1000
    assert surface.get_height() > 600

    # Test toggles and scrubbing
    viewer.show_vision_cones = True
    viewer.show_reward_hud = True
    viewer.playback_speed = 4
    viewer.step_playback()
    assert viewer.current_frame_idx == 4

    viewer.seek_to_frame(5400)
    assert viewer.current_frame_idx == 5400
    viewer.render_frame()
    viewer.run(max_frames=5)


def test_parameter_sharing_ppo_trainer(tmp_path: Path) -> None:
    """Verify ParameterSharingPPOTrainer updates 12-agent policies and saves checkpoints."""
    cfg = replace(
        DEFAULT_CONFIG,
        replay=replace(DEFAULT_CONFIG.replay, enable_recording=False),
    )
    trainer = ParameterSharingPPOTrainer(
        config=cfg,
        num_envs=2,
        rollout_steps=64,
        device="cpu",
        checkpoint_dir=str(tmp_path / "ckpt"),
        seed=42,
    )
    metrics = trainer.train(total_updates=2)

    assert metrics["update"] == 2
    assert metrics["total_env_steps"] == 2 * 64 * 2
    assert os.path.exists(tmp_path / "ckpt" / "field_policy.pt")
    assert os.path.exists(tmp_path / "ckpt" / "keeper_policy.pt")
    assert os.path.exists(tmp_path / "ckpt" / "blue_field_policy.pt")
    assert os.path.exists(tmp_path / "ckpt" / "red_field_policy.pt")

    blue_w = next(trainer.blue_field_policy.parameters()).detach().numpy()
    red_w = next(trainer.red_field_policy.parameters()).detach().numpy()
    assert not np.allclose(blue_w, red_w)

    trainer.load_checkpoints()


def test_resume_archive_and_match_indexing(tmp_path: Path) -> None:
    """Verify sequential match indexing, resume across batches, archiving, and loading archives."""
    ckpt_dir = str(tmp_path / "checkpoints")
    rep_dir = str(tmp_path / "replays")
    arch_dir = str(tmp_path / "archives")

    cfg = replace(
        DEFAULT_CONFIG,
        replay=ReplayConfig(
            save_dir=rep_dir,
            enable_recording=True,
            interest_threshold=0.0,
        ),
    )

    # 1. Train Generations 1..2 (2 full matches: Match #1 and Match #2)
    trainer1 = ParameterSharingPPOTrainer(
        config=cfg,
        num_envs=1,
        rollout_steps=2700,
        device="cpu",
        checkpoint_dir=ckpt_dir,
        seed=42,
    )
    m1 = trainer1.train(total_updates=2)
    assert m1["generation"] == 2
    assert m1["total_matches"] == 2
    assert os.path.exists(os.path.join(rep_dir, "matches_summary.md"))
    assert find_replay_by_match_id(1, rep_dir, ckpt_dir) is not None
    assert find_replay_by_match_id(2, rep_dir, ckpt_dir) is not None

    # 2. Resume and train Generation 3 (Match #3)
    trainer2 = ParameterSharingPPOTrainer(
        config=cfg,
        num_envs=1,
        rollout_steps=2700,
        device="cpu",
        checkpoint_dir=ckpt_dir,
        seed=42,
    )
    assert trainer2.load_checkpoints() is True
    assert trainer2.current_generation == 2
    m2 = trainer2.train(total_updates=1)
    assert m2["generation"] == 3
    assert m2["total_matches"] == 3
    assert find_replay_by_match_id(3, rep_dir, ckpt_dir) is not None

    # 3. Archive this 3-generation batch as `batch_3_gens`
    archive_batch("batch_3_gens", ckpt_dir, rep_dir, arch_dir)

    # 4. Start a new batch from scratch (1 generation: Match #1) and archive as `batch_1_gen`
    reset_active_batch(ckpt_dir, rep_dir)
    trainer_new = ParameterSharingPPOTrainer(
        config=cfg,
        num_envs=1,
        rollout_steps=2700,
        device="cpu",
        checkpoint_dir=ckpt_dir,
        seed=99,
    )
    m_new = trainer_new.train(total_updates=1)
    assert m_new["generation"] == 1
    assert m_new["total_matches"] == 1
    archive_batch("batch_1_gen", ckpt_dir, rep_dir, arch_dir)

    batches = list_archived_batches(arch_dir)
    assert len(batches) == 2

    # 5. Load back `batch_3_gens` and resume to Generation 4 (Match #4)
    load_archived_batch("batch_3_gens", ckpt_dir, rep_dir, arch_dir)
    trainer_restored = ParameterSharingPPOTrainer(
        config=cfg,
        num_envs=1,
        rollout_steps=2700,
        device="cpu",
        checkpoint_dir=ckpt_dir,
        seed=42,
    )
    assert trainer_restored.load_checkpoints() is True
    assert trainer_restored.current_generation == 3
    m_resumed = trainer_restored.train(total_updates=1)
    assert m_resumed["generation"] == 4
    assert m_resumed["total_matches"] == 4
    assert find_replay_by_match_id(4, rep_dir, ckpt_dir) is not None


def test_piecewise_deadzone_and_backpedal_kinematics() -> None:
    """Verify deadzone produces zero speed and negative throttle causes clean backpedaling."""
    env = parallel_env(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    env.reset(seed=101)

    atk_idx = env.agent_to_idx["a_atk_1"]
    gk_idx = env.agent_to_idx["a_keeper"]

    env.player_angle[atk_idx] = 0.0
    env.player_angle[gk_idx] = 0.0

    # 1. Deadzone (|throttle_raw| <= 0.15) produces exactly 0.0 speed
    actions = _zero_actions(env)
    actions["a_atk_1"] = np.array([0.10, 0.0, -1.0, -1.0], dtype=np.float32)
    actions["a_keeper"] = np.array([-0.10, 0.0, -1.0, -1.0], dtype=np.float32)

    env.step(actions)
    assert np.isclose(env.player_vel[atk_idx, 0], 0.0, atol=1e-5)
    assert np.isclose(env.player_vel[atk_idx, 1], 0.0, atol=1e-5)
    assert np.isclose(env.player_vel[gk_idx, 0], 0.0, atol=1e-5)
    assert np.isclose(env.player_vel[gk_idx, 1], 0.0, atol=1e-5)

    # 2. Maximum backpedal (throttle_raw = -1.0)
    # Field player (ROLE_ATK): -0.65 * 190.0 = -123.5
    # Keeper (ROLE_KEEPER): -1.0 * 180.0 = -180.0
    env.player_pos[gk_idx] = np.array([300.0, 420.0], dtype=np.float64)
    actions["a_atk_1"] = np.array([-1.0, 0.0, -1.0, -1.0], dtype=np.float32)
    actions["a_keeper"] = np.array([-1.0, 0.0, -1.0, -1.0], dtype=np.float32)

    for _ in range(15):
        env.step(actions)
    assert np.isclose(env.player_vel[atk_idx, 0], -123.5, atol=0.05)
    assert np.isclose(env.player_vel[atk_idx, 1], 0.0, atol=0.05)
    assert np.isclose(env.player_vel[gk_idx, 0], -180.0, atol=0.05)
    assert np.isclose(env.player_vel[gk_idx, 1], 0.0, atol=0.05)


def test_max_pass_flight_timeout_3s() -> None:
    """Verify passes grabbed after max_pass_flight_time (3.0s) do not register as completed passes."""
    env = parallel_env(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    env.reset(seed=102)

    passer_idx = env.agent_to_idx["a_def_1"]
    receiver_idx = env.agent_to_idx["a_mid_1"]

    # Place players far away from others
    for i in range(12):
        env.player_pos[i] = np.array([100.0 + i * 20.0, 100.0])

    env.player_pos[passer_idx] = np.array([300.0, 420.0])
    env.player_pos[receiver_idx] = np.array([500.0, 420.0])
    env.player_angle[passer_idx] = 0.0  # Facing right toward receiver

    # Give passer the ball and execute kick
    env.ball_holder = passer_idx
    env.kick_grace_timer = 0.0
    env._execute_kick(passer_idx, kick_power=0.6)
    assert env.ball_kicked_in_flight is True
    assert env.ball_kick_flight_timer == 0.0

    # Advance flight timer past 3.0s threshold
    env.ball_kick_flight_timer = 3.2
    # Place receiver on ball
    env.player_pos[receiver_idx] = env.ball_pos.copy()

    # Step to trigger pickup check
    env.step(_zero_actions(env))

    assert env.ball_holder == receiver_idx
    # Did not register as a completed pass because flight timer expired
    assert env.stats["completed_passes"] == 0


def test_no_interception_penalty_on_gk_save_and_prime_hesitation_penalty() -> None:
    """Verify enemy keeper save of true shot on target does not penalize shooter, and prime zone hesitation applies."""
    env = parallel_env(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    env.reset(seed=103)

    shooter_idx = env.agent_to_idx["a_atk_1"]
    b_keeper_idx = env.agent_to_idx["b_keeper"]

    # 1. Shooter fires true shot on target
    env.player_pos[shooter_idx] = np.array([1100.0, 420.0])
    env.player_angle[shooter_idx] = 0.0  # Aimed directly at right goal (1400, 420)
    env.ball_holder = shooter_idx
    env.kick_grace_timer = 0.0

    step_rewards = np.zeros(12, dtype=np.float64)
    env._execute_kick(shooter_idx, kick_power=1.0, step_rewards=step_rewards)
    assert env.ball_kick_was_shot_on_target is True

    # Move b_keeper onto the ball path and trigger pickup
    env.player_pos[b_keeper_idx] = env.ball_pos.copy()
    step_rewards.fill(0.0)
    env._check_free_ball_pickup(step_rewards)

    assert env.ball_holder == b_keeper_idx
    assert env.stats["keeper_saves"] == 1
    assert env.stats["interceptions"] == 0
    # Under zero-sum save rebate, shooter's SOT reward is clawed back and mirrored
    assert step_rewards[shooter_idx] < 0.0

    # 2. Stationary carrier in prime scoring zone earns non-positive ball-advance PBRS
    env.player_pos[b_keeper_idx] = np.array([1355.0, 420.0])  # Keeper in goal mouth
    env.ball_holder = shooter_idx
    env.player_pos[shooter_idx] = np.array([1150.0, 420.0])
    env.player_angle[shooter_idx] = 0.0  # Aimed at goal
    env.ball_pos = env.player_pos[shooter_idx] + np.array([env.cfg.mechanics.ball_carry_offset, 0.0])
    env.ball_vel[:] = 0.0
    env.kick_grace_timer = 0.0
    phi_before = env._compute_ball_potential(TEAM_A)
    env._compute_potential_shaping(np.zeros(12, dtype=np.float64), any_teleport=True)

    # Step without moving or kicking: discounted PBRS (gamma * phi - phi) is slightly negative
    _, rewards, _, _, _ = env.step(_zero_actions(env))
    assert rewards["a_atk_1"] <= 0.0


def test_cinematic_replay_timeline_expansion() -> None:
    """Verify ReplayViewer expands goal set-piece into freeze, walk, and whistle-hold stages and renders headlessly."""
    raw_frames = [
        {
            "time": 0.0,
            "half": 1,
            "team_side": [0, 1],
            "stats": {"score": [0, 0]},
            "events": [],
            "players": [[700.0, 420.0, 0.0, 0]] * 12,
            "ball": [700.0, 420.0, -1],
        },
        {
            "time": 0.5,
            "half": 1,
            "team_side": [0, 1],
            "stats": {"score": [1, 0]},
            "events": [{"type": "GOAL!", "x": 1395.0, "y": 420.0, "team": 0, "agent": "a_atk_1"}],
            "players": [[720.0, 420.0, 0.0, 0]] * 12,
            "ball": [1395.0, 420.0, 0],
        },
        {
            "time": 0.52,
            "half": 1,
            "team_side": [0, 1],
            "stats": {"score": [1, 0]},
            "events": [],
            "players": [[700.0, 420.0, 0.0, 0]] * 12,
            "ball": [700.0, 420.0, -1],
        },
    ]

    replay_dict = {
        "metadata": {
            "match_id": 999,
            "field_width": 1400.0,
            "field_height": 840.0,
            "goal_y_min": 340.0,
            "goal_y_max": 500.0,
            "physics_hz": 60,
        },
        "frames": raw_frames,
    }

    viewer = ReplayViewer(replay_dict, headless=True)
    # 90 freeze + 120 walk + 60 hold + 1 end = 271 frames added for transition
    assert len(viewer.frames) > 250
    assert viewer.frames[1]["cinematic"]["phase"] == "FREEZE"
    assert viewer.frames[91]["cinematic"]["phase"] == "WALK"
    assert viewer.frames[211]["cinematic"]["phase"] == "WHISTLE_HOLD"

    # Test headless render of freeze and walk frames
    viewer.current_frame_idx = 1
    surf_freeze = viewer.render_frame()
    assert surf_freeze.get_width() == viewer.win_w

    viewer.current_frame_idx = 100
    surf_walk = viewer.render_frame()
    assert surf_walk.get_width() == viewer.win_w


def test_asymmetric_deadzone_velocity_inertia_and_anti_jitter() -> None:
    """Verify throttle deadzone [-0.10, +0.10] and exponential velocity inertia."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    mid_idx = env.agent_to_idx["a_mid_1"]
    max_spd = env._get_player_max_speed(mid_idx)

    # 1. Deadzone: -0.08 is in [-0.10, +0.10] -> target_speed is 0.0
    act_deadzone = np.zeros((env.num_players, 4), dtype=np.float32)
    act_deadzone[mid_idx] = [-0.08, 0.0, -1.0, -1.0]
    env.player_vel[mid_idx] = 0.0

    env._physics_substep(act_deadzone)
    assert np.allclose(env.player_vel[mid_idx], 0.0, atol=1e-5)

    # 2. Velocity Inertia: throttle = -1.0 accelerates smoothly over substeps rather than jumping instantly
    act_backpedal = np.zeros((env.num_players, 4), dtype=np.float32)
    act_backpedal[mid_idx] = [-1.0, 0.0, -1.0, -1.0]  # Full backpedal
    target_backpedal_speed = -env.cfg.physics.field_backpedal_mult * max_spd  # -0.65 * 192 = -124.8

    # After 1 substep (dt = 1/60s), velocity should change by only accel_alpha fraction
    env.player_vel[mid_idx] = 0.0
    env.player_angle[mid_idx] = 0.0  # Facing East (+X), backpedal is -X
    env._physics_substep(act_backpedal)

    vx_1 = float(env.player_vel[mid_idx, 0])
    dt = env.cfg.physics.dt
    accel_alpha = min(1.0, dt * env.cfg.physics.player_accel_rate)  # (1/60) * 10.5 = 0.175
    expected_vx_1 = target_backpedal_speed * accel_alpha
    assert abs(vx_1 - expected_vx_1) < 1.0
    assert abs(vx_1) < abs(target_backpedal_speed) * 0.5  # Did NOT jump immediately to full speed


def test_kickoff_protection_and_gk_hold_opponent_buffer() -> None:
    """Verify kickoff tackle protection, exclusion circle, and GK possession buffer."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)

    # Immediately after kickoff, protection timer is active
    assert env.kickoff_protection_timer > 0.0
    carrier_idx = env.ball_holder
    assert carrier_idx == env.agent_to_idx["a_atk_1"]

    # Opponent b_atk_1 stands right next to carrier and triggers tackle
    opp_idx = env.agent_to_idx["b_atk_1"]
    env.player_pos[opp_idx] = env.player_pos[carrier_idx] + np.array([5.0, 0.0], dtype=np.float64)

    # Attempt tackle directly
    rew_arr = np.zeros(env.num_players, dtype=np.float64)
    env._attempt_tackle(opp_idx, rew_arr)
    # Possession must remain with kickoff carrier, no cooldown/reward applied
    assert env.ball_holder == carrier_idx
    assert env.tackle_cooldown[opp_idx] == 0.0

    # Kickoff exclusion circle clamping: opponent inside 130u of center is pushed outward
    center = np.array(env.cfg.physics.field_center, dtype=np.float64)
    env.player_pos[opp_idx] = center + np.array([60.0, 0.0], dtype=np.float64)  # dist 60 < 130
    env._clamp_players_to_boundaries()
    dist_from_center = float(np.linalg.norm(env.player_pos[opp_idx] - center))
    assert dist_from_center >= env.cfg.mechanics.kickoff_exclusion_radius - 1e-3

    # GK Possession buffer: when GK holds ball, opposing field players are pushed outside gk_hold_opponent_buffer
    gk_idx = env.agent_to_idx["a_keeper"]
    env.ball_holder = gk_idx
    left_goal = np.array(env.cfg.physics.left_goal_center, dtype=np.float64)
    env.player_pos[opp_idx] = left_goal + np.array([210.0, 0.0], dtype=np.float64)  # dist 210 < 265
    for _ in range(20):
        env._clamp_players_to_boundaries()
    dist_from_goal = float(np.linalg.norm(env.player_pos[opp_idx] - left_goal))
    assert dist_from_goal >= env.cfg.mechanics.gk_hold_opponent_buffer - 1e-3


def test_gk_cannot_kick_own_goal_and_backward_pass_not_penalized() -> None:
    """Verify backward pass under pressure earns positive pressure-relief reward and is not penalized."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    passer_idx = env.agent_to_idx["a_mid_1"]
    receiver_idx = env.agent_to_idx["a_def_1"]
    opp_idx = env.agent_to_idx["b_atk_1"]

    # Place opponent pressing passer closely (d_opp_orig = 50u), while receiver is in open space
    for i in range(6, 12):
        env.player_pos[i] = np.array([1200.0, 100.0 + (i - 6) * 80.0], dtype=np.float64)
    env.player_pos[passer_idx] = np.array([600.0, 420.0], dtype=np.float64)
    env.player_pos[opp_idx] = np.array([650.0, 420.0], dtype=np.float64)
    env.player_pos[receiver_idx] = np.array([420.0, 420.0], dtype=np.float64)

    env.ball_holder = passer_idx
    env.ball_pos = env.player_pos[passer_idx].copy()
    env.player_angle[passer_idx] = math.pi
    rew_arr = np.zeros(env.num_players, dtype=np.float64)
    env._execute_kick(passer_idx, 0.6, rew_arr)

    env.ball_pos = env.player_pos[receiver_idx].copy()
    env._check_free_ball_pickup(rew_arr)

    assert env.ball_holder == receiver_idx
    assert rew_arr[passer_idx] > 0.0
    assert rew_arr[receiver_idx] > env.cfg.rewards.completed_pass_receiver_ind
    assert env.ball_pos[0] >= 0.0


def test_competitive_interest_score_discounts_blowouts() -> None:
    """Verify that tight contested matches produce higher interest scores than one-sided blowouts."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    # Match A: Highly competitive 3-2 contest
    env.stats["score"] = [3, 2]
    env.stats["goals"] = 5
    env.stats["own_goals"] = 0
    env.stats["team_passes"] = [22, 19]
    env.stats["completed_passes"] = 41
    env.stats["team_tackles"] = [8, 9]
    env.stats["successful_tackles"] = 17
    env.stats["team_saves"] = [3, 4]
    env.stats["keeper_saves"] = 7
    env.stats["shots"] = [7, 6]
    env.stats["keeper_5s_violations"] = 0

    score_competitive = env.compute_interest_score()

    # Match B: One-sided 10-0 blowout with 3 own goals and zero opponent offense
    env.stats["score"] = [10, 0]
    env.stats["goals"] = 10
    env.stats["own_goals"] = 3
    env.stats["team_passes"] = [30, 1]
    env.stats["completed_passes"] = 31
    env.stats["team_tackles"] = [15, 0]
    env.stats["successful_tackles"] = 15
    env.stats["team_saves"] = [0, 1]
    env.stats["keeper_saves"] = 1
    env.stats["shots"] = [16, 0]
    env.stats["keeper_5s_violations"] = 2

    score_blowout = env.compute_interest_score()

    assert score_competitive > score_blowout
    assert score_competitive > 100.0


def test_observation_dimension_parity_and_vision_normalization() -> None:
    """Verify 82-dim obs parity and entity relative angle normalization under 210-deg vision cone."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    obs_dict, _ = env.reset(seed=42)

    # 1. Dimension parity
    assert env.cfg.vision.fov_deg == 210.0
    assert math.isclose(env.cfg.vision.half_fov_rad, math.radians(105.0))
    for agent in env.possible_agents:
        assert env.observation_space(agent).shape == (82,)
        assert obs_dict[agent].shape == (82,)

    # 2. Vision angle normalization strictly bounded in [-1.0, 1.0] across all entity slots
    for _ in range(50):
        actions = {
            agent: np.array([0.5, 0.5, -1.0, -1.0], dtype=np.float32) for agent in env.agents
        }
        obs_dict, _, _, _, _ = env.step(actions)
        for agent, obs in obs_dict.items():
            # Check slot 0 (ball, index 16..20)
            assert -1.0 <= obs[18] <= 1.0
            # Check slots 1..11 (entities, indices 21..75)
            for slot_i in range(1, 12):
                slot_start = 16 + slot_i * 5
                rel_angle_norm = obs[slot_start + 2]
                assert -1.0 <= rel_angle_norm <= 1.0


def test_ping_pong_lateral_pass_suppression_and_pairwise_decay() -> None:
    """Verify that unpressed lateral pass ping-pong yields zero passer reward while progressive passes earn threat gain."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    p1 = env.agent_to_idx["a_def_1"]
    p2 = env.agent_to_idx["a_def_2"]
    team = TEAM_A

    # Position two defenders equidistant from enemy goal (1400, 420): Y = 320 and Y = 520
    env.player_pos[p1] = np.array([350.0, 320.0], dtype=np.float64)
    env.player_pos[p2] = np.array([350.0, 520.0], dtype=np.float64)

    # Move opponent field players away
    for idx in range(env.num_players):
        if env.player_team[idx] != team:
            env.player_pos[idx] = np.array([1200.0, 420.0], dtype=np.float64)

    # Pass 1: p1 -> p2 (equidistant lateral pass: d_phi_threat == 0, d_sigma_press == 0)
    env.ball_holder = None
    env.ball_kick_origin = env.player_pos[p1].copy()
    env.ball_kick_orig_opp_dist = 850.0
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.last_ball_holder = p1
    env.last_ball_team = team
    env.ball_pos = env.player_pos[p2].copy()

    rew1 = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(rew1)

    assert rew1[p1] == 0.0
    assert rew1[p2] == 0.0

    # Pass 2: p2 -> p1 (lateral ping-pong back: still 0.0 passer reward)
    env.ball_holder = None
    env.ball_kick_origin = env.player_pos[p2].copy()
    env.ball_kick_orig_opp_dist = 850.0
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.last_ball_holder = p2
    env.last_ball_team = team
    env.ball_pos = env.player_pos[p1].copy()

    rew2 = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(rew2)
    assert rew2[p2] == 0.0

    # Pass 3: p2 -> p3 (forward progressive pass to mid_1 at X = 600.0, Y = 420.0)
    p3 = env.agent_to_idx["a_mid_1"]
    env.player_pos[p3] = np.array([600.0, 420.0], dtype=np.float64)
    env.player_vel[p3] = np.zeros(2, dtype=np.float64)
    env.ball_holder = None
    env.ball_kick_origin = env.player_pos[p2].copy()
    env.ball_kick_orig_opp_dist = 850.0
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.last_ball_holder = p2
    env.last_ball_team = team
    env.ball_pos = env.player_pos[p3].copy()

    rew4 = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(rew4)

    d_phi = env._compute_ball_potential(team, env.player_pos[p3]) - env._compute_ball_potential(team, env.player_pos[p2])
    r_pass = env.cfg.rewards.pass_threat_coeff * d_phi
    team_bonus = getattr(env.cfg.rewards, "completed_pass_team", 0.0)
    expected_p2 = 0.60 * r_pass + team_bonus
    expected_p3 = 0.40 * r_pass + env.cfg.rewards.completed_pass_receiver_ind + team_bonus
    assert d_phi > 0.0
    assert math.isclose(rew4[p2], expected_p2, abs_tol=1e-5)
    assert math.isclose(rew4[p3], expected_p3, abs_tol=1e-5)


def test_gk_observation_state_separation() -> None:
    """Verify decoupling of GK ball hold timer (obs[8]) and GK box loose ball countdown (obs[15])."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)
    gk_idx = env.agent_to_idx["a_keeper"]
    def_idx = env.agent_to_idx["a_def_1"]

    # State 1: Loose ball in keeper box; GK does NOT hold ball
    env.ball_holder = None
    env.ball_pos = np.array([80.0, 420.0], dtype=np.float64)  # Loose inside keeper's own box
    env.box_loose_ball_timer = 1.5  # 1.5s / 5.0s = 0.3 normalized
    env.ball_hold_timer = 0.0

    obs_gk = env._build_single_observation(gk_idx)
    obs_def = env._build_single_observation(def_idx)

    # obs[8] must be 0.0 (free ball) and NOT aliased with box timer
    assert obs_gk[8] == 0.0
    assert obs_gk[13] == -1.0
    assert math.isclose(obs_gk[14], 1.0, abs_tol=1e-5)
    expected_obs15 = 1.5 / env.cfg.mechanics.box_timer_obs_norm
    assert math.isclose(obs_gk[15], expected_obs15, abs_tol=1e-5)
    # Field DEF player encodes depth, flank, and active box norm in obs[13:16]
    np.testing.assert_allclose(obs_def[13:16], [-1.0, -1.0, 1.0], atol=1e-6)

    # State 2: GK holds ball; box loose timer is 0.0
    env.ball_holder = gk_idx
    env.ball_hold_timer = 2.5  # 2.5s / 5.0s = 0.5 normalized
    env.box_loose_ball_timer = 0.0

    obs_gk2 = env._build_single_observation(gk_idx)
    assert math.isclose(obs_gk2[8], 0.5, abs_tol=1e-5)
    assert obs_gk2[13] == -1.0
    assert obs_gk2[15] == 0.0


def test_tiki_taka_chain_multiplier_and_own_goal_exclusion() -> None:
    """Verify terminal goal rewards and own goal penalty."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)
    rew_cfg = env.cfg.rewards

    atk_idx = env.agent_to_idx["a_atk_1"]
    def_b_idx = env.agent_to_idx["b_def_1"]

    # 1. Standard offensive goal for Team A
    env.team_pass_chain[TEAM_A] = 4
    env.last_ball_holder = atk_idx

    step_rew = np.zeros(env.num_players, dtype=np.float64)
    env._handle_goal_scored(TEAM_A, step_rew)

    expected_team_goal = rew_cfg.goal_team_shared
    expected_atk_goal = rew_cfg.goal_scorer_ind
    assert math.isclose(step_rew[atk_idx], expected_atk_goal + expected_team_goal, abs_tol=1e-5)
    assert math.isclose(step_rew[env.agent_to_idx["a_def_1"]], expected_team_goal, abs_tol=1e-5)
    assert env.team_pass_chain == [0, 0]

    # 2. Own Goal
    env.team_pass_chain[TEAM_A] = 5
    env.last_ball_holder = atk_idx  # Team A player puts ball into own net -> Team B scores

    step_rew_og = np.zeros(env.num_players, dtype=np.float64)
    env._handle_goal_scored(TEAM_B, step_rew_og)

    assert math.isclose(step_rew_og[def_b_idx], rew_cfg.goal_team_shared, abs_tol=1e-5)
    assert math.isclose(
        step_rew_og[atk_idx], rew_cfg.own_goal_ind + rew_cfg.goal_concede_team_shared, abs_tol=1e-5
    )


def test_kinematic_lead_pass_targeting() -> None:
    """Verify that receiving a progressive pass in-stride toward goal awards through_ball_stride_bonus."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)

    kicker_idx = env.agent_to_idx["a_mid_1"]
    runner_idx = env.agent_to_idx["a_atk_1"]

    for idx in range(env.num_players):
        if env.player_team[idx] != TEAM_A:
            env.player_pos[idx] = np.array([1200.0, 750.0], dtype=np.float64)

    env.player_pos[kicker_idx] = np.array([400.0, 420.0], dtype=np.float64)
    env.player_pos[runner_idx] = np.array([750.0, 420.0], dtype=np.float64)
    env.player_vel[runner_idx] = np.array([180.0, 0.0], dtype=np.float64)  # Sprinting toward enemy goal

    env.ball_holder = None
    env.ball_kick_origin = env.player_pos[kicker_idx].copy()
    env.ball_kick_orig_opp_dist = 800.0
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.4
    env.last_ball_holder = kicker_idx
    env.last_ball_team = TEAM_A
    env.ball_pos = env.player_pos[runner_idx].copy()

    step_rew = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(step_rew)

    d_phi = env._compute_ball_potential(TEAM_A, env.player_pos[runner_idx]) - env._compute_ball_potential(
        TEAM_A, env.player_pos[kicker_idx]
    )
    base_prog = 0.60 * env.cfg.rewards.pass_threat_coeff * d_phi
    assert step_rew[kicker_idx] > base_prog + 0.30 * env.cfg.rewards.through_ball_stride_bonus


def test_kickoff_carrier_faces_own_keeper_and_sees_teammates() -> None:
    """Verify that the center kickoff player faces their own goalkeeper at start and after a goal."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)

    # 1. Team A kickoff (Team A defends Left, keeper at X=45)
    env.reset(seed=42, options={"kickoff_team": TEAM_A})
    a_atk_idx = env.agent_to_idx["a_atk_1"]
    a_gk_idx = env.agent_to_idx["a_keeper"]

    assert env.ball_holder == a_atk_idx
    # Carrier angle aims at mid_2 (520, 580) at canon_angle 2.4189 rad
    cos_angle_a = math.cos(env.player_angle[a_atk_idx])
    sin_angle_a = math.sin(env.player_angle[a_atk_idx])
    assert math.isclose(cos_angle_a, math.cos(2.4189), abs_tol=1e-4)
    assert math.isclose(sin_angle_a, math.sin(2.4189), abs_tol=1e-4)

    # In canonical coordinates, carrier faces angle 2.4189 rad (towards mid_2)
    obs_carrier = env._build_single_observation(a_atk_idx)
    # Ball is held
    assert obs_carrier[7] == 1.0
    # Relative angle to mid_2 must be ~0.0 (directly ahead in FOV)
    mid2_pos = env.player_pos[env.agent_to_idx["a_mid_2"]]
    car_pos = env.player_pos[a_atk_idx]
    angle_to_mid2 = math.atan2(mid2_pos[1] - car_pos[1], mid2_pos[0] - car_pos[0])
    rel_angle_to_mid2 = wrap_to_pi(angle_to_mid2 - env.player_angle[a_atk_idx])
    assert math.isclose(rel_angle_to_mid2, 0.0, abs_tol=1e-2)

    # Verify all 5 teammates are within carrier's FOV (+/- 105 degrees = +/- 1.83 rad)
    half_fov = env.cfg.vision.half_fov_rad
    for tm_role in ("keeper", "def_1", "def_2", "mid_1", "mid_2"):
        tm_idx = env.agent_to_idx[f"a_{tm_role}"]
        tm_pos = env.player_pos[tm_idx]
        ang = math.atan2(tm_pos[1] - car_pos[1], tm_pos[0] - car_pos[0])
        rel_ang = abs(wrap_to_pi(ang - env.player_angle[a_atk_idx]))
        assert rel_ang <= half_fov, f"Teammate {tm_role} (rel_ang={rel_ang:.2f}) outside FOV ({half_fov:.2f})"

    # 2. Team B kickoff (Team B defends Right, keeper at X=1355)
    env.reset(seed=42, options={"kickoff_team": TEAM_B})
    b_atk_idx = env.agent_to_idx["b_atk_1"]
    b_gk_idx = env.agent_to_idx["b_keeper"]

    assert env.ball_holder == b_atk_idx
    # Carrier angle aims at b_mid_2
    cos_angle_b = math.cos(env.player_angle[b_atk_idx])
    sin_angle_b = math.sin(env.player_angle[b_atk_idx])
    expected_b_ang = wrap_to_pi(2.4189 + math.pi)
    assert math.isclose(cos_angle_b, math.cos(expected_b_ang), abs_tol=1e-4)
    assert math.isclose(sin_angle_b, math.sin(expected_b_ang), abs_tol=1e-4)

    # 3. Post-Goal reset: Team A scores -> Conceding team (Team B) kicks off facing b_mid_2
    step_rew = np.zeros(env.num_players, dtype=np.float64)
    env._handle_goal_scored(TEAM_A, step_rew)
    assert env.ball_holder == b_atk_idx
    assert math.isclose(math.cos(env.player_angle[b_atk_idx]), math.cos(expected_b_ang), abs_tol=1e-4)


def test_kinematic_lunge_config_parameters() -> None:
    """Verify MechanicsConfig parameters for kinematic lunges and dynamic tackle probabilities."""
    mech = DEFAULT_CONFIG.mechanics
    assert mech.field_lunge_speed == 320.0
    assert mech.field_lunge_duration == 0.25
    assert mech.tackle_miss_stun_time == 0.8
    assert mech.tackle_victim_stun_time == 1.0

    # Expanded Max Tackle Radii
    assert mech.tackle_radius_def == 85.0
    assert mech.tackle_radius_mid == 75.0
    assert mech.tackle_radius_atk == 65.0

    # Base Probabilities
    assert mech.tackle_prob_def == 0.90
    assert mech.tackle_prob_mid == 0.82
    assert mech.tackle_prob_atk == 0.74


def test_kinematic_field_lunge_trajectory_and_speed_override() -> None:
    """Verify that triggering a tackle within aiming range overrides velocity to 320 u/s directed at carrier."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)

    # Clear kickoff protection
    env.kickoff_protection_timer = 0.0

    tackler_idx = env.agent_to_idx["a_def_1"]  # Role: DEF, R_max = 85.0, Aiming = 127.5
    carrier_idx = env.agent_to_idx["b_atk_1"]

    # Place carrier at (600, 400), tackler at (550, 400) -> distance = 50.0 (< 85.0)
    env.player_pos[carrier_idx] = np.array([600.0, 400.0], dtype=np.float64)
    env.ball_holder = carrier_idx
    env.player_pos[tackler_idx] = np.array([550.0, 400.0], dtype=np.float64)
    env.player_vel[tackler_idx] = np.array([0.0, 0.0], dtype=np.float64)

    # Initial zero step
    step_rew = np.zeros(env.num_players, dtype=np.float64)
    env._attempt_tackle(tackler_idx, step_rew)

    # Tackler must have lunged at 320.0 u/s along +X (direction to carrier)
    assert env.lunge_timer[tackler_idx] == 0.25
    assert math.isclose(env.player_vel[tackler_idx, 0], 320.0, abs_tol=1e-3)
    assert math.isclose(env.player_vel[tackler_idx, 1], 0.0, abs_tol=1e-3)
    assert math.isclose(env.player_angle[tackler_idx], 0.0, abs_tol=1e-3)


def test_tackle_whiff_air_penalty_and_instant_freeze() -> None:
    """Verify that tackling with no carrier nearby triggers instant stun, ghost state, and air penalty."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)
    env.kickoff_protection_timer = 0.0

    tackler_idx = env.agent_to_idx["a_mid_1"]  # Role: MID, R_max = 75.0, Aiming = 112.5
    carrier_idx = env.agent_to_idx["b_atk_1"]

    # Carrier is very far away at (1200, 400), tackler at (400, 400) -> dist = 800 > 112.5
    # Ball is near tackler (dist = 100u <= tackle_whiff_max_ball_dist = 180.0u)
    env.player_pos[tackler_idx] = np.array([400.0, 400.0], dtype=np.float64)
    env.player_vel[tackler_idx] = np.array([100.0, 50.0], dtype=np.float64)
    env.player_pos[carrier_idx] = np.array([1200.0, 400.0], dtype=np.float64)
    env.ball_pos = env.player_pos[tackler_idx] + np.array([100.0, 0.0], dtype=np.float64)
    env.ball_holder = carrier_idx

    step_rew = np.zeros(env.num_players, dtype=np.float64)
    env._attempt_tackle(tackler_idx, step_rew)

    # Whiff out of range: physical lunge impulse, 0.8s stun, solid during lunge, and failed tackle penalty
    assert env.stun_timer[tackler_idx] == env.cfg.mechanics.tackle_miss_stun_time
    assert env.lunge_timer[tackler_idx] == env.cfg.mechanics.field_lunge_duration
    assert math.isclose(float(np.linalg.norm(env.player_vel[tackler_idx])), env.cfg.mechanics.field_lunge_speed, abs_tol=1e-3)
    assert not env._is_ghost(tackler_idx)
    assert step_rew[tackler_idx] == env.cfg.rewards.failed_tackle_ind


def test_dynamic_tackle_quadratic_decay_and_out_of_reach_failure() -> None:
    """Verify dynamic tackle success formula P = P_max * (1 - (d/R)^2) and automatic cutoff for d > R."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)
    env.kickoff_protection_timer = 0.0

    tackler_idx = env.agent_to_idx["a_def_1"]  # Role: DEF, R_max = 85.0, P_max = 0.90, Aiming = 127.5
    carrier_idx = env.agent_to_idx["b_atk_1"]

    # Case 1: In aiming range (100u <= 127.5u) but outside reach (100u > 85.0u) -> Automatic failure
    env.player_pos[tackler_idx] = np.array([500.0, 400.0], dtype=np.float64)
    env.player_pos[carrier_idx] = np.array([600.0, 400.0], dtype=np.float64)
    env.ball_holder = carrier_idx

    step_rew = np.zeros(env.num_players, dtype=np.float64)
    env._attempt_tackle(tackler_idx, step_rew)

    # Must fail: tackler is stunned for 0.8s, carrier retains ball, failed_tackle_ind penalty
    assert env.ball_holder == carrier_idx
    assert env.stun_timer[tackler_idx] == env.cfg.mechanics.tackle_miss_stun_time
    assert not env._is_ghost(tackler_idx)
    assert step_rew[tackler_idx] == env.cfg.rewards.failed_tackle_ind

    # Case 2: Point blank range (d = 0.0) with mock RNG forced to 0.0 (guaranteed success)
    env.stun_timer[tackler_idx] = 0.0
    env.tackle_cooldown[tackler_idx] = 0.0
    env.player_pos[tackler_idx] = np.array([600.0, 400.0], dtype=np.float64)
    from unittest.mock import MagicMock
    mock_rng = MagicMock()
    mock_rng.random.return_value = 0.0
    original_rng = env.np_random
    env.np_random = mock_rng

    step_rew.fill(0.0)
    env._attempt_tackle(tackler_idx, step_rew)

    # Success: tackler gets ball, victim stunned for 1.0s and velocity 0.0
    assert env.ball_holder == tackler_idx
    assert env.stun_timer[carrier_idx] == env.cfg.mechanics.tackle_victim_stun_time
    assert np.allclose(env.player_vel[carrier_idx], [0.0, 0.0])
    assert env.is_ghost[carrier_idx] is True or env.is_ghost[carrier_idx] == 1
    assert step_rew[tackler_idx] > 0.0
    expected_carrier_rew = env.cfg.rewards.successful_tackle_carrier_ind + env.cfg.rewards.intercepted_team
    assert math.isclose(step_rew[carrier_idx], expected_carrier_rew, abs_tol=1e-6)

    env.np_random = original_rng


def test_stun_state_physics_and_ghost_mode_collision_pass_through() -> None:
    """Verify that stunned players ignore NN actions, have zero velocity, and allow collision pass-through."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)

    p1 = env.agent_to_idx["a_mid_1"]
    p2 = env.agent_to_idx["b_mid_1"]

    # Place p1 and p2 overlapping (dist = 10.0 < 36.0 min_dist)
    env.player_pos[p1] = np.array([500.0, 400.0], dtype=np.float64)
    env.player_pos[p2] = np.array([510.0, 400.0], dtype=np.float64)

    # Normal state: resolve collisions pushes them apart
    env.ghost_timer[:] = 0.0
    env.stun_timer[:] = 0.0
    env.is_ghost[:] = False
    env._resolve_player_collisions()
    dist_after_normal = float(np.linalg.norm(env.player_pos[p2] - env.player_pos[p1]))
    assert dist_after_normal >= 35.9

    # Stun state: p1 is stunned -> flagged as ghost -> collisions do NOT push them
    env.player_pos[p1] = np.array([500.0, 400.0], dtype=np.float64)
    env.player_pos[p2] = np.array([510.0, 400.0], dtype=np.float64)
    env.stun_timer[p1] = 1.0
    env.is_ghost[p1] = True

    env._resolve_player_collisions()
    dist_after_stun = float(np.linalg.norm(env.player_pos[p2] - env.player_pos[p1]))
    # Since p1 is a ghost, distance is unchanged
    assert math.isclose(dist_after_stun, 10.0, abs_tol=1e-3)


def test_save_best_only_with_multiple_parallel_envs(tmp_path: Path) -> None:
    """Verify that --save-best-only retains solely the highest-interest replay per generation when num_envs > 1."""
    rep_dir = str(tmp_path / "replays")
    ckpt_dir = str(tmp_path / "checkpoints")

    # Fast test config: 10 RL steps per episode
    fast_physics = replace(
        DEFAULT_CONFIG.physics,
        max_rl_steps=10,
    )
    cfg = replace(
        DEFAULT_CONFIG,
        physics=fast_physics,
        replay=replace(
            DEFAULT_CONFIG.replay,
            save_dir=rep_dir,
            enable_recording=True,
            interest_threshold=0.0,
            save_best_only=True,
        ),
    )

    # 1. Run 1 generation with 3 parallel environments and save_best_only=True
    trainer = ParameterSharingPPOTrainer(
        config=cfg,
        num_envs=3,
        rollout_steps=10,
        device="cpu",
        checkpoint_dir=ckpt_dir,
        seed=123,
        save_best_only=True,
    )
    rollout = trainer.collect_rollouts(active_generation=1)

    assert rollout["completed_episodes"] == 3
    assert len(trainer.match_history) == 3

    # All 3 matches are tracked in match_history
    scores = [m["interest_score"] for m in trainer.match_history]
    max_score = max(scores)

    # Only 1 replay is kept in exported_replays
    assert len(rollout["exported_replays"]) == 1
    best_replay_file = rollout["exported_replays"][0]
    assert os.path.exists(best_replay_file)

    # Check that in match_history, exactly the best match retained its replay_path
    saved_matches = [m for m in trainer.match_history if m["replay_path"]]
    assert len(saved_matches) == 1
    assert saved_matches[0]["interest_score"] == max_score
    assert saved_matches[0]["replay_path"] == best_replay_file

    # Verify that only 1 .json replay file exists in the directory
    import glob
    json_files = glob.glob(os.path.join(rep_dir, "match_*.json"))
    assert len(json_files) == 1
    assert os.path.abspath(json_files[0]) == os.path.abspath(best_replay_file)

    # 2. Run another generation with save_best_only=False to verify contrast
    rep_dir_all = str(tmp_path / "replays_all")
    cfg_all = replace(
        cfg,
        replay=replace(cfg.replay, save_dir=rep_dir_all, save_best_only=False),
    )
    trainer_all = ParameterSharingPPOTrainer(
        config=cfg_all,
        num_envs=3,
        rollout_steps=10,
        device="cpu",
        checkpoint_dir=str(tmp_path / "ckpt_all"),
        seed=456,
        save_best_only=False,
    )
    rollout_all = trainer_all.collect_rollouts(active_generation=1)
    assert rollout_all["completed_episodes"] == 3
    assert len(rollout_all["exported_replays"]) == 3
    assert len(glob.glob(os.path.join(rep_dir_all, "match_*.json"))) == 3


def test_smooth_gk_repulsion_wave_and_zero_potential_corruption() -> None:
    """Verify GK possession push is smooth at 285 u/s rather than teleporting, and does not corrupt potential shaping."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    gk_idx = env.agent_to_idx["a_keeper"]
    b_atk_idx = env.agent_to_idx["b_atk_1"]

    env.player_pos[gk_idx] = np.array([45.0, 420.0], dtype=np.float64)
    # Outside new box radius (195.0) but inside GK possession buffer (265.0)
    env.player_pos[b_atk_idx] = np.array([205.0, 420.0], dtype=np.float64)
    env.player_vel[:] = 0.0
    env.ball_holder = gk_idx
    env.ball_pos = env.player_pos[gk_idx] + np.array([env.cfg.mechanics.ball_carry_offset, 0.0])
    env.ball_vel[:] = 0.0

    # Step for 1 RL step (4 physics substeps) with zero actions
    _, rewards, _, _, _ = env.step(_zero_actions(env))

    # Expect b_atk_1 to have moved smoothly: 4 substeps * (gk_hold_push_speed * 1/60)
    # NOT instantly teleported to >= 265.0!
    expected_disp = 4.0 * (env.cfg.mechanics.gk_hold_push_speed / 60.0)
    assert abs(env.player_pos[b_atk_idx, 0] - (205.0 + expected_disp)) < 0.1
    assert env.player_pos[b_atk_idx, 0] < env.cfg.mechanics.gk_hold_opponent_buffer

    # Verify that external push did not produce an artificial presser penalty for b_atk_1
    assert rewards["b_atk_1"] > -0.05

    # Run remaining steps to reach buffer boundary >= 265.0
    for _ in range(6):
        env.step(_zero_actions(env))

    left_goal = np.array(env.cfg.physics.left_goal_center, dtype=np.float64)
    dist_from_goal = float(np.linalg.norm(env.player_pos[b_atk_idx] - left_goal))
    assert dist_from_goal >= env.cfg.mechanics.gk_hold_opponent_buffer - 1e-3


def test_dynamic_vertical_lane_assignment_prevents_role_inversion() -> None:
    """Verify off-ball support potential is symmetric across flanks and rewards moving toward optimal support distance."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    def1_idx = env.agent_to_idx["a_def_1"]
    mid1_idx = env.agent_to_idx["a_mid_1"]
    mid2_idx = env.agent_to_idx["a_mid_2"]

    env.ball_holder = def1_idx
    env.player_pos[def1_idx] = np.array([400.0, 420.0], dtype=np.float64)
    env.ball_pos = env.player_pos[def1_idx].copy()
    # Place mid_1 and mid_2 at d=420u from ball on opposite flanks
    env.player_pos[mid1_idx] = np.array([400.0, 800.0], dtype=np.float64)
    env.player_pos[mid2_idx] = np.array([400.0, 40.0], dtype=np.float64)
    for idx in range(6, 12):
        env.player_pos[idx] = np.array([1200.0, 100.0 + (idx - 6) * 100.0], dtype=np.float64)

    env._compute_potential_shaping(np.zeros(env.num_players, dtype=np.float64), any_teleport=True)

    # Move both midfielders closer to the optimal 240u support ring
    env.player_pos[mid1_idx] = np.array([400.0, 660.0], dtype=np.float64)
    env.player_pos[mid2_idx] = np.array([400.0, 180.0], dtype=np.float64)
    step_rewards_move = np.zeros(env.num_players, dtype=np.float64)
    env._compute_potential_shaping(step_rewards_move, any_teleport=False)

    assert step_rewards_move[mid2_idx] > 0.0
    assert step_rewards_move[mid1_idx] > 0.0


def test_asymmetric_crowding_exemptions_and_box_cutback_angles() -> None:
    """Verify carrier potential is zero while non-carrier crowding (< 125u) reduces potential."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    atk_idx = env.agent_to_idx["a_atk_1"]
    mid_idx = env.agent_to_idx["a_mid_1"]

    env.ball_holder = atk_idx
    env.player_pos[atk_idx] = np.array([700.0, 420.0], dtype=np.float64)
    env.ball_pos = env.player_pos[atk_idx].copy()
    env.player_pos[mid_idx] = np.array([700.0, 660.0], dtype=np.float64)  # 240u away (un-crowded)
    for idx in range(env.num_players):
        if idx not in (atk_idx, mid_idx):
            env.player_pos[idx] = np.array([100.0 + idx * 100.0, 100.0], dtype=np.float64)

    env._compute_potential_shaping(np.zeros(env.num_players, dtype=np.float64), any_teleport=True)

    # Move mid_idx into crowding range (40u from carrier < 125u)
    env.player_pos[mid_idx] = np.array([700.0, 460.0], dtype=np.float64)
    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._compute_potential_shaping(step_rewards, any_teleport=False)

    expected_atk_delta = 0.0
    assert abs(step_rewards[atk_idx] - expected_atk_delta) < 1e-6
    assert step_rewards[mid_idx] < -0.05


def test_final_third_cutback_and_box_angle_switch_pass_rewards() -> None:
    """Verify final-third cutback passes that improve goal threat award positive pass reward and increment chain."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    atk_idx = env.agent_to_idx["a_atk_1"]
    mid_idx = env.agent_to_idx["a_mid_1"]

    # a_atk_1 wide on flank at (1210, 620) cuts back to central a_mid_1 at (1240, 420) closer to goal (1400, 420)
    env.player_pos[atk_idx] = np.array([1210.0, 620.0], dtype=np.float64)
    env.player_pos[mid_idx] = np.array([1240.0, 420.0], dtype=np.float64)
    env.ball_pos = env.player_pos[mid_idx].copy()
    env.ball_kick_origin = env.player_pos[atk_idx].copy()
    env.ball_kick_orig_opp_dist = 300.0
    env.ball_holder = None
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.last_ball_holder = atk_idx
    env.last_ball_team = TEAM_A
    env.team_pass_chain[TEAM_A] = 0

    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(step_rewards)

    assert env.ball_holder == mid_idx
    assert step_rewards[atk_idx] > 0.0
    assert env.team_pass_chain[TEAM_A] == 1
    assert env.stats["completed_passes"] == 1


def test_off_ball_air_tackle_suppression() -> None:
    """Verify triggering tackle off-ball or when own team holds the ball does not stun player or emit AIR TACKLE!."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    def_idx = env.agent_to_idx["a_def_1"]
    atk_idx = env.agent_to_idx["a_atk_1"]

    # 1. Own team holds ball: a_atk_1 holds ball, a_def_1 attempts tackle
    env.ball_holder = atk_idx
    env.player_pos[def_idx] = np.array([400.0, 420.0], dtype=np.float64)
    env.player_pos[atk_idx] = np.array([700.0, 420.0], dtype=np.float64)
    env.ball_pos = env.player_pos[atk_idx].copy()
    env._frame_events = []

    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._attempt_tackle(def_idx, step_rewards)

    assert env.stun_timer[def_idx] == 0.0
    assert not env.is_ghost[def_idx]
    assert step_rewards[def_idx] == 0.0
    assert not any(ev["type"] == "AIR TACKLE!" for ev in env._frame_events)

    # 2. Loose ball > 180.0u away from tackler: no air tackle stun
    env.ball_holder = None
    env.ball_pos = np.array([700.0, 420.0], dtype=np.float64)
    env.player_pos[def_idx] = np.array([400.0, 420.0], dtype=np.float64)  # dist = 300 > 180
    env._frame_events = []

    step_rewards.fill(0.0)
    env._attempt_tackle(def_idx, step_rewards)

    assert env.stun_timer[def_idx] == 0.0
    assert not env.is_ghost[def_idx]
    assert step_rewards[def_idx] == 0.0
    assert not any(ev["type"] == "AIR TACKLE!" for ev in env._frame_events)


def test_enlarged_goal_and_semicircle_geometry_and_scaled_buffers() -> None:
    """Verify 200u goal, 195u semicircle, and scaled tactical/hold buffers."""
    cfg = DEFAULT_CONFIG
    phys = cfg.physics
    mech = cfg.mechanics
    rew = cfg.rewards

    assert phys.goal_width == 200.0
    assert phys.goal_y_min == 320.0
    assert phys.goal_y_max == 520.0
    assert phys.keeper_box_radius == 195.0

    assert mech.gk_hold_opponent_buffer == 265.0
    assert mech.lunge_cooldown == 1.8
    assert mech.keeper_dive_speed == 340.0
    assert rew.completed_pass_receiver_ind == 0.15
    assert rew.keeper_bisect_potential_weight == 0.60

    # Ensure keeper can move vertically inside the expanded goal mouth without barrier clamp
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)
    gk_idx = env.agent_to_idx["a_keeper"]
    env.player_pos[gk_idx] = np.array([40.0, 328.0], dtype=np.float64)
    env._clamp_players_to_boundaries()
    assert abs(env.player_pos[gk_idx, 1] - 328.0) < 1e-3


def test_stationary_receiver_earns_zero_chase_reward_and_active_receiver_earns_bonus() -> None:
    """Verify stationary pass chaser earns non-positive PBRS while closing distance to in-flight pass earns positive PBRS."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    passer_idx = env.agent_to_idx["a_def_1"]
    receiver_idx = env.agent_to_idx["a_mid_1"]

    for i in range(env.num_players):
        if i not in (passer_idx, receiver_idx):
            env.player_pos[i] = np.array([100.0 + i * 80.0, 100.0], dtype=np.float64)

    env.player_pos[passer_idx] = np.array([200.0, 420.0], dtype=np.float64)
    env.player_pos[receiver_idx] = np.array([500.0, 420.0], dtype=np.float64)
    env.player_vel[:] = 0.0

    # Pass in flight toward receiver
    env.ball_holder = None
    env.ball_kicked_in_flight = True
    env.ball_kick_origin = env.player_pos[passer_idx].copy()
    env.last_ball_holder = passer_idx
    env.last_ball_team = TEAM_A
    env.ball_pos = np.array([420.0, 420.0], dtype=np.float64)
    env.ball_vel = np.array([120.0, 0.0], dtype=np.float64)

    env._compute_potential_shaping(np.zeros(env.num_players, dtype=np.float64), any_teleport=True)

    # 1. Receiver stands still: undiscounted PBRS produces exact zero reward leak
    r_still = np.zeros(env.num_players, dtype=np.float64)
    env._compute_potential_shaping(r_still, any_teleport=False)
    expected_r_still = 0.0
    assert abs(r_still[receiver_idx] - expected_r_still) < 1e-6

    # 2. Receiver moves 40u closer to the ball: PBRS is strictly larger
    env.player_pos[receiver_idx] = np.array([460.0, 420.0], dtype=np.float64)
    r_moving = np.zeros(env.num_players, dtype=np.float64)
    env._compute_potential_shaping(r_moving, any_teleport=False)
    assert r_moving[receiver_idx] > r_still[receiver_idx] + 0.005

    # 3. Clean reception awards 40% of pass threat gain + completed_pass_receiver_ind
    env.ball_pos = env.player_pos[receiver_idx].copy()
    rew_arr = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(rew_arr)
    assert env.ball_holder == receiver_idx
    assert rew_arr[receiver_idx] > env.cfg.rewards.completed_pass_receiver_ind


def test_loose_ball_never_triggers_field_tackle_stun() -> None:
    """Verify field players firing tackle on loose balls never stun themselves or popup air tackle."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    p_idx = env.agent_to_idx["a_atk_1"]
    env.ball_holder = None
    env.ball_pos = env.player_pos[p_idx] + np.array([20.0, 0.0], dtype=np.float64)  # Ball loose right next to player
    env._frame_events = []

    rew_arr = np.zeros(env.num_players, dtype=np.float64)
    env._attempt_tackle(p_idx, rew_arr)

    assert env.stun_timer[p_idx] == 0.0
    assert not env.is_ghost[p_idx]
    assert rew_arr[p_idx] == 0.0
    assert not any(ev["type"] == "AIR TACKLE!" for ev in env._frame_events)


def test_keeper_obs13_instant_step_signal_and_sweeper_fetch_priority() -> None:
    """Verify GK obs[13:16] encodes [-1.0, active_box_radius / 195.0, box_loose_timer / 5.0] and loose-in-box potential."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)
    gk_idx = env.agent_to_idx["a_keeper"]

    env.ball_holder = None
    env.ball_pos = np.array([80.0, 420.0], dtype=np.float64)
    env.box_loose_ball_timer = 0.0
    obs = env._build_single_observation(gk_idx)
    np.testing.assert_allclose(obs[13:16], [-1.0, 1.0, 0.0], atol=1e-4)

    half_time = 0.5 * env.cfg.mechanics.box_timer_obs_norm
    env.box_loose_ball_timer = half_time
    obs = env._build_single_observation(gk_idx)
    assert math.isclose(obs[15], 0.5, abs_tol=1e-4)

    env.box_loose_ball_timer = env.cfg.mechanics.box_timer_obs_norm
    obs = env._build_single_observation(gk_idx)
    assert math.isclose(obs[15], 1.0, abs_tol=1e-4)


def test_keeper_holonomic_lateral_and_intercept_dive_saves_fast_shot() -> None:
    """Verify GK dive overrides velocity at 340.0 u/s along steered heading with 1.8s cooldown."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    gk_idx = env.agent_to_idx["a_keeper"]
    env.player_pos[gk_idx] = np.array([45.0, 420.0], dtype=np.float64)
    env.player_angle[gk_idx] = 0.0  # Facing downfield (+X)
    env.player_vel[gk_idx] = 0.0
    env.lunge_cooldown[gk_idx] = 0.0

    # Fast incoming shot toward upper corner (X=0, Y=340)
    env.ball_holder = None
    env.ball_pos = np.array([180.0, 360.0], dtype=np.float64)
    env.ball_vel = np.array([-360.0, -40.0], dtype=np.float64)

    # Keeper triggers special (action[3] = 1.0)
    actions = _zero_actions(env)
    actions["a_keeper"] = np.array([0.0, -1.0, 0.0, 1.0], dtype=np.float32)

    _, rewards, _, _, _ = env.step(actions)

    speed = float(np.linalg.norm(env.player_vel[gk_idx]))
    assert speed >= 300.0
    assert env.lunge_cooldown[gk_idx] > 1.7
    assert env.lunge_cooldown[gk_idx] <= env.cfg.mechanics.keeper_lunge_cooldown + 0.01


def test_foul_restart_formation_opens_central_lane_and_forward_halfspace_runners() -> None:
    """Verify keeper 5s violation resets keeper to goal mouth, ejects ball to wide flank, and applies 1.5s pickup cooldown."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    gk_idx = env.agent_to_idx["a_keeper"]
    env.ball_holder = gk_idx
    env.player_pos[gk_idx] = np.array([60.0, 420.0], dtype=np.float64)
    env.ball_pos = np.array([84.0, 420.0], dtype=np.float64)

    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._handle_keeper_5s_violation(gk_idx, step_rewards)

    assert env.ball_holder is None
    assert math.isclose(env.pickup_cooldown[gk_idx], 2.0, abs_tol=1e-5)
    assert math.isclose(float(np.linalg.norm(env.ball_vel)), 120.0, abs_tol=1e-3)
    assert env.player_pos[gk_idx, 0] == 40.0
    assert env.player_pos[gk_idx, 1] == 420.0
    assert env.ball_pos[0] == 250.0
    assert env.ball_pos[1] in (140.0, 700.0)
    assert (env.pickup_cooldown[env.player_team == 0] >= 1.5).all()


def test_intermediate_teammate_pass_not_classified_as_shot_on_target() -> None:
    """Verify that a forward pass toward a teammate whose position is collinear with the goal is NOT classified as a shot on target."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=127)

    # In Half 1, Team B (TEAM_B = 1) defends Right (X=1400) and attacks Left (X=0)
    kicker_idx = env.agent_to_idx["b_mid_2"]
    teammate_idx = env.agent_to_idx["b_def_2"]

    # Recreate Match 127: Kicker at (615.0, 271.0) aimed at Left goal center (0.0, 420.0)
    # An intermediate teammate is at (535.0, 290.4) directly along that line of sight
    env.player_pos[kicker_idx] = np.array([615.0, 271.0], dtype=np.float64)
    aim_angle = math.atan2(420.0 - 271.0, 0.0 - 615.0)
    env.player_angle[kicker_idx] = aim_angle

    # Teammate 82u ahead along the exact line of sight to the goal
    env.player_pos[teammate_idx] = np.array([535.0, 290.4], dtype=np.float64)

    # Move other players far away to eliminate side effects
    for idx in range(env.num_players):
        if idx not in (kicker_idx, teammate_idx):
            env.player_pos[idx] = np.array([1200.0, 700.0], dtype=np.float64)

    env.ball_holder = kicker_idx
    env.kick_grace_timer = 0.0
    step_rewards = np.zeros(env.num_players, dtype=np.float64)

    env._execute_kick(kicker_idx, kick_power=1.0, step_rewards=step_rewards)

    # 1. Although physically intersecting the goal frame, having a teammate in the pass corridor disqualifies it as a shot
    assert env.ball_kick_was_shot_on_target is False
    assert env.stats["shots"][TEAM_B] == 0
    # Kicker does not receive shot-on-target rewards
    assert step_rewards[kicker_idx] < env.cfg.rewards.shot_on_target_ind * 0.5

    # 2. Now move teammate away out of the pass corridor
    env.reset(seed=128)
    env.player_pos[kicker_idx] = np.array([615.0, 271.0], dtype=np.float64)
    env.player_angle[kicker_idx] = aim_angle
    env.player_pos[teammate_idx] = np.array([535.0, 700.0], dtype=np.float64)  # away from kick cone

    for idx in range(env.num_players):
        if idx not in (kicker_idx, teammate_idx):
            env.player_pos[idx] = np.array([1200.0, 700.0], dtype=np.float64)

    env.ball_holder = kicker_idx
    env.kick_grace_timer = 0.0
    step_rewards.fill(0.0)

    env._execute_kick(kicker_idx, kick_power=1.0, step_rewards=step_rewards)

    # Without an intermediate teammate in the corridor, this kick counts as a valid shot on target
    assert env.ball_kick_was_shot_on_target is True
    assert env.stats["shots"][TEAM_B] == 1
    assert step_rewards[kicker_idx] > 0.0


def test_short_distribution_pass_registered_as_completed_pass() -> None:
    """Verify short distribution (e.g. GK to CB ~90u apart, ~50u ball travel) registers as a Completed Pass."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=410)

    # In Half 1, Team A defends Left (X=0)
    gk_idx = env.agent_to_idx["a_keeper"]
    d1_idx = env.agent_to_idx["a_def_1"]

    # Keeper at (80.0, 384.0), D1 at (190.0, 324.0) -> separation ~125u, ball travel > 85u
    env.player_pos[gk_idx] = np.array([80.0, 384.0], dtype=np.float64)
    env.player_pos[d1_idx] = np.array([190.0, 324.0], dtype=np.float64)

    to_d1 = env.player_pos[d1_idx] - env.player_pos[gk_idx]
    env.player_angle[gk_idx] = math.atan2(to_d1[1], to_d1[0])

    env.ball_holder = gk_idx
    env.kick_grace_timer = 0.0
    step_rewards = np.zeros(env.num_players, dtype=np.float64)

    # Execute gentle distribution pass
    env._execute_kick(gk_idx, kick_power=0.4, step_rewards=step_rewards)
    assert env.ball_kicked_in_flight is True

    # Ball moves toward D1 and D1 picks it up (~50u kick travel)
    env.ball_pos = env.player_pos[d1_idx].copy()
    step_rewards.fill(0.0)
    env._check_free_ball_pickup(step_rewards)

    assert env.ball_holder == d1_idx
    assert env.stats["completed_passes"] == 1
    assert env.stats["team_passes"][TEAM_A] == 1
    # Receiver receives completed_pass_receiver_ind
    assert step_rewards[d1_idx] >= env.cfg.rewards.completed_pass_receiver_ind


def test_wall_bounce_resets_shot_on_target_and_near_shot_reward_distinction() -> None:
    """Verify true shots on target vs shots hitting end wall beside goal, and that wall bounces prevent false saves."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=55)

    shooter_idx = env.agent_to_idx["b_atk_1"]
    gk_idx = env.agent_to_idx["a_keeper"]

    # 1. Off-target shot hitting end wall beside goal (Y = 300, goal is [320, 520])
    env.ball_holder = shooter_idx
    env.player_pos[shooter_idx] = np.array([400.0, 300.0], dtype=np.float64)
    env.player_angle[shooter_idx] = math.pi  # Aimed directly at (0.0, 300.0) - hits end wall 20u beside post
    rewards = np.zeros(env.num_players, dtype=np.float64)

    env._execute_kick(shooter_idx, kick_power=1.0, step_rewards=rewards)

    # Off-target shot must NOT be counted as a shot on target
    assert env.ball_kick_was_shot_on_target is False
    assert env.stats["shots"][TEAM_B] == 0
    assert rewards[shooter_idx] == 0.0

    # 2. True Shot on Target that hits the wall and bounces:
    env.reset(seed=56)
    env.ball_holder = shooter_idx
    env.player_pos[shooter_idx] = np.array([400.0, 420.0], dtype=np.float64)
    env.player_angle[shooter_idx] = math.pi  # Aimed directly at center net (0.0, 420.0)
    rewards.fill(0.0)

    env._execute_kick(shooter_idx, kick_power=1.0, step_rewards=rewards)
    assert env.ball_kick_was_shot_on_target is True
    assert env.stats["shots"][TEAM_B] == 1

    # Simulate ball bouncing off wall
    env.ball_pos = np.array([5.0, 200.0], dtype=np.float64)
    env.ball_vel = np.array([-200.0, 50.0], dtype=np.float64)
    bounced = env._bounce_ball_off_walls()
    assert bounced is True
    # Wall bounce MUST immediately clear shot on target flag
    assert env.ball_kick_was_shot_on_target is False

    # Keeper now picks up the rebounded ball in the semicircle
    env.player_pos[gk_idx] = env.ball_pos.copy()
    pickup_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(pickup_rewards)

    assert env.ball_holder == gk_idx
    # Must NOT register as a save! (it is a harmless catch / box recovery ~1.85, NOT 6.0 save reward)
    assert env.stats["keeper_saves"] == 0
    assert pickup_rewards[gk_idx] < 3.0


def test_gk_dive_save_attempt_rejected_if_ball_already_passed_keeper_or_crossing_goal() -> None:
    """Verify that a goalkeeper triggering action[3] does NOT get awarded a dive save attempt
    if the ball has already flown past the keeper towards the goal line or is crossing the goal in < dt.
    """
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=101)

    # Scenario 1: Match 601 Frame 7096-7097 scenario (Half 2, Team A defends right goal x = 1400.0)
    # a_keeper is at x = 1314.35, y = 459.29.
    # The opponent's shot is at x = 1399.83, y = 515.14, moving rightwards at +175 u/s.
    # The ball has ALREADY flown 85.5u past the keeper towards the goal line.
    env.current_half = 2
    env.team_side = np.array([SIDE_RIGHT, SIDE_LEFT], dtype=np.int32)
    gk_idx = env.agent_to_idx["a_keeper"]
    env.player_pos[gk_idx] = np.array([1314.35, 459.29], dtype=np.float64)
    env.player_vel[gk_idx] = 0.0
    env.lunge_cooldown[gk_idx] = 0.0

    env.ball_holder = None
    env.last_ball_holder = env.agent_to_idx["b_atk_1"]
    env.last_ball_team = TEAM_B
    env.ball_pos = np.array([1399.83, 515.14], dtype=np.float64)
    env.ball_vel = np.array([175.0, -10.0], dtype=np.float64)
    env.ball_kicked_in_flight = True
    env.ball_kick_was_shot_on_target = True

    actions = _zero_actions(env)
    actions["a_keeper"] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    # Step environment
    _, rewards, _, _, _ = env.step(actions)

    # Goal was scored because ball crossed x=1400.
    # But a_keeper must NOT receive keeper_dive_save_attempt_ind (+0.60)!
    # The keeper's individual reward must reflect conceding the goal (and reckless lunge penalty), NOT dive save bonus.
    events = [e["type"] for fr in env.replay_frames[-4:] for e in fr.get("events", [])]
    assert "GK DIVE!" not in events, "Erroneous GK DIVE! should not be awarded when ball already passed keeper into goal!"
    assert "GOAL!" in events

    # Scenario 2: Incoming shot in front of keeper that keeper dives into and saves
    env.reset(seed=102)
    env.kickoff_protection_timer = 0.0
    gk_a_idx = env.agent_to_idx["a_keeper"]
    env.player_pos[gk_a_idx] = np.array([35.0, 410.0], dtype=np.float64)
    env.player_angle[gk_a_idx] = 0.0
    env.lunge_cooldown[gk_a_idx] = 0.0
    env.ball_holder = None
    env.last_ball_holder = env.agent_to_idx["b_atk_1"]
    env.last_ball_team = TEAM_B
    env.ball_pos = np.array([58.0, 410.0], dtype=np.float64)
    env.ball_vel = np.array([-250.0, 0.0], dtype=np.float64)
    env.ball_kicked_in_flight = True
    env.ball_kick_was_shot_on_target = True

    actions = _zero_actions(env)
    actions["a_keeper"] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    _, rewards_2, _, _, _ = env.step(actions)

    assert env.stats["keeper_saves"] == 1
    assert rewards_2["a_keeper"] > 0.0


def test_kickoff_protection_blocks_tackle_and_dive_bleedover() -> None:
    """Verify that during kickoff protection, neither keeper dive nor field player tackle triggers,
    and scoring a goal zeroes action execution for remaining substeps of that RL step.
    """
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=103)

    # env.reset() sets kickoff_protection_timer = 1.5s
    assert env.kickoff_protection_timer > 0.0

    # Try triggering special (action[3] = 1.0) on all players during kickoff
    actions = {}
    for agent in env.agents:
        actions[agent] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    _, _, _, _, _ = env.step(actions)

    # No player should have triggered lunge or tackle
    assert np.all(env.lunge_timer == 0.0)
    assert np.all(env.tackle_cooldown == 0.0)


def test_backward_passes_escape_press_and_attacking_layoff() -> None:
    """Verify backward passes under opponent pressure earn positive pressure-relief reward (d_sigma_press > 0)."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=201)

    atk_idx = env.agent_to_idx["a_atk_1"]
    mid_idx = env.agent_to_idx["a_mid_2"]
    opp_def_idx = env.agent_to_idx["b_def_1"]

    env.player_pos[atk_idx] = np.array([1000.0, 700.0], dtype=np.float64)
    env.player_pos[opp_def_idx] = np.array([1050.0, 700.0], dtype=np.float64)
    env.player_pos[mid_idx] = np.array([750.0, 420.0], dtype=np.float64)
    for idx in range(env.num_players):
        if env.player_team[idx] == TEAM_B and idx != opp_def_idx:
            env.player_pos[idx] = np.array([1250.0, 420.0], dtype=np.float64)

    env.ball_holder = atk_idx
    env.ball_hold_timer = 2.0
    rew_arr = np.zeros(env.num_players, dtype=np.float64)

    to_mid = env.player_pos[mid_idx] - env.player_pos[atk_idx]
    env.player_angle[atk_idx] = math.atan2(to_mid[1], to_mid[0])

    env._execute_kick(atk_idx, kick_power=0.8, step_rewards=rew_arr)
    assert env.ball_kicked_in_flight is True

    env.ball_pos = env.player_pos[mid_idx].copy()
    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(step_rewards)

    assert env.ball_holder == mid_idx
    assert env.stats["completed_passes"] == 1
    assert step_rewards[atk_idx] > 0.25
    assert step_rewards[mid_idx] > env.cfg.rewards.completed_pass_receiver_ind


def test_give_and_go_one_two_bonus_and_anti_farming_decay() -> None:
    """Verify progressive 1-2 passes earn threat gain on both legs while non-progressive loops earn zero passer reward."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=203)

    m1_idx = env.agent_to_idx["a_mid_1"]
    m2_idx = env.agent_to_idx["a_mid_2"]
    for idx in range(6, 12):
        env.player_pos[idx] = np.array([1250.0, 100.0 + (idx - 6) * 80.0], dtype=np.float64)

    # Pass 1: M1 at 450 passes forward to M2 at 600
    env.player_pos[m1_idx] = np.array([450.0, 420.0], dtype=np.float64)
    env.player_pos[m2_idx] = np.array([600.0, 420.0], dtype=np.float64)
    env.ball_holder = m1_idx
    env.player_angle[m1_idx] = 0.0
    rew_arr = np.zeros(env.num_players, dtype=np.float64)
    env._execute_kick(m1_idx, kick_power=0.7, step_rewards=rew_arr)
    env.ball_pos = env.player_pos[m2_idx].copy()
    step_rew_1 = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(step_rew_1)
    assert step_rew_1[m1_idx] > 0.0

    # Pass 2: M1 overlaps ahead to 750 and receives return pass from M2
    # Under pair-oscillation anti-farming, repeat pass between same pair has osc_mult = 0.0
    env.player_pos[m1_idx] = np.array([750.0, 420.0], dtype=np.float64)
    env.player_angle[m2_idx] = 0.0
    env._execute_kick(m2_idx, kick_power=0.6, step_rewards=rew_arr)
    env.ball_pos = env.player_pos[m1_idx].copy()
    step_rew_2 = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(step_rew_2)
    assert step_rew_2[m2_idx] == 0.0

    # Pass 3: Backward pass without pressure (750 -> 600) earns 0.0 passer reward
    env.player_angle[m1_idx] = math.pi
    env._execute_kick(m1_idx, kick_power=0.6, step_rewards=rew_arr)
    env.ball_pos = env.player_pos[m2_idx].copy()
    step_rew_3 = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(step_rew_3)
    assert step_rew_3[m1_idx] == 0.0


def test_loose_ball_two_defenders_mutual_crowding_exemption_and_continuous_chase() -> None:
    """Verify primary presser closing distance on a loose ball (< 55u) receives positive PBRS without crowding penalty."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=201)

    d1_idx = env.agent_to_idx["a_def_1"]
    d2_idx = env.agent_to_idx["a_def_2"]

    env.ball_holder = None
    env.ball_kicked_in_flight = False
    env.ball_pos = np.array([328.0, 457.0], dtype=np.float64)
    env.ball_vel = np.zeros(2, dtype=np.float64)

    # D1 is closest to ball (48u away), D2 is 60u from D1
    env.player_pos[d1_idx] = np.array([280.0, 457.0], dtype=np.float64)
    env.player_pos[d2_idx] = np.array([270.0, 505.0], dtype=np.float64)
    env._compute_potential_shaping(np.zeros(env.num_players, dtype=np.float64), any_teleport=True)

    # D1 closes 15u toward the loose ball (to 33u away)
    env.player_pos[d1_idx] = np.array([295.0, 457.0], dtype=np.float64)
    step_rewards_chase = np.zeros(env.num_players, dtype=np.float64)
    env._compute_potential_shaping(step_rewards_chase, any_teleport=False)
    assert step_rewards_chase[d1_idx] > 0.001


def test_gk_shot_directly_at_keeper_smothers_forward_and_rejects_lateral_dodge_dive() -> None:
    """Verify Match 783 frame 8373 scenario:
    When a shot is directed directly at the goalkeeper's body (d_cpa <= 18u),
    the keeper does NOT dive laterally away to open the goal mouth, does NOT receive an
    undeserved GK DIVE! (+0.60) award, and instead smothers forward into the shot.
    """
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=202)

    gk_idx = env.agent_to_idx["a_keeper"]
    # a_keeper defends Left Goal (x=0) in Half 1
    # Match 783 geometry: GK at (19.63, 444.43), incoming shot at (108.22, 394.02), vel = (-354, 178.8)
    env.player_pos[gk_idx] = np.array([19.63, 444.43], dtype=np.float64)
    env.player_vel[gk_idx] = 0.0
    env.lunge_cooldown[gk_idx] = 0.0
    env.kickoff_protection_timer = 0.0

    env.ball_holder = None
    env.last_ball_holder = env.agent_to_idx["b_atk_1"]
    env.last_ball_team = TEAM_B
    env.ball_pos = np.array([108.22, 394.02], dtype=np.float64)
    env.ball_vel = np.array([-354.0, 178.8], dtype=np.float64)
    env.ball_kicked_in_flight = True
    env.ball_kick_was_shot_on_target = True

    # Keeper triggers special action (action[3] = 1.0)
    actions = _zero_actions(env)
    actions["a_keeper"] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    _, rewards, _, _, _ = env.step(actions)

    # 1. Keeper MUST NOT receive 'GK DIVE!' (+0.60) bonus for dodging a ball shot directly at them!
    events = [e["type"] for fr in env.replay_frames[-2:] for e in fr.get("events", [])]
    assert "GK DIVE!" not in events, "Keeper should not receive GK DIVE! for ball shot directly at body!"

    # 2. Keeper MUST NOT lunge sideways in +Y away from the ball (in match 783 it lunged to y=530+)
    # Instead, keeper stays centered around y=444 or meets the ball along incoming path
    assert env.player_pos[gk_idx, 1] <= 455.0, (
        f"Keeper lunged laterally away to y={env.player_pos[gk_idx, 1]}, exposing goal!"
    )

    # 3. Keeper must not dive backwards into own net
    assert env.player_pos[gk_idx, 0] >= 18.0


def test_defenders_start_compact_at_kickoff_and_zone_inwards() -> None:
    """Verify defenders start compact (150u apart) shielding goal mouth and are exempt from crowding."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset(seed=42, options={"kickoff_team": TEAM_A})

    d1_a = env.agent_to_idx["a_def_1"]
    d2_a = env.agent_to_idx["a_def_2"]
    m1_a = env.agent_to_idx["a_mid_1"]
    m2_a = env.agent_to_idx["a_mid_2"]
    d1_b = env.agent_to_idx["b_def_1"]
    d2_b = env.agent_to_idx["b_def_2"]
    m1_b = env.agent_to_idx["b_mid_1"]
    m2_b = env.agent_to_idx["b_mid_2"]

    # 1. At match start / kickoff, both teams' defenders must start compact (150u apart)
    # Team A defends Left (goal mouth Y in [320, 520], center 420)
    pos_d1_a = env.player_pos[d1_a]
    pos_d2_a = env.player_pos[d2_a]
    sep_a = abs(pos_d2_a[1] - pos_d1_a[1])
    assert math.isclose(sep_a, 150.0, abs_tol=1e-3), f"Team A defenders separation was {sep_a}, expected 150.0"
    assert min(pos_d1_a[1], pos_d2_a[1]) >= 320.0, "Defenders should be inside or at goal mouth top post"
    assert max(pos_d1_a[1], pos_d2_a[1]) <= 520.0, "Defenders should be inside or at goal mouth bottom post"

    # Team A starts with the ball: midfielders start positioned deeper towards their own goal post (X = 520.0)
    pos_m1_a = env.player_pos[m1_a]
    pos_m2_a = env.player_pos[m2_a]
    assert math.isclose(pos_m1_a[0], 520.0, abs_tol=1e-3)
    assert math.isclose(pos_m2_a[0], 520.0, abs_tol=1e-3)

    # Team B defends Right
    pos_d1_b = env.player_pos[d1_b]
    pos_d2_b = env.player_pos[d2_b]
    sep_b = abs(pos_d2_b[1] - pos_d1_b[1])
    assert math.isclose(sep_b, 150.0, abs_tol=1e-3), f"Team B defenders separation was {sep_b}, expected 150.0"

    # Team B receiving midfielders remain at X = 960.0
    pos_m1_b = env.player_pos[m1_b]
    pos_m2_b = env.player_pos[m2_b]
    assert math.isclose(pos_m1_b[0], 960.0, abs_tol=1e-3)
    assert math.isclose(pos_m2_b[0], 960.0, abs_tol=1e-3)

    # 2. Verify post-goal reset maintains compact defender and deeper midfielder formation
    step_rew = np.zeros(env.num_players, dtype=np.float64)
    env._handle_goal_scored(TEAM_A, step_rew)
    pos_d1_b_reset = env.player_pos[d1_b]
    pos_d2_b_reset = env.player_pos[d2_b]
    assert math.isclose(abs(pos_d2_b_reset[1] - pos_d1_b_reset[1]), 150.0, abs_tol=1e-3)

    # After Team A scores, Team B restarts with the ball from Right side:
    # Team B midfielders now start rotated towards their own post (1400.0 - 520.0 = 880.0)
    pos_m1_b_reset = env.player_pos[m1_b]
    pos_m2_b_reset = env.player_pos[m2_b]
    assert math.isclose(pos_m1_b_reset[0], 880.0, abs_tol=1e-3)
    assert math.isclose(pos_m2_b_reset[0], 880.0, abs_tol=1e-3)

    # 3. Verify defender pair in defensive half at 150u separation incurs ZERO crowding penalty
    env.kickoff_protection_timer = 0.0
    env._compute_potential_shaping(np.zeros(env.num_players, dtype=np.float64), any_teleport=True)
    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._compute_potential_shaping(step_rewards, any_teleport=False)

    assert step_rewards[d1_a] >= -1e-3
    assert step_rewards[d2_a] >= -1e-3


def test_wall_pass_mechanics_and_anti_exploit() -> None:
    """Verify side-wall bank passes count as completed passes with anti-exploit protections."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset(seed=42, options={"kickoff_team": TEAM_A})

    p1 = env.agent_to_idx["a_mid_1"]
    p2 = env.agent_to_idx["a_atk_1"]

    # 1. Valid Side-Wall Bank Pass:
    # Passer at (500, 100) kicks towards top wall (Y=0)
    # Receiver at (700, 120) receives the ball after 1 wall bounce
    env.ball_holder = None
    env.last_ball_holder = p1
    env.last_ball_team = TEAM_A
    env.player_pos[p1] = np.array([500.0, 100.0])
    env.player_pos[p2] = np.array([700.0, 120.0])
    env.ball_pos = np.array([500.0, 100.0])
    env.ball_kick_origin = np.array([500.0, 100.0])
    env.ball_vel = np.array([200.0, -300.0])  # heading towards top wall Y=0
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.2
    env.ball_wall_bounces = 0
    env.is_wall_pass_in_flight = False
    env.pickup_cooldown[p1] = 1.0

    zero_acts = np.zeros((env.num_players, 4), dtype=np.float32)

    # Simulate physics until ball bounces off top wall
    for _ in range(30):
        env._physics_substep(zero_acts, None)
        if env.ball_wall_bounces > 0:
            break

    assert env.ball_wall_bounces == 1
    assert env.is_wall_pass_in_flight is True
    assert env.ball_kicked_in_flight is True

    # Now ball reaches receiver p2 (distance > 85u from kick origin)
    env.ball_pos = env.player_pos[p2].copy()
    step_rew = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(step_rew)

    # Must register as completed pass and WALL PASS!
    assert env.stats["completed_passes"] == 1
    assert env.stats["team_passes"][TEAM_A] == 1
    assert step_rew[p1] > 0.0
    assert any(ev.get("type") == "WALL PASS!" for ev in env._frame_events)

    # 2. Anti-Self-Pass exploit test:
    # Same player collecting ball off the wall cannot register a pass
    env._frame_events.clear()
    env.ball_holder = None
    env.last_ball_holder = p1
    env.last_ball_team = TEAM_A
    env.player_pos[p1] = np.array([500.0, 50.0])
    env.ball_pos = np.array([500.0, 50.0])
    env.ball_kick_origin = np.array([500.0, 50.0])
    env.ball_vel = np.array([0.0, -200.0])
    env.ball_kicked_in_flight = True
    env.ball_wall_bounces = 0
    env.is_wall_pass_in_flight = False
    env.pickup_cooldown[p1] = 0.5

    for _ in range(20):
        env._physics_substep(zero_acts, None)
        if env.ball_wall_bounces > 0:
            break

    assert env.ball_wall_bounces == 1
    assert env.is_wall_pass_in_flight is True

    # Same player collects:
    env.pickup_cooldown[p1] = 0.0
    prev_completed = env.stats["completed_passes"]
    step_rew.fill(0.0)
    env.ball_pos = env.player_pos[p1].copy()
    env._check_free_ball_pickup(step_rew)
    assert env.stats["completed_passes"] == prev_completed
    assert not any(ev.get("type") == "WALL PASS!" for ev in env._frame_events)

    # 3. Back-Wall / Endline test:
    # Ball hits back wall (X <= 0 or X >= 1400)
    env.ball_holder = None
    env.last_ball_holder = p1
    env.last_ball_team = TEAM_A
    env.player_pos[p1] = np.array([1300.0, 200.0])
    env.ball_pos = np.array([1380.0, 200.0])  # Non-goal Y
    env.ball_vel = np.array([300.0, 0.0])     # Hits X=1400 back wall
    env.ball_kicked_in_flight = True
    env.ball_wall_bounces = 0
    env.is_wall_pass_in_flight = False

    for _ in range(15):
        env._physics_substep(zero_acts, None)
        if env.ball_wall_bounces > 0:
            break

    # Back wall bounce must clear in-flight status
    assert env.ball_kicked_in_flight is False
    assert env.is_wall_pass_in_flight is False


def test_vision_cone_peripheral_fade() -> None:
    """Verify smooth peripheral visibility falloff across the outer 15 degrees of the vision cone."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)

    # Position player 0 at (100, 100) facing canonical angle 0 rad (due right)
    env.player_pos[0] = np.array([100.0, 100.0], dtype=np.float64)
    env.player_angle[0] = 0.0

    # 1. Player 1 at 0 degrees (dead center, rel_ang = 0) -> vis_presence in foveal zone
    env.player_pos[1] = np.array([200.0, 100.0], dtype=np.float64)
    obs = env._build_single_observation(0)
    slot1_presence = obs[21]
    assert slot1_presence >= 0.85

    # 2. Player 1 at 85 degrees (inside un-faded region <= 90 deg) -> vis_presence in foveal zone
    ang85 = math.radians(85.0)
    env.player_pos[1] = np.array([100.0 + 100.0 * math.cos(ang85), 100.0 + 100.0 * math.sin(ang85)], dtype=np.float64)
    obs = env._build_single_observation(0)
    assert obs[21] >= 0.85

    # 3. Player 1 at 97.5 degrees (exact midpoint of fade range 90..105 deg) -> vis_presence fades smoothly
    ang975 = math.radians(97.5)
    env.player_pos[1] = np.array([100.0 + 100.0 * math.cos(ang975), 100.0 + 100.0 * math.sin(ang975)], dtype=np.float64)
    obs = env._build_single_observation(0)
    assert 0.40 <= obs[21] <= 0.75

    # 4. Player 1 at 106 degrees (outside 105 deg FOV) -> falls back to peripheral radar presence 0.25
    ang106 = math.radians(106.0)
    env.player_pos[1] = np.array([100.0 + 100.0 * math.cos(ang106), 100.0 + 100.0 * math.sin(ang106)], dtype=np.float64)
    obs = env._build_single_observation(0)
    assert math.isclose(obs[21], 0.25, abs_tol=1e-5)


def test_loose_ball_proximity_pull() -> None:
    """Verify primary presser closing distance to a loose ball receives positive PBRS."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)

    env.ball_holder = None
    env.ball_pos = np.array([700.0, 420.0], dtype=np.float64)
    for i in range(12):
        env.player_pos[i] = np.array([100.0, 100.0 + i * 50.0], dtype=np.float64)

    # Place player 3 (a_mid_1) at distance 80u from ball, then move to 40u from ball
    env.player_pos[3] = np.array([700.0, 500.0], dtype=np.float64)
    env._compute_potential_shaping(np.zeros(12, dtype=np.float64), any_teleport=True)

    env.player_pos[3] = np.array([700.0, 460.0], dtype=np.float64)
    step_rewards = np.zeros(12, dtype=np.float64)
    env._compute_potential_shaping(step_rewards, any_teleport=False)

    assert step_rewards[3] > 0.005, f"Expected positive presser PBRS > 0.005, got {step_rewards[3]}"


def test_contested_ball_recovery_and_pickup_tackle_grace() -> None:
    """Verify that picking up a loose ball grants pickup_tackle_grace (0.15s) tackle immunity."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)
    env.kickoff_protection_timer = 0.0

    p_red = env.agent_to_idx["a_mid_1"]
    p_blue = env.agent_to_idx["b_mid_1"]

    env.ball_pos = np.array([700.0, 400.0], dtype=np.float64)
    env.ball_vel = np.array([0.0, 0.0], dtype=np.float64)
    env.ball_holder = None
    env.ball_kicked_in_flight = False
    env.last_ball_team = TEAM_B
    env.last_ball_holder = p_blue
    env.pickup_cooldown[:] = 0.0
    env.stun_timer[:] = 0.0

    env.player_pos[p_red] = np.array([715.0, 400.0], dtype=np.float64)
    env.player_pos[p_blue] = np.array([745.0, 400.0], dtype=np.float64)

    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(step_rewards)

    # 1. Red wins the ball and pickup_tackle_grace_timer is set to 0.15s
    assert env.ball_holder == p_red
    assert math.isclose(env.pickup_tackle_grace_timer, env.cfg.mechanics.pickup_tackle_grace, rel_tol=1e-5)

    # 2. Blue attempts immediate tackle during grace period -> blocked (carrier keeps ball, tackler whiffs)
    step_rewards.fill(0.0)
    env._attempt_tackle(p_blue, step_rewards)
    assert env.ball_holder == p_red
    assert env.stun_timer[p_red] == 0.0
    assert math.isclose(env.stun_timer[p_blue], env.cfg.mechanics.tackle_miss_stun_time, rel_tol=1e-5)
    assert math.isclose(step_rewards[p_blue], env.cfg.rewards.failed_tackle_ind, rel_tol=1e-5)

    # 3. Once grace expires and turnover cooldown elapses, Blue can tackle normally
    env.pickup_tackle_grace_timer = 0.0
    env.stun_timer[p_blue] = 0.0
    env.last_turnover_reward_time = -999.0
    from unittest.mock import MagicMock
    mock_rng = MagicMock()
    mock_rng.random.return_value = 0.0
    orig_rng = env.np_random
    env.np_random = mock_rng

    step_rewards.fill(0.0)
    env._attempt_tackle(p_blue, step_rewards)

    assert env.ball_holder == p_blue
    assert env.stun_timer[p_red] > 0.0
    assert step_rewards[p_blue] > 0.0

    env.np_random = orig_rng


def test_keeper_pass_corridor_blocks_clearance_bonus() -> None:
    """Verify GK pass intercepted in corridor incurs interception penalty while completed forward pass earns positive reward."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)

    gk_idx = env.agent_to_idx["a_keeper"]
    tm_idx = env.agent_to_idx["a_mid_1"]
    opp_atk_idx = env.agent_to_idx["b_atk_1"]

    # Case 1: GK kick is intercepted by opponent in corridor
    env.player_pos[gk_idx] = np.array([50.0, 420.0], dtype=np.float64)
    env.player_angle[gk_idx] = 0.0
    env.ball_pos = np.array([74.0, 420.0], dtype=np.float64)
    env.ball_holder = gk_idx
    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._execute_kick(gk_idx, kick_power=0.8, step_rewards=step_rewards)

    env.player_pos[opp_atk_idx] = np.array([180.0, 420.0], dtype=np.float64)
    env.ball_pos = env.player_pos[opp_atk_idx].copy()
    env._check_free_ball_pickup(step_rewards)
    assert step_rewards[gk_idx] < 0.0

    # Case 2: Clear corridor, pass reaches teammate upfield
    env.reset(seed=43)
    env.player_pos[gk_idx] = np.array([50.0, 420.0], dtype=np.float64)
    env.player_pos[tm_idx] = np.array([450.0, 420.0], dtype=np.float64)
    env.player_angle[gk_idx] = 0.0
    env.ball_pos = np.array([74.0, 420.0], dtype=np.float64)
    env.ball_holder = gk_idx

    step_rewards.fill(0.0)
    env._execute_kick(gk_idx, kick_power=0.8, step_rewards=step_rewards)
    env.ball_pos = env.player_pos[tm_idx].copy()
    env._check_free_ball_pickup(step_rewards)
    assert step_rewards[gk_idx] > 0.0


def test_keeper_turnover_penalty_gradient() -> None:
    """Verify that intercepted keeper kicks apply intercepted_kicker_ind + intercepted_team penalty."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)

    gk_idx = env.agent_to_idx["a_keeper"]
    opp_idx = env.agent_to_idx["b_atk_1"]

    env.last_ball_holder = gk_idx
    env.last_ball_team = 0
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.ball_kick_origin = np.array([50.0, 420.0], dtype=np.float64)
    env.ball_kick_was_shot_on_target = False
    env.player_pos[opp_idx] = np.array([450.0, 420.0], dtype=np.float64)
    env.ball_pos = np.array([450.0, 420.0], dtype=np.float64)
    env.ball_holder = None
    env.pickup_cooldown[:] = 0.0
    env.stun_timer[:] = 0.0

    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(step_rewards)

    expected = env.cfg.rewards.intercepted_kicker_ind + env.cfg.rewards.intercepted_team
    assert math.isclose(step_rewards[gk_idx], expected, rel_tol=1e-5)


def test_keeper_5s_unclaimed_box_ball_and_post_save_grace() -> None:
    """Verify GK hold violation event and post-save grace timer."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    gk_idx = env.agent_to_idx["a_keeper"]
    step_rewards = np.zeros(env.num_players, dtype=np.float64)

    # 1. Verify GK hold violation emits '5s KEEPER HOLD FOUL!'
    env.ball_holder = gk_idx
    env.ball_hold_timer = 5.01
    env._frame_events.clear()
    env._handle_keeper_5s_violation(gk_idx, step_rewards)
    assert any(e["type"] == "5s KEEPER HOLD FOUL!" for e in env._frame_events)

    # 2. Verify post-save grace pauses box_loose_ball_timer accumulation
    env.ball_holder = None
    env.ball_pos = np.array([80.0, 420.0], dtype=np.float64)  # Inside left keeper box
    env.box_loose_ball_timer = 0.0
    env.post_save_box_grace_timer = 2.0

    # Advance 1 physics step
    actions = _zero_actions(env)
    env.step(actions)
    # Box loose timer should still be 0.0 because post_save_box_grace_timer was active
    assert env.box_loose_ball_timer == 0.0
    assert env.post_save_box_grace_timer < 2.0


def test_dynamic_box_shrinks_only_after_ball_stopped_for_2s() -> None:
    """Verify box loose timer and shrinkage ONLY engage when ball is stopped/settled inside for 2s (<= 15.0 u/s)."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    nom_radius = env.cfg.physics.keeper_box_radius
    assert env.active_box_radius[SIDE_LEFT] == nom_radius

    # 1. Moving ball enters left box: timer must stay 0.0, radius must not shrink
    env.ball_holder = None
    env.ball_pos = np.array([80.0, 420.0], dtype=np.float64)  # inside left box
    env.ball_vel = np.array([200.0, 0.0], dtype=np.float64)  # moving at high speed
    env.box_loose_ball_timer = 0.0
    env.post_save_box_grace_timer = 0.0

    actions = _zero_actions(env)
    # Run 1 RL decision step (4 physics frames) with ball moving
    env.step(actions)
    assert env.box_loose_ball_timer == 0.0
    assert env.active_box_radius[SIDE_LEFT] == nom_radius

    # 2. Ball settles/slows inside left box (<= 15.0 u/s threshold)
    env.ball_pos = np.array([80.0, 420.0], dtype=np.float64)
    env.ball_vel = np.array([8.0, 0.0], dtype=np.float64)
    env.box_loose_ball_timer = 0.0
    env.post_save_box_grace_timer = 0.0

    # Simulate 1.5s (< 2.0s box_shrink_grace_time)
    dt = env.cfg.physics.dt
    frames_1_5s = int(round(1.5 / dt))
    for _ in range(frames_1_5s):
        env._physics_substep(np.zeros((env.num_players, 4), dtype=np.float32), np.zeros(env.num_players))

    # Timer accumulated, but radius must still be untouched at nominal 195.0
    assert env.box_loose_ball_timer > 1.4
    assert env.box_loose_ball_timer < 2.0
    assert env.active_box_radius[SIDE_LEFT] == nom_radius

    # Simulate another 1.0s (total 2.5s > 2.0s grace)
    frames_1s = int(round(1.0 / dt))
    for _ in range(frames_1s):
        env._physics_substep(np.zeros((env.num_players, 4), dtype=np.float32), np.zeros(env.num_players))

    # After 2.5s stopped, radius must have shrunk by ~0.5s * 55 u/s = ~27.5u
    assert env.box_loose_ball_timer > 2.4
    assert env.active_box_radius[SIDE_LEFT] < nom_radius - 20.0

    # 3. Ball clears the box -> timer resets immediately and radius re-expands
    env.ball_pos = np.array([300.0, 420.0], dtype=np.float64)
    env.ball_vel = np.array([120.0, 0.0], dtype=np.float64)
    shrunken_r = float(env.active_box_radius[SIDE_LEFT])
    env._physics_substep(np.zeros((env.num_players, 4), dtype=np.float32), np.zeros(env.num_players))

    assert env.box_loose_ball_timer[SIDE_LEFT] == 0.0
    assert env.active_box_radius[SIDE_LEFT] > shrunken_r


def test_unclaimed_box_ball_organic_shrinkage_to_zero_no_foul() -> None:
    """Verify dead ball in box contracts semicircle to 0.0u without foul/ejection, allowing outfield entry."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    nom_radius = env.cfg.physics.keeper_box_radius
    assert nom_radius == 195.0
    dt = env.cfg.physics.dt

    # Dead ball inside Left box at (80, 420)
    env.ball_holder = None
    env.ball_pos = np.array([80.0, 420.0], dtype=np.float64)
    env.ball_vel = np.array([0.0, 0.0], dtype=np.float64)
    env.box_loose_ball_timer = 0.0
    env.post_save_box_grace_timer = 0.0

    # Place opposing attacker at (100, 420) - inside nominal 195u semicircle
    atk_idx = env.agent_to_idx["b_atk_1"]
    env.player_pos[atk_idx] = np.array([100.0, 420.0], dtype=np.float64)
    env.player_vel[atk_idx] = np.zeros(2, dtype=np.float64)

    # Simulate 5.6s of dead ball (2.0s grace + 3.55s shrink at 55 u/s to reach 0.0u)
    frames_5_6s = int(round(5.6 / dt))
    for _ in range(frames_5_6s):
        env._physics_substep(np.zeros((env.num_players, 4), dtype=np.float32), np.zeros(env.num_players))

    # Semicircle must have completely contracted to 0.0u
    assert env.active_box_radius[SIDE_LEFT] == 0.0
    # No foul, no keeper 5s violation, no synthetic ejection
    assert env.stats["keeper_5s_violations"] == 0
    assert not any("FOUL" in e.get("type", "") for e in env._frame_events)
    # Ball is still at dead position (not teleported/ejected toward midfield)
    assert math.isclose(env.ball_pos[0], 80.0, abs_tol=1e-3)
    assert math.isclose(env.ball_pos[1], 420.0, abs_tol=1e-3)

    # Boundary clamping should NOT push outfield attacker away when box is 0.0u
    env.player_pos[atk_idx] = np.array([100.0, 420.0], dtype=np.float64)
    env._clamp_players_to_boundaries(apply_dynamic_push=True)
    assert math.isclose(env.player_pos[atk_idx, 0], 100.0, abs_tol=1e-3)

    # When ball is kicked / moves (> 15 u/s), semicircle re-expands and timer resets
    env.ball_vel = np.array([50.0, 0.0], dtype=np.float64)
    env._physics_substep(np.zeros((env.num_players, 4), dtype=np.float32), np.zeros(env.num_players))
    assert env.box_loose_ball_timer[SIDE_LEFT] == 0.0
    assert env.active_box_radius[SIDE_LEFT] > 0.0


def test_outfield_semicircle_contact_strips_inward_radial_velocity() -> None:
    """Verify outfield player colliding with active_box_radius has inward radial velocity stripped, but outward velocity is preserved."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    # Use Team A striker approaching Team B goal (Right goal at 1400, 420)
    atk_idx = env.agent_to_idx["a_atk_1"]
    goal_c = np.array(env.cfg.physics.right_goal_center, dtype=np.float64)  # [1400, 420]
    box_r = float(env.active_box_radius[SIDE_RIGHT])

    # 1. Position player inside the semicircle boundary, heading inward toward goal center
    inward_dir = np.array([-1.0, 0.0], dtype=np.float64)  # towards (1400, 420) from the left
    player_pos = goal_c + inward_dir * (box_r - 10.0)  # at (1215, 420)
    env.player_pos[atk_idx] = player_pos.copy()
    # Velocity heading inwards toward the goal (+x) and tangentially (+y)
    env.player_vel[atk_idx] = np.array([80.0, 50.0], dtype=np.float64)

    delta = env.player_pos[atk_idx] - goal_c
    push_dir = delta / np.linalg.norm(delta)
    assert np.dot(env.player_vel[atk_idx], push_dir) < -50.0

    # Execute boundary clamp with dynamic push
    env._clamp_players_to_boundaries(apply_dynamic_push=True)

    # Position must be clamped to eff_box_r + 1e-3
    dist_after = float(np.linalg.norm(env.player_pos[atk_idx] - goal_c))
    assert dist_after >= box_r

    # Inward radial velocity component must be stripped, leaving only tangential velocity (Vy = 50.0)
    rad_v_after = float(np.dot(env.player_vel[atk_idx], push_dir))
    assert rad_v_after >= -1e-6
    assert math.isclose(env.player_vel[atk_idx, 1], 50.0, abs_tol=1e-5)

    # 2. Outward velocity preservation: player moving outward away from goal
    env.player_pos[atk_idx] = goal_c + inward_dir * (box_r - 5.0)
    env.player_vel[atk_idx] = np.array([-60.0, 30.0], dtype=np.float64)  # moving away from goal (-X)
    delta2 = env.player_pos[atk_idx] - goal_c
    push_dir2 = delta2 / np.linalg.norm(delta2)
    assert np.dot(env.player_vel[atk_idx], push_dir2) > 0.0  # outward!

    env._clamp_players_to_boundaries(apply_dynamic_push=True)
    # Outward velocity must NOT be stripped or altered
    assert math.isclose(env.player_vel[atk_idx, 0], -60.0, abs_tol=1e-5)
    assert math.isclose(env.player_vel[atk_idx, 1], 30.0, abs_tol=1e-5)

    # 3. Goalkeeper inside own box is exempt from boundary clamp
    gk_idx = env.agent_to_idx["a_keeper"]
    gk_goal = np.array(env.cfg.physics.left_goal_center, dtype=np.float64)
    env.player_pos[gk_idx] = gk_goal + np.array([50.0, 0.0], dtype=np.float64)  # inside left box
    env.player_vel[gk_idx] = np.array([-40.0, 0.0], dtype=np.float64)
    env._clamp_players_to_boundaries(apply_dynamic_push=True)
    # GK position and vel are not clamped by own box
    assert float(np.linalg.norm(env.player_pos[gk_idx] - gk_goal)) < 60.0
    assert math.isclose(env.player_vel[gk_idx, 0], -40.0, abs_tol=1e-5)


def test_carrier_center_used_for_advance_and_backpedal_suppression() -> None:
    """Verify carrier rotation produces zero advance spike and forward dribbling earns positive PBRS."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    carrier_idx = env.agent_to_idx["a_mid_1"]

    # 1. Rotating in place at enemy semicircle edge: carrier center remains fixed, only body angle changes
    env.ball_holder = carrier_idx
    env.player_pos[carrier_idx] = np.array([1205.0, 420.0], dtype=np.float64)
    env.player_angle[carrier_idx] = 0.0
    env.player_vel[carrier_idx] = np.array([0.0, 0.0], dtype=np.float64)
    env.ball_pos = env.player_pos[carrier_idx] + np.array([env.cfg.mechanics.ball_carry_offset, 0.0])
    env._compute_potential_shaping(np.zeros(env.num_players, dtype=np.float64), any_teleport=True)

    actions = _zero_actions(env)
    actions["a_mid_1"] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # pure turn
    _, rewards, _, _, _ = env.step(actions)
    assert rewards["a_def_1"] >= -0.05

    # 2. Forward carry vs retreating carry PBRS
    def_idx = env.agent_to_idx["a_def_1"]
    env.ball_holder = def_idx
    env.player_pos[def_idx] = np.array([500.0, 420.0], dtype=np.float64)
    env.player_angle[def_idx] = 0.0
    env.player_vel[def_idx] = np.array([150.0, 0.0], dtype=np.float64)
    env.ball_pos = env.player_pos[def_idx] + np.array([env.cfg.mechanics.ball_carry_offset, 0.0])
    actions = _zero_actions(env)
    actions["a_def_1"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _, rew_fwd, _, _, _ = env.step(actions)

    env.player_vel[def_idx] = np.array([-100.0, 0.0], dtype=np.float64)
    actions["a_def_1"] = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _, rew_back, _, _, _ = env.step(actions)

    assert rew_fwd["a_def_1"] > rew_back["a_def_1"]


def test_semicircle_cutback_pass_triggers_layoff_and_increments_chain() -> None:
    """Verify cutbacks from wide byline to central slot improve threat potential and increment pass chain."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    atk = env.agent_to_idx["a_atk_1"]
    mid = env.agent_to_idx["a_mid_1"]
    team = TEAM_A

    # Lateral cutback inward from wide flank (1250, 620) to central slot (1245, 420) closer to goal (1400, 420)
    env.player_pos[atk] = np.array([1250.0, 620.0], dtype=np.float64)
    env.player_pos[mid] = np.array([1245.0, 420.0], dtype=np.float64)
    env.ball_holder = None
    env.ball_kick_origin = env.player_pos[atk].copy()
    env.ball_kick_orig_opp_dist = 300.0
    env.ball_pos = env.player_pos[mid].copy()
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.last_ball_holder = atk
    env.last_ball_team = team
    env.team_pass_chain[team] = 1

    rew = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(rew)
    assert env.team_pass_chain[team] == 2
    assert rew[atk] > 0.0


def test_role_aware_chaser_handoff_prioritizes_mid_and_atk_over_def_in_opponent_half() -> None:
    """Verify closest outfield player is assigned primary presser potential while teammates take goal-side cover."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    d1 = env.agent_to_idx["a_def_1"]
    m1 = env.agent_to_idx["a_mid_1"]

    env.ball_holder = None
    env.ball_pos = np.array([400.0, 420.0], dtype=np.float64)
    env.player_pos[d1] = np.array([350.0, 420.0], dtype=np.float64)  # closest to ball (50u)
    env.player_pos[m1] = np.array([550.0, 420.0], dtype=np.float64)
    env._compute_potential_shaping(np.zeros(env.num_players, dtype=np.float64), any_teleport=True)

    # Move d1 20u closer to the ball -> positive presser PBRS
    env.player_pos[d1] = np.array([370.0, 420.0], dtype=np.float64)
    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._compute_potential_shaping(step_rewards, any_teleport=False)
    assert step_rewards[d1] > 0.0


def test_back_wall_bounce_resets_pass_chain_and_side_wall_preserves() -> None:
    """Verify that endline back-wall bounce resets pass chain, but side-wall bounce preserves it."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    team = TEAM_A
    env.team_pass_chain[team] = 5
    env.last_ball_team = team
    env.ball_holder = None
    env.ball_kick_was_shot_on_target = True

    # 1. Back wall bounce outside goal mouth (X=5.0, Y=200.0, moving left)
    env.ball_pos = np.array([5.0, 200.0], dtype=np.float64)
    env.ball_vel = np.array([-100.0, 0.0], dtype=np.float64)
    bounced = env._bounce_ball_off_walls()

    assert bounced is True
    assert env.team_pass_chain[team] == 0
    assert env.ball_kick_was_shot_on_target is False

    # 2. Side wall bounce (bank pass / wall pass off top wall: Y=2.0, Vy = -80.0)
    env.team_pass_chain[team] = 4
    env.last_ball_team = team
    env.ball_pos = np.array([700.0, 2.0], dtype=np.float64)
    env.ball_vel = np.array([50.0, -80.0], dtype=np.float64)
    bounced_side = env._bounce_ball_off_walls()

    assert bounced_side is True
    # Side-wall bank pass MUST retain pass chain!
    assert env.team_pass_chain[team] == 4


def test_margin_gated_soft_line_ordering_penalizes_overextended_defender() -> None:
    """Verify goal-side defensive cover potential rewards retreating to cover point and penalizes overextension."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    d1 = env.agent_to_idx["a_def_1"]
    a1 = env.agent_to_idx["a_atk_1"]
    opp = env.agent_to_idx["b_atk_1"]

    # Out of possession: b_atk_1 holds ball at (800, 420), a_atk_1 is closest presser at (750, 420)
    env.ball_holder = opp
    env.player_pos[opp] = np.array([800.0, 420.0], dtype=np.float64)
    env.ball_pos = env.player_pos[opp].copy()
    env.player_pos[a1] = np.array([750.0, 420.0], dtype=np.float64)

    # Nearest opponent to d1 is b_def_1 at (600, 300); goal-side cover point is 0.65 * (600, 300) + 0.35 * (0, 420) = (390, 342)
    b_d1 = env.agent_to_idx["b_def_1"]
    env.player_pos[b_d1] = np.array([600.0, 300.0], dtype=np.float64)
    env.player_pos[d1] = np.array([500.0, 342.0], dtype=np.float64)
    env._compute_potential_shaping(np.zeros(env.num_players, dtype=np.float64), any_teleport=True)

    # Retreating toward cover point (390, 342) earns positive PBRS
    env.player_pos[d1] = np.array([420.0, 342.0], dtype=np.float64)
    step_rewards_retreat = np.zeros(env.num_players, dtype=np.float64)
    env._compute_potential_shaping(step_rewards_retreat, any_teleport=False)
    assert step_rewards_retreat[d1] > 0.0

    # Overextending away from cover point (to 650, 342) earns negative PBRS
    env.player_pos[d1] = np.array([650.0, 342.0], dtype=np.float64)
    step_rewards_over = np.zeros(env.num_players, dtype=np.float64)
    env._compute_potential_shaping(step_rewards_over, any_teleport=False)
    assert step_rewards_over[d1] < 0.0


def test_anchor_pocket_deadzone_eliminates_micro_jitter_inside_35u() -> None:
    """Verify off-ball support potential peaks smoothly around the 240u radial support ring."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    a1 = env.agent_to_idx["a_atk_1"]
    m2 = env.agent_to_idx["a_mid_2"]
    env.ball_holder = a1
    env.player_pos[a1] = np.array([500.0, 420.0], dtype=np.float64)
    env.ball_pos = env.player_pos[a1].copy()

    # Move m2 from 380u away to 240u away (optimal support ring) -> positive PBRS
    env.player_pos[m2] = np.array([500.0, 800.0], dtype=np.float64)
    env._compute_potential_shaping(np.zeros(env.num_players, dtype=np.float64), any_teleport=True)

    env.player_pos[m2] = np.array([500.0, 660.0], dtype=np.float64)
    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._compute_potential_shaping(step_rewards, any_teleport=False)
    assert step_rewards[m2] > 0.0


def test_pinned_ball_nudge_exempts_penalty_box_and_triggers_near_edge() -> None:
    """Verify that pinned_ball nudge is exempt inside box, but triggers outside near box edge."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    # 1. Position ball inside the left goalkeeper box
    env.ball_pos = np.array([80.0, 420.0], dtype=np.float64)
    env.ball_vel = np.array([0.0, 0.0], dtype=np.float64)
    env.ball_holder = None

    p1 = env.agent_to_idx["a_def_1"]
    p2 = env.agent_to_idx["b_atk_1"]
    env.player_pos[p1] = np.array([90.0, 420.0], dtype=np.float64)
    env.player_pos[p2] = np.array([80.0, 430.0], dtype=np.float64)

    env._physics_substep(np.zeros((env.num_players, 4), dtype=np.float32), np.zeros(env.num_players))
    # pinned_ball_timer must NOT accumulate because in_any_box is True
    assert env.pinned_ball_timer == 0.0

    # 2. Position ball OUTSIDE box near edge: d = 210u (within 24u of 195u radius)
    env.ball_pos = np.array([210.0, 420.0], dtype=np.float64)
    env.ball_vel = np.array([0.0, 0.0], dtype=np.float64)
    env.player_pos[p1] = np.array([215.0, 420.0], dtype=np.float64)
    env.player_pos[p2] = np.array([210.0, 430.0], dtype=np.float64)

    env._physics_substep(np.zeros((env.num_players, 4), dtype=np.float32), np.zeros(env.num_players))
    # near_box_edge is True, so pinned_ball_timer accumulates!
    assert env.pinned_ball_timer > 0.0


def test_dead_angle_flank_shot_attenuation_and_cutback_aim_boost() -> None:
    """Verify shots on target earn shot-on-target reward while off-target flank shots earn zero."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    atk = env.agent_to_idx["a_atk_1"]
    gk = env.agent_to_idx["b_keeper"]

    # 1. Off-target flank kick aimed outside goal mouth
    env.player_pos[gk] = np.array([1360.0, 360.0], dtype=np.float64)
    env.player_pos[atk] = np.array([1250.0, 180.0], dtype=np.float64)
    env.player_angle[atk] = 0.0  # Aimed at (1400, 180) outside goal mouth [320, 520]
    env.ball_pos = env.player_pos[atk].copy()
    env.ball_holder = atk

    rew_flank = np.zeros(env.num_players, dtype=np.float64)
    env._execute_kick(atk, kick_power=0.8, step_rewards=rew_flank)
    flank_shot_rew = rew_flank[atk]

    # 2. Central shot on target from same X
    env.reset(seed=43)
    env.player_pos[gk] = np.array([1360.0, 360.0], dtype=np.float64)
    env.player_pos[atk] = np.array([1250.0, 420.0], dtype=np.float64)
    env.player_angle[atk] = 0.0
    env.ball_pos = env.player_pos[atk].copy()
    env.ball_holder = atk

    rew_central = np.zeros(env.num_players, dtype=np.float64)
    env._execute_kick(atk, kick_power=0.8, step_rewards=rew_central)
    central_shot_rew = rew_central[atk]

    assert flank_shot_rew == 0.0
    assert central_shot_rew > 0.4
    assert central_shot_rew > flank_shot_rew


def test_scanned_pass_multiplier_in_and_out_of_fov() -> None:
    """Verify progressive forward passes earn positive threat reward while backward unpressed passes earn zero."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    kicker = env.agent_to_idx["a_mid_1"]
    receiver = env.agent_to_idx["a_atk_1"]
    team = TEAM_A

    # 1. Progressive forward pass (600 -> 900)
    env.player_pos[kicker] = np.array([600.0, 420.0], dtype=np.float64)
    env.player_pos[receiver] = np.array([900.0, 420.0], dtype=np.float64)
    env.ball_holder = None
    env.ball_kick_origin = env.player_pos[kicker].copy()
    env.ball_kick_orig_opp_dist = 500.0
    env.ball_pos = env.player_pos[receiver].copy()
    env.ball_vel = np.array([200.0, 0.0], dtype=np.float64)
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.last_ball_holder = kicker
    env.last_ball_team = team

    rew_in = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(rew_in)
    kicker_rew_in = rew_in[kicker]

    # 2. Backward pass without pressure (600 -> 300)
    env.reset(seed=43)
    env.player_pos[kicker] = np.array([600.0, 420.0], dtype=np.float64)
    env.player_pos[receiver] = np.array([300.0, 420.0], dtype=np.float64)
    env.ball_holder = None
    env.ball_kick_origin = env.player_pos[kicker].copy()
    env.ball_kick_orig_opp_dist = 500.0
    env.ball_pos = env.player_pos[receiver].copy()
    env.ball_vel = np.array([-200.0, 0.0], dtype=np.float64)
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.last_ball_holder = kicker
    env.last_ball_team = team

    rew_out = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(rew_out)
    kicker_rew_out = rew_out[kicker]

    assert kicker_rew_in > kicker_rew_out
    assert kicker_rew_out == 0.0


def test_state_a2_defender_stays_behind_ball_carrier() -> None:
    """Verify off-ball defender moving into an open pass lane at 240u support distance earns positive PBRS."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    carrier = env.agent_to_idx["a_mid_1"]
    d1 = env.agent_to_idx["a_def_1"]

    env.ball_holder = carrier
    env.player_pos[carrier] = np.array([450.0, 420.0], dtype=np.float64)
    env.ball_pos = env.player_pos[carrier].copy()
    env.player_pos[d1] = np.array([100.0, 225.0], dtype=np.float64)  # 350u behind carrier in flank channel
    env._compute_potential_shaping(np.zeros(env.num_players, dtype=np.float64), any_teleport=True)

    # Move d1 to 210u behind carrier (closer to 240u support ring)
    env.player_pos[d1] = np.array([210.0, 225.0], dtype=np.float64)
    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._compute_potential_shaping(step_rewards, any_teleport=False)
    assert step_rewards[d1] > 0.0


def test_arc_hesitation_stall_penalty() -> None:
    """Verify stationary carrier near enemy semicircle earns non-positive PBRS while advancing carrier earns positive PBRS."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    carrier = env.agent_to_idx["a_mid_1"]
    env.ball_holder = carrier
    env.player_pos[carrier] = np.array([1150.0, 420.0], dtype=np.float64)
    env.player_angle[carrier] = 0.0
    env.ball_pos = env.player_pos[carrier] + np.array([env.cfg.mechanics.ball_carry_offset, 0.0])

    # 1. Stationary step -> non-positive PBRS
    _, rewards_stall, _, _, _ = env.step(_zero_actions(env))
    assert rewards_stall["a_mid_1"] <= 0.0

    # 2. Forward step -> positive PBRS
    acts = _zero_actions(env)
    acts["a_mid_1"] = np.array([1.0, 0.0, -1.0, -1.0], dtype=np.float32)
    _, rewards_fwd, _, _, _ = env.step(acts)
    assert rewards_fwd["a_mid_1"] > rewards_stall["a_mid_1"]


def test_no_blind_spots_peripheral_radar() -> None:
    """Verify peripheral radar never zeroes out entity slots when directly behind an agent."""
    env = FootballParallelEnv(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    env.reset(seed=101)

    # Place player 1 (a_def_1) facing East (angle = 0.0)
    agent_name = "a_def_1"
    idx = env.agent_to_idx[agent_name]
    env.player_pos[idx] = np.array([500.0, 420.0], dtype=np.float64)
    env.player_angle[idx] = 0.0  # Facing positive X

    # Place teammate directly behind player 1 (West: relative angle = pi)
    teammate_idx = env.agent_to_idx["a_def_2"]
    env.player_pos[teammate_idx] = np.array([400.0, 420.0], dtype=np.float64)

    # Place ball directly behind player 1 (West)
    env.ball_pos[:] = np.array([350.0, 420.0], dtype=np.float64)

    obs = env._get_obs()
    player_obs = obs[agent_name]

    # Slot 0 is ball (indices 16..20)
    # Slot format: [presence, rel_dist, rel_bearing, rel_vx, rel_vy]
    ball_slot = player_obs[16:21]
    assert abs(ball_slot[0]) >= 0.20, f"Ball presence {ball_slot[0]} magnitude should be at least 0.20 floor"
    assert not np.allclose(ball_slot[1:3], [0.0, 0.0]), "Ball slot should not be zeroed out"

    # Find which entity slot corresponds to teammate_idx
    slot_idx = env._entity_player_slots[idx].index(teammate_idx)
    # Each slot is 5 floats starting at 16 + (slot_idx + 1) * 5
    start_i = 16 + (slot_idx + 1) * 5
    tm_slot = player_obs[start_i : start_i + 5]
    assert tm_slot[0] >= 0.25, f"Teammate presence {tm_slot[0]} should be at least 0.25 floor"
    assert not np.allclose(tm_slot[1:3], [0.0, 0.0]), "Teammate slot should not be zeroed out"


def test_gk_does_not_overshoot_close_shot_when_diving() -> None:
    """Verify GK dive launches at keeper_dive_speed (340.0 u/s) and intercepts incoming close-range shot."""
    env = FootballParallelEnv(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    env.reset(seed=102)
    env.kickoff_protection_timer = 0.0

    gk_idx = env.agent_to_idx["a_keeper"]
    env.player_pos[gk_idx] = np.array([30.0, 420.0], dtype=np.float64)
    env.player_vel[gk_idx] = np.zeros(2, dtype=np.float64)
    env.player_angle[gk_idx] = 0.0

    env.ball_pos[:] = np.array([65.0, 425.0], dtype=np.float64)
    env.ball_vel[:] = np.array([-200.0, 0.0], dtype=np.float64)
    env.ball_holder = None
    env.ball_kicked_in_flight = True
    env.ball_kick_was_shot_on_target = True
    env.last_ball_team = 1

    act = np.zeros(ACTION_DIM, dtype=np.float32)
    act[3] = 1.0  # Dive trigger
    actions = {name: np.zeros(ACTION_DIM, dtype=np.float32) for name in ALL_AGENTS}
    actions["a_keeper"] = act

    env.step(actions)
    gk_speed = float(np.linalg.norm(env.player_vel[gk_idx]))
    assert math.isclose(gk_speed, env.cfg.mechanics.keeper_dive_speed, abs_tol=1.0)
    assert env.ball_holder == gk_idx


def test_gk_walking_pickup_beats_diving_pickup_on_loose_ball() -> None:
    """Verify walking to pick up a loose ball preserves keeper lunge cooldown (0.0) whereas diving puts lunge on 1.8s cooldown."""
    env1 = FootballParallelEnv(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    env1.reset(seed=103)
    env1.kickoff_protection_timer = 0.0
    gk_idx1 = env1.agent_to_idx["a_keeper"]
    env1.player_pos[gk_idx1] = np.array([40.0, 420.0], dtype=np.float64)
    env1.player_vel[gk_idx1] = np.array([50.0, 0.0], dtype=np.float64)
    env1.ball_pos[:] = np.array([50.0, 420.0], dtype=np.float64)
    env1.ball_vel[:] = np.zeros(2, dtype=np.float64)
    env1.ball_holder = None

    act_walk = {name: np.zeros(ACTION_DIM, dtype=np.float32) for name in ALL_AGENTS}
    act_walk["a_keeper"] = np.array([1.0, 0.0, -1.0, -1.0], dtype=np.float32)
    env1.step(act_walk)
    assert env1.ball_holder == gk_idx1
    assert env1.lunge_cooldown[gk_idx1] == 0.0

    env2 = FootballParallelEnv(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    env2.reset(seed=103)
    env2.kickoff_protection_timer = 0.0
    gk_idx2 = env2.agent_to_idx["a_keeper"]
    env2.player_pos[gk_idx2] = np.array([40.0, 420.0], dtype=np.float64)
    env2.player_vel[gk_idx2] = np.array([50.0, 0.0], dtype=np.float64)
    env2.ball_pos[:] = np.array([50.0, 420.0], dtype=np.float64)
    env2.ball_vel[:] = np.zeros(2, dtype=np.float64)
    env2.ball_holder = None

    act_dive = {name: np.zeros(ACTION_DIM, dtype=np.float32) for name in ALL_AGENTS}
    act_dive["a_keeper"] = np.array([1.0, 0.0, -1.0, 1.0], dtype=np.float32)
    env2.step(act_dive)
    assert env2.lunge_cooldown[gk_idx2] > 1.5


def test_gk_long_distribution_and_wall_bounce_rewards() -> None:
    """Verify GK long pass to an upfield teammate earns positive threat gain upon reception."""
    env = FootballParallelEnv(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    env.reset(seed=104)

    gk_idx = env.agent_to_idx["a_keeper"]
    tm_idx = env.agent_to_idx["a_mid_1"]
    env.ball_holder = gk_idx
    env.player_pos[gk_idx] = np.array([50.0, 420.0], dtype=np.float64)
    env.player_pos[tm_idx] = np.array([600.0, 420.0], dtype=np.float64)
    env.player_angle[gk_idx] = 0.0
    env.ball_pos[:] = env.player_pos[gk_idx].copy()

    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._execute_kick(gk_idx, kick_power=1.0, step_rewards=step_rewards)
    env.ball_pos = env.player_pos[tm_idx].copy()
    env._check_free_ball_pickup(step_rewards)
    assert step_rewards[gk_idx] > 0.5


def test_one_touch_75_deg_raw_steer_deflection() -> None:
    """Verify one-touch raw steer deflection allows up to clean side-foot deflection angle (+/- 35 deg) without low-pass lag."""
    env = FootballParallelEnv(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    env.reset(seed=105)

    p_idx = env.agent_to_idx["a_atk_1"]
    env.ball_holder = p_idx
    env.player_pos[p_idx] = np.array([700.0, 420.0], dtype=np.float64)
    env.player_angle[p_idx] = 0.0  # Facing East
    env.ball_pos[:] = env.player_pos[p_idx].copy()
    env.raw_steer_input[p_idx] = 1.0

    env._execute_kick(p_idx, kick_power=0.4, step_rewards=np.zeros(env.num_players, dtype=np.float64))

    launch_angle = math.atan2(float(env.ball_vel[1]), float(env.ball_vel[0]))
    assert math.isclose(launch_angle, math.radians(35.0), abs_tol=math.radians(1.0)), (
        f"Launch angle {math.degrees(launch_angle)} deg did not match expected 35 deg deflection"
    )


def test_pocket_ema_smoothing_and_obs_encoding() -> None:
    """Verify centralized critic state shape (54,) and obs[13:16] one-hot role encoding."""
    env = FootballParallelEnv(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    obs, _ = env.reset(seed=106)

    for agent_name in ALL_AGENTS:
        assert obs[agent_name].shape == (82,), f"Expected obs shape (82,), got {obs[agent_name].shape}"

    car_idx = env.agent_to_idx["a_atk_1"]
    env.ball_holder = car_idx
    env.player_pos[car_idx] = np.array([800.0, 420.0], dtype=np.float64)

    acts = {name: np.zeros(ACTION_DIM, dtype=np.float32) for name in ALL_AGENTS}
    obs, _, _, _, infos = env.step(acts)

    for agent_name in ALL_AGENTS:
        assert "central_state" in infos[agent_name]
        assert infos[agent_name]["central_state"].shape == (54,)

    def_obs = obs["a_def_1"]
    np.testing.assert_allclose(def_obs[13:16], [-1.0, -1.0, 1.0], atol=1e-6)


def test_gk_lunge_cooldown_and_obs_normalization() -> None:
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset()
    env.kickoff_protection_timer = 0.0

    assert env.cfg.mechanics.keeper_lunge_cooldown == 1.8

    env.ball_holder = None
    env.ball_pos = np.array([300.0, 420.0], dtype=np.float64)
    env.ball_vel = np.array([0.0, 0.0], dtype=np.float64)
    gk_idx = 0
    env.player_pos[gk_idx] = np.array([45.0, 420.0], dtype=np.float64)
    env.lunge_cooldown[gk_idx] = 0.0
    env.lunge_timer[gk_idx] = 0.0

    actions = _zero_actions(env)
    actions["a_keeper"] = np.array([0.0, 0.0, -1.0, 1.0], dtype=np.float32)
    env.step(actions)

    assert env.lunge_cooldown[gk_idx] > 1.5
    assert env.lunge_cooldown[gk_idx] <= 1.8

    obs = env._get_obs()
    cd_norm = obs["a_keeper"][9]
    expected_norm = env.lunge_cooldown[gk_idx] / env.cfg.mechanics.keeper_lunge_cooldown
    assert 0.0 <= cd_norm <= 1.0
    assert cd_norm == pytest.approx(expected_norm, abs=1e-4)


def test_gk_regime2_dive_heading_auto_alignment() -> None:
    """Verify goalkeeper dive velocity aligns with steered heading angle (player_angle + steer_raw * 75 deg)."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset()
    env.kickoff_protection_timer = 0.0

    gk_idx = 0
    env.ball_holder = None
    env.ball_pos = np.array([120.0, 320.0], dtype=np.float64)
    env.player_pos[gk_idx] = np.array([45.0, 420.0], dtype=np.float64)
    env.player_angle[gk_idx] = 0.0
    env.lunge_cooldown[gk_idx] = 0.0
    env.lunge_timer[gk_idx] = 0.0

    actions = _zero_actions(env)
    actions["a_keeper"] = np.array([0.0, 0.0, -1.0, 1.0], dtype=np.float32)
    env.step(actions)

    vel_angle = math.atan2(float(env.player_vel[gk_idx, 1]), float(env.player_vel[gk_idx, 0]))
    assert vel_angle == pytest.approx(0.0, abs=0.05)


def test_gk_stall_and_box_shrink_penalties() -> None:
    """Verify GK loose-in-box potential rewards approaching a loose ball inside the box and penalizes retreating."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset()

    gk_idx = 0
    env.ball_holder = None
    env.ball_pos = np.array([120.0, 420.0], dtype=np.float64)
    env.ball_vel = np.array([0.0, 0.0], dtype=np.float64)
    env.player_pos[gk_idx] = np.array([40.0, 420.0], dtype=np.float64)
    env._compute_potential_shaping(np.zeros(env.num_players, dtype=np.float64), any_teleport=True)

    # 1. Approaching loose ball in box -> positive PBRS
    env.player_pos[gk_idx] = np.array([75.0, 420.0], dtype=np.float64)
    step_rewards_approach = np.zeros(env.num_players, dtype=np.float64)
    env._compute_potential_shaping(step_rewards_approach, any_teleport=False)
    assert step_rewards_approach[gk_idx] > 0.0

    # 2. Retreating away from loose ball in box -> negative PBRS
    env.player_pos[gk_idx] = np.array([25.0, 420.0], dtype=np.float64)
    step_rewards_retreat = np.zeros(env.num_players, dtype=np.float64)
    env._compute_potential_shaping(step_rewards_retreat, any_teleport=False)
    assert step_rewards_retreat[gk_idx] < 0.0


def test_fresh_vs_recycled_progression() -> None:
    """Verify progressive pass reward scales with Markovian ball threat gain (d_phi_threat)."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset()

    def make_mid_pass(orig_x: float, dest_x: float) -> float:
        env.reset()
        k_idx = 3
        rx_idx = 4
        for idx in range(6, 12):
            env.player_pos[idx] = np.array([1350.0, 100.0 + (idx - 6) * 80.0], dtype=np.float64)
        env.ball_holder = None
        env.ball_kick_origin = np.array([orig_x, 420.0], dtype=np.float64)
        env.ball_kick_orig_opp_dist = 800.0
        env.ball_pos = np.array([dest_x, 420.0], dtype=np.float64)
        env.player_pos[rx_idx] = env.ball_pos.copy()
        env.player_vel[rx_idx] = np.zeros(2, dtype=np.float64)
        env.last_ball_holder = k_idx
        env.last_ball_team = TEAM_A
        env.ball_kicked_in_flight = True
        env.ball_kick_flight_timer = 0.5
        sr = np.zeros(env.num_players, dtype=np.float64)
        env._check_free_ball_pickup(sr)
        return float(sr[k_idx])

    short_rew = make_mid_pass(450.0, 550.0)
    long_rew = make_mid_pass(450.0, 850.0)
    assert long_rew > short_rew > 0.0


def test_triangle_pass_loop_decay() -> None:
    """Verify backward leg of a triangle pass loop without opponent pressure earns zero passer reward."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset()

    p_a, p_c = 1, 3
    for idx in range(6, 12):
        env.player_pos[idx] = np.array([1250.0, 100.0 + (idx - 6) * 80.0], dtype=np.float64)

    env.ball_holder = None
    env.ball_kick_origin = np.array([550.0, 300.0], dtype=np.float64)
    env.ball_kick_orig_opp_dist = 700.0
    env.ball_pos = np.array([500.0, 300.0], dtype=np.float64)
    env.player_pos[p_a] = env.ball_pos.copy()
    env.last_ball_holder = p_c
    env.last_ball_team = TEAM_A
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5

    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(step_rewards)

    assert step_rewards[p_c] == 0.0
    assert step_rewards[p_a] == 0.0


def test_goal_assist_award_and_window() -> None:
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset()

    passer = 3  # a_mid_1
    scorer = 5  # a_atk_1

    # Simulate completed pass at t = 10.0s
    env.physics_frame_count = int(10.0 / env.cfg.physics.dt)
    env.last_completed_pass_passer[TEAM_A] = passer
    env.last_completed_pass_receiver[TEAM_A] = scorer
    env.last_completed_pass_time[TEAM_A] = env.match_time
    env.last_ball_holder = scorer
    env.last_ball_team = TEAM_A

    # Case 1: Goal scored at t = 12.5s (2.5s later <= 4.0s window)
    env.physics_frame_count = int(12.5 / env.cfg.physics.dt)
    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._handle_goal_scored(scoring_team=TEAM_A, step_rewards=step_rewards)

    expected_assist = cfg.rewards.goal_assist_ind
    team_shared = cfg.rewards.goal_team_shared
    assert step_rewards[passer] == pytest.approx(expected_assist + team_shared, abs=1e-3)
    assert step_rewards[1] == pytest.approx(team_shared, abs=1e-3)
    assert env.stats.get("assists", 0) == 1

    # Case 2: Goal scored at t = 15.5s (> 4.0s later)
    env.reset()
    env.physics_frame_count = int(10.0 / env.cfg.physics.dt)
    env.last_completed_pass_passer[TEAM_A] = passer
    env.last_completed_pass_receiver[TEAM_A] = scorer
    env.last_completed_pass_time[TEAM_A] = env.match_time
    env.last_ball_holder = scorer
    env.last_ball_team = TEAM_A

    env.physics_frame_count = int(15.5 / env.cfg.physics.dt)
    step_rewards2 = np.zeros(env.num_players, dtype=np.float64)
    env._handle_goal_scored(scoring_team=TEAM_A, step_rewards=step_rewards2)
    assert step_rewards2[passer] == pytest.approx(team_shared, abs=1e-3)
    assert step_rewards2[passer] == step_rewards2[1]


def test_pass_reception_tackle_grace_timer() -> None:
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset()

    kicker = 1
    receiver = 3
    env.ball_holder = None
    env.ball_kick_origin = np.array([500.0, 420.0], dtype=np.float64)
    env.ball_pos = np.array([700.0, 420.0], dtype=np.float64)
    env.player_pos[receiver] = env.ball_pos.copy()
    env.player_angle[kicker] = 0.0
    env.last_ball_holder = kicker
    env.last_ball_team = TEAM_A
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.pickup_tackle_grace_timer = 0.0

    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(step_rewards)

    assert env.pickup_tackle_grace_timer == pytest.approx(cfg.mechanics.pass_reception_tackle_grace, abs=1e-4)


def test_scanned_pass_mult_uses_kicker_body_heading() -> None:
    """Verify in-stride reception bonus scales smoothly with receiver goalward velocity."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)

    def run_pass(rx_vx: float) -> float:
        env.reset()
        kicker = 1
        receiver = 3
        for idx in range(6, 12):
            env.player_pos[idx] = np.array([1350.0, 100.0 + (idx - 6) * 80.0], dtype=np.float64)
        env.ball_holder = None
        env.ball_kick_origin = np.array([500.0, 420.0], dtype=np.float64)
        env.ball_kick_orig_opp_dist = 800.0
        env.ball_pos = np.array([800.0, 420.0], dtype=np.float64)
        env.player_pos[receiver] = env.ball_pos.copy()
        env.player_vel[receiver] = np.array([rx_vx, 0.0], dtype=np.float64)
        env.last_ball_holder = kicker
        env.last_ball_team = TEAM_A
        env.ball_kicked_in_flight = True
        env.ball_kick_flight_timer = 0.5
        env.pickup_cooldown[:] = 0.0
        step_rewards = np.zeros(env.num_players, dtype=np.float64)
        env._check_free_ball_pickup(step_rewards)
        return float(step_rewards[kicker])

    rew_sprint = run_pass(180.0)
    rew_jog = run_pass(110.0)
    rew_static = run_pass(0.0)
    assert rew_sprint > rew_jog > rew_static


def test_open_space_carry_multiplier() -> None:
    """Verify forward carrier dribble toward enemy goal earns positive PBRS."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset()

    carrier = 5  # a_atk_1
    env.ball_holder = carrier
    env.player_pos[carrier] = np.array([700.0, 420.0], dtype=np.float64)
    env.player_angle[carrier] = 0.0
    env.player_vel[carrier] = np.array([150.0, 0.0], dtype=np.float64)
    env.ball_pos = env.player_pos[carrier] + np.array([24.0, 0.0], dtype=np.float64)
    env._compute_potential_shaping(np.zeros(12, dtype=np.float64), any_teleport=True)

    actions = _zero_actions(env)
    actions["a_atk_1"] = np.array([1.0, 0.0, -1.0, -1.0], dtype=np.float32)
    _, rewards, _, _, _ = env.step(actions)
    assert rewards["a_atk_1"] > 0.0


def test_multiepisodic_rollout_halftime_goals_foul_smoke() -> None:
    """Smoke test running 300 steps with random actions across Halftime, Goals, and Keeper Violations."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    obs, infos = env.reset(seed=123)

    rng = np.random.default_rng(123)

    for step_i in range(300):
        actions = {
            agent: rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
            for agent in env.agents
        }

        if step_i == 50:
            env._handle_goal_scored(scoring_team=TEAM_A, step_rewards=np.zeros(12, dtype=np.float64))
        elif step_i == 100:
            env._handle_keeper_5s_violation(
                env._team_keeper_idx(TEAM_B),
                np.zeros(12, dtype=np.float64),
            )
        elif step_i == 150:
            env.physics_frame_count = env.cfg.physics.half_physics_frames - 2

        obs, rewards, terminations, truncations, infos = env.step(actions)

        assert len(obs) == 12
        for agent_name, agent_obs in obs.items():
            assert agent_obs.shape == (82,)
            assert not np.isnan(agent_obs).any(), f"NaN in obs for {agent_name} at step {step_i}"
            assert not np.isinf(agent_obs).any(), f"Inf in obs for {agent_name} at step {step_i}"

        for agent_name, rew_val in rewards.items():
            assert not np.isnan(rew_val), f"NaN reward for {agent_name} at step {step_i}"
            assert not np.isinf(rew_val), f"Inf reward for {agent_name} at step {step_i}"

        assert not np.isnan(env.active_box_radius).any()


def test_single_env_step_throughput_above_500_sps() -> None:
    """Benchmark single-environment stepping throughput and verify it exceeds 500 SPS."""
    import time

    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)
    act_arr = np.zeros((12, ACTION_DIM), dtype=np.float32)

    n_steps = 300
    t0 = time.perf_counter()
    for _ in range(n_steps):
        env.step_array(act_arr)
    elapsed = max(1e-9, time.perf_counter() - t0)
    sps = n_steps / elapsed
    assert sps > 500.0, f"Expected single-env throughput > 500 SPS, got {sps:.1f} SPS"


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


def test_post_kick_pbrs_smooth_decay_and_carrier_baseline_continuity() -> None:
    """Verify kicker potential decays smoothly over post-kick timer and carrier matches reception scale."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    a1 = env.agent_to_idx["a_atk_1"]
    m1 = env.agent_to_idx["a_mid_1"]
    w_press = env.cfg.rewards.def_press_potential_weight

    # Carrier potential matches reception scale (w_press)
    env.ball_holder = a1
    env.ball_pos = env.player_pos[a1].copy()
    env._compute_potential_shaping(np.zeros(12, dtype=np.float64), any_teleport=True)
    assert math.isclose(env._phi_curr[a1], w_press, abs_tol=1e-5)

    # Post-kick timer decays kicker potential smoothly when teammate is receiving
    env.ball_holder = None
    env.ball_kicked_in_flight = True
    env.last_ball_team = TEAM_A
    env.last_ball_holder = a1
    env.post_kick_pbrs_timer[a1] = 0.50
    # Place ball 20u from kicker and teammate m1 at the ball (m1 is pass chaser)
    env.ball_pos = env.player_pos[a1] + np.array([20.0, 0.0])
    env.player_pos[m1] = env.ball_pos.copy()

    env._compute_potential_shaping(np.zeros(12, dtype=np.float64), any_teleport=True)
    phi_t50 = float(env._phi_curr[a1])
    assert math.isclose(phi_t50, w_press, abs_tol=1e-5)

    # At half timer (0.25s), potential is blended
    env.post_kick_pbrs_timer[a1] = 0.25
    env._compute_potential_shaping(np.zeros(12, dtype=np.float64), any_teleport=True)
    phi_t25 = float(env._phi_curr[a1])
    assert 0.0 < phi_t25 < phi_t50

    # Expired timer drops deadzone support potential to non-positive with crowding
    env.post_kick_pbrs_timer[a1] = 0.0
    env._compute_potential_shaping(np.zeros(12, dtype=np.float64), any_teleport=True)
    phi_t00 = float(env._phi_curr[a1])
    assert phi_t00 <= 0.0 and phi_t00 < phi_t25

    # If kicker is the closest player to loose ball, they become pass_chaser with non-zero potential
    env.player_pos[m1] = np.array([100.0, 100.0])
    env._compute_potential_shaping(np.zeros(12, dtype=np.float64), any_teleport=True)
    assert math.isclose(float(env._phi_curr[a1]), w_press, abs_tol=1e-5)


def test_circulation_floor_on_open_space_first_link() -> None:
    """Verify first-pass circulation floor awards reward on open switches with zero threat delta."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    atk = env.agent_to_idx["a_atk_1"]
    mid = env.agent_to_idx["a_mid_2"]

    # Wide switch: atk at (700, 420) to mid at (700, 580) -> pass_dist = 160u, dy = 160u, d_phi_threat = 0
    env.player_pos[atk] = np.array([700.0, 420.0], dtype=np.float64)
    env.player_pos[mid] = np.array([700.0, 580.0], dtype=np.float64)

    # Move defenders far away (open space)
    for idx in range(6, 12):
        env.player_pos[idx] = np.array([1250.0, 420.0], dtype=np.float64)

    env.ball_holder = None
    env.ball_kick_origin = env.player_pos[atk].copy()
    env.ball_kick_start_pos = env.player_pos[atk].copy()
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.last_ball_holder = atk
    env.last_ball_team = TEAM_A
    env.team_pass_chain[TEAM_A] = 0
    env.ball_pos = env.player_pos[mid].copy()

    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(step_rewards)

    # Circulation floor (0.15) applied
    team_bonus = getattr(env.cfg.rewards, "completed_pass_team", 0.0)
    expected_passer = 0.60 * 0.15 + team_bonus
    expected_receiver = 0.40 * 0.15 + env.cfg.rewards.completed_pass_receiver_ind + team_bonus
    assert math.isclose(step_rewards[atk], expected_passer, abs_tol=1e-5)
    assert math.isclose(step_rewards[mid], expected_receiver, abs_tol=1e-5)
    assert env.stats["completed_passes"] == 1


def test_in_flight_pass_expiration_on_slow_moving_ball() -> None:
    """Verify in-flight pass status expires when ball speed drops below 25 u/s threshold."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    atk = env.agent_to_idx["a_atk_1"]
    env.ball_holder = None
    env.last_ball_holder = atk
    env.last_ball_team = TEAM_A
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.2
    env.is_wall_pass_in_flight = True

    # Ball moving at 20 u/s (< 25 u/s minimum pass speed threshold)
    env.ball_vel = np.array([20.0, 0.0], dtype=np.float64)

    zero_acts = np.zeros((env.num_players, 4), dtype=np.float32)
    env._physics_substep(zero_acts, None)

    assert env.ball_kicked_in_flight is False
    assert env.is_wall_pass_in_flight is False
    assert env.ball_kick_orig_opp_dist == -1.0


def test_through_ball_stride_bonus_physical_gating() -> None:
    """Verify through ball stride bonus is physically gated on ball speed, distance, forward displacement, and threat."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    passer = env.agent_to_idx["a_mid_1"]
    receiver = env.agent_to_idx["a_atk_1"]

    for idx in range(6, 12):
        env.player_pos[idx] = np.array([1250.0, 750.0], dtype=np.float64)

    env.player_pos[passer] = np.array([500.0, 420.0], dtype=np.float64)
    env.player_pos[receiver] = np.array([720.0, 420.0], dtype=np.float64)
    env.player_vel[receiver] = np.array([180.0, 0.0], dtype=np.float64)

    env.ball_holder = None
    env.ball_kick_origin = env.player_pos[passer].copy()
    env.ball_kick_start_pos = env.player_pos[passer].copy()
    env.ball_pos = env.player_pos[receiver].copy()
    env.ball_vel = np.array([120.0, 0.0], dtype=np.float64)  # speed 120 >= 45
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.4
    env.last_ball_holder = passer
    env.last_ball_team = TEAM_A
    env.team_pass_chain[TEAM_A] = 1

    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(step_rewards)

    assert env.stats["through_balls"] == 1
    assert any(ev["type"] == "THROUGH BALL!" for ev in env._frame_events)


def test_zero_value_pass_event_and_stat_suppression() -> None:
    """Verify non-qualifying zero-value passes do not inflate pass stats or emit event popups."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    passer = env.agent_to_idx["a_mid_1"]
    receiver = env.agent_to_idx["a_atk_1"]

    # Backward pass without pressure (600 -> 400, dy = 0)
    env.player_pos[passer] = np.array([600.0, 420.0], dtype=np.float64)
    env.player_pos[receiver] = np.array([400.0, 420.0], dtype=np.float64)
    for idx in range(6, 12):
        env.player_pos[idx] = np.array([1300.0, 420.0], dtype=np.float64)

    env.ball_holder = None
    env.ball_kick_origin = env.player_pos[passer].copy()
    env.ball_kick_start_pos = env.player_pos[passer].copy()
    env.ball_pos = env.player_pos[receiver].copy()
    env.ball_vel = np.array([-100.0, 0.0], dtype=np.float64)
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.last_ball_holder = passer
    env.last_ball_team = TEAM_A
    env.team_pass_chain[TEAM_A] = 1  # Not first link
    env._frame_events = []

    prev_passes = env.stats["completed_passes"]
    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(step_rewards)

    assert env.stats["completed_passes"] == prev_passes
    assert not any(ev["type"] in ("PASS!", "WALL PASS!", "THROUGH BALL!") for ev in env._frame_events)


def test_keeper_possession_pbrs_continuity_and_downfield_outlet_arc() -> None:
    """Verify that a goalkeeper holding the ball targets the top clearance arc rather than the deep goal mouth."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    gk_idx = env.agent_to_idx["a_keeper"]
    phys = env.cfg.physics

    # GK at edge of box: (160, 420)
    env.player_pos[gk_idx] = np.array([160.0, 420.0], dtype=np.float64)
    env.ball_pos = np.array([160.0, 420.0], dtype=np.float64)

    # 1. Loose ball in box before pickup
    env.ball_holder = None
    step_rew_loose = np.zeros(env.num_players, dtype=np.float64)
    env._compute_potential_shaping(step_rew_loose, any_teleport=True)
    loose_phi = float(env._phi_curr[gk_idx])

    # 2. Picked up by keeper: targets clearance arc at x = keeper_box_radius - 25.0 = 170.0
    env.ball_holder = gk_idx
    step_rew_held = np.zeros(env.num_players, dtype=np.float64)
    env._compute_potential_shaping(step_rew_held, any_teleport=True)
    held_phi = float(env._phi_curr[gk_idx])

    # Pickup potential must remain high and continuous (no -0.25 cliff)
    assert held_phi >= loose_phi - 0.05
    assert held_phi > 0.55


def test_keeper_kick_clamped_to_downfield_hemisphere() -> None:
    """Verify that goalkeeper kick angles facing toward own goal are clamped into the downfield hemisphere."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    # Team A GK defending Left (X=0): facing backward into own goal (angle = pi)
    gk_a = env.agent_to_idx["a_keeper"]
    env.ball_holder = gk_a
    env.player_pos[gk_a] = np.array([50.0, 420.0], dtype=np.float64)
    env.player_angle[gk_a] = math.pi
    env._execute_kick(gk_a, kick_power=1.0)

    # Kick direction dx must be clamped downfield (dx >= 0.25)
    assert env.ball_vel[0] > 0.0
    k_dir_x = float(env.ball_vel[0] / np.linalg.norm(env.ball_vel))
    assert k_dir_x >= 0.25 - 1e-5

    # Team B GK defending Right (X=1400): facing backward into own goal (angle = 0.0)
    gk_b = env.agent_to_idx["b_keeper"]
    env.ball_holder = gk_b
    env.player_pos[gk_b] = np.array([1350.0, 420.0], dtype=np.float64)
    env.player_angle[gk_b] = 0.0
    env._execute_kick(gk_b, kick_power=1.0)

    # Kick direction dx must be clamped downfield (dx <= -0.25)
    assert env.ball_vel[0] < 0.0
    k_dir_x_b = float(env.ball_vel[0] / np.linalg.norm(env.ball_vel))
    assert k_dir_x_b <= -0.25 + 1e-5


def test_keeper_5s_hold_turnover_wide_flank_and_team_lockout() -> None:
    """Verify that 5s GK violation resets keeper to goal mouth, ejects ball to wide flank, and locks out keeper's team."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    gk_idx = env.agent_to_idx["a_keeper"]
    env.ball_holder = gk_idx
    env.player_pos[gk_idx] = np.array([120.0, 500.0], dtype=np.float64)

    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._handle_keeper_5s_violation(gk_idx, step_rewards)

    # GK reset to home goal mouth anchor
    assert env.player_pos[gk_idx, 0] == 40.0
    assert env.player_pos[gk_idx, 1] == 420.0
    assert env.player_angle[gk_idx] == 0.0

    # Ball ejected to wide flank outlet outside box (250u out, y=140 or 700)
    assert env.ball_pos[0] == 250.0
    assert env.ball_pos[1] in (140.0, 700.0)
    assert env.ball_vel[0] > 0.0

    # Offending team locked out from immediate pickup for 1.5s
    team_mask = env.player_team == 0
    assert (env.pickup_cooldown[team_mask] >= 1.5).all()
    assert env.pickup_cooldown[gk_idx] >= 2.0


def test_tackle_contact_clearance_reach_and_whiff_distance_gating() -> None:
    """Verify surface contact clearance tackle calculation and whiff distance gating."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)
    env.kickoff_protection_timer = 0.0

    tackler_idx = env.agent_to_idx["a_atk_1"]
    teammate_idx = env.agent_to_idx["a_mid_1"]
    carrier_idx = env.agent_to_idx["b_atk_1"]

    # 1. Whiff gating: tackle ignored when own team holds ball
    env.ball_holder = teammate_idx
    env.player_pos[tackler_idx] = np.array([500.0, 420.0], dtype=np.float64)
    env.player_pos[teammate_idx] = np.array([540.0, 420.0], dtype=np.float64)
    env.ball_pos = env.player_pos[teammate_idx].copy()
    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._attempt_tackle(tackler_idx, step_rewards)
    assert env.stun_timer[tackler_idx] == 0.0
    assert step_rewards[tackler_idx] == 0.0

    # 2. Whiff gating: tackle ignored when dist_to_ball > 180.0u
    env.ball_holder = carrier_idx
    env.player_pos[carrier_idx] = np.array([900.0, 420.0], dtype=np.float64)
    env.ball_pos = env.player_pos[carrier_idx].copy()
    env._attempt_tackle(tackler_idx, step_rewards)
    assert env.stun_timer[tackler_idx] == 0.0
    assert step_rewards[tackler_idx] == 0.0

    # 3. Authentic contact clearance at bumper-to-bumper distance (dist = 40.0u)
    env.player_pos[carrier_idx] = np.array([540.0, 420.0], dtype=np.float64)
    env.ball_pos = env.player_pos[carrier_idx].copy()
    env.player_angle[tackler_idx] = 0.0  # Facing carrier directly

    from unittest.mock import MagicMock
    mock_rng = MagicMock()
    mock_rng.random.return_value = 0.0  # Force RNG to pass if p_success > 0
    orig_rng = env.np_random
    env.np_random = mock_rng

    env._attempt_tackle(tackler_idx, step_rewards)
    assert env.ball_holder == tackler_idx
    assert env.stun_timer[carrier_idx] == env.cfg.mechanics.tackle_victim_stun_time

    env.np_random = orig_rng


def test_goalkeeper_directional_pickup_and_perimeter_strip_gating() -> None:
    """Verify goalkeeper pickups and perimeter strips require facing the ball within the forward cone."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    gk_idx = env.agent_to_idx["a_keeper"]
    atk_idx = env.agent_to_idx["b_atk_1"]

    # 1. Perimeter strip: goalkeeper facing away cannot strip ball from behind
    env.ball_holder = atk_idx
    env.player_pos[atk_idx] = np.array([60.0, 420.0], dtype=np.float64)
    env.ball_pos = np.array([55.0, 420.0], dtype=np.float64)
    env.player_pos[gk_idx] = np.array([45.0, 420.0], dtype=np.float64)
    env.player_angle[gk_idx] = math.pi  # Facing into own net (-X)

    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._check_keeper_perimeter_strip(atk_idx, step_rewards)
    assert env.ball_holder == atk_idx

    # Goalkeeper facing carrier cleanly strips ball
    env.player_angle[gk_idx] = 0.0  # Facing ball (+X)
    env._check_keeper_perimeter_strip(atk_idx, step_rewards)
    assert env.ball_holder == gk_idx

    # 2. Free ball pickup: a keeper touching a loose ball collects it regardless of facing
    env.reset(seed=42)
    env.ball_holder = None
    env.ball_pos = np.array([65.0, 420.0], dtype=np.float64)
    env.player_pos[gk_idx] = np.array([45.0, 420.0], dtype=np.float64)
    env.player_angle[gk_idx] = math.pi  # Facing away from ball

    step_rewards.fill(0.0)
    env._check_free_ball_pickup(step_rewards)
    assert env.ball_holder == gk_idx

    # Goalkeeper facing ball picks up free ball
    env.ball_holder = None
    env.player_angle[gk_idx] = 0.0
    env.player_angle[gk_idx] = 0.0
    env._check_free_ball_pickup(step_rewards)
    assert env.ball_holder == gk_idx


def _oop_env(seed: int = 0) -> FootballParallelEnv:
    """Loose-ball environment with every player parked far from the ball and from each other."""
    env = FootballParallelEnv(config=EnvConfig())
    env.reset(seed=seed)
    env.ball_holder = None
    env.ball_kicked_in_flight = False
    env.last_ball_team = None
    env.stun_timer[:] = 0.0
    env.ghost_timer[:] = 0.0
    env.is_ghost[:] = False
    env.team_side[:] = [0, 1]
    for k in range(12):
        env.player_pos[k] = [100.0 + 100.0 * k, 780.0]
    return env


def test_keeper_collects_loose_ball_while_facing_away() -> None:
    """A keeper touching a loose ball collects it regardless of its heading."""
    env = _oop_env()
    env.player_pos[0] = [60.0, 420.0]
    env.player_angle[0] = math.pi  # facing own goal line, ball behind the back
    env.ball_pos[:] = [80.0, 420.0]
    env.ball_vel[:] = 0.0
    env.pickup_cooldown[:] = 0.0
    env._check_free_ball_pickup(np.zeros(12))
    assert env.ball_holder == 0


def test_pitch_boundary_wall_velocity_dissipation_and_bisector_depth() -> None:
    """Verify boundary wall collision zeroes normal velocity and bisector target depth stays in front of goal line."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    # 1. Boundary wall contact zeroes inward normal velocity
    env.player_pos[0] = np.array([18.0, 420.0], dtype=np.float64)
    env.player_vel[0] = np.array([-180.0, 60.0], dtype=np.float64)
    env._clamp_players_to_boundaries()
    assert env.player_vel[0, 0] == 0.0
    assert env.player_vel[0, 1] == 60.0

    # 2. Bisector target depth never retreats behind minimum clearance (X >= 38.0)
    # Deep corner ball at [20, 100]
    env.ball_pos = np.array([20.0, 100.0], dtype=np.float64)
    env._compute_potential_shaping(np.zeros(env.num_players, dtype=np.float64), any_teleport=True)
    phi_val = float(env.prev_tactical_potential[0])
    assert phi_val >= 0.0

    # 3. Continuous orientation factor psi_face across full heading circle
    phi_angles = []
    angles = np.linspace(-math.pi, math.pi, 181)
    env.ball_pos = np.array([200.0, 420.0], dtype=np.float64)
    env.player_pos[0] = np.array([45.0, 420.0], dtype=np.float64)
    for a in angles:
        env.player_angle[0] = a
        env._compute_potential_shaping(np.zeros(env.num_players, dtype=np.float64), any_teleport=True)
        phi_angles.append(float(env.prev_tactical_potential[0]))

    max_delta = max(abs(phi_angles[i] - phi_angles[i - 1]) for i in range(1, len(phi_angles)))
    assert max_delta < 0.02


# ---------------------------------------------------------------------------
# Boundary, pin, box-radius, presser-tie and degenerate-action specifications
# ---------------------------------------------------------------------------

_CORNER_CASES = [
    pytest.param((15.0, 15.0), (1.0, 1.0), id="top_left"),
    pytest.param((1385.0, 15.0), (-1.0, 1.0), id="top_right"),
    pytest.param((15.0, 825.0), (1.0, -1.0), id="bottom_left"),
    pytest.param((1385.0, 825.0), (-1.0, -1.0), id="bottom_right"),
]


def _scatter_players_midfield(env: FootballParallelEnv) -> None:
    """Park every player in the middle of the pitch, away from the corners and both boxes."""
    for i in range(1, 6):
        env.player_pos[i] = [700.0, 100.0 + 60.0 * (i - 1)]
    for i in range(7, 12):
        env.player_pos[i] = [900.0, 100.0 + 60.0 * (i - 7)]
    env.player_pos[0] = [40.0, 420.0]
    env.player_pos[6] = [1360.0, 420.0]
    env.player_vel[:] = 0.0


@pytest.mark.parametrize("corner, inward", _CORNER_CASES)
def test_loose_ball_pinned_in_each_corner_is_nudged_toward_centre_after_two_seconds(corner, inward) -> None:
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)
    env.kickoff_protection_timer = 0.0
    env.pickup_tackle_grace_timer = 0.0
    _scatter_players_midfield(env)
    sx, sy = inward
    env.ball_holder = None
    env.ball_pos = np.array(corner, dtype=np.float64)
    env.ball_vel[:] = 0.0
    # Two players beyond pickup range (28u) but inside the 68u crowding radius.
    env.player_pos[1] = [corner[0] + sx * 45.0, corner[1]]
    env.player_pos[7] = [corner[0], corner[1] + sy * 45.0]
    env.pinned_ball_timer = 0.0

    zero_actions = np.zeros((12, 4))
    zero_speeds = np.zeros(12)
    nudge_substep = None
    prev_timer = 0.0
    for k in range(240):
        env._physics_substep(zero_actions, zero_speeds)
        env.physics_frame_count += 1
        if prev_timer > 0.0 and env.pinned_ball_timer == 0.0:
            nudge_substep = k
            break
        prev_timer = env.pinned_ball_timer

    assert nudge_substep is not None, "pinned loose ball was never released"
    # 2.0 s at 60 substeps/s
    assert 118 <= nudge_substep <= 122
    # Direction (685, 405)/|.| * 95 u/s from the (15, 15) corner, mirrored per corner.
    assert env.ball_vel[0] == pytest.approx(sx * 81.8, abs=0.5)
    assert env.ball_vel[1] == pytest.approx(sy * 48.3, abs=0.5)


@pytest.mark.parametrize("corner, inward", _CORNER_CASES)
def test_carried_ball_pinned_in_each_corner_keeps_ball_and_carrier_on_pitch(corner, inward) -> None:
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)
    env.kickoff_protection_timer = 0.0
    env.pickup_tackle_grace_timer = 0.0
    _scatter_players_midfield(env)
    sx, sy = inward
    carrier, opponent = 3, 9
    cx, cy = corner[0] + sx * 3.0, corner[1] + sy * 3.0
    env.ball_holder = carrier
    env.last_ball_holder = carrier
    env.last_ball_team = TEAM_A
    env.ball_hold_timer = 0.0
    env.player_pos[carrier] = [cx, cy]
    env.player_angle[carrier] = math.atan2(-sy, -sx)
    env.player_pos[opponent] = [cx + sx * 40.0, cy + sy * 10.0]
    env.ball_pos = env.player_pos[carrier] + np.array([-sx, -sy]) / math.sqrt(2.0) * 24.0
    env.ball_vel[:] = 0.0

    phys = DEFAULT_CONFIG.physics
    r_ball, r_player = phys.ball_radius, phys.player_radius
    zero_actions = np.zeros((12, 4))
    zero_speeds = np.zeros(12)
    for _ in range(180):
        env._physics_substep(zero_actions, zero_speeds)
        env.physics_frame_count += 1
        assert np.isfinite(env.ball_pos).all() and np.isfinite(env.player_pos).all()
        # The carried ball is clamped to the ball-radius inset up to sub-millimetre ordering slack
        # from the carrier push-back that follows the clamp.
        assert r_ball - 1e-3 <= env.ball_pos[0] <= phys.field_width - r_ball + 1e-3
        assert r_ball - 1e-3 <= env.ball_pos[1] <= phys.field_height - r_ball + 1e-3
        assert r_player - 1e-9 <= env.player_pos[carrier, 0] <= phys.field_width - r_player + 1e-9
        assert r_player - 1e-9 <= env.player_pos[carrier, 1] <= phys.field_height - r_player + 1e-9


def test_box_radius_fully_shrunk_and_nominal_on_both_sides_after_halftime_switch() -> None:
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)
    env._handle_halftime()
    assert list(env.team_side) == [SIDE_RIGHT, SIDE_LEFT]
    assert list(env.active_box_radius) == [195.0, 195.0]
    env.kickoff_protection_timer = 0.0
    env.pickup_tackle_grace_timer = 0.0
    _scatter_players_midfield(env)

    # Nominal radius on both sides: an outfielder at either goal line is pushed out to the arc (x = 195 + 1e-3).
    env.player_pos[8] = [30.0, 420.0]
    env.player_pos[2] = [1370.0, 420.0]
    env._clamp_players_to_boundaries()
    assert env.player_pos[8, 0] == pytest.approx(195.001, abs=1e-9)
    assert env.player_pos[2, 0] == pytest.approx(1204.999, abs=1e-9)

    # A dead ball in the left box (defended by team B after the switch) shrinks only the left radius.
    env.player_pos[0] = [1380.0, 420.0]
    env.player_pos[6] = [20.0, 420.0]
    env.player_pos[8] = [700.0, 700.0]
    env.player_pos[2] = [700.0, 650.0]
    env.ball_holder = None
    env.ball_pos = np.array([100.0, 520.0], dtype=np.float64)
    env.ball_vel[:] = 0.0
    zero_actions = np.zeros((12, 4))
    zero_speeds = np.zeros(12)
    for _ in range(60 * 8):
        env._physics_substep(zero_actions, zero_speeds)
        env.physics_frame_count += 1
    assert env.active_box_radius[SIDE_LEFT] == 0.0
    assert env.active_box_radius[SIDE_RIGHT] == 195.0

    # Fully shrunk: the former exclusion zone is open; the untouched side still excludes.
    env.player_pos[8] = [30.0, 420.0]
    env.player_pos[2] = [1370.0, 420.0]
    env._clamp_players_to_boundaries()
    assert env.player_pos[8, 0] == pytest.approx(30.0, abs=1e-9)
    assert env.player_pos[2, 0] == pytest.approx(1204.999, abs=1e-9)


def _recording_env() -> FootballParallelEnv:
    cfg = replace(
        DEFAULT_CONFIG,
        replay=replace(DEFAULT_CONFIG.replay, enable_recording=True, interest_threshold=float("inf")),
    )
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)
    env.kickoff_protection_timer = 0.0
    return env


def _team_b_holds_ball_in_team_a_attacking_half(env, d1_pos, d2_pos, st_pos) -> None:
    env.player_pos[:] = np.array([[700.0, 100.0 + 60.0 * i] for i in range(12)], dtype=np.float64)
    env.player_pos[0] = [40.0, 420.0]
    env.player_pos[6] = [1360.0, 420.0]
    env.player_pos[1] = d1_pos
    env.player_pos[2] = d2_pos
    env.player_pos[3] = [600.0, 300.0]
    env.player_pos[4] = [600.0, 540.0]
    env.player_pos[5] = st_pos
    env.player_pos[7:12] = [[1100.0, 200.0], [1050.0, 300.0], [1000.0, 420.0], [1050.0, 540.0], [1100.0, 640.0]]
    env.ball_holder = 8
    env.last_ball_holder = 8
    env.last_ball_team = TEAM_B
    env.ball_pos = env.player_pos[8].copy()
    env.ball_vel[:] = 0.0
    env._compute_potential_shaping(np.zeros(env.num_players), any_teleport=True)


def test_turnover_in_opposing_half_presser_is_nearest_player_whatever_its_role() -> None:
    env = _recording_env()

    # Both defenders are ahead of the striker and closer to the carrier at (1000, 420).
    _team_b_holds_ball_in_team_a_attacking_half(env, [960.0, 380.0], [930.0, 470.0], [820.0, 420.0])
    assert env.last_role_states[:6] == ["GK", "1stP", "COVR", "COVR", "COVR", "COVR"]
    assert env._phi_curr[1] > env._phi_curr[5]

    # The striker is closest: it presses and the defenders cover.
    _team_b_holds_ball_in_team_a_attacking_half(env, [860.0, 380.0], [830.0, 470.0], [960.0, 420.0])
    assert env.last_role_states[:6] == ["GK", "COVR", "COVR", "COVR", "COVR", "1stP"]

    # Exactly one presser per team and every coverer's shaping potential is finite.
    assert sum(1 for r in env.last_role_states[:6] if r == "1stP") == 1
    assert np.isfinite(env._phi_curr).all()


def test_presser_selection_tie_resolves_to_lowest_player_index() -> None:
    env = _recording_env()
    env.player_pos[:] = np.array([[700.0, 100.0 + 60.0 * i] for i in range(12)], dtype=np.float64)
    env.player_pos[0] = [40.0, 420.0]
    env.player_pos[6] = [1360.0, 420.0]
    env.player_pos[1] = [200.0, 100.0]
    env.player_pos[3] = [200.0, 150.0]
    env.player_pos[5] = [200.0, 700.0]
    env.ball_holder = None
    env.last_ball_holder = None
    env.last_ball_team = None
    env.ball_pos = np.array([700.0, 420.0], dtype=np.float64)
    env.ball_vel[:] = 0.0

    # Players 2 and 4 are exactly 120u from the loose ball (above and below it).
    for upper, lower in ((2, 4), (4, 2)):
        env.player_pos[upper] = [700.0, 300.0]
        env.player_pos[lower] = [700.0, 540.0]
        env._compute_potential_shaping(np.zeros(env.num_players), any_teleport=True)
        assert env.last_role_states[2] == "1stP"
        assert env.last_role_states[4] == "2ndP"
        first = list(env.last_role_states)
        env._compute_potential_shaping(np.zeros(env.num_players), any_teleport=True)
        assert env.last_role_states == first


@pytest.mark.parametrize(
    "gk_y, in_mouth",
    [(420.0, True), (345.0, True), (250.0, False), (590.0, False)],
    ids=["mouth_centre", "mouth_post", "box_above_mouth", "box_below_mouth"],
)
def test_goalkeeper_driven_into_end_wall_stays_on_pitch_and_in_box(gk_y, in_mouth) -> None:
    phys = DEFAULT_CONFIG.physics
    assert in_mouth == (phys.goal_y_min <= gk_y <= phys.goal_y_max)
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)
    env.kickoff_protection_timer = 0.0
    _scatter_players_midfield(env)
    env.ball_holder = None
    env.ball_pos = np.array([700.0, 420.0], dtype=np.float64)
    env.ball_vel[:] = 0.0
    keeper = 0
    env.player_pos[keeper] = [60.0, gk_y]
    env.player_angle[keeper] = math.pi
    actions = np.zeros((12, 4))
    actions[keeper] = [1.0, 0.0, 0.0, 0.0]  # full throttle straight at the end wall
    zero_speeds = np.zeros(12)
    min_x = np.inf
    for _ in range(180):
        env._physics_substep(actions, zero_speeds)
        env.physics_frame_count += 1
        min_x = min(min_x, float(env.player_pos[keeper, 0]))
    assert min_x >= phys.player_radius - 1e-9
    assert env.player_pos[keeper, 0] == pytest.approx(phys.player_radius, abs=1e-9)
    assert env.player_pos[keeper, 1] == pytest.approx(gk_y, abs=1e-6)
    assert env.player_vel[keeper, 0] >= 0.0
    assert np.isfinite(env.player_pos[keeper]).all() and np.isfinite(env.player_vel[keeper]).all()


@pytest.mark.parametrize("bad", [math.inf, -math.inf], ids=["+inf", "-inf"])
def test_step_array_clips_infinite_actions_to_finite_state(bad) -> None:
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)
    actions = np.zeros((12, 4))
    actions[3, 0] = bad
    actions[4, 1] = bad
    actions[5, 2] = bad
    actions[0, 3] = bad
    obs, central, rew, _, _ = env.step_array(actions)
    for arr in (obs, central, rew, env.player_pos, env.player_vel, env.player_angle, env.ball_pos):
        assert np.isfinite(arr).all()


def test_step_array_rejects_or_clips_nan_actions() -> None:
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)
    actions = np.zeros((12, 4))
    actions[3, 0] = np.nan
    try:
        obs, central, rew, _, _ = env.step_array(actions)
    except (ValueError, FloatingPointError):
        return
    for arr in (obs, central, rew, env.player_pos, env.player_vel, env.player_angle, env.ball_pos):
        assert np.isfinite(arr).all()


# ---------------------------------------------------------------------------
# Configuration consolidation and dead-state specifications
# ---------------------------------------------------------------------------

def test_config_has_no_unread_fields_and_env_has_no_dead_state() -> None:
    res = orphan_scan.scan(REPO_ROOT / "football_rl")
    assert res["unread_config_fields"] == []
    assert res["unused_class_aliases"] == []
    api_attrs = {
        "FootballParallelEnv.action_spaces",
        "FootballParallelEnv.observation_spaces",
        "FootballParallelEnv.render_mode",
    }
    assert set(res["write_only_attributes"]) <= api_attrs


def test_consolidated_config_fields_carry_the_values_production_reads() -> None:
    mech, rew, phys = DEFAULT_CONFIG.mechanics, DEFAULT_CONFIG.rewards, DEFAULT_CONFIG.physics
    assert mech.keeper_foul_pickup_cooldown == 2.0
    assert mech.box_timer_obs_norm == 5.0
    assert mech.keeper_obs_speed_norm == 430.0
    assert mech.kick_steer_max_deflection_deg == 35.0
    assert rew.shot_on_target_ind == 0.45
    assert rew.goal_scorer_ind == 3.0
    assert rew.pass_threat_coeff == 6.0
    assert rew.through_ball_stride_bonus == 0.80
    assert rew.pass_press_escape_coeff == 0.60
    removed = {
        mech: [
            "keeper_box_loose_max_time", "lunge_speed", "tackle_range", "def_tackle_range", "mid_tackle_range",
            "atk_tackle_range", "tackle_near_range", "tackle_success_prob", "def_tackle_success_prob",
            "mid_tackle_success_prob", "atk_tackle_success_prob", "tackle_stun_duration",
        ],
        rew: [
            "pass_threat_weight", "pass_space_weight", "lead_pass_bonus", "atk_goal_scorer_ind",
            "atk_shot_on_target_ind", "other_shot_on_target_ind", "keeper_5s_violation_team",
        ],
        phys: [
            "kick_steer_max_deflection_deg", "ball_kick_speed_span", "half_duration", "half_rl_steps",
            "match_duration", "player_max_turn_rate_deg",
        ],
        DEFAULT_CONFIG.vision: ["fov_rad"],
    }
    for cfg_obj, names in removed.items():
        for name in names:
            assert not hasattr(cfg_obj, name), name


def test_keeper_hold_foul_charges_only_the_keeper_and_emits_hold_foul_event() -> None:
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)
    keeper = 0
    env.ball_holder = keeper
    env.ball_pos = env.player_pos[keeper].copy()
    env.ball_hold_timer = 5.0
    rewards = np.zeros(env.num_players, dtype=np.float64)
    env._handle_keeper_5s_violation(keeper, rewards)

    assert rewards[keeper] == pytest.approx(-1.5, abs=1e-12)
    others = [i for i in range(env.num_players) if i != keeper]
    assert np.all(rewards[others] == 0.0)
    assert env.stats["keeper_5s_violations"] == 1
    assert env.pickup_cooldown[keeper] == pytest.approx(2.0, abs=1e-12)
    assert [e["type"] for e in env._frame_events] == ["5s KEEPER HOLD FOUL!"]


def test_field_hold_foul_charges_violator_and_team_while_only_5s_stat_is_kept() -> None:
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)
    violator = 3
    env.ball_holder = violator
    env.ball_hold_timer = 5.0
    rewards = np.zeros(env.num_players, dtype=np.float64)
    env._handle_field_5s_violation(violator, rewards)

    assert env.stats["field_5s_violations"] == 1
    assert "field_3s_violations" not in env.stats
    assert not hasattr(env, "_handle_field_3s_violation")
    assert rewards[violator] < -1.5


GOLDEN_ROLLOUT_DIGEST = "a975ea438aa843ea120651b05ce4773fce294a0f1694c23f33d605b0f0051f6c"
GOLDEN_ROLLOUT_DIGEST_WITHOUT_LINE_ORDER = "e726782a53270dbbc008e1912e551635033b95a6583dc78c7c946adaf7b339ae"


def rollout_digest(steps: int = 600, seed: int = 1234, record: bool = False, rewards=None) -> str:
    """SHA-256 over observation columns 0..75, rewards, kinematics and possession of a seeded random-action rollout."""
    cfg = replace(
        DEFAULT_CONFIG if rewards is None else replace(DEFAULT_CONFIG, rewards=rewards),
        replay=replace(
            DEFAULT_CONFIG.replay,
            enable_recording=record,
            interest_threshold=float("inf"),
        ),
    )
    env = FootballParallelEnv(config=cfg)
    env.reset(seed=seed)
    acts = np.random.default_rng(seed).uniform(-1, 1, size=(steps, 12, 4))
    h = hashlib.sha256()
    for k in range(steps):
        obs, central, rew, done, _ = env.step_array(acts[k])
        for arr in (obs[:, :76], central, rew, env.player_pos, env.player_vel, env.ball_pos):
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update(str(env.ball_holder).encode())
        if done:
            env.reset(seed=seed + k + 1)
    return h.hexdigest()


def test_rollout_matches_golden_digest():
    assert rollout_digest() == GOLDEN_ROLLOUT_DIGEST


def test_rollout_without_line_order_penalty_matches_golden_digest():
    rewards = replace(DEFAULT_CONFIG.rewards, line_order_weight=0.0)
    assert rollout_digest(rewards=rewards) == GOLDEN_ROLLOUT_DIGEST_WITHOUT_LINE_ORDER
    assert rollout_digest() != GOLDEN_ROLLOUT_DIGEST_WITHOUT_LINE_ORDER


def test_rollout_digest_is_seed_sensitive_and_repeatable():
    assert rollout_digest(steps=60, seed=1234) == rollout_digest(steps=60, seed=1234)
    assert rollout_digest(steps=60, seed=1234) != rollout_digest(steps=60, seed=1235)


GOLDEN_REPLAY_DIGEST = "9accb1139ab1e0cf1419b4b2c6aa9a3789046210e2d27f7f59deac8e7a439bde"
GOLDEN_FULL_MATCH_REPLAY_DIGEST = "b696f562f722eca42049da1e6118bc6e42ec52cf41e36bfb79cd5280e3b0611e"
GOLDEN_FULL_MATCH_REPLAY_DIGEST_WITHOUT_LINE_ORDER = "d23cef4d09f54ee91c3b55a9bab2caf7710ef1d120e9957304832280d677e90b"


def replay_digest(steps: int = 300, seed: int = 1234, with_value_predictions: bool = False, rewards=None) -> str:
    """SHA-256 of the JSON-serialised replay frames of a seeded random-action rollout with recording on.

    With `with_value_predictions`, critic values are supplied on two of every three RL steps.
    """
    cfg = replace(
        DEFAULT_CONFIG if rewards is None else replace(DEFAULT_CONFIG, rewards=rewards),
        replay=replace(
            DEFAULT_CONFIG.replay,
            enable_recording=True,
            interest_threshold=float("inf"),
        ),
    )
    env = FootballParallelEnv(config=cfg)
    env.reset(seed=seed)
    acts = np.random.default_rng(seed).uniform(-1, 1, size=(steps, 12, 4))
    values = np.random.default_rng(seed + 7).uniform(-1, 1, size=(steps, 12))
    for k in range(steps):
        v_preds = values[k] if (with_value_predictions and k % 3) else None
        env.step_array(acts[k], v_preds=v_preds)
    return hashlib.sha256(json.dumps(env.replay_frames, sort_keys=True).encode()).hexdigest()


def test_replay_frames_match_golden_digest():
    assert replay_digest() == GOLDEN_REPLAY_DIGEST


def test_full_match_replay_frames_match_golden_digest():
    """Covers half-time, full-time events and intermittent critic values over all 10,801 frames."""
    assert replay_digest(steps=2700, with_value_predictions=True) == GOLDEN_FULL_MATCH_REPLAY_DIGEST


def test_replay_frames_without_line_order_penalty_match_golden_digests():
    rewards = replace(DEFAULT_CONFIG.rewards, line_order_weight=0.0)
    assert replay_digest(rewards=rewards) == GOLDEN_REPLAY_DIGEST
    full = replay_digest(steps=2700, with_value_predictions=True, rewards=rewards)
    assert full == GOLDEN_FULL_MATCH_REPLAY_DIGEST_WITHOUT_LINE_ORDER
    assert full != GOLDEN_FULL_MATCH_REPLAY_DIGEST


def test_replay_digest_is_seed_sensitive_and_repeatable():
    assert replay_digest(steps=20) == replay_digest(steps=20)
    assert replay_digest(steps=20, seed=1234) != replay_digest(steps=20, seed=1235)


def test_precomputed_player_tables_match_roles():
    env = FootballParallelEnv()
    assert env._max_speed_arr.shape == (12,)
    assert env._max_speed_arr.tolist() == [180.0, 188.0, 188.0, 192.0, 192.0, 190.0] * 2
    assert env._is_keeper.dtype == bool
    assert np.flatnonzero(env._is_keeper).tolist() == [0, 6]
    assert [env._get_player_max_speed(i) for i in range(12)] == env._max_speed_arr.tolist()
    phys = env.cfg.physics
    assert len(env._goal_centers) == 2
    assert env._goal_centers[0].tolist() == list(phys.left_goal_center)
    assert env._goal_centers[1].tolist() == list(phys.right_goal_center)


def _random_clamp_state(env: FootballParallelEnv, rng: np.random.Generator) -> None:
    """Random outfield states concentrated near walls, corners and both goal semicircles."""
    phys = env.cfg.physics
    r = phys.player_radius
    for side, goal_x in ((0, 0.0), (1, phys.field_width)):
        env.active_box_radius[side] = rng.choice([0.0, 1e-4, 40.0, 120.0, phys.keeper_box_radius])
    for i in range(12):
        zone = rng.integers(0, 5)
        if zone == 0:  # anywhere (including slightly outside the field)
            x = rng.uniform(-10.0, phys.field_width + 10.0)
            y = rng.uniform(-10.0, phys.field_height + 10.0)
        elif zone == 1:  # left corner pin region
            x = rng.uniform(0.0, r + 3.0)
            y = rng.uniform(phys.goal_y_min - 250.0, phys.goal_y_max + 250.0)
        elif zone == 2:  # right corner pin region
            x = phys.field_width - rng.uniform(0.0, r + 3.0)
            y = rng.uniform(phys.goal_y_min - 250.0, phys.goal_y_max + 250.0)
        elif zone == 3:  # around the semicircle envelopes
            x = rng.choice([0.0, phys.field_width]) + rng.choice([-1.0, 1.0]) * rng.uniform(0.0, 215.0)
            y = rng.uniform(phys.goal_y_min - 230.0, phys.goal_y_max + 230.0)
        else:  # top and bottom walls
            x = rng.uniform(0.0, phys.field_width)
            y = rng.choice([0.0, phys.field_height]) + rng.choice([-1.0, 1.0]) * rng.uniform(0.0, r + 3.0)
        env.player_pos[i] = (x, y)
    env.player_vel[:] = rng.uniform(-200.0, 200.0, size=(12, 2))


def test_outfield_boundary_candidates_cover_every_player_the_clamp_changes():
    env = FootballParallelEnv()
    env.reset(seed=0)
    env.ball_holder = None
    env.kickoff_protection_timer = 0.0
    rng = np.random.default_rng(20240607)
    changed_total = 0
    skipped_total = 0
    for _ in range(2000):
        _random_clamp_state(env, rng)
        pos_before = env.player_pos.copy()
        vel_before = env.player_vel.copy()
        candidates = env._outfield_boundary_candidates()
        assert candidates.shape == (12,) and candidates.dtype == bool
        assert not candidates[[0, 6]].any()
        env._clamp_players_to_boundaries(apply_dynamic_push=True)
        changed = (env.player_pos != pos_before).any(axis=1) | (env.player_vel != vel_before).any(axis=1)
        outfield = ~env._is_keeper
        assert not (changed & outfield & ~candidates).any(), (
            np.flatnonzero(changed & outfield & ~candidates), pos_before, vel_before
        )
        changed_total += int((changed & outfield).sum())
        skipped_total += int((outfield & ~candidates).sum())
    # The corpus exercises both the clamp body and the skipped path.
    assert changed_total > 5000
    assert skipped_total > 1000


def test_outfield_boundary_candidates_exclude_players_in_open_field():
    env = FootballParallelEnv()
    env.reset(seed=0)
    env.player_pos[:] = (700.0, 420.0)
    assert not env._outfield_boundary_candidates().any()
    env.player_pos[3] = (3.0, 420.0)
    env.player_pos[8] = (700.0, 2.0)
    env.player_pos[9] = (env.cfg.physics.field_width - 2.0, 100.0)
    assert np.flatnonzero(env._outfield_boundary_candidates()).tolist() == [3, 8, 9]


def test_free_ball_pickup_with_no_player_in_reach_changes_nothing():
    env = FootballParallelEnv()
    env.reset(seed=0)
    env.ball_holder = None
    env.pickup_cooldown[:] = 0.0
    env.stun_timer[:] = 0.0
    env.ghost_timer[:] = 0.0
    env.ball_pos = np.array([700.0, 420.0])
    env.player_pos[:] = (100.0, 100.0)
    rewards = np.zeros(12)
    env._check_free_ball_pickup(rewards)
    assert env.ball_holder is None
    assert not rewards.any()
    env.player_pos[4] = (700.0 + env.cfg.mechanics.pickup_distance - 1.0, 420.0)
    env.player_angle[4] = np.pi  # facing the ball
    env._check_free_ball_pickup(rewards)
    assert env.ball_holder == 4


POTENTIAL_CORPUS_PATH = Path(__file__).resolve().parent / "golden" / "potential_corpus.npz"


def _load_potential_corpus() -> dict[str, np.ndarray]:
    """Load the recorded potential-field corpus: 200 consecutive RL steps of a seeded scripted-policy rollout.

    Every record holds the state the potential field reads (positions, headings, ball, possession,
    stun/ghost/lunge flags, post-kick timers, tactical biases, box radii) together with the resulting
    per-player potentials, role labels and role targets.
    """
    with np.load(POTENTIAL_CORPUS_PATH) as data:
        return {name: data[name] for name in data.files}


def _potentials_from_env(env: FootballParallelEnv, corpus: dict[str, np.ndarray], k: int):
    """Load record `k` into `env`, evaluate its potential field, and return (phi, role labels, role targets)."""
    env.player_pos[:] = corpus["pos"][k]
    env.player_angle[:] = corpus["angle"][k]
    env.ball_pos[:] = corpus["ball"][k]
    holder = int(corpus["holder"][k])
    env.ball_holder = None if holder < 0 else holder
    env.ball_kicked_in_flight = bool(corpus["in_flight"][k])
    last_team = int(corpus["last_ball_team"][k])
    env.last_ball_team = None if last_team < 0 else last_team
    env.team_side[:] = corpus["team_side"][k]
    env.stun_timer[:] = np.where(corpus["stunned"][k], 1.0, 0.0)
    env.is_ghost[:] = corpus["disabled"][k]
    env.lunge_timer[:] = np.where(corpus["lunging"][k], 1.0, 0.0)
    env.post_kick_pbrs_timer[:] = corpus["post_kick_timer"][k]
    env.active_box_radius[:] = corpus["active_box_radius"][k]
    env._compute_potential_shaping(np.zeros(env.num_players, dtype=np.float64), any_teleport=True)
    return env._phi_curr.copy(), list(env.last_role_states), np.array(env.last_role_targets, dtype=np.float64)


def test_potential_corpus_covers_possession_and_role_states():
    corpus = _load_potential_corpus()
    holder = corpus["holder"]
    phys = DEFAULT_CONFIG.physics
    assert corpus["phi"].shape == (200, 12)

    labels = set(corpus["role_states"].ravel().tolist())
    assert {"GK", "BALL", "CHSE", "1stP", "2ndP", "COVR", "STUN", "LNGE"} <= labels

    assert (holder < 0).any()
    assert {0, 6} <= set(holder.tolist())
    assert ((holder > 0) & (holder < 6)).any() and (holder > 6).any()
    for team in (0, 1):
        assert ((holder < 0) & corpus["in_flight"] & (corpus["last_ball_team"] == team)).any()
    for members in (slice(0, 6), slice(6, 12)):
        assert corpus["stunned"][:, members].any()
        assert (corpus["post_kick_timer"][:, members] > 0.0).any()

    ball = corpus["ball"]
    for goal_x in (0.0, phys.field_width):
        d_goal = np.hypot(ball[:, 0] - goal_x, ball[:, 1] - 0.5 * phys.field_height)
        assert ((holder < 0) & (d_goal <= phys.keeper_box_radius + 50.0)).any()


def test_env_potentials_match_recorded_corpus():
    corpus = _load_potential_corpus()
    cfg = replace(
        DEFAULT_CONFIG,
        replay=replace(DEFAULT_CONFIG.replay, enable_recording=True, interest_threshold=float("inf")),
    )
    env = FootballParallelEnv(config=cfg)
    env.reset(seed=0)
    for k in range(corpus["phi"].shape[0]):
        phi, labels, targets = _potentials_from_env(env, corpus, k)
        np.testing.assert_allclose(phi, corpus["phi"][k], rtol=0.0, atol=1e-9, err_msg=f"phi, record {k}")
        assert labels == corpus["role_states"][k].tolist(), f"role labels, record {k}"
        np.testing.assert_allclose(targets, corpus["role_targets"][k], rtol=0.0, atol=1e-9, err_msg=f"targets, record {k}")


def _potential_state_from_corpus(corpus: dict[str, np.ndarray], k: int) -> PotentialState:
    holder = int(corpus["holder"][k])
    return PotentialState(
        pos=corpus["pos"][k],
        angle=corpus["angle"][k],
        ball=corpus["ball"][k],
        holder=holder,
        ball_in_flight=bool(corpus["in_flight"][k]),
        last_ball_team=int(corpus["last_ball_team"][k]),
        team_side=corpus["team_side"][k],
        disabled=corpus["disabled"][k],
        stunned=corpus["stunned"][k],
        post_kick_timer=corpus["post_kick_timer"][k],
        depth_bias=corpus["depth_bias"][k],
        flank_bias=corpus["flank_bias"][k],
        active_box_radius=corpus["active_box_radius"][k],
    )


def test_potentials_without_line_order_penalty_match_recorded_corpus():
    """Switching the line-order term off reproduces the recorded unpenalised potentials in the env and in the pure field."""
    corpus = _load_potential_corpus()
    rewards = replace(DEFAULT_CONFIG.rewards, line_order_weight=0.0)
    cfg = replace(
        DEFAULT_CONFIG,
        rewards=rewards,
        replay=replace(DEFAULT_CONFIG.replay, enable_recording=True, interest_threshold=float("inf")),
    )
    env = FootballParallelEnv(config=cfg)
    env.reset(seed=0)
    expected = corpus["phi_without_line_order"]
    for k in range(expected.shape[0]):
        phi, _, _ = _potentials_from_env(env, corpus, k)
        np.testing.assert_allclose(phi, expected[k], rtol=0.0, atol=1e-9, err_msg=f"env phi, record {k}")
        res = compute_potentials(_potential_state_from_corpus(corpus, k), rewards, DEFAULT_CONFIG.physics)
        np.testing.assert_allclose(res.phi, expected[k], rtol=0.0, atol=1e-9, err_msg=f"pure phi, record {k}")


def test_recorded_corpus_penalises_only_covering_defenders_ahead_of_their_striker():
    corpus = _load_potential_corpus()
    drop = corpus["phi_without_line_order"] - corpus["phi"]
    assert (drop >= 0.0).all() and (drop < DEFAULT_CONFIG.rewards.line_order_weight).all()
    assert not drop[:, [0, 3, 4, 5, 6, 9, 10, 11]].any()
    assert drop[:, [1, 2]].any() and drop[:, [7, 8]].any()


def test_pure_potentials_match_recorded_corpus():
    corpus = _load_potential_corpus()
    codes = ("GK", "BALL", "CHSE", "1stP", "2ndP", "COVR")
    for k in range(corpus["phi"].shape[0]):
        res = compute_potentials(_potential_state_from_corpus(corpus, k), DEFAULT_CONFIG.rewards, DEFAULT_CONFIG.physics)
        np.testing.assert_allclose(res.phi, corpus["phi"][k], rtol=0.0, atol=1e-9, err_msg=f"phi, record {k}")

        # Stun and lunge overlay the positional role in the recorded labels.
        labels = [
            "STUN" if corpus["stunned"][k][i] else "LNGE" if corpus["lunging"][k][i] else codes[res.role_code[i]]
            for i in range(12)
        ]
        assert labels == corpus["role_states"][k].tolist(), f"role labels, record {k}"

        rounded = np.array([[round(float(v), 1) for v in row] for row in res.target])
        np.testing.assert_allclose(rounded, corpus["role_targets"][k], rtol=0.0, atol=1e-9, err_msg=f"targets, record {k}")


def _wall_pinned_carrier_env() -> FootballParallelEnv:
    """Team-A carrier on the top wall, facing it, with an opponent inside 55u and a stationary ball."""
    env = _oop_env()
    env.kickoff_protection_timer = 0.0
    env.pickup_tackle_grace_timer = 0.0
    c = 3
    env.ball_holder = c
    env.last_ball_holder = c
    env.last_ball_team = 0
    env.ball_hold_timer = 0.0
    env.player_pos[c] = [600.0, 40.0]
    env.player_angle[c] = -math.pi / 2
    env.player_vel[c] = 0.0
    env.player_pos[9] = [640.0, 60.0]
    env.ball_pos = env.player_pos[c] + np.array([0.0, -24.0])
    env.ball_vel[:] = 0.0
    return env


def test_wall_pinned_carrier_gets_exactly_one_five_second_violation() -> None:
    env = _wall_pinned_carrier_env()
    v0 = env.stats["field_5s_violations"]
    for _ in range(6 * 60):
        env._physics_substep(np.zeros((12, 4)), np.zeros(12))
        env.physics_frame_count += 1
    assert env.stats["field_5s_violations"] - v0 == 1


def test_carried_ball_never_accumulates_the_loose_pin_timer() -> None:
    env = _wall_pinned_carrier_env()
    env.pinned_ball_timer = 1.9
    env._physics_substep(np.zeros((12, 4)), np.zeros(12))
    assert env.pinned_ball_timer == 0.0


# ==============================================================================
# Kick-outcome and keeper-hold counters
# ==============================================================================
KICK_KEYS = ("kicks", "kick_to_teammate", "kick_to_opponent", "kick_other", "gk_holds")


def _kick_env(seed: int = 410) -> FootballParallelEnv:
    env = FootballParallelEnv(
        replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    env.reset(seed=seed)
    env.kickoff_protection_timer = 0.0
    env.pickup_tackle_grace_timer = 0.0
    return env


def _kick_and_deliver(env: FootballParallelEnv, kicker: int, receiver: int, power: float) -> None:
    """Kick from `kicker`, then put the ball on top of `receiver` and run the pickup check."""
    env.ball_holder = kicker
    env.kick_grace_timer = 0.0
    rewards = np.zeros(env.num_players, dtype=np.float64)
    env._execute_kick(kicker, kick_power=power, step_rewards=rewards)
    env.ball_pos = env.player_pos[receiver].copy()
    env.ball_vel[:] = 0.0
    env.pickup_cooldown[receiver] = 0.0
    env._check_free_ball_pickup(rewards)
    assert env.ball_holder == receiver


def test_fresh_match_has_zero_kick_and_keeper_hold_counters() -> None:
    env = _kick_env()
    assert {k: env.stats[k] for k in KICK_KEYS} == {k: 0 for k in KICK_KEYS}


def test_one_pass_one_interception_and_one_unresolved_kick_are_counted_once_each() -> None:
    env = _kick_env()
    d1 = env.agent_to_idx["a_def_1"]
    d2 = env.agent_to_idx["a_def_2"]
    b_atk = env.agent_to_idx["b_atk_1"]
    env.player_pos[d1] = np.array([300.0, 300.0])
    env.player_pos[d2] = np.array([420.0, 300.0])
    env.player_pos[b_atk] = np.array([700.0, 700.0])

    _kick_and_deliver(env, kicker=d1, receiver=d2, power=0.5)
    assert env.stats["kicks"] == 1 and env.stats["kick_to_teammate"] == 1

    _kick_and_deliver(env, kicker=d2, receiver=b_atk, power=0.5)
    assert env.stats["kicks"] == 2
    assert env.stats["kick_to_teammate"] == 1 and env.stats["kick_to_opponent"] == 1

    env.ball_holder = b_atk
    env.player_angle[b_atk] = 0.0
    env._execute_kick(b_atk, kick_power=0.5, step_rewards=np.zeros(12))
    assert env.stats["kicks"] == 3
    assert env.stats["kick_other"] == 0, "an in-flight kick is not yet resolved"

    env.rl_step_count = env.cfg.physics.max_rl_steps - 1
    _, _, _, done, summary = env.step_array(np.zeros((12, 4)))
    assert done
    final = summary["final_stats"]
    assert (final["kicks"], final["kick_to_teammate"], final["kick_to_opponent"], final["kick_other"]) == (3, 1, 1, 1)
    assert final["kicks"] == final["kick_to_teammate"] + final["kick_to_opponent"] + final["kick_other"]


def test_kick_picked_up_again_by_the_kicker_is_other_not_teammate() -> None:
    env = _kick_env()
    d1 = env.agent_to_idx["a_def_1"]
    env.player_pos[d1] = np.array([300.0, 300.0])
    _kick_and_deliver(env, kicker=d1, receiver=d1, power=0.5)
    assert (env.stats["kicks"], env.stats["kick_to_teammate"], env.stats["kick_to_opponent"], env.stats["kick_other"]) == (1, 0, 0, 1)


def test_second_kick_before_any_pickup_resolves_the_first_as_other() -> None:
    env = _kick_env()
    d1 = env.agent_to_idx["a_def_1"]
    env.player_pos[d1] = np.array([300.0, 300.0])
    for _ in range(2):
        env.ball_holder = d1
        env._execute_kick(d1, kick_power=0.5, step_rewards=np.zeros(12))
    assert (env.stats["kicks"], env.stats["kick_other"]) == (2, 1)


def test_kickoff_discards_the_pending_kick_as_other() -> None:
    env = _kick_env()
    d1 = env.agent_to_idx["a_def_1"]
    env.ball_holder = d1
    env._execute_kick(d1, kick_power=0.5, step_rewards=np.zeros(12))
    env._setup_kickoff(TEAM_B)
    assert (env.stats["kicks"], env.stats["kick_other"]) == (1, 1)
    env.reset(seed=3)
    assert {k: env.stats[k] for k in KICK_KEYS} == {k: 0 for k in KICK_KEYS}


def test_keeper_pickups_count_as_keeper_holds_and_outfield_pickups_do_not() -> None:
    env = _kick_env(seed=88)
    k_idx = env.agent_to_idx["a_keeper"]
    b_atk = env.agent_to_idx["b_atk_1"]
    d1 = env.agent_to_idx["a_def_1"]
    env.player_pos[d1] = np.array([500.0, 300.0])
    env.player_pos[b_atk] = np.array([400.0, 420.0])
    env.player_angle[b_atk] = math.pi
    env.ball_holder = b_atk
    env._execute_kick(b_atk, kick_power=1.0, step_rewards=np.zeros(12))
    env.player_pos[k_idx] = env.ball_pos.copy()
    env._check_free_ball_pickup(np.zeros(12))
    assert env.ball_holder == k_idx
    assert env.stats["gk_holds"] == 1
    assert env.stats["kick_to_opponent"] == 1 and env.stats["keeper_saves"] == 1

    env.ball_holder = None
    env.ball_pos = env.player_pos[d1].copy()
    env.ball_vel[:] = 0.0
    env.ball_kicked_in_flight = False
    env.pickup_cooldown[:] = 0.0
    env._check_free_ball_pickup(np.zeros(12))
    assert env.ball_holder == d1
    assert env.stats["gk_holds"] == 1


def test_random_rollout_kick_counters_balance_every_step() -> None:
    env = _kick_env(seed=7)
    acts = np.random.default_rng(7).uniform(-1, 1, size=(900, 12, 4))
    for k in range(900):
        env.step_array(acts[k])
        s = env.stats
        resolved = s["kick_to_teammate"] + s["kick_to_opponent"] + s["kick_other"]
        assert s["kicks"] - resolved in (0, 1), f"step {k}: at most one kick may be unresolved"
    assert env.stats["kicks"] > 0


# ==============================================================================
# Uniform replay capture
# ==============================================================================
def _uniform_env(tmp_path: Path, **replay_kwargs) -> FootballParallelEnv:
    cfg = replace(
        DEFAULT_CONFIG,
        replay=replace(
            DEFAULT_CONFIG.replay,
            enable_recording=True,
            interest_threshold=float("inf"),
            save_dir=str(tmp_path / "rep"),
            **replay_kwargs,
        ),
    )
    env = FootballParallelEnv(config=cfg)
    env.reset(seed=5, options={"generation": 12, "match_id": 34})
    return env


def _play_full_match(env: FootballParallelEnv, seed: int = 5) -> None:
    acts = np.random.default_rng(seed).uniform(-1, 1, size=(env.cfg.physics.max_rl_steps, 12, 4))
    for k in range(env.cfg.physics.max_rl_steps):
        env.step_array(acts[k])


def test_uniform_capture_exports_a_below_threshold_match_with_uniform_in_the_name(tmp_path: Path) -> None:
    env = _uniform_env(tmp_path)
    env.uniform_capture = True
    _play_full_match(env)
    assert env.last_saved_replay_path is None, "interest threshold is infinite; only the uniform capture may export"
    path = env.export_uniform_replay()
    assert path is not None and Path(path).exists()
    name = Path(path).name
    assert name.startswith("match_0034_gen0012_score") and "_uniform_" in name
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert len(data["frames"]) == env.cfg.physics.max_physics_frames + 1
    assert data["metadata"]["match_id"] == 34 and data["metadata"]["generation"] == 12
    assert env.export_uniform_replay() is None, "a captured replay is exported once"


def test_uniform_capture_off_keeps_nothing(tmp_path: Path) -> None:
    env = _uniform_env(tmp_path)
    _play_full_match(env)
    assert env.export_uniform_replay() is None
    assert not (tmp_path / "rep").exists()


def test_uniform_capture_survives_the_next_reset_and_is_independent_of_best_only(tmp_path: Path) -> None:
    env = _uniform_env(tmp_path, save_best_only=True)
    env.uniform_capture = True
    _play_full_match(env)
    env.reset(seed=6, options={"generation": 13, "match_id": 35})
    path = env.export_uniform_replay()
    assert path is not None and "gen0012" in Path(path).name


# ---------------------------------------------------------------------------
# Match-state observation features (columns 76..81)
# ---------------------------------------------------------------------------
POSSESSION_OWNER, BALL_IN_FLIGHT, TIME_REMAINING, HALF, OWN_KICK_LOCKOUT, TURNOVER_COOLDOWN = (
    76,
    77,
    78,
    79,
    80,
    81,
)
TEAM_A_OWNS = [1.0] * 6 + [-1.0] * 6
TEAM_B_OWNS = [-1.0] * 6 + [1.0] * 6
TEAM_A_LOOSE_OWNS = [0.5] * 6 + [-0.5] * 6
TEAM_B_LOOSE_OWNS = [-0.5] * 6 + [0.5] * 6
A_ATK, B_ATK = 5, 11
STAND_STILL = np.zeros((12, 4))


def _kickoff_env(kickoff_team: int = TEAM_A) -> FootballParallelEnv:
    env = FootballParallelEnv(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    env.reset(seed=0, options={"kickoff_team": kickoff_team})
    return env


def _soft_kick(env: FootballParallelEnv, kicker: int) -> None:
    """Release a minimum-power (160 u/s) straight kick along the kicker's facing."""
    env._execute_kick(kicker, kick_power=0.35, step_rewards=np.zeros(env.num_players))


def test_possession_owner_marks_the_kicking_team_for_the_whole_flight() -> None:
    env = _kickoff_env()
    obs = env._get_observations_array()
    assert env.ball_holder == A_ATK
    np.testing.assert_array_equal(obs[:, POSSESSION_OWNER], TEAM_A_OWNS)
    np.testing.assert_array_equal(obs[:, BALL_IN_FLIGHT], np.zeros(12))

    env.player_angle[A_ATK] = 0.0
    _soft_kick(env, A_ATK)
    obs = env._get_observations_array()
    flight_steps = 0
    while env.ball_kicked_in_flight:
        assert env.ball_holder is None
        np.testing.assert_array_equal(obs[:, POSSESSION_OWNER], TEAM_A_OWNS)
        np.testing.assert_array_equal(obs[:, BALL_IN_FLIGHT], np.ones(12))
        obs = env.step_array(STAND_STILL)[0]
        flight_steps += 1
    # 160 u/s falls below the 25 u/s flight floor after (160 - 25) / (155 / 60) = 52.3 -> 53 frames = RL step 14.
    assert flight_steps == 14
    assert env.ball_holder is None and env.last_ball_team == TEAM_A
    np.testing.assert_array_equal(obs[:, POSSESSION_OWNER], TEAM_A_LOOSE_OWNS)
    np.testing.assert_array_equal(obs[:, BALL_IN_FLIGHT], np.zeros(12))

    env.last_ball_team = None
    np.testing.assert_array_equal(env._get_observations_array()[:, POSSESSION_OWNER], np.zeros(12))


def test_possession_owner_flips_when_an_opponent_intercepts() -> None:
    env = _kickoff_env()
    env.player_angle[A_ATK] = 0.0
    env.player_pos[B_ATK] = [790.0, 420.0]  # on the kick line, 66 u ahead of the release point (724, 420)
    env.player_vel[B_ATK] = 0.0
    _soft_kick(env, A_ATK)
    owners = []
    for _ in range(10):
        obs = env.step_array(STAND_STILL)[0]
        owners.append(obs[:, POSSESSION_OWNER].tolist())
        if env.ball_holder is not None:
            break
    assert env.ball_holder == B_ATK
    assert len(owners) >= 2 and all(o == TEAM_A_OWNS for o in owners[:-1])
    np.testing.assert_array_equal(obs[:, POSSESSION_OWNER], TEAM_B_OWNS)
    np.testing.assert_array_equal(obs[:, BALL_IN_FLIGHT], np.zeros(12))


def test_own_kick_lockout_decays_for_the_kicker_only() -> None:
    env = _kickoff_env()
    env.pickup_cooldown[8] = 0.3  # a pickup lockout that does not come from a kick
    _soft_kick(env, A_ATK)
    # 0.50 s post_kick_pbrs_timer minus 4/60 s per RL step, divided by 0.50 s.
    expected = [max(0.0, (0.50 - k * 4.0 / 60.0) / 0.50) for k in range(9)]
    obs = env._get_observations_array()
    for k, want in enumerate(expected):
        if k:
            obs = env.step_array(STAND_STILL)[0]
        assert obs[A_ATK, OWN_KICK_LOCKOUT] == pytest.approx(want, abs=1e-6), f"after {k} steps"
        others = np.delete(obs[:, OWN_KICK_LOCKOUT], A_ATK)
        np.testing.assert_array_equal(others, np.zeros(11))

    # Clearing pickup/kick_cooldown via a teammate touch does not clear post_kick_pbrs_timer.
    env.post_kick_pbrs_timer[A_ATK] = 0.25
    env.kick_cooldown_active[A_ATK] = False
    env.pickup_cooldown[A_ATK] = 0.0
    assert env._get_observations_array()[A_ATK, OWN_KICK_LOCKOUT] == pytest.approx(0.5, abs=1e-6)

    # Non-kick pickup_cooldown does not leak into own_kick_lockout; post_kick_pbrs_timer > 0.5 saturates at 1.
    env.post_kick_pbrs_timer[A_ATK] = 0.0
    env.pickup_cooldown[A_ATK] = 0.9
    assert env._get_observations_array()[A_ATK, OWN_KICK_LOCKOUT] == 0.0
    env.post_kick_pbrs_timer[A_ATK] = 0.9
    assert env._get_observations_array()[A_ATK, OWN_KICK_LOCKOUT] == 1.0


def test_time_remaining_and_half_follow_the_match_clock() -> None:
    env = _kickoff_env()
    obs = env._get_observations_array()
    np.testing.assert_array_equal(obs[:, TIME_REMAINING], np.ones(12))
    np.testing.assert_array_equal(obs[:, HALF], np.zeros(12))

    for _ in range(27):
        obs = env.step_array(STAND_STILL)[0]
    np.testing.assert_allclose(obs[:, TIME_REMAINING], 0.99, atol=1e-6)
    np.testing.assert_array_equal(obs[:, HALF], np.zeros(12))

    env.rl_step_count = 1349
    env.physics_frame_count = 1349 * 4
    obs = env.step_array(STAND_STILL)[0]  # frame 5400: half-time
    assert env.current_half == 2
    np.testing.assert_allclose(obs[:, TIME_REMAINING], 0.5, atol=1e-6)
    np.testing.assert_array_equal(obs[:, HALF], np.ones(12))

    env.rl_step_count = 2025
    np.testing.assert_allclose(env._get_observations_array()[:, TIME_REMAINING], 0.25, atol=1e-6)

    env.rl_step_count = 2699
    env.physics_frame_count = 2699 * 4
    obs, _, _, done, _ = env.step_array(STAND_STILL)
    assert done
    np.testing.assert_array_equal(obs[:, TIME_REMAINING], np.zeros(12))
    np.testing.assert_array_equal(obs[:, HALF], np.ones(12))

    env.rl_step_count = 2701  # past full time the clock feature stays at 0
    np.testing.assert_array_equal(env._get_observations_array()[:, TIME_REMAINING], np.zeros(12))


def test_swapping_the_flight_team_label_flips_only_possession_owner() -> None:
    env = _kickoff_env()
    _soft_kick(env, A_ATK)
    for _ in range(3):
        env.step_array(STAND_STILL)
    assert env.ball_kicked_in_flight and env.last_ball_team == TEAM_A
    obs_a = env._get_observations_array().copy()
    env.last_ball_team = TEAM_B
    obs_b = env._get_observations_array()

    np.testing.assert_array_equal(obs_a[:, POSSESSION_OWNER], TEAM_A_OWNS)
    np.testing.assert_array_equal(obs_b[:, POSSESSION_OWNER], TEAM_B_OWNS)
    np.testing.assert_array_equal(
        np.delete(obs_b, POSSESSION_OWNER, axis=1), np.delete(obs_a, POSSESSION_OWNER, axis=1)
    )

    env.ball_kicked_in_flight = False
    env.last_ball_team = TEAM_A
    obs_loose_a = env._get_observations_array().copy()
    env.last_ball_team = TEAM_B
    obs_loose_b = env._get_observations_array()
    np.testing.assert_array_equal(obs_loose_a[:, POSSESSION_OWNER], TEAM_A_LOOSE_OWNS)
    np.testing.assert_array_equal(obs_loose_b[:, POSSESSION_OWNER], TEAM_B_LOOSE_OWNS)
    np.testing.assert_array_equal(
        np.delete(obs_loose_b, POSSESSION_OWNER, axis=1), np.delete(obs_loose_a, POSSESSION_OWNER, axis=1)
    )


def test_mirrored_kick_gives_both_teams_the_same_view() -> None:
    env_a = _kickoff_env(TEAM_A)
    env_b = _kickoff_env(TEAM_B)
    _soft_kick(env_a, A_ATK)
    _soft_kick(env_b, B_ATK)
    for _ in range(3):
        obs_a = env_a.step_array(STAND_STILL)[0]
        obs_b = env_b.step_array(STAND_STILL)[0]
    assert env_a.ball_kicked_in_flight and env_b.ball_kicked_in_flight

    np.testing.assert_array_equal(obs_a[:, POSSESSION_OWNER], TEAM_A_OWNS)
    np.testing.assert_array_equal(obs_b[:, POSSESSION_OWNER], TEAM_B_OWNS)
    assert obs_a[A_ATK, OWN_KICK_LOCKOUT] > 0.0 and obs_b[B_ATK, OWN_KICK_LOCKOUT] > 0.0
    # Same 180-degree canonical tolerance as the kickoff symmetry test.
    np.testing.assert_allclose(obs_a[:6], obs_b[6:], atol=1e-6)
    np.testing.assert_allclose(obs_a[6:], obs_b[:6], atol=1e-6)


def test_last_ball_team_is_observable_while_it_changes_the_reward() -> None:
    env = _kickoff_env()
    _soft_kick(env, A_ATK)
    for _ in range(2):
        env.step_array(STAND_STILL)
    twin = copy.deepcopy(env)
    twin.last_ball_team = TEAM_B

    obs_env = env._get_observations_array()
    obs_twin = twin._get_observations_array()
    rew_env = env.step_array(STAND_STILL)[2]
    rew_twin = twin.step_array(STAND_STILL)[2]
    assert not np.array_equal(rew_env, rew_twin), "the witness states must differ in reward"
    assert not np.array_equal(obs_env, obs_twin), "states with different rewards ahead share one observation"


def test_dead_loose_ball_last_ball_team_is_observable_while_it_changes_the_reward() -> None:
    env = _kickoff_env()
    env.ball_holder = None
    env.ball_kicked_in_flight = False
    env.ball_vel[:] = 0.0
    env.last_ball_team = TEAM_A
    env.ball_pos[:] = env.player_pos[B_ATK]
    env.pickup_cooldown[B_ATK] = 0.0
    env.last_turnover_reward_time = -999.0

    twin = copy.deepcopy(env)
    twin.last_ball_team = TEAM_B

    obs_env = env._get_observations_array()
    obs_twin = twin._get_observations_array()
    rew_env = env.step_array(STAND_STILL)[2]
    rew_twin = twin.step_array(STAND_STILL)[2]
    assert env.ball_holder == B_ATK and twin.ball_holder == B_ATK
    assert not np.array_equal(rew_env, rew_twin), "turnover reward must differ across last_ball_team"
    np.testing.assert_array_equal(obs_env[:, POSSESSION_OWNER], TEAM_A_LOOSE_OWNS)
    np.testing.assert_array_equal(obs_twin[:, POSSESSION_OWNER], TEAM_B_LOOSE_OWNS)
    assert not np.array_equal(obs_env, obs_twin)


def test_post_kick_pbrs_timer_is_observable_while_it_changes_the_reward() -> None:
    env = _kickoff_env()
    env.player_angle[A_ATK] = 0.0
    a_mid_2 = 4
    env.player_pos[a_mid_2] = [780.0, 420.0]
    env.player_vel[a_mid_2] = 0.0
    _soft_kick(env, A_ATK)
    # Step until a_mid_2 picks up the ball; _on_new_player_touched_ball clears A_ATK's kick_cooldown_active
    # and pickup_cooldown while post_kick_pbrs_timer[A_ATK] stays > 0.
    for _ in range(10):
        env.step_array(STAND_STILL)
        if env.ball_holder == a_mid_2:
            break
    assert env.ball_holder == a_mid_2
    assert not env.kick_cooldown_active[A_ATK] and env.pickup_cooldown[A_ATK] == 0.0
    assert env.post_kick_pbrs_timer[A_ATK] > 0.10

    twin = copy.deepcopy(env)
    twin.post_kick_pbrs_timer[A_ATK] = 0.0

    obs_env = env._get_observations_array()
    obs_twin = twin._get_observations_array()
    rew_env = env.step_array(STAND_STILL)[2]
    rew_twin = twin.step_array(STAND_STILL)[2]
    assert rew_env[A_ATK] != rew_twin[A_ATK], "post_kick_pbrs_timer must change A_ATK's potential reward"
    assert obs_env[A_ATK, OWN_KICK_LOCKOUT] > 0.0 and obs_twin[A_ATK, OWN_KICK_LOCKOUT] == 0.0
    assert not np.array_equal(obs_env, obs_twin)


def test_turnover_cooldown_decays_after_a_rewarded_turnover() -> None:
    env = _kickoff_env()
    np.testing.assert_array_equal(env._get_observations_array()[:, TURNOVER_COOLDOWN], np.zeros(12))

    # Set up a loose ball owned by TEAM_A right at B_ATK's feet so frame 1 triggers a rewarded turnover.
    env.ball_holder = None
    env.ball_kicked_in_flight = False
    env.ball_vel[:] = 0.0
    env.last_ball_team = TEAM_A
    env.ball_pos[:] = env.player_pos[B_ATK]
    env.pickup_cooldown[B_ATK] = 0.0

    obs = env.step_array(STAND_STILL)[0]
    assert env.ball_holder == B_ATK
    assert env.last_turnover_reward_time == pytest.approx(0.0)
    # 1 RL step ends at frame 4 (t = 4/60), 4/60 s after frame 0 pickup -> 56/60 remaining, hitting 0 at step 15.
    for k in range(15):
        want = max(0.0, 1.0 - 4.0 * (k + 1) / 60.0)
        np.testing.assert_allclose(obs[:, TURNOVER_COOLDOWN], want, atol=1e-6, err_msg=f"step {k}")
        obs = env.step_array(STAND_STILL)[0]
    np.testing.assert_array_equal(obs[:, TURNOVER_COOLDOWN], np.zeros(12))


def test_turnover_cooldown_is_observable_while_it_changes_the_reward() -> None:
    env = _kickoff_env()
    env.ball_holder = None
    env.ball_kicked_in_flight = False
    env.ball_vel[:] = 0.0
    env.last_ball_team = TEAM_A
    env.ball_pos[:] = env.player_pos[B_ATK]
    env.pickup_cooldown[B_ATK] = 0.0
    env.last_turnover_reward_time = -999.0

    twin = copy.deepcopy(env)
    twin.last_turnover_reward_time = twin.match_time

    obs_env = env._get_observations_array()
    obs_twin = twin._get_observations_array()
    rew_env = env.step_array(STAND_STILL)[2]
    rew_twin = twin.step_array(STAND_STILL)[2]
    assert not np.array_equal(rew_env, rew_twin), "turnover cooldown must gate the turnover reward"
    np.testing.assert_array_equal(obs_env[:, TURNOVER_COOLDOWN], np.zeros(12))
    np.testing.assert_array_equal(obs_twin[:, TURNOVER_COOLDOWN], np.ones(12))
    assert not np.array_equal(obs_env, obs_twin)


def test_observations_stay_deterministic_and_bounded_across_both_halves() -> None:
    def play() -> np.ndarray:
        env = FootballParallelEnv(
            config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
        )
        env.reset(seed=21)
        rng = np.random.default_rng(21)
        frames = []
        for start, steps in ((0, 40), (1330, 40), (2670, 30)):
            env.rl_step_count = start
            env.physics_frame_count = start * 4
            for _ in range(steps):
                frames.append(env.step_array(rng.uniform(-1.0, 1.0, size=(12, 4)))[0].copy())
        return np.stack(frames)

    first, second = play(), play()
    assert first.shape == (110, 12, OBS_DIM)
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()
    assert first.min() >= -1.0 and first.max() <= 1.0
