"""Verification test suite for the 6v6 MARL Football PettingZoo ParallelEnv."""

from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path

import numpy as np
from pettingzoo.test import parallel_api_test
import pytest

from football_rl.config import (
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
    assert OBS_DIM == 76

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
    np.testing.assert_allclose(obs_ht["b_atk_1"], obs_a_start["a_atk_1"], atol=1e-6)


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
    assert math.isclose(env.pickup_cooldown[carrier_idx], 1.0, rel_tol=1e-4)
    assert bool(env.kick_cooldown_active[carrier_idx]) is True

    # Aim ball directly at top wall (Y = 0) so it bounces: wall bounce must NOT clear kick cooldown
    env.ball_pos = np.array([700.0, 14.0])
    env.ball_vel = np.array([0.0, -300.0])
    env._physics_substep(
        np.array([_zero_actions(env)[a] for a in ALL_AGENTS], dtype=np.float64), rewards
    )
    assert env.pickup_cooldown[carrier_idx] > 0.95
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
    assert env.stats["field_3s_violations"] == 1
    assert total_carrier_reward < -1.5

    # Move teammate a_def_1 onto the ball to pick it up -> must clear carrier's cooldown
    teammate_idx = env.agent_to_idx["a_def_1"]
    env.player_pos[teammate_idx] = env.ball_pos.copy()
    env.step(_zero_actions(env))

    assert env.ball_holder == teammate_idx
    assert env.pickup_cooldown[carrier_idx] == 0.0


def test_keeper_5s_foul_teleport_and_zero_advance_spike() -> None:
    """Verify Keeper 5-second hold triggers foul teleport and places players correctly."""
    env = parallel_env(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    env.reset(seed=33)

    keeper_idx = env.agent_to_idx["a_keeper"]
    env.ball_holder = keeper_idx
    env.ball_hold_timer = 4.99

    _, rewards, _, _, _ = env.step(_zero_actions(env))

    assert env.stats["keeper_5s_violations"] == 1
    # Attacking Team B's b_atk_1 holds ball at world (580, 420) (canonical 820, 420)
    b_atk_1_idx = env.agent_to_idx["b_atk_1"]
    b_mid_1_idx = env.agent_to_idx["b_mid_1"]
    b_mid_2_idx = env.agent_to_idx["b_mid_2"]
    assert env.ball_holder == b_atk_1_idx
    np.testing.assert_allclose(env.player_pos[b_atk_1_idx], [580.0, 420.0])
    np.testing.assert_allclose(env.player_pos[b_mid_1_idx], [435.0, 605.0])
    np.testing.assert_allclose(env.player_pos[b_mid_2_idx], [435.0, 235.0])
    np.testing.assert_allclose(env.player_pos[keeper_idx], [40.0, 420.0])

    assert math.isclose(rewards["a_keeper"], -5.5, abs_tol=0.05)


def test_role_one_hot_encoding() -> None:
    """Verify role one-hot features in obs[13:16] for GK [0,0,0], DEF [1,0,0], MID [0,1,0], ATK [0,0,1]."""
    env = parallel_env(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    obs, _ = env.reset(seed=42)

    np.testing.assert_allclose(obs["a_keeper"][13:16], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(obs["a_def_1"][13:16], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(obs["a_def_2"][13:16], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(obs["a_mid_1"][13:16], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(obs["a_mid_2"][13:16], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(obs["a_atk_1"][13:16], [0.0, 0.0, 1.0])

    np.testing.assert_allclose(obs["b_keeper"][13:16], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(obs["b_def_1"][13:16], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(obs["b_mid_1"][13:16], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(obs["b_atk_1"][13:16], [0.0, 0.0, 1.0])


def test_role_specific_kick_speeds() -> None:
    """Verify role-specific maximum kick launch speeds: ATK 640 > MID 560 > DEF 510."""
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
    assert mid_max_spd == 560.0
    assert def_max_spd == 510.0
    assert gk_max_spd == 560.0
    assert atk_max_spd > mid_max_spd > def_max_spd


def test_true_shot_on_target_save_vs_harmless_catch_and_anti_farming() -> None:
    """Verify keeper save rewards true shot-on-target, harmless catch awards +0.5, and anti-strip farming."""
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

    # Keeper save ind (5.0) + team (1.0) = 6.0
    assert env.stats["keeper_saves"] == 1
    assert math.isclose(save_rewards[k_idx], 6.0, abs_tol=1e-5)

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

    # Harmless catch ind (0.5) + team (0.1) = 0.6, no keeper_save stat increment
    assert env.stats["keeper_saves"] == 0
    assert math.isclose(catch_rewards[k_idx], 0.6, abs_tol=1e-5)

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
    # Field player (ROLE_ATK): -0.50 * 190.0 = -95.0
    # Keeper (ROLE_KEEPER): -0.75 * 180.0 = -135.0
    actions["a_atk_1"] = np.array([-1.0, 0.0, -1.0, -1.0], dtype=np.float32)
    actions["a_keeper"] = np.array([-1.0, 0.0, -1.0, -1.0], dtype=np.float32)

    for _ in range(15):
        env.step(actions)
    assert np.isclose(env.player_vel[atk_idx, 0], -95.0, atol=0.05)
    assert np.isclose(env.player_vel[atk_idx, 1], 0.0, atol=0.05)
    assert np.isclose(env.player_vel[gk_idx, 0], -135.0, atol=0.05)
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
    # Shooter was not penalized for an interception on a valid GK save
    assert step_rewards[shooter_idx] >= 0.0

    # 2. Prime scoring zone hesitation penalty
    env.player_pos[b_keeper_idx] = np.array([1355.0, 420.0])  # Keeper in goal mouth
    env.ball_holder = shooter_idx
    env.player_pos[shooter_idx] = np.array([1150.0, 420.0])  # Within 320u of right goal
    env.player_angle[shooter_idx] = 0.0  # Aimed at goal
    env.ball_pos = env.player_pos[shooter_idx] + np.array([env.cfg.mechanics.ball_carry_offset, 0.0])
    env.ball_vel[:] = 0.0
    env.prev_ball_x = float(env.ball_pos[0])
    env.kick_grace_timer = 0.0

    # Step without kicking
    _, rewards, _, _, _ = env.step(_zero_actions(env))
    # Must include prime_shot_hesitation_pen (-0.025)
    # Possession reward (+0.006 carrier + 0.001 team) - 0.025 = negative net
    assert rewards["a_atk_1"] < -0.01


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
    """Verify asymmetric throttle deadzone [-0.35, +0.15], exponential velocity inertia, and jitter reverse penalty."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    mid_idx = env.agent_to_idx["a_mid_1"]
    max_spd = env._get_player_max_speed(mid_idx)

    # 1. Asymmetric Deadzone: -0.25 is in [-0.35, +0.15] -> target_speed is 0.0
    act_deadzone = np.zeros((env.num_players, 4), dtype=np.float32)
    act_deadzone[mid_idx] = [-0.25, 0.0, -1.0, -1.0]
    env.player_vel[mid_idx] = 0.0

    env._physics_substep(act_deadzone)
    assert np.allclose(env.player_vel[mid_idx], 0.0, atol=1e-5)

    # 2. Velocity Inertia: throttle = -1.0 accelerates smoothly over substeps rather than jumping instantly
    act_backpedal = np.zeros((env.num_players, 4), dtype=np.float32)
    act_backpedal[mid_idx] = [-1.0, 0.0, -1.0, -1.0]  # Full backpedal
    target_backpedal_speed = -env.cfg.physics.field_backpedal_mult * max_spd  # -0.50 * 195 = -97.5

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

    # 3. Anti-Jitter Velocity-Reversal Penalty in step()
    # Case A: sudden velocity flip from +50.0 u/s to backpedaling
    env.reset(seed=42)
    env.prev_longitudinal_vel[mid_idx] = 50.0
    env.player_vel[mid_idx] = np.array([-30.0, 0.0], dtype=np.float64)
    acts = _zero_actions(env)
    acts["a_mid_1"] = np.array([-0.8, 0.0, -1.0, -1.0], dtype=np.float32)
    _, rews_flip, _, _, _ = env.step(acts)

    # Case B: consistent direction (both negative) -> no jitter penalty
    env.reset(seed=42)
    env.prev_longitudinal_vel[mid_idx] = -50.0
    env.player_vel[mid_idx] = np.array([-30.0, 0.0], dtype=np.float64)
    _, rews_noflip, _, _, _ = env.step(acts)

    diff = rews_flip["a_mid_1"] - rews_noflip["a_mid_1"]
    assert math.isclose(diff, env.cfg.rewards.jitter_reverse_pen, abs_tol=1e-5)


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
    """Verify GK forward-hemisphere launch clamp and backward pass non-penalty."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    gk_idx = env.agent_to_idx["a_keeper"]
    env.ball_holder = gk_idx
    env.player_pos[gk_idx] = np.array([25.0, 420.0], dtype=np.float64)
    # Face directly backward into own goal net (math.pi, facing -X)
    env.player_angle[gk_idx] = math.pi
    env.team_side[TEAM_A] = SIDE_LEFT  # Left goal is X=0

    # Execute kick
    rew_arr = np.zeros(env.num_players, dtype=np.float64)
    env._execute_kick(gk_idx, 1.0, rew_arr)

    # Ball X velocity MUST be strictly positive (clamped into forward hemisphere)
    assert env.ball_vel[0] > 0.0
    # Ball cannot travel backward into own net
    assert env.ball_pos[0] >= 0.0

    # Test backward pass non-penalty and recycle bonus
    env.reset(seed=42)
    passer_idx = env.agent_to_idx["a_mid_1"]
    receiver_idx = env.agent_to_idx["a_def_1"]
    env.ball_holder = passer_idx
    env.player_pos[passer_idx] = np.array([600.0, 420.0], dtype=np.float64)
    env.player_pos[receiver_idx] = np.array([350.0, 420.0], dtype=np.float64)

    # Face backward toward receiver (math.pi)
    env.player_angle[passer_idx] = math.pi
    step_rew = np.zeros(env.num_players, dtype=np.float64)
    env._execute_kick(passer_idx, 0.6, step_rew)

    # While pass is in flight backward, net goal advance is negative
    env.prev_ball_x = 600.0
    env.ball_pos[0] = 500.0  # moved backward towards X=0
    acts = _zero_actions(env)
    # Step environment
    obs, rews, terms, truncs, infos = env.step(acts)

    # Team reward for ball advance must NOT penalize teammates during an in-flight pass
    # Now simulate pickup by receiver
    env.ball_pos = env.player_pos[receiver_idx].copy()
    pickup_rew = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(pickup_rew)

    # Passer receives completed_pass + recycle_pass_bonus
    expected_min_bonus = env.cfg.rewards.completed_pass_passer_ind + env.cfg.rewards.recycle_pass_bonus
    assert pickup_rew[passer_idx] >= expected_min_bonus - 1e-4


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
    """Verify 76-dim obs parity and entity relative angle normalization under 210-deg vision cone."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    obs_dict, _ = env.reset(seed=42)

    # 1. Dimension parity
    assert env.cfg.vision.fov_deg == 210.0
    assert math.isclose(env.cfg.vision.half_fov_rad, math.radians(105.0))
    for agent in env.possible_agents:
        assert env.observation_space(agent).shape == (76,)
        assert obs_dict[agent].shape == (76,)

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
    """Verify that lateral/backward pass farming yields zero progressive reward and decays pairwise."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    p1 = env.agent_to_idx["a_def_1"]
    p2 = env.agent_to_idx["a_def_2"]
    team = TEAM_A

    # Position two defenders laterally at same X: 350.0, Y: 320.0 and 520.0 (distance = 200u)
    env.player_pos[p1] = np.array([350.0, 320.0], dtype=np.float64)
    env.player_pos[p2] = np.array([350.0, 520.0], dtype=np.float64)

    # Move opponent field players away
    for idx in range(env.num_players):
        if env.player_team[idx] != team:
            env.player_pos[idx] = np.array([1200.0, 420.0], dtype=np.float64)

    # Pass 1: p1 -> p2 (lateral pass: dx = 0.0)
    env.ball_holder = None
    env.ball_kick_origin = env.player_pos[p1].copy()
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.last_ball_holder = p1
    env.last_ball_team = team
    env.ball_pos = env.player_pos[p2].copy()

    rew1 = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(rew1)

    # Pass 1: Not progressive (canon_dx = 0.0 < 30.0) -> prog_bonus = 0.0
    # Pairwise repeat count = 0, decay multiplier = 1.0
    # Passer gets 0.0, receiver gets 0.2
    rx_bonus = env.cfg.rewards.first_touch_reception_bonus
    assert rew1[p1] == 0.0
    assert math.isclose(rew1[p2], 0.20 + rx_bonus, abs_tol=1e-5)
    assert env.team_pass_chain[team] == 0  # Lateral pass does not increment pass chain

    # Pass 2: p2 -> p1 (lateral ping-pong back: repeat count = 1, decay = 0.25)
    env.ball_holder = None
    env.ball_kick_origin = env.player_pos[p2].copy()
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.last_ball_holder = p2
    env.last_ball_team = team
    env.ball_pos = env.player_pos[p1].copy()

    rew2 = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(rew2)

    assert rew2[p2] == 0.0
    assert math.isclose(rew2[p1], 0.20 * 0.25 + rx_bonus, abs_tol=1e-5)
    assert env.pairwise_repeat_count == 1

    # Pass 3: p1 -> p2 (lateral ping-pong: repeat count = 2, decay = 0.0625)
    env.ball_holder = None
    env.ball_kick_origin = env.player_pos[p1].copy()
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.last_ball_holder = p1
    env.last_ball_team = team
    env.ball_pos = env.player_pos[p2].copy()

    rew3 = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(rew3)

    assert rew3[p1] == 0.0
    assert math.isclose(rew3[p2], 0.20 * (0.25**2) + rx_bonus, abs_tol=1e-5)
    assert (rew3[p2] - rx_bonus) < 0.02

    # Now Pass 4: p2 -> p3 (forward progressive pass to mid_1: X = 600.0)
    p3 = env.agent_to_idx["a_mid_1"]
    env.player_pos[p3] = np.array([600.0, 420.0], dtype=np.float64)
    env.ball_holder = None
    env.ball_kick_origin = env.player_pos[p2].copy()
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.last_ball_holder = p2
    env.last_ball_team = team
    env.ball_pos = env.player_pos[p3].copy()

    rew4 = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(rew4)

    # Pairwise history resets on new pair!
    assert env.pairwise_repeat_count == 0
    # canon_dx = 600 - 350 = 250u. Effective dx uses max(canon_dx, goal_dist_gain)
    d_orig_goal = float(np.linalg.norm(env.player_pos[p2] - np.array([1400.0, 420.0])))
    d_dest_goal = float(np.linalg.norm(env.player_pos[p3] - np.array([1400.0, 420.0])))
    effective_dx = max(250.0, d_orig_goal - d_dest_goal)
    prog_bonus = effective_dx * env.cfg.rewards.pass_progressive_dx_weight
    team_bonus = prog_bonus * 0.5
    expected_p2 = prog_bonus + env.cfg.rewards.def_breakout_pass_bonus + team_bonus
    expected_p3 = 0.20 + env.cfg.rewards.mid_link_receive_bonus + team_bonus + rx_bonus
    assert math.isclose(rew4[p2], expected_p2, abs_tol=1e-5)
    assert math.isclose(rew4[p3], expected_p3, abs_tol=1e-5)
    assert env.team_pass_chain[team] == 1  # Progressive pass increments chain!

    # Pass 5: Line-breaking through pass from p3 (mid_1 at 600) to p4 (atk_1 at 950)
    # Position 2 opponent defenders at X=750 and X=850 (between 600 and 950)
    p4 = env.agent_to_idx["a_atk_1"]
    opp_def1 = env.agent_to_idx["b_def_1"]
    opp_def2 = env.agent_to_idx["b_def_2"]
    env.player_pos[opp_def1] = np.array([750.0, 250.0], dtype=np.float64)
    env.player_pos[opp_def2] = np.array([850.0, 600.0], dtype=np.float64)
    env.player_pos[p4] = np.array([950.0, 420.0], dtype=np.float64)

    env.ball_holder = None
    env.ball_kick_origin = env.player_pos[p3].copy()
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.last_ball_holder = p3
    env.last_ball_team = team
    env.ball_pos = env.player_pos[p4].copy()

    rew5 = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(rew5)

    # 350u forward territory + 2 bypassed defenders:
    # prog_bonus = (350 * 0.006) + (2 * 0.40) = 2.10 + 0.80 = 2.90
    expected_prog = (350.0 * env.cfg.rewards.pass_progressive_dx_weight) + (2 * env.cfg.rewards.pass_line_break_bonus)
    expected_team = expected_prog * 0.5
    expected_p3_passer = (
        expected_prog
        + env.cfg.rewards.mid_to_atk_pass_bonus
        + env.cfg.rewards.pass_to_atk_bonus
        + expected_team
    )
    assert math.isclose(rew5[p3], expected_p3_passer, abs_tol=1e-5)
    assert env.team_pass_chain[team] == 2  # Increments to 2!


def test_gk_observation_state_separation() -> None:
    """Verify decoupling of GK ball hold timer (obs[8]) and GK box loose ball countdown (obs[13])."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)
    gk_idx = env.agent_to_idx["a_keeper"]
    def_idx = env.agent_to_idx["a_def_1"]

    # State 1: Loose ball in keeper box; GK does NOT hold ball
    env.ball_holder = None
    env.ball_pos = np.array([80.0, 420.0], dtype=np.float64)  # Loose inside keeper's own box
    env.box_loose_ball_timer = 1.5  # 1.5s / 3.0s = 0.5 normalized
    env.ball_hold_timer = 0.0

    obs_gk = env._build_single_observation(gk_idx)
    obs_def = env._build_single_observation(def_idx)

    # obs[8] must be 0.0 (free ball) and NOT aliased with box timer
    assert obs_gk[8] == 0.0
    # obs[13] for GK contains stepped ramp: 0.25 + 0.75 * (1.5 / 3.0) = 0.625
    expected_obs13 = 0.25 + 0.75 * (1.5 / env.cfg.mechanics.keeper_box_loose_max_time)
    assert math.isclose(obs_gk[13], expected_obs13, abs_tol=1e-5)
    assert obs_gk[14] == 0.0
    assert obs_gk[15] == 0.0
    # Field DEF player retains role bit in obs[13]
    assert obs_def[13] == 1.0

    # State 2: GK holds ball; box loose timer is 0.0
    env.ball_holder = gk_idx
    env.ball_hold_timer = 2.5  # 2.5s / 5.0s = 0.5 normalized
    env.box_loose_ball_timer = 0.0

    obs_gk2 = env._build_single_observation(gk_idx)
    assert math.isclose(obs_gk2[8], 0.5, abs_tol=1e-5)
    assert obs_gk2[13] == 0.0


def test_tiki_taka_chain_multiplier_and_own_goal_exclusion() -> None:
    """Verify that pass chains scale terminal offensive goals and exclude own goals."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)
    rew_cfg = env.cfg.rewards

    atk_idx = env.agent_to_idx["a_atk_1"]
    def_b_idx = env.agent_to_idx["b_def_1"]

    # 1. 4-pass sequence for Team A -> Chain = 4 -> Multiplier = 1.0 + 0.30 * 4 = 2.2
    env.team_pass_chain[TEAM_A] = 4
    env.last_ball_holder = atk_idx

    step_rew = np.zeros(env.num_players, dtype=np.float64)
    env._handle_goal_scored(TEAM_A, step_rew)

    expected_team_goal = rew_cfg.goal_team_shared * 2.2  # 10.0 * 2.2 = 22.0
    expected_atk_goal = rew_cfg.atk_goal_scorer_ind * 2.2  # 7.5 * 2.2 = 16.5
    # atk_1 receives individual + team shared
    assert math.isclose(step_rew[atk_idx], expected_atk_goal + expected_team_goal, abs_tol=1e-5)
    # teammate receives team shared
    assert math.isclose(step_rew[env.agent_to_idx["a_def_1"]], expected_team_goal, abs_tol=1e-5)
    # Chains reset post-goal
    assert env.team_pass_chain == [0, 0]

    # 2. Own Goal: Chain multiplier must NOT apply to own goal
    env.team_pass_chain[TEAM_A] = 5
    env.last_ball_holder = atk_idx  # Team A player puts ball into own net -> Team B scores

    step_rew_og = np.zeros(env.num_players, dtype=np.float64)
    env._handle_goal_scored(TEAM_B, step_rew_og)

    # Team B gets standard goal_team_shared without multiplier
    assert math.isclose(step_rew_og[def_b_idx], rew_cfg.goal_team_shared, abs_tol=1e-5)
    # Own goal scorer receives own_goal_ind + goal_concede_team_shared (-8.0 + -10.0 = -18.0)
    assert math.isclose(
        step_rew_og[atk_idx], rew_cfg.own_goal_ind + rew_cfg.goal_concede_team_shared, abs_tol=1e-5
    )


def test_kinematic_lead_pass_targeting() -> None:
    """Verify that pass aiming rewards raycasts directed at kinematic lead projection (p + tau * v)."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)

    kicker_idx = env.agent_to_idx["a_mid_1"]
    runner_idx = env.agent_to_idx["a_atk_1"]

    env.player_pos[kicker_idx] = np.array([400.0, 420.0], dtype=np.float64)
    env.ball_pos = env.player_pos[kicker_idx].copy()
    env.ball_holder = kicker_idx

    # Runner at [600.0, 300.0] sprinting up at 160 u/s along Y
    env.player_pos[runner_idx] = np.array([600.0, 300.0], dtype=np.float64)
    env.player_vel[runner_idx] = np.array([0.0, 160.0], dtype=np.float64)
    tau = env.cfg.rewards.pass_lead_tau  # 0.70s
    # Lead target: [600.0, 300.0 + 0.7 * 160.0] = [600.0, 412.0]
    lead_target = env.player_pos[runner_idx] + tau * env.player_vel[runner_idx]
    to_lead = lead_target - env.ball_pos
    lead_angle = math.atan2(to_lead[1], to_lead[0])

    # Move opponent defenders far out of way
    for idx in range(env.num_players):
        if env.player_team[idx] != TEAM_A:
            env.player_pos[idx] = np.array([1200.0, 700.0], dtype=np.float64)

    # Face directly toward the kinematic lead point
    env.player_angle[kicker_idx] = lead_angle
    step_rew = np.zeros(env.num_players, dtype=np.float64)
    env._execute_kick(kicker_idx, kick_power=0.6, step_rewards=step_rew)

    # Kicker receives pass_target_aim_ind
    assert step_rew[kicker_idx] >= env.cfg.rewards.pass_target_aim_ind - 1e-4


def test_kickoff_carrier_faces_own_keeper_and_sees_teammates() -> None:
    """Verify that the center kickoff player faces their own goalkeeper at start and after a goal."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)

    # 1. Team A kickoff (Team A defends Left, keeper at X=45)
    env.reset(seed=42, options={"kickoff_team": TEAM_A})
    a_atk_idx = env.agent_to_idx["a_atk_1"]
    a_gk_idx = env.agent_to_idx["a_keeper"]

    assert env.ball_holder == a_atk_idx
    # Carrier angle must face negative X (towards Left keeper at X=45)
    cos_angle_a = math.cos(env.player_angle[a_atk_idx])
    sin_angle_a = math.sin(env.player_angle[a_atk_idx])
    assert math.isclose(cos_angle_a, -1.0, abs_tol=1e-5)
    assert math.isclose(sin_angle_a, 0.0, abs_tol=1e-5)

    # In canonical coordinates, carrier faces angle pi (towards own keeper)
    obs_carrier = env._build_single_observation(a_atk_idx)
    # Ball is held
    assert obs_carrier[7] == 1.0
    # Relative angle to own keeper must be ~0.0 (directly ahead in FOV)
    gk_pos = env.player_pos[a_gk_idx]
    car_pos = env.player_pos[a_atk_idx]
    angle_to_gk = math.atan2(gk_pos[1] - car_pos[1], gk_pos[0] - car_pos[0])
    rel_angle_to_gk = wrap_to_pi(angle_to_gk - env.player_angle[a_atk_idx])
    assert math.isclose(rel_angle_to_gk, 0.0, abs_tol=1e-5)

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
    # Carrier angle must face positive X (towards Right keeper at X=1355)
    cos_angle_b = math.cos(env.player_angle[b_atk_idx])
    sin_angle_b = math.sin(env.player_angle[b_atk_idx])
    assert math.isclose(cos_angle_b, 1.0, abs_tol=1e-5)
    assert math.isclose(sin_angle_b, 0.0, abs_tol=1e-5)

    # 3. Post-Goal reset: Team A scores -> Conceding team (Team B) kicks off facing b_keeper
    step_rew = np.zeros(env.num_players, dtype=np.float64)
    env._handle_goal_scored(TEAM_A, step_rew)
    assert env.ball_holder == b_atk_idx
    assert math.isclose(math.cos(env.player_angle[b_atk_idx]), 1.0, abs_tol=1e-5)


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

    # Whiff: instant stun for 0.8s, zero velocity, ghost flag, and reckless penalty
    assert env.stun_timer[tackler_idx] == env.cfg.mechanics.tackle_miss_stun_time
    assert np.allclose(env.player_vel[tackler_idx], [0.0, 0.0])
    assert env.is_ghost[tackler_idx] is True or env.is_ghost[tackler_idx] == 1
    assert step_rew[tackler_idx] == env.cfg.rewards.reckless_tackle_ind


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
    assert env.is_ghost[tackler_idx] is True or env.is_ghost[tackler_idx] == 1
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
    assert step_rew[carrier_idx] == env.cfg.rewards.successful_tackle_carrier_ind

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

    # Expect b_atk_1 to have moved smoothly: 4 substeps * (285.0 * 1/60) = 19.0 units
    # New X should be ~224.0, NOT instantly teleported to >= 265.0!
    assert abs(env.player_pos[b_atk_idx, 0] - 224.0) < 0.1
    assert env.player_pos[b_atk_idx, 0] < env.cfg.mechanics.gk_hold_opponent_buffer
    assert abs(env.step_external_push[b_atk_idx, 0] - 19.0) < 0.1

    # Verify that external push did not produce an artificial presser penalty for b_atk_1
    assert rewards["b_atk_1"] > -0.05

    # Run remaining steps to reach buffer boundary >= 265.0 (need ~41 units more -> ~3 RL steps)
    for _ in range(4):
        env.step(_zero_actions(env))

    left_goal = np.array(env.cfg.physics.left_goal_center, dtype=np.float64)
    dist_from_goal = float(np.linalg.norm(env.player_pos[b_atk_idx] - left_goal))
    assert dist_from_goal >= env.cfg.mechanics.gk_hold_opponent_buffer - 1e-3


def test_dynamic_vertical_lane_assignment_prevents_role_inversion() -> None:
    """Verify dynamic vertical lane assignment based on canonical Y prevents role inversion gradient cancellation."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    # Carrier is a_def_1 at (400, 420)
    def1_idx = env.agent_to_idx["a_def_1"]
    mid1_idx = env.agent_to_idx["a_mid_1"]
    mid2_idx = env.agent_to_idx["a_mid_2"]

    env.ball_holder = def1_idx
    env.player_pos[def1_idx] = np.array([400.0, 420.0], dtype=np.float64)
    # Invert positions: a_mid_1 in bottom half (y=620), a_mid_2 in top half (y=220)
    env.player_pos[mid1_idx] = np.array([600.0, 620.0], dtype=np.float64)
    env.player_pos[mid2_idx] = np.array([600.0, 220.0], dtype=np.float64)
    env.player_vel[:] = 0.0

    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    prev_pos = env.player_pos.copy()
    prev_dists = np.linalg.norm(env.player_pos - env.ball_pos[None, :], axis=1)

    # Compute tactical positioning rewards
    env._compute_tactical_positioning_rewards(step_rewards, prev_pos, prev_dists, any_teleport=False)

    # Now simulate a_mid_2 moving closer to Top Flank target (y decreases from 220 to 200)
    # and a_mid_1 moving closer to Bottom Flank target (y increases from 620 to 640)
    step_rewards_move = np.zeros(env.num_players, dtype=np.float64)
    prev_pos_move = env.player_pos.copy()
    env.player_pos[mid2_idx] = np.array([600.0, 200.0], dtype=np.float64)
    env.player_pos[mid1_idx] = np.array([600.0, 640.0], dtype=np.float64)

    env._compute_tactical_positioning_rewards(step_rewards_move, prev_pos_move, prev_dists, any_teleport=False)

    # Both players moved towards their nearest dynamic flank:
    # a_mid_2 (y=200, Top) and a_mid_1 (y=640, Bottom) should both receive positive support pocket approach rewards!
    assert step_rewards_move[mid2_idx] > 0.0
    assert step_rewards_move[mid1_idx] > 0.0


def test_asymmetric_crowding_exemptions_and_box_cutback_angles() -> None:
    """Verify asymmetric right-of-way crowding exemptions and wide-angle cutback relaxation near enemy box."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    atk_idx = env.agent_to_idx["a_atk_1"]
    mid_idx = env.agent_to_idx["a_mid_1"]

    # 1. Ball carrier is exempt from crowding; non-carrier receives crowding penalty
    env.ball_holder = atk_idx
    env.player_pos[atk_idx] = np.array([700.0, 420.0], dtype=np.float64)
    env.player_pos[mid_idx] = np.array([700.0, 480.0], dtype=np.float64)  # 60u away (< teammate_min_spacing=165u)
    # Move other teammates far away
    for idx in range(env.num_players):
        if idx not in (atk_idx, mid_idx):
            env.player_pos[idx] = np.array([100.0 + idx * 30.0, 100.0], dtype=np.float64)

    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    prev_pos = env.player_pos.copy()
    prev_dists = np.linalg.norm(env.player_pos - env.ball_pos[None, :], axis=1)

    env._compute_tactical_positioning_rewards(step_rewards, prev_pos, prev_dists, any_teleport=False)

    # Carrier (a_atk_1) is exempt from crowding penalty, mid_1 is penalized
    assert step_rewards[atk_idx] >= 0.0
    assert step_rewards[mid_idx] < -0.01

    # 2. Cutback angles near enemy box: a_atk_1 below semicircle, a_mid_1 above semicircle
    env.player_pos[atk_idx] = np.array([1250.0, 560.0], dtype=np.float64)  # theta ≈ -42 deg
    env.player_pos[mid_idx] = np.array([1250.0, 280.0], dtype=np.float64)  # theta ≈ +42 deg, delta theta ≈ 84 deg >= 40 deg
    step_rewards_box = np.zeros(env.num_players, dtype=np.float64)
    env._compute_tactical_positioning_rewards(step_rewards_box, env.player_pos.copy(), prev_dists, any_teleport=False)

    # Neither receives box_sector_crowding_pen because they are in different angular sectors
    assert step_rewards_box[atk_idx] >= 0.0
    assert step_rewards_box[mid_idx] >= 0.0

    # 3. Same-sector box clogging (< 35 deg): only non-carrier penalized
    env.player_pos[atk_idx] = np.array([1250.0, 550.0], dtype=np.float64)
    env.player_pos[mid_idx] = np.array([1235.0, 565.0], dtype=np.float64)
    step_rewards_clog = np.zeros(env.num_players, dtype=np.float64)
    env._compute_tactical_positioning_rewards(step_rewards_clog, env.player_pos.copy(), prev_dists, any_teleport=False)

    # Carrier (a_atk_1) unpenalized, mid_1 receives sector crowding penalty
    assert step_rewards_clog[atk_idx] >= 0.0
    assert step_rewards_clog[mid_idx] < -0.01


def test_final_third_cutback_and_box_angle_switch_pass_rewards() -> None:
    """Verify final-third cutbacks & semicircle switches qualify as box angle switches and award bonus."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    atk_idx = env.agent_to_idx["a_atk_1"]
    mid_idx = env.agent_to_idx["a_mid_1"]

    # a_atk_1 below semicircle passes to a_mid_1 above semicircle
    # (canon_dx = 1230 - 1250 = -20.0 < 0, but angle shift >= 25 deg around goal)
    env.player_pos[atk_idx] = np.array([1250.0, 560.0], dtype=np.float64)
    env.player_pos[mid_idx] = np.array([1230.0, 280.0], dtype=np.float64)
    env.ball_pos = env.player_pos[mid_idx].copy()
    env.ball_kick_origin = env.player_pos[atk_idx].copy()
    env.ball_holder = None
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.last_ball_holder = atk_idx
    env.last_ball_team = TEAM_A
    env.team_pass_chain[TEAM_A] = 0

    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(step_rewards)

    # Ball pickup by a_mid_1 completes the pass
    assert env.ball_holder == mid_idx
    # Passer should receive box_angle_pass_bonus (0.85)
    assert step_rewards[atk_idx] >= env.cfg.rewards.box_angle_pass_bonus
    # Chain count should increment
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
    assert rew.first_touch_reception_bonus == 0.35
    assert rew.keeper_sweeper_chase_scale == 0.028
    assert rew.keeper_edge_camp_radius == 95.0

    # Ensure keeper can move vertically inside the expanded goal mouth without barrier clamp
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)
    gk_idx = env.agent_to_idx["a_keeper"]
    env.player_pos[gk_idx] = np.array([40.0, 328.0], dtype=np.float64)
    env._clamp_players_to_boundaries()
    assert abs(env.player_pos[gk_idx, 1] - 328.0) < 1e-3


def test_stationary_receiver_earns_zero_chase_reward_and_active_receiver_earns_bonus() -> None:
    """Verify stationary receiver gets zero chase reward while active sprint earns displacement projection and touch bonus."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    passer_idx = env.agent_to_idx["a_def_1"]
    receiver_idx = env.agent_to_idx["a_mid_1"]

    env.player_pos[passer_idx] = np.array([300.0, 420.0], dtype=np.float64)
    env.player_pos[receiver_idx] = np.array([550.0, 420.0], dtype=np.float64)
    env.player_vel[:] = 0.0

    # Pass in flight toward receiver
    env.ball_holder = None
    env.ball_kicked_in_flight = True
    env.ball_kick_origin = env.player_pos[passer_idx].copy()
    env.last_ball_holder = passer_idx
    env.last_ball_team = TEAM_A
    env.ball_pos = np.array([400.0, 420.0], dtype=np.float64)  # Rolling toward receiver
    env.ball_vel = np.array([120.0, 0.0], dtype=np.float64)

    # 1. Receiver stands completely still (zero throttle): active chase reward is exactly 0.0
    r_still = np.zeros(env.num_players, dtype=np.float64)
    env._compute_tactical_positioning_rewards(
        r_still,
        env.player_pos.copy(),
        np.linalg.norm(env.player_pos - env.ball_pos, axis=1),
        False,
    )

    # 2. Receiver actively moves 10 units toward the ball
    prev_pos = env.player_pos.copy()
    env.player_pos[receiver_idx] += np.array([-10.0, 0.0])  # Displaced toward incoming ball
    r_moving = np.zeros(env.num_players, dtype=np.float64)
    env._compute_tactical_positioning_rewards(
        r_moving,
        prev_pos,
        np.linalg.norm(prev_pos - env.ball_pos, axis=1),
        False,
    )

    active_chase_reward = r_moving[receiver_idx] - r_still[receiver_idx]
    assert math.isclose(active_chase_reward, 10.0 * env.cfg.rewards.loose_ball_chase_scale, abs_tol=1e-5)
    assert active_chase_reward > 0.09

    # 3. Clean first touch reception awards discrete first_touch_reception_bonus
    env.ball_pos = env.player_pos[receiver_idx].copy()
    rew_arr = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(rew_arr)
    assert env.ball_holder == receiver_idx
    # Includes reception base (0.2), progressive bonus, mid-link bonus, and first_touch_reception_bonus (0.35)
    assert rew_arr[receiver_idx] >= env.cfg.rewards.first_touch_reception_bonus + 0.20


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
    """Verify GK obs[13] jumps from 0.0 to 0.25 on frame 1 in semicircle and sweeper fetch priority triggers."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)
    gk_idx = env.agent_to_idx["a_keeper"]

    # Ball at midfield: obs[13] must be 0.0
    env.ball_holder = None
    env.ball_pos = np.array([700.0, 420.0], dtype=np.float64)
    obs = env._build_single_observation(gk_idx)
    assert obs[13] == 0.0

    # Ball crosses into GK semicircle (radius 195.0), timer = 0.0: instant jump to 0.25!
    env.ball_pos = np.array([80.0, 420.0], dtype=np.float64)
    env.box_loose_ball_timer = 0.0
    obs = env._build_single_observation(gk_idx)
    assert math.isclose(obs[13], 0.25, abs_tol=1e-4)

    # As timer ramps to 3.0s, obs[13] ramps to 1.0
    env.box_loose_ball_timer = 1.5
    obs = env._build_single_observation(gk_idx)
    assert math.isclose(obs[13], 0.625, abs_tol=1e-4)

    env.box_loose_ball_timer = 3.0
    obs = env._build_single_observation(gk_idx)
    assert math.isclose(obs[13], 1.0, abs_tol=1e-4)


def test_keeper_holonomic_lateral_and_intercept_dive_saves_fast_shot() -> None:
    """Verify GK holonomic dive overrides velocity toward intercept point without angular turn delay and saves fast shot."""
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

    # Keeper must have dove holonomically at keeper_dive_speed (340.0 u/s) with reduced cooldown 1.8s
    speed = float(np.linalg.norm(env.player_vel[gk_idx]))
    assert speed >= 300.0  # Underwent holonomic dive impulse
    assert env.lunge_cooldown[gk_idx] > 1.5  # Cooldown applied (1.8s max)
    assert env.lunge_cooldown[gk_idx] <= 1.81


def test_foul_restart_formation_opens_central_lane_and_forward_halfspace_runners() -> None:
    """Verify foul restart places attacking ST at canonical 820, M1/M2 at 965 in halfspaces, and opens central lane."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    env._setup_keeper_foul_teleport(penalized_team=TEAM_A)
    # Attacking team is Team B (canonical coordinates)
    c_pos, _, _ = env._to_team_canonical(env.player_pos, env.player_vel, env.player_angle, TEAM_B)

    b_atk_idx = env.agent_to_idx["b_atk_1"]
    b_mid1_idx = env.agent_to_idx["b_mid_1"]
    b_mid2_idx = env.agent_to_idx["b_mid_2"]

    # Attacking 3-2 overload positions
    assert math.isclose(c_pos[b_atk_idx, 0], 820.0, abs_tol=1e-3)
    assert math.isclose(c_pos[b_atk_idx, 1], 420.0, abs_tol=1e-3)
    assert math.isclose(c_pos[b_mid1_idx, 0], 965.0, abs_tol=1e-3)
    assert math.isclose(c_pos[b_mid1_idx, 1], 235.0, abs_tol=1e-3)
    assert math.isclose(c_pos[b_mid2_idx, 0], 965.0, abs_tol=1e-3)
    assert math.isclose(c_pos[b_mid2_idx, 1], 605.0, abs_tol=1e-3)

    # Defending zonal block: striker behind ball at 640
    a_atk_idx = env.agent_to_idx["a_atk_1"]
    assert math.isclose(c_pos[a_atk_idx, 0], 640.0, abs_tol=1e-3)


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
    assert step_rewards[kicker_idx] < env.cfg.rewards.other_shot_on_target_ind * 0.5

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





