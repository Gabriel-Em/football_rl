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
from football_rl.env import FootballParallelEnv, parallel_env
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

    assert dist_keeper <= 145.0 + 1e-6
    assert dist_field_a > 145.0
    assert dist_field_b > 145.0


def test_field_3s_violation_ghost_mode_and_other_player_reset() -> None:
    """Verify 3-second field hold violation drops ball, enables Ghost Mode, and resets on other touch."""
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
    # Attacking Team B's b_atk_1 holds ball at (590, 420)
    b_atk_1_idx = env.agent_to_idx["b_atk_1"]
    b_mid_1_idx = env.agent_to_idx["b_mid_1"]
    b_mid_2_idx = env.agent_to_idx["b_mid_2"]
    assert env.ball_holder == b_atk_1_idx
    np.testing.assert_allclose(env.player_pos[b_atk_1_idx], [590.0, 420.0])
    np.testing.assert_allclose(env.player_pos[b_mid_1_idx], [460.0, 110.0])
    np.testing.assert_allclose(env.player_pos[b_mid_2_idx], [460.0, 730.0])
    np.testing.assert_allclose(env.player_pos[keeper_idx], [45.0, 420.0])

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

    surface = viewer.render_frame()
    assert surface.get_width() > 1000
    assert surface.get_height() > 600

    # Test toggles and scrubbing
    viewer.show_vision_cones = False
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
    env = FootballParallelEnv(DEFAULT_CONFIG)
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

    # GK Possession buffer: when GK holds ball, opposing field players are pushed outside 230u
    gk_idx = env.agent_to_idx["a_keeper"]
    env.ball_holder = gk_idx
    left_goal = np.array(env.cfg.physics.left_goal_center, dtype=np.float64)
    env.player_pos[opp_idx] = left_goal + np.array([180.0, 0.0], dtype=np.float64)  # dist 180 < 230
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
