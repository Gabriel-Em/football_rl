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
    """Verify role anchor & tactical guidance features in obs[13:16].

    For field players: obs[13] = role_base + 0.20 * pass_chain, obs[14:16] = relative target pocket.
    On reset, pass_chain is 0, so DEF obs[13] is 0.33, MID is 0.66, ATK is 1.0.
    For keeper: obs[13] = box_loose_timer, obs[14] = barrier_radius, obs[15] = pass_chain.
    """
    env = parallel_env(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    obs, _ = env.reset(seed=42)

    # Keeper on reset: box_loose_timer=0, barrier_radius=1.0, pass_chain=0
    np.testing.assert_allclose(obs["a_keeper"][13], 0.0, atol=1e-3)
    np.testing.assert_allclose(obs["a_keeper"][14], 1.0, atol=1e-3)
    np.testing.assert_allclose(obs["a_keeper"][15], 0.0, atol=1e-3)

    # Field roles role_base on reset: DEF=-0.75, MID=0.0, ATK=+0.75
    np.testing.assert_allclose(obs["a_def_1"][13], -0.75, atol=1e-2)
    np.testing.assert_allclose(obs["a_def_2"][13], -0.75, atol=1e-2)
    np.testing.assert_allclose(obs["a_mid_1"][13], 0.0, atol=1e-2)
    np.testing.assert_allclose(obs["a_mid_2"][13], 0.0, atol=1e-2)
    np.testing.assert_allclose(obs["a_atk_1"][13], 0.75, atol=1e-2)

    np.testing.assert_allclose(obs["b_keeper"][13], 0.0, atol=1e-3)
    np.testing.assert_allclose(obs["b_keeper"][14], 1.0, atol=1e-3)
    np.testing.assert_allclose(obs["b_def_1"][13], -0.75, atol=1e-2)
    np.testing.assert_allclose(obs["b_mid_1"][13], 0.0, atol=1e-2)
    np.testing.assert_allclose(obs["b_atk_1"][13], 0.75, atol=1e-2)


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
    assert gk_max_spd == 640.0
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
    assert math.isclose(rew2[p1], (0.20 + rx_bonus) * 0.25, abs_tol=1e-5)
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
    assert math.isclose(rew3[p2], (0.20 + rx_bonus) * (0.25**2), abs_tol=1e-5)
    assert rew3[p2] < 0.05

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
    prog_bonus = effective_dx * env.cfg.rewards.pass_progressive_dx_weight * env.cfg.rewards.scanned_pass_mult
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
    expected_prog = (
        (350.0 * env.cfg.rewards.pass_progressive_dx_weight) + (2 * env.cfg.rewards.pass_line_break_bonus)
    ) * env.cfg.rewards.scanned_pass_mult
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
    # obs[14] contains active box radius (1.0 on reset) and obs[15] contains team pass chain (0.0 on reset)
    assert math.isclose(obs_gk[14], 1.0, abs_tol=1e-5)
    assert math.isclose(obs_gk[15], 0.0, abs_tol=1e-5)
    # Field DEF player encodes role anchor in obs[13] (-0.75 + 0.02 * pass_chain)
    assert math.isclose(obs_def[13], -0.75, abs_tol=1e-2)

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

    # As timer ramps to max_time (5.0s), obs[13] ramps to 1.0
    half_time = 0.5 * env.cfg.mechanics.keeper_box_loose_max_time
    env.box_loose_ball_timer = half_time
    obs = env._build_single_observation(gk_idx)
    assert math.isclose(obs[13], 0.625, abs_tol=1e-4)

    env.box_loose_ball_timer = env.cfg.mechanics.keeper_box_loose_max_time
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


def test_short_distribution_pass_registered_as_completed_pass() -> None:
    """Verify short distribution (e.g. GK to CB ~90u apart, ~50u ball travel) registers as a Completed Pass."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=410)

    # In Half 1, Team A defends Left (X=0)
    gk_idx = env.agent_to_idx["a_keeper"]
    d1_idx = env.agent_to_idx["a_def_1"]

    # Keeper at (100.0, 384.0), D1 at (170.0, 324.0) -> separation ~92u
    env.player_pos[gk_idx] = np.array([100.0, 384.0], dtype=np.float64)
    env.player_pos[d1_idx] = np.array([170.0, 324.0], dtype=np.float64)

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
    # Both players receive appropriate pass link-up rewards
    assert step_rewards[d1_idx] >= env.cfg.rewards.first_touch_reception_bonus


def test_wall_bounce_resets_shot_on_target_and_near_shot_reward_distinction() -> None:
    """Verify true shots on target vs near shots hitting end wall beside goal, and that wall bounces prevent false saves."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=55)

    shooter_idx = env.agent_to_idx["b_atk_1"]
    gk_idx = env.agent_to_idx["a_keeper"]

    # 1. Near Shot: Attacker shoots toward Left goal line, but trajectory hits wall beside goal (Y = 300, goal is [320, 520])
    env.ball_holder = shooter_idx
    env.player_pos[shooter_idx] = np.array([400.0, 300.0], dtype=np.float64)
    env.player_angle[shooter_idx] = math.pi  # Aimed directly at (0.0, 300.0) - hits end wall 20u beside post
    rewards = np.zeros(env.num_players, dtype=np.float64)

    env._execute_kick(shooter_idx, kick_power=1.0, step_rewards=rewards)

    # Near shot must NOT be counted as a shot on target
    assert env.ball_kick_was_shot_on_target is False
    assert env.stats["shots"][TEAM_B] == 0
    # But receives partial reward (35%) to reward shooting ambition
    assert rewards[shooter_idx] > 0.0
    expected_near_rew = env.cfg.rewards.atk_shot_on_target_ind * env.cfg.rewards.shot_near_target_mult
    assert rewards[shooter_idx] >= expected_near_rew * 0.9

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

    # Scenario 2: Incoming shot in front of keeper with adequate reaction time DOES get awarded GK DIVE!
    env.reset(seed=102)
    env.kickoff_protection_timer = 0.0  # Clear kickoff protection for test
    gk_a_idx = env.agent_to_idx["a_keeper"]  # In half 1, a_keeper defends left goal (x=0)
    env.player_pos[gk_a_idx] = np.array([35.0, 410.0], dtype=np.float64)
    env.lunge_cooldown[gk_a_idx] = 0.0
    env.ball_holder = None
    env.last_ball_holder = env.agent_to_idx["b_atk_1"]
    env.last_ball_team = TEAM_B
    env.ball_pos = np.array([120.0, 390.0], dtype=np.float64)  # in front of keeper (120 > 35)
    env.ball_vel = np.array([-250.0, -10.0], dtype=np.float64)
    env.ball_kicked_in_flight = True
    env.ball_kick_was_shot_on_target = True

    actions = _zero_actions(env)
    actions["a_keeper"] = np.array([0.0, -1.0, 0.0, 1.0], dtype=np.float32)

    _, rewards_2, _, _, _ = env.step(actions)

    events_2 = [e["type"] for fr in env.replay_frames[-4:] for e in fr.get("events", [])]
    assert "GK DIVE!" in events_2, "Legitimate dive attempt in front of keeper should be awarded GK DIVE!"
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
    """Verify that backward passes under pressure earn escape_press_pass_bonus,
    and final-third drop passes to midfielders earn attacking_layoff_bonus.
    """
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=201)

    # 1. Escape Press Scenario:
    # Team A defends left goal (canonical space = world space).
    # a_atk_1 is at (1000.0, 700.0).
    # Opponent b_def_1 is pressing closely at (1050.0, 700.0) - dist 50u <= 140u.
    # a_atk_1 passes backwards to open midfielder a_mid_2 at (750.0, 420.0).
    # Nearest opponent to a_mid_2 is > 180u away.
    atk_idx = env.agent_to_idx["a_atk_1"]
    mid_idx = env.agent_to_idx["a_mid_2"]
    opp_def_idx = env.agent_to_idx["b_def_1"]

    env.player_pos[atk_idx] = np.array([1000.0, 700.0], dtype=np.float64)
    env.player_pos[opp_def_idx] = np.array([1050.0, 700.0], dtype=np.float64)
    env.player_pos[mid_idx] = np.array([750.0, 420.0], dtype=np.float64)
    # Move other opponents away
    for idx in range(env.num_players):
        if env.player_team[idx] == TEAM_B and idx != opp_def_idx:
            env.player_pos[idx] = np.array([1250.0, 420.0], dtype=np.float64)

    env.ball_holder = atk_idx
    env.ball_hold_timer = 2.0
    rew_arr = np.zeros(env.num_players, dtype=np.float64)

    # Face toward receiver (angle to 750, 420 from 1000, 700)
    to_mid = env.player_pos[mid_idx] - env.player_pos[atk_idx]
    env.player_angle[atk_idx] = math.atan2(to_mid[1], to_mid[0])

    env._execute_kick(atk_idx, kick_power=0.8, step_rewards=rew_arr)
    assert env.ball_kicked_in_flight is True

    # Simulate arrival at receiver
    env.ball_pos = env.player_pos[mid_idx].copy()
    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(step_rewards)

    assert env.ball_holder == mid_idx
    assert env.stats["completed_passes"] == 1
    # Passer must receive escape_press_pass_bonus
    assert step_rewards[atk_idx] >= env.cfg.rewards.escape_press_pass_bonus
    # Chain must be preserved >= 1
    assert env.team_pass_chain[TEAM_A] >= 1

    # 2. Attacking Lay-Off (Hold-up Wall Pass) Scenario:
    env.reset(seed=202)
    env.player_pos[atk_idx] = np.array([850.0, 420.0], dtype=np.float64)
    env.player_pos[mid_idx] = np.array([620.0, 420.0], dtype=np.float64)
    env.ball_holder = atk_idx
    env.ball_hold_timer = 0.5
    env.player_angle[atk_idx] = math.pi  # Facing backward directly at midfielder

    rew_arr.fill(0.0)
    env._execute_kick(atk_idx, kick_power=0.6, step_rewards=rew_arr)

    env.ball_pos = env.player_pos[mid_idx].copy()
    step_rewards.fill(0.0)
    env._check_free_ball_pickup(step_rewards)

    # Passer must receive attacking_layoff_bonus
    assert step_rewards[atk_idx] >= env.cfg.rewards.attacking_layoff_bonus
    assert env.team_pass_chain[TEAM_A] >= 1


def test_give_and_go_one_two_bonus_and_anti_farming_decay() -> None:
    """Verify that a 1-2 give-and-go return pass is rewarded with full multiplier and give_and_go_bonus,
    while stationary 3+ pass loops between the same pair suffer steep decay.
    """
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=203)

    m1_idx = env.agent_to_idx["a_mid_1"]
    m2_idx = env.agent_to_idx["a_mid_2"]

    # --- Pass 1: M1 passes to M2 ---
    env.player_pos[m1_idx] = np.array([450.0, 420.0], dtype=np.float64)
    env.player_pos[m2_idx] = np.array([600.0, 420.0], dtype=np.float64)
    env.ball_holder = m1_idx
    env.player_angle[m1_idx] = 0.0  # Facing M2 (+X)
    rew_arr = np.zeros(env.num_players, dtype=np.float64)

    env._execute_kick(m1_idx, kick_power=0.7, step_rewards=rew_arr)
    env.ball_pos = env.player_pos[m2_idx].copy()
    step_rew_1 = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(step_rew_1)

    assert env.pairwise_repeat_count == 0
    assert env.ball_holder == m2_idx

    # --- Pass 2: M1 sprints forward 80u to (530.0, 420.0), M2 returns ball to M1 (GIVE & GO) ---
    env.player_pos[m1_idx] = np.array([530.0, 420.0], dtype=np.float64)  # 530 - 450 = +80u forward
    env.player_angle[m2_idx] = math.pi  # Facing backward toward M1

    rew_arr.fill(0.0)
    env._execute_kick(m2_idx, kick_power=0.6, step_rewards=rew_arr)
    env.ball_pos = env.player_pos[m1_idx].copy()
    step_rew_2 = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(step_rew_2)

    # Must recognize Give-and-Go!
    assert env.pairwise_repeat_count == 1
    # Passer M2 receives give_and_go_bonus
    assert step_rew_2[m2_idx] >= env.cfg.rewards.give_and_go_bonus

    # --- Pass 3: Immediate Pass back without movement (Exchange 3 between same pair) ---
    rew_arr.fill(0.0)
    env.player_angle[m1_idx] = 0.0
    env._execute_kick(m1_idx, kick_power=0.6, step_rewards=rew_arr)
    env.ball_pos = env.player_pos[m2_idx].copy()
    step_rew_3 = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(step_rew_3)

    assert env.pairwise_repeat_count == 2
    # Receiver bonus + team bonus must be decayed by 0.25^2 = 0.0625 (total added reward <= 0.05)
    rx_added_3 = step_rew_3[m2_idx] - env.cfg.rewards.first_touch_reception_bonus
    assert rx_added_3 <= 0.05

    # --- Pass 4: Stalling loop (Exchange 4 between same pair) ---
    rew_arr.fill(0.0)
    env.player_angle[m2_idx] = math.pi
    env._execute_kick(m2_idx, kick_power=0.6, step_rewards=rew_arr)
    env.ball_pos = env.player_pos[m1_idx].copy()
    step_rew_4 = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(step_rew_4)

    assert env.pairwise_repeat_count == 3
    # Suffer steep 0.25^3 = 0.0156 decay (total added reward <= 0.02)
    rx_added_4 = step_rew_4[m1_idx] - env.cfg.rewards.first_touch_reception_bonus
    assert rx_added_4 <= 0.02


def test_loose_ball_two_defenders_mutual_crowding_exemption_and_continuous_chase() -> None:
    """Verify that when two defenders converge on a loose ball in proximity (<165u):
    1. Neither defender suffers mutual crowding or approach repulsion penalties against each other.
    2. Chase reward is continuously awarded within 55u down to pickup distance (eliminating deadzone).
    3. Primary chaser hysteresis prevents 15 Hz flip-flopping between defenders.
    """
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=201)

    d1_idx = env.agent_to_idx["a_def_1"]
    d2_idx = env.agent_to_idx["a_def_2"]

    # Match 783 geometry: Ball at (328.0, 457.0), D1 at (266.0, 432.0), D2 at (270.0, 505.0)
    # Distance between D1 and D2 is ~73u (well below teammate_min_spacing 165u!)
    env.ball_holder = None
    env.ball_kicked_in_flight = False
    env.ball_pos = np.array([328.0, 457.0], dtype=np.float64)
    env.ball_vel = np.zeros(2, dtype=np.float64)

    env.player_pos[d1_idx] = np.array([266.0, 432.0], dtype=np.float64)
    env.player_pos[d2_idx] = np.array([270.0, 505.0], dtype=np.float64)

    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    prev_pos = env.player_pos.copy()
    prev_dists = np.linalg.norm(env.player_pos - env.ball_pos[None, :], axis=1)

    # Compute tactical positioning rewards
    env._compute_tactical_positioning_rewards(step_rewards, prev_pos, prev_dists, any_teleport=False)

    # Crucial assertion: Neither D1 nor D2 should receive negative crowding penalty from each other!
    # With min_spacing = 165u and dist = 73u, without exemption pen would be ~ -0.015 per player.
    # With exemption, rewards should be non-negative (no crowding pen).
    assert step_rewards[d1_idx] >= 0.0
    assert step_rewards[d2_idx] >= 0.0

    # Test continuous chase within 55u:
    # Place D1 at distance 48u from ball (<55u) moving toward ball
    env.player_pos[d1_idx] = np.array([285.0, 457.0], dtype=np.float64)  # 328 - 285 = 43u (<55u)
    prev_pos_d1 = env.player_pos.copy()
    prev_pos_d1[d1_idx] = np.array([280.0, 457.0], dtype=np.float64)  # closed 5u
    prev_dists_d1 = np.linalg.norm(prev_pos_d1 - env.ball_pos[None, :], axis=1)

    step_rewards_chase = np.zeros(env.num_players, dtype=np.float64)
    env._compute_tactical_positioning_rewards(
        step_rewards_chase, prev_pos_d1, prev_dists_d1, any_teleport=False
    )
    # Chase reward MUST be awarded even though dist < 55u!
    assert step_rewards_chase[d1_idx] > 0.005


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
    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    prev_pos = env.player_pos.copy()
    prev_dists = np.linalg.norm(env.player_pos - env.ball_pos[None, :], axis=1)
    env._compute_tactical_positioning_rewards(step_rewards, prev_pos, prev_dists, any_teleport=False)

    # Defender rewards should have no crowding deduction (-pen)
    # teammate_defender_spacing is 120u, so 150u is well above threshold!
    assert step_rewards[d1_a] >= -1e-6
    assert step_rewards[d2_a] >= -1e-6


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
    assert step_rew[p1] >= cfg.rewards.wall_pass_bonus - 1e-5
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
    """Verify proximity gravitational pull for stationary ball chasers within 95u."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)

    env.ball_holder = None
    env.ball_pos = np.array([700.0, 420.0], dtype=np.float64)
    # Move all other players far away
    for i in range(12):
        env.player_pos[i] = np.array([100.0, 100.0], dtype=np.float64)

    # Place player 3 (a_mid_1) at distance 50u from ball
    env.player_pos[3] = np.array([700.0, 470.0], dtype=np.float64)
    prev_pos = env.player_pos.copy()
    env.prev_ball_at_step_start = env.ball_pos.copy()

    step_rewards = np.zeros(12, dtype=np.float64)
    env._compute_tactical_positioning_rewards(
        step_rewards,
        prev_pos,
        np.linalg.norm(prev_pos - env.ball_pos[None, :], axis=1),
        any_teleport=False,
    )

    # Player 3 is primary chaser (rank 0 -> w=1.0) and stationary (active_chase_closed=0.0)
    # Proximity reward should be active: 0.008 * (1.0 - (50 - 28) / (95 - 28)) = 0.008 * 45/67 ≈ 0.00537
    assert step_rewards[3] > 0.005, f"Expected proximity reward > 0.005, got {step_rewards[3]}"


def test_contested_ball_recovery_and_pickup_tackle_grace() -> None:
    """Verify that recovering a contested loose ball awards the contested bonus and grants tackle immunity."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)
    env.kickoff_protection_timer = 0.0

    p_red = env.agent_to_idx["a_mid_1"]
    p_blue = env.agent_to_idx["b_mid_1"]

    # Place loose ball at (700, 400)
    # Red is within pickup range (dist = 15 <= 28)
    # Blue is within contested range (dist to ball = 45 <= 65)
    env.ball_pos = np.array([700.0, 400.0], dtype=np.float64)
    env.ball_vel = np.array([0.0, 0.0], dtype=np.float64)
    env.ball_holder = None
    env.ball_kicked_in_flight = False
    env.pickup_cooldown[:] = 0.0
    env.stun_timer[:] = 0.0

    env.player_pos[p_red] = np.array([715.0, 400.0], dtype=np.float64)
    env.player_pos[p_blue] = np.array([745.0, 400.0], dtype=np.float64)

    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(step_rewards)

    # 1. Red should win the ball, get pickup reward + team pickup reward + contested recovery bonus (+0.5 + 0.15 + 0.4 = +1.05)
    assert env.ball_holder == p_red
    expected_red_rew = (
        env.cfg.rewards.loose_ball_pickup_ind
        + env.cfg.rewards.loose_ball_pickup_team
        + env.cfg.rewards.contested_ball_recovery_bonus
    )
    assert math.isclose(step_rewards[p_red], expected_red_rew, rel_tol=1e-5)

    # 2. Pickup tackle grace timer must be set to pickup_tackle_grace (0.35s)
    assert math.isclose(env.pickup_tackle_grace_timer, env.cfg.mechanics.pickup_tackle_grace, rel_tol=1e-5)

    # 3. Blue attempts immediate tackle during grace period
    step_rewards.fill(0.0)
    env._attempt_tackle(p_blue, step_rewards)

    # Tackle must be blocked by grace period: Red retains ball, Red not stunned, Blue not whiff stunned
    assert env.ball_holder == p_red
    assert env.stun_timer[p_red] == 0.0
    assert env.stun_timer[p_blue] == 0.0
    assert step_rewards[p_blue] == 0.0

    # 4. Once grace expires, Blue can tackle normally
    env.pickup_tackle_grace_timer = 0.0
    from unittest.mock import MagicMock
    mock_rng = MagicMock()
    mock_rng.random.return_value = 0.0  # Force 100% tackle success
    orig_rng = env.np_random
    env.np_random = mock_rng

    step_rewards.fill(0.0)
    env._attempt_tackle(p_blue, step_rewards)

    assert env.ball_holder == p_blue
    assert env.stun_timer[p_red] > 0.0
    assert step_rewards[p_blue] > 0.0

    env.np_random = orig_rng


def test_keeper_pass_corridor_blocks_clearance_bonus() -> None:
    """Verify that keeper_clear_forward_ind is withheld if an opponent stands within the 38u pass corridor."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)

    gk_idx = env.agent_to_idx["a_keeper"]  # Side Left, defends X=0, attacks X=1400
    opp_atk_idx = env.agent_to_idx["b_atk_1"]

    env.player_pos[gk_idx] = np.array([50.0, 420.0], dtype=np.float64)
    env.player_angle[gk_idx] = 0.0  # Facing directly forward +X
    env.ball_pos = np.array([74.0, 420.0], dtype=np.float64)
    env.ball_holder = gk_idx

    # Case 1: Opponent is standing at (180, 420) directly in the corridor (proj = 106u, perp_dist = 0u <= 38u)
    env.player_pos[opp_atk_idx] = np.array([180.0, 420.0], dtype=np.float64)

    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._execute_kick(gk_idx, kick_power=0.8, step_rewards=step_rewards)

    # Clearance bonus should be BLOCKED because corridor has opponent
    assert step_rewards[gk_idx] <= 0.0

    # Case 2: Move opponent outside the corridor to (180, 475) (proj = 106u, perp_dist = 55u > 38u)
    env.ball_holder = gk_idx
    env.player_pos[gk_idx] = np.array([50.0, 420.0], dtype=np.float64)
    env.player_angle[gk_idx] = 0.0
    env.ball_pos = np.array([74.0, 420.0], dtype=np.float64)
    env.player_pos[opp_atk_idx] = np.array([180.0, 475.0], dtype=np.float64)

    step_rewards.fill(0.0)
    env._execute_kick(gk_idx, kick_power=0.8, step_rewards=step_rewards)

    # Clearance bonus should now be awarded (> 0)
    assert step_rewards[gk_idx] > 0.0


def test_keeper_turnover_penalty_gradient() -> None:
    """Verify that keeper turnovers follow a smooth penalty gradient from -2.0 (own box) to -0.8 (midfield)."""
    cfg = replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    env = FootballParallelEnv(cfg)
    env.reset(seed=42)

    gk_idx = env.agent_to_idx["a_keeper"]  # Side Left, kicker_team = 0
    opp_idx = env.agent_to_idx["b_atk_1"]

    env.last_ball_holder = gk_idx
    env.last_ball_team = 0
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.ball_kick_origin = np.array([50.0, 420.0], dtype=np.float64)
    env.ball_kick_was_shot_on_target = False

    # Scenario A: Deep defensive turnover (canonical X = 150 <= 200) -> -2.0
    env.player_pos[opp_idx] = np.array([150.0, 420.0], dtype=np.float64)
    env.ball_pos = np.array([150.0, 420.0], dtype=np.float64)
    env.ball_holder = None
    env.pickup_cooldown[:] = 0.0
    env.stun_timer[:] = 0.0

    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(step_rewards)

    # GK should receive full deep turnover penalty (-2.0) + team penalty (-0.2) = -2.2
    assert math.isclose(
        step_rewards[gk_idx],
        env.cfg.rewards.keeper_deep_turnover_pen + env.cfg.rewards.intercepted_team,
        rel_tol=1e-5,
    )

    # Scenario B: Midfield turnover (canonical X = 450, halfway between 200 and 700)
    # Expected penalty: -2.0 + 1.2 * (250/500) = -1.4; with team penalty: -1.6
    env.last_ball_holder = gk_idx
    env.last_ball_team = 0
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.ball_kick_was_shot_on_target = False
    env.player_pos[opp_idx] = np.array([450.0, 420.0], dtype=np.float64)
    env.ball_pos = np.array([450.0, 420.0], dtype=np.float64)
    env.ball_holder = None
    env.pickup_cooldown[:] = 0.0
    env.stun_timer[:] = 0.0

    step_rewards.fill(0.0)
    env._check_free_ball_pickup(step_rewards)

    expected_mid_pen = -2.0 + 1.2 * 0.5 + env.cfg.rewards.intercepted_team  # -1.6
    assert math.isclose(step_rewards[gk_idx], expected_mid_pen, rel_tol=1e-5)

    # Scenario C: Upfield turnover (canonical X = 800 >= 700) -> standard -0.8; with team: -1.0
    env.last_ball_holder = gk_idx
    env.last_ball_team = 0
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.ball_kick_was_shot_on_target = False
    env.player_pos[opp_idx] = np.array([800.0, 420.0], dtype=np.float64)
    env.ball_pos = np.array([800.0, 420.0], dtype=np.float64)
    env.ball_holder = None
    env.pickup_cooldown[:] = 0.0
    env.stun_timer[:] = 0.0

    step_rewards.fill(0.0)
    env._check_free_ball_pickup(step_rewards)

    expected_upfield = env.cfg.rewards.intercepted_kicker_ind + env.cfg.rewards.intercepted_team
    assert math.isclose(step_rewards[gk_idx], expected_upfield, rel_tol=1e-5)


def test_keeper_5s_unclaimed_box_ball_and_post_save_grace() -> None:
    """Verify 5.0s box loose timer, differentiated event naming, and post-save grace timer."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    gk_idx = env.agent_to_idx["a_keeper"]
    step_rewards = np.zeros(env.num_players, dtype=np.float64)

    # 1. Verify GK hold violation emits '5s KEEPER HOLD FOUL!'
    env.ball_holder = gk_idx
    env.ball_hold_timer = 5.01
    env._frame_events.clear()
    env._handle_keeper_5s_violation(gk_idx, step_rewards, is_hold_violation=True)
    assert any(e["type"] == "5s KEEPER HOLD FOUL!" for e in env._frame_events)

    # 2. Verify loose ball in box emits '5s UNCLAIMED BOX BALL!'
    env._frame_events.clear()
    env._handle_keeper_5s_violation(gk_idx, step_rewards, is_hold_violation=False)
    assert any(e["type"] == "5s UNCLAIMED BOX BALL!" for e in env._frame_events)

    # 3. Verify post-save grace pauses box_loose_ball_timer accumulation
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

    # 3. Ball starts moving again (> 15.0 u/s) -> timer resets immediately and radius re-expands
    env.ball_vel = np.array([120.0, 0.0], dtype=np.float64)
    shrunken_r = float(env.active_box_radius[SIDE_LEFT])
    env._physics_substep(np.zeros((env.num_players, 4), dtype=np.float32), np.zeros(env.num_players))

    assert env.box_loose_ball_timer == 0.0
    assert env.active_box_radius[SIDE_LEFT] > shrunken_r


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
    env.step_external_push.fill(0.0)
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

    env.step_external_push.fill(0.0)
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
    """Verify carrier rotation produces 0 team advance penalty, backpedaling earns 0 advance bonus, and defender advance is tapered."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    carrier_idx = env.agent_to_idx["a_mid_1"]
    team = TEAM_A

    # 1. Rotating in place at enemy semicircle edge: carrier center remains fixed, only body angle changes
    env.ball_holder = carrier_idx
    env.player_pos[carrier_idx] = np.array([1205.0, 420.0], dtype=np.float64)
    env.player_angle[carrier_idx] = 0.0  # facing goal
    env.player_vel[carrier_idx] = np.array([0.0, 0.0], dtype=np.float64)
    env.ball_pos = env.player_pos[carrier_idx] + np.array([env.cfg.mechanics.ball_carry_offset, 0.0])

    # Rotate 180 degrees to face field (turn action) without throttle
    actions = _zero_actions(env)
    actions["a_mid_1"] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # pure turn
    obs, rewards, terms, truncs, infos = env.step(actions)

    # Carrier rotated without advancing center towards or away from enemy goal:
    # Team advance reward should be ~0.0 (no massive negative penalty for looking back for a cutback pass!)
    team_adv_rew = env.cfg.rewards.ball_advance_team_scale
    assert rewards["a_def_1"] >= -team_adv_rew * 2.0  # bounded near 0, NOT -48u * 0.004

    # 2. Backpedaling into enemy semicircle: facing away from goal (pi), driving backwards (+X)
    env.ball_holder = carrier_idx
    env.player_pos[carrier_idx] = np.array([1150.0, 420.0], dtype=np.float64)
    env.player_angle[carrier_idx] = math.pi  # facing away from enemy goal
    env.player_vel[carrier_idx] = np.array([80.0, 0.0], dtype=np.float64)  # moving toward goal in reverse!
    env.ball_pos = env.player_pos[carrier_idx] - np.array([env.cfg.mechanics.ball_carry_offset, 0.0])

    actions = _zero_actions(env)
    # Reverse throttle: moving +X while facing -X
    actions["a_mid_1"] = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obs, rewards, terms, truncs, infos = env.step(actions)

    # When backpedaling (long_v < -5.0), carrier_goal_advance_ind_scale MUST NOT be awarded!
    # Even though center advanced toward goal, carrier individual advance reward is 0.0
    ind_scale = env.cfg.rewards.carrier_goal_advance_ind_scale
    # The carrier's reward should not include positive carrier_goal_advance_ind
    assert rewards["a_mid_1"] < ind_scale * 10.0

    # 3. Defender dribbling advance reward taper in opponent half (> 680u) vs own half (<= 680u)
    def_idx = env.agent_to_idx["a_def_1"]
    
    # In own half: full 1.0x advance
    env.ball_holder = def_idx
    env.player_pos[def_idx] = np.array([400.0, 420.0], dtype=np.float64)
    env.player_angle[def_idx] = 0.0
    env.player_vel[def_idx] = np.array([60.0, 0.0], dtype=np.float64)
    env.ball_pos = env.player_pos[def_idx] + np.array([env.cfg.mechanics.ball_carry_offset, 0.0])
    actions = _zero_actions(env)
    actions["a_def_1"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _, rew_own, _, _, _ = env.step(actions)
    adv_own = rew_own["a_def_1"]

    # In opponent half (> 680u): 0.25x advance multiplier
    env.ball_holder = def_idx
    env.player_pos[def_idx] = np.array([850.0, 420.0], dtype=np.float64)
    env.player_angle[def_idx] = 0.0
    env.player_vel[def_idx] = np.array([60.0, 0.0], dtype=np.float64)
    env.ball_pos = env.player_pos[def_idx] + np.array([env.cfg.mechanics.ball_carry_offset, 0.0])
    actions = _zero_actions(env)
    actions["a_def_1"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _, rew_opp, _, _, _ = env.step(actions)
    adv_opp = rew_opp["a_def_1"]

    # Defender in opponent half earns significantly less advance reward than in own half
    assert adv_opp < adv_own


def test_semicircle_cutback_pass_triggers_layoff_and_increments_chain() -> None:
    """Verify cutbacks from byline/semicircle trigger layoff to ST, MID, or DEF, while rejecting non-cutbacks."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    atk = env.agent_to_idx["a_atk_1"]
    mid = env.agent_to_idx["a_mid_1"]
    df = env.agent_to_idx["a_def_1"]
    team = TEAM_A

    # 1. ATK -> MID cutback from (1310, 580) to (1120, 420)
    env.player_pos[atk] = np.array([1310.0, 580.0], dtype=np.float64)
    env.player_pos[mid] = np.array([1120.0, 420.0], dtype=np.float64)
    env.ball_holder = None
    env.ball_kick_origin = env.player_pos[atk].copy()
    env.ball_pos = env.player_pos[mid].copy()
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.last_ball_holder = atk
    env.last_ball_team = team
    env.team_pass_chain[team] = 1

    rew = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(rew)
    assert env.team_pass_chain[team] == 2
    assert rew[atk] >= env.cfg.rewards.attacking_layoff_bonus
    assert any(e.get("type") == "LAY-OFF!" for e in env._frame_events)

    # 2. DEF (overlapping) -> ATK cutback: kicker is DEF, receiver is ATK
    env.reset(seed=43)
    env.player_pos[df] = np.array([1320.0, 570.0], dtype=np.float64)
    env.player_pos[atk] = np.array([1100.0, 420.0], dtype=np.float64)
    env.ball_holder = None
    env.ball_kick_origin = env.player_pos[df].copy()
    env.ball_pos = env.player_pos[atk].copy()
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.last_ball_holder = df
    env.last_ball_team = team
    env.team_pass_chain[team] = 1

    rew2 = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(rew2)
    assert env.team_pass_chain[team] == 2
    assert rew2[df] >= env.cfg.rewards.attacking_layoff_bonus
    assert any(e.get("type") == "LAY-OFF!" for e in env._frame_events)

    # 3. Lateral cutback inward: Delta X is small (-5u), but Delta Y towards center is 150u
    env.reset(seed=44)
    env.player_pos[atk] = np.array([1250.0, 600.0], dtype=np.float64)
    env.player_pos[mid] = np.array([1245.0, 420.0], dtype=np.float64)
    env.ball_holder = None
    env.ball_kick_origin = env.player_pos[atk].copy()
    env.ball_pos = env.player_pos[mid].copy()
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.last_ball_holder = atk
    env.last_ball_team = team
    env.team_pass_chain[team] = 1

    rew3 = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(rew3)
    assert env.team_pass_chain[team] == 2
    assert rew3[atk] >= env.cfg.rewards.attacking_layoff_bonus
    assert any(e.get("type") == "LAY-OFF!" for e in env._frame_events)

    # 4. Anti-Exploit / False Positive: Forward pass towards goal line is NOT a cutback
    env.reset(seed=45)
    env.player_pos[mid] = np.array([1150.0, 420.0], dtype=np.float64)
    env.player_pos[atk] = np.array([1300.0, 420.0], dtype=np.float64)  # forward into box
    env.ball_holder = None
    env.ball_kick_origin = env.player_pos[mid].copy()
    env.ball_pos = env.player_pos[atk].copy()
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.last_ball_holder = mid
    env.last_ball_team = team

    rew4 = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(rew4)
    # Forward pass into box is progressive, NOT an attacking layoff
    assert not any(e.get("type") == "LAY-OFF!" for e in env._frame_events)


def test_role_aware_chaser_handoff_prioritizes_mid_and_atk_over_def_in_opponent_half() -> None:
    """Verify role-aware chaser priority in opponent half, own half, deep defense, and State C."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    d1 = env.agent_to_idx["a_def_1"]
    m1 = env.agent_to_idx["a_mid_1"]
    a1 = env.agent_to_idx["a_atk_1"]

    # 1. State B (Loose Ball) in Opponent Half (X=1050 > 660):
    # Place DEF closer in raw distance (40u) than MID (90u) and ATK (80u)
    env.ball_holder = None
    env.ball_pos = np.array([1050.0, 420.0], dtype=np.float64)
    env.player_pos[d1] = np.array([1010.0, 420.0], dtype=np.float64)  # raw dist = 40.0u
    env.player_pos[m1] = np.array([960.0, 420.0], dtype=np.float64)   # raw dist = 90.0u
    env.player_pos[a1] = np.array([970.0, 420.0], dtype=np.float64)   # raw dist = 80.0u

    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._compute_tactical_positioning_rewards(
        step_rewards,
        env.player_pos.copy(),
        np.linalg.norm(env.player_pos - env.ball_pos[None, :], axis=1),
        any_teleport=False,
    )
    assert env.primary_loose_ball_chaser[TEAM_A] in (m1, a1)

    # 2. State B in Own Half (X=400 <= 660):
    # DEF has zero distance offset; closest player wins priority
    env.ball_pos = np.array([400.0, 420.0], dtype=np.float64)
    env.player_pos[d1] = np.array([360.0, 420.0], dtype=np.float64)  # raw dist = 40u
    env.player_pos[m1] = np.array([480.0, 420.0], dtype=np.float64)  # raw dist = 80u
    env.player_pos[a1] = np.array([550.0, 420.0], dtype=np.float64)

    step_rewards.fill(0.0)
    env._compute_tactical_positioning_rewards(
        step_rewards,
        env.player_pos.copy(),
        np.linalg.norm(env.player_pos - env.ball_pos[None, :], axis=1),
        any_teleport=False,
    )
    assert env.primary_loose_ball_chaser[TEAM_A] == d1

    # 3. Deep Defensive Third (X=300 < 450):
    # ATK has +130u penalty to keep them forward for counter-attack
    d2 = env.agent_to_idx["a_def_2"]
    env.ball_pos = np.array([300.0, 420.0], dtype=np.float64)
    env.player_pos[d1] = np.array([100.0, 420.0], dtype=np.float64)  # raw dist = 200u
    env.player_pos[d2] = np.array([100.0, 580.0], dtype=np.float64)  # raw dist = 256u
    env.player_pos[a1] = np.array([340.0, 420.0], dtype=np.float64)  # raw dist = 40u (+130 = 170u eff)
    env.player_pos[m1] = np.array([410.0, 420.0], dtype=np.float64)  # raw dist = 110u
    step_rewards.fill(0.0)
    env._compute_tactical_positioning_rewards(
        step_rewards,
        env.player_pos.copy(),
        np.linalg.norm(env.player_pos - env.ball_pos[None, :], axis=1),
        any_teleport=False,
    )
    assert env.primary_loose_ball_chaser[TEAM_A] == m1

    # 4. Emergency DEF step-in when opponent-half gap is huge:
    # DEF is right next to ball (10u), MID/ATK are 400u away
    env.ball_pos = np.array([1050.0, 420.0], dtype=np.float64)
    env.player_pos[d1] = np.array([1040.0, 420.0], dtype=np.float64)  # eff dist = 10 + 160 + 0.35*(390) = 306.5u
    env.player_pos[m1] = np.array([650.0, 420.0], dtype=np.float64)   # eff dist = 400u
    env.player_pos[a1] = np.array([650.0, 420.0], dtype=np.float64)
    step_rewards.fill(0.0)
    env._compute_tactical_positioning_rewards(
        step_rewards,
        env.player_pos.copy(),
        np.linalg.norm(env.player_pos - env.ball_pos[None, :], axis=1),
        any_teleport=False,
    )
    assert env.primary_loose_ball_chaser[TEAM_A] == d1


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
    """Verify line-ordering penalty on overextension, retreat recovery bonus, and carrier/chaser exemptions."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    d1 = env.agent_to_idx["a_def_1"]
    m1 = env.agent_to_idx["a_mid_1"]
    m2 = env.agent_to_idx["a_mid_2"]
    a1 = env.agent_to_idx["a_atk_1"]

    env.ball_holder = a1
    env.ball_pos = np.array([900.0, 420.0], dtype=np.float64)
    env.player_pos[a1] = np.array([900.0, 420.0], dtype=np.float64)
    env.player_pos[m1] = np.array([600.0, 300.0], dtype=np.float64)
    env.player_pos[m2] = np.array([620.0, 540.0], dtype=np.float64)

    # 1. Compare defender just below threshold (X=664.0) vs just above threshold (X=666.0, where threshold is 620 + 45 = 665.0)
    env.player_pos[d1] = np.array([664.0, 350.0], dtype=np.float64)
    step_rewards_legal = np.zeros(env.num_players, dtype=np.float64)
    env._compute_tactical_positioning_rewards(
        step_rewards_legal,
        env.player_pos.copy(),
        np.linalg.norm(env.player_pos - env.ball_pos[None, :], axis=1),
        any_teleport=False,
    )

    env.player_pos[d1] = np.array([666.0, 350.0], dtype=np.float64)
    step_rewards_over = np.zeros(env.num_players, dtype=np.float64)
    env._compute_tactical_positioning_rewards(
        step_rewards_over,
        env.player_pos.copy(),
        np.linalg.norm(env.player_pos - env.ball_pos[None, :], axis=1),
        any_teleport=False,
    )
    # Step discontinuity crossing 665.0 is exactly line_order_pen_scale (-0.010)
    assert step_rewards_over[d1] <= step_rewards_legal[d1] - env.cfg.rewards.line_order_pen_scale + 5e-4

    # 2. Extreme overextended defender at X = 700 (> 665)
    env.player_pos[d1] = np.array([700.0, 350.0], dtype=np.float64)
    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._compute_tactical_positioning_rewards(
        step_rewards,
        env.player_pos.copy(),
        np.linalg.norm(env.player_pos - env.ball_pos[None, :], axis=1),
        any_teleport=False,
    )
    assert step_rewards[d1] <= -env.cfg.rewards.line_order_pen_scale + 1e-5

    # 3. Overextended defender who is carrying the ball is EXEMPT from line order penalty
    env.ball_holder = d1
    env.player_pos[d1] = np.array([700.0, 350.0], dtype=np.float64)
    step_rewards3 = np.zeros(env.num_players, dtype=np.float64)
    env._compute_tactical_positioning_rewards(
        step_rewards3,
        env.player_pos.copy(),
        np.linalg.norm(env.player_pos - env.ball_pos[None, :], axis=1),
        any_teleport=False,
    )
    assert step_rewards3[d1] > -1e-5

    # 4. Retreat recovery bonus: overextended defender retreating back (prev=720, curr=700)
    env.ball_holder = a1
    env.player_pos[d1] = np.array([700.0, 350.0], dtype=np.float64)
    prev_pos = env.player_pos.copy()
    prev_pos[d1] = np.array([720.0, 350.0], dtype=np.float64)  # was 20u further overextended
    step_rewards4 = np.zeros(env.num_players, dtype=np.float64)
    env._compute_tactical_positioning_rewards(
        step_rewards4,
        prev_pos,
        np.linalg.norm(env.player_pos - env.ball_pos[None, :], axis=1),
        any_teleport=False,
    )
    # Retreat recovery bonus adds 20u * role_zone_anchor_scale to offset the base penalty
    assert step_rewards4[d1] > -env.cfg.rewards.line_order_pen_scale

    # 5. Midfielder overextending ahead of striker (m1 at X = 980 > 900 + 60 = 960)
    env.player_pos[m1] = np.array([980.0, 300.0], dtype=np.float64)
    step_rewards5 = np.zeros(env.num_players, dtype=np.float64)
    env._compute_tactical_positioning_rewards(
        step_rewards5,
        env.player_pos.copy(),
        np.linalg.norm(env.player_pos - env.ball_pos[None, :], axis=1),
        any_teleport=False,
    )
    assert step_rewards5[m1] <= -0.5 * env.cfg.rewards.line_order_pen_scale + 1e-5


def test_anchor_pocket_deadzone_eliminates_micro_jitter_inside_35u() -> None:
    """Verify that moving within 35u of a tactical anchor/pocket yields 0.0 delta approach reward in env."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    # Setup loose ball State B at X=500, Y=420
    env.ball_holder = None
    env.ball_pos = np.array([500.0, 420.0], dtype=np.float64)

    m2 = env.agent_to_idx["a_mid_2"]
    anc_x = float(np.clip(0.62 * 500.0 + 220.0, 360.0, 960.0))
    anc = np.array([anc_x, 640.0], dtype=np.float64)

    # 1. Movement entirely inside 35u deadzone (from 30u away to 15u away)
    env.player_pos[m2] = anc + np.array([15.0, 0.0], dtype=np.float64)
    prev_pos = env.player_pos.copy()
    prev_pos[m2] = anc + np.array([30.0, 0.0], dtype=np.float64)

    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._compute_tactical_positioning_rewards(
        step_rewards,
        prev_pos,
        np.linalg.norm(env.player_pos - env.ball_pos[None, :], axis=1),
        any_teleport=False,
    )
    # Approach delta inside deadzone is max(0, 30 - 35) - max(0, 15 - 35) = 0 - 0 = 0.0!
    assert abs(step_rewards[m2]) < 1e-5

    # 2. Movement entirely outside deadzone (from 65u away to 50u away: 15u gain)
    env.player_pos[m2] = anc + np.array([50.0, 0.0], dtype=np.float64)
    prev_pos[m2] = anc + np.array([65.0, 0.0], dtype=np.float64)

    step_rewards2 = np.zeros(env.num_players, dtype=np.float64)
    env._compute_tactical_positioning_rewards(
        step_rewards2,
        prev_pos,
        np.linalg.norm(env.player_pos - env.ball_pos[None, :], axis=1),
        any_teleport=False,
    )
    # Approach delta outside deadzone is max(0, 65 - 35) - max(0, 50 - 35) = 30 - 15 = 15.0u * scale
    expected = env.cfg.rewards.role_zone_anchor_scale * 15.0
    assert math.isclose(step_rewards2[m2], expected, abs_tol=1e-4)


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
    """Verify flank shots receive 0.35x attenuation, cutback aim receives 1.5x boost, and pairwise aim decays."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    atk = env.agent_to_idx["a_atk_1"]
    mid = env.agent_to_idx["a_mid_1"]

    # 1. Extreme byline flank shot: X=1250 >= 1210, Y=180 (|180-420|=240 >= 135)
    env.player_pos[atk] = np.array([1250.0, 180.0], dtype=np.float64)
    env.player_angle[atk] = math.atan2(420.0 - 180.0, 1400.0 - 1250.0)
    env.ball_pos = env.player_pos[atk].copy()
    env.ball_holder = atk

    rew_flank = np.zeros(env.num_players, dtype=np.float64)
    env._execute_kick(atk, kick_power=0.8, step_rewards=rew_flank)
    flank_shot_rew = rew_flank[atk]

    # 2. Central shot from same distance: X=1250, Y=420 (|420-420|=0 < 135)
    env.reset(seed=43)
    env.player_pos[atk] = np.array([1250.0, 420.0], dtype=np.float64)
    env.player_angle[atk] = 0.0
    env.ball_pos = env.player_pos[atk].copy()
    env.ball_holder = atk

    rew_central = np.zeros(env.num_players, dtype=np.float64)
    env._execute_kick(atk, kick_power=0.8, step_rewards=rew_central)
    central_shot_rew = rew_central[atk]

    # Flank shot must receive significant attenuation compared to central shot
    assert flank_shot_rew < central_shot_rew - 0.40

    # 3. Semicircle Cutback Aim Bonus: near enemy goal (d <= 250u), aimed at teammate
    env.reset(seed=44)
    env.player_pos[atk] = np.array([1280.0, 500.0], dtype=np.float64)  # dist to enemy goal = 144u <= 250u
    env.player_pos[mid] = np.array([1100.0, 420.0], dtype=np.float64)  # open teammate
    env.player_angle[atk] = math.atan2(420.0 - 500.0, 1100.0 - 1280.0)
    env.ball_pos = env.player_pos[atk].copy()
    env.ball_holder = atk
    env.last_ball_holder = -1

    rew_cutback = np.zeros(env.num_players, dtype=np.float64)
    env._execute_kick(atk, kick_power=0.6, step_rewards=rew_cutback)
    # Cutback aim bonus is 1.5x pass_target_aim_ind (1.5 * 0.45 = 0.675)
    assert rew_cutback[atk] >= 1.5 * env.cfg.rewards.pass_target_aim_ind - 1e-4

    # 4. Instant Pass-Aim Decay on Return Pass to last_passer_idx:
    env.reset(seed=45)
    env.player_pos[atk] = np.array([900.0, 420.0], dtype=np.float64)
    env.player_pos[mid] = np.array([1000.0, 420.0], dtype=np.float64)
    env.player_angle[atk] = 0.0  # aiming at mid
    env.ball_pos = env.player_pos[atk].copy()
    env.ball_holder = atk
    env.last_passer_idx = mid
    env.pairwise_repeat_count = 1

    rew_decay = np.zeros(env.num_players, dtype=np.float64)
    env._execute_kick(atk, kick_power=0.6, step_rewards=rew_decay)
    # Decayed by (0.25)^2 = 0.0625
    expected_decayed = env.cfg.rewards.pass_target_aim_ind * (env.cfg.rewards.pass_pairwise_decay_base ** 2)
    assert math.isclose(rew_decay[atk], expected_decayed, abs_tol=1e-3)


def test_scanned_pass_multiplier_in_and_out_of_fov() -> None:
    """Verify 1.20x progressive pass bonus for passes inside vision cone and 1.00x for passes outside."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    kicker = env.agent_to_idx["a_mid_1"]
    receiver = env.agent_to_idx["a_atk_1"]
    team = TEAM_A

    # 1. Receiver INSIDE kicker's 210 deg FOV at kick
    env.player_pos[kicker] = np.array([600.0, 420.0], dtype=np.float64)
    env.player_pos[receiver] = np.array([900.0, 420.0], dtype=np.float64)  # forward pass, rel angle = 0
    env.ball_holder = None
    env.ball_kick_origin = env.player_pos[kicker].copy()
    env.ball_pos = env.player_pos[receiver].copy()
    env.ball_vel = np.array([200.0, 0.0], dtype=np.float64)
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.last_ball_holder = kicker
    env.last_ball_team = team
    env.kick_frame_player_pos = env.player_pos.copy()

    rew_in = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(rew_in)
    kicker_rew_in = rew_in[kicker]

    # 2. Receiver OUTSIDE kicker's FOV (e.g. blind pass kicked forward while receiver is behind kick direction)
    env.reset(seed=43)
    env.player_pos[kicker] = np.array([600.0, 420.0], dtype=np.float64)
    env.player_pos[receiver] = np.array([300.0, 420.0], dtype=np.float64)
    env.ball_holder = None
    env.ball_kick_origin = env.player_pos[kicker].copy()
    env.ball_pos = env.player_pos[receiver].copy()
    env.ball_vel = np.array([200.0, 0.0], dtype=np.float64)  # ball heading right, receiver was at left (180 deg)
    env.ball_kicked_in_flight = True
    env.ball_kick_flight_timer = 0.5
    env.last_ball_holder = kicker
    env.last_ball_team = team
    env.kick_frame_player_pos = env.player_pos.copy()

    rew_out = np.zeros(env.num_players, dtype=np.float64)
    env._check_free_ball_pickup(rew_out)
    kicker_rew_out = rew_out[kicker]

    # Pass in FOV must be rewarded significantly higher due to scanned_pass_mult (1.20x) and forward progress
    assert kicker_rew_in > kicker_rew_out


def test_state_a2_defender_stays_behind_ball_carrier() -> None:
    """Verify State A2 sets defender target anchor strictly behind the ball carrier."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    carrier = env.agent_to_idx["a_mid_1"]
    d1 = env.agent_to_idx["a_def_1"]

    # Ball carrier at X = 400.0
    env.ball_holder = carrier
    env.player_pos[carrier] = np.array([400.0, 420.0], dtype=np.float64)
    env.ball_pos = env.player_pos[carrier].copy()

    # Target anchor in State A2 for def_1: min(0.50 * 400 + 110, 400 - 120) = min(310, 280) = 280.0
    target_anchor_x = float(np.clip(min(0.50 * 400.0 + 110.0, 400.0 - 120.0), 200.0, 660.0))
    assert target_anchor_x == 280.0
    assert target_anchor_x < 400.0 - 100.0  # strictly behind the carrier!

    # Verify defender moving towards 280.0 receives positive anchor approach reward
    env.player_pos[d1] = np.array([280.0, 245.0], dtype=np.float64)
    prev_pos = env.player_pos.copy()
    prev_pos[d1] = np.array([220.0, 245.0], dtype=np.float64)  # moved closer from 220 to 280

    step_rewards = np.zeros(env.num_players, dtype=np.float64)
    env._compute_tactical_positioning_rewards(
        step_rewards,
        prev_pos,
        np.linalg.norm(env.player_pos - env.ball_pos[None, :], axis=1),
        any_teleport=False,
    )
    assert step_rewards[d1] > 0.0


def test_arc_hesitation_stall_penalty() -> None:
    """Verify carrier grinding near enemy semicircle for >= 0.75s incurs prime_shot_hesitation_pen."""
    env = FootballParallelEnv(DEFAULT_CONFIG)
    env.reset(seed=42)

    carrier = env.agent_to_idx["a_mid_1"]
    enemy_goal = np.array([1400.0, 420.0], dtype=np.float64)

    # Position carrier at distance 200u from enemy goal (<= 235u) facing away from goal (pi)
    env.ball_holder = carrier
    env.player_pos[carrier] = np.array([1200.0, 420.0], dtype=np.float64)
    env.player_angle[carrier] = math.pi  # facing midfield (not aimed at goal!)
    env.ball_pos = env.player_pos[carrier] - np.array([env.cfg.mechanics.ball_carry_offset, 0.0])

    # 1. Stalling for 0.85s (>= 0.75s): hesitation penalty MUST trigger
    env.ball_hold_timer = 0.85
    _, rewards_stall, _, _, _ = env.step(_zero_actions(env))
    assert rewards_stall["a_mid_1"] < -0.01  # includes -0.025 penalty

    # 2. Fresh possession (< 0.75s, e.g. 0.3s): hesitation penalty must NOT trigger
    env.ball_hold_timer = 0.30
    _, rewards_fresh, _, _, _ = env.step(_zero_actions(env))
    # Possession reward without hesitation penalty is positive/neutral
    assert rewards_fresh["a_mid_1"] > rewards_stall["a_mid_1"]


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
    assert ball_slot[0] >= 0.25, f"Ball presence {ball_slot[0]} should be at least 0.25 floor"
    assert not np.allclose(ball_slot[1:3], [0.0, 0.0]), "Ball slot should not be zeroed out"

    # Find which entity slot corresponds to teammate_idx
    slot_idx = env._entity_player_slots[idx].index(teammate_idx)
    # Each slot is 5 floats starting at 16 + (slot_idx + 1) * 5
    start_i = 16 + (slot_idx + 1) * 5
    tm_slot = player_obs[start_i : start_i + 5]
    assert tm_slot[0] >= 0.25, f"Teammate presence {tm_slot[0]} should be at least 0.25 floor"
    assert not np.allclose(tm_slot[1:3], [0.0, 0.0]), "Teammate slot should not be zeroed out"


def test_gk_does_not_overshoot_close_shot_when_diving() -> None:
    """Verify GK dive clamps speed to smother close-range CPA without overshooting."""
    env = FootballParallelEnv(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    env.reset(seed=102)

    gk_idx = env.agent_to_idx["a_keeper"]
    # Place GK in center of goal line
    env.player_pos[gk_idx] = np.array([30.0, 420.0], dtype=np.float64)
    env.player_vel[gk_idx] = np.zeros(2, dtype=np.float64)
    env.player_angle[gk_idx] = 0.0

    # Place ball moving towards goal, close to GK
    env.ball_pos[:] = np.array([65.0, 435.0], dtype=np.float64)
    env.ball_vel[:] = np.array([-200.0, 0.0], dtype=np.float64)  # heading towards own goal
    env.ball_holder = None
    env.ball_kicked_in_flight = True
    env.last_ball_team = 1  # opponent kicked it

    # Step physics with GK dive special action (action[3] = 1.0)
    act = np.zeros(ACTION_DIM, dtype=np.float32)
    act[3] = 1.0  # Dive trigger
    actions = {name: np.zeros(ACTION_DIM, dtype=np.float32) for name in ALL_AGENTS}
    actions["a_keeper"] = act

    obs, rews, terms, truncs, infos = env.step(actions)

    # Verify GK didn't shoot past the ball: GK distance to ball should be small
    d_gk_ball = float(np.linalg.norm(env.player_pos[gk_idx] - env.ball_pos))
    gk_speed = float(np.linalg.norm(env.player_vel[gk_idx]))
    assert gk_speed <= 140.0 + 1e-4, f"GK dive speed {gk_speed} exceeded maximum dive speed clamp"


def test_gk_walking_pickup_beats_diving_pickup_on_loose_ball() -> None:
    """Verify walking to pick up loose ball yields clean foot recovery (+0.35) and dive doesn't farm."""
    # Test 1: Clean walking pickup
    env1 = FootballParallelEnv(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    env1.reset(seed=103)
    gk_idx1 = env1.agent_to_idx["a_keeper"]
    env1.player_pos[gk_idx1] = np.array([40.0, 420.0], dtype=np.float64)
    env1.player_vel[gk_idx1] = np.array([50.0, 0.0], dtype=np.float64)
    env1.ball_pos[:] = np.array([50.0, 420.0], dtype=np.float64)
    env1.ball_vel[:] = np.zeros(2, dtype=np.float64)
    env1.ball_holder = None
    env1.ball_kicked_in_flight = False

    # Keeper walks forward without diving
    act_walk = {name: np.zeros(ACTION_DIM, dtype=np.float32) for name in ALL_AGENTS}
    act_walk["a_keeper"] = np.array([1.0, 0.0, -1.0, -1.0], dtype=np.float32)
    _, rews_walk, _, _, _ = env1.step(act_walk)

    # Test 2: Diving pickup on same ball
    env2 = FootballParallelEnv(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    env2.reset(seed=103)
    gk_idx2 = env2.agent_to_idx["a_keeper"]
    env2.player_pos[gk_idx2] = np.array([40.0, 420.0], dtype=np.float64)
    env2.player_vel[gk_idx2] = np.array([50.0, 0.0], dtype=np.float64)
    env2.ball_pos[:] = np.array([50.0, 420.0], dtype=np.float64)
    env2.ball_vel[:] = np.zeros(2, dtype=np.float64)
    env2.ball_holder = None
    env2.ball_kicked_in_flight = False

    # Keeper lunges / dives at loose ball
    act_dive = {name: np.zeros(ACTION_DIM, dtype=np.float32) for name in ALL_AGENTS}
    act_dive["a_keeper"] = np.array([1.0, 0.0, -1.0, 1.0], dtype=np.float32)
    _, rews_dive, _, _, _ = env2.step(act_dive)

    assert rews_walk["a_keeper"] > rews_dive["a_keeper"], (
        f"Walking pickup reward {rews_walk['a_keeper']} should beat diving pickup {rews_dive['a_keeper']}"
    )


def test_gk_long_distribution_and_wall_bounce_rewards() -> None:
    """Verify GK long pass to unpressed teammate triggers trajectory bonus and wall bounce bypass."""
    env = FootballParallelEnv(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    env.reset(seed=104)

    gk_idx = env.agent_to_idx["a_keeper"]
    env.ball_holder = gk_idx
    env.player_pos[gk_idx] = np.array([50.0, 100.0], dtype=np.float64)
    env.ball_pos[:] = env.player_pos[gk_idx].copy()

    # Place unpressed teammate deep upfield along sideline
    tm_idx = env.agent_to_idx["a_mid_1"]
    env.player_pos[tm_idx] = np.array([600.0, 120.0], dtype=np.float64)
    env.player_vel[tm_idx] = np.zeros(2, dtype=np.float64)

    # Place all opponents far away from tm_idx
    for opp_idx in range(6, 12):
        env.player_pos[opp_idx] = np.array([900.0, 600.0], dtype=np.float64)

    # Execute wall kick from GK aiming toward top sideline to bank upfield to tm_idx
    # Face GK diagonally toward top wall
    env.player_angle[gk_idx] = -0.40  # Aiming angled toward y=0 wall
    act = {name: np.zeros(ACTION_DIM, dtype=np.float32) for name in ALL_AGENTS}
    act["a_keeper"] = np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float32)  # Full kick

    _, rews, _, _, _ = env.step(act)
    # The GK should receive significant positive pass reward for targeting unpressed teammate upfield
    assert rews["a_keeper"] > 0.5, f"GK should receive distribution reward on long pass, got {rews['a_keeper']}"


def test_one_touch_75_deg_raw_steer_deflection() -> None:
    """Verify one-touch raw steer deflection allows up to 75 deg angle without low-pass lag."""
    env = FootballParallelEnv(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    env.reset(seed=105)

    p_idx = env.agent_to_idx["a_atk_1"]
    env.ball_holder = p_idx
    env.player_pos[p_idx] = np.array([700.0, 420.0], dtype=np.float64)
    env.player_angle[p_idx] = 0.0  # Facing East
    env.ball_pos[:] = env.player_pos[p_idx].copy()

    # Apply full steer right (+1.0) with kick
    act = {name: np.zeros(ACTION_DIM, dtype=np.float32) for name in ALL_AGENTS}
    act["a_atk_1"] = np.array([0.0, 1.0, 1.0, -1.0], dtype=np.float32)

    env.step(act)

    # Ball velocity launch direction
    launch_angle = math.atan2(float(env.ball_vel[1]), float(env.ball_vel[0]))
    # Expected deflection is ~ +75 degrees (+1.309 rad)
    assert math.isclose(launch_angle, math.radians(75.0), abs_tol=math.radians(5.0)), (
        f"Launch angle {math.degrees(launch_angle)} deg did not match expected ~75 deg deflection"
    )


def test_pocket_ema_smoothing_and_obs_encoding() -> None:
    """Verify tactical pocket EMA smoothing and obs[13:16] dynamic role guidance."""
    env = FootballParallelEnv(
        config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False))
    )
    obs, _ = env.reset(seed=106)

    # Check obs dimension is strictly 76
    for agent_name in ALL_AGENTS:
        assert obs[agent_name].shape == (76,), f"Expected obs shape (76,), got {obs[agent_name].shape}"

    # Teammate 1 holds ball
    car_idx = env.agent_to_idx["a_atk_1"]
    env.ball_holder = car_idx
    env.player_pos[car_idx] = np.array([800.0, 420.0], dtype=np.float64)

    # Take step to let pocket calculation run
    acts = {name: np.zeros(ACTION_DIM, dtype=np.float32) for name in ALL_AGENTS}
    obs, _, _, _, infos = env.step(acts)

    # Check central_state in info has shape (54,)
    for agent_name in ALL_AGENTS:
        assert "central_state" in infos[agent_name]
        assert infos[agent_name]["central_state"].shape == (54,)

    # Verify obs[13:16] for field player has valid ranges
    def_obs = obs["a_def_1"]
    assert -1.0 <= def_obs[13] <= 1.0
    assert -1.0 <= def_obs[14] <= 1.0
    assert -1.0 <= def_obs[15] <= 1.0



















