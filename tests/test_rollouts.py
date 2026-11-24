"""Specification tests for counterfactual rollouts (`tools/audit/rollouts.py`)."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from football_rl.config import DEFAULT_CONFIG  # noqa: E402
from football_rl.env import FootballParallelEnv  # noqa: E402
from tests.test_eval_metrics import (  # noqa: E402
    make_trace,
    quick_spec,
    write_constant_hybrid_checkpoint,
    write_head_checkpoint,
)
from tools.audit import harness  # noqa: E402


def _dir_digest(directory: Path) -> dict[str, str]:
    if not directory.exists():
        return {}
    return {
        str(p.relative_to(directory)): hashlib.md5(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def _make_hybrid_trace_run(tmp_path: Path, *, max_steps: int = 120, n: int = 2, seed: int = 11, z_kick: float = -3.0):
    ckpt = write_constant_hybrid_checkpoint(tmp_path / "ckpt_hybrid", z_kick=z_kick, z_spec=-10.0)
    trace_dir = tmp_path / "traces_hybrid"
    spec = quick_spec(ckpt, opponent="self", mode="sampled", n=n, seed=seed, max_steps=max_steps)
    traces = harness.run_matches(spec)
    harness.save_traces(traces, trace_dir, protected=[ckpt])
    return ckpt, trace_dir, traces


def _make_gaussian_trace_run(tmp_path: Path, *, max_steps: int = 120, n: int = 2, seed: int = 11):
    ckpt = write_head_checkpoint(tmp_path / "ckpt_gauss", {"blue": "gaussian4", "red": "gaussian4"}, seed=3)
    trace_dir = tmp_path / "traces_gauss"
    spec = quick_spec(ckpt, opponent="self", mode="sampled", n=n, seed=seed, max_steps=max_steps)
    traces = harness.run_matches(spec)
    harness.save_traces(traces, trace_dir, protected=[ckpt])
    return ckpt, trace_dir, traces


# ----------------------------------------------------------------------------
# 1. Re-simulation & determinism
# ----------------------------------------------------------------------------
class TestResimulation:
    def test_resimulate_full_match_within_1e_3(self, tmp_path):
        from tools.audit import rollouts

        ckpt = write_head_checkpoint(tmp_path / "ckpt", {"blue": "gaussian4", "red": "gaussian4"}, seed=7)
        spec = quick_spec(ckpt, opponent="self", mode="sampled", n=1, seed=11, max_steps=2700)
        trace = harness.play_match(spec, 0)
        ok, max_err, _ = rollouts.resimulate_match(trace, match_seed=11 * 1000 + 0)
        assert ok is True
        assert max_err < 1e-3

    def test_corrupted_trace_is_skipped_and_counted(self, tmp_path):
        from tools.audit import rollouts

        ckpt, trace_dir, traces = _make_gaussian_trace_run(tmp_path, max_steps=90, n=2, seed=11)
        # Corrupt match 0 player position at step 40 by 0.05 u (> 1e-3 u)
        bad = dict(traces[0])
        bad["pos"] = bad["pos"].copy()
        bad["pos"][40, 0, 0] += 0.05
        np.savez_compressed(trace_dir / "m000.npz", **bad)

        res = rollouts.run_gk(trace_dir=trace_dir, trace_seed=11, workers=1)
        assert res["n_matches_total"] == 2
        assert res["n_matches_ok"] == 1
        assert res["n_matches_skipped"] == 1

    def test_ball_corruption_and_pos_err_limit_boundary(self, tmp_path):
        from tools.audit import rollouts

        _, _, traces = _make_gaussian_trace_run(tmp_path, max_steps=90, n=1, seed=11)
        base = traces[0]
        match_seed = 11 * 1000 + 0

        # Just below POS_ERR_LIMIT (5e-4 < 1e-3) must pass
        near_ok = {k: v.copy() for k, v in base.items()}
        near_ok["pos"][40, 0, 0] += 5e-4
        near_ok["ball"][40, 0] += 5e-4
        ok, max_err, _ = rollouts.resimulate_match(near_ok, match_seed=match_seed)
        assert ok is True
        assert 4e-4 <= max_err < rollouts.POS_ERR_LIMIT

        # Ball corruption above 1e-3 (2e-3) with players untouched must fail
        bad_ball = {k: v.copy() for k, v in base.items()}
        bad_ball["ball"][40, 0] += 2e-3
        ok_ball, err_ball, _ = rollouts.resimulate_match(bad_ball, match_seed=match_seed)
        assert ok_ball is False
        assert err_ball >= rollouts.POS_ERR_LIMIT

        # Player corruption just above 1e-3 (2e-3) must fail
        bad_pos = {k: v.copy() for k, v in base.items()}
        bad_pos["pos"][40, 0, 0] += 2e-3
        ok_pos, err_pos, _ = rollouts.resimulate_match(bad_pos, match_seed=match_seed)
        assert ok_pos is False
        assert err_pos >= rollouts.POS_ERR_LIMIT

    def test_hold_skips_corrupted_trace_and_counts(self, tmp_path):
        from tools.audit import rollouts

        ckpt, trace_dir, traces = _make_hybrid_trace_run(tmp_path, max_steps=90, n=2, seed=11, z_kick=-2.0)
        bad = dict(traces[0])
        bad["ball"] = bad["ball"].copy()
        bad["ball"][40, 0] += 0.05
        np.savez_compressed(trace_dir / "m000.npz", **bad)

        res = rollouts.run_hold(
            trace_dir=trace_dir,
            checkpoint_dir=ckpt,
            trace_seed=11,
            seed=7,
            workers=1,
        )
        assert res["n_matches_total"] == 2
        assert res["n_matches_ok"] == 1
        assert res["n_matches_skipped"] == 1


# ----------------------------------------------------------------------------
# 2. GK opportunities and variants
# ----------------------------------------------------------------------------
class TestGkOpportunities:
    def test_opportunity_filtering_boundaries(self):
        from tools.audit import rollouts

        tr = make_trace(160)
        tr["ball"][:] = (100.0, 420.0)
        tr["pos"][:, 0] = (20.0, 420.0)     # dist 80 u from ball, ball_goal 100 <= 195
        tr["bvel"][:] = (79.9, 0.0)         # < 80 u/s

        # Step 0: valid for team 0. Step 44: < 45 steps spacing -> skip. Step 45: valid.
        # Put ball at dist 39.99 at step 90 -> skip; dist 140.01 at step 91 -> skip;
        # speed 80.0 at step 92 -> skip; outside box (195.1) at step 93 -> skip;
        # held at step 94 -> skip; step 95 valid; step 150 has < 15 steps left -> skip.
        tr["pos"][90, 0] = (60.01, 420.0)   # dist 39.99
        tr["pos"][91, 0] = (100.0 - 140.01, 420.0)  # dist 140.01 (negative x)
        tr["bvel"][92] = (80.0, 0.0)
        tr["ball"][93] = (195.1, 420.0)
        tr["pos"][93, 0] = (95.1, 420.0)
        tr["holder"][94] = 1
        # Mask out steps between 0..44 and 46..94 and 96..159 by putting ball far
        keep = {0, 44, 45, 90, 91, 92, 93, 94, 95, 150}
        for s in range(160):
            if s not in keep:
                tr["ball"][s] = (700.0, 420.0)
        tr["ball"][150] = (100.0, 420.0)
        tr["pos"][150, 0] = (20.0, 420.0)

        opps = rollouts.find_gk_opportunities(tr)
        assert opps == [(0, 0), (45, 0), (95, 0)]

    def test_scripted_go_to_ball_action_hand_values(self):
        from tools.audit import rollouts

        env = FootballParallelEnv(config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False)))
        env.reset(seed=0)
        env.player_pos[0] = (40.0, 420.0)
        env.ball_pos[:] = (120.0, 420.0)     # bearing 0 rad

        env.player_angle[0] = 0.0
        np.testing.assert_allclose(rollouts.gk_scripted_action(env, 0), [1.0, 0.0, -1.0, -1.0], atol=1e-6)

        env.player_angle[0] = -0.3           # error +0.3 -> steer +0.6
        np.testing.assert_allclose(rollouts.gk_scripted_action(env, 0), [1.0, 0.6, -1.0, -1.0], atol=1e-6)

        env.player_angle[0] = -0.8           # error +0.8 -> clipped to +1.0
        np.testing.assert_allclose(rollouts.gk_scripted_action(env, 0), [1.0, 1.0, -1.0, -1.0], atol=1e-6)

        env.player_angle[0] = 0.8            # error -0.8 -> clipped to -1.0
        np.testing.assert_allclose(rollouts.gk_scripted_action(env, 0), [1.0, -1.0, -1.0, -1.0], atol=1e-6)

    def test_gk_scripted_beats_inert_recorded_on_hand_scene(self, tmp_path):
        from tools.audit import rollouts

        env = FootballParallelEnv(config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False)))
        env.reset(seed=5)
        # Put ball loose 65 u in front of GK 0, all other players far away; GK faces sideways.
        env.ball_holder = None
        env.ball_pos[:] = (110.0, 420.0)
        env.ball_vel[:] = (0.0, 0.0)
        env.player_pos[0] = (45.0, 420.0)
        env.player_angle[0] = math.pi / 2.0
        env.player_vel[:] = 0.0
        for idx in range(1, 12):
            env.player_pos[idx] = (700.0 + 30.0 * (idx % 6), 80.0 if idx < 6 else 760.0)

        snap = copy.deepcopy(env)
        # Recorded actions: everyone inert for 15 steps
        rec_acts = np.tile(np.array([0.0, 0.0, -1.0, -1.0], dtype=np.float32), (15, 12, 1))
        out = rollouts.eval_gk_opportunity(snap, team=0, recorded_actions=rec_acts)
        assert out["recorded"]["pickup"] == 0.0
        assert out["inert"]["pickup"] == 0.0
        assert out["scripted"]["pickup"] == 1.0
        assert out["diff_scripted_minus_recorded"]["pickup"] == 1.0

    def test_gk_runs_on_hybrid5_trace_with_opportunity(self, tmp_path):
        from tools.audit import rollouts

        ckpt = write_constant_hybrid_checkpoint(tmp_path / "ckpt_hybrid", z_kick=0.0, z_spec=-10.0)
        trace_dir = tmp_path / "traces_hybrid"
        spec = quick_spec(ckpt, opponent="scripted", mode="sampled", n=2, seed=11, max_steps=300)
        traces = harness.run_matches(spec)
        assert traces[0]["head"].tolist() == [1, 0]
        harness.save_traces(traces, trace_dir, protected=[ckpt])

        out_path = tmp_path / "gk_hybrid.json"
        rc = rollouts.main(["gk", "--trace-dir", str(trace_dir), "--trace-seed", "11", "--workers", "1", "--out", str(out_path)])
        assert rc == 0
        payload = json.loads(out_path.read_text())
        assert payload["n_matches_ok"] == 2
        assert payload["n_matches_skipped"] == 0
        assert payload["n_opportunities"] >= 1
        for group in ("recorded", "scripted", "inert", "diff_scripted_minus_recorded"):
            for metric in ("pickup", "gk_reward", "team_reward"):
                entry = payload["variants"][group][metric]
                assert entry["n"] == payload["n_opportunities"]
                assert math.isfinite(entry["mean"])


# ----------------------------------------------------------------------------
# 3. HOLD decision points, splits, hand test, identical-branch identity & determinism
# ----------------------------------------------------------------------------
class TestHoldRollouts:
    def test_hold_decision_selection_and_splits(self):
        from tools.audit import rollouts

        tr = make_trace(120)
        tr["hold_t"] = np.zeros(120, dtype=np.float64)
        # Step 0: GK holding -> skip
        tr["holder"][0] = 0
        tr["hold_t"][0] = 0.5
        # Step 1: hold_t 0.079 < 0.08 -> skip
        tr["holder"][1] = 5
        tr["hold_t"][1] = 0.079
        # Step 2: valid! Team 0 player 5 at (300, 420), opp at (360, 420) -> dist 60 < 70, defensive third (ball x=300 < 466.67)
        tr["holder"][2] = 5
        tr["hold_t"][2] = 0.08
        tr["pos"][2, 5] = (300.0, 420.0)
        tr["ball"][2] = (300.0, 420.0)
        tr["pos"][2, 7] = (360.0, 420.0)
        # Step 46: only 44 steps after 2 -> skip
        tr["holder"][46] = 5
        tr["hold_t"][46] = 0.2
        # Step 47: valid! Team 1 player 11, side[47, 1] == 1 (attacks left, so canonical_x = 1400 - x).
        # Put ball at x=300 -> canonical_x = 1100 >= 933.33 -> attacking third; opp at dist 120 -> >=120 bin
        tr["holder"][47] = 11
        tr["hold_t"][47] = 0.15
        tr["pos"][47, 11] = (300.0, 420.0)
        tr["ball"][47] = (300.0, 420.0)
        tr["pos"][47, 1] = (420.0, 420.0)

        pts = rollouts.find_hold_decisions(tr)
        assert [p["step"] for p in pts] == [2, 47]
        assert pts[0]["pressure"] == "lt70" and pts[0]["third"] == "defensive"
        assert pts[1]["pressure"] == "ge120" and pts[1]["third"] == "attacking"

    def test_hand_two_player_kick_and_hold_physics(self, tmp_path):
        from tools.audit import rollouts

        # Hand-derived test on a two-player scene (a_atk_1 vs deterministic opponent b_atk_1)
        # Ball held by a_atk_1 (slot 5, ROLE_ATK, max_kick_spd=640) at (700, 420), angle=0.
        # Opponent b_atk_1 at (1200, 420), all others out of play.
        ckpt_no_kick = write_constant_hybrid_checkpoint(tmp_path / "c_nokick", z_kick=-10.0, z_spec=-10.0)
        ckpt_kick = write_constant_hybrid_checkpoint(tmp_path / "c_kick", z_kick=+10.0, z_spec=-10.0)

        def make_scene():
            env = FootballParallelEnv(config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False)))
            env.reset(seed=0)
            env.player_pos[5] = (700.0, 420.0)
            env.player_angle[5] = 0.0
            env.player_vel[:] = 0.0
            env.ball_holder = 5
            env.ball_hold_timer = 0.10
            env.kick_grace_timer = 0.0
            env.ball_pos[:] = (724.0, 420.0)
            env.player_pos[11] = (1200.0, 420.0)
            for idx in range(12):
                if idx not in (5, 11):
                    env.player_pos[idx] = (100.0 + 40.0 * (idx % 6), 40.0 if idx < 6 else 800.0)
            return env

        # 1) KICK branch: policy naturally does not kick (z_kick=-10 -> p=0.005), forced KICK (+1.0) on step 0
        #    launches ball at 640 u/s; after 1 RL step (4 substeps @ 1/60 s, decel 155 u/s^2):
        #    speed = 640 - 4*(155/60) = 629.6666667 u/s along steer deflection tanh(-0.2)*35 deg.
        pol_nk = (harness.TeamPolicy(str(ckpt_no_kick), "blue"), harness.TeamPolicy(str(ckpt_no_kick), "red"))
        kick_state = {}

        def _track_kick(step_k: int, cur_env) -> None:
            if step_k == 0:
                kick_state["holder"] = cur_env.ball_holder
                kick_state["vel"] = cur_env.ball_vel.copy()

        rollouts.roll_hold_branches(
            make_scene(),
            holder_player=5,
            team_a=pol_nk[0],
            team_b=pol_nk[1],
            branch_seed=42,
            sigma_scale=0.0,
            horizons=(15,),
            after_kick_step=_track_kick,
        )
        assert kick_state["holder"] is None
        steer_rad = math.tanh(-0.2) * math.radians(35.0)
        expected_spd = 640.0 - 4.0 * 155.0 / 60.0
        np.testing.assert_allclose(
            kick_state["vel"],
            [expected_spd * math.cos(steer_rad), expected_spd * math.sin(steer_rad)],
            atol=1e-5,
        )

        # 2) HOLD branch: policy naturally kicks (z_kick=+10 -> p=0.995) with sigma_scale=0,
        #    forcing HOLD (act[5, 2] = -1.0) for 15 decisions keeps ball_holder == 5 for all 15 steps.
        pol_k = (harness.TeamPolicy(str(ckpt_kick), "blue"), harness.TeamPolicy(str(ckpt_kick), "red"))
        holders_seen = []

        def _track_hold(step_k: int, cur_env) -> None:
            holders_seen.append(cur_env.ball_holder)

        res = rollouts.roll_hold_branches(
            make_scene(),
            holder_player=5,
            team_a=pol_k[0],
            team_b=pol_k[1],
            branch_seed=99,
            sigma_scale=0.0,
            horizons=(15,),
            hold_steps=15,
            after_hold_step=_track_hold,
        )
        assert holders_seen == [5] * 15
        assert res["hold"][15]["p_team_ball_1s"] == 1.0
        assert res["kick"][15]["p_team_ball_1s"] == 0.0

    def test_identical_branches_paired_diff_is_exactly_zero_hybrid5(self, tmp_path):
        from tools.audit import rollouts

        ckpt, _, _ = _make_hybrid_trace_run(tmp_path, max_steps=60, n=1, seed=11, z_kick=0.0)
        team_a = harness.TeamPolicy(str(ckpt), "blue")
        team_b = harness.TeamPolicy(str(ckpt), "red")

        env = FootballParallelEnv(config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False)))
        env.reset(seed=11000)
        for _ in range(3):
            env.step_array(np.zeros((12, 4), dtype=np.float32))
        holder = env.ball_holder
        assert holder is not None and holder % 6 != 0

        res = rollouts.roll_hold_branches(
            env,
            holder_player=holder,
            team_a=team_a,
            team_b=team_b,
            branch_seed=12345,
            sigma_scale=1.0,
            horizons=(75, 225),
            identical_override=True,
        )
        for h in (75, 225):
            for key, val in res["diff"][h].items():
                if isinstance(val, dict):
                    for ck, cv in val.items():
                        assert cv == 0.0, f"horizon {h} component {ck} diff was {cv}"
                else:
                    assert val == 0.0, f"horizon {h} metric {key} diff was {val}"

    def test_non_shaping_fallback_and_bootstrap_formula(self):
        from tools.audit import rollouts

        # Stub env without last_shaping_rewards (uses last_rew_breakdown's r_sparse column)
        class _StubEnv:
            def __init__(self):
                self.cfg = DEFAULT_CONFIG
                self.step_i = 0
                self.ball_holder = 5
                self.last_ball_holder = 5
                self.last_ball_team = 0
                self.last_turnover_reward_time = -999.0
                self.shot_on_target_cd = np.zeros(12)
                self.tackle_cooldown = np.zeros(12)
                self.stun_timer = np.zeros(12)
                self.stats = {
                    "score": [0, 0], "own_goals": 0, "team_assists": [0, 0],
                    "field_5s_violations": 0, "keeper_5s_violations": 0,
                    "failed_tackles": 0, "shots": [0, 0], "team_passes": [0, 0], "team_saves": [0, 0],
                }
                self.last_rew_breakdown = [{"r_sparse": 0.0} for _ in range(12)]

            def _get_observations_array(self):
                return np.zeros((12, 82), dtype=np.float32)

            def get_centralized_state(self):
                return np.zeros((12, 54), dtype=np.float32)

            def step_array(self, act, val=None):
                self.step_i += 1
                rew = np.full(12, 0.25, dtype=np.float32)
                self.last_rew_breakdown = [{"r_sparse": 0.10} for _ in range(12)]
                return self._get_observations_array(), self.get_centralized_state(), rew, False, None

        class _StubPol:
            def act(self, obs, cen, mode, sigma_scale, gen):
                return np.zeros((6, 4), dtype=np.float32), np.full(6, 2.0, dtype=np.float32), np.zeros((6, 4), dtype=np.float32)

        gamma = float(DEFAULT_CONFIG.ppo.gamma)
        out = rollouts.roll_hold_branches(
            _StubEnv(),
            holder_player=5,
            team_a=_StubPol(),
            team_b=_StubPol(),
            branch_seed=7,
            horizons=(3,),
            kick_steps=1,
            hold_steps=2,
        )
        expected_ns = sum((gamma ** k) * 0.10 for k in range(3))
        expected_full_boot = sum((gamma ** k) * 0.25 for k in range(3)) + (gamma ** 3) * 2.0
        assert out["kick"][3]["non_shaping_return"] == pytest.approx(expected_ns, rel=1e-6)
        assert out["kick"][3]["full_return_bootstrap"] == pytest.approx(expected_full_boot, rel=1e-6)

    def test_hold_report_deterministic_for_fixed_seed_and_no_write(self, tmp_path):
        from tools.audit import rollouts

        ckpt, trace_dir, _ = _make_hybrid_trace_run(tmp_path, max_steps=120, n=2, seed=11, z_kick=-2.0)
        ckpt_digest = _dir_digest(ckpt)
        trace_digest = _dir_digest(trace_dir)

        out1 = tmp_path / "hold1.json"
        out2 = tmp_path / "hold2.json"
        assert rollouts.main([
            "hold", "--trace-dir", str(trace_dir), "--checkpoint-dir", str(ckpt),
            "--trace-seed", "11", "--seed", "77", "--workers", "1", "--out", str(out1),
        ]) == 0
        assert rollouts.main([
            "hold", "--trace-dir", str(trace_dir), "--checkpoint-dir", str(ckpt),
            "--trace-seed", "11", "--seed", "77", "--workers", "2", "--out", str(out2),
        ]) == 0

        d1 = json.loads(out1.read_text())
        d2 = json.loads(out2.read_text())
        assert d1 == d2
        assert d1["n_matches_ok"] == 2
        assert _dir_digest(ckpt) == ckpt_digest
        assert _dir_digest(trace_dir) == trace_digest

        # Component attribution sum matches non_shaping_return
        for horizon_key in ("75", "225"):
            overall = d1["horizons"][horizon_key]["overall"]
            comp_sum = sum(overall["components"][k]["mean"] for k in overall["components"] if overall["components"][k]["mean"] is not None)
            assert comp_sum == pytest.approx(overall["non_shaping_return"]["mean"], abs=1e-5)

        # Protected output paths must raise
        with pytest.raises(ValueError):
            rollouts.main([
                "hold", "--trace-dir", str(trace_dir), "--checkpoint-dir", str(ckpt),
                "--out", str(trace_dir / "bad.json"),
            ])
        with pytest.raises(ValueError):
            rollouts.main([
                "gk", "--trace-dir", str(trace_dir), "--out", str(REPO_ROOT / "replays" / "bad.json"),
            ])
        with pytest.raises(ValueError):
            rollouts.main([
                "gk", "--trace-dir", str(trace_dir), "--out", str(REPO_ROOT / "checkpoints" / "bad.json"),
            ])

    def test_hold_production_override_scope_and_constants(self, tmp_path):
        from tools.audit import rollouts

        assert rollouts.KICK_OVERRIDE_STEPS == 1
        assert rollouts.HOLD_OVERRIDE_STEPS == 15

        ckpt = write_constant_hybrid_checkpoint(tmp_path / "c_const", z_kick=3.0, z_spec=-10.0)
        pol_a_raw = harness.TeamPolicy(str(ckpt), "blue")
        pol_b_raw = harness.TeamPolicy(str(ckpt), "red")

        class _SpyPol:
            def __init__(self, inner, sink):
                self.inner = inner
                self.sink = sink

            def act(self, obs, cen, mode, sigma_scale, gen):
                act, val, mean = self.inner.act(obs, cen, mode, sigma_scale, gen)
                self.sink.append(act.copy())
                return act, val, mean

        class _RecordingEnv(FootballParallelEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.recorded_acts = []

            def step_array(self, actions_array, v_preds=None, action_means=None):
                self.recorded_acts.append(np.asarray(actions_array, dtype=np.float32).copy())
                return super().step_array(actions_array, v_preds=v_preds, action_means=action_means)

        env = _RecordingEnv(config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False)))
        env.reset(seed=0)
        env.player_pos[5] = (700.0, 420.0)
        env.player_angle[5] = 0.0
        env.player_vel[:] = 0.0
        env.ball_holder = 5
        env.ball_hold_timer = 0.10
        env.kick_grace_timer = 0.0
        env.ball_pos[:] = (724.0, 420.0)
        for idx in range(12):
            if idx != 5:
                env.player_pos[idx] = (100.0 + 40.0 * (idx % 6), 40.0 if idx < 6 else 800.0)

        unforced_a = []
        unforced_b = []
        kick_actual = []
        hold_actual = []

        def _cap_kick(step_k, cur_env):
            if step_k == 19:
                kick_actual[:] = cur_env.recorded_acts

        def _cap_hold(step_k, cur_env):
            if step_k == 19:
                hold_actual[:] = cur_env.recorded_acts

        rollouts.roll_hold_branches(
            env,
            holder_player=5,
            team_a=_SpyPol(pol_a_raw, unforced_a),
            team_b=_SpyPol(pol_b_raw, unforced_b),
            branch_seed=42,
            mode="greedy",
            sigma_scale=0.0,
            horizons=(20,),
            after_kick_step=_cap_kick,
            after_hold_step=_cap_hold,
        )

        # Each branch runs 20 policy steps + 1 end-bootstrap call = 21 calls per team.
        assert len(unforced_a) == 42 and len(unforced_b) == 42
        assert len(kick_actual) == 20 and len(hold_actual) == 20

        kick_unforced = [np.concatenate([unforced_a[k], unforced_b[k]]) for k in range(20)]
        hold_unforced = [np.concatenate([unforced_a[21 + k], unforced_b[21 + k]]) for k in range(20)]

        # Unforced kick column is ~0.7985 (strictly distinct from both +1 and -1).
        for k in range(20):
            diff_kick = np.abs(kick_actual[k] - kick_unforced[k]) > 1e-6
            if k == 0:
                expected_mask = np.zeros((12, 4), dtype=bool)
                expected_mask[5, 2] = True
                assert np.array_equal(diff_kick, expected_mask)
                assert kick_actual[k][5, 2] == pytest.approx(1.0)
            else:
                assert not np.any(diff_kick), f"unexpected kick override at step {k}"

            diff_hold = np.abs(hold_actual[k] - hold_unforced[k]) > 1e-6
            if k < 15:
                expected_mask = np.zeros((12, 4), dtype=bool)
                expected_mask[5, 2] = True
                assert np.array_equal(diff_hold, expected_mask)
                assert hold_actual[k][5, 2] == pytest.approx(-1.0)
            else:
                assert not np.any(diff_hold), f"unexpected hold override at step {k}"

    def test_non_shaping_return_matches_recording_r_sparse(self, tmp_path):
        from tools.audit import rollouts

        ckpt = write_constant_hybrid_checkpoint(tmp_path / "c_ns", z_kick=-3.0, z_spec=-10.0)
        pol_a = harness.TeamPolicy(str(ckpt), "blue")
        pol_b = harness.TeamPolicy(str(ckpt), "red")

        replay_dir = tmp_path / "replays"
        cfg = replace(
            DEFAULT_CONFIG,
            replay=replace(
                DEFAULT_CONFIG.replay,
                enable_recording=True,
                interest_threshold=1e9,
                save_dir=str(replay_dir),
            ),
        )
        env = FootballParallelEnv(config=cfg)
        env.reset(seed=0)
        # Put holder 5 in shooting range facing the goal (with team_side[0]==0 -> right goal at x=1400)
        goal_x = 1400.0 if env.team_side[0] == 0 else 0.0
        env.player_angle[5] = 0.0 if env.team_side[0] == 0 else math.pi
        dir_x = 1.0 if env.team_side[0] == 0 else -1.0
        env.player_pos[5] = (goal_x - 350.0 * dir_x, 420.0)
        env.player_vel[:] = 0.0
        env.ball_holder = 5
        env.ball_hold_timer = 0.10
        env.kick_grace_timer = 0.0
        env.ball_pos[:] = (env.player_pos[5][0] + 24.0 * dir_x, 420.0)
        for idx in range(12):
            if idx != 5:
                env.player_pos[idx] = (700.0 + 30.0 * (idx % 6), 40.0 if idx < 6 else 800.0)

        kick_bds = []
        kick_rews = []

        def _rec_kick(step_k, cur_env):
            kick_bds.append(copy.deepcopy(cur_env.last_rew_breakdown))
            kick_rews.append(cur_env.last_step_rewards.copy())

        res = rollouts.roll_hold_branches(
            env,
            holder_player=5,
            team_a=pol_a,
            team_b=pol_b,
            branch_seed=19,
            mode="greedy",
            sigma_scale=0.0,
            horizons=(15,),
            after_kick_step=_rec_kick,
        )
        gamma = float(DEFAULT_CONFIG.ppo.gamma)
        expected_ns = sum(
            (gamma ** k) * float(np.mean([row["r_sparse"] for row in bd[:6]]))
            for k, bd in enumerate(kick_bds)
        )
        expected_full = sum(
            (gamma ** k) * float(np.mean(rew[:6]))
            for k, rew in enumerate(kick_rews)
        )
        # Confirm both shaping and sparse rewards are nonzero so full != non_shaping
        assert abs(expected_ns) > 1e-2
        assert abs(expected_full - expected_ns) > 1e-2
        assert res["kick"][15]["non_shaping_return"] == pytest.approx(expected_ns, abs=15 * 6e-5)
        assert not replay_dir.exists()

    def test_bootstrap_uses_holder_team_slice_not_all_players(self):
        from tools.audit import rollouts

        class _StubEnv:
            def __init__(self):
                self.cfg = DEFAULT_CONFIG
                self.ball_holder = 11
                self.last_ball_holder = 11
                self.last_ball_team = 1
                self.last_turnover_reward_time = -999.0
                self.shot_on_target_cd = np.zeros(12)
                self.tackle_cooldown = np.zeros(12)
                self.stun_timer = np.zeros(12)
                self.stats = {
                    "score": [0, 0], "own_goals": 0, "team_assists": [0, 0],
                    "field_5s_violations": 0, "keeper_5s_violations": 0,
                    "failed_tackles": 0, "shots": [0, 0], "team_passes": [0, 0], "team_saves": [0, 0],
                }
                self.last_shaping_rewards = np.zeros(12, dtype=np.float64)

            def _get_observations_array(self):
                return np.zeros((12, 82), dtype=np.float32)

            def get_centralized_state(self):
                return np.zeros((12, 54), dtype=np.float32)

            def step_array(self, act, val=None):
                rew = np.concatenate([np.full(6, 0.15, dtype=np.float32), np.full(6, 0.25, dtype=np.float32)])
                self.last_shaping_rewards = np.concatenate([np.full(6, 0.10), np.full(6, 0.15)])
                return self._get_observations_array(), self.get_centralized_state(), rew, False, None

        class _StubPol:
            def __init__(self, val_fill: float):
                self.val_fill = val_fill

            def act(self, obs, cen, mode, sigma_scale, gen):
                return np.zeros((6, 4), dtype=np.float32), np.full(6, self.val_fill, dtype=np.float32), np.zeros((6, 4), dtype=np.float32)

        gamma = float(DEFAULT_CONFIG.ppo.gamma)
        out = rollouts.roll_hold_branches(
            _StubEnv(),
            holder_player=11,  # team 1
            team_a=_StubPol(2.0),
            team_b=_StubPol(5.0),
            branch_seed=7,
            horizons=(2, 4),
            kick_steps=1,
            hold_steps=2,
        )
        for h in (2, 4):
            expected_ns = sum((gamma ** k) * 0.10 for k in range(h))
            expected_full_boot = sum((gamma ** k) * 0.25 for k in range(h)) + (gamma ** h) * 5.0
            assert out["kick"][h]["non_shaping_return"] == pytest.approx(expected_ns, rel=1e-6)
            assert out["kick"][h]["full_return_bootstrap"] == pytest.approx(expected_full_boot, rel=1e-6)

    def test_keeper_strip_after_intrastep_pickup_attributed_to_turnover(self):
        from tools.audit import rollouts

        env = FootballParallelEnv(config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False)))
        env.reset(seed=0)
        # Ensure team 1 defends the right goal (x=1400).
        if env.team_side[1] != 1:
            env.team_side[0], env.team_side[1] = 1, 0
        env.ball_holder = None
        env.last_ball_team = 0
        env.last_ball_holder = 5
        env.ball_kicked_in_flight = False
        env.pickup_cooldown[:] = 0.0
        env.stun_timer[:] = 0.0
        env.ghost_timer[:] = 0.0
        env.last_keeper_kick_time[:] = -999.0
        env.player_vel[:] = 0.0
        # Place player 5 outside right box (1400 - 195 - 18 = 1187 -> 1185) facing right,
        # ball 12 u ahead (1197), and team-1 keeper 6 at 1228 facing left (31 u from free ball,
        # 19 u from player 5's carried ball at 1209 -> strips in same RL step).
        env.player_pos[5] = (1185.0, 420.0)
        env.player_angle[5] = 0.0
        env.ball_pos[:] = (1197.0, 420.0)
        env.ball_vel[:] = (0.0, 0.0)
        env.player_pos[6] = (1228.0, 420.0)
        env.player_angle[6] = math.pi
        for idx in range(12):
            if idx not in (5, 6):
                env.player_pos[idx] = (400.0 + 30.0 * (idx % 6), 60.0 if idx < 6 else 780.0)

        act = np.zeros((12, 4), dtype=np.float32)
        act[:, 2:] = -1.0
        pre = rollouts._snapshot_step_state(env)
        _, _, rew, _, _ = env.step_array(act)
        r_ns = rollouts._non_shaping_rewards(env, rew)

        assert pre["holder"] is None
        assert env.ball_holder == 6
        c0 = rollouts._attribute_sparse_components(pre, env, act, r_ns, 0)
        c1 = rollouts._attribute_sparse_components(pre, env, act, r_ns, 1)
        expected_to = float(DEFAULT_CONFIG.rewards.successful_tackle_tackler_ind) / 6.0 + float(
            DEFAULT_CONFIG.rewards.successful_tackle_team
        )
        assert abs(c0["other"]) < 1e-6
        assert abs(c1["other"]) < 1e-6
        assert c0["turnover"] == pytest.approx(-expected_to, abs=1e-5)
        assert c1["turnover"] == pytest.approx(expected_to, abs=1e-5)

    def test_real_match_step_other_component_below_1e_6(self, tmp_path):
        from tools.audit import rollouts

        ckpt = write_constant_hybrid_checkpoint(tmp_path / "ckpt_match", z_kick=0.0, z_spec=0.0)
        spec = quick_spec(ckpt, opponent="scripted", mode="sampled", n=1, seed=11, max_steps=1200)
        trace = harness.play_match(spec, 0)

        env = rollouts._build_env()
        env.reset(seed=11 * 1000 + 0)
        act = np.asarray(trace["act"], dtype=np.float32)
        nonzero_sparse_steps = 0
        for t in range(act.shape[0]):
            pre = rollouts._snapshot_step_state(env)
            _, _, rew, done, _ = env.step_array(act[t])
            r_ns = rollouts._non_shaping_rewards(env, rew)
            if float(np.max(np.abs(r_ns))) > 1e-6:
                nonzero_sparse_steps += 1
            for team in (0, 1):
                comp = rollouts._attribute_sparse_components(pre, env, act[t], r_ns, team)
                assert abs(comp["other"]) < 1e-6, f"step {t} team {team} had other={comp['other']}"
            if done:
                break
        assert nonzero_sparse_steps >= 5


# ----------------------------------------------------------------------------
# 4. E5: X4 leftovers (F5b limit boundary, F6 checkpoint-trace match & metadata)
# ----------------------------------------------------------------------------
class TestRolloutsE5:
    def test_f5b_pos_err_limit_tight_bracket(self, tmp_path):
        """[GUARD] POS_ERR_LIMIT=1e-3 tightly bracketed: 9.0e-4 passes, 1.1e-3 is skipped and counted."""
        from tools.audit import rollouts

        _, trace_dir, traces = _make_gaussian_trace_run(tmp_path, max_steps=90, n=2, seed=11)
        base = traces[0]
        match_seed = 11 * 1000 + 0

        # Pass at 9.0e-4 u (pos and ball), both in resimulate_match and run_gk
        for field in ("pos", "ball"):
            near_ok = {k: v.copy() for k, v in base.items()}
            if field == "pos":
                near_ok["pos"][40, 0, 0] += np.float32(9.0e-4)
            else:
                near_ok["ball"][40, 0] += np.float32(9.0e-4)
            ok, max_err, _ = rollouts.resimulate_match(near_ok, match_seed=match_seed)
            assert ok is True
            assert 8.0e-4 <= max_err < rollouts.POS_ERR_LIMIT

        ok_trace = {k: v.copy() for k, v in base.items()}
        ok_trace["pos"][40, 0, 0] += np.float32(9.0e-4)
        np.savez_compressed(trace_dir / "m000.npz", **ok_trace)
        res_ok = rollouts.run_gk(trace_dir=trace_dir, trace_seed=11, workers=1)
        assert res_ok["n_matches_total"] == 2
        assert res_ok["n_matches_ok"] == 2
        assert res_ok["n_matches_skipped"] == 0

        # Skip and count at 1.1e-3 u (pos and ball), both in resimulate_match and run_gk
        for field in ("pos", "ball"):
            bad = {k: v.copy() for k, v in base.items()}
            if field == "pos":
                bad["pos"][40, 0, 0] += np.float32(1.1e-3)
            else:
                bad["ball"][40, 0] += np.float32(1.1e-3)
            ok_bad, err_bad, _ = rollouts.resimulate_match(bad, match_seed=match_seed)
            assert ok_bad is False
            assert err_bad >= rollouts.POS_ERR_LIMIT

        bad_trace = {k: v.copy() for k, v in base.items()}
        bad_trace["ball"][40, 0] += np.float32(1.1e-3)
        np.savez_compressed(trace_dir / "m000.npz", **bad_trace)
        res_bad = rollouts.run_gk(trace_dir=trace_dir, trace_seed=11, workers=1)
        assert res_bad["n_matches_total"] == 2
        assert res_bad["n_matches_ok"] == 1
        assert res_bad["n_matches_skipped"] == 1

    def test_f6_checkpoint_matches_its_own_trace_and_rejects_wrong_checkpoint(self, tmp_path):
        """[RED] A gaussian4 trace (1 match, workers=1) passes with the checkpoint that played it; another checkpoint raises."""
        from tools.audit import rollouts

        own_ckpt, trace_dir, traces = _make_gaussian_trace_run(tmp_path, max_steps=120, n=1, seed=11)
        other_ckpt = write_head_checkpoint(tmp_path / "ckpt_other", {"blue": "gaussian4", "red": "gaussian4"}, seed=4)
        assert len(rollouts.find_hold_decisions(traces[0])) >= 1

        res = rollouts.run_hold(
            trace_dir=trace_dir,
            checkpoint_dir=own_ckpt,
            trace_seed=11,
            seed=1,
            max_matches=1,
            workers=1,
        )
        assert res["n_matches_ok"] == 1
        assert res["n_matches_skipped"] == 0
        assert res["n_decisions"] >= 1
        assert res["shaping_mode"] == str(DEFAULT_CONFIG.rewards.shaping_mode)
        assert res["gamma"] == pytest.approx(float(DEFAULT_CONFIG.ppo.gamma))

        with pytest.raises(ValueError) as exc_info:
            rollouts.run_hold(
                trace_dir=trace_dir,
                checkpoint_dir=other_ckpt,
                trace_seed=11,
                seed=1,
                max_matches=1,
                workers=1,
            )
        msg = str(exc_info.value)
        assert re.search(r"\bmatch 0 step 0 seat \d+\b", msg), msg

    def test_f6_tolerance_1e_5_and_checks_matches_after_first(self, tmp_path):
        """[RED] Tight 1e-5 val/mean check at step 0 across every match and seat (kills 1e-5->1e-2 and first-match-only)."""
        from tools.audit import rollouts

        ckpt, trace_dir, traces = _make_hybrid_trace_run(tmp_path, max_steps=120, n=2, seed=11, z_kick=-2.0)

        # 5e-6 perturbation at step 0 of match 1 must pass
        ok_m1 = {k: v.copy() for k, v in traces[1].items()}
        ok_m1["val"][0, 4] += np.float32(5.0e-6)
        np.savez_compressed(trace_dir / "m001.npz", **ok_m1)
        res_ok = rollouts.run_hold(
            trace_dir=trace_dir, checkpoint_dir=ckpt, trace_seed=11, seed=1, max_matches=2, workers=1,
        )
        assert res_ok["n_matches_ok"] == 2

        # 5e-5 perturbation on val at step 0 of match 1 (match 0 clean) must raise, naming match 1, step 0, seat 4
        bad_val = {k: v.copy() for k, v in traces[1].items()}
        bad_val["val"][0, 4] += np.float32(5.0e-5)
        np.savez_compressed(trace_dir / "m001.npz", **bad_val)
        with pytest.raises(ValueError) as exc_val:
            rollouts.run_hold(
                trace_dir=trace_dir, checkpoint_dir=ckpt, trace_seed=11, seed=1, max_matches=2, workers=1,
            )
        msg_val = str(exc_val.value)
        assert re.search(r"\bmatch 1 step 0 seat 4\b", msg_val), msg_val

        # 5e-5 perturbation on mean at step 0 of match 1 seat 9 (team B) must raise, naming match 1, step 0, seat 9
        bad_mean = {k: v.copy() for k, v in traces[1].items()}
        bad_mean["mean"][0, 9, 2] += np.float32(5.0e-5)
        np.savez_compressed(trace_dir / "m001.npz", **bad_mean)
        with pytest.raises(ValueError) as exc_mean:
            rollouts.run_hold(
                trace_dir=trace_dir, checkpoint_dir=ckpt, trace_seed=11, seed=1, max_matches=2, workers=1,
            )
        msg_mean = str(exc_mean.value)
        assert re.search(r"\bmatch 1 step 0 seat 9\b", msg_mean), msg_mean

    def test_f6_verifies_step_zero_even_without_hold_decisions(self, tmp_path):
        """[RED] Checkpoint verification runs at step 0 even when a match has no hold decision points."""
        from tools.audit import rollouts

        ckpt, trace_dir, traces = _make_hybrid_trace_run(tmp_path, max_steps=2, n=1, seed=11, z_kick=-2.0)
        assert rollouts.find_hold_decisions(traces[0]) == []

        bad = {k: v.copy() for k, v in traces[0].items()}
        bad["val"][0, 2] += np.float32(5.0e-5)
        np.savez_compressed(trace_dir / "m000.npz", **bad)
        with pytest.raises(ValueError) as exc_info:
            rollouts.run_hold(
                trace_dir=trace_dir, checkpoint_dir=ckpt, trace_seed=11, seed=1, max_matches=1, workers=1,
            )
        msg = str(exc_info.value)
        assert re.search(r"\bmatch 0 step 0 seat 2\b", msg), msg

    def test_gk_and_hold_json_include_shaping_mode_and_gamma(self, tmp_path):
        """[RED] Both gk and hold JSON outputs record shaping_mode and gamma."""
        from tools.audit import rollouts

        ckpt, trace_dir, _ = _make_hybrid_trace_run(tmp_path, max_steps=60, n=1, seed=11, z_kick=-2.0)
        gk_out = tmp_path / "gk.json"
        hold_out = tmp_path / "hold.json"
        assert rollouts.main([
            "gk", "--trace-dir", str(trace_dir), "--trace-seed", "11",
            "--workers", "1", "--shaping-mode", "all", "--out", str(gk_out),
        ]) == 0
        assert rollouts.main([
            "hold", "--trace-dir", str(trace_dir), "--checkpoint-dir", str(ckpt),
            "--trace-seed", "11", "--workers", "1", "--shaping-mode", "ball", "--out", str(hold_out),
        ]) == 0

        gk_data = json.loads(gk_out.read_text(encoding="utf-8"))
        hold_data = json.loads(hold_out.read_text(encoding="utf-8"))
        assert gk_data["shaping_mode"] == "all"
        assert gk_data["gamma"] == pytest.approx(float(DEFAULT_CONFIG.ppo.gamma))
        assert hold_data["shaping_mode"] == "ball"
        assert hold_data["gamma"] == pytest.approx(float(DEFAULT_CONFIG.ppo.gamma))



# ----------------------------------------------------------------------------
# 5. Intent split and forced-hold length in `hold`
# ----------------------------------------------------------------------------
HOLD_GOLDEN_DIR = REPO_ROOT / "tests" / "golden" / "rollouts_hold_default"
INTENT_BUCKETS = ("intent_kick", "intent_mid", "intent_hold", "def_intent_kick", "def_intent_hold")
INTENT_LABELS = ("intent=kick", "intent=mid", "intent=hold", "def&intent=kick", "def&intent=hold")


def _hold_fixture(tmp_path: Path, head: str):
    torch.manual_seed(20261119)  # the hybrid5 helper leaves the critic at its random init; seed it so `val` is reproducible
    if head == "hybrid5":
        return _make_hybrid_trace_run(tmp_path, max_steps=120, n=2, seed=11, z_kick=-2.0)
    return _make_gaussian_trace_run(tmp_path, max_steps=120, n=2, seed=11)


def _without_intent_buckets(summary: dict) -> dict:
    out = copy.deepcopy(summary)
    out.pop("hold_steps", None)
    for h_data in out["horizons"].values():
        for bucket in INTENT_BUCKETS:
            h_data.pop(bucket, None)
    return out


def _report_without_intent_lines(text: str) -> list[str]:
    lines = text.splitlines()
    kept: list[str] = []
    skip_components = False
    for line in lines:
        if skip_components and line.strip().startswith("components:"):
            skip_components = False
            continue
        skip_components = False
        if line.strip().startswith(INTENT_LABELS):
            skip_components = True
            continue
        kept.append(line)
    return kept


class TestHoldIntentSplit:
    @pytest.mark.parametrize("head", ["hybrid5", "gaussian4"])
    def test_default_hold_output_matches_frozen_golden(self, tmp_path, head):
        """[GUARD] Default `rollouts hold` (no `--hold-steps`) keeps every pre-existing key and value and every pre-existing
        report line, in order. The only additions allowed are the intent buckets under each horizon (and their report lines,
        after the existing buckets) and the top-level `hold_steps`.

        Golden produced at commit f6cd858 (unmodified `tools/audit/rollouts.py`) on the fixtures of `_hold_fixture`
        (`torch.manual_seed(20261119)`, then `_make_hybrid_trace_run(max_steps=120, n=2, seed=11, z_kick=-2.0)`, `_make_gaussian_trace_run(max_steps=120, n=2,
        seed=11)`) with `rollouts.main(["hold", "--trace-dir", T, "--checkpoint-dir", C, "--trace-seed", "11", "--seed", "77",
        "--workers", "1", "--out", OUT])`; `OUT/rollouts.json` and `OUT/report.txt` were copied to
        `tests/golden/rollouts_hold_default/<head>.json` and `<head>_report.txt`.
        """
        from tools.audit import rollouts

        ckpt, trace_dir, _ = _hold_fixture(tmp_path, head)
        out = tmp_path / "out"
        assert rollouts.main([
            "hold", "--trace-dir", str(trace_dir), "--checkpoint-dir", str(ckpt),
            "--trace-seed", "11", "--seed", "77", "--workers", "1", "--out", str(out),
        ]) == 0
        data = json.loads((out / "rollouts.json").read_text(encoding="utf-8"))
        frozen = json.loads((HOLD_GOLDEN_DIR / f"{head}.json").read_text(encoding="utf-8"))
        assert frozen["n_decisions"] >= 1

        assert set(data) - set(frozen) <= {"hold_steps"}
        assert set(data["horizons"]) == set(frozen["horizons"])
        for h, h_data in data["horizons"].items():
            assert set(h_data) - set(frozen["horizons"][h]) <= set(INTENT_BUCKETS)
        assert _without_intent_buckets(data) == frozen

        new_lines = _report_without_intent_lines((out / "report.txt").read_text(encoding="utf-8"))
        old_lines = (HOLD_GOLDEN_DIR / f"{head}_report.txt").read_text(encoding="utf-8").splitlines()
        assert len(new_lines) == len(old_lines)
        assert new_lines[0].split("  matches ok=", 1)[1] == old_lines[0].split("  matches ok=", 1)[1]
        assert new_lines[1:] == old_lines[1:]

    @staticmethod
    def _intent_trace(head):
        tr = make_trace(10)
        tr["mean"] = np.zeros((10, 12, 4), dtype=np.float64)
        if head is not None:
            tr["head"] = np.asarray(head, dtype=np.int8)
        return tr

    def test_intent_hybrid5_reads_p_from_the_kick_mean(self):
        """[RED] hybrid5 (`head[team] == 1`): p = (mean_kick + 1) / 2; kick if p > 0.5, hold if p < 0.05, else mid."""
        from tools.audit import rollouts

        tr = self._intent_trace([1, 0])
        cases = {
            0.2: "kick",     # p = 0.60
            0.0: "mid",      # p = 0.50, not > 0.5
            -0.88: "mid",    # p = 0.06
            -0.92: "hold",   # p = 0.04
            -1.0: "hold",    # p = 0
        }
        for step, (mean_kick, expected) in enumerate(cases.items()):
            tr["mean"][step, 3, 2] = mean_kick
            tr["mean"][step, 9, 2] = 0.9            # the other team's seat is never read for holder 3
            assert rollouts.decision_intent(tr, step, 3) == expected, (mean_kick, expected)

    def test_intent_gaussian4_threshold_is_strict_at_0_35(self):
        """[RED] gaussian4 (`head[team] == 0`, or no `head` field): kick if mean_kick > 0.35, else hold (never mid)."""
        from tools.audit import rollouts

        for head in ([1, 0], None):
            tr = self._intent_trace(head)
            for step, (mean_kick, expected) in enumerate(((0.35, "hold"), (0.3500001, "kick"), (0.9, "kick"), (-0.95, "hold"), (0.2, "hold"))):
                tr["mean"][step, 9, 2] = mean_kick
                tr["mean"][step, 3, 2] = -1.0 if mean_kick > 0.35 else 1.0   # the other team's seat is never read
                assert rollouts.decision_intent(tr, step, 9) == expected, (head, mean_kick, expected)

    @staticmethod
    def _hand_record(intent, third, value):
        diff = {m: value for m in ("non_shaping_return", "full_return_bootstrap", "p_team_ball_1s", "p_goal_for", "p_goal_against")}
        diff["components"] = {ck: value for ck in ("turnover", "pass_bonus", "shot_on_target", "goal", "failed_tackle", "five_s_rule", "save", "other")}
        return {"intent": intent, "third": third, "pressure": "mid", "rollout": {"diff": {75: diff, 225: diff}}}

    def test_summary_intent_buckets_on_hand_records(self):
        """[RED] Five records: kick/defensive 0.3, kick/middle 0.1, mid/defensive -0.2, hold/defensive 0.5, hold/attacking 0.7.
        intent_kick = {0.3, 0.1}: mean 0.2, sd 0.141421, se 0.1; intent_mid = {-0.2}; intent_hold = {0.5, 0.7}: mean 0.6, se 0.1;
        def_intent_kick = {0.3}; def_intent_hold = {0.5}; overall n = 5 and third_defensive = {0.3, -0.2, 0.5} as before."""
        from tools.audit import rollouts

        records = [
            self._hand_record("kick", "defensive", 0.3),
            self._hand_record("kick", "middle", 0.1),
            self._hand_record("mid", "defensive", -0.2),
            self._hand_record("hold", "defensive", 0.5),
            self._hand_record("hold", "attacking", 0.7),
        ]
        s = rollouts.summarize_hold(records, 1, 1, 0, 0.0, seed=1)
        for h in ("75", "225"):
            b = s["horizons"][h]
            for bucket, n, mean, se in (
                ("intent_kick", 2, 0.2, 0.1),
                ("intent_mid", 1, -0.2, None),
                ("intent_hold", 2, 0.6, 0.1),
                ("def_intent_kick", 1, 0.3, None),
                ("def_intent_hold", 1, 0.5, None),
                ("overall", 5, 0.28, None),
                ("third_defensive", 3, 0.2, None),
            ):
                stat = b[bucket]["non_shaping_return"]
                assert b[bucket]["n"] == n and stat["n"] == n, bucket
                assert stat["mean"] == pytest.approx(mean), bucket
                if se is not None:
                    assert stat["se"] == pytest.approx(se), bucket
                assert b[bucket]["components"]["goal"]["mean"] == pytest.approx(mean), bucket
        assert "def_intent_mid" not in s["horizons"]["75"]

        lines = rollouts.format_hold_report(s, title="hand").splitlines()
        labels = [ln.strip().split(":", 1)[0].strip() for ln in lines if ln.startswith("    ") and not ln.startswith("      ")]
        per_horizon = ["overall", "opp < 70 u", "opp >= 120 u", "defensive 1/3", "middle 1/3", "attacking 1/3", *INTENT_LABELS]
        assert labels == per_horizon * 2

    def test_records_carry_intent_end_to_end(self, tmp_path):
        """[RED] On the z_kick = -2 hybrid5 fixture every hold decision records mean_kick = -0.73883 (p = 0.1306): both decisions
        land in intent_mid, none in intent_kick or intent_hold."""
        from tools.audit import rollouts

        ckpt, trace_dir, _ = _hold_fixture(tmp_path, "hybrid5")
        res = rollouts.run_hold(trace_dir=trace_dir, checkpoint_dir=ckpt, trace_seed=11, seed=77, workers=1)
        for h in ("75", "225"):
            b = res["horizons"][h]
            assert (b["overall"]["n"], b["intent_mid"]["n"], b["intent_kick"]["n"], b["intent_hold"]["n"]) == (2, 2, 0, 0)

    @staticmethod
    def _observe_hold_branch_kicks(monkeypatch, rollouts):
        """Wrap `roll_hold_branches` with a HOLD-branch hook reading the env's kick counter after every decision."""
        real = rollouts.roll_hold_branches
        logs = []

        def observed(snap, **kw):
            start = int(snap.stats["kicks"])
            seen = []
            logs.append(seen)
            return real(snap, after_hold_step=lambda k, env: seen.append(int(env.stats["kicks"]) - start), **kw)

        monkeypatch.setattr(rollouts, "roll_hold_branches", observed)
        return logs

    def test_hold_steps_option_sets_the_forced_hold_length(self, tmp_path, monkeypatch):
        """[RED] z_kick = +1 hybrid5 fixture (1 match): one hold decision (step 2, holder 5, p = 0.7269 -> intent kick). In
        greedy branches that policy fires whenever it may, so the HOLD branch's first kick comes on the first unforced
        decision: index 1 with `--hold-steps 1`, index 15 by default."""
        from tools.audit import rollouts

        ckpt, trace_dir, _ = _make_hybrid_trace_run(tmp_path, max_steps=120, n=1, seed=11, z_kick=1.0)
        logs = self._observe_hold_branch_kicks(monkeypatch, rollouts)
        for flag, first_kick in ((["--hold-steps", "1"], 1), ([], 15)):
            logs.clear()
            out = tmp_path / f"hold{len(flag)}.json"
            assert rollouts.main([
                "hold", "--trace-dir", str(trace_dir), "--checkpoint-dir", str(ckpt), "--trace-seed", "11",
                "--mode", "greedy", "--workers", "1", *flag, "--out", str(out),
            ]) == 0
            assert len(logs) == 1
            assert logs[0].index(1) == first_kick
            assert logs[0][:first_kick] == [0] * first_kick
            data = json.loads(out.read_text(encoding="utf-8"))
            assert data["hold_steps"] == first_kick
            assert data["horizons"]["75"]["intent_kick"]["n"] == 1

    def test_hold_steps_below_one_is_rejected(self, tmp_path, capsys):
        """[RED] `--hold-steps` 0 and -1 are argparse errors naming the flag; `run_hold(hold_steps=0)` raises ValueError."""
        from tools.audit import rollouts

        for bad in ("0", "-1", "x"):
            with pytest.raises(SystemExit):
                rollouts.main(["hold", "--trace-dir", str(tmp_path), "--checkpoint-dir", str(tmp_path),
                               "--hold-steps", bad, "--out", str(tmp_path / "o.json")])
            assert "--hold-steps" in capsys.readouterr().err
        with pytest.raises(ValueError, match="hold_steps"):
            rollouts.run_hold(trace_dir=tmp_path, checkpoint_dir=tmp_path, hold_steps=0)


# ----------------------------------------------------------------------------
# 6. Aim counterfactual (`aim`): POLICY / AIM / AIMP at real kicks with an open teammate in the kickable arc
# ----------------------------------------------------------------------------
from tests.test_eval_metrics import aim_scene  # noqa: E402


def _aim_hand_trace(T: int) -> dict:
    tr = make_trace(T)
    tr["ang"] = np.zeros((T, 12), dtype=np.float32)
    tr["mean"] = np.zeros((T, 12, 4), dtype=np.float32)
    tr["head"] = np.array([1, 1], dtype=np.int8)
    return tr


class TestAimRollouts:
    def test_point_selection_spacing_target_and_reach(self):
        """[RED] Kicker A3 holds and kicks (ball (300, 420), facing 0 at the kick row, +90 deg on later rows):
          5   open A4 +10, A5 -30, direction 0  -> point, target A4 (smallest |angle|), already aimed (A4 within 15 deg)
          30  open A4 +10                        -> skipped: 25 decisions after 5
          50  open A4 +40 only                   -> no point (40 > 35 deg); the spacing still counts from 5
          60  open A4 -34.5                      -> point (55 after 5), not already aimed
          104 open A4 +5                         -> skipped: 44 after 60
          150 open A4 +5, but nobody holds       -> no point
        Intent from the kick mean (hybrid5): 0.2 -> p 0.6 -> kick; -0.95 -> p 0.025 -> hold. Ball x 300, side 0 -> defensive."""
        from tools.audit import rollouts

        tr = _aim_hand_trace(200)
        aim_scene(tr, 5, {4: 10.0, 5: -30.0})
        aim_scene(tr, 30, {4: 10.0})
        aim_scene(tr, 50, {4: 40.0})
        aim_scene(tr, 60, {4: -34.5})
        aim_scene(tr, 104, {4: 5.0})
        aim_scene(tr, 150, {4: 5.0})
        tr["holder"][150] = -1
        tr["mean"][5, 3, 2] = 0.2
        tr["mean"][60, 3, 2] = -0.95
        pts = rollouts.find_aim_points(tr)
        assert [(p["step"], p["kicker"], p["target"], p["already"]) for p in pts] == [(5, 3, 4, True), (60, 3, 4, False)]
        assert pts[0]["deflection"] == pytest.approx(math.radians(10.0), abs=1e-6)
        assert pts[1]["deflection"] == pytest.approx(math.radians(-34.5), abs=1e-6)
        assert pts[0]["target_dist"] == pytest.approx(200.0, abs=1e-3)
        assert [p["intent"] for p in pts] == ["kick", "hold"]
        assert [p["third"] for p in pts] == ["defensive", "defensive"]
        assert [p["team"] for p in pts] == [0, 0]

    def test_point_needs_two_rows_after_the_kick(self):
        """[RED] A kick at row 97 is a point in a 100-row trace (97 + 2 < 100), not in a 99-row one."""
        from tools.audit import rollouts

        for T, n_points in ((100, 1), (99, 0)):
            tr = _aim_hand_trace(T)
            aim_scene(tr, 97, {4: 10.0})
            assert len(rollouts.find_aim_points(tr)) == n_points, T

    def test_steer_and_power_hand_values(self):
        """[RED] Steer = clip(deflection / 35 deg, -1, 1): +17.5 deg -> 0.5, -17.5 -> -0.5, 40 -> 1. AIMP power for a target
        300 u away and v_max 600 (midfielder): v0 = sqrt(200^2 + 2 * 155 * 300) = sqrt(133000) = 364.69165,
        pn = (364.69165 - 160) / 440 = 0.4652083, power = 0.35 + 0.65 * 0.4652083 + 0.0001 = 0.6524854. At 2000 u, v0 is
        capped at v_max: pn 1, power 1.0001. At 0 u: v0 = 200, pn = 40 / 440 = 0.0909091."""
        from tools.audit import rollouts

        assert rollouts.aim_steer(math.radians(17.5)) == pytest.approx(0.5)
        assert rollouts.aim_steer(math.radians(-17.5)) == pytest.approx(-0.5)
        assert rollouts.aim_steer(math.radians(40.0)) == 1.0
        v0, pn, power = rollouts.aimp_power(300.0, 600.0)
        assert v0 == pytest.approx(364.69165, abs=1e-4)
        assert pn == pytest.approx(0.4652083, abs=1e-6)
        assert power == pytest.approx(0.6524854, abs=1e-6)
        assert rollouts.aimp_power(2000.0, 600.0) == pytest.approx((600.0, 1.0, 1.0001))
        assert rollouts.aimp_power(0.0, 600.0)[1] == pytest.approx(0.0909091, abs=1e-6)

    @staticmethod
    def _fixture_point(tmp_path):
        """The hybrid5 `_hold_fixture` (z_kick = -2): match 0 has one aim point, step 2, striker A5 (global 5) kicking,
        target A4 at +12.3 deg of its facing."""
        from tools.audit import rollouts

        ckpt, trace_dir, traces = _hold_fixture(tmp_path, "hybrid5")
        tr = traces[0]
        pts = rollouts.find_aim_points(tr)
        assert [(p["step"], p["kicker"], p["target"], p["already"]) for p in pts] == [(2, 5, 4, False)]
        ok, _, snaps = rollouts.resimulate_match(tr, match_seed=11 * 1000 + 0, target_steps=[2])
        assert ok
        pols = (harness.TeamPolicy(str(ckpt), "blue"), harness.TeamPolicy(str(ckpt), "red"))
        return ckpt, trace_dir, tr, pts[0], snaps[2], pols

    def test_policy_branch_replays_the_kick_and_aim_points_at_the_target(self, tmp_path):
        """[RED] POLICY's step 0 plays the recorded actions, so its ball velocity after step 0 equals the trace's bvel[3]
        (< 1e-4). AIM differs from them only in the kicker's steer, AIMP only in its steer and power; AIM's ball is loose
        after step 0 and leaves within 1 deg of the direction from the ball to the target."""
        from tools.audit import rollouts

        _, _, tr, p, snap, (team_a, team_b) = self._fixture_point(tmp_path)
        steer = rollouts.aim_steer(p["deflection"])
        _, _, power = rollouts.aimp_power(p["target_dist"], 640.0)          # striker: atk_max_kick_speed 640
        seen = {}

        def hook(branch, k, env, act):
            if k == 0:
                seen[branch] = (env.ball_vel.copy(), env.ball_holder, np.array(act, dtype=np.float32, copy=True))

        res = rollouts.roll_aim_branches(
            snap, recorded_actions=tr["act"][2], kicker=5, target=4, branch_seed=123,
            team_a=team_a, team_b=team_b, steer=steer, power=power, after_step=hook,
        )
        np.testing.assert_allclose(seen["policy"][0], tr["bvel"][3], atol=1e-4)
        assert np.array_equal(seen["policy"][2], tr["act"][2])
        mask_aim = np.zeros((12, 4), dtype=bool)
        mask_aim[5, 1] = True
        mask_aimp = mask_aim.copy()
        mask_aimp[5, 2] = True
        assert np.array_equal(seen["aim"][2] != tr["act"][2], mask_aim)
        assert np.array_equal(seen["aimp"][2] != tr["act"][2], mask_aimp)
        assert seen["aim"][2][5, 1] == pytest.approx(steer, abs=1e-7)
        assert seen["aimp"][2][5, 2] == pytest.approx(power, abs=1e-7)
        assert seen["aim"][1] is None
        tgt = snap.player_pos[4] - snap.ball_pos
        v = seen["aim"][0]
        err = abs((math.atan2(v[1], v[0]) - math.atan2(tgt[1], tgt[0]) + math.pi) % (2 * math.pi) - math.pi)
        assert math.degrees(err) < 1.0
        assert res["aim"]["kick_err_deg"] == pytest.approx(math.degrees(err), abs=1e-9)
        assert res["policy"]["kick_err_deg"] > 15.0                          # the recorded kick was not aimed

    def test_branches_share_one_generator_seed(self, tmp_path):
        """[RED] With AIM's steer and AIMP's steer and power equal to the recorded ones, the three sampled branches are the
        same rollout: every recorded value is identical."""
        from tools.audit import rollouts

        _, _, tr, p, snap, (team_a, team_b) = self._fixture_point(tmp_path)
        rec = tr["act"][2]
        res = rollouts.roll_aim_branches(
            snap, recorded_actions=rec, kicker=5, target=4, branch_seed=987,
            team_a=team_a, team_b=team_b, steer=float(rec[5, 1]), power=float(rec[5, 2]),
        )
        assert set(res) == {"policy", "aim", "aimp"}
        assert set(res["policy"]["returns"]) == {15, 45, 75}
        assert res["aim"] == res["policy"]
        assert res["aimp"] == res["policy"]

    def test_full_return_adds_the_critic_bootstrap_at_every_horizon(self, tmp_path):
        """[GUARD] On the fixture point (120-step match, kick at step 2, so no terminal step within 75 decisions), the full
        return at h = 15, 45, 75 is sum_{k<h} gamma^k * mean(r_k over the kicking team) + gamma^h * mean(V(s_h) over the
        kicking team). The oracle reads the step rewards and the state s_h through `after_step` and evaluates V(s_h) with
        the harness team policy; gamma = 0.995 is the config value."""
        from tools.audit import rollouts

        _, _, tr, p, snap, (team_a, team_b) = self._fixture_point(tmp_path)
        gamma = 0.995
        rewards, states = [], {}

        def hook(branch, k, env, act):
            if branch == "policy":
                rewards.append(float(np.asarray(env.last_step_rewards, dtype=np.float64)[:6].mean()))
                if k + 1 in (15, 45, 75):
                    states[k + 1] = (env._get_observations_array(), env.get_centralized_state())

        res = rollouts.roll_aim_branches(
            snap, recorded_actions=tr["act"][2], kicker=5, target=4, branch_seed=321,
            team_a=team_a, team_b=team_b, steer=0.0, power=0.5, after_step=hook,
        )["policy"]
        assert len(rewards) == 75
        for h in (15, 45, 75):
            obs, cen = states[h]
            _, val, _ = team_a.act(obs[:6], cen[:6], "sampled", 1.0, torch.Generator().manual_seed(0))
            boot = gamma ** h * float(np.mean(val))
            assert abs(boot) > 1e-3, h                                          # the bootstrap is visible at this tolerance
            plain = sum(gamma ** k * r for k, r in enumerate(rewards[:h]))
            assert res["returns"][h]["full_return_bootstrap"] == pytest.approx(plain + boot, abs=1e-6), h

    @staticmethod
    def _hand_record(already, third, intent, firsts, ns75):
        branches = {}
        for b, first, ns in zip(("policy", "aim", "aimp"), firsts, ns75):
            branches[b] = {
                "returns": {h: {"non_shaping_return": ns, "full_return_bootstrap": 2.0 * ns} for h in (15, 45, 75)},
                "possession": {15: 1.0 if first == "target" else 0.0, 30: 0.0},
                "first_receiver": first, "goal_for": 0.0, "goal_against": 0.0,
                "kick_err_deg": None if first == "none" else 3.0,
            }
        return {"already": already, "third": third, "intent": intent, "branches": branches}

    def test_summary_buckets_on_hand_records(self):
        """[RED] Records (already, third, intent, first receiver P/AIM/AIMP, non-shaping return P/AIM/AIMP):
          A  no,  defensive, kick  opponent/target/target  0.0/0.3/0.1
          B  no,  middle,    kick  none/target/opponent    0.2/0.3/0.2
          C  yes, defensive, hold  target/target/target    0.5/0.5/0.4
        not_aimed (A, B): target share 0 / 1 / 0.5, opponent 0.5 / 0 / 0.5; AIM - POLICY {0.3, 0.1}: mean 0.2, se 0.1;
        AIMP - POLICY {0.1, 0.0}: mean 0.05, se 0.05; full return doubles them. already_aimed (C): n 1, AIM - POLICY 0.
        Thirds and intents split the not-aimed records only: defensive {A}, middle {B}, attacking {}; kick {A, B}, hold {}."""
        from tools.audit import rollouts

        recs = [
            self._hand_record(False, "defensive", "kick", ("opponent", "target", "target"), (0.0, 0.3, 0.1)),
            self._hand_record(False, "middle", "kick", ("none", "target", "opponent"), (0.2, 0.3, 0.2)),
            self._hand_record(True, "defensive", "hold", ("target", "target", "target"), (0.5, 0.5, 0.4)),
        ]
        s = rollouts.summarize_aim(recs, 1, 1, 0, 0.0, seed=1)
        assert s["mode"] == "aim" and s["n_points"] == 3
        b = s["buckets"]["not_aimed"]
        assert b["n"] == 2
        assert [b["branches"][x]["first_receiver_target"]["mean"] for x in ("policy", "aim", "aimp")] == pytest.approx([0.0, 1.0, 0.5])
        assert [b["branches"][x]["first_receiver_opponent"]["mean"] for x in ("policy", "aim", "aimp")] == pytest.approx([0.5, 0.0, 0.5])
        assert b["branches"]["policy"]["ball_held_after_kick"]["mean"] == pytest.approx(0.5)
        aim = b["paired"]["aim_minus_policy"]["non_shaping_return_75"]
        assert (aim["mean"], aim["se"], aim["n"]) == (pytest.approx(0.2), pytest.approx(0.1), 2)
        aimp = b["paired"]["aimp_minus_policy"]["non_shaping_return_75"]
        assert (aimp["mean"], aimp["se"]) == (pytest.approx(0.05), pytest.approx(0.05))
        assert b["paired"]["aim_minus_policy"]["full_return_bootstrap_75"]["mean"] == pytest.approx(0.4)
        assert b["paired"]["aim_minus_policy"]["possession_15"]["mean"] == pytest.approx(1.0)
        c = s["buckets"]["already_aimed"]
        assert c["n"] == 1 and c["paired"]["aim_minus_policy"]["non_shaping_return_75"]["mean"] == pytest.approx(0.0)
        n_of = {k: v["n"] for k, v in s["buckets"].items()}
        assert n_of == {
            "not_aimed": 2, "already_aimed": 1,
            "not_aimed_third_defensive": 1, "not_aimed_third_middle": 1, "not_aimed_third_attacking": 0,
            "not_aimed_intent_kick": 2, "not_aimed_intent_mid": 0, "not_aimed_intent_hold": 0,
        }
        assert s["buckets"]["not_aimed_third_middle"]["paired"]["aim_minus_policy"]["non_shaping_return_75"]["mean"] == pytest.approx(0.1)
        text = rollouts.format_aim_report(s, title="hand")
        assert text.splitlines()[0].startswith("===== rollouts aim hand")
        assert "-- not already aimed (n=2) --" in text and "-- already aimed (n=1) --" in text

    def test_aim_cli_end_to_end(self, tmp_path):
        """[RED] `rollouts aim` on the hybrid5 fixture (2 matches, 1 point): JSON and report written, the same JSON with 1 and
        2 workers, inputs untouched, protected outputs refused, and a missing --checkpoint-dir is an argparse error."""
        from tools.audit import rollouts

        ckpt, trace_dir, _ = _hold_fixture(tmp_path, "hybrid5")
        digests = (_dir_digest(ckpt), _dir_digest(trace_dir))
        out1, out2 = tmp_path / "aim1", tmp_path / "aim2.json"
        base = ["aim", "--trace-dir", str(trace_dir), "--checkpoint-dir", str(ckpt), "--trace-seed", "11", "--seed", "5",
                "--shaping-mode", "legacy"]
        assert rollouts.main([*base, "--workers", "1", "--out", str(out1)]) == 0
        assert rollouts.main([*base, "--workers", "2", "--max-matches", "2", "--out", str(out2)]) == 0
        d1 = json.loads((out1 / "rollouts.json").read_text(encoding="utf-8"))
        d2 = json.loads(out2.read_text(encoding="utf-8"))
        assert d1 == d2
        assert (d1["mode"], d1["seed"], d1["shaping_mode"], d1["n_matches_ok"], d1["n_points"]) == ("aim", 5, "legacy", 2, 1)
        assert d1["gamma"] == pytest.approx(float(DEFAULT_CONFIG.ppo.gamma))
        rec = d1["records"][0]
        assert (rec["match_idx"], rec["step"], rec["kicker"], rec["target"], rec["already"]) == (0, 2, 5, 4, False)
        assert (rec["third"], rec["intent"]) == ("middle", "mid")
        assert d1["buckets"]["not_aimed"]["n"] == 1 and d1["buckets"]["not_aimed_intent_mid"]["n"] == 1
        assert (out1 / "report.txt").read_text(encoding="utf-8").startswith("===== rollouts aim")
        assert (_dir_digest(ckpt), _dir_digest(trace_dir)) == digests
        for bad in (trace_dir / "bad.json", ckpt / "bad.json"):
            with pytest.raises(ValueError):
                rollouts.main([*base, "--out", str(bad)])
        with pytest.raises(SystemExit):
            rollouts.main(["aim", "--trace-dir", str(trace_dir), "--out", str(tmp_path / "x.json")])


# ----------------------------------------------------------------------------
# Credit: how much of a decision's counterfactual benefit reaches the trainer's advantage
# ----------------------------------------------------------------------------
from tests.test_eval_metrics import pass_scene  # noqa: E402

CREDIT_COMPONENTS = np.array([[0, 1, 0, 0], [10, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
CREDIT_VALUES = np.array([0.5, 0.25, 1.0, 0.5], dtype=np.float64)


def _credit_scene(holder: int, pos, angle: float, ball):
    """Every player parked on a back line except `holder`, who holds the ball at `pos` facing `angle`; ball at `ball`."""
    env = FootballParallelEnv(config=replace(DEFAULT_CONFIG, replay=replace(DEFAULT_CONFIG.replay, enable_recording=False)))
    env.reset(seed=0)
    env.player_vel[:] = 0.0
    for idx in range(12):
        env.player_pos[idx] = (100.0 + 40.0 * (idx % 6), 40.0 if idx < 6 else 800.0)
    env.player_pos[holder] = pos
    env.player_angle[holder] = angle
    env.ball_holder = holder
    env.ball_hold_timer = 0.10
    env.kick_grace_timer = 0.0
    env.ball_pos[:] = ball
    return env


def _credit_recorded(kicker: int, steer: float = 0.0, kick: float = 1.0) -> np.ndarray:
    rec = np.zeros((12, 4), dtype=np.float32)
    rec[:, 2:] = -1.0
    rec[kicker] = (0.0, steer, kick, -1.0)
    return rec


def _credit_policies(ckpt: Path):
    return harness.TeamPolicy(str(ckpt), "blue"), harness.TeamPolicy(str(ckpt), "red")


def _attack_fixture(tmp_path: Path):
    """Two 700-row matches of a random hybrid5 checkpoint (seed 5, spread 1.0, trace seed 11): match 1 has two outfield kicks of
    team B's seat 3 (player 9) in team B's attacking third, at rows 548 and 630."""
    ckpt = write_head_checkpoint(tmp_path / "ckpt_attack", {"blue": "hybrid5", "red": "hybrid5"}, seed=5, spread=1.0)
    trace_dir = tmp_path / "traces_attack"
    traces = harness.run_matches(quick_spec(ckpt, opponent="self", mode="sampled", n=2, seed=11, max_steps=700))
    harness.save_traces(traces, trace_dir, protected=[ckpt])
    return ckpt, trace_dir, traces


class TestCreditEstimator:
    def test_gae_at_decision_zero_hand_values(self):
        """[RED] Three steps, reward components (goal, sparse, ball, role) = (0, 1, 0, 0), (10, 0, 0, 0), (0, 0, 1, 0), critic
        V = [0.5, 0.25, 1.0, 0.5], gamma 0.5. The advantage at decision 0 is the reward part sum_l (gamma lam)^l r_l plus the
        critic part sum_l (gamma lam)^l (gamma V_{l+1} - V_l).
          lam 0: reward (0, 1, 0, 0), critic 0.125 - 0.5 = -0.375, total 0.625.
          lam 1: weights 1, 0.5, 0.25: reward (5, 1, 0.25, 0), critic (0.125 - 0.5) + 0.5 (0.5 - 0.25) + 0.25 (0.25 - 1) = -0.4375,
                 total 5.8125. A terminal at the last step drops gamma V_3 = 0.25 (weight 0.25): critic -0.5, total 5.75.
          lam 0.5: weights 1, 0.25, 0.0625: reward (2.5, 1, 0.0625, 0), critic -0.375 + 0.25 * 0.25 + 0.0625 * (0.25 - 1) = -0.359375,
                 total 3.203125."""
        from tools.audit import rollouts

        def total(parts, critic):
            return float(parts.sum() + critic)

        parts, critic = rollouts.credit_gae(CREDIT_COMPONENTS, CREDIT_VALUES, 0.5, 0.0, terminal=False)
        assert parts.tolist() == [0.0, 1.0, 0.0, 0.0]
        assert critic == pytest.approx(-0.375)
        assert total(parts, critic) == pytest.approx(0.625)
        parts, critic = rollouts.credit_gae(CREDIT_COMPONENTS, CREDIT_VALUES, 0.5, 1.0, terminal=False)
        np.testing.assert_allclose(parts, [5.0, 1.0, 0.25, 0.0])
        assert critic == pytest.approx(-0.4375)
        assert total(parts, critic) == pytest.approx(5.8125)
        parts, critic = rollouts.credit_gae(CREDIT_COMPONENTS, CREDIT_VALUES, 0.5, 1.0, terminal=True)
        np.testing.assert_allclose(parts, [5.0, 1.0, 0.25, 0.0])
        assert critic == pytest.approx(-0.5)
        assert total(parts, critic) == pytest.approx(5.75)
        parts, critic = rollouts.credit_gae(CREDIT_COMPONENTS, CREDIT_VALUES, 0.5, 0.5, terminal=False)
        np.testing.assert_allclose(parts, [2.5, 1.0, 0.0625, 0.0])
        assert critic == pytest.approx(-0.359375)
        assert total(parts, critic) == pytest.approx(3.203125)

    def test_gae_reproduces_the_trainer_on_random_branches(self):
        """[RED] Against the trainer's backward recursion (`train_ppo.gae_advantages`, an independent implementation): the
        advantage at decision 0 of seeded random 75-step branches, bootstrapped from V_75, and of a branch that ends in a
        terminal step after 40 decisions, for lambda in {0, 0.5, 0.8, 0.95, 1}, gamma 0.995."""
        from football_rl.train_ppo import gae_advantages
        from tools.audit import rollouts

        rng = np.random.default_rng(20261122)
        for n, terminal in ((75, False), (40, True), (1, True), (1, False)):
            comps = rng.normal(size=(n, 4))
            values = rng.normal(size=n + 1)
            dones = np.zeros((n, 1))
            dones[-1, 0] = 1.0 if terminal else 0.0
            for lam in (0.0, 0.5, 0.8, 0.95, 1.0):
                want = gae_advantages(comps.sum(axis=1)[:, None], values[:n, None], dones, values[n:n + 1], 0.995, lam)[0, 0]
                parts, critic = rollouts.credit_gae(comps, values, 0.995, lam, terminal=terminal)
                assert float(parts.sum() + critic) == pytest.approx(float(want), rel=1e-10, abs=1e-12), (n, terminal, lam)

    def test_gae_rejects_degenerate_input(self):
        """[RED] No decisions, a lambda outside [0, 1] and a value vector shorter than n + 1 are errors; a terminal branch needs no
        V_n, so n values suffice there."""
        from tools.audit import rollouts

        with pytest.raises(ValueError):
            rollouts.credit_gae(np.zeros((0, 4)), np.zeros(1), 0.5, 0.5, terminal=False)
        for lam in (-0.01, 1.01, float("nan")):
            with pytest.raises(ValueError):
                rollouts.credit_gae(CREDIT_COMPONENTS, CREDIT_VALUES, 0.5, lam, terminal=False)
        with pytest.raises(ValueError):
            rollouts.credit_gae(CREDIT_COMPONENTS, CREDIT_VALUES[:3], 0.5, 0.5, terminal=False)
        parts, critic = rollouts.credit_gae(CREDIT_COMPONENTS, CREDIT_VALUES[:3], 0.5, 1.0, terminal=True)
        assert float(parts.sum() + critic) == pytest.approx(5.75)

    def test_step_components_hand_values(self):
        """[RED] One step of the kicker's reward split into goal, sparse, ball and role, advance weight 6:
          reward 13.0, shaping 0.25, one goal for, potential 0.10 -> 0.12: goal 10, sparse (13 - 0.25) - 10 = 2.75,
             ball 6 * 0.02 = 0.12, role 0.25 - 0.12 = 0.13.
          reward -10.5, shaping -0.5, one goal against, potential 0.30 -> 0.27: goal -10, sparse (-10.5 + 0.5) + 10 = 0,
             ball -0.18, role -0.5 + 0.18 = -0.32.
          a goal each way in one step: goal 0. No goal: goal 0 and sparse = reward - shaping."""
        from tools.audit import rollouts

        got = rollouts.credit_step_components(13.0, 0.25, 1, 0, 0.10, 0.12, 6.0)
        np.testing.assert_allclose(got, [10.0, 2.75, 0.12, 0.13], atol=1e-12)
        got = rollouts.credit_step_components(-10.5, -0.5, 0, 1, 0.30, 0.27, 6.0)
        np.testing.assert_allclose(got, [-10.0, 0.0, -0.18, -0.32], atol=1e-12)
        got = rollouts.credit_step_components(0.4, 0.1, 1, 1, 0.0, 0.0, 6.0)
        np.testing.assert_allclose(got, [0.0, 0.3, 0.0, 0.1], atol=1e-12)
        got = rollouts.credit_step_components(0.4, 0.1, 0, 0, 0.0, 0.0, 6.0)
        np.testing.assert_allclose(got, [0.0, 0.3, 0.0, 0.1], atol=1e-12)

    def test_branch_seed_hand_values(self):
        """[RED] Branch seed ((seed + 7919 j) * 1_000_003 + match * 10_000 + step) & 0x7FFFFFFF.
        seed 1, j 0, match 2, step 30: 1_000_003 + 20_000 + 30 = 1_020_033.
        seed 1, j 1: 7920 * 1_000_003 = 7_920_023_760, + 20_030 = 7_920_043_790, minus 3 * 2^31 = 1_477_592_846.
        With j 0 it is the seed of `rollouts aim`. Seeds of different j differ."""
        from tools.audit import rollouts

        assert rollouts.credit_branch_seed(1, 0, 2, 30) == 1_020_033
        assert rollouts.credit_branch_seed(1, 1, 2, 30) == 1_477_592_846
        assert rollouts.credit_branch_seed(5, 0, 3, 9) == (5 * 1_000_003 + 30_000 + 9) & 0x7FFF_FFFF
        assert len({rollouts.credit_branch_seed(1, j, 2, 30) for j in range(4)}) == 4


def _hand_branches():
    """2 points x 2 seeds x (recorded, alternative) x 1 decision x (goal, sparse, ball, role); V = 0; every branch is one terminal
    decision, so the advantage is the step reward whatever lambda. Totals (recorded / alternative):
      point 1: seeds (1, 3) / (2, 6)   all goal;     point 2: (0, 2) / (1, 2)   recorded: goal 0 / sparse 2, alternative: goal 1 / sparse 2."""
    comps = np.zeros((2, 2, 2, 1, 4))
    comps[0, :, 0, 0, 0] = (1.0, 3.0)
    comps[0, :, 1, 0, 0] = (2.0, 6.0)
    comps[1, 1, 0, 0, 1] = 2.0
    comps[1, 0, 1, 0, 0] = 1.0
    comps[1, 1, 1, 0, 1] = 2.0
    return comps, np.zeros((2, 2, 2, 2)), np.ones((2, 2, 2), dtype=np.int64), np.ones((2, 2, 2), dtype=bool)


class TestCreditStatistics:
    def test_signal_noise_and_shares_on_hand_arrays(self):
        """[RED] `_hand_branches`. Seed-averaged difference alternative - recorded: point 1 (1 + 3) / 2 = 2, point 2 (1 + 0) / 2 = 0.5.
        S = 1.25, SE = sd(2, 0.5) / sqrt(2) = (1.5 / sqrt(2)) / sqrt(2) = 0.75. Recorded branch across seeds: variances 2 and 2, so
        N = sqrt(2) = 1.4142136 and SNR = 1.25 / sqrt(2) = 0.8838835. Paired SD of the four differences (1, 3, 1, 0): mean 1.25,
        variance (0.0625 + 3.0625 + 0.0625 + 1.5625) / 3 = 1.5833333, sd 1.2583057. Variance shares: goal (2 + 0) / 2 = 1 of 2,
        sparse (0 + 2) / 2 = 1 of 2, the rest 0. Signal parts: goal 1.25, sparse 0. Same at every lambda (one terminal decision).
        With sign -1 (the benefit of the recorded kick over the alternative) the signal and its parts flip, noise stays."""
        from tools.audit import rollouts

        comps, vals, n_steps, terminal = _hand_branches()
        out = rollouts.credit_statistics(comps, vals, n_steps, terminal, lambdas=(0.8, 0.95), gamma=0.995, sign=1.0)
        assert list(out) == ["0.8", "0.95"]
        for key in out:
            r = out[key]
            assert r["S"] == pytest.approx(1.25)
            assert r["S_se"] == pytest.approx(0.75)
            assert r["N"] == pytest.approx(math.sqrt(2.0))
            assert r["SNR"] == pytest.approx(1.25 / math.sqrt(2.0))
            assert r["paired_sd"] == pytest.approx(math.sqrt(4.75 / 3.0))
            assert r["var_share"] == pytest.approx({"goal": 0.5, "sparse": 0.5, "ball": 0.0, "role": 0.0, "critic": 0.0})
            assert r["signal_parts"] == pytest.approx({"goal": 1.25, "sparse": 0.0, "ball": 0.0, "role": 0.0, "critic": 0.0})
        flipped = rollouts.credit_statistics(comps, vals, n_steps, terminal, lambdas=(0.8,), gamma=0.995, sign=-1.0)["0.8"]
        assert flipped["S"] == pytest.approx(-1.25) and flipped["S_se"] == pytest.approx(0.75)
        assert flipped["N"] == pytest.approx(math.sqrt(2.0)) and flipped["SNR"] == pytest.approx(-1.25 / math.sqrt(2.0))
        assert flipped["paired_sd"] == pytest.approx(math.sqrt(4.75 / 3.0))
        assert flipped["signal_parts"]["goal"] == pytest.approx(-1.25)

    def test_critic_part_enters_the_signal_and_the_noise(self):
        """[RED] One decision, non-terminal, gamma 0.5, V_0 = 0 and V_1 = 2 in the alternative branch of seed 0 only, no reward:
        the advantage there is 0.5 * 2 - 0 = 1 (critic part), all other branches 0. One point, two seeds: S = 0.5 (seed average of
        1 and 0), N = 0 (the recorded branch is constant), so SNR and the variance shares are None; the critic part carries the whole signal (0.5)."""
        from tools.audit import rollouts

        comps = np.zeros((1, 2, 2, 1, 4))
        vals = np.zeros((1, 2, 2, 2))
        vals[0, 0, 1, 1] = 2.0
        out = rollouts.credit_statistics(comps, vals, np.ones((1, 2, 2), dtype=np.int64), np.zeros((1, 2, 2), dtype=bool),
                                         lambdas=(0.5,), gamma=0.5, sign=1.0)["0.5"]
        assert out["S"] == pytest.approx(0.5) and out["N"] == 0.0 and out["SNR"] is None
        assert out["S_se"] is None
        assert out["signal_parts"]["critic"] == pytest.approx(0.5)
        assert all(v is None for v in out["var_share"].values())

    def test_degenerate_point_and_seed_counts(self):
        """[RED] No points: every statistic is None. One point: S defined, its SE None. One seed: an error (the noise needs two)."""
        from tools.audit import rollouts

        comps, vals, n_steps, terminal = _hand_branches()
        none = rollouts.credit_statistics(comps[:0], vals[:0], n_steps[:0], terminal[:0], lambdas=(0.8,), gamma=0.995, sign=1.0)["0.8"]
        assert all(none[k] is None for k in ("S", "S_se", "N", "SNR", "paired_sd"))
        assert all(v is None for v in none["var_share"].values()) and all(v is None for v in none["signal_parts"].values())
        one = rollouts.credit_statistics(comps[:1], vals[:1], n_steps[:1], terminal[:1], lambdas=(0.8,), gamma=0.995, sign=1.0)["0.8"]
        assert one["S"] == pytest.approx(2.0) and one["S_se"] is None and one["N"] == pytest.approx(math.sqrt(2.0))
        with pytest.raises(ValueError):
            rollouts.credit_statistics(comps[:, :1], vals[:, :1], n_steps[:, :1], terminal[:, :1], lambdas=(0.8,), gamma=0.995, sign=1.0)


class TestCreditSummary:
    @staticmethod
    def _records(goals):
        comps, vals, n_steps, terminal = _hand_branches()
        return [
            {"point": {"match_idx": 0, "step": 10 * p, "kicker": 3}, "components": comps[p], "values": vals[p],
             "n_steps": n_steps[p], "terminal": terminal[p], "goals": goals[p]}
            for p in range(2)
        ]

    def test_sign_convention_and_goal_rates_on_hand_records(self):
        """[RED] The hand arrays of `_hand_branches` as records. aim: S = A_AIM - A_POLICY = +1.25. attack: S = A_KICK - A_HOLD, the
        recorded branch minus the alternative = -1.25, with the noise of the recorded branch unchanged (sqrt 2). Goal counts per
        point, seed and branch (for, against), recorded branch first: point 1 seed 0 recorded (1, 0) and alternative (0, 1), seed 1
        recorded (0, 0) and alternative (0, 1); point 2 seed 0 alternative (0, 2), everything else 0. Means over the 4 (point, seed)
        pairs: recorded for 1/4, against 0; alternative for 0, against (1 + 1 + 2) / 4 = 1. Aim reports no goal rates."""
        from tools.audit import rollouts

        goals = np.zeros((2, 2, 2, 2), dtype=np.int64)
        goals[0, 0, 0, 0] = 1
        goals[0, 0, 1, 1] = 1
        goals[0, 1, 1, 1] = 1
        goals[1, 0, 1, 1] = 2
        aim = rollouts.summarize_credit(self._records(goals), 2, 2, 0, 0.0, decision="aim", seed=1, seeds=2, lambdas=[0.8])
        attack = rollouts.summarize_credit(self._records(goals), 2, 2, 0, 0.0, decision="attack", seed=1, seeds=2, lambdas=[0.8])
        assert aim["estimators"]["0.8"]["S"] == pytest.approx(1.25) and attack["estimators"]["0.8"]["S"] == pytest.approx(-1.25)
        assert aim["estimators"]["0.8"]["N"] == pytest.approx(math.sqrt(2.0)) == pytest.approx(attack["estimators"]["0.8"]["N"])
        assert "goals_5s" not in aim
        assert attack["goals_5s"] == {"kick": {"for": 0.25, "against": 0.0}, "hold": {"for": 0.0, "against": 1.0}}
        assert [p["step"] for p in attack["points"]] == [0, 10] and attack["n_points"] == 2
        assert attack["steps"] == 75 and attack["lambdas"] == [0.8]
        empty = rollouts.summarize_credit([], 2, 2, 0, None, decision="attack", seed=1, seeds=3, lambdas=[0.8, 0.95])
        assert empty["goals_5s"] == {"kick": {"for": None, "against": None}, "hold": {"for": None, "against": None}}
        assert empty["estimators"]["0.95"]["S"] is None and empty["points"] == []


class TestCreditPoints:
    def test_attack_point_selection(self):
        """[RED] Outfield kicks of a 300-row trace (ball at row = canonical x; side 0 for team A, 1 for team B):
          5   A3, ball x 1000                      -> point
          30  A3, 1000, 25 after 5                 -> skipped
          40  A3, ball x 600 (middle third)        -> no point; the spacing counts from the last point
          50  A3, 1000, 45 after 5                 -> point (exactly 45 is enough)
          100 A3, ball x 933.0 (< 2/3 of 1400)     -> no point
          110 B9, ball x 400 = canonical 1000      -> point (60 after 50)
          154 A3, 1000, 44 after 110               -> skipped
          160 A3, ball x 934 (> 933.33)            -> point (50 after 110; the skipped kick at 154 did not move the spacing)
          190 A0 (keeper) kicks at 1000            -> no point
          200 A3, 1000, but nobody holds the ball  -> no point
          297 A3, 1000                             -> point in a 300-row trace (297 + 2 < 300), not in a 299-row one."""
        from tools.audit import rollouts

        def scene(T):
            tr = make_trace(T)
            for t0, kicker, x in ((5, 3, 1000.0), (30, 3, 1000.0), (40, 3, 600.0), (50, 3, 1000.0), (100, 3, 933.0), (110, 9, 400.0),
                                  (154, 3, 1000.0), (160, 3, 934.0), (190, 0, 1000.0), (200, 3, 1000.0), (297, 3, 1000.0)):
                if t0 + 1 < T:
                    pass_scene(tr, t0, kicker=kicker, ball=(x, 420.0))
            tr["holder"][200] = -1
            return tr

        pts = rollouts.find_attack_points(scene(300))
        assert [(p["step"], p["kicker"], p["team"]) for p in pts] == [(5, 3, 0), (50, 3, 0), (110, 9, 1), (160, 3, 0), (297, 3, 0)]
        assert [p["step"] for p in rollouts.find_attack_points(scene(299))] == [5, 50, 110, 160]
        assert rollouts.find_attack_points(make_trace(0)) == []

    def test_attack_points_on_a_real_trace_are_kicks_of_the_holder(self, tmp_path):
        """[RED] The attack fixture's match 1 has the two hand-checked points (rows 548 and 630, player 9)."""
        from tools.audit import rollouts

        _, _, traces = _attack_fixture(tmp_path)
        assert rollouts.find_attack_points(traces[0]) == []
        assert [(p["step"], p["kicker"]) for p in rollouts.find_attack_points(traces[1])] == [(548, 9), (630, 9)]

    def test_override_scope_of_the_alternative_branches(self, tmp_path):
        """[RED] Constant hybrid5 checkpoint (z_kick +10, so the policy fires almost always; its outputs do not depend on the
        observation), sigma 0, kicker A5 holding at (700, 420). Per seed j, the actions applied in the 75 decisions:
          attack: KICK plays the recorded actions at decision 0 and the policy afterwards; HOLD differs from KICK only in the
            kicker's kick column, which is -1 for decisions 0-14 and equal to KICK's from decision 15 on (whole rows equal).
          aim: POLICY plays the recorded actions at decision 0; AIM differs only in the kicker's steer at decision 0 (0.5),
            everything else equal in all 75 decisions."""
        from tools.audit import rollouts

        ckpt = write_constant_hybrid_checkpoint(tmp_path / "c", z_kick=10.0, z_spec=-10.0)
        team_a, team_b = _credit_policies(ckpt)
        rec = _credit_recorded(5, steer=0.2, kick=1.0)
        logs: dict = {}

        def hook(label, j, k, env, act):
            logs.setdefault((label, j), []).append(np.array(act, dtype=np.float32, copy=True))

        def acts(label, j):
            arr = np.stack(logs[(label, j)])
            assert arr.shape == (75, 12, 4)
            return arr

        snap = _credit_scene(5, (700.0, 420.0), 0.0, (724.0, 420.0))
        out = rollouts.roll_credit_point(snap, decision="attack", recorded_actions=rec, kicker=5, branch_seeds=[41, 42],
                                         team_a=team_a, team_b=team_b, sigma_scale=0.0, after_step=hook)
        assert out["components"].shape == (2, 2, 75, 4) and out["values"].shape == (2, 2, 76)
        for j in (0, 1):
            kick, hold = acts("kick", j), acts("hold", j)
            np.testing.assert_array_equal(kick[0], rec)
            assert (hold[:15, 5, 2] == -1.0).all()
            assert (kick[15:20, 5, 2] > -1.0).all()                      # the policy fires, so a 16th held decision would show
            np.testing.assert_array_equal(hold[15:], kick[15:])
            differs = kick != hold
            expected = np.zeros_like(differs)
            expected[:15, 5, 2] = kick[:15, 5, 2] != -1.0
            np.testing.assert_array_equal(differs, expected)
            assert expected[0, 5, 2] and expected[1:15, 5, 2].any()

        logs.clear()
        snap = _credit_scene(5, (700.0, 420.0), 0.0, (724.0, 420.0))
        rollouts.roll_credit_point(snap, decision="aim", recorded_actions=rec, kicker=5, branch_seeds=[41, 42],
                                   team_a=team_a, team_b=team_b, steer=0.5, sigma_scale=0.0, after_step=hook)
        for j in (0, 1):
            pol, aim = acts("policy", j), acts("aim", j)
            np.testing.assert_array_equal(pol[0], rec)
            expected = np.zeros(aim.shape, dtype=bool)
            expected[0, 5, 1] = True
            np.testing.assert_array_equal(pol != aim, expected)
            assert aim[0, 5, 1] == np.float32(0.5)

    def test_policy_branch_replays_the_trace_and_the_aim_policy_branch(self, tmp_path):
        """[RED] On the hybrid5 fixture point (match 0, kick at row 2 by A5): the credit POLICY branch of seed index 0 plays the
        trace's kick (ball velocity after decision 0 equals the trace's bvel[3] within 1e-4), and every ball and player position of
        its 75 decisions equals the `rollouts aim` POLICY branch run from the same generator seed."""
        from tools.audit import rollouts

        _, _, tr, p, snap, (team_a, team_b) = TestAimRollouts._fixture_point(tmp_path)
        seed = rollouts.credit_branch_seed(5, 0, 0, p["step"])
        assert seed == (5 * 1_000_003 + 0 * 10_000 + p["step"]) & 0x7FFF_FFFF
        credit_traj, aim_traj = [], []

        def credit_hook(label, j, k, env, act):
            if label == "policy":
                credit_traj.append((env.ball_pos.copy(), env.player_pos.copy(), env.ball_vel.copy()))

        def aim_hook(branch, k, env, act):
            if branch == "policy":
                aim_traj.append((env.ball_pos.copy(), env.player_pos.copy(), env.ball_vel.copy()))

        rollouts.roll_credit_point(snap, decision="aim", recorded_actions=tr["act"][p["step"]], kicker=p["kicker"], branch_seeds=[seed],
                                   team_a=team_a, team_b=team_b, steer=rollouts.aim_steer(p["deflection"]), after_step=credit_hook)
        rollouts.roll_aim_branches(snap, recorded_actions=tr["act"][p["step"]], kicker=p["kicker"], target=p["target"], branch_seed=seed,
                                   team_a=team_a, team_b=team_b, steer=rollouts.aim_steer(p["deflection"]), power=0.5, after_step=aim_hook)
        assert len(credit_traj) == len(aim_traj) == 75
        np.testing.assert_allclose(credit_traj[0][2], tr["bvel"][p["step"] + 1], atol=1e-4)
        for (cb, cp, cv), (ab, ap, av) in zip(credit_traj, aim_traj):
            np.testing.assert_array_equal(cb, ab)
            np.testing.assert_array_equal(cp, ap)
            np.testing.assert_array_equal(cv, av)


class TestCreditArguments:
    def test_aim_points_exclude_kicks_that_are_already_aimed(self):
        """[RED] The `aim` points of `TestAimRollouts.test_point_selection_spacing_target_and_reach` are rows 5 (already aimed: A4 at
        +10 deg, direction 0) and 60 (not aimed); the credit points of the aim decision are row 60 only. An unknown decision is an
        error."""
        from tools.audit import rollouts

        tr = _aim_hand_trace(200)
        aim_scene(tr, 5, {4: 10.0, 5: -30.0})
        aim_scene(tr, 60, {4: -34.5})
        assert [p["step"] for p in rollouts.find_aim_points(tr)] == [5, 60]
        assert [(p["step"], p["kicker"], p["target"]) for p in rollouts.find_credit_points(tr, "aim")] == [(60, 3, 4)]
        with pytest.raises(ValueError):
            rollouts.find_credit_points(tr, "shoot")

    def test_run_rejects_bad_arguments(self, tmp_path):
        """[RED] `run_credit` refuses fewer than 2 seeds, an empty or out-of-range lambda list, an unknown decision and no workers
        before it looks at any trace: with a trace directory that does not exist the error is a ValueError, and with good arguments
        the same call gets as far as the missing traces (FileNotFoundError)."""
        from tools.audit import rollouts

        ckpt, _, _ = _hold_fixture(tmp_path, "hybrid5")
        base = dict(trace_dir=tmp_path / "no_such_traces", checkpoint_dir=ckpt, trace_seed=11)
        for bad in (dict(seeds=1), dict(lambdas=[]), dict(lambdas=[0.5, 1.5]), dict(lambdas=[-0.1]), dict(decision="shoot"), dict(workers=0)):
            with pytest.raises(ValueError):
                rollouts.run_credit(**base, **bad)
        with pytest.raises(FileNotFoundError):
            rollouts.run_credit(**base)


class TestCreditBranch:
    def test_values_are_the_kickers_critic_with_a_bootstrap_at_the_end(self, tmp_path):
        """[RED] On the fixture point (a 120-row trace, kick at row 2: no terminal step within 75 decisions) the recorded values
        V_0..V_75 are the kicker's critic at the state before each decision and, after the last one, at the final state (the
        bootstrap). The oracle evaluates the harness team policy on the states `after_step` sees. V_75 is non-zero, the branch is
        not terminal, and the step count is 75."""
        from tools.audit import rollouts

        _, _, tr, p, snap, (team_a, team_b) = TestAimRollouts._fixture_point(tmp_path)
        kicker = p["kicker"]
        gen = torch.Generator().manual_seed(0)

        def critic(env):
            obs, cen = env._get_observations_array(), env.get_centralized_state()
            _, val, _ = team_a.act(obs[:6], cen[:6], "sampled", 1.0, gen)
            return float(val[kicker])

        oracle = [critic(snap)]

        def hook(k, env, act):
            oracle.append(critic(env))

        res = rollouts.roll_credit_branch(snap, recorded_actions=tr["act"][p["step"]], kicker=kicker, branch_seed=321,
                                          team_a=team_a, team_b=team_b, after_step=hook)
        assert res["n_steps"] == 75 and res["terminal"] is False
        assert res["values"].shape == (76,) and len(oracle) == 76
        np.testing.assert_allclose(res["values"], oracle, atol=1e-6)
        assert abs(res["values"][75]) > 1e-3

    def test_goal_split_on_scored_and_conceded_goals(self, tmp_path):
        """[RED] Empty-net scenes, constant checkpoint that never kicks (z_kick -10), kicker A5 (team A), sigma 0.
        Team A's seat 5 holds at (1300, 420) facing the goal and kicks into the empty net at decision 0: one goal for, 13 = 10 (team)
        + 3 (scorer) in the goal decision, so goal = +10 and sparse = +3 there. Team B's seat 5 (player 11) holds at (100, 420)
        facing the other goal and scores against A: goal = -10 and sparse = 0 for the kicker. The goal sums are +10 and -10. In the
        goal decision the env pays no shaping (the ball teleports to the centre): ball + role = 0 there, whatever the ball
        potential did. The four parts add up to the kicker's step reward in every decision."""
        from tools.audit import rollouts

        ckpt = write_constant_hybrid_checkpoint(tmp_path / "c", z_kick=-10.0, z_spec=-10.0)
        team_a, team_b = _credit_policies(ckpt)
        for holder, pos, angle, ball, want_goals, want_goal, want_sparse in (
            (5, (1300.0, 420.0), 0.0, (1324.0, 420.0), (1, 0), 10.0, 3.0),
            (11, (100.0, 420.0), math.pi, (76.0, 420.0), (0, 1), -10.0, 0.0),
        ):
            rewards = []

            def hook(k, env, act):
                rewards.append(float(env.last_step_rewards[5]))

            res = rollouts.roll_credit_branch(
                _credit_scene(holder, pos, angle, ball), recorded_actions=_credit_recorded(holder), kicker=5,
                branch_seed=7, team_a=team_a, team_b=team_b, sigma_scale=0.0, after_step=hook)
            assert (res["goals_for"], res["goals_against"]) == want_goals
            comps = res["components"]
            assert comps[:, 0].sum() == pytest.approx(want_goal)
            assert comps[1, 0] == pytest.approx(want_goal)
            assert comps[1, 1] == pytest.approx(want_sparse)
            assert comps[1, 2] + comps[1, 3] == pytest.approx(0.0, abs=1e-9)
            np.testing.assert_allclose(comps.sum(axis=1), rewards, atol=1e-9)

    def test_ball_part_is_the_kicker_teams_territorial_gain(self, tmp_path):
        """[RED] A weak kick (action 0.4, about 200 u/s: it stops within 125 u) that stays in play: team A's seat 5 kicks east from
        (800, 420), then team B's seat 5 (player 11) kicks west from (600, 420). The ball moves toward the attacked goal of team A in the first and of team B in the second, so the kicker's
        ball part in decision 0 is positive in the first scene and negative in the second."""
        from tools.audit import rollouts

        ckpt = write_constant_hybrid_checkpoint(tmp_path / "c", z_kick=-10.0, z_spec=-10.0)
        team_a, team_b = _credit_policies(ckpt)
        signs = []
        for holder, pos, angle, ball in ((5, (800.0, 420.0), 0.0, (824.0, 420.0)), (11, (600.0, 420.0), math.pi, (576.0, 420.0))):
            res = rollouts.roll_credit_branch(
                _credit_scene(holder, pos, angle, ball), recorded_actions=_credit_recorded(holder, kick=0.4), kicker=5,
                branch_seed=7, team_a=team_a, team_b=team_b, sigma_scale=0.0)
            assert res["goals_for"] == res["goals_against"] == 0
            signs.append(res["components"][0, 2])
        assert signs[0] > 1e-3 and signs[1] < -1e-3


class TestCreditTerminal:
    def test_branch_that_ends_stops_and_reads_no_bootstrap(self, tmp_path):
        """[GUARD] A match with 10 decisions left ends in the 10th decision of the branch: 10 decisions are played, the branch is
        terminal, nothing is recorded after them (components and values zero) and no V_10 is read. A match with exactly 75 decisions
        left ends in the last decision of the branch: 75 decisions, terminal, and V_75 is not read (0). With 76 left, the branch is
        not terminal and V_75 is the bootstrap."""
        from tools.audit import rollouts

        ckpt = write_constant_hybrid_checkpoint(tmp_path / "c", z_kick=-10.0, z_spec=-10.0)
        team_a, team_b = _credit_policies(ckpt)
        for left, n_steps, terminal in ((10, 10, True), (75, 75, True), (76, 75, False)):
            snap = _credit_scene(5, (800.0, 420.0), 0.0, (824.0, 420.0))
            snap.rl_step_count = snap.cfg.physics.max_rl_steps - left
            res = rollouts.roll_credit_branch(snap, recorded_actions=_credit_recorded(5, kick=0.4), kicker=5, branch_seed=7,
                                              team_a=team_a, team_b=team_b, sigma_scale=0.0)
            assert (res["n_steps"], res["terminal"]) == (n_steps, terminal), left
            assert (res["components"][n_steps:] == 0.0).all() and (res["values"][n_steps + (0 if terminal else 1):] == 0.0).all()
            assert (res["values"][:n_steps] != 0.0).all()
            assert (res["values"][75] != 0.0) == (not terminal)


class TestCreditCli:
    def test_aim_decision_end_to_end(self, tmp_path):
        """[RED] `rollouts credit` (default decision aim) on the hybrid5 fixture, 2 seeds: one point; JSON and report written; the
        same JSON with 1 and 2 workers; the noise is positive (the seeds differ per index); inputs untouched; protected outputs
        refused; a missing --checkpoint-dir, one seed and a lambda outside [0, 1] are argparse errors."""
        from tools.audit import rollouts

        ckpt, trace_dir, _ = _hold_fixture(tmp_path, "hybrid5")
        digests = (_dir_digest(ckpt), _dir_digest(trace_dir))
        out1, out2 = tmp_path / "credit1", tmp_path / "credit2.json"
        base = ["credit", "--trace-dir", str(trace_dir), "--checkpoint-dir", str(ckpt), "--trace-seed", "11", "--seed", "5",
                "--seeds", "2", "--shaping-mode", "legacy", "--lambdas", "0", "0.8", "0.95"]
        assert rollouts.main([*base, "--workers", "1", "--out", str(out1)]) == 0
        assert rollouts.main([*base, "--workers", "2", "--max-matches", "2", "--out", str(out2)]) == 0
        d1 = json.loads((out1 / "rollouts.json").read_text(encoding="utf-8"))
        d2 = json.loads(out2.read_text(encoding="utf-8"))
        assert d1 == d2
        assert (d1["mode"], d1["decision"], d1["seed"], d1["seeds"], d1["steps"], d1["shaping_mode"]) == ("credit", "aim", 5, 2, 75, "legacy")
        assert (d1["n_matches_ok"], d1["n_points"]) == (2, 1)
        assert d1["gamma"] == pytest.approx(float(DEFAULT_CONFIG.ppo.gamma))
        assert d1["points"] == [{"match_idx": 0, "step": 2, "kicker": 5, "target": 4}]
        assert list(d1["estimators"]) == ["0", "0.8", "0.95"]
        assert d1["estimators"]["0.95"]["N"] > 0.0
        assert d1["estimators"]["0"]["N"] == pytest.approx(0.0, abs=1e-6)       # the decision step is deterministic given the actions
        assert "goals_5s" not in d1
        text = (out1 / "report.txt").read_text(encoding="utf-8")
        assert text.startswith("===== rollouts credit") and "decision=aim" in text and "0.95 |" in text
        assert (_dir_digest(ckpt), _dir_digest(trace_dir)) == digests
        for bad in (trace_dir / "bad.json", ckpt / "bad.json"):
            with pytest.raises(ValueError):
                rollouts.main([*base, "--out", str(bad)])
        for extra in (["--seeds", "1"], ["--lambdas", "1.5"], ["--lambdas", "-0.1"], ["--decision", "shoot"]):
            with pytest.raises(SystemExit):
                rollouts.main([*base, *extra, "--out", str(tmp_path / "x.json")])
        with pytest.raises(SystemExit):
            rollouts.main(["credit", "--trace-dir", str(trace_dir), "--out", str(tmp_path / "x.json")])

    def test_attack_decision_end_to_end(self, tmp_path):
        """[RED] `rollouts credit --decision attack` on the attack fixture (2 points in match 1), 2 seeds: the points, the goal counts
        within 5 s per branch (KICK and HOLD, for and against, in [0, 1] as per-point means), the sign convention (S is the benefit
        of the recorded kick over holding), identical JSON with 1 and 2 workers; and, on the hybrid5 fixture that has no
        attacking-third kick, an empty result that is reported, not an error."""
        from tools.audit import rollouts

        ckpt, trace_dir, _ = _attack_fixture(tmp_path)
        base = ["credit", "--decision", "attack", "--trace-dir", str(trace_dir), "--checkpoint-dir", str(ckpt), "--trace-seed", "11",
                "--seed", "3", "--seeds", "2", "--shaping-mode", "legacy", "--lambdas", "0.8", "0.95"]
        assert rollouts.main([*base, "--workers", "1", "--out", str(tmp_path / "a1.json")]) == 0
        assert rollouts.main([*base, "--workers", "2", "--out", str(tmp_path / "a2.json")]) == 0
        d1 = json.loads((tmp_path / "a1.json").read_text(encoding="utf-8"))
        d2 = json.loads((tmp_path / "a2.json").read_text(encoding="utf-8"))
        assert d1 == d2
        assert (d1["decision"], d1["n_points"], d1["n_matches_ok"]) == ("attack", 2, 2)
        assert d1["points"] == [{"match_idx": 1, "step": 548, "kicker": 9}, {"match_idx": 1, "step": 630, "kicker": 9}]
        assert set(d1["goals_5s"]) == {"kick", "hold"}
        for branch in ("kick", "hold"):
            assert set(d1["goals_5s"][branch]) == {"for", "against"}
            assert all(0.0 <= v <= 1.0 for v in d1["goals_5s"][branch].values())
        assert d1["estimators"]["0.95"]["N"] > 0.0

        hold_ckpt, hold_traces, _ = _hold_fixture(tmp_path / "hold", "hybrid5")
        empty = tmp_path / "empty"
        assert rollouts.main(["credit", "--decision", "attack", "--trace-dir", str(hold_traces), "--checkpoint-dir", str(hold_ckpt),
                              "--trace-seed", "11", "--seeds", "2", "--out", str(empty)]) == 0
        d0 = json.loads((empty / "rollouts.json").read_text(encoding="utf-8"))
        assert d0["n_points"] == 0 and d0["points"] == []
        assert all(d0["estimators"]["0.95"][k] is None for k in ("S", "N", "SNR"))
        assert "no points" in (empty / "report.txt").read_text(encoding="utf-8")

    def test_report_layout_on_a_hand_summary(self):
        """[RED] The text report lists one row per lambda in the order given, with S +- SE, N, SNR, the paired SD and the variance
        shares and signal parts of goal / sparse / ball / role / critic; the goal line appears for attack only."""
        from tools.audit import rollouts

        comps, vals, n_steps, terminal = _hand_branches()
        stats = rollouts.credit_statistics(comps, vals, n_steps, terminal, lambdas=(0.8, 0.95), gamma=0.995, sign=1.0)
        summary = {"mode": "credit", "decision": "aim", "seed": 1, "seeds": 2, "steps": 75, "shaping_mode": "legacy",
                   "gamma": 0.995, "lambdas": [0.8, 0.95], "n_matches_total": 2, "n_matches_ok": 2, "n_matches_skipped": 0,
                   "max_pos_err": 0.0, "n_points": 2, "points": [], "estimators": stats}
        lines = rollouts.format_credit_report(summary, title="hand").splitlines()
        assert lines[0].startswith("===== rollouts credit hand  decision=aim matches ok=2/2 skipped=0 points=2 seeds=2 seed=1")
        rows = [ln for ln in lines if ln.startswith(("0.8 |", "0.95 |"))]
        assert len(rows) == 2 and rows[0].startswith("0.8 | +1.2500 +- 0.7500 | 1.4142 | +0.884 | 1.258 |")
        assert "0.50/0.50/0.00/0.00/0.00" in rows[0] and "+1.250/+0.000/+0.000/+0.000/+0.000" in rows[0]
        assert not any(ln.startswith("goals within 5 s") for ln in lines)
