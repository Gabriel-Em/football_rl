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
