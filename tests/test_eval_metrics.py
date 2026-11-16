"""Specification tests for the evaluation harness (`tools/audit/harness.py`) and the behaviour
census (`tools/audit/census.py`).

Census tests drive the metrics with tiny hand-built traces whose answers are solved by hand.
Trace layout (one row per decision; row t holds the state *before* step t's action, except the
cumulative `s_*` counters and `lastkick`, which are read *after* step t):
  pos (T,12,2)  ball (T,2)  bvel (T,2)  holder (T,) -1 = loose  side (T,2) 0 = defends left
  box (T,2) active box radius per side  lastkick (T,12) -999 = never  s_score (T,2)
  s_team_passes (T,2)  s_own_goals (T,)
Players 0..5 are team A (GK, D1, D2, M1, M2, ST), 6..11 team B.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from football_rl.train_ppo import ActorCritic  # noqa: E402
from tools.audit import census, harness  # noqa: E402

STEP_S = 1.0 / 15.0


# ----------------------------------------------------------------------------
# Hand-built trace helpers
# ----------------------------------------------------------------------------
def make_trace(T: int) -> dict:
    far = np.zeros((T, 12, 2), dtype=np.float32)
    far[:, :, 0] = 700.0
    far[:, :, 1] = 800.0 - 10.0 * np.arange(12)[None, :]
    return {
        "pos": far,
        "ball": np.tile(np.array([700.0, 20.0], dtype=np.float32), (T, 1)),
        "bvel": np.zeros((T, 2), dtype=np.float32),
        "holder": np.full(T, -1, dtype=np.int8),
        "side": np.tile(np.array([0, 1], dtype=np.int8), (T, 1)),
        "box": np.full((T, 2), 195.0, dtype=np.float32),
        "lastkick": np.full((T, 12), -999.0, dtype=np.float32),
        "s_score": np.zeros((T, 2), dtype=np.int32),
        "s_team_passes": np.zeros((T, 2), dtype=np.int32),
        "s_own_goals": np.zeros(T, dtype=np.int32),
        "n_steps": np.array(T),
    }


def set_kick(trace: dict, step: int, player: int, time_s: float = 1.0) -> None:
    """Record a kick by `player` during `step` (last_kick_time changes at that step and stays)."""
    trace["lastkick"][step:, player] = time_s + step * STEP_S


# ----------------------------------------------------------------------------
# Possession metrics
# ----------------------------------------------------------------------------
class TestPossession:
    def test_hold_and_loose_share(self):
        tr = make_trace(11)
        tr["holder"][:] = np.array([-1, 3, 3, 3, -1, 4, 4, -1, -1, 3, 6], dtype=np.int8)
        np.testing.assert_allclose(census.hold_share(tr), [6 / 11, 1 / 11])
        assert census.loose_share(tr) == pytest.approx(4 / 11)

    def test_shares_sum_to_one(self):
        tr = make_trace(9)
        tr["holder"][:] = np.array([0, 5, 6, 11, -1, -1, 2, 7, -1], dtype=np.int8)
        assert census.hold_share(tr).sum() + census.loose_share(tr) == pytest.approx(1.0)

    def test_holder_spells_split_on_loose_and_on_holder_change(self):
        tr = make_trace(10)
        tr["holder"][:] = np.array([-1, 3, 3, 3, -1, 4, 4, 5, 5, 5], dtype=np.int8)
        players, seconds = census.holder_spells(tr)
        assert players.tolist() == [3, 4, 5]
        np.testing.assert_allclose(seconds, np.array([3, 2, 3]) * STEP_S)

    def test_holder_spells_empty_when_ball_always_loose(self):
        players, seconds = census.holder_spells(make_trace(5))
        assert players.size == 0 and seconds.size == 0

    def test_single_step_trace(self):
        tr = make_trace(1)
        tr["holder"][0] = 2
        players, seconds = census.holder_spells(tr)
        assert players.tolist() == [2]
        np.testing.assert_allclose(seconds, [STEP_S])


# ----------------------------------------------------------------------------
# Kick outcomes
# ----------------------------------------------------------------------------
class TestKickOutcome:
    def _one_kick(self, holder_at: dict, T: int = 140, t0: int = 5, player: int = 3, score_at=None):
        tr = make_trace(T)
        tr["holder"][t0] = player
        set_kick(tr, t0, player)
        for step, who in holder_at.items():
            tr["holder"][step] = who
        if score_at is not None:
            step, team = score_at
            tr["s_score"][step:, team] += 1
        return census.kick_outcomes(tr)

    def test_teammate_reception(self):
        out = self._one_kick({15: 4})                     # A3 kicks at 5, A4 holds at t0+10
        assert [(k["player"], k["outcome"]) for k in out] == [(3, "teammate")]

    def test_opponent_reception(self):
        out = self._one_kick({15: 7})                     # B1 (global 7)
        assert out[0]["outcome"] == "opponent"

    def test_none_after_61_quiet_steps(self):
        out = self._one_kick({})
        assert out[0]["outcome"] == "none"

    def test_window_boundary_is_60_steps(self):
        assert self._one_kick({65: 4})[0]["outcome"] == "teammate"   # t0 + 60 inclusive
        assert self._one_kick({66: 4})[0]["outcome"] == "none"       # t0 + 61 is outside

    def test_same_player_recovers_after_eight_steps(self):
        assert self._one_kick({14: 3})[0]["outcome"] == "self"       # t0 + 9
        assert self._one_kick({13: 3})[0]["outcome"] == "none"       # t0 + 8 is still the release

    def test_goal_beats_a_later_reception(self):
        out = self._one_kick({35: 7}, score_at=(25, 0))
        assert out[0]["outcome"] == "goal"

    def test_goal_scored_in_the_kick_step_counts(self):
        out = self._one_kick({}, score_at=(5, 0))
        assert out[0]["outcome"] == "goal"

    def test_first_event_wins(self):
        out = self._one_kick({20: 4, 30: 7})
        assert out[0]["outcome"] == "teammate"

    def test_kick_in_final_window_without_outcome_is_censored(self):
        tr = make_trace(40)
        tr["holder"][30] = 3
        set_kick(tr, 30, 3)
        assert census.kick_outcomes(tr) == []

    def test_kick_in_final_window_with_outcome_is_kept(self):
        tr = make_trace(40)
        tr["holder"][30] = 3
        set_kick(tr, 30, 3)
        tr["holder"][35] = 4
        assert census.kick_outcomes(tr)[0]["outcome"] == "teammate"

    def test_kick_on_the_first_step_is_detected(self):
        tr = make_trace(100)
        tr["holder"][0] = 3
        set_kick(tr, 0, 3)
        tr["holder"][10] = 4
        out = census.kick_outcomes(tr)
        assert len(out) == 1 and out[0]["step"] == 0

    def test_team_b_kick_and_goalkeeper_slot(self):
        tr = make_trace(100)
        tr["holder"][5] = 6
        set_kick(tr, 5, 6)                                # B's keeper
        tr["holder"][12] = 9                              # B3: teammate of B
        out = census.kick_outcomes(tr)
        assert (out[0]["team"], out[0]["slot"], out[0]["outcome"]) == (1, 0, "teammate")

    def test_summary_counts_only_outfield_kicks(self):
        tr = make_trace(200)
        for t0, who, recv in ((5, 3, 4), (30, 1, 8), (60, 0, 2)):
            tr["holder"][t0] = who
            set_kick(tr, t0, who)
            tr["holder"][t0 + 10] = recv
        counts = census.outfield_outcome_counts(census.kick_outcomes(tr))
        assert counts == {"teammate": 1, "opponent": 1, "self": 0, "none": 0, "goal": 0, "n": 2}

    def test_halftime_reset_of_last_kick_is_not_a_kick(self):
        tr = make_trace(100)
        tr["lastkick"][10:50, 3] = 20.0
        tr["lastkick"][50:, 3] = -999.0
        assert [k["step"] for k in census.kick_outcomes(tr)] == [10]


# ----------------------------------------------------------------------------
# Geometry metrics
# ----------------------------------------------------------------------------
def team_a_geometry(offsets, T=3, ball=(0.0, 0.0)):
    tr = make_trace(T)
    tr["ball"][:] = ball
    for slot, (dx, dy) in zip(range(1, 6), offsets):
        tr["pos"][:, slot] = (ball[0] + dx, ball[1] + dy)
    return tr


class TestNWithin:
    OFFSETS = [(50, 0), (0, 80), (120, 0), (200, 0), (0, 300)]

    def test_hand_solved_counts(self):
        tr = team_a_geometry(self.OFFSETS)
        assert census.n_within(tr, 100.0)[0] == pytest.approx(2.0)
        assert census.n_within(tr, 150.0)[0] == pytest.approx(3.0)
        assert census.n_within(tr, 40.0)[0] == pytest.approx(0.0)
        assert census.n_within(tr, 1000.0)[0] == pytest.approx(5.0)

    def test_radius_is_strict(self):
        tr = team_a_geometry([(100, 0), (0, 300), (0, 400), (0, 500), (0, 600)])
        assert census.n_within(tr, 100.0)[0] == 0.0
        assert census.n_within(tr, 100.001)[0] == 1.0

    def test_goalkeeper_is_not_counted(self):
        tr = team_a_geometry([(500, 0)] * 5)
        tr["pos"][:, 0] = (10.0, 0.0)
        assert census.n_within(tr, 100.0)[0] == 0.0

    def test_mean_over_decisions(self):
        tr = team_a_geometry(self.OFFSETS, T=4)
        tr["pos"][2:, 1] = (900.0, 900.0)                 # player 1 leaves for the last two decisions
        assert census.n_within(tr, 100.0)[0] == pytest.approx((2 + 2 + 1 + 1) / 4)

    def test_teams_are_independent(self):
        tr = team_a_geometry(self.OFFSETS)
        for slot in range(7, 12):
            tr["pos"][:, slot] = (60.0 * (slot - 6), 0.0)   # B at 60,120,180,240,300 from the ball at the origin
        assert census.n_within(tr, 100.0)[1] == pytest.approx(1.0)


class TestCluster:
    def test_line_of_five_spaced_80_is_one_component(self):
        tr = team_a_geometry([(0, 0), (80, 0), (160, 0), (240, 0), (320, 0)], ball=(500.0, 500.0))
        sizes = census.max_component_size(tr, 90.0)[0]
        assert sizes.tolist() == [5, 5, 5]
        assert census.cluster_ge4(tr)[0] == 1.0

    def test_b6_formation_is_not_a_cluster(self):
        tr = team_a_geometry(TestNWithin.OFFSETS)
        assert census.max_component_size(tr, 90.0)[0].tolist() == [3, 3, 3]
        assert census.cluster_ge4(tr)[0] == 0.0

    def test_link_radius_is_strict(self):
        tr = team_a_geometry([(0, 0), (90, 0), (180, 0), (270, 0), (360, 0)], ball=(500.0, 500.0))
        assert census.max_component_size(tr, 90.0)[0].tolist() == [1, 1, 1]

    def test_chain_through_an_intermediate_player(self):
        # Ends are 4 links apart (needs transitive closure), player 5 is isolated
        tr = team_a_geometry([(0, 0), (60, 0), (120, 0), (180, 0), (700, 700)], ball=(300.0, 300.0))
        assert census.max_component_size(tr, 90.0)[0].tolist() == [4, 4, 4]
        assert census.cluster_ge4(tr)[0] == 1.0

    def test_share_over_decisions(self):
        tr = team_a_geometry([(0, 0), (80, 0), (160, 0), (240, 0), (320, 0)], T=4, ball=(500.0, 500.0))
        tr["pos"][2:, 5] = (900.0, 800.0)                  # last two decisions: the fifth player leaves
        tr["pos"][3:, 4] = (900.0, 700.0)                  # last one: the fourth leaves too
        assert census.cluster_ge4(tr)[0] == pytest.approx(3 / 4)


class TestRoleInversion:
    def _trace(self, d_x, st_x, side_a=0, d_slot=1):
        tr = make_trace(2)
        tr["side"][:, 0] = side_a
        tr["side"][:, 1] = 1 - side_a
        behind = st_x - 100.0 if side_a == 0 else st_x + 100.0
        tr["pos"][:, 1, 0] = behind
        tr["pos"][:, 2, 0] = behind
        tr["pos"][:, d_slot, 0] = d_x
        tr["pos"][:, 5, 0] = st_x
        return tr

    def test_defender_ahead_by_50_is_inverted(self):
        assert census.role_inversion(self._trace(500.0, 450.0))[0] == 1.0

    def test_defender_ahead_by_20_is_not(self):
        assert census.role_inversion(self._trace(470.0, 450.0))[0] == 0.0

    def test_exactly_30_is_not(self):
        assert census.role_inversion(self._trace(480.0, 450.0))[0] == 0.0

    def test_just_over_30_is_inverted(self):
        assert census.role_inversion(self._trace(480.5, 450.0))[0] == 1.0

    def test_either_defender_counts(self):
        assert census.role_inversion(self._trace(500.0, 450.0, d_slot=2))[0] == 1.0

    def test_attack_axis_follows_the_side(self):
        # Team A defends the right goal: it attacks toward x = 0, so ahead means smaller x.
        assert census.role_inversion(self._trace(400.0, 450.0, side_a=1))[0] == 1.0
        assert census.role_inversion(self._trace(500.0, 450.0, side_a=1))[0] == 0.0

    def test_midfielders_do_not_count(self):
        tr = self._trace(450.0, 450.0)
        tr["pos"][:, 3, 0] = 900.0
        tr["pos"][:, 4, 0] = 900.0
        assert census.role_inversion(tr)[0] == 0.0


# ----------------------------------------------------------------------------
# Goalkeeper opportunities
# ----------------------------------------------------------------------------
class TestGkOpportunities:
    def _trace(self):
        tr = make_trace(48)
        # Team A keeper (player 0) defends the left goal (centre (0, 420), box 195).
        def seg(a, b, ball, gk):
            tr["ball"][a:b] = ball
            tr["pos"][a:b, 0] = gk
        seg(2, 10, (100.0, 420.0), (90.0, 420.0))      # loose, slow, keeper 10 u away; picked up at 10
        seg(10, 12, (80.0, 420.0), (80.0, 420.0))
        tr["holder"][10:12] = 0
        seg(15, 23, (150.0, 420.0), (50.0, 420.0))     # keeper exactly 100 u away, never closes, no pickup
        seg(23, 26, (400.0, 420.0), (50.0, 420.0))     # ball leaves the box
        seg(28, 34, (170.0, 420.0), (49.0, 420.0))     # keeper 121 u away: not an opportunity
        seg(36, 40, (100.0, 420.0), (60.0, 420.0))     # in the box but moving fast
        tr["bvel"][36:40] = (200.0, 0.0)
        return tr

    def test_episodes_pickup_and_proximity(self):
        eps = census.gk_opportunities(self._trace())
        mine = [e for e in eps if e["team"] == 0]
        assert [(e["start"], e["end"]) for e in mine] == [(2, 10), (15, 23)]
        assert [e["pickup"] for e in mine] == [True, False]
        assert [e["within30"] for e in mine] == [True, False]
        assert [e for e in eps if e["team"] == 1] == []

    def test_keeper_exactly_120_away_counts(self):
        tr = make_trace(10)
        tr["ball"][2:6] = (150.0, 420.0)
        tr["pos"][2:6, 0] = (30.0, 420.0)
        assert len(census.gk_opportunities(tr)) == 1

    def test_ball_outside_box_radius_is_ignored(self):
        tr = make_trace(10)
        tr["ball"][2:6] = (196.0, 420.0)
        tr["pos"][2:6, 0] = (150.0, 420.0)
        assert census.gk_opportunities(tr) == []

    def test_dynamic_box_radius_is_used(self):
        tr = make_trace(10)
        tr["ball"][2:6] = (196.0, 420.0)
        tr["pos"][2:6, 0] = (150.0, 420.0)
        tr["box"][:] = 210.0
        assert len(census.gk_opportunities(tr)) == 1

    def test_team_b_goalkeeper_uses_the_right_goal(self):
        tr = make_trace(12)
        tr["ball"][3:8] = (1300.0, 420.0)
        tr["pos"][3:8, 6] = (1330.0, 420.0)             # within 30 u of the ball
        eps = census.gk_opportunities(tr)
        assert [(e["team"], e["within30"]) for e in eps] == [(1, False)]   # exactly 30 u is not "within 30"

    def test_second_half_swap_follows_side(self):
        tr = make_trace(12)
        tr["side"][:, 0] = 1
        tr["side"][:, 1] = 0                            # A now defends the right goal
        tr["ball"][3:8] = (1300.0, 420.0)
        tr["pos"][3:8, 0] = (1330.0, 420.0)
        eps = census.gk_opportunities(tr)
        assert [e["team"] for e in eps] == [0]

    def test_episode_open_at_end_of_trace(self):
        tr = make_trace(10)
        tr["ball"][6:] = (100.0, 420.0)
        tr["pos"][6:, 0] = (60.0, 420.0)
        eps = census.gk_opportunities(tr)
        assert len(eps) == 1 and eps[0]["start"] == 6 and eps[0]["pickup"] is False

    def test_pickup_two_steps_after_leaving_is_credited(self):
        tr = make_trace(20)
        tr["ball"][3:6] = (100.0, 420.0)
        tr["pos"][3:6, 0] = (60.0, 420.0)
        tr["ball"][6:9] = (250.0, 420.0)                # leaves the box, still moving slowly
        tr["holder"][7] = 0                             # keeper holds one step after the episode closed
        assert census.gk_opportunities(tr)[0]["pickup"] is True
        tr["holder"][7] = -1
        tr["holder"][9] = 0                             # three steps after: too late
        assert census.gk_opportunities(tr)[0]["pickup"] is False


# ----------------------------------------------------------------------------
# Goal types
# ----------------------------------------------------------------------------
class TestGoalTypes:
    def _trace(self):
        tr = make_trace(100)
        # Goal 1: A's striker kicks at step 20 from 400 u out, scores at step 30, nobody touches the ball.
        tr["pos"][20, 5] = (1000.0, 420.0)
        tr["holder"][20] = 5
        set_kick(tr, 20, 5)
        tr["s_score"][30:, 0] = 1
        # Goal 2: M2 kicks at 50 from 500 u (3-4-5 triangle), M1 touches it at 55, scores at 58 -> not direct.
        tr["pos"][50, 4] = (1100.0, 820.0)
        tr["holder"][50] = 4
        set_kick(tr, 50, 4)
        tr["holder"][55] = 3
        tr["s_team_passes"][45:, 0] = 2
        tr["s_score"][58:, 0] = 2
        # Own goal credited to A at step 70.
        tr["s_score"][70:, 0] = 3
        tr["s_own_goals"][70:] = 1
        return tr

    def test_goal_events(self):
        goals = census.goal_events(self._trace())
        assert [(g["step"], g["team"], g["own_goal"]) for g in goals] == [(30, 0, False), (58, 0, False), (70, 0, True)]
        assert [g["direct"] for g in goals[:2]] == [True, False]
        assert goals[0]["kick_distance"] == pytest.approx(400.0)
        assert goals[1]["kick_distance"] == pytest.approx(500.0)
        assert goals[0]["passes_prior_10s"] == 0 and goals[1]["passes_prior_10s"] == 2
        assert goals[0]["seconds_since_kick"] == pytest.approx(10 * STEP_S)
        assert goals[0]["kicker_slot"] == 5 and goals[1]["kicker_slot"] == 4

    def test_goal_types_summary_excludes_own_goals_from_kick_stats(self):
        s = census.goal_type_counts(census.goal_events(self._trace()))
        assert s["n"] == 3 and s["n_open_play"] == 2
        assert s["direct"] == 1
        assert s["no_pass_10s"] == 1
        assert s["kick_distance_sum"] == pytest.approx(900.0)

    def test_attacking_goal_follows_the_side(self):
        tr = make_trace(60)
        tr["side"][:, 0] = 1                              # A attacks the goal at x = 0
        tr["pos"][10, 5] = (300.0, 420.0)
        tr["holder"][10] = 5
        set_kick(tr, 10, 5)
        tr["s_score"][15:, 0] = 1
        assert census.goal_events(tr)[0]["kick_distance"] == pytest.approx(300.0)

    def test_goal_in_first_step(self):
        tr = make_trace(10)
        tr["s_score"][0:, 1] = 1
        goals = census.goal_events(tr)
        assert [(g["step"], g["team"]) for g in goals] == [(0, 1)]

    def test_no_goals(self):
        assert census.goal_events(make_trace(10)) == []


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------
class TestAggregation:
    def test_mean_se_n(self):
        m, se, n = census.mean_se([1.0, 2.0, 3.0, 4.0])
        assert (m, n) == (2.5, 4)
        assert se == pytest.approx(math.sqrt(5.0 / 3.0) / 2.0)

    def test_mean_se_degenerate(self):
        assert census.mean_se([]) == (None, None, 0)
        m, se, n = census.mean_se([3.0])
        assert (m, se, n) == (3.0, None, 1)

    def test_ratio_se_matches_hand_value(self):
        # Matches contribute (num, den) = (1, 2), (3, 4): pooled ratio 4/6. Residuals y - r x: 1 - 4/3 = -1/3 and 3 - 8/3 = 1/3.
        r, se, n = census.ratio_se([1, 3], [2, 4])
        assert r == pytest.approx(4 / 6)
        assert n == 2
        assert se == pytest.approx(math.sqrt((1 / 9 + 1 / 9) * 2 / 1) / 6)

    def test_ratio_se_zero_denominator(self):
        assert census.ratio_se([0, 0], [0, 0]) == (None, None, 0)


# ----------------------------------------------------------------------------
# Harness: determinism, isolation, modes
# ----------------------------------------------------------------------------
def write_checkpoint(directory: Path, seed: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    for prefix in ("blue", "red"):
        for kind in ("field", "keeper"):
            torch.save(ActorCritic().state_dict(), directory / f"{prefix}_{kind}_policy.pt")
    return directory


def dir_digest(directory: Path) -> dict:
    return {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())}


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory) -> Path:
    return write_checkpoint(tmp_path_factory.mktemp("ckpt"), seed=0)


def quick_spec(ckpt, **kw):
    base = dict(checkpoint_dir=str(ckpt), opponent="self", mode="sampled", n=2, seed=5, max_steps=45)
    base.update(kw)
    return harness.EvalSpec(**base)


class TestHarness:
    def test_same_seed_gives_identical_traces(self, ckpt):
        a = harness.run_matches(quick_spec(ckpt))
        b = harness.run_matches(quick_spec(ckpt))
        assert len(a) == len(b) == 2
        for ta, tb in zip(a, b):
            assert set(ta) == set(tb)
            for key in ta:
                assert np.array_equal(ta[key], tb[key]), key

    def test_different_seed_differs(self, ckpt):
        a = harness.run_matches(quick_spec(ckpt, seed=5))
        b = harness.run_matches(quick_spec(ckpt, seed=6))
        assert not np.array_equal(a[0]["act"], b[0]["act"])
        assert not np.array_equal(a[0]["pos"], b[0]["pos"])

    def test_matches_within_a_run_differ(self, ckpt):
        a = harness.run_matches(quick_spec(ckpt))
        assert not np.array_equal(a[0]["act"], a[1]["act"])

    def test_match_does_not_depend_on_n_or_workers(self, ckpt):
        solo = harness.run_matches(quick_spec(ckpt, n=1))
        pair = harness.run_matches(quick_spec(ckpt, n=2, workers=2))
        assert np.array_equal(solo[0]["pos"], pair[0]["pos"])
        assert np.array_equal(solo[0]["act"], pair[0]["act"])

    def test_match_in_a_sequence_equals_the_same_match_played_alone(self, ckpt):
        spec = quick_spec(ckpt, n=2)
        in_sequence = harness.run_matches(spec)[1]
        fresh = harness.play_match(spec, 1)
        for key in fresh:
            assert np.array_equal(fresh[key], in_sequence[key]), key

    def test_result_does_not_depend_on_ambient_torch_threads(self, ckpt):
        import torch

        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            one = harness.run_matches(quick_spec(ckpt, n=1, max_steps=45))[0]
            torch.set_num_threads(2)
            two = harness.run_matches(quick_spec(ckpt, n=1, max_steps=45))[0]
            assert torch.get_num_threads() == 2          # the session restores the caller's setting
        finally:
            torch.set_num_threads(previous)
        for key in one:
            assert np.array_equal(one[key], two[key]), key

    def test_trace_schema(self, ckpt):
        t = harness.run_matches(quick_spec(ckpt, n=1))[0]
        T = 45
        assert int(t["n_steps"]) == T
        for key, shape in {
            "pos": (T, 12, 2), "vel": (T, 12, 2), "ang": (T, 12), "ball": (T, 2), "bvel": (T, 2),
            "holder": (T,), "act": (T, 12, 4), "mean": (T, 12, 4), "val": (T, 12), "rew": (T, 12),
            "box": (T, 2), "side": (T, 2), "lastkick": (T, 12), "s_score": (T, 2), "s_team_passes": (T, 2),
            "s_own_goals": (T,),
        }.items():
            assert t[key].shape == shape, key
        assert t["final_score"].shape == (2,)
        assert np.all(np.isfinite(t["act"])) and np.all(np.abs(t["act"]) <= 1.0)

    def test_modes_shape_the_actions(self, ckpt):
        greedy = harness.run_matches(quick_spec(ckpt, mode="greedy", n=1))[0]
        assert np.array_equal(greedy["act"], np.clip(greedy["mean"], -1, 1))
        mixed = harness.run_matches(quick_spec(ckpt, mode="mixed", n=1))[0]
        assert np.array_equal(mixed["act"][..., :2], np.clip(mixed["mean"][..., :2], -1, 1))
        assert not np.array_equal(mixed["act"][..., 2:], np.clip(mixed["mean"][..., 2:], -1, 1))
        sampled = harness.run_matches(quick_spec(ckpt, n=1))[0]
        assert not np.array_equal(sampled["act"][..., :2], np.clip(sampled["mean"][..., :2], -1, 1))

    def test_sigma_scale_zero_equals_greedy_actions(self, ckpt):
        zero = harness.run_matches(quick_spec(ckpt, sigma_scale=0.0, n=1))[0]
        assert np.array_equal(zero["act"], np.clip(zero["mean"], -1, 1))

    def test_swap_exchanges_the_policies(self, tmp_path):
        d = write_checkpoint(tmp_path / "c", seed=1)
        straight = harness.run_matches(quick_spec(d, mode="greedy", n=1))[0]
        swapped = harness.run_matches(quick_spec(d, mode="greedy", n=1, swap=True))[0]
        # Team A's action mean in the swapped run is produced by the red_* weights.
        assert not np.array_equal(straight["mean"][:, :6], swapped["mean"][:, :6])

    def test_frozen_opponent_uses_its_own_red_policies(self, tmp_path):
        mine = write_checkpoint(tmp_path / "mine", seed=2)
        other = write_checkpoint(tmp_path / "other", seed=3)
        self_play = harness.run_matches(quick_spec(mine, mode="greedy", n=1))[0]
        frozen = harness.run_matches(quick_spec(mine, mode="greedy", n=1, opponent=f"frozen:{other}"))[0]
        assert np.array_equal(self_play["mean"][0, :6], frozen["mean"][0, :6])      # same team A, same first observation
        assert not np.array_equal(self_play["mean"][0, 6:], frozen["mean"][0, 6:])  # different B weights

    def test_scripted_opponent_actions_are_the_bot(self, ckpt):
        t = harness.run_matches(quick_spec(ckpt, opponent="scripted", n=1))[0]
        assert np.all(t["val"][:, 6:] == 0.0)
        assert set(np.unique(t["act"][:, 6:, 2])) <= {-1.0, 1.0}

    def test_unknown_opponent_and_mode_are_rejected(self, ckpt):
        with pytest.raises(ValueError):
            harness.run_matches(quick_spec(ckpt, opponent="mystery"))
        with pytest.raises(ValueError):
            harness.run_matches(quick_spec(ckpt, mode="loud"))

    def test_checkpoint_required_for_networks(self):
        with pytest.raises(ValueError):
            harness.run_matches(harness.EvalSpec(opponent="self", n=1, max_steps=5))

    def test_breakdown_and_store_obs_add_columns(self, ckpt):
        t = harness.run_matches(quick_spec(ckpt, n=1, breakdown=True, store_obs=True))[0]
        assert t["bd"].shape == (45, 12, 5)
        assert t["role"].shape == (45, 12)
        assert t["phi"].shape == (45, 12)
        assert t["obs_s"].shape[1:] == (12, 82) and t["cen_s"].shape[1:] == (12, 54)
        plain = harness.run_matches(quick_spec(ckpt, n=1))[0]
        assert "bd" not in plain and "obs_s" not in plain
        # The breakdown run plays the same match.
        assert np.array_equal(t["pos"], plain["pos"])

    def test_no_writes_to_checkpoint_or_replays(self, tmp_path, monkeypatch):
        d = write_checkpoint(tmp_path / "checkpoints", seed=4)
        replays = tmp_path / "replays"
        replays.mkdir()
        before = dir_digest(d)
        monkeypatch.chdir(tmp_path)
        trace = harness.run_matches(quick_spec(d, n=1, breakdown=True, max_steps=None))[0]
        assert int(trace["n_steps"]) == 2700            # a finished match is when an environment would export a replay
        assert dir_digest(d) == before
        assert list(replays.iterdir()) == []
        assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoints", "replays"]

    def test_output_directory_guard(self, tmp_path):
        d = write_checkpoint(tmp_path / "checkpoints", seed=4)
        with pytest.raises(ValueError):
            harness.check_output_dir(d, [d])
        with pytest.raises(ValueError):
            harness.check_output_dir(d / "sub", [d])
        with pytest.raises(ValueError):
            harness.check_output_dir(REPO_ROOT / "replays" / "x", [])
        with pytest.raises(ValueError):
            harness.check_output_dir(REPO_ROOT / "checkpoints", [])
        harness.check_output_dir(tmp_path / "out", [d])

    def test_save_and_load_round_trip(self, ckpt, tmp_path):
        traces = harness.run_matches(quick_spec(ckpt, n=2))
        out = tmp_path / "out"
        harness.save_traces(traces, out, [Path(ckpt)])
        loaded = census.load_traces(out)
        assert len(loaded) == 2
        for a, b in zip(traces, loaded):
            for key in a:
                assert np.array_equal(a[key], b[key]), key


class TestTeamPolicy:
    @pytest.mark.parametrize("old_obs_dim", [76, 81])
    def test_a_76_input_checkpoint_loads_zero_padded(self, tmp_path, old_obs_dim):
        d = tmp_path / "legacy"
        d.mkdir()
        torch.manual_seed(3)
        legacy = {}
        for prefix in ("blue", "red"):
            for kind in ("field", "keeper"):
                legacy[prefix, kind] = ActorCritic(obs_dim=old_obs_dim)
                torch.save(legacy[prefix, kind].state_dict(), d / f"{prefix}_{kind}_policy.pt")
        team = harness.TeamPolicy(str(d), "blue")
        for (_, actor, critic, _), kind in zip(team.heads, ("keeper", "field")):
            old = legacy["blue", kind]
            actor_w, critic_w = actor[0][0], critic[0][0]
            assert actor_w.shape[1] == 82 and critic_w.shape[1] == 136
            assert torch.equal(actor_w[:, :old_obs_dim], old.actor_mean[0].weight)
            assert not actor_w[:, old_obs_dim:].any()
            assert torch.equal(critic_w[:, :old_obs_dim], old.critic[0].weight[:, :old_obs_dim])
            assert not critic_w[:, old_obs_dim:82].any()
            assert torch.equal(critic_w[:, 82:], old.critic[0].weight[:, old_obs_dim:])
        obs = np.zeros((6, 82), dtype=np.float32)
        cen = np.zeros((6, 54), dtype=np.float32)
        actions, values, _ = team.act(obs, cen, "greedy", 1.0, torch.Generator().manual_seed(0))
        assert actions.shape == (6, 4) and values.shape == (6,)

    def test_forward_is_bit_identical_to_the_networks(self, tmp_path):
        d = write_checkpoint(tmp_path / "c", seed=7)
        rng = np.random.default_rng(0)
        obs = rng.normal(size=(6, 82)).astype(np.float32)
        cen = rng.normal(size=(6, 54)).astype(np.float32)
        team = harness.TeamPolicy(str(d), "blue")
        actions, values, means = team.act(obs, cen, "greedy", 1.0, torch.Generator().manual_seed(0))
        for file, sl in (("keeper", slice(0, 1)), ("field", slice(1, 6))):
            net = ActorCritic()
            net.load_state_dict(torch.load(d / f"blue_{file}_policy.pt", weights_only=True))
            with torch.no_grad():
                o, c = torch.as_tensor(obs[sl]), torch.as_tensor(cen[sl])
                assert np.array_equal(means[sl], net.actor_mean(o).numpy())
                assert np.array_equal(values[sl], net.get_value(o, c).numpy())
        assert np.array_equal(actions, means)

    def test_noise_scales_with_the_checkpoint_sigma(self, tmp_path):
        d = write_checkpoint(tmp_path / "c", seed=7)
        obs = np.zeros((6, 82), dtype=np.float32)
        cen = np.zeros((6, 54), dtype=np.float32)
        team = harness.TeamPolicy(str(d), "blue")
        a, _, m = team.act(obs, cen, "sampled", 1.0, torch.Generator().manual_seed(4))
        b, _, _ = team.act(obs, cen, "sampled", 2.0, torch.Generator().manual_seed(4))
        # Same noise draw: the deviation from the mean doubles (logstd = -0.5 -> sigma = exp(-0.5)).
        np.testing.assert_allclose((b - m), 2.0 * (a - m), rtol=1e-4, atol=1e-6)
        assert np.abs(a - m).max() > 0.0


class TestCensusEndToEnd:
    def test_match_metrics_and_report_on_harness_traces(self, ckpt):
        traces = harness.run_matches(quick_spec(ckpt, n=2, max_steps=120))
        per_match = [census.match_metrics(t) for t in traces]
        assert all(0.0 <= m["hold_share_A"] <= 1.0 for m in per_match)
        text = census.report(per_match, title="synthetic")
        assert "n_within(100)" in text and "kick outcome" in text and "n=2" in text


# ----------------------------------------------------------------------------
# E1: Kick propensity
# ----------------------------------------------------------------------------
class TestKickPropensity:
    def test_four_decision_kick_plus_two_decision_tackle_is_one_sixth(self):
        tr = make_trace(20)
        tr["holder"][2:6] = 3
        set_kick(tr, 5, 3)
        tr["holder"][10:12] = 3
        kp = census.kick_propensity(tr)
        assert kp["A"] == pytest.approx(1.0 / 6.0)
        assert kp["pooled"] == pytest.approx(1.0 / 6.0)
        assert kp["M"] == pytest.approx(1.0 / 6.0)
        assert kp["B"] is None

    def test_one_decision_spell_ending_in_kick_is_one(self):
        tr = make_trace(10)
        tr["holder"][4] = 5
        set_kick(tr, 4, 5)
        kp = census.kick_propensity(tr)
        assert kp["A"] == pytest.approx(1.0)
        assert kp["ST"] == pytest.approx(1.0)
        assert kp["pooled"] == pytest.approx(1.0)

    def test_no_holding_returns_none_not_zero_or_nan(self):
        tr = make_trace(10)
        kp = census.kick_propensity(tr)
        for key in ("A", "B", "pooled", "GK", "D", "M", "ST"):
            assert kp[key] is None

    def test_role_groups_and_teams_aggregate_independently(self):
        tr = make_trace(30)
        tr["holder"][1:3] = 0
        set_kick(tr, 2, 0)         # A GK: 1/2
        tr["holder"][5:9] = 7      # B D1: 0/4
        tr["holder"][12:14] = 11   # B ST: 1/2
        set_kick(tr, 13, 11)
        kp = census.kick_propensity(tr)
        assert kp["A"] == pytest.approx(0.5)
        assert kp["B"] == pytest.approx(1.0 / 6.0)
        assert kp["pooled"] == pytest.approx(2.0 / 8.0)
        assert kp["GK"] == pytest.approx(0.5)
        assert kp["D"] == pytest.approx(0.0)
        assert kp["ST"] == pytest.approx(0.5)
        assert kp["M"] is None


# ----------------------------------------------------------------------------
# E1: Per-keeper geometry
# ----------------------------------------------------------------------------
class TestGkGeometry:
    def test_hand_solved_distance_inbox_and_goalline(self):
        tr = make_trace(4)
        tr["pos"][0:3, 0] = (10.0, 420.0)
        tr["pos"][3, 0] = (200.0, 420.0)
        geo = census.gk_geometry(tr)
        assert geo["gk_dist_goal_A"] == pytest.approx(57.5)
        assert geo["gk_in_box_A"] == pytest.approx(0.75)
        assert geo["gk_goal_line_A"] == pytest.approx(0.75)

    def test_exact_boundary_values_are_inclusive(self):
        tr = make_trace(2)
        tr["pos"][0, 0] = (19.5, 420.0)
        tr["pos"][1, 0] = (195.0, 420.0)
        tr["pos"][0, 6] = (1400.0 - 19.5, 420.0)
        tr["pos"][1, 6] = (1400.0 - 195.0, 420.0)
        geo = census.gk_geometry(tr)
        assert geo["gk_goal_line_A"] == pytest.approx(0.5)
        assert geo["gk_in_box_A"] == pytest.approx(1.0)
        assert geo["gk_goal_line_B"] == pytest.approx(0.5)
        assert geo["gk_in_box_B"] == pytest.approx(1.0)
        assert geo["gk_in_box_pooled"] == pytest.approx(1.0)

    def test_ignores_dynamic_box_and_follows_side_swap(self):
        tr = make_trace(2)
        tr["box"][:] = 100.0
        tr["side"][:, 0] = 1
        tr["side"][:, 1] = 0
        tr["pos"][:, 0] = (1400.0 - 150.0, 420.0)
        tr["pos"][:, 6] = (195.1, 420.0)
        geo = census.gk_geometry(tr)
        assert geo["gk_in_box_A"] == pytest.approx(1.0)
        assert geo["gk_in_box_B"] == pytest.approx(0.0)
        assert geo["gk_dist_goal_A"] == pytest.approx(150.0)


# ----------------------------------------------------------------------------
# E1: Per-team hold/spell/cluster/n_within/inversion & summary
# ----------------------------------------------------------------------------
class TestPerTeamAndSummary:
    def test_b6_formation_team_a_and_spread_team_b(self):
        tr = team_a_geometry(TestNWithin.OFFSETS, T=3, ball=(0.0, 0.0))
        for idx, slot in enumerate(range(7, 12)):
            tr["pos"][:, slot] = (200.0 + 150.0 * idx, 0.0)
        nw = census.n_within(tr, 100.0)
        assert nw[0] == pytest.approx(2.0)
        assert nw[1] == pytest.approx(0.0)
        m = census.match_metrics(tr)
        assert m["n_within_100_A"] == pytest.approx(2.0)
        assert m["n_within_100_B"] == pytest.approx(0.0)
        assert m["n_within_100_pooled"] == pytest.approx(1.0)

    def test_per_team_hold_spell_cluster_inversion_and_scoreless_in_summary(self):
        tr = make_trace(10)
        tr["holder"][0:3] = 2
        tr["holder"][4:9] = 8
        m = census.match_metrics(tr)
        assert m["hold_share_A"] == pytest.approx(0.3)
        assert m["hold_share_B"] == pytest.approx(0.5)
        assert m["hold_share_pooled"] == pytest.approx(0.8)
        assert m["spells_n_A"] == 1
        assert m["spells_sum_A"] == pytest.approx(3 * STEP_S)
        assert m["spells_n_B"] == 1
        assert m["spells_sum_B"] == pytest.approx(5 * STEP_S)
        assert m["scoreless"] == 1.0
        s = census.summarize([m])["metrics"]
        assert s["hold_share_pooled"]["mean"] == pytest.approx(0.8)
        assert s["holder_spell_s_A"]["mean"] == pytest.approx(3 * STEP_S)
        assert s["holder_spell_s_B"]["mean"] == pytest.approx(5 * STEP_S)
        assert s["holder_spell_s_pooled"]["mean"] == pytest.approx(4 * STEP_S)
        assert s["scoreless"]["mean"] == pytest.approx(1.0)


# ----------------------------------------------------------------------------
# E1: Time-binned metrics (last 225 decisions vs rest)
# ----------------------------------------------------------------------------
class TestTimeBinnedMetrics:
    def test_defender_mean_canonical_x_300_decisions(self):
        tr = make_trace(300)
        tr["s_shots"] = np.zeros((300, 2), dtype=np.int32)
        tr["s_field_5s_violations"] = np.zeros(300, dtype=np.int32)
        tr["s_keeper_5s_violations"] = np.zeros(300, dtype=np.int32)
        tr["pos"][:75, 1:3, 0] = 300.0
        tr["pos"][75:, 1:3, 0] = 500.0
        tb = census.time_binned_metrics(tr)
        assert tb["early_def_x_A"] == pytest.approx(300.0)
        assert tb["late_def_x_A"] == pytest.approx(500.0)

    def test_all_time_binned_submetrics_hand_solved(self):
        tr = make_trace(300)
        tr["s_shots"] = np.zeros((300, 2), dtype=np.int32)
        tr["s_field_5s_violations"] = np.zeros(300, dtype=np.int32)
        tr["s_keeper_5s_violations"] = np.zeros(300, dtype=np.int32)
        # Early [0:75] (5 s = 1/12 min), Late [75:300] (15 s = 1/4 min).
        # GK A holds 3 steps in early (kicks at step 2 -> within 0.5 s), 10 steps in late (kicks at step 109 -> >0.5 s).
        tr["holder"][0:3] = 0
        set_kick(tr, 2, 0)
        tr["holder"][100:110] = 0
        set_kick(tr, 109, 0)
        # Inversion in early: 0; in late: all 225 steps inverted for team A.
        tr["pos"][:, 5, 0] = 400.0
        tr["pos"][:75, 1:3, 0] = 300.0
        tr["pos"][75:, 1:3, 0] = 500.0
        # Shots: 1 by A in early (12/min), 3 by A in late (12/min); 0 by B.
        tr["s_shots"][30:, 0] = 1
        tr["s_shots"][200:, 0] = 4
        # 5s violations: 1 field violation in late (4/min).
        tr["s_field_5s_violations"][250:] = 1
        # Outfield kick-to-self: 1 kick in early -> teammate, 2 kicks in late -> 1 self, 1 opponent (0.5).
        tr["holder"][10] = 3
        set_kick(tr, 10, 3)
        tr["holder"][20] = 4
        tr["holder"][150] = 3
        set_kick(tr, 150, 3)
        tr["holder"][160] = 3
        tr["holder"][180] = 4
        set_kick(tr, 180, 4)
        tr["holder"][190] = 7

        tb = census.time_binned_metrics(tr)
        assert tb["early_gk_hold_s_A"] == pytest.approx(3 * STEP_S)
        assert tb["late_gk_hold_s_A"] == pytest.approx(10 * STEP_S)
        assert tb["early_gk_quick_kick_share_A"] == pytest.approx(1.0)
        assert tb["late_gk_quick_kick_share_A"] == pytest.approx(0.0)
        assert tb["early_role_inversion_A"] == pytest.approx(0.0)
        assert tb["late_role_inversion_A"] == pytest.approx(1.0)
        assert tb["early_shots_per_min_A"] == pytest.approx(12.0)
        assert tb["late_shots_per_min_A"] == pytest.approx(12.0)
        assert tb["early_kick_self_share_A"] == pytest.approx(0.0)
        assert tb["late_kick_self_share_A"] == pytest.approx(0.5)
        assert tb["early_violations_5s_per_min_both"] == pytest.approx(0.0)
        assert tb["late_violations_5s_per_min_both"] == pytest.approx(4.0)

    def test_short_trace_under_225_steps_has_none_early_metrics(self):
        tr = make_trace(100)
        tr["s_shots"] = np.zeros((100, 2), dtype=np.int32)
        tr["s_field_5s_violations"] = np.zeros(100, dtype=np.int32)
        tr["s_keeper_5s_violations"] = np.zeros(100, dtype=np.int32)
        tb = census.time_binned_metrics(tr)
        assert tb["early_def_x_A"] is None
        assert tb["early_shots_per_min_both"] is None
        assert tb["late_def_x_A"] == pytest.approx(700.0)


# ----------------------------------------------------------------------------
# E1: Gates
# ----------------------------------------------------------------------------
def _fake_census_point(**overrides):
    metrics = {
        "hold_share_pooled": {"mean": 0.19, "se": 0.002, "n": 96},
        "hold_share_A": {"mean": 0.095, "se": 0.002, "n": 96},
        "hold_share_B": {"mean": 0.095, "se": 0.002, "n": 96},
        "holder_spell_s_pooled": {"mean": 0.27, "se": 0.004, "n": 96},
        "gk_in_box_A": {"mean": 0.99, "se": 0.002, "n": 96},
        "gk_in_box_B": {"mean": 0.99, "se": 0.002, "n": 96},
        "gk_dist_goal_A": {"mean": 75.0, "se": 1.0, "n": 96},
        "gk_dist_goal_B": {"mean": 75.0, "se": 1.0, "n": 96},
        "gk_goal_line_A": {"mean": 0.20, "se": 0.01, "n": 96},
        "gk_goal_line_B": {"mean": 0.15, "se": 0.01, "n": 96},
        "n_within_100_pooled": {"mean": 0.50, "se": 0.01, "n": 96},
        "cluster_ge4_pooled": {"mean": 0.06, "se": 0.003, "n": 96},
        "cluster_ge4_A": {"mean": 0.08, "se": 0.004, "n": 96},
        "cluster_ge4_B": {"mean": 0.04, "se": 0.003, "n": 96},
        "role_inversion_pooled": {"mean": 0.14, "se": 0.005, "n": 96},
        "role_inversion_A": {"mean": 0.14, "se": 0.005, "n": 96},
        "role_inversion_B": {"mean": 0.14, "se": 0.005, "n": 96},
        "goals_total": {"mean": 2.2, "se": 0.1, "n": 96},
        "scoreless": {"mean": 0.10, "se": 0.03, "n": 96},
        "kick_propensity_pooled": {"mean": 0.245, "se": 0.003, "n": 96},
    }
    for k, v in overrides.items():
        if isinstance(v, tuple):
            metrics[k] = {"mean": v[0], "se": v[1], "n": 96}
        else:
            metrics[k] = {"mean": v, "se": metrics[k]["se"], "n": 96}
    return {"summary": {"n_matches": 96, "metrics": metrics}}


class TestGates:
    def test_delta_se_and_significance_star(self):
        from tools.audit import gates

        res = gates.delta_stat(0.091, 0.003, 0.054, 0.002)
        assert res["delta"] == pytest.approx(0.037)
        assert res["se"] == pytest.approx(math.sqrt(0.003 ** 2 + 0.002 ** 2))
        assert res["se"] == pytest.approx(0.00360555, rel=1e-4)
        assert res["significant"] is True

        non_sig = gates.delta_stat(0.091, 0.01, 0.085, 0.01)
        assert non_sig["significant"] is False

    def test_two_consecutive_exceedances_rule_on_cluster_ge4(self):
        from tools.audit import gates

        pass_arm = {"s1": {
            "1050": {"self_sampled": _fake_census_point(cluster_ge4_A=0.12)},
            "1100": {"self_sampled": _fake_census_point(cluster_ge4_A=0.16)},
            "1150": {"self_sampled": _fake_census_point(cluster_ge4_A=0.14)},
            "1200": {"self_sampled": _fake_census_point(cluster_ge4_A=0.16)},
        }}
        fail_arm = {"s1": {
            "1100": {"self_sampled": _fake_census_point(cluster_ge4_A=0.12)},
            "1150": {"self_sampled": _fake_census_point(cluster_ge4_A=0.16)},
            "1200": {"self_sampled": _fake_census_point(cluster_ge4_A=0.16)},
        }}
        assert gates.evaluate_gates(pass_arm)["gates"]["G3"]["passed"] is True
        assert gates.evaluate_gates(fail_arm)["gates"]["G3"]["passed"] is False

    def test_g9_prime_b_and_seed_ranges(self):
        from tools.audit import gates

        b3 = {
            "s42": {"1200": {"self_sampled": _fake_census_point(kick_propensity_pooled=0.245)}},
            "s43": {"1200": {"self_sampled": _fake_census_point(kick_propensity_pooled=0.240)}},
        }
        b4_pass = {
            "s42": {"1200": {"self_sampled": _fake_census_point(kick_propensity_pooled=0.205)}},
            "s43": {"1200": {"self_sampled": _fake_census_point(kick_propensity_pooled=0.200)}},
        }
        b4_fail = {
            "s42": {"1200": {"self_sampled": _fake_census_point(kick_propensity_pooled=0.205)}},
            "s43": {"1200": {"self_sampled": _fake_census_point(kick_propensity_pooled=0.242)}},
        }
        res_pass = gates.evaluate_gates(b4_pass, comparator=b3, gens=["1200"])
        assert res_pass["gates"]["G9_prime_b"]["passed"] is True
        assert res_pass["gates"]["G9_prime_b"]["points"]["1200"]["arm_range"] == pytest.approx([0.200, 0.205])
        assert res_pass["gates"]["G9_prime_b"]["points"]["1200"]["comp_range"] == pytest.approx([0.240, 0.245])
        md = gates.format_markdown(res_pass)
        assert "0.200..0.205" in md or "0.200" in md

        res_fail = gates.evaluate_gates(b4_fail, comparator=b3, gens=["1200"])
        assert res_fail["gates"]["G9_prime_b"]["passed"] is False


# ----------------------------------------------------------------------------
# E1: evaluate_arm
# ----------------------------------------------------------------------------
class TestEvaluateArm:
    def test_determinism_two_runs_byte_identical_census_json(self, ckpt, tmp_path):
        from tools.audit import evaluate_arm

        snap = tmp_path / "gen1200"
        snap.mkdir()
        for p in ckpt.iterdir():
            (snap / p.name).write_bytes(p.read_bytes())

        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        evaluate_arm.evaluate_arm(
            [snap], out1, workers=1, frozen_dir=snap, n_override=2, max_steps=30,
        )
        evaluate_arm.evaluate_arm(
            [snap], out2, workers=1, frozen_dir=snap, n_override=2, max_steps=30,
        )
        for tag in ("self_sampled", "self_mixed", "vs_scripted_sampled", "vs_frozen"):
            p1 = out1 / "gen1200" / tag / "census.json"
            p2 = out2 / "gen1200" / tag / "census.json"
            assert p1.is_file() and p2.is_file()
            assert p1.read_bytes() == p2.read_bytes()

    def test_no_writes_to_checkpoint_or_replays(self, tmp_path, monkeypatch):
        from tools.audit import evaluate_arm

        snap = write_checkpoint(tmp_path / "checkpoints" / "gen1200", seed=9)
        replays = tmp_path / "replays"
        replays.mkdir()
        before = dir_digest(snap)
        monkeypatch.chdir(tmp_path)
        evaluate_arm.evaluate_arm(
            [snap], tmp_path / "eval_out", workers=1, n_override=2, max_steps=20, tags=["self_sampled"],
        )
        assert dir_digest(snap) == before
        assert list(replays.iterdir()) == []


# ----------------------------------------------------------------------------
# E2: Policy internals (tools/audit/internals.py)
# ----------------------------------------------------------------------------
def make_internals_trace(T: int) -> dict:
    tr = make_trace(T)
    tr["ang"] = np.zeros((T, 12), dtype=np.float32)
    tr["hold_t"] = np.zeros(T, dtype=np.float32)
    tr["act"] = np.full((T, 12, 4), -1.0, dtype=np.float32)
    tr["mean"] = np.full((T, 12, 4), -1.0, dtype=np.float32)
    return tr


class TestInternals:
    def test_saturation_and_grad_scale_hand_values(self):
        from tools.audit import internals

        tr = make_internals_trace(4)
        vals = np.array([0.995, -1.0, 0.5, 0.0], dtype=np.float32)
        for col in range(4):
            tr["mean"][:, 1:6, col] = vals[:, None]
            tr["mean"][:, 7:12, col] = vals[:, None]
        # Keepers sit at 0.0; they must not enter field-player averages.
        tr["mean"][:, [0, 6], :] = 0.0
        m = internals.match_internals(tr)
        for head in ("throttle", "steer", "kick", "special"):
            assert m[f"sat99_{head}"] == pytest.approx(0.5)
            assert m[f"grad_scale_{head}"] == pytest.approx(0.43999375)

    def test_kick_decomposition_hand_values(self):
        from tools.audit import internals

        tr = make_internals_trace(10)
        means = [0.9, 0.9, 0.9, -0.9, -0.9, -0.9, -0.9, -0.9]
        # Rows 0..7 are eligible outfield holds; row 8 fails hold_t grace; row 9 is GK hold.
        for t, km in enumerate(means):
            tr["holder"][t] = 3
            tr["hold_t"][t] = 0.10
            tr["mean"][t, 3, 2] = km
            tr["act"][t, 3, 2] = 0.8 if t in (0, 1, 7) else -0.5
        tr["holder"][8] = 3
        tr["hold_t"][8] = 0.04
        tr["mean"][8, 3, 2] = 0.9
        tr["act"][8, 3, 2] = 0.9
        tr["holder"][9] = 0
        tr["hold_t"][9] = 0.20
        tr["mean"][9, 0, 2] = 0.9
        tr["act"][9, 0, 2] = 0.9

        m = internals.match_internals(tr)
        assert m["p_kick_obs"] == pytest.approx(0.375)
        assert m["p_kick_intent"] == pytest.approx(0.375)
        assert m["kick_from_intent"] == pytest.approx(2.0 / 3.0)
        assert m["p_kick_given_intent"] == pytest.approx(2.0 / 3.0)
        assert m["p_kick_given_no_intent"] == pytest.approx(0.2)

    def test_kick_decomposition_no_eligible_decisions_returns_none(self):
        from tools.audit import internals

        tr = make_internals_trace(4)
        tr["holder"][0] = 3
        tr["hold_t"][0] = 0.05  # below 0.08 grace
        tr["holder"][1] = 0     # keeper
        tr["hold_t"][1] = 0.50
        m = internals.match_internals(tr)
        for key in (
            "p_kick_obs", "p_kick_intent", "kick_from_intent",
            "p_kick_given_intent", "p_kick_given_no_intent",
            "power_mean", "power_sd", "full_power_share",
            "aim_teammate_share_sampled", "aim_teammate_share_mean",
        ):
            assert m[key] is None

    def test_power_hand_values(self):
        from tools.audit import internals

        tr = make_internals_trace(3)
        actions = [1.0, 0.675, 0.415]
        for t, a in enumerate(actions):
            tr["holder"][t] = 4
            tr["hold_t"][t] = 0.10
            tr["act"][t, 4, 2] = a
        m = internals.match_internals(tr)
        expected = np.array([1.0, 0.5, 0.1])
        assert m["power_mean"] == pytest.approx(0.5333333333)
        assert m["power_sd"] == pytest.approx(float(expected.std()))
        assert m["full_power_share"] == pytest.approx(1.0 / 3.0)

    def test_aim_teammate_share_hand_cases(self):
        from tools.audit import internals

        def one_kick(teammate_xy, steer_act, steer_mean=0.0):
            tr = make_internals_trace(1)
            tr["holder"][0] = 1
            tr["hold_t"][0] = 0.10
            tr["act"][0, 1, 2] = 0.9
            tr["act"][0, 1, 1] = steer_act
            tr["mean"][0, 1, 1] = steer_mean
            tr["pos"][0, 1] = (0.0, 0.0)
            tr["ang"][0, 1] = 0.0
            tr["pos"][0, 2] = teammate_xy
            return internals.match_internals(tr)

        assert one_kick((200.0, 0.0), 0.0)["aim_teammate_share_sampled"] == pytest.approx(1.0)
        assert one_kick((200.0, 60.0), 0.0)["aim_teammate_share_sampled"] == pytest.approx(0.0)
        assert one_kick((200.0, 60.0), 0.5, steer_mean=0.0)["aim_teammate_share_sampled"] == pytest.approx(1.0)
        assert one_kick((200.0, 60.0), 0.5, steer_mean=0.0)["aim_teammate_share_mean"] == pytest.approx(0.0)
        assert one_kick((50.0, 0.0), 0.0)["aim_teammate_share_sampled"] == pytest.approx(0.0)

    def test_special_rate_and_intent_hand_values(self):
        from tools.audit import internals

        tr = make_internals_trace(2)
        # 2 steps * 10 field players = 20 field decisions; fire on 3, 2 of which have mean > 0.5.
        tr["act"][0, 1, 3] = 0.8
        tr["mean"][0, 1, 3] = 0.7
        tr["act"][0, 2, 3] = 0.6
        tr["mean"][0, 2, 3] = 0.6
        tr["act"][1, 7, 3] = 0.9
        tr["mean"][1, 7, 3] = -0.2
        tr["act"][:, 0, 3] = 1.0  # keeper fire ignored
        m = internals.match_internals(tr)
        assert m["special_rate"] == pytest.approx(3.0 / 20.0)
        assert m["special_from_intent"] == pytest.approx(2.0 / 3.0)

    def test_cli_writes_json_and_prints_text(self, tmp_path, capsys):
        from tools.audit import internals

        tr1 = make_internals_trace(2)
        tr1["holder"][:] = 1
        tr1["hold_t"][:] = 0.1
        tr1["act"][0, 1, 2] = 1.0
        tr2 = make_internals_trace(2)
        tr2["holder"][:] = 1
        tr2["hold_t"][:] = 0.1
        tr2["act"][:, 1, 2] = 1.0
        d = tmp_path / "traces"
        d.mkdir()
        np.savez_compressed(d / "m000.npz", **tr1)
        np.savez_compressed(d / "m001.npz", **tr2)
        out_json = tmp_path / "internals.json"
        rc = internals.main([str(d), "--json", str(out_json)])
        assert rc == 0
        txt = capsys.readouterr().out
        assert "sat99" in txt and "kick" in txt and "n=2" in txt
        data = json.loads(out_json.read_text(encoding="utf-8"))
        met = data[str(d)]["summary"]["metrics"]
        assert met["p_kick_obs"]["mean"] == pytest.approx(0.75)
        assert met["p_kick_obs"]["se"] == pytest.approx(0.25)
        assert met["p_kick_obs"]["n"] == 2


# ----------------------------------------------------------------------------
# E2: goal_diff in census.py
# ----------------------------------------------------------------------------
class TestGoalDiff:
    def test_goal_diff_mean_and_se_hand_values(self):
        t1 = make_trace(2)
        t1["final_score"] = np.array([1, 3], dtype=np.int32)
        t2 = make_trace(2)
        t2["final_score"] = np.array([2, 2], dtype=np.int32)
        per_match = [census.match_metrics(t1), census.match_metrics(t2)]
        assert [m["goal_diff"] for m in per_match] == [-2.0, 0.0]
        gd = census.summarize(per_match)["metrics"]["goal_diff"]
        assert gd["mean"] == pytest.approx(-1.0)
        assert gd["se"] == pytest.approx(1.0)
        assert gd["n"] == 2


# ----------------------------------------------------------------------------
# E2: fresh300 gates, EBE targets, and between-seed SD (tools/audit/gates.py)
# ----------------------------------------------------------------------------
class TestGatesE2:
    def test_fresh300_g1_checks_last_gen_only(self):
        from tools.audit import gates

        arm_pass = {"s42": {
            "200": {"self_sampled": _fake_census_point(hold_share_pooled=0.16)},
            "300": {"self_sampled": _fake_census_point(hold_share_pooled=0.18)},
        }}
        arm_fail = {"s42": {
            "200": {"self_sampled": _fake_census_point(hold_share_pooled=0.18)},
            "300": {"self_sampled": _fake_census_point(hold_share_pooled=0.16)},
        }}
        res_pass = gates.evaluate_gates(arm_pass, gens=["200", "300"], protocol="fresh300")
        res_fail = gates.evaluate_gates(arm_fail, gens=["200", "300"], protocol="fresh300")
        assert res_pass["gates"]["G1"]["passed"] is True
        assert res_fail["gates"]["G1"]["passed"] is False
        assert "G9_prime_b" not in res_pass["gates"]
        # Under resume150, 0.16 at gen 200 fails G1.
        assert gates.evaluate_gates(arm_pass, gens=["200", "300"], protocol="resume150")["gates"]["G1"]["passed"] is False

    def test_fresh300_g3_checks_last_two_gens(self):
        from tools.audit import gates

        arm_pass = {"s42": {
            "200": {"self_sampled": _fake_census_point(cluster_ge4_pooled=0.12)},
            "300": {"self_sampled": _fake_census_point(cluster_ge4_pooled=0.09)},
        }}
        arm_fail = {"s42": {
            "200": {"self_sampled": _fake_census_point(cluster_ge4_pooled=0.12)},
            "300": {"self_sampled": _fake_census_point(cluster_ge4_pooled=0.11)},
        }}
        arm_recovered = {"s42": {
            "100": {"self_sampled": _fake_census_point(cluster_ge4_pooled=0.12)},
            "200": {"self_sampled": _fake_census_point(cluster_ge4_pooled=0.12)},
            "300": {"self_sampled": _fake_census_point(cluster_ge4_pooled=0.09)},
        }}
        assert gates.evaluate_gates(arm_pass, gens=["200", "300"], protocol="fresh300")["gates"]["G3"]["passed"] is True
        assert gates.evaluate_gates(arm_fail, gens=["200", "300"], protocol="fresh300")["gates"]["G3"]["passed"] is False
        assert gates.evaluate_gates(arm_recovered, gens=["100", "200", "300"], protocol="fresh300")["gates"]["G3"]["passed"] is True

    def _two_seed_target_arm(self, tag: str, key: str, v42: float, v43: float):
        return {
            "s42": {"300": {"self_sampled": _fake_census_point(), tag: _fake_census_point(**{key: (v42, 0.4)})}},
            "s43": {"300": {"self_sampled": _fake_census_point(), tag: _fake_census_point(**{key: (v43, 0.4)})}},
        }

    def test_ebe_targets_hand_cases(self):
        from tools.audit import gates

        comp = self._two_seed_target_arm("vs_scripted_sampled", "goals_A", -10.5, -11.0)
        spec = "goals_A@vs_scripted_sampled:higher:2.0"

        beats = gates.evaluate_gates(
            self._two_seed_target_arm("vs_scripted_sampled", "goals_A", -8.0, -7.5),
            comparator=comp, gens=["300"], protocol="fresh300", targets=[spec],
        )["targets"][0]
        assert beats["verdict"] == "BEATS"
        assert beats["mean_diff"] == pytest.approx(3.0)

        split = gates.evaluate_gates(
            self._two_seed_target_arm("vs_scripted_sampled", "goals_A", -8.0, -10.8),
            comparator=comp, gens=["300"], protocol="fresh300", targets=[spec],
        )["targets"][0]
        assert split["verdict"] == "NEUTRAL"

        small_diff = gates.evaluate_gates(
            self._two_seed_target_arm("vs_scripted_sampled", "goals_A", -9.9, -9.6),
            comparator=comp, gens=["300"], protocol="fresh300", targets=[spec],
        )["targets"][0]
        assert small_diff["verdict"] == "NEUTRAL"
        assert small_diff["mean_diff"] == pytest.approx(1.0)

        arm_high = self._two_seed_target_arm("vs_scripted_sampled", "goals_A", -8.0, -7.5)
        loses = gates.evaluate_gates(
            self._two_seed_target_arm("vs_scripted_sampled", "goals_A", -10.5, -11.0),
            comparator=arm_high, gens=["300"], protocol="fresh300", targets=[spec],
        )["targets"][0]
        assert loses["verdict"] == "LOSES"

        one_seed = {"s42": {"300": {"self_sampled": _fake_census_point(), "vs_scripted_sampled": _fake_census_point(goals_A=(-5.0, 0.4))}}}
        under = gates.evaluate_gates(
            one_seed, comparator=comp, gens=["300"], protocol="fresh300", targets=[spec],
        )["targets"][0]
        assert under["verdict"] == "NEUTRAL"
        assert under["note"]

        # Lower-is-better direction
        lower_beats = gates.evaluate_gates(
            self._two_seed_target_arm("vs_scripted_sampled", "goals_B", 4.0, 5.0),
            comparator=self._two_seed_target_arm("vs_scripted_sampled", "goals_B", 8.0, 9.0),
            gens=["300"], protocol="fresh300", targets=["goals_B@vs_scripted_sampled:lower:2.0"],
        )["targets"][0]
        assert lower_beats["verdict"] == "BEATS"

    def test_between_seed_sample_sd_at_last_gen(self):
        from tools.audit import gates

        arm = {
            "s42": {"300": {"self_sampled": _fake_census_point(hold_share_pooled=0.1906)}},
            "s43": {"300": {"self_sampled": _fake_census_point(hold_share_pooled=0.1739)}},
        }
        res = gates.evaluate_gates(arm, gens=["300"], protocol="fresh300")
        sd = res["gates"]["G1"]["metrics"]["hold_share_pooled"]["points"]["300"]["arm_sd"]
        assert sd == pytest.approx(0.01180868, abs=1e-5)
        md = gates.format_markdown(res)
        assert "0.0118" in md

    def test_cli_fresh300_with_multi_tag_target(self, tmp_path, capsys):
        from tools.audit import gates

        for seed, gd, hs in (("s42", -8.0, 0.19), ("s43", -7.5, 0.18)):
            for tag, pt in (("self_sampled", _fake_census_point(hold_share_pooled=hs)),
                            ("vs_scripted_sampled", {"summary": {"n_matches": 48, "metrics": {"goal_diff": {"mean": gd, "se": 0.4, "n": 48}}}})):
                p = tmp_path / "arm" / seed / "gen300" / tag
                p.mkdir(parents=True)
                (p / "census.json").write_text(json.dumps(pt), encoding="utf-8")
        for seed, gd, hs in (("s42", -10.5, 0.18), ("s43", -11.0, 0.18)):
            for tag, pt in (("self_sampled", _fake_census_point(hold_share_pooled=hs)),
                            ("vs_scripted_sampled", {"summary": {"n_matches": 48, "metrics": {"goal_diff": {"mean": gd, "se": 0.4, "n": 48}}}})):
                p = tmp_path / "comp" / seed / "gen300" / tag
                p.mkdir(parents=True)
                (p / "census.json").write_text(json.dumps(pt), encoding="utf-8")

        rc = gates.main([
            "--arm", f"s42={tmp_path}/arm/s42", f"s43={tmp_path}/arm/s43",
            "--comparator", f"s42={tmp_path}/comp/s42", f"s43={tmp_path}/comp/s43",
            "--protocol", "fresh300", "--gens", "300",
            "--target", "goal_diff@vs_scripted_sampled:higher:2.0",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "BEATS" in out and "+3.000" in out


# ----------------------------------------------------------------------------
# E2: evaluate_arm writes internals.json for self_sampled
# ----------------------------------------------------------------------------
class TestEvaluateArmE2:
    def test_internals_json_written_deterministic_and_no_writes(self, tmp_path, monkeypatch):
        from tools.audit import evaluate_arm

        snap = write_checkpoint(tmp_path / "checkpoints" / "gen1200", seed=11)
        replays = tmp_path / "replays"
        replays.mkdir()
        before = dir_digest(snap)
        monkeypatch.chdir(tmp_path)

        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        evaluate_arm.evaluate_arm(
            [snap], out1, workers=1, n_override=2, max_steps=20, tags=["self_sampled", "self_mixed"],
        )
        evaluate_arm.evaluate_arm(
            [snap], out2, workers=1, n_override=2, max_steps=20, tags=["self_sampled", "self_mixed"],
        )
        i1 = out1 / "gen1200" / "self_sampled" / "internals.json"
        i2 = out2 / "gen1200" / "self_sampled" / "internals.json"
        assert i1.is_file() and i2.is_file()
        assert i1.read_bytes() == i2.read_bytes()
        assert not (out1 / "gen1200" / "self_mixed" / "internals.json").exists()
        assert dir_digest(snap) == before
        assert list(replays.iterdir()) == []

