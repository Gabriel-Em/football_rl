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
        "kick_outcome_opponent": {"mean": 0.48, "se": 0.01, "n": 96},
        "goal_no_pass_10s_share": {"mean": 0.30, "se": 0.02, "n": 96},
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
# E2: EBE targets and between-seed SD (tools/audit/gates.py)
# ----------------------------------------------------------------------------
class TestGatesE2:
    def test_fresh_g1_checks_last_gen_only(self):
        """Anchor hold share {0.18, 0.20, 0.19, 0.21} (mean 0.195, delta 0.02582): 0.15 at gen 200 followed by 0.19 at
        gen 300 passes G1 (0.19 is inside the anchor range, so gen 200 is not judged); 0.19 at gen 200 followed by
        0.15 at gen 300 fails (0.15 is below every anchor seed and 0.195 - 0.15 = 0.045 >= 0.02582)."""
        from tools.audit import gates

        cal = _make_fake_calibration()
        arm_pass = {"s42": _fresh_arm_seed({"200": {"hold_share_pooled": 0.15}, "300": {"hold_share_pooled": 0.19}})}
        arm_fail = {"s42": _fresh_arm_seed({"200": {"hold_share_pooled": 0.19}, "300": {"hold_share_pooled": 0.15}})}
        res_pass = gates.evaluate_gates(arm_pass, gens=["200", "300"], protocol="fresh", calibration=cal)
        res_fail = gates.evaluate_gates(arm_fail, gens=["200", "300"], protocol="fresh", calibration=cal)
        assert res_pass["gates"]["G1"]["passed"] is True
        assert res_fail["gates"]["G1"]["passed"] is False
        assert res_fail["gates"]["G1"]["metrics"]["hold_share_pooled"]["mean_diff"] == pytest.approx(-0.045, abs=1e-9)

    def _two_seed_target_arm(self, tag: str, key: str, v42: float, v43: float):
        out = {}
        for seed, v in (("s42", v42), ("s43", v43)):
            out[seed] = _fresh_arm_seed({"300": {}})
            out[seed]["300"][tag]["summary"]["metrics"][key] = {"mean": v, "se": 0.4, "n": 48}
        return out

    def test_ebe_targets_hand_cases(self):
        from tools.audit import gates

        cal = _make_fake_calibration()
        comp = self._two_seed_target_arm("vs_scripted_sampled", "goals_A", -10.5, -11.0)
        spec = "goals_A@vs_scripted_sampled:higher:2.0"

        beats = gates.evaluate_gates(
            self._two_seed_target_arm("vs_scripted_sampled", "goals_A", -8.0, -7.5),
            comparator=comp, gens=["300"], protocol="fresh", calibration=cal, targets=[spec],
        )["targets"][-1]
        assert beats["verdict"] == "BEATS"
        assert beats["mean_diff"] == pytest.approx(3.0)

        split = gates.evaluate_gates(
            self._two_seed_target_arm("vs_scripted_sampled", "goals_A", -8.0, -10.8),
            comparator=comp, gens=["300"], protocol="fresh", calibration=cal, targets=[spec],
        )["targets"][-1]
        assert split["verdict"] == "NEUTRAL"

        small_diff = gates.evaluate_gates(
            self._two_seed_target_arm("vs_scripted_sampled", "goals_A", -9.9, -9.6),
            comparator=comp, gens=["300"], protocol="fresh", calibration=cal, targets=[spec],
        )["targets"][-1]
        assert small_diff["verdict"] == "NEUTRAL"
        assert small_diff["mean_diff"] == pytest.approx(1.0)

        arm_high = self._two_seed_target_arm("vs_scripted_sampled", "goals_A", -8.0, -7.5)
        loses = gates.evaluate_gates(
            self._two_seed_target_arm("vs_scripted_sampled", "goals_A", -10.5, -11.0),
            comparator=arm_high, gens=["300"], protocol="fresh", calibration=cal, targets=[spec],
        )["targets"][-1]
        assert loses["verdict"] == "LOSES"

        one_seed = {"s42": self._two_seed_target_arm("vs_scripted_sampled", "goals_A", -5.0, -5.0)["s42"]}
        under = gates.evaluate_gates(
            one_seed, comparator=comp, gens=["300"], protocol="fresh", calibration=cal, targets=[spec],
        )["targets"][-1]
        assert under["verdict"] == "NEUTRAL"
        assert under["note"]

        # Lower-is-better direction
        lower_beats = gates.evaluate_gates(
            self._two_seed_target_arm("vs_scripted_sampled", "goals_B", 4.0, 5.0),
            comparator=self._two_seed_target_arm("vs_scripted_sampled", "goals_B", 8.0, 9.0),
            gens=["300"], protocol="fresh", calibration=cal, targets=["goals_B@vs_scripted_sampled:lower:2.0"],
        )["targets"][-1]
        assert lower_beats["verdict"] == "BEATS"

    def test_between_seed_sample_sd_at_last_gen(self):
        """Two seeds at 0.1906 and 0.1739: mean 0.18225, deviations +-0.00835, sample SD 0.00835 * sqrt(2) = 0.0118087."""
        from tools.audit import gates

        arm = {
            "s42": _fresh_arm_seed({"300": {"hold_share_pooled": 0.1906}}),
            "s43": _fresh_arm_seed({"300": {"hold_share_pooled": 0.1739}}),
        }
        res = gates.evaluate_gates(arm, gens=["300"], protocol="fresh", calibration=_make_fake_calibration())
        sd = res["gates"]["G1"]["metrics"]["hold_share_pooled"]["points"]["300"]["arm_sd"]
        assert sd == pytest.approx(0.01180868, abs=1e-5)
        md = gates.format_markdown(res)
        assert "0.0118" in md

    def test_cli_explicit_target_against_comparator(self, tmp_path, capsys):
        """`--target goal_diff@vs_scripted_sampled:higher:2.0` is judged after T1-T3: arm {-8.0, -7.5} against comparator
        {-10.5, -11.0} on the blue tag gives mean diff +3.000 and BEATS (the arm's lowest seed beats the comparator's highest,
        and 3.0 >= 2.0). The red tag is the same in both runs, so T1 (the blue/red average) moves by only +1.5."""
        from tools.audit import gates

        cal_path = tmp_path / "cal.json"
        cal_path.write_text(json.dumps(_make_fake_calibration()), encoding="utf-8")
        for name, values in (("arm", {"s42": -8.0, "s43": -7.5}), ("comp", {"s42": -10.5, "s43": -11.0})):
            for seed, gd in values.items():
                _write_e3_anchor_dir(tmp_path / name / seed, "300", self_overrides=_FRESH_SELF_BASE, bot_blue_gd=gd, bot_red_gd=-20.0)

        rc = gates.main([
            "--arm", f"s42={tmp_path}/arm/s42", f"s43={tmp_path}/arm/s43",
            "--comparator", f"s42={tmp_path}/comp/s42", f"s43={tmp_path}/comp/s43",
            "--protocol", "fresh", "--calibration", str(cal_path), "--gens", "300",
            "--target", "goal_diff@vs_scripted_sampled:higher:2.0",
        ])
        assert rc == 0
        rows = {ln.split("|")[1].strip(): [c.strip() for c in ln.split("|")[1:-1]] for ln in capsys.readouterr().out.splitlines()
                if ln.startswith("| T1 (") or ln.startswith("| goal_diff@")}
        explicit = rows["goal_diff@vs_scripted_sampled"]
        assert explicit[6] == "+3.000" and explicit[7] == "BEATS"
        t1 = rows["T1 (informational at gen 300)"]
        assert t1[6] == "+1.500" and t1[7] == "NEUTRAL"


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



# ----------------------------------------------------------------------------
# X1: decoupled action heads in the harness and in policy internals
# ----------------------------------------------------------------------------
def write_head_checkpoint(directory: Path, heads: dict, seed: int = 0, spread: float = 0.3) -> Path:
    """Checkpoint whose `blue_*` / `red_*` networks use the head named in `heads`; hybrid5 outputs depend on the observation."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    for prefix in ("blue", "red"):
        for kind in ("field", "keeper"):
            net = ActorCritic(policy_head=heads[prefix])
            if heads[prefix] == "hybrid5":
                with torch.no_grad():
                    net.actor_mean[4].weight.normal_(0.0, spread)
            torch.save(net.state_dict(), directory / f"{prefix}_{kind}_policy.pt")
    return directory


def write_constant_hybrid_checkpoint(directory: Path, z_kick: float, z_spec: float, floor: float = 0.005) -> Path:
    """Hybrid5 checkpoint with observation-independent outputs: throttle 0.7, steer -0.2, power 0.4 (pre-tanh)."""
    directory.mkdir(parents=True, exist_ok=True)
    for prefix in ("blue", "red"):
        for kind in ("field", "keeper"):
            net = ActorCritic(policy_head="hybrid5", fire_prob_floor=floor)
            with torch.no_grad():
                net.actor_mean[4].weight.zero_()
                net.actor_mean[4].bias.copy_(torch.tensor([0.7, -0.2, 0.4, z_kick, z_spec]))
            torch.save(net.state_dict(), directory / f"{prefix}_{kind}_policy.pt")
    return directory


# tanh(0.7), tanh(-0.2); power mean (tanh(0.4) + 1) / 2 = 0.689974 -> kick action 0.35 + 0.65 * 0.689974.
CONST_THR, CONST_STEER, CONST_KICK_ACTION = 0.6043678, -0.1973753, 0.7984834
# floor 0.005, bound 6: p(z = +3) = 0.9367710 -> 2p - 1 = 0.8735419.
CONST_FIRE_MEAN = 0.8735419


def _obs_cen(seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(6, 82)).astype(np.float32), rng.normal(size=(6, 54)).astype(np.float32)


class TestHybridHarness:
    def test_team_policy_detects_the_head_from_the_output_width(self, tmp_path):
        d = write_head_checkpoint(tmp_path / "c", {"blue": "gaussian4", "red": "hybrid5"})
        assert harness.TeamPolicy(str(d), "blue").head == 0
        assert harness.TeamPolicy(str(d), "red").head == 1

    def test_gaussian4_teams_keep_their_four_tuple_heads_and_default_head_field(self, ckpt):
        team = harness.TeamPolicy(str(ckpt), "blue")
        assert all(len(h) == 4 for h in team.heads)
        trace = harness.play_match(quick_spec(ckpt, opponent="scripted", max_steps=3), 0)
        assert trace["head"].tolist() == [0, 0] and trace["head"].dtype == np.int8

    def test_a_hybrid5_checkpoint_is_deterministic(self, tmp_path):
        d = write_head_checkpoint(tmp_path / "c", {"blue": "hybrid5", "red": "hybrid5"}, seed=3)
        a = harness.run_matches(quick_spec(d, max_steps=40))
        b = harness.run_matches(quick_spec(d, max_steps=40))
        for ta, tb in zip(a, b):
            assert set(ta) == set(tb) and "head" in ta
            for key in ta:
                assert np.array_equal(ta[key], tb[key]), key

    @pytest.mark.parametrize(
        "opponent,swap,expected",
        [
            ("self", False, [0, 1]),
            ("self", True, [1, 0]),
            ("scripted", False, [0, 0]),
            ("scripted", True, [1, 0]),
            ("scripted2", False, [0, 0]),
            ("frozen", False, [0, 0]),
            ("frozen", True, [1, 0]),
        ],
    )
    def test_head_records_the_seats_with_and_without_swap(self, tmp_path, opponent, swap, expected):
        # blue_* = gaussian4, red_* = hybrid5; the frozen directory is a gaussian4 checkpoint.
        d = write_head_checkpoint(tmp_path / "c", {"blue": "gaussian4", "red": "hybrid5"})
        frozen = write_head_checkpoint(tmp_path / "f", {"blue": "gaussian4", "red": "gaussian4"}, seed=1)
        opp = f"frozen:{frozen}" if opponent == "frozen" else opponent
        spec = quick_spec(d, opponent=opp, swap=swap, n=1, max_steps=4)
        trace = harness.play_match(spec, 0)
        assert trace["head"].tolist() == expected

    @pytest.mark.parametrize("swap,expected", [(False, [0, 1]), (True, [1, 0])])
    def test_cli_swap_writes_the_seat_heads_into_the_traces(self, tmp_path, swap, expected):
        # blue_* = gaussian4, red_* = hybrid5; `--swap` hands team A the red_* policies.
        d = write_head_checkpoint(tmp_path / "c", {"blue": "gaussian4", "red": "hybrid5"})
        out = tmp_path / "out"
        argv = ["--checkpoint-dir", str(d), "--opp", "self", "--n", "2", "--seed", "3", "--workers", "1",
                "--max-steps", "4", "--out-dir", str(out)]
        assert harness.main(argv + (["--swap"] if swap else [])) == 0
        files = sorted(out.glob("m*.npz"))
        assert [f.name for f in files] == ["m000.npz", "m001.npz"]
        for f in files:
            with np.load(f) as trace:
                assert trace["head"].tolist() == expected

    def test_a_hybrid5_arm_plays_a_gaussian4_frozen_opponent(self, tmp_path):
        arm = write_head_checkpoint(tmp_path / "arm", {"blue": "hybrid5", "red": "hybrid5"})
        frozen = write_head_checkpoint(tmp_path / "f", {"blue": "gaussian4", "red": "gaussian4"}, seed=1)
        for swap in (False, True):
            trace = harness.play_match(quick_spec(arm, opponent=f"frozen:{frozen}", swap=swap, max_steps=30), 0)
            assert trace["head"].tolist() == [1, 0]
            assert trace["act"].shape[1:] == (12, 4) and np.isfinite(trace["act"]).all()
        # A gaussian4 learner against a hybrid5 frozen opponent: heads [0, 1].
        g_arm = write_head_checkpoint(tmp_path / "g", {"blue": "gaussian4", "red": "gaussian4"}, seed=2)
        h_frozen = write_head_checkpoint(tmp_path / "h", {"blue": "hybrid5", "red": "hybrid5"}, seed=3)
        assert harness.play_match(quick_spec(g_arm, opponent=f"frozen:{h_frozen}", max_steps=3), 0)["head"].tolist() == [0, 1]

    def test_hybrid5_trace_rows_are_env_decoded_and_means_hold_fire_probabilities(self, tmp_path):
        d = write_head_checkpoint(tmp_path / "c", {"blue": "hybrid5", "red": "hybrid5"}, seed=4)
        traces = harness.run_matches(quick_spec(d, max_steps=60))
        act = np.concatenate([t["act"] for t in traces])
        mean = np.concatenate([t["mean"] for t in traces])
        kick, special = act[:, :, 2], act[:, :, 3]
        assert ((kick == -1.0) | (kick > 0.35)).all() and ((special == -1.0) | (special == 1.0)).all()
        assert (kick > 0.35).any() and (kick == -1.0).any()
        # 2p - 1 with p in [0.005 + 0.99 * sigmoid(-6), 0.995 - 0.99 * sigmoid(-6)] = [0.007448, 0.992552].
        assert (np.abs(mean[:, :, 2:]) <= 0.98511).all()

    def test_mixed_mode_leaves_throttle_and_steer_at_their_means(self, tmp_path):
        d = write_head_checkpoint(tmp_path / "c", {"blue": "hybrid5", "red": "hybrid5"}, seed=5)
        team = harness.TeamPolicy(str(d), "blue")
        obs, cen = _obs_cen()
        actions, _, means = team.act(obs, cen, "mixed", 1.0, torch.Generator().manual_seed(1))
        np.testing.assert_array_equal(actions[:, :2], means[:, :2])
        sampled, _, _ = team.act(obs, cen, "sampled", 1.0, torch.Generator().manual_seed(1))
        assert np.abs(sampled[:, :2] - means[:, :2]).max() > 0.0

    def test_mixed_mode_fires_kicks_at_the_deterministic_power_mean(self, tmp_path):
        # Constant head: power mean (tanh(0.4) + 1) / 2 = 0.689974 -> kick action CONST_KICK_ACTION; kick fires with
        # p = 0.9368, so six slots under one seeded generator contain fired kicks.
        team = harness.TeamPolicy(str(write_constant_hybrid_checkpoint(tmp_path / "c", 3.0, 0.0)), "blue")
        obs, cen = _obs_cen()
        mixed, _, _ = team.act(obs, cen, "mixed", 1.0, torch.Generator().manual_seed(1))
        fired = mixed[:, 2] != -1.0
        assert fired.sum() >= 3
        np.testing.assert_allclose(mixed[fired, 2], CONST_KICK_ACTION, atol=1e-6)
        # The sampled mode moves the fired power off the mean, so the assertion above can fail.
        sampled, _, _ = team.act(obs, cen, "sampled", 1.0, torch.Generator().manual_seed(1))
        assert np.abs(sampled[sampled[:, 2] != -1.0, 2] - CONST_KICK_ACTION).max() > 1e-3

    def test_greedy_uses_the_means_and_fires_above_one_half(self, tmp_path):
        obs, cen = _obs_cen()
        fires = harness.TeamPolicy(str(write_constant_hybrid_checkpoint(tmp_path / "on", 3.0, -3.0)), "blue")
        a, _, m = fires.act(obs, cen, "greedy", 1.0, torch.Generator().manual_seed(0))
        np.testing.assert_allclose(a, [[CONST_THR, CONST_STEER, CONST_KICK_ACTION, -1.0]] * 6, atol=1e-6)
        np.testing.assert_allclose(m, [[CONST_THR, CONST_STEER, CONST_FIRE_MEAN, -CONST_FIRE_MEAN]] * 6, atol=1e-6)
        silent = harness.TeamPolicy(str(write_constant_hybrid_checkpoint(tmp_path / "off", -3.0, 3.0)), "blue")
        a, _, _ = silent.act(obs, cen, "greedy", 1.0, torch.Generator().manual_seed(0))
        np.testing.assert_allclose(a, [[CONST_THR, CONST_STEER, -1.0, 1.0]] * 6, atol=1e-6)

    def test_sigma_scale_moves_the_gaussian_heads_only(self, tmp_path):
        team = harness.TeamPolicy(str(write_constant_hybrid_checkpoint(tmp_path / "c", 0.0, 0.0)), "blue")
        obs, cen = _obs_cen()
        zero, _, _ = team.act(obs, cen, "sampled", 0.0, torch.Generator().manual_seed(3))
        one, _, _ = team.act(obs, cen, "sampled", 1.0, torch.Generator().manual_seed(3))
        two, _, _ = team.act(obs, cen, "sampled", 2.0, torch.Generator().manual_seed(3))
        np.testing.assert_allclose(zero[:, :2], [[CONST_THR, CONST_STEER]] * 6, atol=1e-6)
        np.testing.assert_allclose(two[:, :2] - zero[:, :2], 2.0 * (one[:, :2] - zero[:, :2]), rtol=1e-4, atol=1e-6)
        # Same generator state: the Bernoulli draws are the same whatever the scale, and sigma = 0 still fires at random.
        np.testing.assert_array_equal(zero[:, 2:] > 0.35, one[:, 2:] > 0.35)
        fired = np.concatenate(
            [team.act(obs, cen, "sampled", 0.0, torch.Generator().manual_seed(s))[0][:, 2] for s in range(60)]
        )
        assert (fired > 0.35).any() and (fired == -1.0).any()

    def test_fire_bounds_come_from_the_checkpoint_not_from_defaults(self, tmp_path):
        """A checkpoint trained with floor 0.01 evaluates with floor 0.01, not the default 0.005."""
        d = write_constant_hybrid_checkpoint(tmp_path / "c", -1e6, -1e6, floor=0.01)
        team = harness.TeamPolicy(str(d), "blue")
        obs, cen = _obs_cen()
        _, _, m = team.act(obs, cen, "greedy", 1.0, torch.Generator().manual_seed(0))
        # p = 0.01 + 0.98 * 0.0024726 = 0.0124232 -> 2p - 1 = -0.9751536 (the default floor 0.005 would give -0.985104).
        np.testing.assert_allclose(m[:, 2], -0.9751536, atol=1e-6)


class TestInternalsHybridHead:
    @staticmethod
    def _trace(head):
        from tools.audit import internals  # noqa: F401

        tr = make_internals_trace(8)
        if head is not None:
            tr["head"] = np.array(head, dtype=np.int8)
        # Team A holder 3 (rows 0-3) with kick means 2p - 1; team B holder 8 (rows 4-7).
        for t, (m, a) in enumerate(zip([0.2, 0.2, -0.2, -0.2], [0.8, -1.0, -1.0, 0.5])):
            tr["holder"][t], tr["hold_t"][t] = 3, 0.10
            tr["mean"][t, 3, 2], tr["act"][t, 3, 2] = m, a
        for t, (m, a) in zip(range(4, 8), zip([0.2, 0.2, 0.9, 0.9], [-1.0, -1.0, 0.8, -1.0])):
            tr["holder"][t], tr["hold_t"][t] = 8, 0.10
            tr["mean"][t, 8, 2], tr["act"][t, 8, 2] = m, a
        return tr

    def test_intent_is_p_above_one_half_for_hybrid_seats_and_mean_above_0_35_for_gaussian_seats(self):
        from tools.audit import internals

        m = internals.match_internals(self._trace([1, 0]))
        # Intent rows: team A rows 0, 1 (mean 0.2 > 0); team B rows 6, 7 (mean 0.9 > 0.35); rows 4, 5 (0.2) are not.
        assert m["p_kick_obs"] == pytest.approx(3 / 8)
        assert m["p_kick_intent"] == pytest.approx(4 / 8)
        assert m["kick_from_intent"] == pytest.approx(2 / 3)
        assert m["p_kick_given_intent"] == pytest.approx(2 / 4)
        assert m["p_kick_given_no_intent"] == pytest.approx(1 / 4)

    def test_traces_without_a_head_field_read_as_gaussian4(self):
        from tools.audit import internals

        m = internals.match_internals(self._trace(None))
        assert m["p_kick_intent"] == pytest.approx(2 / 8)          # only the two 0.9 rows exceed 0.35
        assert internals.match_internals(self._trace([0, 0])) == m

    def test_special_intent_uses_each_seats_own_threshold(self):
        from tools.audit import internals

        tr = self._trace([1, 0])
        for t, m in enumerate([0.2, -0.2, 0.2]):                    # player 2, team A, hybrid: intent when mean > 0
            tr["act"][t, 2, 3], tr["mean"][t, 2, 3] = 1.0, m
        for t, m in enumerate([0.2, 0.9]):                          # player 9, team B, gaussian: intent when mean > 0.5
            tr["act"][t, 9, 3], tr["mean"][t, 9, 3] = 1.0, m
        r = internals.match_internals(tr)
        assert r["n_special_fires"] == 5
        assert r["special_from_intent"] == pytest.approx(3 / 5)     # 2 hybrid + 1 gaussian
        assert r["special_rate"] == pytest.approx(5 / (8 * 10))

    def test_saturation_and_grad_scale_skip_probability_heads(self):
        from tools.audit import internals

        r = internals.match_internals(self._trace([1, 0]))
        # Throttle and steer are tanh means for every seat (all -1 -> saturated, gradient scale 0).
        assert r["sat99_throttle"] == 1.0 and r["grad_scale_steer"] == 0.0
        # Kick and special: only team B's field players (gaussian) count.
        # Player 8 kick means: rows 0-3 -1 (sat), 4-5 0.2, 6-7 0.9 -> 4 sat of 8; the other four B players are always -1.
        assert r["sat99_kick"] == pytest.approx((4 + 4 * 8) / 40)
        assert r["grad_scale_kick"] == pytest.approx((2 * 0.96 + 2 * 0.19) / 40)
        both = internals.match_internals(self._trace([1, 1]))
        assert both["sat99_kick"] is None and both["grad_scale_kick"] is None
        assert both["sat99_special"] is None and both["grad_scale_special"] is None
        assert both["sat99_throttle"] == 1.0

    def test_head_field_is_read_from_stored_traces(self, tmp_path):
        from tools.audit import internals

        d = tmp_path / "t"
        d.mkdir()
        tr = self._trace([1, 0])
        np.savez_compressed(d / "m000.npz", **tr)
        loaded = list(internals._iter_internal_traces(d))
        assert loaded[0]["head"].tolist() == [1, 0]
        assert internals.match_internals(loaded[0])["p_kick_intent"] == pytest.approx(4 / 8)
        old = self._trace(None)
        np.savez_compressed(d / "m000.npz", **old)
        assert "head" not in next(iter(internals._iter_internal_traces(d)))

# ----------------------------------------------------------------------------
# E3: Anchor calibration, relative fresh gates, red tags, and --recensus
# ----------------------------------------------------------------------------
def _write_e3_anchor_dir(
    root: Path,
    gen: str,
    *,
    self_overrides: dict | None = None,
    bot_blue_gd: float | None = -16.0,
    bot_red_gd: float | None = -18.0,
    frozen_blue_gd: float | None = -1.0,
    frozen_red_gd: float | None = -1.2,
    omit_tags: tuple[str, ...] = (),
) -> Path:
    """Write fake evaluate_arm output for one seed at one generation."""
    gen_dir = root / f"gen{int(gen):04d}"
    tag_payloads = {
        "self_sampled": _fake_census_point(**(self_overrides or {})),
        "vs_scripted_sampled": {"summary": {"n_matches": 48, "metrics": {"goal_diff": {"mean": bot_blue_gd, "se": 0.4, "n": 48}}}},
        "vs_scripted_sampled_red": {"summary": {"n_matches": 48, "metrics": {"goal_diff": {"mean": bot_red_gd, "se": 0.4, "n": 48}}}},
        "vs_frozen": {"summary": {"n_matches": 48, "metrics": {"goal_diff": {"mean": frozen_blue_gd, "se": 0.2, "n": 48}}}},
        "vs_frozen_red": {"summary": {"n_matches": 48, "metrics": {"goal_diff": {"mean": frozen_red_gd, "se": 0.2, "n": 48}}}},
    }
    for tag, payload in tag_payloads.items():
        if tag in omit_tags:
            continue
        tdir = gen_dir / tag
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / "census.json").write_text(json.dumps(payload), encoding="utf-8")
        if tag == "self_sampled":
            (tdir / "internals.json").write_text(
                json.dumps({"summary": {"n_matches": 96, "metrics": {"aim_teammate_share_sampled": {"mean": 0.12, "se": 0.004, "n": 96}}}}),
                encoding="utf-8",
            )
    return root


def _make_fake_calibration(
    *,
    gen: str = "300",
    guard_overrides: dict | None = None,
    target_overrides: dict | None = None,
) -> dict:
    """Build an in-memory calibration dict matching calibrate.py's schema."""
    guards = {
        "hold_share_pooled": {"gate": "G1", "tag": "self_sampled", "worse": "lower", "floor": 0.010, "per_seed": {"42": 0.18, "43": 0.20, "44": 0.19, "45": 0.21}, "mean": 0.195, "sd": 0.012909944487358056, "delta": 0.025819888974716113},
        "holder_spell_s_pooled": {"gate": "G1", "tag": "self_sampled", "worse": "lower", "floor": 0.020, "per_seed": {"42": 0.27, "43": 0.27, "44": 0.27, "45": 0.27}, "mean": 0.27, "sd": 0.0, "delta": 0.020},
        "gk_in_box_pooled": {"gate": "G2", "tag": "self_sampled", "worse": "lower", "floor": 0.010, "per_seed": {"42": 0.99, "43": 0.99, "44": 0.99, "45": 0.99}, "mean": 0.99, "sd": 0.0, "delta": 0.010},
        "gk_goal_line_pooled": {"gate": "G2", "tag": "self_sampled", "worse": "higher", "floor": 0.030, "per_seed": {"42": 0.175, "43": 0.175, "44": 0.175, "45": 0.175}, "mean": 0.175, "sd": 0.0, "delta": 0.030},
        "n_within_100_pooled": {"gate": "G3", "tag": "self_sampled", "worse": "higher", "floor": 0.030, "per_seed": {"42": 0.50, "43": 0.50, "44": 0.50, "45": 0.50}, "mean": 0.50, "sd": 0.0, "delta": 0.030},
        "cluster_ge4_pooled": {"gate": "G3", "tag": "self_sampled", "worse": "higher", "floor": 0.010, "per_seed": {"42": 0.06, "43": 0.06, "44": 0.06, "45": 0.06}, "mean": 0.06, "sd": 0.0, "delta": 0.010},
        "role_inversion_pooled": {"gate": "G4", "tag": "self_sampled", "worse": "higher", "floor": 0.030, "per_seed": {"42": 0.28, "43": 0.30, "44": 0.29, "45": 0.31}, "mean": 0.295, "sd": 0.012909944487358056, "delta": 0.05},
        "goals_total": {"gate": "G5", "tag": "self_sampled", "worse": "lower", "floor": 0.30, "per_seed": {"42": 2.2, "43": 2.2, "44": 2.2, "45": 2.2}, "mean": 2.2, "sd": 0.0, "delta": 0.30},
        "scoreless": {"gate": "G5", "tag": "self_sampled", "worse": "higher", "floor": 0.05, "per_seed": {"42": 0.10, "43": 0.10, "44": 0.10, "45": 0.10}, "mean": 0.10, "sd": 0.0, "delta": 0.05},
    }
    if guard_overrides:
        for k, v in guard_overrides.items():
            guards[k] = {**guards[k], **v}
    targets = {
        "T1": {"id": "T1", "key": "goal_diff", "tags": ["vs_scripted_sampled", "vs_scripted_sampled_red"], "direction": "higher", "floor": 2.0, "per_seed": {"42": -19.0, "43": -18.0, "44": -20.0, "45": -17.0}, "mean": -18.5, "sd": 1.2909944487358056, "delta": 2.581988897471611},
        "T2": {"id": "T2", "key": "goal_diff", "tags": ["vs_frozen", "vs_frozen_red"], "direction": "higher", "floor": 0.5, "per_seed": {"42": -1.0, "43": -1.2, "44": -1.1, "45": -0.9}, "mean": -1.05, "sd": 0.12909944487358058, "delta": 0.5},
        "T3": {"id": "T3", "key": "kick_outcome_teammate", "tags": ["self_sampled"], "direction": "higher", "floor": 0.03, "per_seed": {"42": 0.31, "43": 0.30, "44": 0.32, "45": 0.29}, "mean": 0.305, "sd": 0.012909944487358056, "delta": 0.03},
    }
    if target_overrides:
        for k, v in target_overrides.items():
            targets[k] = {**targets[k], **v}
    return {
        "gen": gen,
        "git_commit": "abc1234",
        "anchor_seeds": ["42", "43", "44", "45"],
        "anchor_dirs": {"42": "a42", "43": "a43", "44": "a44", "45": "a45"},
        "guards": guards,
        "targets": targets,
    }


def _fresh_arm_seed(gens_vals: dict[str, dict], *, bot_b: float = -18.0, bot_r: float = -18.0, fr_b: float = -1.0, fr_r: float = -1.0, kot: float = 0.31) -> dict:
    out = {}
    for g, ov in gens_vals.items():
        self_ov = {
            "kick_outcome_teammate": (kot, 0.01),
            "gk_in_box_pooled": (0.99, 0.002),
            "gk_goal_line_pooled": (0.175, 0.01),
            "aim_teammate_share_sampled": (0.12, 0.004),
            **ov,
        }
        out[g] = {
            "self_sampled": _fake_census_point(**self_ov),
            "vs_scripted_sampled": {"summary": {"n_matches": 48, "metrics": {"goal_diff": {"mean": bot_b, "se": 0.4, "n": 48}}}},
            "vs_scripted_sampled_red": {"summary": {"n_matches": 48, "metrics": {"goal_diff": {"mean": bot_r, "se": 0.4, "n": 48}}}},
            "vs_frozen": {"summary": {"n_matches": 48, "metrics": {"goal_diff": {"mean": fr_b, "se": 0.2, "n": 48}}}},
            "vs_frozen_red": {"summary": {"n_matches": 48, "metrics": {"goal_diff": {"mean": fr_r, "se": 0.2, "n": 48}}}},
        }
    return out


class TestCalibrateE3:
    def test_calibrate_hand_values_sd_floor_and_two_tag_target(self, tmp_path):
        """[RED] Exact sample SD (ddof=1), floor binding, and two-tag per-seed mean."""
        from tools.audit import calibrate

        # 4 seeds: hold_share {0.18, 0.20, 0.19, 0.21} (s=0.012910, delta=0.025820)
        # cluster_ge4_pooled {0.190, 0.191, 0.192, 0.193} (s=0.001291, floor 0.010 binds)
        specs = [
            ("42", 0.18, 0.190),
            ("43", 0.20, 0.191),
            ("44", 0.19, 0.192),
            ("45", 0.21, 0.193),
        ]
        anchor_dirs = {}
        for seed, hs, cl in specs:
            anchor_dirs[seed] = _write_e3_anchor_dir(
                tmp_path / f"s{seed}",
                "300",
                self_overrides={
                    "hold_share_pooled": hs,
                    "cluster_ge4_pooled": cl,
                    "gk_in_box_pooled": (0.99, 0.002),
                    "gk_goal_line_pooled": (0.175, 0.01),
                    "kick_outcome_teammate": (0.31, 0.01),
                },
            )
        cal4 = calibrate.calibrate_from_dirs(anchor_dirs, gen="300")
        hs_entry = cal4["guards"]["hold_share_pooled"]
        assert hs_entry["mean"] == pytest.approx(0.195, abs=1e-9)
        assert hs_entry["sd"] == pytest.approx(0.012910, abs=1e-6)
        assert hs_entry["delta"] == pytest.approx(0.025820, abs=1e-6)

        cl_entry = cal4["guards"]["cluster_ge4_pooled"]
        assert cl_entry["sd"] == pytest.approx(0.001291, abs=1e-6)
        assert cl_entry["delta"] == pytest.approx(0.010, abs=1e-9)

        # 3 seeds for two-tag target: blue {-10, -18, -16}, red {-14, -20, -16}
        three_dirs = {}
        for seed, bb, br in (("42", -10.0, -14.0), ("43", -18.0, -20.0), ("44", -16.0, -16.0)):
            three_dirs[seed] = _write_e3_anchor_dir(
                tmp_path / f"t3_{seed}",
                "300",
                self_overrides={"gk_in_box_pooled": (0.99, 0.002), "gk_goal_line_pooled": (0.175, 0.01), "kick_outcome_teammate": (0.31, 0.01)},
                bot_blue_gd=bb,
                bot_red_gd=br,
            )
        cal3 = calibrate.calibrate_from_dirs(three_dirs, gen="300")
        t1 = cal3["targets"]["T1"]
        assert t1["per_seed"] == {"42": pytest.approx(-12.0), "43": pytest.approx(-19.0), "44": pytest.approx(-16.0)}
        assert t1["mean"] == pytest.approx(-15.6667, abs=1e-4)
        assert t1["sd"] == pytest.approx(3.5119, abs=1e-4)
        assert t1["delta"] == pytest.approx(7.0238, abs=1e-4)

    def test_calibrate_rejects_fewer_than_three_seeds_and_none_values(self, tmp_path):
        """[RED] Fewer than 3 seeds or None/missing metric errors naming seed, tag, and metric."""
        from tools.audit import calibrate

        two_dirs = {
            seed: _write_e3_anchor_dir(
                tmp_path / f"two_{seed}",
                "300",
                self_overrides={"gk_in_box_pooled": (0.99, 0.002), "gk_goal_line_pooled": (0.175, 0.01), "kick_outcome_teammate": (0.31, 0.01)},
            )
            for seed in ("42", "43")
        }
        with pytest.raises(ValueError, match="3"):
            calibrate.calibrate_from_dirs(two_dirs, gen="300")

        bad_dirs = {
            seed: _write_e3_anchor_dir(
                tmp_path / f"bad_{seed}",
                "300",
                self_overrides={
                    "gk_in_box_pooled": (0.99, 0.002),
                    "gk_goal_line_pooled": (0.175, 0.01),
                    "kick_outcome_teammate": (0.31, 0.01),
                    **({"hold_share_pooled": (None, 0.0)} if seed == "43" else {}),
                },
            )
            for seed in ("42", "43", "44")
        }
        with pytest.raises(ValueError) as exc_info:
            calibrate.calibrate_from_dirs(bad_dirs, gen="300")
        msg = str(exc_info.value)
        assert "43" in msg and "self_sampled" in msg and "hold_share_pooled" in msg

        missing_tag_dirs = {
            seed: _write_e3_anchor_dir(
                tmp_path / f"mtag_{seed}",
                "300",
                self_overrides={"gk_in_box_pooled": (0.99, 0.002), "gk_goal_line_pooled": (0.175, 0.01), "kick_outcome_teammate": (0.31, 0.01)},
                omit_tags=("vs_frozen_red",) if seed == "44" else (),
            )
            for seed in ("42", "43", "44")
        }
        with pytest.raises(ValueError) as exc_tag:
            calibrate.calibrate_from_dirs(missing_tag_dirs, gen="300")
        assert "44" in str(exc_tag.value) and "vs_frozen_red" in str(exc_tag.value)

    def test_calibrate_cli_writes_json_md_and_calls_git_once(self, tmp_path, monkeypatch):
        """[RED] CLI writes JSON+MD, records gen, anchor_dirs, and git_commit via one helper call."""
        from tools.audit import calibrate

        anchor_args = []
        for seed in ("42", "43", "44"):
            d = _write_e3_anchor_dir(
                tmp_path / f"cli_{seed}",
                "300",
                self_overrides={"gk_in_box_pooled": (0.99, 0.002), "gk_goal_line_pooled": (0.175, 0.01), "kick_outcome_teammate": (0.31, 0.01)},
            )
            anchor_args.append(f"{seed}={d}")
        calls = []

        def fake_git():
            calls.append(1)
            return "deadbee-dirty"

        monkeypatch.setattr(calibrate, "current_git_commit", fake_git)
        out_json = tmp_path / "cal.json"
        out_md = tmp_path / "cal.md"
        rc = calibrate.main(["--anchor", *anchor_args, "--gen", "300", "--json", str(out_json), "--md", str(out_md)])
        assert rc == 0
        assert len(calls) == 1
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert data["gen"] == "300"
        assert data["git_commit"] == "deadbee-dirty"
        assert set(data["anchor_dirs"]) == {"42", "43", "44"}
        assert set(data["guards"]) == {
            "hold_share_pooled", "holder_spell_s_pooled", "gk_in_box_pooled", "gk_goal_line_pooled",
            "n_within_100_pooled", "cluster_ge4_pooled", "role_inversion_pooled", "goals_total", "scoreless",
        }
        assert set(data["targets"]) == {"T1", "T2", "T3"}
        assert out_md.is_file() and "T1" in out_md.read_text(encoding="utf-8")

    def test_calibrate_cli_requires_json_and_sorts_seeds_numerically(self, tmp_path):
        """[RED] --json is required on CLI and --anchor seeds sort numerically."""
        from tools.audit import calibrate

        anchor_args = []
        anchor_dirs = {}
        for seed in ("100", "42", "43"):
            d = _write_e3_anchor_dir(
                tmp_path / f"num_{seed}",
                "300",
                self_overrides={"gk_in_box_pooled": (0.99, 0.002), "gk_goal_line_pooled": (0.175, 0.01), "kick_outcome_teammate": (0.31, 0.01)},
            )
            anchor_args.append(f"{seed}={d}")
            anchor_dirs[seed] = d

        with pytest.raises(SystemExit) as exc_info:
            calibrate.main(["--anchor", *anchor_args, "--gen", "300"])
        assert exc_info.value.code != 0

        cal = calibrate.calibrate_from_dirs(anchor_dirs, gen="300", git_commit="abc1234")
        assert cal["anchor_seeds"] == ["42", "43", "100"]
        md = calibrate.format_markdown(cal)
        assert "seeds=[42, 43, 100]" in md.splitlines()[0]
        g1_line = next(ln for ln in md.splitlines() if "| hold_share_pooled |" in ln)
        assert g1_line.index("42=") < g1_line.index("43=") < g1_line.index("100=")


class TestGatesFreshE3:
    def test_relative_guard_lower_and_higher_hand_cases(self):
        """[RED] Relative guards require both strict seed separation and mean diff >= delta."""
        from tools.audit import gates

        cal = _make_fake_calibration()

        # Worse = lower: anchor {0.18, 0.20, 0.19, 0.21}, delta 0.025820
        fail_arm = {
            "s1": _fresh_arm_seed({"300": {"hold_share_pooled": 0.15}}),
            "s2": _fresh_arm_seed({"300": {"hold_share_pooled": 0.16}}),
        }
        res_fail = gates.evaluate_gates(fail_arm, gens=["300"], protocol="fresh", calibration=cal)
        assert res_fail["gates"]["G1"]["passed"] is False
        assert res_fail["gates"]["G1"]["metrics"]["hold_share_pooled"]["passed"] is False
        assert res_fail["gates"]["G1"]["metrics"]["hold_share_pooled"]["mean_diff"] == pytest.approx(-0.040, abs=1e-9)

        unsep_arm = {
            "s1": _fresh_arm_seed({"300": {"hold_share_pooled": 0.15}}),
            "s2": _fresh_arm_seed({"300": {"hold_share_pooled": 0.185}}),
        }
        assert gates.evaluate_gates(unsep_arm, gens=["300"], protocol="fresh", calibration=cal)["gates"]["G1"]["passed"] is True

        small_diff_arm = {
            "s1": _fresh_arm_seed({"300": {"hold_share_pooled": 0.17}}),
            "s2": _fresh_arm_seed({"300": {"hold_share_pooled": 0.175}}),
        }
        res_small = gates.evaluate_gates(small_diff_arm, gens=["300"], protocol="fresh", calibration=cal)
        assert res_small["gates"]["G1"]["passed"] is True
        assert res_small["gates"]["G1"]["metrics"]["hold_share_pooled"]["mean_diff"] == pytest.approx(-0.0225, abs=1e-9)

        # Worse = higher: anchor {0.28, 0.30, 0.29, 0.31}, delta 0.05
        inv_fail = {
            "s1": _fresh_arm_seed({"300": {"role_inversion_pooled": 0.40}}),
            "s2": _fresh_arm_seed({"300": {"role_inversion_pooled": 0.42}}),
        }
        res_inv_fail = gates.evaluate_gates(inv_fail, gens=["300"], protocol="fresh", calibration=cal)
        assert res_inv_fail["gates"]["G4"]["passed"] is False
        assert res_inv_fail["gates"]["G4"]["metrics"]["role_inversion_pooled"]["mean_diff"] == pytest.approx(0.115, abs=1e-9)

        inv_pass = {
            "s1": _fresh_arm_seed({"300": {"role_inversion_pooled": 0.32}}),
            "s2": _fresh_arm_seed({"300": {"role_inversion_pooled": 0.33}}),
        }
        res_inv_pass = gates.evaluate_gates(inv_pass, gens=["300"], protocol="fresh", calibration=cal)
        assert res_inv_pass["gates"]["G4"]["passed"] is True
        assert res_inv_pass["gates"]["G4"]["metrics"]["role_inversion_pooled"]["mean_diff"] == pytest.approx(0.030, abs=1e-9)

        # Worse = higher, not separated: anchor {0.28, 0.30, 0.29, 0.31}, arm {0.29, 0.60} -> diff 0.15 >= 0.05, PASS
        inv_unsep = {
            "s1": _fresh_arm_seed({"300": {"role_inversion_pooled": 0.29}}),
            "s2": _fresh_arm_seed({"300": {"role_inversion_pooled": 0.60}}),
        }
        res_inv_unsep = gates.evaluate_gates(inv_unsep, gens=["300"], protocol="fresh", calibration=cal)
        assert res_inv_unsep["gates"]["G4"]["passed"] is True
        assert res_inv_unsep["gates"]["G4"]["metrics"]["role_inversion_pooled"]["mean_diff"] == pytest.approx(0.15, abs=1e-9)

    @pytest.mark.parametrize(
        "metric,inside_val,outside_val",
        [
            ("hold_share_pooled", 0.05, 0.049),
            ("holder_spell_s_pooled", 0.15, 0.149),
            ("gk_in_box_A", 0.80, 0.799),
            ("gk_in_box_B", 0.80, 0.799),
            ("gk_goal_line_A", 0.60, 0.601),
            ("gk_goal_line_B", 0.60, 0.601),
            ("n_within_100_pooled", 1.00, 1.001),
            ("cluster_ge4_pooled", 0.30, 0.301),
            ("role_inversion_pooled", 0.55, 0.551),
            ("goals_total", 0.40, 0.399),
            ("goals_total", 5.00, 5.001),
            ("scoreless", 0.70, 0.701),
        ],
    )
    def test_catastrophe_floors_all_twelve_bounds(self, metric, inside_val, outside_val):
        """Parametrised C0 check covering all 12 bounds: inside/at-bound passes, outside fails."""
        from tools.audit import gates

        cal = _make_fake_calibration()
        inside_arm = {"s42": _fresh_arm_seed({"300": {metric: inside_val}})}
        res_in = gates.evaluate_gates(inside_arm, gens=["300"], protocol="fresh", calibration=cal)
        assert res_in["gates"]["C0"]["passed"] is True
        assert res_in["gates"]["C0"]["violations"] == []

        outside_arm = {"s42": _fresh_arm_seed({"300": {metric: outside_val}})}
        res_out = gates.evaluate_gates(outside_arm, gens=["300"], protocol="fresh", calibration=cal)
        assert res_out["gates"]["C0"]["passed"] is False
        assert len(res_out["gates"]["C0"]["violations"]) == 1
        assert res_out["gates"]["C0"]["violations"][0]["metric"] == metric

    def test_catastrophe_floors_gen_filtering_and_strict_bounds(self):
        """[RED] C0 checks gens >= 200 per seed with strict inequality bounds and per-keeper checks."""
        from tools.audit import gates

        cal = _make_fake_calibration()

        # hold share [0.02, 0.15, 0.16] over [100, 200, 300] -> gen 100 ignored -> PASS
        pass_100 = {"s42": _fresh_arm_seed({
            "100": {"hold_share_pooled": 0.02},
            "200": {"hold_share_pooled": 0.15},
            "300": {"hold_share_pooled": 0.16},
        })}
        assert gates.evaluate_gates(pass_100, gens=["100", "200", "300"], protocol="fresh", calibration=cal)["gates"]["C0"]["passed"] is True

        # [0.15, 0.04, 0.16] -> FAIL at 200
        fail_200 = {"s42": _fresh_arm_seed({
            "100": {"hold_share_pooled": 0.15},
            "200": {"hold_share_pooled": 0.04},
            "300": {"hold_share_pooled": 0.16},
        })}
        res_200 = gates.evaluate_gates(fail_200, gens=["100", "200", "300"], protocol="fresh", calibration=cal)
        assert res_200["gates"]["C0"]["passed"] is False
        assert res_200["gates"]["C0"]["violations"][0]["seed"] == "s42"
        assert res_200["gates"]["C0"]["violations"][0]["gen"] == "200"
        assert res_200["gates"]["C0"]["violations"][0]["value"] == pytest.approx(0.04)

        # gk_goal_line_B 0.61 at 300 -> FAIL
        fail_gk = {"s42": _fresh_arm_seed({"300": {"gk_goal_line_B": 0.61}})}
        assert gates.evaluate_gates(fail_gk, gens=["300"], protocol="fresh", calibration=cal)["gates"]["C0"]["passed"] is False

        # goals_total 5.1 -> FAIL; exact 0.40 goals_total and exact 0.05 hold_share -> PASS
        fail_goals = {"s42": _fresh_arm_seed({"300": {"goals_total": 5.1}})}
        assert gates.evaluate_gates(fail_goals, gens=["300"], protocol="fresh", calibration=cal)["gates"]["C0"]["passed"] is False

        exact_bounds = {"s42": _fresh_arm_seed({"300": {"goals_total": 0.40, "hold_share_pooled": 0.05}})}
        assert gates.evaluate_gates(exact_bounds, gens=["300"], protocol="fresh", calibration=cal)["gates"]["C0"]["passed"] is True

    def test_fresh_protocol_validation_errors(self):
        """[RED] Last gen != calibration gen or missing calibration raises ValueError."""
        from tools.audit import gates

        cal = _make_fake_calibration(gen="300")
        arm = {"s42": _fresh_arm_seed({"200": {}})}
        with pytest.raises(ValueError, match="calibration"):
            gates.evaluate_gates(arm, gens=["200"], protocol="fresh", calibration=None)
        with pytest.raises(ValueError, match="300"):
            gates.evaluate_gates(arm, gens=["200"], protocol="fresh", calibration=cal)

    def test_fresh_protocol_targets_cli_and_markdown_header(self, tmp_path, capsys):
        """[RED] T1-T3 against anchor or --comparator, explicit --target after T1-T3, header names protocol/path/gen."""
        from tools.audit import gates

        cal = _make_fake_calibration(gen="300")
        cal_path = tmp_path / "cal.json"
        cal_path.write_text(json.dumps(cal), encoding="utf-8")

        # Anchor T1 is {-19, -18, -20, -17}, mean -18.5, delta 2.582.
        # Arm T1 = (-12 + -14)/2 = -13 and (-13 + -15)/2 = -14 -> mean -13.5, beats anchor (+5.0 >= 2.582)
        for seed, bb, br in (("42", -12.0, -14.0), ("43", -13.0, -15.0)):
            _write_e3_anchor_dir(
                tmp_path / "arm" / seed,
                "300",
                self_overrides={"gk_in_box_pooled": (0.99, 0.002), "gk_goal_line_pooled": (0.175, 0.01), "kick_outcome_teammate": (0.31, 0.01)},
                bot_blue_gd=bb,
                bot_red_gd=br,
            )

        rc = gates.main([
            "--arm", f"42={tmp_path}/arm/42", f"43={tmp_path}/arm/43",
            "--protocol", "fresh", "--calibration", str(cal_path),
            "--gens", "300", "--target", "goals_total@self_sampled:higher:0.1",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        first_line = out.splitlines()[0]
        assert "protocol=fresh" in first_line
        assert str(cal_path) in first_line
        assert "300" in first_line
        assert "T1" in out and "BEATS" in out
        assert out.index("T3") < out.index("goals_total@self_sampled")
        assert "Between-seed SD at gen 300" in out

        # When --comparator is given, T1-T3 use --comparator while G1-G5 still use anchor.
        # Comparator hold_share_pooled=0.30 so G1 would FAIL if wrongly compared against comparator.
        for seed, bb, br in (("91", -12.0, -14.0), ("92", -13.0, -15.0)):
            _write_e3_anchor_dir(
                tmp_path / "comp" / seed,
                "300",
                self_overrides={"hold_share_pooled": 0.30, "gk_in_box_pooled": (0.99, 0.002), "gk_goal_line_pooled": (0.175, 0.01), "kick_outcome_teammate": (0.31, 0.01)},
                bot_blue_gd=bb,
                bot_red_gd=br,
            )
        rc2 = gates.main([
            "--arm", f"42={tmp_path}/arm/42", f"43={tmp_path}/arm/43",
            "--comparator", f"91={tmp_path}/comp/91", f"92={tmp_path}/comp/92",
            "--protocol", "fresh", "--calibration", str(cal_path),
            "--gens", "300",
        ])
        assert rc2 == 0
        out2 = capsys.readouterr().out
        # Same T1 as comparator -> T1 row verdict is NEUTRAL, and G1 still compares vs anchor (PASS)
        t1_rows = [ln for ln in out2.splitlines() if ln.startswith("| T1 (")]
        assert len(t1_rows) == 1
        t1_cells = [c.strip() for c in t1_rows[0].split("|")[1:-1]]
        assert t1_cells[7] == "NEUTRAL"
        assert "91=-13.000" in t1_cells[5] and "92=-14.000" in t1_cells[5]
        assert "G1 | ball holding (relative) | PASS" in out2

    def test_fresh_protocol_fails_closed_on_missing_data(self):
        """[RED] Fresh protocol raises ValueError naming seed, gen, tag, and metric on missing data."""
        from tools.audit import gates

        cal = _make_fake_calibration(gen="300")

        # 1a. Missing generation in C0
        arm_missing_gen = {
            "s42": _fresh_arm_seed({"300": {}}),
            "s43": _fresh_arm_seed({"200": {}, "300": {}}),
        }
        with pytest.raises(ValueError) as exc_gen:
            gates.evaluate_gates(arm_missing_gen, gens=["200", "300"], protocol="fresh", calibration=cal)
        msg_gen = str(exc_gen.value)
        assert "s42" in msg_gen and "200" in msg_gen and "self_sampled" in msg_gen and "hold_share_pooled" in msg_gen

        # 1b. Missing tag in C0
        arm_missing_c0_tag = {"s42": _fresh_arm_seed({"200": {}, "300": {}})}
        del arm_missing_c0_tag["s42"]["200"]["self_sampled"]
        with pytest.raises(ValueError) as exc_c0_tag:
            gates.evaluate_gates(arm_missing_c0_tag, gens=["200", "300"], protocol="fresh", calibration=cal)
        msg_c0_tag = str(exc_c0_tag.value)
        assert "s42" in msg_c0_tag and "200" in msg_c0_tag and "self_sampled" in msg_c0_tag and "hold_share_pooled" in msg_c0_tag

        # 1c. None value in C0
        arm_none_c0 = {"s42": _fresh_arm_seed({"200": {"gk_in_box_A": (None, 0.0)}, "300": {}})}
        with pytest.raises(ValueError) as exc_none_c0:
            gates.evaluate_gates(arm_none_c0, gens=["200", "300"], protocol="fresh", calibration=cal)
        msg_none_c0 = str(exc_none_c0.value)
        assert "s42" in msg_none_c0 and "200" in msg_none_c0 and "self_sampled" in msg_none_c0 and "gk_in_box_A" in msg_none_c0

        # 2. Missing arm value for a guard metric at last_gen
        arm_missing_guard = {"s42": _fresh_arm_seed({"300": {"gk_in_box_pooled": (None, 0.0)}})}
        with pytest.raises(ValueError) as exc_guard:
            gates.evaluate_gates(arm_missing_guard, gens=["300"], protocol="fresh", calibration=cal)
        msg_guard = str(exc_guard.value)
        assert "s42" in msg_guard and "300" in msg_guard and "self_sampled" in msg_guard and "gk_in_box_pooled" in msg_guard

        # 3a. Missing tag for T1-T3 on arm
        arm_missing_t_tag = {
            "s42": _fresh_arm_seed({"300": {}}),
            "s43": _fresh_arm_seed({"300": {}}),
        }
        del arm_missing_t_tag["s43"]["300"]["vs_scripted_sampled_red"]
        with pytest.raises(ValueError) as exc_t_arm:
            gates.evaluate_gates(arm_missing_t_tag, gens=["300"], protocol="fresh", calibration=cal)
        msg_t_arm = str(exc_t_arm.value)
        assert "s43" in msg_t_arm and "300" in msg_t_arm and "vs_scripted_sampled_red" in msg_t_arm and "goal_diff" in msg_t_arm

        # 3b. Missing tag for T1-T3 on comparator
        arm_ok = {"s42": _fresh_arm_seed({"300": {}}), "s43": _fresh_arm_seed({"300": {}})}
        comp_missing_t_tag = {"c91": _fresh_arm_seed({"300": {}}), "c92": _fresh_arm_seed({"300": {}})}
        del comp_missing_t_tag["c91"]["300"]["vs_frozen"]
        with pytest.raises(ValueError) as exc_t_comp:
            gates.evaluate_gates(arm_ok, comparator=comp_missing_t_tag, gens=["300"], protocol="fresh", calibration=cal)
        msg_t_comp = str(exc_t_comp.value)
        assert "c91" in msg_t_comp and "300" in msg_t_comp and "vs_frozen" in msg_t_comp and "goal_diff" in msg_t_comp

    def test_explicit_target_on_an_absent_tag_is_neutral_not_an_error(self):
        """[GUARD] An explicit `--target` whose tag no arm seed has is reported NEUTRAL with a note (no seed has a value),
        while the gates themselves fail closed on missing data (see `test_fresh_protocol_fails_closed_on_missing_data`)."""
        from tools.audit import gates

        arm = {s: _fresh_arm_seed({"300": {}}) for s in ("s42", "s43")}
        res = gates.evaluate_gates(arm, gens=["300"], protocol="fresh", calibration=_make_fake_calibration(),
                                   targets=["goal_diff@absent_tag:higher:0.5"])
        target = res["targets"][-1]
        assert target["key"] == "goal_diff" and target["tag"] == "absent_tag"
        assert target["verdict"] == "NEUTRAL"
        assert target["mean_diff"] is None
        assert "fewer than 2 seeds (arm=0, comparator=0)" in target["note"]

    def test_fresh_protocol_rejects_mismatched_calibration_constants(self, tmp_path):
        """[RED] Loading/evaluating calibration rejects guard key, worse direction, or target tag mismatch."""
        from tools.audit import gates

        arm = {"s42": _fresh_arm_seed({"300": {}})}

        # Mismatched guard worse direction
        bad_worse = _make_fake_calibration(guard_overrides={"hold_share_pooled": {"worse": "higher"}})
        with pytest.raises(ValueError, match="hold_share_pooled"):
            gates.evaluate_gates(arm, gens=["300"], protocol="fresh", calibration=bad_worse)

        # Mismatched guard metric keys
        bad_keys = _make_fake_calibration()
        del bad_keys["guards"]["scoreless"]
        bad_keys["guards"]["extra_metric"] = {"gate": "G5", "tag": "self_sampled", "worse": "higher", "floor": 0.05, "per_seed": {"42": 0.1}, "mean": 0.1, "sd": 0.0, "delta": 0.05}
        with pytest.raises(ValueError, match="scoreless|guards"):
            gates.evaluate_gates(arm, gens=["300"], protocol="fresh", calibration=bad_keys)

        # Mismatched target tags via CLI --calibration
        bad_tags = _make_fake_calibration(target_overrides={"T1": {"tags": ["vs_scripted_sampled"]}})
        cal_path = tmp_path / "bad_cal.json"
        cal_path.write_text(json.dumps(bad_tags), encoding="utf-8")
        _write_e3_anchor_dir(
            tmp_path / "arm" / "42",
            "300",
            self_overrides={"gk_in_box_pooled": (0.99, 0.002), "gk_goal_line_pooled": (0.175, 0.01), "kick_outcome_teammate": (0.31, 0.01)},
        )
        with pytest.raises(ValueError, match="T1"):
            gates.main(["--arm", f"42={tmp_path}/arm/42", "--protocol", "fresh", "--calibration", str(cal_path), "--gens", "300"])

    def test_gates_docstring_mentions_protocol_and_calibration(self):
        """[RED] gates.py module docstring documents --protocol and --calibration."""
        from tools.audit import gates

        assert gates.__doc__ is not None
        assert "--protocol" in gates.__doc__
        assert "--calibration" in gates.__doc__


class TestEvaluateArmE3:
    def test_red_tags_match_direct_swap_harness_and_report_red_goals_A(self, tmp_path):
        """[RED] vs_scripted_sampled_red matches swap=True array-for-array and reports red goals_A."""
        from tools.audit import census, evaluate_arm, harness

        ckpt_dir = tmp_path / "ckpt" / "gen0300"
        ckpt_dir.mkdir(parents=True)
        torch.manual_seed(7)
        for prefix in ("blue", "red"):
            for kind in ("field", "keeper"):
                ac = ActorCritic()
                # Distinguish blue vs red deterministically via output bias
                with torch.no_grad():
                    ac.actor_mean[-2].bias.fill_(1.0 if prefix == "red" else -1.0)
                torch.save(ac.state_dict(), ckpt_dir / f"{prefix}_{kind}_policy.pt")

        frozen_dir = write_checkpoint(tmp_path / "frozen" / "gen1050", seed=9)
        out_dir = tmp_path / "eval_out"
        evaluate_arm.evaluate_arm(
            [ckpt_dir],
            out_dir,
            workers=1,
            frozen_dir=frozen_dir,
            tags=["vs_scripted_sampled_red", "vs_frozen_red"],
            n_override=2,
            max_steps=30,
        )

        # Direct harness runs with swap=True
        direct_bot = harness.run_matches(
            harness.EvalSpec(
                checkpoint_dir=str(ckpt_dir), opponent="scripted", mode="sampled",
                n=2, seed=13, swap=True, workers=1, max_steps=30,
            )
        )
        direct_straight = harness.run_matches(
            harness.EvalSpec(
                checkpoint_dir=str(ckpt_dir), opponent="scripted", mode="sampled",
                n=2, seed=13, swap=False, workers=1, max_steps=30,
            )
        )
        assert not np.array_equal(direct_bot[0]["mean"][:, :6], direct_straight[0]["mean"][:, :6])

        red_traces = census.load_traces(out_dir / "gen0300" / "vs_scripted_sampled_red")
        assert len(red_traces) == 2
        for got, exp in zip(red_traces, direct_bot):
            assert set(got) == set(exp)
            for k in exp:
                assert np.array_equal(got[k], exp[k]), k

        c_red = json.loads((out_dir / "gen0300" / "vs_scripted_sampled_red" / "census.json").read_text(encoding="utf-8"))
        exp_goals_a = census.summarize([census.match_metrics(t) for t in direct_bot])["metrics"]["goals_A"]["mean"]
        assert c_red["summary"]["metrics"]["goals_A"]["mean"] == pytest.approx(exp_goals_a)

        direct_fr = harness.run_matches(
            harness.EvalSpec(
                checkpoint_dir=str(ckpt_dir), opponent=f"frozen:{frozen_dir}", mode="sampled",
                n=2, seed=15, swap=True, workers=1, max_steps=30,
            )
        )
        fr_traces = census.load_traces(out_dir / "gen0300" / "vs_frozen_red")
        for got, exp in zip(fr_traces, direct_fr):
            for k in exp:
                assert np.array_equal(got[k], exp[k]), k

    def test_adding_tags_leaves_existing_tags_byte_identical_and_recensus_reproduces_bytes(self, tmp_path, monkeypatch):
        """[RED] Adding tags preserves existing tag files; --recensus reproduces JSONs without running harness."""
        from tools.audit import evaluate_arm, harness

        snap = write_checkpoint(tmp_path / "snaps" / "gen0300", seed=12)
        out_dir = tmp_path / "eval_out"
        evaluate_arm.evaluate_arm(
            [snap], out_dir, workers=1, tags=["self_sampled"], n_override=2, max_steps=20,
        )
        before_self = dir_digest(out_dir / "gen0300" / "self_sampled")
        census_bytes = (out_dir / "gen0300" / "self_sampled" / "census.json").read_bytes()
        internals_bytes = (out_dir / "gen0300" / "self_sampled" / "internals.json").read_bytes()

        # Adding another tag leaves self_sampled byte-identical
        evaluate_arm.evaluate_arm(
            [snap], out_dir, workers=1, tags=["vs_scripted_sampled_red"], n_override=2, max_steps=20,
        )
        assert dir_digest(out_dir / "gen0300" / "self_sampled") == before_self
        assert (out_dir / "gen0300" / "vs_scripted_sampled_red" / "census.json").is_file()

        # Corrupt JSONs, block harness completely, and run --recensus
        (out_dir / "gen0300" / "self_sampled" / "census.json").write_text("{}", encoding="utf-8")
        (out_dir / "gen0300" / "self_sampled" / "internals.json").write_text("{}", encoding="utf-8")

        def forbid_harness(*args, **kwargs):
            raise AssertionError("harness must not run during --recensus")

        monkeypatch.setattr(harness, "iter_matches", forbid_harness)
        monkeypatch.setattr(harness, "Session", forbid_harness)
        rc = evaluate_arm.main([
            "--snapshots", str(tmp_path / "nonexistent" / "gen0300"),
            "--out", str(out_dir),
            "--recensus",
            "--tags", "self_sampled", "vs_scripted_sampled_red",
        ])
        assert rc == 0
        assert (out_dir / "gen0300" / "self_sampled" / "census.json").read_bytes() == census_bytes
        assert (out_dir / "gen0300" / "self_sampled" / "internals.json").read_bytes() == internals_bytes

        # Missing traces in a tag directory errors naming it
        empty_tag = out_dir / "gen0300" / "self_mixed"
        empty_tag.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="self_mixed"):
            evaluate_arm.main([
                "--snapshots", str(tmp_path / "nonexistent" / "gen0300"),
                "--out", str(out_dir),
                "--recensus",
                "--tags", "self_mixed",
            ])

    def test_red_goal_attribution_differs_between_swap_true_and_false(self, tmp_path, monkeypatch):
        """When only the red pair scores, goals_A is positive with swap=True and 0 with swap=False."""
        from tools.audit import evaluate_arm, harness

        ckpt_dir = tmp_path / "ckpt" / "gen0300"
        ckpt_dir.mkdir(parents=True)
        for prefix in ("blue", "red"):
            for kind in ("field", "keeper"):
                ac = ActorCritic()
                with torch.no_grad():
                    for p in ac.parameters():
                        p.zero_()
                    # Distinguishable blue vs red weights; bias[0] drives Team A throttle when kept as TeamPolicy
                    ac.actor_mean[-2].bias.fill_(1.0 if prefix == "red" else -1.0)
                torch.save(ac.state_dict(), ckpt_dir / f"{prefix}_{kind}_policy.pt")

        frozen_dir = tmp_path / "frozen" / "gen1050"
        frozen_dir.mkdir(parents=True)
        for prefix in ("blue", "red"):
            for kind in ("field", "keeper"):
                ac = ActorCritic()
                with torch.no_grad():
                    for p in ac.parameters():
                        p.zero_()
                torch.save(ac.state_dict(), frozen_dir / f"{prefix}_{kind}_policy.pt")

        orig_load = harness._load_policies

        def load_scoring_red(spec):
            team_a, team_b = orig_load(spec)
            # When Team A loaded the red policy (bias[0] > 0), drive Team A with scripted_actions (scores in <=160 steps)
            if team_a is not None and float(team_a.heads[0][1][-1][1][0]) > 0.0:
                return None, team_b
            return team_a, team_b

        monkeypatch.setattr(harness, "_load_policies", load_scoring_red)
        out_dir = tmp_path / "eval_out"
        evaluate_arm.evaluate_arm(
            [ckpt_dir],
            out_dir,
            workers=1,
            frozen_dir=frozen_dir,
            tags=["vs_frozen", "vs_frozen_red"],
            n_override=1,
            max_steps=160,
        )
        c_blue = json.loads((out_dir / "gen0300" / "vs_frozen" / "census.json").read_text(encoding="utf-8"))
        c_red = json.loads((out_dir / "gen0300" / "vs_frozen_red" / "census.json").read_text(encoding="utf-8"))
        goals_blue = c_blue["summary"]["metrics"]["goals_A"]["mean"]
        goals_red = c_red["summary"]["metrics"]["goals_A"]["mean"]
        assert goals_blue == pytest.approx(0.0)
        assert goals_red >= 1.0
        assert goals_red != goals_blue

    def test_recensus_multitag_computes_all_before_writing(self, tmp_path):
        """[RED] A multi-tag --recensus failure on a later tag leaves earlier tags untouched."""
        from tools.audit import evaluate_arm

        snap = write_checkpoint(tmp_path / "snaps" / "gen0300", seed=12)
        out_dir = tmp_path / "eval_out"
        evaluate_arm.evaluate_arm(
            [snap], out_dir, workers=1, tags=["self_sampled"], n_override=2, max_steps=20,
        )
        sentinel = '{"sentinel": "keep_me"}\n'
        (out_dir / "gen0300" / "self_sampled" / "census.json").write_text(sentinel, encoding="utf-8")
        (out_dir / "gen0300" / "self_sampled" / "internals.json").write_text(sentinel, encoding="utf-8")
        (out_dir / "gen0300" / "self_sampled" / "report.txt").write_text(sentinel, encoding="utf-8")

        empty_tag = out_dir / "gen0300" / "vs_scripted_sampled_red"
        empty_tag.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="vs_scripted_sampled_red"):
            evaluate_arm.main([
                "--snapshots", str(tmp_path / "nonexistent" / "gen0300"),
                "--out", str(out_dir),
                "--recensus",
                "--tags", "self_sampled", "vs_scripted_sampled_red",
            ])
        assert (out_dir / "gen0300" / "self_sampled" / "census.json").read_text(encoding="utf-8") == sentinel
        assert (out_dir / "gen0300" / "self_sampled" / "internals.json").read_text(encoding="utf-8") == sentinel
        assert (out_dir / "gen0300" / "self_sampled" / "report.txt").read_text(encoding="utf-8") == sentinel




# ----------------------------------------------------------------------------
# Fresh gates: catastrophe floors, learning floor, takeoff, primary target, CLI flags
# ----------------------------------------------------------------------------
FIRE_ROLES = ("blue_field", "blue_keeper", "red_field", "red_keeper")


def _arm_seed_t2(t2_by_gen: dict, **kw) -> dict:
    """`_fresh_arm_seed` whose frozen-opponent goal_diff differs per generation: {gen: (blue, red)}."""
    out = _fresh_arm_seed({g: {} for g in t2_by_gen}, **kw)
    for g, (blue, red) in t2_by_gen.items():
        out[g]["vs_frozen"]["summary"]["metrics"]["goal_diff"]["mean"] = blue
        out[g]["vs_frozen_red"]["summary"]["metrics"]["goal_diff"]["mean"] = red
    return out


def _goal_history(goals_by_gen: dict, **match_extra) -> dict:
    """Training history with one match per entry of `goals_by_gen[gen]` (a list of goal counts)."""
    matches = []
    for gen in sorted(goals_by_gen):
        for goals in goals_by_gen[gen]:
            matches.append({"match_id": len(matches), "generation": gen, "goals": goals, **match_extra})
    return {"matches": matches}


def _flat_history(first: int, last: int, goals: float, per_gen: int = 8) -> dict:
    return _goal_history({g: [goals] * per_gen for g in range(first, last + 1)})


def _takeoff_history(takeoff_gen, last: int = 300) -> dict:
    """Goals 1.5 per match from generation `takeoff_gen - 24` on (0.0 before); never 1.5 for None."""
    start = None if takeoff_gen is None else takeoff_gen - 24
    return _goal_history({g: [1.5 if start is not None and g >= start else 0.0] for g in range(1, last + 1)})


def _hybrid_history(n_gens: int) -> dict:
    """Linear-in-generation fire telemetry for four roles (role index i = 0..3)."""
    hist = _goal_history({g: [1.0] for g in range(1, n_gens + 1)})
    hist["generations"] = [
        {
            "generation": g,
            "fire_sat": {r: {"kick": 0.10 * i + 0.02 * g, "spec": 0.10 * i + 0.50 + 0.02 * g} for i, r in enumerate(FIRE_ROLES)},
            "p_kick": {r: 0.20 + 0.10 * i + 0.01 * g for i, r in enumerate(FIRE_ROLES)},
            "p_spec": {r: 0.60 - 0.10 * i - 0.01 * g for i, r in enumerate(FIRE_ROLES)},
        }
        for g in range(1, n_gens + 1)
    ]
    return hist


def _write_history(path: Path, hist: dict) -> str:
    path.write_text(json.dumps(hist), encoding="utf-8")
    return str(path)


_FRESH_SELF_BASE = {"gk_in_box_pooled": (0.99, 0.002), "gk_goal_line_pooled": (0.175, 0.01), "kick_outcome_teammate": (0.31, 0.01)}


def _fresh_cli_argv(tmp_path: Path, t2_by_gen: dict, *, self_by_gen: dict | None = None, takeoffs: dict | None = None) -> list:
    """Arguments for `gates.main` over four on-disk arm seeds (42-45), a calibration file and optional histories."""
    cal_path = tmp_path / "cal.json"
    cal_path.write_text(json.dumps(_make_fake_calibration()), encoding="utf-8")
    argv = ["--arm"]
    for seed in ("42", "43", "44", "45"):
        root = tmp_path / "arm" / seed
        for gen, (blue, red) in t2_by_gen.items():
            _write_e3_anchor_dir(root, gen, self_overrides={**_FRESH_SELF_BASE, **(self_by_gen or {}).get(gen, {})}, frozen_blue_gd=blue, frozen_red_gd=red)
        argv.append(f"{seed}={root}")
    argv += ["--protocol", "fresh", "--calibration", str(cal_path), "--gens", *t2_by_gen]
    if takeoffs is not None:
        argv.append("--history")
        for seed, t in takeoffs.items():
            argv.append(f"{seed}={_write_history(tmp_path / f'h{seed}.json', _takeoff_history(t))}")
    return argv


class TestCatastropheFloors:
    def test_registered_rules_pin_three_changed_rows(self):
        """[RED] The eleven catastrophe rows carry exactly the registered rule text."""
        from tools.audit import gates

        assert [(m, r) for m, r, _ in gates.CATASTROPHE_SPECS] == [
            ("hold_share_pooled", "< 0.05"),
            ("holder_spell_s_pooled", "< 0.15"),
            ("gk_in_box_A", "< 0.80"),
            ("gk_in_box_B", "< 0.80"),
            ("gk_goal_line_A", "> 0.60"),
            ("gk_goal_line_B", "> 0.60"),
            ("n_within_100_pooled", "> 1.00"),
            ("cluster_ge4_pooled", "> 0.30"),
            ("role_inversion_pooled", "> 0.55"),
            ("goals_total", "< 0.40 or > 5.0"),
            ("scoreless", "> 0.70"),
        ]

    def test_baseline_run_that_trips_older_floors_passes(self):
        """[RED] The late-takeoff anchor seed (gen 300: hold 0.0920, goals 0.938, scoreless 0.427) is not a catastrophe."""
        from tools.audit import gates

        cal = _make_fake_calibration()
        arm = {"43": _fresh_arm_seed({
            "100": {"hold_share_pooled": 0.069, "goals_total": 0.531, "scoreless": 0.573},
            "200": {"hold_share_pooled": 0.097, "goals_total": 0.792, "scoreless": 0.448},
            "300": {"hold_share_pooled": 0.0920, "goals_total": 0.938, "scoreless": 0.427},
        })}
        res = gates.evaluate_gates(arm, gens=["100", "200", "300"], protocol="fresh", calibration=cal)
        assert res["gates"]["C0"]["passed"] is True
        assert res["gates"]["C0"]["violations"] == []

    @pytest.mark.parametrize("metric,value", [("hold_share_pooled", 0.049), ("goals_total", 0.399), ("scoreless", 0.701)])
    def test_values_below_the_registered_floors_fail_at_gen_200(self, metric, value):
        """[GUARD] Each changed row still fails just outside its registered bound, at gen 200."""
        from tools.audit import gates

        arm = {"42": _fresh_arm_seed({"200": {metric: value}, "300": {}})}
        res = gates.evaluate_gates(arm, gens=["200", "300"], protocol="fresh", calibration=_make_fake_calibration())
        assert [(v["gen"], v["metric"]) for v in res["gates"]["C0"]["violations"]] == [("200", metric)]


class TestLearningFloor:
    def _run(self, t100, t300, **kw):
        from tools.audit import gates

        arm = {"42": _arm_seed_t2({"100": t100, "300": t300})}
        return gates.evaluate_gates(arm, gens=["100", "300"], protocol="fresh", calibration=_make_fake_calibration(), **kw)["gates"]["C0L"]

    def test_constant_is_registered_value(self):
        from tools.audit import gates

        assert gates.LEARNING_FLOOR_GAIN == 0.50

    def test_gain_above_floor_passes_and_reports_hand_values(self):
        """[RED] T2(100) = -4.0, T2(300) = (-3.5 + -3.4)/2 = -3.45: gain 0.55 -> PASS."""
        c0l = self._run((-4.0, -4.0), (-3.5, -3.4))
        assert c0l["passed"] is True
        assert set(c0l) >= {"passed", "rule", "per_seed"}
        assert set(c0l["per_seed"]["42"]) == {"t_first", "t_last", "gain", "passed"}
        s = c0l["per_seed"]["42"]
        assert s["t_first"] == pytest.approx(-4.0)
        assert s["t_last"] == pytest.approx(-3.45)
        assert s["gain"] == pytest.approx(0.55)
        assert s["passed"] is True

    def test_gain_below_floor_fails(self):
        """[RED] T2(300) = (-3.6 + -3.5)/2 = -3.55: gain 0.45 -> FAIL."""
        c0l = self._run((-4.0, -4.0), (-3.6, -3.5))
        assert c0l["passed"] is False
        assert c0l["per_seed"]["42"]["gain"] == pytest.approx(0.45)
        assert c0l["per_seed"]["42"]["passed"] is False

    def test_gain_exactly_at_floor_passes(self):
        """[RED] The bound is strict: gain exactly 0.50 passes."""
        c0l = self._run((-4.0, -4.0), (-3.5, -3.5))
        assert c0l["per_seed"]["42"]["gain"] == 0.5
        assert c0l["passed"] is True

    def test_each_frozen_tag_contributes_equally(self):
        """[RED] Only the red pair improves: (-4, -4) -> (-4, -3) is a gain of 0.5, not 1.0 and not 0.0."""
        c0l = self._run((-4.0, -4.0), (-4.0, -3.0))
        assert c0l["per_seed"]["42"]["gain"] == 0.5

    def test_one_failing_seed_fails_the_floor(self):
        """[RED] Seeds are judged one by one; the gate fails if any seed fails."""
        from tools.audit import gates

        arm = {
            "42": _arm_seed_t2({"100": (-4.0, -4.0), "300": (-1.0, -1.0)}),
            "43": _arm_seed_t2({"100": (-4.0, -4.0), "300": (-3.9, -3.9)}),
        }
        c0l = gates.evaluate_gates(arm, gens=["100", "300"], protocol="fresh", calibration=_make_fake_calibration())["gates"]["C0L"]
        assert c0l["passed"] is False
        assert c0l["per_seed"]["42"]["passed"] is True and c0l["per_seed"]["43"]["passed"] is False
        assert c0l["per_seed"]["42"]["gain"] == pytest.approx(3.0)

    @pytest.mark.parametrize("gens", [["200", "300"], ["300"]])
    def test_not_evaluated_without_gen_100(self, gens):
        """[RED] Without generation 100 in --gens the floor is not evaluated (passed None) and says why."""
        from tools.audit import gates

        arm = {"42": _arm_seed_t2({g: (-1.0, -1.0) for g in gens})}
        c0l = gates.evaluate_gates(arm, gens=gens, protocol="fresh", calibration=_make_fake_calibration())["gates"]["C0L"]
        assert c0l["passed"] is None
        assert "not evaluated: gen 100 not in --gens" in c0l["note"]

    def test_missing_red_tag_at_gen_100_fails_closed(self):
        """[RED] A missing vs_frozen_red at gen 100 raises ValueError naming role, seed, gen, tag and metric."""
        from tools.audit import gates

        seed = _arm_seed_t2({"100": (-4.0, -4.0), "300": (-1.0, -1.0)})
        del seed["100"]["vs_frozen_red"]
        with pytest.raises(ValueError) as exc:
            gates.evaluate_gates({"42": seed}, gens=["100", "300"], protocol="fresh", calibration=_make_fake_calibration())
        msg = str(exc.value)
        assert "arm" in msg and "'42'" in msg and "'100'" in msg and "vs_frozen_red" in msg and "goal_diff" in msg

    @pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
    @pytest.mark.parametrize("gen", ["100", "300"])
    def test_none_or_non_finite_value_fails_closed(self, bad, gen):
        """[RED] None, NaN or Inf in a T2 term at gen 100 or at the last generation raises ValueError naming it."""
        from tools.audit import gates

        seed = _arm_seed_t2({"100": (-4.0, -4.0), "300": (-1.0, -1.0)})
        seed[gen]["vs_frozen"]["summary"]["metrics"]["goal_diff"]["mean"] = bad
        with pytest.raises(ValueError) as exc:
            gates.evaluate_gates({"42": seed}, gens=["100", "300"], protocol="fresh", calibration=_make_fake_calibration())
        msg = str(exc.value)
        assert "'42'" in msg and f"'{gen}'" in msg and "vs_frozen" in msg and "goal_diff" in msg

    def test_comparator_is_not_checked(self):
        """[RED] A comparator with zero gain, and one with no gen 100 at all, does not change C0L."""
        comp_flat = {"91": _arm_seed_t2({"100": (-1.0, -1.0), "300": (-1.0, -1.0)}), "92": _arm_seed_t2({"100": (-1.0, -1.0), "300": (-1.0, -1.0)})}
        c0l = self._run((-4.0, -4.0), (-3.0, -3.0), comparator=comp_flat)
        assert c0l["passed"] is True and c0l["per_seed"]["42"]["gain"] == pytest.approx(1.0)
        comp_no_100 = {"91": _arm_seed_t2({"300": (-1.0, -1.0)}), "92": _arm_seed_t2({"300": (-1.0, -1.0)})}
        assert self._run((-4.0, -4.0), (-3.0, -3.0), comparator=comp_no_100)["passed"] is True

    def test_markdown_row_and_section(self):
        """[RED] The summary table has a C0L row; the section lists seed, T2 at 100, T2 at last, gain and status."""
        from tools.audit import gates

        arm = {"42": _arm_seed_t2({"100": (-4.0, -4.0), "300": (-3.5, -3.4)}), "43": _arm_seed_t2({"100": (-4.0, -4.0), "300": (-3.6, -3.5)})}
        md = gates.format_markdown(gates.evaluate_gates(arm, gens=["100", "300"], protocol="fresh", calibration=_make_fake_calibration()))
        lines = md.splitlines()
        assert any(ln.startswith("| C0L |") and ln.endswith("| FAIL |") for ln in lines)
        assert "| 42 | -4.000 | -3.450 | +0.550 | PASS |" in lines
        assert "| 43 | -4.000 | -3.550 | +0.450 | FAIL |" in lines



class TestTakeoffGeneration:
    def test_step_history_hand_value(self):
        """[RED] 1.0 for gens 1-30, 2.0 for 31-60: gen 42 averages 1.48, gen 43 averages 1.52 -> 43."""
        from tools.audit import training_progress

        hist = _goal_history({g: [1.0 if g <= 30 else 2.0] * 8 for g in range(1, 61)})
        assert training_progress.takeoff_generation(hist) == 43
        cut = _goal_history({g: [1.0 if g <= 30 else 2.0] * 8 for g in range(1, 43)})
        assert training_progress.takeoff_generation(cut) is None

    def test_threshold_is_inclusive_and_earliest_is_first_plus_window_minus_one(self):
        """[RED] A constant 1.5 reaches the threshold exactly at gen 25."""
        from tools.audit import training_progress

        assert training_progress.takeoff_generation(_flat_history(1, 60, 1.5)) == 25

    def test_just_below_threshold_never_qualifies(self):
        """[RED] A constant 1.49 never reaches 1.5."""
        from tools.audit import training_progress

        assert training_progress.takeoff_generation(_flat_history(1, 60, 1.49)) is None

    def test_history_shorter_than_window_is_none_and_exactly_window_qualifies(self):
        """[RED] 24 generations of 3.0 give no full window; 25 give takeoff at 25."""
        from tools.audit import training_progress

        assert training_progress.takeoff_generation(_flat_history(1, 24, 3.0)) is None
        assert training_progress.takeoff_generation(_flat_history(1, 25, 3.0)) == 25

    def test_missing_generation_is_an_error_naming_it(self):
        """[RED] Every generation from first to last needs a match."""
        from tools.audit import training_progress

        hist = _goal_history({g: [3.0] for g in range(1, 61) if g != 10})
        with pytest.raises(ValueError, match=r"\b10\b"):
            training_progress.takeoff_generation(hist)

    def test_window_starts_at_first_generation_in_history(self):
        """[RED] A history starting at gen 101 with all 3.0 takes off at 125; a hole at 150 is named."""
        from tools.audit import training_progress

        assert training_progress.takeoff_generation(_flat_history(101, 160, 3.0)) == 125
        holed = _goal_history({g: [3.0] for g in range(101, 161) if g != 150})
        with pytest.raises(ValueError, match=r"\b150\b"):
            training_progress.takeoff_generation(holed)

    def test_per_generation_mean_is_taken_before_the_window_mean(self):
        """[RED] Window 2: gen 1 = [0,0,0,3] (mean 0.75), gen 2 = [3] (mean 3.0) -> 1.875 >= 1.5 -> 2. Pooled over matches it would be 6/5 = 1.2."""
        from tools.audit import training_progress

        hist = _goal_history({1: [0, 0, 0, 3], 2: [3]})
        assert training_progress.takeoff_generation(hist, goals=1.5, window=2) == 2

    def test_order_of_matches_does_not_matter(self):
        """[RED] Matches are grouped by their generation field."""
        from tools.audit import training_progress

        hist = _goal_history({g: [1.0 if g <= 30 else 2.0] * 8 for g in range(1, 61)})
        hist["matches"] = hist["matches"][::-1]
        assert training_progress.takeoff_generation(hist) == 43

    def test_window_of_one_and_custom_threshold(self):
        """[RED] Window 1 is the first generation whose own mean reaches the threshold."""
        from tools.audit import training_progress

        hist = _goal_history({1: [0], 2: [1, 2], 3: [4]})
        assert training_progress.takeoff_generation(hist, goals=1.5, window=1) == 2
        assert training_progress.takeoff_generation(hist, goals=1.51, window=1) == 3

    @pytest.mark.parametrize("hist", [{"matches": []}, {}, {"matches": [{"generation": 1}]}, {"matches": [{"generation": 1, "goals": float("nan")}]}])
    def test_degenerate_histories_are_errors(self, hist):
        """[RED] No matches, a match without goals, or a non-finite goal count raises ValueError."""
        from tools.audit import training_progress

        with pytest.raises(ValueError):
            training_progress.takeoff_generation(hist)

    @pytest.mark.parametrize("kwargs", [{"window": 0}, {"window": -3}, {"goals": float("nan")}])
    def test_invalid_parameters_are_errors(self, kwargs):
        """[RED] A non-positive window or a non-finite threshold raises ValueError."""
        from tools.audit import training_progress

        with pytest.raises(ValueError):
            training_progress.takeoff_generation(_flat_history(1, 30, 3.0), **kwargs)

    def test_cli_prints_takeoff_line_and_json(self, tmp_path, capsys):
        """[RED] --takeoff prints the registered line; --json gains the takeoff block only with the flag."""
        from tools.audit import training_progress

        h = _write_history(tmp_path / "h.json", _goal_history({g: [1.0 if g <= 30 else 2.0] * 8 for g in range(1, 61)}))
        out_json = tmp_path / "o.json"
        assert training_progress.main(["--history", h, "--bucket", "30", "--takeoff", "--json", str(out_json)]) == 0
        assert "takeoff_gen=43 (trailing 25-gen mean training goals >= 1.5)" in capsys.readouterr().out.splitlines()
        assert json.loads(out_json.read_text(encoding="utf-8"))["takeoff"] == {"gen": 43, "goals": 1.5, "window": 25, "first_gen": 1, "last_gen": 60}

        assert training_progress.main(["--history", h, "--bucket", "30", "--json", str(out_json)]) == 0
        assert "takeoff" not in json.loads(out_json.read_text(encoding="utf-8"))
        assert "takeoff_gen" not in capsys.readouterr().out

    def test_cli_options_and_none_result(self, tmp_path, capsys):
        """[RED] --takeoff-goals / --takeoff-window apply; no takeoff prints `none` with the last generation."""
        from tools.audit import training_progress

        h = _write_history(tmp_path / "h.json", _goal_history({g: [1.0 if g <= 30 else 2.0] * 8 for g in range(1, 61)}))
        out_json = tmp_path / "o.json"
        training_progress.main(["--history", h, "--takeoff", "--takeoff-goals", "2.0", "--takeoff-window", "5", "--json", str(out_json)])
        assert "takeoff_gen=35 (trailing 5-gen mean training goals >= 2.0)" in capsys.readouterr().out.splitlines()
        assert json.loads(out_json.read_text(encoding="utf-8"))["takeoff"]["gen"] == 35

        training_progress.main(["--history", h, "--takeoff", "--takeoff-goals", "2.5", "--json", str(out_json)])
        assert "takeoff_gen=none (not reached by gen 60)" in capsys.readouterr().out.splitlines()
        assert json.loads(out_json.read_text(encoding="utf-8"))["takeoff"] == {"gen": None, "goals": 2.5, "window": 25, "first_gen": 1, "last_gen": 60}


class TestTrainingProgressOutput:
    GUARD_HISTORY = {"matches": [
        {"generation": 1, "interest_score": 10.0, "goals": 1, "completed_passes": 2, "shots": [1, 1], "successful_tackles": 3,
         "keeper_saves": 0, "true_possession_pct": [10.0, 10.0, 80.0], "stationary_ball_frames": 100},
        {"generation": 1, "interest_score": 20.0, "goals": 3, "completed_passes": 4, "shots": [2, 0], "successful_tackles": 5,
         "keeper_saves": 2, "true_possession_pct": [10.0, 10.0, 90.0], "stationary_ball_frames": 300},
        {"generation": 2, "interest_score": 60.0, "goals": 2, "completed_passes": 0, "shots": [0, 0], "successful_tackles": 1,
         "keeper_saves": 1, "true_possession_pct": [10.0, 10.0, 70.0], "stationary_ball_frames": 200},
        {"generation": 3, "interest_score": 5.0, "goals": 0, "completed_passes": 1, "shots": [1, 0], "successful_tackles": 0,
         "keeper_saves": 0, "true_possession_pct": [10.0, 10.0, 50.0], "stationary_ball_frames": 0},
    ]}
    GUARD_STDOUT = (
        "training_progress: 4 matches in 2 bucket(s) of 2 gens\n"
        "  gens     1-    2 n=   3 interest= 30.0 (med  20.0 max  60.0) goals=2.00 passes=2.0 shots=1.3 tackles=3.0 saves=1.00 loose%=80.0 stat=200\n"
        "  gens     3-    3 n=   1 interest=  5.0 (med   5.0 max   5.0) goals=0.00 passes=1.0 shots=1.0 tackles=0.0 saves=0.00 loose%=50.0 stat=0\n"
        "  newest 2 generation(s):\n"
        "  gens     2-    2 n=   1 interest= 60.0 (med  60.0 max  60.0) goals=2.00 passes=0.0 shots=0.0 tackles=1.0 saves=1.00 loose%=70.0 stat=200\n"
        "  gens     3-    3 n=   1 interest=  5.0 (med   5.0 max   5.0) goals=0.00 passes=1.0 shots=1.0 tackles=0.0 saves=0.00 loose%=50.0 stat=0\n"
    )

    def test_output_without_takeoff_or_fire_is_unchanged(self, tmp_path, capsys):
        """[GUARD] Plain stdout is the literal below (taken from the tool before --takeoff/--fire existed; hand values: interest mean 30, shots 4/3, ...) and the JSON keeps its keys."""
        from tools.audit import training_progress

        h = _write_history(tmp_path / "h.json", self.GUARD_HISTORY)
        out_json = tmp_path / "o.json"
        assert training_progress.main(["--history", h, "--bucket", "2", "--last-gens", "2", "--json", str(out_json)]) == 0
        assert capsys.readouterr().out == self.GUARD_STDOUT
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert set(data) == {"buckets", "last_generations"}
        assert all("fire" not in b and "takeoff" not in b for b in data["buckets"])
        assert data["buckets"][0]["shots"] == pytest.approx(4.0 / 3.0)

    def test_fire_and_takeoff_append_after_the_plain_output(self, tmp_path, capsys):
        """[RED] With both flags the plain lines come first and unchanged."""
        from tools.audit import training_progress

        h = _write_history(tmp_path / "h.json", self.GUARD_HISTORY)
        assert training_progress.main(["--history", h, "--bucket", "2", "--last-gens", "2", "--takeoff", "--fire"]) == 0
        out = capsys.readouterr().out
        assert out.startswith(self.GUARD_STDOUT)
        assert len(out) > len(self.GUARD_STDOUT)


class TestFireSummary:
    def _run(self, tmp_path, capsys, hist, *extra):
        from tools.audit import training_progress

        h = _write_history(tmp_path / "h.json", hist)
        out_json = tmp_path / "o.json"
        rc = training_progress.main(["--history", h, "--bucket", "2", "--fire", "--json", str(out_json), *extra])
        return rc, capsys.readouterr().out, json.loads(out_json.read_text(encoding="utf-8"))

    def test_bucket_means_of_hybrid_telemetry(self, tmp_path, capsys):
        """[RED] Two buckets of two generations: per-role means of fire_sat kick/spec, p_kick and p_spec (hand values)."""
        rc, out, data = self._run(tmp_path, capsys, _hybrid_history(4))
        assert rc == 0
        # generations {1,2}: base + 0.03 (kick/spec), p_kick base + 0.015, p_spec base - 0.015; generations {3,4}: 0.07 / 0.035
        expected = [
            {"kick": [0.03, 0.13, 0.23, 0.33], "spec": [0.53, 0.63, 0.73, 0.83], "p_kick": [0.215, 0.315, 0.415, 0.515], "p_spec": [0.585, 0.485, 0.385, 0.285]},
            {"kick": [0.07, 0.17, 0.27, 0.37], "spec": [0.57, 0.67, 0.77, 0.87], "p_kick": [0.235, 0.335, 0.435, 0.535], "p_spec": [0.565, 0.465, 0.365, 0.265]},
        ]
        assert len(data["buckets"]) == 2
        for bucket, exp in zip(data["buckets"], expected):
            fire = bucket["fire"]
            assert fire["n"] == 2
            for i, role in enumerate(FIRE_ROLES):
                assert fire["fire_sat"][role]["kick"] == pytest.approx(exp["kick"][i], abs=1e-12)
                assert fire["fire_sat"][role]["spec"] == pytest.approx(exp["spec"][i], abs=1e-12)
                assert fire["p_kick"][role] == pytest.approx(exp["p_kick"][i], abs=1e-12)
                assert fire["p_spec"][role] == pytest.approx(exp["p_spec"][i], abs=1e-12)
        lines = out.splitlines()
        assert "  fire gens     1-    2 (n=2)" in lines
        assert "    fire_sat.kick  A/red field=0.030 A/red keeper=0.130 B/blue field=0.230 B/blue keeper=0.330" in lines
        assert "    fire_sat.spec  A/red field=0.530 A/red keeper=0.630 B/blue field=0.730 B/blue keeper=0.830" in lines
        assert "    p_kick         A/red field=0.215 A/red keeper=0.315 B/blue field=0.415 B/blue keeper=0.515" in lines
        assert "    p_spec         A/red field=0.585 A/red keeper=0.485 B/blue field=0.385 B/blue keeper=0.285" in lines
        assert "  fire gens     3-    4 (n=2)" in lines
        assert "    fire_sat.kick  A/red field=0.070 A/red keeper=0.170 B/blue field=0.270 B/blue keeper=0.370" in lines
        assert "fire: n/a (no hybrid5 telemetry)" not in out

    def test_gaussian_history_prints_not_available_and_exits_zero(self, tmp_path, capsys):
        """[RED] No fire_sat in the generation records -> the n/a line, exit 0, null fire blocks."""
        hist = _hybrid_history(4)
        for rec in hist["generations"]:
            for k in ("fire_sat", "p_kick", "p_spec"):
                del rec[k]
        rc, out, data = self._run(tmp_path, capsys, hist)
        assert rc == 0
        assert "fire: n/a (no hybrid5 telemetry)" in out.splitlines()
        assert [b["fire"] for b in data["buckets"]] == [None, None]

    def test_history_without_generation_records_is_not_available(self, tmp_path, capsys):
        """[RED] An old history with no generation list prints the n/a line."""
        hist = _hybrid_history(4)
        del hist["generations"]
        rc, out, data = self._run(tmp_path, capsys, hist)
        assert rc == 0 and "fire: n/a (no hybrid5 telemetry)" in out.splitlines()

    def test_incomplete_record_is_an_error_naming_the_generation(self, tmp_path):
        """[RED] A record with fire_sat but a role missing raises ValueError naming the generation."""
        from tools.audit import training_progress

        hist = _hybrid_history(4)
        del hist["generations"][2]["fire_sat"]["red_field"]
        with pytest.raises(ValueError, match=r"generation 3\b"):
            training_progress.fire_progress(hist, bucket=2)

    def test_bucket_without_records_between_buckets_with_records(self, tmp_path, capsys):
        """[RED] Records only for generations 3-4: the first bucket prints n/a, the second its means."""
        hist = _hybrid_history(4)
        hist["generations"] = hist["generations"][2:]
        rc, out, data = self._run(tmp_path, capsys, hist)
        assert data["buckets"][0]["fire"] is None
        assert data["buckets"][1]["fire"]["n"] == 2
        assert "  fire gens     1-    2: n/a" in out.splitlines()


    def test_fire_lines_label_the_teams_by_drawn_colour(self, tmp_path, capsys):
        """[RED] Team A (stored `blue_*`, seats 0-5) is drawn red and team B (stored `red_*`) blue, so the fire lines print
        `A/red field`, `A/red keeper`, `B/blue field`, `B/blue keeper`. Two generations, one bucket, hand values:
        fire_sat.kick of role i is 0.10*i + 0.02*g averaged over g = 1, 2, i.e. 0.03, 0.13, 0.23, 0.33; the JSON keys stay the stored names."""
        rc, out, data = self._run(tmp_path, capsys, _hybrid_history(2))
        assert rc == 0
        lines = out.splitlines()
        assert "    fire_sat.kick  A/red field=0.030 A/red keeper=0.130 B/blue field=0.230 B/blue keeper=0.330" in lines
        assert "    p_kick         A/red field=0.215 A/red keeper=0.315 B/blue field=0.415 B/blue keeper=0.515" in lines
        assert not any(name in out for name in FIRE_ROLES)
        fire = data["buckets"][0]["fire"]
        assert set(fire["p_kick"]) == set(FIRE_ROLES)
        assert fire["fire_sat"]["blue_field"]["kick"] == pytest.approx(0.03, abs=1e-12)
        assert fire["fire_sat"]["red_keeper"]["kick"] == pytest.approx(0.33, abs=1e-12)


class TestGatesTakeoff:
    @staticmethod
    def _g6(takeoffs: dict, **kw):
        from tools.audit import gates

        arm = {s: _fresh_arm_seed({"300": {}}) for s in takeoffs}
        hist = {s: _takeoff_history(t) for s, t in takeoffs.items()}
        return gates.evaluate_gates(arm, gens=["300"], protocol="fresh", calibration=_make_fake_calibration(), histories=hist, **kw)

    def test_constant_is_registered_value(self):
        from tools.audit import gates

        assert gates.TAKEOFF_FAIL_SHARE == 0.5

    def test_anchor_like_split_passes(self):
        """[RED] Takeoff gens {37, none, 242, 37}: one of four without takeoff -> PASS."""
        res = self._g6({"42": 37, "43": None, "44": 242, "45": 37})
        g6 = res["gates"]["G6"]
        assert g6["passed"] is True
        assert g6["per_seed"] == {"42": 37, "43": None, "44": 242, "45": 37}

    def test_three_of_four_without_takeoff_fails(self):
        """[RED] {none, none, none, 37}: share 0.75 > 0.5 -> FAIL."""
        assert self._g6({"42": None, "43": None, "44": None, "45": 37})["gates"]["G6"]["passed"] is False

    def test_exactly_half_without_takeoff_passes(self):
        """[RED] {none, none, 37, 37}: share 0.5 is not above 0.5 -> PASS."""
        assert self._g6({"42": None, "43": None, "44": 37, "45": 37})["gates"]["G6"]["passed"] is True

    def test_missing_history_gives_none_and_names_the_seeds(self):
        """[RED] A seed without a history makes G6 not evaluated, and the note names it."""
        from tools.audit import gates

        arm = {s: _fresh_arm_seed({"300": {}}) for s in ("42", "43", "44", "45")}
        hist = {s: _takeoff_history(37) for s in ("42", "43", "44")}
        g6 = gates.evaluate_gates(arm, gens=["300"], protocol="fresh", calibration=_make_fake_calibration(), histories=hist)["gates"]["G6"]
        assert g6["passed"] is None and "45" in g6["note"]
        no_hist = gates.evaluate_gates(arm, gens=["300"], protocol="fresh", calibration=_make_fake_calibration())["gates"]["G6"]
        assert no_hist["passed"] is None and all(s in no_hist["note"] for s in arm)

    def test_history_for_a_non_arm_seed_is_an_error(self):
        """[RED] A history seed that is not an arm seed raises ValueError naming it."""
        from tools.audit import gates

        arm = {"42": _fresh_arm_seed({"300": {}})}
        hist = {"42": _takeoff_history(37), "99": _takeoff_history(37)}
        with pytest.raises(ValueError, match="99"):
            gates.evaluate_gates(arm, gens=["300"], protocol="fresh", calibration=_make_fake_calibration(), histories=hist)

    def test_markdown_takeoff_table(self):
        """[RED] The Takeoff section lists seed, takeoff generation (none when absent) and the history range."""
        from tools.audit import gates

        md = gates.format_markdown(self._g6({"42": 37, "43": None, "44": 242, "45": 37}))
        lines = md.splitlines()
        assert any(ln.startswith("| G6 |") and ln.endswith("| PASS |") for ln in lines)
        assert "| 42 | 37 | 1-300 |" in lines
        assert "| 43 | none | 1-300 |" in lines
        assert "| 44 | 242 | 1-300 |" in lines

    def test_cli_history_flag_end_to_end(self, tmp_path, capsys):
        """[RED] --history SEED=PATH feeds G6 through the CLI and the JSON report."""
        from tools.audit import gates

        argv = _fresh_cli_argv(tmp_path, {"300": (-1.0, -1.0)}, takeoffs={"42": 37, "43": None, "44": 242, "45": 37})
        out_json = tmp_path / "g.json"
        assert gates.main([*argv, "--json", str(out_json)]) == 0
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert data["gates"]["G6"]["passed"] is True
        assert data["gates"]["G6"]["per_seed"] == {"42": 37, "43": None, "44": 242, "45": 37}
        assert "| 43 | none | 1-300 |" in capsys.readouterr().out.splitlines()

    def test_cli_history_flag_rejects_malformed_and_duplicate_items(self, tmp_path):
        """[RED] Items must be SEED=PATH, and a seed may be given once."""
        from tools.audit import gates

        argv = _fresh_cli_argv(tmp_path, {"300": (-1.0, -1.0)})
        h = _write_history(tmp_path / "h.json", _takeoff_history(37))
        with pytest.raises(ValueError, match="SEED=PATH"):
            gates.main([*argv, "--history", h])
        with pytest.raises(ValueError, match="42"):
            gates.main([*argv, "--history", f"42={h}", f"42={h}"])


class TestPrimaryTarget:
    def test_constant_labels_and_json(self):
        """[RED] T2 is primary, T1 informational at gen 300, T3 unchanged; JSON names the primary target."""
        from tools.audit import gates

        assert gates.PRIMARY_TARGET == "T2"
        arm = {s: _fresh_arm_seed({"300": {}}) for s in ("42", "43")}
        res = gates.evaluate_gates(arm, gens=["300"], protocol="fresh", calibration=_make_fake_calibration())
        assert res["primary_target"] == "T2"
        rows = {ln.split("|")[1].strip(): ln for ln in gates.format_markdown(res).splitlines() if ln.startswith("| T")}
        assert "T2 (primary)" in rows
        assert "T1 (informational at gen 300)" in rows
        assert any(k.startswith("T3 (") for k in rows)

    def test_target_verdicts_do_not_depend_on_the_labels(self):
        """[GUARD] Arm T1 -13.5 beats the anchor -18.5 (delta 2.582); T2 and T3 are neutral."""
        from tools.audit import gates

        arm = {"42": _fresh_arm_seed({"300": {}}, bot_b=-12.0, bot_r=-14.0), "43": _fresh_arm_seed({"300": {}}, bot_b=-13.0, bot_r=-15.0)}
        res = gates.evaluate_gates(arm, gens=["300"], protocol="fresh", calibration=_make_fake_calibration())
        assert {t["id"]: t["verdict"] for t in res["targets"]} == {"T1": "BEATS", "T2": "NEUTRAL", "T3": "NEUTRAL"}


class TestExitOnFailure:
    FAILING_C0 = {"200": {"hold_share_pooled": 0.04}}

    def test_returns_two_on_catastrophe(self, tmp_path):
        """[RED] A C0 violation returns 2 with the flag and 0 without it."""
        from tools.audit import gates

        t2 = {"100": (-4.0, -4.0), "200": (-2.0, -2.0), "300": (-1.0, -1.0)}
        argv = _fresh_cli_argv(tmp_path, t2, self_by_gen=self.FAILING_C0)
        assert gates.main(argv) == 0
        assert gates.main([*argv, "--exit-on-c0"]) == 2

    def test_returns_two_on_learning_floor_failure(self, tmp_path):
        """[RED] A flat T2 (gain 0.0 < 0.50) returns 2 with the flag and 0 without it."""
        from tools.audit import gates

        argv = _fresh_cli_argv(tmp_path, {"100": (-1.0, -1.0), "300": (-1.0, -1.0)})
        assert gates.main(argv) == 0
        assert gates.main([*argv, "--exit-on-c0"]) == 2

    def test_returns_two_on_takeoff_failure(self, tmp_path):
        """[RED] Three of four seeds without takeoff returns 2 with the flag and 0 without it."""
        from tools.audit import gates

        argv = _fresh_cli_argv(tmp_path, {"100": (-4.0, -4.0), "300": (-1.0, -1.0)}, takeoffs={"42": None, "43": None, "44": None, "45": 37})
        assert gates.main(argv) == 0
        assert gates.main([*argv, "--exit-on-c0"]) == 2

    def test_returns_zero_when_everything_passes(self, tmp_path):
        """[RED] C0, C0L and G6 all pass -> 0 even with the flag."""
        from tools.audit import gates

        argv = _fresh_cli_argv(tmp_path, {"100": (-4.0, -4.0), "200": (-2.0, -2.0), "300": (-1.0, -1.0)}, takeoffs={"42": 37, "43": None, "44": 242, "45": 37})
        assert gates.main([*argv, "--exit-on-c0"]) == 0

    def test_not_evaluated_does_not_trigger(self, tmp_path):
        """[RED] Without gen 100 and without histories C0L and G6 are None; the flag still returns 0."""
        from tools.audit import gates

        argv = _fresh_cli_argv(tmp_path, {"300": (-1.0, -1.0)})
        assert gates.main([*argv, "--exit-on-c0"]) == 0



class TestFreshCliFlags:
    def test_default_protocol_is_fresh(self, tmp_path, capsys):
        """[RED] Without `--protocol` the fresh protocol runs: the report header names it and the C0L, G6 and style sections exist."""
        from tools.audit import gates

        argv = _fresh_cli_argv(tmp_path, {"100": (-4.0, -4.0), "300": (-1.0, -1.0)})
        i = argv.index("--protocol")
        del argv[i:i + 2]
        out_json = tmp_path / "g.json"
        assert gates.main([*argv, "--json", str(out_json)]) == 0
        out = capsys.readouterr().out
        assert "protocol=fresh" in out.splitlines()[0]
        assert "## C0L Learning Floor" in out and "## Takeoff (G6)" in out
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert data["protocol"] == "fresh" and data["primary_target"] == "T2"

    def test_bare_call_without_calibration_raises_the_required_error(self, tmp_path):
        """[RED] `gates` with neither `--protocol` nor `--calibration` raises the fresh protocol's "--calibration is required" error."""
        from tools.audit import gates

        _write_e3_anchor_dir(tmp_path / "arm" / "42", "300", self_overrides=_FRESH_SELF_BASE)
        with pytest.raises(ValueError, match="--calibration is required"):
            gates.main(["--arm", f"42={tmp_path}/arm/42", "--gens", "300"])

    @pytest.mark.parametrize("protocol", ["resume150", "fresh300"])
    def test_retired_protocol_names_are_rejected(self, tmp_path, capsys, protocol):
        """[RED] argparse rejects `--protocol resume150|fresh300` (exit status 2, "invalid choice"), and so does `evaluate_gates`;
        the only protocol left is `fresh`."""
        from tools.audit import gates

        assert gates.PROTOCOLS == ("fresh",)
        _write_e3_anchor_dir(tmp_path / "arm" / "42", "300", self_overrides=_FRESH_SELF_BASE)
        with pytest.raises(SystemExit) as exc:
            gates.main(["--arm", f"42={tmp_path}/arm/42", "--protocol", protocol, "--gens", "300"])
        assert exc.value.code == 2
        assert "invalid choice" in capsys.readouterr().err
        with pytest.raises(ValueError, match="protocol"):
            gates.evaluate_gates({"42": _fresh_arm_seed({"300": {}})}, gens=["300"], protocol=protocol, calibration=_make_fake_calibration())


class TestCalibrationValidation:
    @staticmethod
    def _evaluate(cal, tag="self_sampled"):
        from tools.audit import gates

        arm = {"42": _fresh_arm_seed({"300": {}})}
        if tag != "self_sampled":
            arm["42"]["300"][tag] = arm["42"]["300"]["self_sampled"]
        return gates.evaluate_gates(arm, gens=["300"], tag=tag, protocol="fresh", calibration=cal)

    def test_valid_calibration_is_accepted(self):
        """[GUARD] The fixture calibration passes every check."""
        assert self._evaluate(_make_fake_calibration())["gates"]["G1"]["passed"] is True

    @pytest.mark.parametrize("tid", ["T1", "T2", "T3"])
    def test_rejects_target_direction_other_than_registered(self, tid):
        """[RED] A target registered as `higher` with direction `lower` is rejected."""
        cal = _make_fake_calibration(target_overrides={tid: {"direction": "lower"}})
        with pytest.raises(ValueError, match=rf"{tid}.*direction"):
            self._evaluate(cal)

    @pytest.mark.parametrize("kind,entry,bad", [
        ("guards", "hold_share_pooled", float("nan")),
        ("guards", "scoreless", float("inf")),
        ("targets", "T2", float("nan")),
        ("targets", "T1", float("inf")),
    ])
    def test_rejects_non_finite_delta(self, kind, entry, bad):
        """[RED] A NaN or infinite delta is rejected."""
        cal = _make_fake_calibration()
        cal[kind][entry]["delta"] = bad
        with pytest.raises(ValueError, match=rf"{entry}.*delta must be finite and > 0"):
            self._evaluate(cal)

    @pytest.mark.parametrize("kind,entry,bad", [
        ("guards", "hold_share_pooled", 0.0),
        ("guards", "goals_total", -0.30),
        ("targets", "T2", 0.0),
        ("targets", "T3", -0.03),
    ])
    def test_rejects_non_positive_delta(self, kind, entry, bad):
        """[RED] A delta of 0 or below is rejected."""
        cal = _make_fake_calibration()
        cal[kind][entry]["delta"] = bad
        with pytest.raises(ValueError, match=rf"{entry}.*delta must be finite and > 0"):
            self._evaluate(cal)

    @pytest.mark.parametrize("kind,entry,bad", [
        ("guards", "hold_share_pooled", 0.0099),
        ("guards", "goals_total", 0.29),
        ("targets", "T1", 1.99),
        ("targets", "T2", 0.4999),
    ])
    def test_rejects_delta_below_its_floor(self, kind, entry, bad):
        """[RED] A positive delta just under the entry's own floor is rejected; a delta equal to the floor is accepted."""
        cal = _make_fake_calibration()
        cal[kind][entry]["delta"] = bad
        with pytest.raises(ValueError, match=rf"{entry}.*floor"):
            self._evaluate(cal)
        cal[kind][entry]["delta"] = cal[kind][entry]["floor"]
        self._evaluate(cal)

    @pytest.mark.parametrize("kind,entry", [("guards", "hold_share_pooled"), ("guards", "scoreless"), ("targets", "T1"), ("targets", "T3")])
    def test_rejects_fewer_than_three_anchor_seeds(self, kind, entry):
        """[RED] An entry built from two anchor seeds is rejected; three are accepted."""
        cal = _make_fake_calibration()
        seeds = dict(cal[kind][entry]["per_seed"])
        cal[kind][entry]["per_seed"] = dict(list(seeds.items())[:2])
        with pytest.raises(ValueError, match=rf"{entry}.*anchor seeds"):
            self._evaluate(cal)
        cal[kind][entry]["per_seed"] = dict(list(seeds.items())[:3])
        self._evaluate(cal)

    def test_rejects_guard_tag_that_differs_from_tag_in_use(self):
        """[RED] Guards calibrated on self_sampled cannot be applied to --tag self_mixed; the error names both."""
        with pytest.raises(ValueError) as exc:
            self._evaluate(_make_fake_calibration(), tag="self_mixed")
        assert "self_sampled" in str(exc.value) and "self_mixed" in str(exc.value)

    def test_gates_constants_match_the_calibration_specs(self):
        """[GUARD] gates.py and calibrate.py register the same worse directions, target tags and target directions."""
        from tools.audit import calibrate, gates

        assert gates.FRESH_GUARD_WORSE == {metric: worse for _, metric, worse, _ in calibrate.GUARD_CALIBRATION_SPECS}
        assert gates.FRESH_TARGET_TAGS == {tid: tags for tid, _, tags, _, _ in calibrate.TARGET_CALIBRATION_SPECS}
        assert gates.FRESH_TARGET_DIRECTION == {tid: direction for tid, _, _, direction, _ in calibrate.TARGET_CALIBRATION_SPECS}


class TestRedTagSpecs:
    @pytest.mark.parametrize("red,blue,seed", [("vs_scripted_sampled_red", "vs_scripted_sampled", 13), ("vs_frozen_red", "vs_frozen", 15)])
    def test_red_tag_equals_blue_twin_except_swap(self, red, blue, seed):
        """[RED] Opponent, mode, match count and seed are those of the blue twin; only the seats are swapped."""
        from tools.audit import evaluate_arm

        r, b = evaluate_arm.TAG_SPECS[red], evaluate_arm.TAG_SPECS[blue]
        for key in ("opp", "mode", "n", "seed"):
            assert r[key] == b[key], key
        assert r["n"] == 48 and r["seed"] == seed
        assert r["swap"] is True
        assert not b.get("swap", False)


_POOL_TAG_B = {"frozen_team": "B", "member_gen": 50}
_POOL_TAG_A = {"frozen_team": "A", "member_gen": 100}


def _pool_history(self_goals_by_gen: dict, pool_goals_by_gen: dict) -> dict:
    """History with self-play matches (`pool` null, as the trainer writes it) and pool matches (`pool` set) per generation."""
    matches = []
    for gen in sorted(set(self_goals_by_gen) | set(pool_goals_by_gen)):
        for goals in self_goals_by_gen.get(gen, []):
            matches.append({"match_id": len(matches), "generation": gen, "goals": goals, "pool": None})
        for goals in pool_goals_by_gen.get(gen, []):
            matches.append({"match_id": len(matches), "generation": gen, "goals": goals, "pool": dict(_POOL_TAG_B)})
    return {"matches": matches}


class TestPoolMatchExclusion:
    """Training-progress statistics and the takeoff generation read self-play matches only; pool matches (frozen seats
    from the opponent snapshot pool, match record `pool` set) are counted per bucket and excluded."""

    MIXED_HISTORY = {"matches": [
        {"generation": 1, "interest_score": 10.0, "goals": 1, "completed_passes": 2, "successful_tackles": 3, "pool": None},
        {"generation": 1, "interest_score": 30.0, "goals": 3, "completed_passes": 4, "successful_tackles": 1, "pool": None},
        {"generation": 1, "interest_score": 90.0, "goals": 9, "completed_passes": 20, "successful_tackles": 9, "pool": _POOL_TAG_B},
        {"generation": 2, "interest_score": 20.0, "goals": 2, "completed_passes": 0, "successful_tackles": 2},
        {"generation": 2, "interest_score": 80.0, "goals": 8, "completed_passes": 10, "successful_tackles": 8, "pool": _POOL_TAG_A},
        {"generation": 2, "interest_score": 80.0, "goals": 8, "completed_passes": 10, "successful_tackles": 8, "pool": _POOL_TAG_A},
    ]}

    def test_bucket_means_use_self_play_matches_only(self):
        """[RED] Hand values over the three self-play matches (goals 1, 3, 2; interest 10, 30, 20); pooling all six would
        give goals 31/6 and interest max 90."""
        from tools.audit import training_progress

        (row,) = training_progress.bucket_progress(self.MIXED_HISTORY, bucket=2)
        assert (row["gen_start"], row["gen_end"], row["n"], row["n_pooled"]) == (1, 2, 3, 3)
        assert row["goals"] == pytest.approx(2.0)
        assert row["passes"] == pytest.approx(2.0)
        assert row["tackles"] == pytest.approx(2.0)
        assert (row["interest_mean"], row["interest_median"], row["interest_max"]) == (20.0, 20.0, 30.0)

    def test_per_generation_rows_count_pool_matches_per_generation(self):
        """[RED] gen 1: 2 self-play + 1 pool, mean goals 2.0; gen 2: 1 self-play + 2 pool, goals 2.0."""
        from tools.audit import training_progress

        rows = training_progress.per_generation(self.MIXED_HISTORY, last_gens=2)
        assert [(r["gen_start"], r["n"], r["n_pooled"]) for r in rows] == [(1, 2, 1), (2, 1, 2)]
        assert [r["goals"] for r in rows] == [pytest.approx(2.0), pytest.approx(2.0)]

    def test_history_without_pool_matches_counts_zero(self):
        """[RED] Both encodings of self-play (no `pool` key, `pool: null`) count as self-play."""
        from tools.audit import training_progress

        rows = training_progress.bucket_progress(TestTrainingProgressOutput.GUARD_HISTORY, bucket=2)
        assert [(r["n"], r["n_pooled"]) for r in rows] == [(3, 0), (1, 0)]
        rows = training_progress.bucket_progress(_pool_history({1: [1.0, 2.0]}, {}), bucket=1)
        assert [(r["n"], r["n_pooled"], r["goals"]) for r in rows] == [(2, 0, 1.5)]

    def test_bucket_with_only_pool_matches_has_no_statistics(self):
        """[RED] A bucket whose matches are all pool matches keeps its row: n 0, the pool count, every statistic None."""
        from tools.audit import training_progress

        rows = training_progress.bucket_progress(_pool_history({1: [1.0]}, {2: [5.0, 7.0]}), bucket=1)
        assert [(r["gen_start"], r["gen_end"], r["n"], r["n_pooled"]) for r in rows] == [(1, 1, 1, 0), (2, 2, 0, 2)]
        empty = rows[1]
        for key in ("interest_mean", "interest_median", "interest_max", "goals", "passes", "shots", "tackles", "saves",
                    "loose_pct", "stationary_frames"):
            assert empty[key] is None, key

    def test_stdout_reports_the_excluded_pool_matches(self, tmp_path, capsys):
        """[RED] Header and bucket lines state the pool counts; an all-pool bucket prints its own line."""
        from tools.audit import training_progress

        h = _write_history(tmp_path / "h.json", self.MIXED_HISTORY)
        assert training_progress.main(["--history", h, "--bucket", "2"]) == 0
        assert capsys.readouterr().out == (
            "training_progress: 3 matches in 1 bucket(s) of 2 gens (3 pool matches excluded)\n"
            "  gens     1-    2 n=   3 interest= 20.0 (med  20.0 max  30.0) goals=2.00 passes=2.0 shots=0.0 tackles=2.0 saves=0.00 loose%=- stat=0 pooled=3\n"
        )
        h = _write_history(tmp_path / "h2.json", _pool_history({1: [1.0]}, {2: [5.0]}))
        assert training_progress.main(["--history", h, "--bucket", "1"]) == 0
        out = capsys.readouterr().out
        assert out.startswith("training_progress: 1 matches in 2 bucket(s) of 1 gens (1 pool matches excluded)\n")
        assert out.endswith("  gens     2-    2 n=   0 (no self-play match) pooled=1\n")

    def test_json_rows_carry_the_pool_count(self, tmp_path):
        """[RED] Every JSON bucket row has `n_pooled`; the top-level keys are unchanged."""
        from tools.audit import training_progress

        h = _write_history(tmp_path / "h.json", self.MIXED_HISTORY)
        out_json = tmp_path / "o.json"
        assert training_progress.main(["--history", h, "--bucket", "2", "--last-gens", "2", "--json", str(out_json)]) == 0
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert set(data) == {"buckets", "last_generations"}
        assert [b["n_pooled"] for b in data["buckets"]] == [3]
        assert [b["n_pooled"] for b in data["last_generations"]] == [1, 2]

    def test_pool_matches_cannot_cause_an_early_takeoff(self):
        """[RED] Self-play goals 1.0 (6 per gen) and pool goals 9.0 (2 per gen), gens 1-30: no takeoff. Pooled per-generation
        means would be 3.0 and give takeoff at 25."""
        from tools.audit import training_progress

        hist = _pool_history({g: [1.0] * 6 for g in range(1, 31)}, {g: [9.0] * 2 for g in range(1, 31)})
        assert training_progress.takeoff_summary(hist) == {"gen": None, "goals": 1.5, "window": 25, "first_gen": 1, "last_gen": 30}

    def test_pool_matches_cannot_delay_the_takeoff(self):
        """[RED] Card E4's step history (self-play 1.0 for gens 1-30, 2.0 for 31-60) gives 43 with two 0-goal pool matches per
        generation added. Pooled means (0.75, then 1.5) would give 55."""
        from tools.audit import training_progress

        self_goals = {g: [1.0 if g <= 30 else 2.0] * 6 for g in range(1, 61)}
        hist = _pool_history(self_goals, {g: [0.0, 0.0] for g in range(1, 61)})
        assert training_progress.takeoff_generation(hist) == 43

    def test_generation_with_only_pool_matches_is_an_error(self):
        """[RED] Takeoff needs a self-play match in every generation; a generation with pool matches only is named."""
        from tools.audit import training_progress

        hist = _pool_history({g: [1.0] for g in range(1, 31) if g != 5}, {5: [3.0]})
        with pytest.raises(ValueError, match=r"generation 5 has only pool matches"):
            training_progress.takeoff_generation(hist)
        with pytest.raises(ValueError, match=r"no self-play match"):
            training_progress.takeoff_generation(_pool_history({}, {g: [3.0] for g in range(1, 31)}))

    def test_takeoff_cli_states_the_exclusion(self, tmp_path, capsys):
        """[RED] With pool matches, `--takeoff` prints one extra line with the excluded count; the JSON block keeps its keys."""
        from tools.audit import training_progress

        hist = _pool_history({g: [1.0] * 6 for g in range(1, 31)}, {g: [9.0] * 2 for g in range(1, 31)})
        h = _write_history(tmp_path / "h.json", hist)
        out_json = tmp_path / "o.json"
        assert training_progress.main(["--history", h, "--bucket", "100", "--takeoff", "--json", str(out_json)]) == 0
        out = capsys.readouterr().out
        assert "takeoff_gen=none (not reached by gen 30)\ntakeoff: self-play matches only (60 pool matches excluded)\n" in out
        assert json.loads(out_json.read_text(encoding="utf-8"))["takeoff"] == {
            "gen": None, "goals": 1.5, "window": 25, "first_gen": 1, "last_gen": 30}
        h = _write_history(tmp_path / "h2.json", _flat_history(1, 30, 2.0))
        assert training_progress.main(["--history", h, "--takeoff"]) == 0
        assert "pool matches excluded" not in capsys.readouterr().out

    def test_g6_reads_self_play_takeoff(self):
        """[RED] Four arm seeds whose self-play never takes off but whose pool matches would lift the pooled mean to 3.0:
        G6 sees four seeds without takeoff and fails."""
        from tools.audit import gates

        hist = _pool_history({g: [1.0] * 6 for g in range(1, 61)}, {g: [9.0] * 2 for g in range(1, 61)})
        seeds = ("42", "43", "44", "45")
        arm = {s: _fresh_arm_seed({"300": {}}) for s in seeds}
        res = gates.evaluate_gates(arm, gens=["300"], protocol="fresh", calibration=_make_fake_calibration(),
                                   histories={s: hist for s in seeds})
        g6 = res["gates"]["G6"]
        assert g6["per_seed"] == {s: None for s in seeds}
        assert g6["passed"] is False


# ----------------------------------------------------------------------------
# E5: Fire-behaviour internals (tools/audit/internals.py)
# ----------------------------------------------------------------------------
class TestInternalsFireE5:
    def test_fire_internals_hand_trace_and_aggregation(self, tmp_path, monkeypatch):
        """[RED] Hand-solved 4-decision trace for keeper_special_rate, field_special_rate_far,
        keeper_kick_fire_holding (including zero-eligible None skipping), and cumulative tackle totals.
        """
        from football_rl.config import DEFAULT_CONFIG
        from tools.audit import internals

        # Match 1: 4 decisions (T=4), ball at (100, 420), all players start at (100, 420)
        tr1 = make_internals_trace(4)
        tr1["ball"][:] = (100.0, 420.0)
        tr1["pos"][:] = (100.0, 420.0)

        # Keepers (slots 0 and 6): 8 total keeper decisions.
        # - step 0: holder=0, hold_t=0.04 (<0.08 grace); act[0,0,2]=0.9 (excluded by grace),
        #           act[0,0,3]=0.9 (excluded from keeper_special_rate because holding);
        #           keeper 6 not holding, act[0,6,3]=0.8 (>0.5 -> fire #1).
        # - step 1: holder=0, hold_t=0.10 (>=0.08 grace); act[1,0,2]=0.8 (>0.35 -> kick fire #1),
        #           act[1,0,3]=0.9 (excluded because holding); keeper 6 not holding, act[1,6,3]=-1.0.
        # - step 2: holder=6, hold_t=0.20 (>=0.08 grace); act[2,6,2]=0.35 (not >0.35 -> no kick),
        #           act[2,6,3]=0.9 (excluded because holding); keeper 0 not holding, act[2,0,3]=0.5 (not >0.5).
        # - step 3: holder=6, hold_t=0.25 (>=0.08 grace); act[3,6,2]=0.6 (>0.35 -> kick fire #2),
        #           act[3,6,3]=-1.0 (excluded because holding); keeper 0 not holding, act[3,0,3]=-1.0.
        # Hand values for tr1:
        #   keeper_special_rate: 1 fire (step 0 k6) out of 4 non-holding keeper decisions = 1/4 = 0.25
        #   (if holding states were included: 4 fires out of 8 = 0.50)
        #   keeper_kick_fire_holding: 2 fires (steps 1, 3) out of 3 eligible GK holds (steps 1, 2, 3) = 2/3
        #   (if grace ignored: 3 fires out of 4 = 0.75)
        tr1["holder"][:] = np.array([0, 0, 6, 6], dtype=np.int8)
        tr1["hold_t"][:] = np.array([0.04, 0.10, 0.20, 0.25], dtype=np.float32)
        tr1["act"][0, 0, 2] = 0.9
        tr1["act"][0, 0, 3] = 0.9
        tr1["act"][0, 6, 3] = 0.8
        tr1["act"][1, 0, 2] = 0.8
        tr1["act"][1, 0, 3] = 0.9
        tr1["act"][2, 6, 2] = 0.35
        tr1["act"][2, 6, 3] = 0.9
        tr1["act"][2, 0, 3] = 0.5
        tr1["act"][3, 6, 2] = 0.6

        # Field players (slots 1..5, 7..11): 4 * 10 = 40 field decisions.
        # - step 0 slot 1: dist 179.9 u (not > 180), act=0.9 (near fire)
        # - step 1 slot 2: dist 180.0 u (exact boundary, not > 180), act=0.9 (near fire)
        # - step 2 slot 3: dist 180.1 u (> 180), act=0.9 (far fire #1)
        # - step 3 slot 7: dist 200.0 u (> 180), act=0.7 (far fire #2)
        # - step 3 slot 8: dist 200.0 u (> 180), act=0.5 (not > 0.5, no fire)
        # Hand values for tr1:
        #   special_rate = 4 / 40 = 0.10
        #   field_special_rate_far = 2 / 40 = 0.05 (if >= instead of >: 3 / 40 = 0.075)
        tr1["pos"][0, 1] = (279.9, 420.0)
        tr1["act"][0, 1, 3] = 0.9
        tr1["pos"][1, 2] = (280.0, 420.0)
        tr1["act"][1, 2, 3] = 0.9
        tr1["pos"][2, 3] = (280.1, 420.0)
        tr1["act"][2, 3, 3] = 0.9
        tr1["pos"][3, 7] = (100.0, 620.0)
        tr1["act"][3, 7, 3] = 0.7
        tr1["pos"][3, 8] = (100.0, 620.0)
        tr1["act"][3, 8, 3] = 0.5

        # Cumulative tackle counters: final value is match total (sum would be 9 and 4)
        tr1["s_failed_tackles"] = np.array([1, 1, 3, 4], dtype=np.int32)
        tr1["s_successful_tackles"] = np.array([0, 1, 1, 2], dtype=np.int32)

        m1 = internals.match_internals(tr1)
        assert m1["keeper_special_rate"] == pytest.approx(0.25)
        assert m1["field_special_rate_far"] == pytest.approx(0.05)
        assert m1["special_rate"] == pytest.approx(0.10)
        assert m1["keeper_kick_fire_holding"] == pytest.approx(2.0 / 3.0)
        assert m1["failed_tackles_pm"] == pytest.approx(4.0)
        assert m1["successful_tackles_pm"] == pytest.approx(2.0)

        # Head-agnostic: hybrid5 team flags leave all env-act fire metrics identical
        tr1_hyb = {k: v.copy() for k, v in tr1.items()}
        tr1_hyb["head"] = np.array([1, 1], dtype=np.int8)
        m1_hyb = internals.match_internals(tr1_hyb)
        for key in (
            "keeper_special_rate",
            "field_special_rate_far",
            "keeper_kick_fire_holding",
            "failed_tackles_pm",
            "successful_tackles_pm",
        ):
            assert m1_hyb[key] == pytest.approx(m1[key])

        # Verify config threshold matches DEFAULT_CONFIG.mechanics.tackle_whiff_max_ball_dist
        assert internals.TACKLE_WHIFF_MAX_BALL_DIST == pytest.approx(
            float(DEFAULT_CONFIG.mechanics.tackle_whiff_max_ball_dist)
        )
        monkeypatch.setattr(internals, "TACKLE_WHIFF_MAX_BALL_DIST", 190.0)
        assert internals.match_internals(tr1)["field_special_rate_far"] == pytest.approx(1.0 / 40.0)
        monkeypatch.undo()

        # Match 2: zero eligible keeper holding decisions (holder=0 below grace at step 0, loose/outfield otherwise)
        #   keeper_kick_fire_holding = None (skipped in summarize)
        #   keeper_special_rate = 0 / 7 = 0.0
        #   field_special_rate_far = 0 / 40 = 0.0
        #   failed_tackles_pm = 2.0, successful_tackles_pm = 0.0
        tr2 = make_internals_trace(4)
        tr2["ball"][:] = (100.0, 420.0)
        tr2["pos"][:] = (100.0, 420.0)
        tr2["holder"][:] = np.array([0, -1, 3, -1], dtype=np.int8)
        tr2["hold_t"][:] = np.array([0.05, 0.0, 0.20, 0.0], dtype=np.float32)
        tr2["act"][0, 0, 2] = 0.9
        tr2["s_failed_tackles"] = np.array([0, 2, 2, 2], dtype=np.int32)
        tr2["s_successful_tackles"] = np.array([0, 0, 0, 0], dtype=np.int32)

        m2 = internals.match_internals(tr2)
        assert m2["keeper_kick_fire_holding"] is None
        assert m2["keeper_special_rate"] == pytest.approx(0.0)
        assert m2["field_special_rate_far"] == pytest.approx(0.0)
        assert m2["failed_tackles_pm"] == pytest.approx(2.0)
        assert m2["successful_tackles_pm"] == pytest.approx(0.0)

        # Aggregated across [m1, m2]:
        #   keeper_kick_fire_holding: n=1, mean=2/3, se=None
        #   keeper_special_rate: n=2, mean=(0.25+0)/2=0.125, se=0.125
        #   field_special_rate_far: n=2, mean=(0.05+0)/2=0.025, se=0.025
        #   failed_tackles_pm: n=2, mean=(4+2)/2=3.0, se=1.0
        #   successful_tackles_pm: n=2, mean=(2+0)/2=1.0, se=1.0
        summ = internals.summarize([m1, m2])["metrics"]
        assert summ["keeper_kick_fire_holding"] == {"mean": pytest.approx(2.0 / 3.0), "se": None, "n": 1}
        assert summ["keeper_special_rate"] == {"mean": pytest.approx(0.125), "se": pytest.approx(0.125), "n": 2}
        assert summ["field_special_rate_far"] == {"mean": pytest.approx(0.025), "se": pytest.approx(0.025), "n": 2}
        assert summ["failed_tackles_pm"] == {"mean": pytest.approx(3.0), "se": pytest.approx(1.0), "n": 2}
        assert summ["successful_tackles_pm"] == {"mean": pytest.approx(1.0), "se": pytest.approx(1.0), "n": 2}

        # CLI JSON + report output includes all new keys and preserves existing lines
        d = tmp_path / "traces_e5"
        d.mkdir()
        np.savez_compressed(d / "m000.npz", **tr1)
        np.savez_compressed(d / "m001.npz", **tr2)
        out_json = tmp_path / "internals_e5.json"
        assert internals.main([str(d), "--json", str(out_json)]) == 0
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        for key in (
            "keeper_special_rate",
            "field_special_rate_far",
            "keeper_kick_fire_holding",
            "failed_tackles_pm",
            "successful_tackles_pm",
        ):
            assert key in payload[str(d)]["summary"]["metrics"]

    def test_fire_internals_reads_whiff_dist_from_config(self, monkeypatch):
        """Reloading internals picks up DEFAULT_CONFIG.mechanics.tackle_whiff_max_ball_dist (not hard-coded 180)."""
        import importlib
        from dataclasses import replace
        import football_rl.config as cfg_mod
        from tools.audit import internals

        tr = make_internals_trace(2)
        tr["ball"][:] = (100.0, 420.0)
        tr["pos"][:] = (100.0, 420.0)
        # Step 0 slot 1: dist 160 u (> 150, <= 180) with special fire; step 1 slot 1: dist 190 u (> 180) with fire
        tr["pos"][0, 1] = (260.0, 420.0)
        tr["act"][0, 1, 3] = 0.9
        tr["pos"][1, 1] = (290.0, 420.0)
        tr["act"][1, 1, 3] = 0.9

        assert internals.match_internals(tr)["field_special_rate_far"] == pytest.approx(1.0 / 20.0)

        custom_cfg = replace(
            cfg_mod.DEFAULT_CONFIG,
            mechanics=replace(cfg_mod.DEFAULT_CONFIG.mechanics, tackle_whiff_max_ball_dist=150.0),
        )
        monkeypatch.setattr(cfg_mod, "DEFAULT_CONFIG", custom_cfg)
        try:
            importlib.reload(internals)
            assert internals.TACKLE_WHIFF_MAX_BALL_DIST == pytest.approx(150.0)
            assert internals.match_internals(tr)["field_special_rate_far"] == pytest.approx(2.0 / 20.0)
        finally:
            monkeypatch.undo()
            importlib.reload(internals)

    def test_keeper_kick_fire_holding_grace_boundary(self):
        """Exact float64 pickup_kick_grace (0.08 s) is included (>=), while float32(0.08) (< 0.08) is excluded."""
        from tools.audit import internals

        tr = make_internals_trace(2)
        tr["holder"][:] = np.array([0, 6], dtype=np.int8)
        # Step 0: exact 0.08 in float64 -> eligible (>= grace), does not kick -> 0/1
        # Step 1: 0.10 in float64 -> eligible, kicks -> 1/2 = 0.5 (if > instead of >=, step 0 excluded -> 1/1 = 1.0)
        tr["hold_t"] = np.array([0.08, 0.10], dtype=np.float64)
        tr["act"][0, 0, 2] = 0.0
        tr["act"][1, 6, 2] = 0.8
        assert internals.match_internals(tr)["keeper_kick_fire_holding"] == pytest.approx(0.5)

        # Confirm float32(0.08) rounds below float64(0.08) and is excluded, leaving only step 1 -> 1.0
        tr_f32 = {k: v.copy() for k, v in tr.items()}
        tr_f32["hold_t"] = np.array([np.float32(0.08), np.float32(0.10)], dtype=np.float32)
        assert internals.match_internals(tr_f32)["keeper_kick_fire_holding"] == pytest.approx(1.0)

    def test_existing_internals_keys_unchanged_against_master(self):
        """[GUARD] Every internals key that existed at 89109c0 keeps its exact value on real FL42 and FX42 traces.

        Fixture: the first 450 steps of match 0 of `archives/{fl,fx}_s42_gen300/eval/gen0300/self_sampled`
        (FL: gaussian4 head, FX: hybrid5 head), with only the trace keys `internals` reads, stored in
        `tests/golden/`. `internals_89109c0_values.json` holds the per-match values and the summary that
        `tools/audit/internals.py` at commit 89109c0 produced on exactly those files.
        """
        from tools.audit import internals

        golden_dir = REPO_ROOT / "tests" / "golden"
        frozen = json.loads((golden_dir / "internals_89109c0_values.json").read_text(encoding="utf-8"))
        old_keys = frozen["metric_keys"]
        assert len(old_keys) == 20
        assert set(old_keys).issubset(set(internals.METRIC_KEYS))

        cur_matches = []
        for name in sorted(frozen["matches"]):
            with np.load(golden_dir / name) as z:
                tr = {k: z[k] for k in z.files}
            cur = internals.match_internals(tr)
            cur_matches.append(cur)
            old = frozen["matches"][name]
            for key in old_keys:
                if old[key] is None:
                    assert cur[key] is None, (name, key)
                else:
                    assert cur[key] == pytest.approx(old[key], rel=0.0, abs=0.0), (name, key)
        assert len(cur_matches) == 2

        cur_sum = internals.summarize(cur_matches)["metrics"]
        for key in old_keys:
            old_sum = frozen["summary"][key]
            assert cur_sum[key]["n"] == old_sum["n"], key
            for sub in ("mean", "se"):
                if old_sum[sub] is None:
                    assert cur_sum[key][sub] is None, (key, sub)
                else:
                    assert cur_sum[key][sub] == pytest.approx(old_sum[sub], rel=0.0, abs=0.0), (key, sub)


# ----------------------------------------------------------------------------
# E6: Style targets in gates.py and paired readout tools/audit/run_compare.py
# ----------------------------------------------------------------------------
def _write_e6_eval_dir(
    root: Path,
    gen: str | int,
    *,
    s1: float = 0.20,
    s2: float | None = 0.12,
    s3: float = 0.48,
    s4: float = 0.80,
    write_internals: bool = True,
    omit_internals_key: bool = False,
    omit_census_key: str | None = None,
    t1_blue: float = -10.0,
    t1_red: float = -12.0,
    t2_blue: float = -0.5,
    t2_red: float = -0.7,
    t3: float = 0.30,
    extra_census: dict | None = None,
    extra_internals: dict | None = None,
) -> Path:
    """Write tiny census.json and internals.json files for one seed and generation."""
    gen_dir = root / f"gen{int(gen):04d}"
    census_ov = {
        "hold_share_pooled": (s1, 0.002),
        "kick_outcome_teammate": (t3, 0.004),
        "kick_outcome_opponent": (s3, 0.005),
        "goal_no_pass_10s_share": (s4, 0.04),
        "gk_in_box_pooled": (0.99, 0.002),
        "gk_goal_line_pooled": (0.175, 0.01),
    }
    if extra_census:
        for k, v in extra_census.items():
            census_ov[k] = v if isinstance(v, tuple) else (v, 0.01)
    self_payload = _fake_census_point(**census_ov)
    if omit_census_key is not None:
        self_payload["summary"]["metrics"].pop(omit_census_key, None)

    ss_dir = gen_dir / "self_sampled"
    ss_dir.mkdir(parents=True, exist_ok=True)
    (ss_dir / "census.json").write_text(json.dumps(self_payload), encoding="utf-8")

    if write_internals:
        int_metrics = {
            "power_mean": {"mean": 0.80, "se": 0.003, "n": 96},
            "full_power_share": {"mean": 0.45, "se": 0.005, "n": 96},
            "keeper_special_rate": {"mean": 0.03, "se": 0.001, "n": 96},
            "special_rate": {"mean": 0.04, "se": 0.001, "n": 96},
            "successful_tackles_pm": {"mean": 2.5, "se": 0.15, "n": 96},
            "failed_tackles_pm": {"mean": 12.0, "se": 0.40, "n": 96},
        }
        if not omit_internals_key:
            int_metrics["aim_teammate_share_sampled"] = {"mean": s2, "se": 0.004, "n": 96}
        if extra_internals:
            for k, v in extra_internals.items():
                int_metrics[k] = {"mean": v[0], "se": v[1], "n": 96} if isinstance(v, tuple) else {"mean": v, "se": 0.001, "n": 96}
        (ss_dir / "internals.json").write_text(
            json.dumps({"summary": {"n_matches": 96, "metrics": int_metrics}}),
            encoding="utf-8",
        )

    tag_payloads = {
        "vs_scripted_sampled": {"summary": {"n_matches": 48, "metrics": {"goal_diff": {"mean": t1_blue, "se": 0.4, "n": 48}}}},
        "vs_scripted_sampled_red": {"summary": {"n_matches": 48, "metrics": {"goal_diff": {"mean": t1_red, "se": 0.5, "n": 48}}}},
        "vs_frozen": {"summary": {"n_matches": 48, "metrics": {"goal_diff": {"mean": t2_blue, "se": 0.2, "n": 48}}}},
        "vs_frozen_red": {"summary": {"n_matches": 48, "metrics": {"goal_diff": {"mean": t2_red, "se": 0.3, "n": 48}}}},
    }
    for tag, payload in tag_payloads.items():
        tdir = gen_dir / tag
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / "census.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


class TestStyleTargetsE6:
    def _build_arm(self, base: Path, per_seed: dict, **common) -> list[str]:
        args = []
        for seed in sorted(per_seed):
            d = _write_e6_eval_dir(base / seed, 300, **common, **per_seed[seed])
            args.append(f"{seed}={d}")
        return args

    def test_style_targets_beats_loses_and_report_schema(self, tmp_path, capsys):
        """[RED] S1-S4 compute comparator ddof=1 SD delta, call _ebe_verdict, and emit MD + JSON."""
        from tools.audit import gates

        cal_path = tmp_path / "cal.json"
        cal_path.write_text(json.dumps(_make_fake_calibration(gen="300")), encoding="utf-8")

        # Comparator series: {x, x+0.02, x+0.04, x+0.06} -> deviations {-0.03, -0.01, +0.01, +0.03}
        # sum_sq = 0.0020, s = sqrt(0.0020 / 3) = 0.025819888974716113, 2*s = 0.051639777949432225 (> all floors)
        # S1 (higher): arm [0.18..0.24] mean 0.21 vs comp [0.10..0.16] mean 0.13 -> diff +0.08 -> BEATS
        # S2 (higher): arm [0.10..0.16] mean 0.13 vs comp [0.20..0.26] mean 0.23 -> diff -0.10 -> LOSES
        # S3 (lower):  arm [0.40..0.46] mean 0.43 vs comp [0.50..0.56] mean 0.53 -> diff -0.10 -> BEATS
        # S4 (lower):  arm [0.70..0.76] mean 0.73 vs comp [0.60..0.66] mean 0.63 -> diff +0.10 -> LOSES
        arm_specs = {
            "42": dict(s1=0.18, s2=0.10, s3=0.40, s4=0.70),
            "43": dict(s1=0.20, s2=0.12, s3=0.42, s4=0.72),
            "44": dict(s1=0.22, s2=0.14, s3=0.44, s4=0.74),
            "45": dict(s1=0.24, s2=0.16, s3=0.46, s4=0.76),
        }
        comp_specs = {
            "42": dict(s1=0.10, s2=0.20, s3=0.50, s4=0.60),
            "43": dict(s1=0.12, s2=0.22, s3=0.52, s4=0.62),
            "44": dict(s1=0.14, s2=0.24, s3=0.54, s4=0.64),
            "45": dict(s1=0.16, s2=0.26, s3=0.56, s4=0.66),
        }
        arm_args = self._build_arm(tmp_path / "arm", arm_specs)
        comp_args = self._build_arm(tmp_path / "comp", comp_specs)
        out_json = tmp_path / "out.json"
        out_md = tmp_path / "out.md"

        rc = gates.main([
            "--arm", *arm_args,
            "--comparator", *comp_args,
            "--gens", "300",
            "--protocol", "fresh",
            "--calibration", str(cal_path),
            "--json", str(out_json),
            "--md", str(out_md),
        ])
        assert rc == 0
        md = out_md.read_text(encoding="utf-8")
        stdout = capsys.readouterr().out
        assert md + "\n" == stdout and "## Style Targets" in md
        assert md.index("## Targets") < md.index("## Style Targets") < md.index("## Per-team Informational")

        data = json.loads(out_json.read_text(encoding="utf-8"))
        st = data["style_targets"]
        assert len(st) == 4
        expected_delta = 2.0 * math.sqrt(0.0020 / 3.0)  # 0.051639777949432225
        expected = [
            ("S1", "hold_share_pooled", "higher", 0.01, +0.08, "BEATS"),
            ("S2", "aim_teammate_share_sampled", "higher", 0.02, -0.10, "LOSES"),
            ("S3", "kick_outcome_opponent", "lower", 0.02, -0.10, "BEATS"),
            ("S4", "goal_no_pass_10s_share", "lower", 0.05, +0.10, "LOSES"),
        ]
        for row, (sid, key, direction, floor, mdiff, verdict) in zip(st, expected):
            assert row["id"] == sid
            assert row["key"] == key
            assert row["direction"] == direction
            assert row["floor"] == pytest.approx(floor)
            assert row["delta"] == pytest.approx(expected_delta, abs=1e-9)
            assert row["mean_diff"] == pytest.approx(mdiff, abs=1e-9)
            assert row["verdict"] == verdict
            assert set(row["arm_per_seed"]) == {"42", "43", "44", "45"}
            assert set(row["comp_per_seed"]) == {"42", "43", "44", "45"}

        style_block = md.split("## Style Targets", 1)[1].split("## Per-team Informational", 1)[0]
        for sid, key, direction, _, _, verdict in expected:
            assert any(
                ln.startswith(f"| {sid} |") and key in ln and direction in ln and verdict in ln
                for ln in style_block.splitlines()
            )

    def test_style_targets_overlap_exact_delta_boundary_and_floor_binding(self, tmp_path):
        """[RED] Overlap gives NEUTRAL; diff == delta gives BEATS; floor binds when 2*s < floor; arm SD does not set delta."""
        from tools.audit import gates

        cal_path = tmp_path / "cal.json"
        cal_path.write_text(json.dumps(_make_fake_calibration(gen="300")), encoding="utf-8")

        # S1: comp constant 0.25 -> s=0, floor 0.01 binds -> delta=0.01.
        #     arm [0.26, 0.26, 0.26, 0.26] -> diff = +0.010000000000000009 >= delta -> BEATS
        # S2: comp [0.200, 0.202, 0.204, 0.206] -> s=0.002582, 2*s=0.005164 < floor 0.02 -> delta=0.02.
        #     arm [0.215, 0.215, 0.215, 0.215] -> diff = +0.012 (< 0.02) -> NEUTRAL (even though > 2*s)
        # S3: comp constant 0.50 -> s=0, floor 0.02 binds -> delta=0.02.
        #     arm [0.48, 0.48, 0.48, 0.48] -> diff = -0.02, comp-arm = +0.02 >= delta -> BEATS
        # S4: comp constant 0.60 -> s=0, floor 0.05 binds -> delta=0.05.
        #     arm [0.40, 0.40, 0.40, 0.61] (overlap: max 0.61 > 0.60, diff -0.1475) -> NEUTRAL
        # Also make arm S4 have large variance so using arm SD instead of comp SD changes delta.
        arm_specs = {
            "42": dict(s1=0.26, s2=0.215, s3=0.48, s4=0.40),
            "43": dict(s1=0.26, s2=0.215, s3=0.48, s4=0.40),
            "44": dict(s1=0.26, s2=0.215, s3=0.48, s4=0.40),
            "45": dict(s1=0.26, s2=0.215, s3=0.48, s4=0.61),
        }
        comp_specs = {
            "42": dict(s1=0.25, s2=0.200, s3=0.50, s4=0.60),
            "43": dict(s1=0.25, s2=0.202, s3=0.50, s4=0.60),
            "44": dict(s1=0.25, s2=0.204, s3=0.50, s4=0.60),
            "45": dict(s1=0.25, s2=0.206, s3=0.50, s4=0.60),
        }
        out_json = tmp_path / "boundary.json"
        rc = gates.main([
            "--arm", *self._build_arm(tmp_path / "arm", arm_specs),
            "--comparator", *self._build_arm(tmp_path / "comp", comp_specs),
            "--gens", "300", "--protocol", "fresh", "--calibration", str(cal_path),
            "--json", str(out_json),
        ])
        assert rc == 0
        by_id = {r["id"]: r for r in json.loads(out_json.read_text(encoding="utf-8"))["style_targets"]}
        assert by_id["S1"]["delta"] == pytest.approx(0.01)
        assert by_id["S1"]["verdict"] == "BEATS"
        assert by_id["S2"]["delta"] == pytest.approx(0.02)
        assert by_id["S2"]["verdict"] == "NEUTRAL"
        assert by_id["S3"]["delta"] == pytest.approx(0.02)
        assert by_id["S3"]["verdict"] == "BEATS"
        assert by_id["S4"]["delta"] == pytest.approx(0.05)
        assert by_id["S4"]["verdict"] == "NEUTRAL"

        # Just under exact boundary on S1 (0.2599 -> diff +0.0099 < 0.01) -> NEUTRAL
        arm_just_under = {s: {**v, "s1": 0.2599} for s, v in arm_specs.items()}
        out_json2 = tmp_path / "under.json"
        gates.main([
            "--arm", *self._build_arm(tmp_path / "arm2", arm_just_under),
            "--comparator", *self._build_arm(tmp_path / "comp2", comp_specs),
            "--gens", "300", "--protocol", "fresh", "--calibration", str(cal_path),
            "--json", str(out_json2),
        ])
        by_id2 = {r["id"]: r for r in json.loads(out_json2.read_text(encoding="utf-8"))["style_targets"]}
        assert by_id2["S1"]["verdict"] == "NEUTRAL"

        # Bit-exact equality (no float rounding noise):
        # S2 (higher): comp [0.5625, 0.5625, 0.5625, 0.3125] -> mean 0.5, ddof-1 SD 0.125 -> delta = 0.25;
        #   arm [0.75, 0.75, 0.75, 0.75] -> mean 0.75, mean_diff = 0.75 - 0.5 = 0.25.
        #   repr(mean_diff) == '0.25' == repr(delta), bit-equal (kills >= -> > in _ebe_verdict).
        # S3 (lower): comp [0.4375, 0.4375, 0.4375, 0.6875] -> mean 0.5, ddof-1 SD 0.125 -> delta = 0.25;
        #   arm [0.25, 0.25, 0.25, 0.25] -> mean 0.25, comp_mean - arm_mean = 0.5 - 0.25 = 0.25.
        #   repr(-mean_diff) == '0.25' == repr(delta), bit-equal (kills >= -> > in _ebe_verdict lower branch).
        exact_arm_specs = {
            "42": dict(s1=0.75, s2=0.75, s3=0.25, s4=0.25),
            "43": dict(s1=0.75, s2=0.75, s3=0.25, s4=0.25),
            "44": dict(s1=0.75, s2=0.75, s3=0.25, s4=0.25),
            "45": dict(s1=0.75, s2=0.75, s3=0.25, s4=0.25),
        }
        exact_comp_specs = {
            "42": dict(s1=0.5625, s2=0.5625, s3=0.4375, s4=0.4375),
            "43": dict(s1=0.5625, s2=0.5625, s3=0.4375, s4=0.4375),
            "44": dict(s1=0.5625, s2=0.5625, s3=0.4375, s4=0.4375),
            "45": dict(s1=0.3125, s2=0.3125, s3=0.6875, s4=0.6875),
        }
        out_exact = tmp_path / "exact.json"
        rc_exact = gates.main([
            "--arm", *self._build_arm(tmp_path / "arm_exact", exact_arm_specs),
            "--comparator", *self._build_arm(tmp_path / "comp_exact", exact_comp_specs),
            "--gens", "300", "--protocol", "fresh", "--calibration", str(cal_path),
            "--json", str(out_exact),
        ])
        assert rc_exact == 0
        by_id_exact = {r["id"]: r for r in json.loads(out_exact.read_text(encoding="utf-8"))["style_targets"]}
        assert repr(by_id_exact["S2"]["mean_diff"]) == "0.25"
        assert repr(by_id_exact["S2"]["delta"]) == "0.25"
        assert by_id_exact["S2"]["mean_diff"] == by_id_exact["S2"]["delta"]
        assert by_id_exact["S2"]["verdict"] == "BEATS"
        assert repr(-by_id_exact["S3"]["mean_diff"]) == "0.25"
        assert repr(by_id_exact["S3"]["delta"]) == "0.25"
        assert -by_id_exact["S3"]["mean_diff"] == by_id_exact["S3"]["delta"]
        assert by_id_exact["S3"]["verdict"] == "BEATS"

    def test_style_targets_missing_internals_or_key_fails_closed(self, tmp_path):
        """[RED] Missing internals.json or style key raises ValueError naming seed, gen and path."""
        from tools.audit import gates

        cal_path = tmp_path / "cal.json"
        cal_path.write_text(json.dumps(_make_fake_calibration(gen="300")), encoding="utf-8")

        base_arm = {s: {} for s in ("42", "43", "44", "45")}
        comp_args = self._build_arm(tmp_path / "comp", base_arm)

        # Missing internals.json on arm seed 43
        arm_no_int = self._build_arm(tmp_path / "arm_no_int", {
            "42": {}, "43": {"write_internals": False}, "44": {}, "45": {},
        })
        expected_path = str(tmp_path / "arm_no_int" / "43" / "gen0300" / "self_sampled" / "internals.json")
        with pytest.raises(ValueError) as exc_arm:
            gates.main([
                "--arm", *arm_no_int, "--comparator", *comp_args,
                "--gens", "300", "--protocol", "fresh", "--calibration", str(cal_path),
            ])
        msg_arm = str(exc_arm.value)
        assert "43" in msg_arm and "300" in msg_arm and expected_path in msg_arm

        # Missing internals.json on comparator seed 44
        arm_ok = self._build_arm(tmp_path / "arm_ok", base_arm)
        comp_no_int = self._build_arm(tmp_path / "comp_no_int", {
            "42": {}, "43": {}, "44": {"write_internals": False}, "45": {},
        })
        expected_comp_path = str(tmp_path / "comp_no_int" / "44" / "gen0300" / "self_sampled" / "internals.json")
        with pytest.raises(ValueError) as exc_comp:
            gates.main([
                "--arm", *arm_ok, "--comparator", *comp_no_int,
                "--gens", "300", "--protocol", "fresh", "--calibration", str(cal_path),
            ])
        msg_comp = str(exc_comp.value)
        assert "44" in msg_comp and "300" in msg_comp and expected_comp_path in msg_comp

        # Missing key inside internals.json on arm seed 42
        arm_no_key = self._build_arm(tmp_path / "arm_no_key", {
            "42": {"omit_internals_key": True}, "43": {}, "44": {}, "45": {},
        })
        key_path = str(tmp_path / "arm_no_key" / "42" / "gen0300" / "self_sampled" / "internals.json")
        with pytest.raises(ValueError) as exc_key:
            gates.main([
                "--arm", *arm_no_key, "--comparator", *comp_args,
                "--gens", "300", "--protocol", "fresh", "--calibration", str(cal_path),
            ])
        msg_key = str(exc_key.value)
        assert "42" in msg_key and "300" in msg_key and "aim_teammate_share_sampled" in msg_key and key_path in msg_key

        # Missing census key (S3 kick_outcome_opponent) on comparator seed 45
        comp_no_s3 = self._build_arm(tmp_path / "comp_no_s3", {
            "42": {}, "43": {}, "44": {}, "45": {"omit_census_key": "kick_outcome_opponent"},
        })
        s3_path = str(tmp_path / "comp_no_s3" / "45" / "gen0300" / "self_sampled" / "census.json")
        with pytest.raises(ValueError) as exc_s3:
            gates.main([
                "--arm", *arm_ok, "--comparator", *comp_no_s3,
                "--gens", "300", "--protocol", "fresh", "--calibration", str(cal_path),
            ])
        msg_s3 = str(exc_s3.value)
        assert "45" in msg_s3 and "300" in msg_s3 and "kick_outcome_opponent" in msg_s3 and s3_path in msg_s3

    def test_style_targets_without_comparator_and_non_interference_with_exit_code(self, tmp_path):
        """[RED] Without comparator -> style_targets is None + MD note; LOSES on S1-S4 never flips gate status or exit code."""
        from tools.audit import gates

        cal_path = tmp_path / "cal.json"
        cal_path.write_text(json.dumps(_make_fake_calibration(gen="300")), encoding="utf-8")

        arm_args = self._build_arm(tmp_path / "arm", {
            "42": dict(s1=0.19, s2=0.05, s3=0.70, s4=0.90),
            "43": dict(s1=0.19, s2=0.05, s3=0.70, s4=0.90),
            "44": dict(s1=0.19, s2=0.05, s3=0.70, s4=0.90),
            "45": dict(s1=0.19, s2=0.05, s3=0.70, s4=0.90),
        })
        # Without comparator
        out_no_comp = tmp_path / "no_comp.json"
        md_no_comp = tmp_path / "no_comp.md"
        rc_no = gates.main([
            "--arm", *arm_args,
            "--gens", "300", "--protocol", "fresh", "--calibration", str(cal_path),
            "--exit-on-c0", "--json", str(out_no_comp), "--md", str(md_no_comp),
        ])
        assert rc_no == 0
        assert json.loads(out_no_comp.read_text(encoding="utf-8"))["style_targets"] is None
        md_text = md_no_comp.read_text(encoding="utf-8")
        assert "## Style Targets" in md_text
        assert "not evaluated (no comparator)" in md_text.split("## Style Targets", 1)[1]

        # With comparator where all 4 style targets LOSE (yet C0/G1-G5/G6 pass) -> exit code still 0
        comp_args = self._build_arm(tmp_path / "comp", {
            "42": dict(s1=0.30, s2=0.30, s3=0.40, s4=0.50),
            "43": dict(s1=0.30, s2=0.30, s3=0.40, s4=0.50),
            "44": dict(s1=0.30, s2=0.30, s3=0.40, s4=0.50),
            "45": dict(s1=0.30, s2=0.30, s3=0.40, s4=0.50),
        })
        out_lose = tmp_path / "lose.json"
        rc_lose = gates.main([
            "--arm", *arm_args, "--comparator", *comp_args,
            "--gens", "300", "--protocol", "fresh", "--calibration", str(cal_path),
            "--exit-on-c0", "--json", str(out_lose),
        ])
        assert rc_lose == 0
        res_lose = json.loads(out_lose.read_text(encoding="utf-8"))
        assert all(r["verdict"] == "LOSES" for r in res_lose["style_targets"])
        assert all(g["passed"] is not False for g in res_lose["gates"].values())

    FRESH_GOLDEN_SEEDS = ("42", "43", "44", "45")

    @classmethod
    def _run_fresh_golden(cls, fixture: Path, out_dir: Path, monkeypatch, capsys):
        """Run `gates` with relative paths from the fixture root; return (exit code, stdout, Markdown bytes, JSON bytes)."""
        from tools.audit import gates

        monkeypatch.chdir(fixture)
        seeds = cls.FRESH_GOLDEN_SEEDS
        rc = gates.main([
            "--arm", *[f"{s}=arm/{s}" for s in seeds],
            "--comparator", *[f"{s}=comp/{s}" for s in seeds],
            "--history", *[f"{s}=histories/{s}.json" for s in seeds],
            "--calibration", "calibration.json",
            "--gens", "100", "200", "300",
            "--protocol", "fresh",
            "--exit-on-c0",
            "--json", str(out_dir / "report.json"), "--md", str(out_dir / "report.md"),
        ])
        out = capsys.readouterr().out
        return rc, out, (out_dir / "report.md").read_bytes(), (out_dir / "report.json").read_bytes()

    def test_fresh_gates_output_matches_the_frozen_golden(self, tmp_path, monkeypatch, capsys):
        """[GUARD] The whole fresh-protocol report (Markdown, JSON and exit code) on the synthetic fixture
        `tests/golden/gates_fresh/` is byte-identical to `tests/golden/gates_fresh_expected.{md,json,exit}`.

        The fixture holds invented values for 4 arm and 4 comparator seeds at gens 100, 200, 300, four training
        histories and a calibration file. It makes C0 fail on seed 44, C0L pass on three seeds and fail on seed 43,
        G3 fail, G6 pass with one seed without takeoff, T1 BEATS with T2 and T3 NEUTRAL, S1 BEATS, S3 LOSES, S2 and
        S4 NEUTRAL, and fills the per-team section. The expected files were produced at commit 5bbfa72, before
        the resume protocols were removed, by running from `tests/golden/gates_fresh/`:
            python -m tools.audit.gates --arm 42=arm/42 43=arm/43 44=arm/44 45=arm/45
              --comparator 42=comp/42 ... 45=comp/45 --history 42=histories/42.json ... 45=histories/45.json
              --calibration calibration.json --gens 100 200 300 --protocol fresh --exit-on-c0
              --json gates_fresh_expected.json --md gates_fresh_expected.md
        (exit status 2, stored in `gates_fresh_expected.exit`).
        """
        golden = REPO_ROOT / "tests" / "golden"
        rc, out, md, js = self._run_fresh_golden(golden / "gates_fresh", tmp_path, monkeypatch, capsys)
        assert md == (golden / "gates_fresh_expected.md").read_bytes()
        assert js == (golden / "gates_fresh_expected.json").read_bytes()
        assert rc == int((golden / "gates_fresh_expected.exit").read_text(encoding="utf-8"))
        assert out.encode("utf-8") == md + b"\n"

    def test_fresh_gates_golden_fails_when_one_census_value_changes(self, tmp_path, monkeypatch, capsys):
        """[GUARD] Changing one census input in a copy of the fixture (seed 42, gen 300, `self_sampled`
        `hold_share_pooled` 0.25 -> 0.35) changes both reports, so the golden comparison is not vacuous."""
        import shutil

        golden = REPO_ROOT / "tests" / "golden"
        fixture = tmp_path / "fixture"
        shutil.copytree(golden / "gates_fresh", fixture)
        census_path = fixture / "arm" / "42" / "gen0300" / "self_sampled" / "census.json"
        payload = json.loads(census_path.read_text(encoding="utf-8"))
        assert payload["summary"]["metrics"]["hold_share_pooled"]["mean"] == 0.25
        payload["summary"]["metrics"]["hold_share_pooled"]["mean"] = 0.35
        census_path.write_text(json.dumps(payload), encoding="utf-8")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        _, _, md, js = self._run_fresh_golden(fixture, out_dir, monkeypatch, capsys)
        assert md != (golden / "gates_fresh_expected.md").read_bytes()
        assert js != (golden / "gates_fresh_expected.json").read_bytes()
        assert "42=0.350" in md.decode("utf-8")

    def test_load_arm_dir_payload_has_no_bookkeeping_and_is_json_serializable(self, tmp_path):
        """[RED] load_arm_dir returns clean JSON-serializable payloads with no internal bookkeeping keys."""
        from tools.audit import gates

        dir_with_int = _write_e6_eval_dir(tmp_path / "with_int", 300)
        dir_no_int = _write_e6_eval_dir(tmp_path / "no_int", 300, write_internals=False)

        for d in (dir_with_int, dir_no_int):
            loaded = gates.load_arm_dir(d)
            dumped = json.dumps(loaded)
            assert isinstance(dumped, str)
            for tag_map in loaded.values():
                for payload in tag_map.values():
                    assert not any(k.startswith("_") for k in payload), f"unexpected private keys in {list(payload)}"


class TestRunCompareE6:
    def test_run_compare_hand_fixture_window_and_fallback_order(self, tmp_path, capsys):
        """[RED] Eval-dir fallback picks first dir containing gen<g:04d>/, window [g-w+1..g] is exact, missing prints '-'."""
        from tools.audit import run_compare

        # runA has two eval dirs: d1 (has gen0200 only) and d2 (has both gen0200 and gen0400).
        # Expect gen0200 from d1 (s1=0.21, NOT d2's 0.99), gen0400 from d2 (s1=0.25).
        d1 = tmp_path / "runA_d1"
        d2 = tmp_path / "runA_d2"
        _write_e6_eval_dir(
            d1, 200,
            s1=0.21, s2=0.14, s3=0.45, s4=0.75,
            t1_blue=-10.0, t1_red=-12.0, t2_blue=-1.0, t2_red=-2.0, t3=0.28,
            extra_census={
                "role_inversion_pooled": (0.04, 0.002),
                "gk_in_box_pooled": (0.98, 0.001),
                "gk_pickup_rate": (0.35, 0.03),
                "holder_spell_s_pooled": (0.41, 0.005),
                "goals_total": (2.2, 0.12),
            },
            extra_internals={
                "power_mean": (0.85, 0.002),
                "full_power_share": (0.48, 0.006),
                "keeper_special_rate": (0.025, 0.001),
                "special_rate": (0.035, 0.001),
                "successful_tackles_pm": (2.4, 0.15),
                "failed_tackles_pm": (11.5, 0.45),
            },
        )
        _write_e6_eval_dir(d2, 200, s1=0.99)  # shadowed by d1
        _write_e6_eval_dir(d2, 400, s1=0.25, s2=0.16, s3=0.42, s4=0.70, t1_blue=-6.0, t1_red=-8.0, t2_blue=0.2, t2_red=0.6, t3=0.33)

        # runB has only gen0200 (gen0400 missing), and no history file
        rb_dir = tmp_path / "runB_d"
        _write_e6_eval_dir(rb_dir, 200, s1=0.19, s2=0.11, s3=0.50, s4=0.82)

        # History for runA with window=3:
        # At gen 200, window is 198, 199, 200 (NOT 197, NOT 201).
        # At gen 400, window is 398, 399, 400 (NOT 397, NOT 401).
        # Put outlier 99.0 at gens 197, 201, 397, 401 so any off-by-one fails.
        roles = ("blue_field", "blue_keeper", "red_field", "red_keeper")
        gen_recs = []
        for g in (197, 198, 199, 200, 201, 397, 398, 399, 400, 401):
            if g in (197, 201, 397, 401):
                val_k = 99.0
                val_s = 99.0
                pk = 99.0
                ps = 99.0
                vc = 99.0
            elif g in (198, 199, 200):
                offset = g - 198  # 0, 1, 2 -> mean offset 1.0
                val_k = 0.01 * (offset + 1)  # 0.01, 0.02, 0.03 -> mean 0.02
                val_s = 0.02 * (offset + 1)  # 0.02, 0.04, 0.06 -> mean 0.04
                pk = 0.10 + 0.01 * offset    # 0.10, 0.11, 0.12 -> mean 0.11
                ps = 0.05 + 0.01 * offset    # 0.05, 0.06, 0.07 -> mean 0.06
                vc = 0.40 + 0.03 * offset    # 0.40, 0.43, 0.46 -> mean 0.43
            else:
                offset = g - 398
                val_k = 0.05 * (offset + 1)  # mean 0.10
                val_s = 0.03 * (offset + 1)  # mean 0.06
                pk = 0.20 + 0.02 * offset    # mean 0.22
                ps = 0.08 + 0.02 * offset    # mean 0.10
                vc = 0.50 + 0.02 * offset    # mean 0.52
            gen_recs.append({
                "generation": g,
                "fire_sat": {r: {"kick": val_k + 0.001 * i, "spec": val_s + 0.001 * i} for i, r in enumerate(roles)},
                "p_kick": {r: pk + 0.001 * i for i, r in enumerate(roles)},
                "p_spec": {r: ps + 0.001 * i for i, r in enumerate(roles)},
                "v_corr_mc": {r: vc + 0.001 * i for i, r in enumerate(roles)},
            })
        hist_a = tmp_path / "hist_a.json"
        hist_a.write_text(json.dumps({"matches": [], "generations": gen_recs}), encoding="utf-8")

        out_md = tmp_path / "cmp.md"
        out_json = tmp_path / "cmp.json"
        rc = run_compare.main([
            "--run", f"runA={d1},{d2}", f"runB={rb_dir}",
            "--history", f"runA={hist_a}",
            "--gens", "200", "400",
            "--window", "3",
            "--md", str(out_md),
            "--json", str(out_json),
        ])
        assert rc == 0
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert data["gens"] == ["200", "400"]
        assert data["window"] == 3
        assert data["eval_dirs"]["runA"]["200"] == str(d1)
        assert data["eval_dirs"]["runA"]["400"] == str(d2)
        assert data["eval_dirs"]["runB"]["400"] is None

        ev_200 = data["eval"]["runA"]["200"]
        assert ev_200["S1"]["mean"] == pytest.approx(0.21)
        assert ev_200["S2"]["mean"] == pytest.approx(0.14)
        assert ev_200["S3"]["mean"] == pytest.approx(0.45)
        assert ev_200["S4"]["mean"] == pytest.approx(0.75)
        assert ev_200["T1_blue"]["mean"] == pytest.approx(-10.0)
        assert ev_200["T1_red"]["mean"] == pytest.approx(-12.0)
        assert ev_200["T1"]["mean"] == pytest.approx(-11.0)
        assert ev_200["T1"]["se"] == pytest.approx(0.5 * math.sqrt(0.4 ** 2 + 0.5 ** 2))
        assert ev_200["T2_blue"]["mean"] == pytest.approx(-1.0)
        assert ev_200["T2_red"]["mean"] == pytest.approx(-2.0)
        assert ev_200["T2"]["mean"] == pytest.approx(-1.5)
        assert ev_200["T2"]["se"] == pytest.approx(0.5 * math.sqrt(0.2 ** 2 + 0.3 ** 2))
        assert ev_200["T3"]["mean"] == pytest.approx(0.28)
        assert ev_200["role_inversion_pooled"]["mean"] == pytest.approx(0.04)
        assert ev_200["gk_in_box_pooled"]["mean"] == pytest.approx(0.98)
        assert ev_200["gk_pickup_rate"]["mean"] == pytest.approx(0.35)
        assert ev_200["holder_spell_s_pooled"]["mean"] == pytest.approx(0.41)
        assert ev_200["goals_total"]["mean"] == pytest.approx(2.2)
        assert ev_200["power_mean"]["mean"] == pytest.approx(0.85)
        assert ev_200["full_power_share"]["mean"] == pytest.approx(0.48)
        assert ev_200["keeper_special_rate"]["mean"] == pytest.approx(0.025)
        assert ev_200["special_rate"]["mean"] == pytest.approx(0.035)
        assert ev_200["successful_tackles_pm"]["mean"] == pytest.approx(2.4)
        assert ev_200["failed_tackles_pm"]["mean"] == pytest.approx(11.5)

        assert data["eval"]["runA"]["400"]["S1"]["mean"] == pytest.approx(0.25)
        assert data["eval"]["runB"]["400"] is None

        h200 = data["history"]["runA"]["200"]
        assert h200["gen_start"] == 198 and h200["gen_end"] == 200
        assert h200["fire"]["fire_sat"]["blue_field"]["kick"] == pytest.approx(0.020)
        assert h200["fire"]["fire_sat"]["red_keeper"]["spec"] == pytest.approx(0.043)
        assert h200["fire"]["p_kick"]["blue_keeper"] == pytest.approx(0.111)
        assert h200["fire"]["p_spec"]["red_field"] == pytest.approx(0.062)
        assert h200["v_corr_mc"]["blue_field"] == pytest.approx(0.430)
        assert h200["v_corr_mc"]["red_keeper"] == pytest.approx(0.433)

        h400 = data["history"]["runA"]["400"]
        assert h400["gen_start"] == 398 and h400["gen_end"] == 400
        assert h400["fire"]["fire_sat"]["blue_field"]["kick"] == pytest.approx(0.100)
        assert h400["v_corr_mc"]["blue_field"] == pytest.approx(0.520)

        assert data["history"]["runB"]["200"] is None
        md = out_md.read_text(encoding="utf-8")
        assert "| - |" in md or " - |" in md

    def test_run_compare_clamps_window_label_and_warns_missing_history(self, tmp_path, capsys):
        """[RED] Window start clamps to 1 when g < window, and missing --history warns once on stderr while rendering '-'."""
        from tools.audit import run_compare

        ra_dir = _write_e6_eval_dir(tmp_path / "runA", 50)
        rb_dir = _write_e6_eval_dir(tmp_path / "runB", 50)
        roles = ("blue_field", "blue_keeper", "red_field", "red_keeper")
        gen_recs = [
            {
                "generation": g,
                "fire_sat": {r: {"kick": 0.01 * g, "spec": 0.02} for r in roles},
                "p_kick": {r: 0.10 for r in roles},
                "p_spec": {r: 0.05 for r in roles},
                "v_corr_mc": {r: 0.25 for r in roles},
            }
            for g in range(1, 51)
        ]
        hist_a = tmp_path / "hist_a.json"
        hist_a.write_text(json.dumps({"matches": [], "generations": gen_recs}), encoding="utf-8")
        missing_b = tmp_path / "missing_hist_b.json"

        out_md = tmp_path / "cmp.md"
        out_json = tmp_path / "cmp.json"
        rc = run_compare.main([
            "--run", f"runA={ra_dir}", f"runB={rb_dir}",
            "--history", f"runA={hist_a}", f"runB={missing_b}",
            "--gens", "50",
            "--window", "100",
            "--md", str(out_md),
            "--json", str(out_json),
        ])
        assert rc == 0
        captured = capsys.readouterr()
        err_lines = [ln for ln in captured.err.splitlines() if ln.strip()]
        assert len(err_lines) == 1
        assert "runB" in err_lines[0] and str(missing_b) in err_lines[0]

        data = json.loads(out_json.read_text(encoding="utf-8"))
        ha = data["history"]["runA"]["50"]
        assert ha["gen_start"] == 1 and ha["gen_end"] == 50
        assert ha["fire"]["fire_sat"]["blue_field"]["kick"] == pytest.approx(0.255)
        assert data["history"]["runB"]["50"] is None

        md = out_md.read_text(encoding="utf-8")
        assert "| 1-50 |" in md
        assert "-49-50" not in md
        assert " - |" in md or "| - |" in md


    def test_run_compare_labels_the_teams_by_drawn_colour(self, tmp_path):
        """[RED] The history table prints `A/red field`, `A/red keeper`, `B/blue field`, `B/blue keeper` for the stored roles
        `blue_field`, `blue_keeper`, `red_field`, `red_keeper`; the JSON keys keep the stored names. Two generations, window 2, hand
        values: fire_sat.kick of role i is 0.10*i + 0.02*g averaged over g = 1, 2 (0.03, 0.13, 0.23, 0.33), v_corr_mc is 0.5 + 0.1*i."""
        from tools.audit import run_compare

        roles = ("blue_field", "blue_keeper", "red_field", "red_keeper")
        hist = _hybrid_history(2)
        for rec in hist["generations"]:
            rec["v_corr_mc"] = {r: 0.5 + 0.1 * i for i, r in enumerate(roles)}
        h = _write_history(tmp_path / "h.json", hist)
        ev = _write_e6_eval_dir(tmp_path / "runA", 2)
        out_json = tmp_path / "cmp.json"
        out_md = tmp_path / "cmp.md"
        assert run_compare.main(["--run", f"runA={ev}", "--history", f"runA={h}", "--gens", "2", "--window", "2",
                                 "--md", str(out_md), "--json", str(out_json)]) == 0
        lines = out_md.read_text(encoding="utf-8").splitlines()
        assert "| fire_sat.kick | A/red field | 1-2 | 0.030 |" in lines
        assert "| fire_sat.kick | A/red keeper | 1-2 | 0.130 |" in lines
        assert "| fire_sat.kick | B/blue field | 1-2 | 0.230 |" in lines
        assert "| fire_sat.kick | B/blue keeper | 1-2 | 0.330 |" in lines
        assert "| v_corr_mc | B/blue keeper | 1-2 | 0.800 |" in lines
        assert not any(f"| {name} |" in ln for ln in lines for name in roles)
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert data["history"]["runA"]["2"]["fire"]["fire_sat"]["blue_field"]["kick"] == pytest.approx(0.03, abs=1e-12)
        assert set(data["history"]["runA"]["2"]["v_corr_mc"]) == set(roles)


# ----------------------------------------------------------------------------
# Pass options at the kick
# ----------------------------------------------------------------------------
def pass_scene(trace: dict, t0: int, *, kicker: int = 3, ball=(300.0, 420.0), bvel=(300.0, 0.0)) -> None:
    """At row `t0`: every player parked on x = 1300 (>= 1000 u from the ball, so no candidate and, beyond every teammate
    segment, no blocker), the ball at `ball`, the kicker 20 u behind it holding and kicking, and `bvel` on row `t0 + 1`."""
    trace["pos"][t0, :, 0] = 1300.0
    trace["pos"][t0, :, 1] = 100.0 + 50.0 * np.arange(12)
    trace["ball"][t0] = ball
    trace["pos"][t0, kicker] = (ball[0] - 20.0, ball[1])
    trace["holder"][t0] = kicker
    set_kick(trace, t0, kicker)
    if t0 + 1 < trace["bvel"].shape[0]:
        trace["bvel"][t0 + 1] = bvel


def one_pass_kick(placements: dict, *, bvel=(300.0, 0.0), T: int = 80, t0: int = 5) -> dict:
    tr = make_trace(T)
    pass_scene(tr, t0, bvel=bvel)
    for who, xy in placements.items():
        tr["pos"][t0, who] = xy
    recs = census.pass_options(tr)
    assert len(recs) == 1
    return recs[0]


class TestPassOptions:
    # Kicker A3 (global 3); ball (300, 420); teammate A4 (global 4) at (500, 420): segment length L = 200 along +x.
    def test_lane_open_when_opponent_is_wide_of_the_segment(self):
        # Opponent B1 (7) at (400, 500): projection s = 100 in (0, 200), perpendicular 80 > 50 -> open.
        rec = one_pass_kick({4: (500.0, 420.0), 7: (400.0, 500.0)})
        assert rec["candidates"] == [4] and rec["open"] == [4]

    def test_lane_blocked_by_opponent_near_the_segment(self):
        # (400, 450): s = 100, perpendicular 30 <= 50 -> blocked.
        rec = one_pass_kick({4: (500.0, 420.0), 7: (400.0, 450.0)})
        assert rec["candidates"] == [4] and rec["open"] == []

    def test_lane_blocked_at_exactly_50(self):
        # (400, 470): perpendicular exactly 50; the lane is open only for > 50 -> blocked.
        rec = one_pass_kick({4: (500.0, 420.0), 7: (400.0, 470.0)})
        assert rec["open"] == []
        # (400, 470.5): 50.5 > 50 -> open.
        rec = one_pass_kick({4: (500.0, 420.0), 7: (400.0, 470.5)})
        assert rec["open"] == [4]

    def test_opponent_beyond_the_receiver_or_behind_the_ball_does_not_block(self):
        # (550, 420): on the line, s = 250 > L = 200 -> outside (0, L) -> open.
        assert one_pass_kick({4: (500.0, 420.0), 7: (550.0, 420.0)})["open"] == [4]
        # (250, 420): s = -50 < 0 -> open.
        assert one_pass_kick({4: (500.0, 420.0), 7: (250.0, 420.0)})["open"] == [4]
        # (500, 420), on the receiver: s = 200 = L is outside the open interval (0, L) -> open.
        assert one_pass_kick({4: (500.0, 420.0), 7: (500.0, 420.0)})["open"] == [4]

    def test_opposing_keeper_blocks_like_any_opponent(self):
        # B's keeper (6) at (400, 450): s = 100, perpendicular 30 -> blocked.
        assert one_pass_kick({4: (500.0, 420.0), 6: (400.0, 450.0)})["open"] == []

    def test_one_blocked_lane_leaves_the_other_open(self):
        # A4 at (500, 420) blocked by B1 at (400, 440) (perp 20); A5 at (300, 620): segment along +y, L = 200; B1 has
        # s = 20, perpendicular 100 -> open for A5.
        rec = one_pass_kick({4: (500.0, 420.0), 5: (300.0, 620.0), 7: (400.0, 440.0)})
        assert rec["candidates"] == [4, 5] and rec["open"] == [5]

    def test_distance_band_is_80_to_700_inclusive(self):
        # Distances from the ball (300, 420): A1 79, A2 80, A4 700, A5 701; A's keeper (0) at 100 is not outfield.
        rec = one_pass_kick({1: (379.0, 420.0), 2: (380.0, 420.0), 4: (1000.0, 420.0), 5: (1001.0, 420.0), 0: (300.0, 520.0)})
        assert rec["candidates"] == [2, 4]
        assert rec["open"] == [2, 4]

    def test_kick_aimed_straight_at_an_open_teammate(self):
        # bvel (300, 0) points along the segment to A4: angle 0 deg -> aimed.
        rec = one_pass_kick({4: (500.0, 420.0)}, bvel=(300.0, 0.0))
        assert rec["has_direction"] and rec["aimed_open"]

    def test_aim_cone_is_15_degrees_inclusive(self):
        def at(deg):
            return (300.0 + 200.0 * math.cos(math.radians(deg)), 420.0 + 200.0 * math.sin(math.radians(deg)))
        # Teammate at 15 deg (float32 positions land within 3e-6 deg of it) -> aimed; 15.01 deg and 16 deg -> not aimed.
        assert one_pass_kick({4: at(15.0)})["aimed_open"] is True
        assert one_pass_kick({4: at(15.01)})["aimed_open"] is False
        assert one_pass_kick({4: at(16.0)})["aimed_open"] is False
        # Below the x axis: -14 deg aimed, -16 deg not aimed.
        assert one_pass_kick({4: at(-14.0)})["aimed_open"] is True
        assert one_pass_kick({4: at(-16.0)})["aimed_open"] is False

    def test_aim_at_a_blocked_teammate_is_not_aimed_open(self):
        # Straight at A4, but B1 at (400, 450) blocks it -> no open candidate, so not aimed at an open one.
        rec = one_pass_kick({4: (500.0, 420.0), 7: (400.0, 450.0)}, bvel=(300.0, 0.0))
        assert rec["open"] == [] and rec["aimed_open"] is False

    def test_kick_without_direction_counts_for_options_only(self):
        # |bvel| = 40 < 50 -> no direction: in pass_option_share, not in aim_open_share. |bvel| = 50 has a direction.
        rec = one_pass_kick({4: (500.0, 420.0)}, bvel=(40.0, 0.0))
        assert rec["open"] == [4] and rec["has_direction"] is False and rec["aimed_open"] is False
        counts = census.pass_option_counts([rec])
        assert (counts["pass_opt_n"], counts["pass_opt_with"]) == (1, 1)
        assert (counts["aim_open_den"], counts["aim_open_n"]) == (0, 0)
        rec50 = one_pass_kick({4: (500.0, 420.0)}, bvel=(50.0, 0.0))
        assert rec50["has_direction"] is True and rec50["aimed_open"] is True

    def test_kick_on_the_last_row_and_keeper_kicks_are_in_no_key(self):
        tr = make_trace(80)
        pass_scene(tr, 79)                                 # last row: no row 80 for the kick direction
        tr["pos"][79, 4] = (500.0, 420.0)
        assert census.pass_options(tr) == []
        tr = make_trace(80)
        pass_scene(tr, 5, kicker=0)                        # A's keeper kicks
        tr["pos"][5, 4] = (500.0, 420.0)
        assert census.pass_options(tr) == []
        assert census.pass_option_counts([]) == {
            "pass_opt_n": 0, "pass_opt_with": 0, "aim_open_den": 0, "aim_open_n": 0,
            "aim_comp_den": 0, "aim_comp_mate": 0, "noopt_den": 0, "noopt_opp": 0,
        }

    def test_outcome_is_the_kick_outcomes_result_and_censored_kicks_leave_the_outcome_keys(self):
        # Kick at 5 received by A4 at 15 -> teammate. A kick at 60 of an 80-row trace with no event is censored by
        # kick_outcomes: it still counts for the option and aim keys, not for completion.
        tr = make_trace(80)
        pass_scene(tr, 5)
        tr["pos"][5, 4] = (500.0, 420.0)
        tr["holder"][15] = 4
        pass_scene(tr, 60)
        tr["pos"][60, 4] = (500.0, 420.0)
        recs = census.pass_options(tr)
        assert [(r["step"], r["outcome"]) for r in recs] == [(5, "teammate"), (60, None)]
        counts = census.pass_option_counts(recs)
        assert counts == {
            "pass_opt_n": 2, "pass_opt_with": 2, "aim_open_den": 2, "aim_open_n": 2,
            "aim_comp_den": 1, "aim_comp_mate": 1, "noopt_den": 0, "noopt_opp": 0,
        }

    @staticmethod
    def _three_matches():
        """Kicks by A3 at rows 5, 80, 155 of 240-row traces; A4 at (500, 420); outcome set by the holder 10 rows later.
        open = A4 open; blocked = B1 at (400, 450); aimed = bvel (300, 0); wide = bvel (0, 300); still = bvel (40, 0).
          match 1: open aimed -> teammate;  blocked aimed -> opponent
          match 2: open aimed -> opponent;  open wide -> teammate
          match 3: blocked aimed -> teammate;  open still -> none;  open aimed -> teammate
        """
        plan = [
            [("open", (300.0, 0.0), 4), ("blocked", (300.0, 0.0), 7)],
            [("open", (300.0, 0.0), 7), ("open", (0.0, 300.0), 4)],
            [("blocked", (300.0, 0.0), 4), ("open", (40.0, 0.0), None), ("open", (300.0, 0.0), 4)],
        ]
        traces = []
        for kicks in plan:
            tr = make_trace(240)
            for t0, (lane, bvel, receiver) in zip((5, 80, 155), kicks):
                pass_scene(tr, t0, bvel=bvel)
                tr["pos"][t0, 4] = (500.0, 420.0)
                if lane == "blocked":
                    tr["pos"][t0, 7] = (400.0, 450.0)
                if receiver is not None:
                    tr["holder"][t0 + 10] = receiver
            traces.append(tr)
        return traces

    def test_four_ratios_on_three_matches_with_hand_values(self):
        per_match = [census.match_metrics(t) for t in self._three_matches()]
        met = census.summarize(per_match)["metrics"]
        # pass_option_share: (with, n) = (1, 2), (2, 2), (2, 3) -> r = 5/7; residuals -3/7, 4/7, -1/7; sum sq 26/49;
        # se = sqrt(3/2 * 26/49) / 7 = sqrt(39) / 49.
        assert met["pass_option_share"]["mean"] == pytest.approx(5 / 7)
        assert met["pass_option_share"]["se"] == pytest.approx(math.sqrt(39) / 49)
        assert met["pass_option_share"]["n"] == 3
        # aim_open_share: (aimed, den) = (1, 1), (1, 2), (1, 1) (match 3's still kick has no direction) -> r = 3/4;
        # residuals 1/4, -1/2, 1/4; sum sq 6/16; se = sqrt(3/2 * 6/16) / 4 = 3/16.
        assert met["aim_open_share"]["mean"] == pytest.approx(3 / 4)
        assert met["aim_open_share"]["se"] == pytest.approx(3 / 16)
        assert met["aim_open_share"]["n"] == 3
        # aim_open_completion: (teammate, aimed) = (1, 1), (0, 1), (1, 1) -> r = 2/3; residuals 1/3, -2/3, 1/3;
        # sum sq 6/9; se = sqrt(3/2 * 6/9) / 3 = 1/3.
        assert met["aim_open_completion"]["mean"] == pytest.approx(2 / 3)
        assert met["aim_open_completion"]["se"] == pytest.approx(1 / 3)
        assert met["aim_open_completion"]["n"] == 3
        # no_option_opponent_share: (opponent, no option) = (1, 1), (0, 0) dropped, (0, 1) -> r = 1/2, n = 2;
        # residuals 1/2, -1/2; se = sqrt(2 * 1/2) / 2 = 1/2.
        assert met["no_option_opponent_share"]["mean"] == pytest.approx(1 / 2)
        assert met["no_option_opponent_share"]["se"] == pytest.approx(1 / 2)
        assert met["no_option_opponent_share"]["n"] == 2
        # The kick outcomes behind these keys are unchanged: 7 outfield kicks, 4 teammate, 2 opponent, 1 none.
        assert sum(m["kicks_n"] for m in per_match) == 7
        assert met["kick_outcome_teammate"]["mean"] == pytest.approx(4 / 7)

    def test_report_prints_the_four_keys(self):
        per_match = [census.match_metrics(t) for t in self._three_matches()]
        text = census.report(per_match, title="hand")
        line = next(ln for ln in text.splitlines() if ln.startswith("pass options"))
        assert "option 71.4%" in line and "aim_open 75.0%" in line
        assert "completion 66.7%" in line and "no_option_opponent 50.0%" in line


# ----------------------------------------------------------------------------
# Chance baselines for aim at the kick (reachable open teammates, kickable arc)
# ----------------------------------------------------------------------------
def aim_scene(trace: dict, t0: int, mates_deg: dict, *, facing_deg: float = 0.0, dir_deg: float = 0.0,
              speed: float = 300.0, dist: float = 200.0) -> None:
    """`pass_scene` at row `t0` (kicker A3, ball (300, 420)) with the kicker facing `facing_deg` at row `t0` and facing
    `facing_deg + 90` on every later row, teammates `{player: bearing in deg}` `dist` u from the ball, and the kick leaving
    at `dir_deg` with `speed` u/s on row `t0 + 1`."""
    T = trace["holder"].shape[0]
    trace.setdefault("ang", np.zeros((T, 12), dtype=np.float32))
    d = math.radians(dir_deg)
    pass_scene(trace, t0, bvel=(speed * math.cos(d), speed * math.sin(d)))
    for who, bearing in mates_deg.items():
        b = math.radians(bearing)
        trace["pos"][t0, who] = (300.0 + dist * math.cos(b), 420.0 + dist * math.sin(b))
    trace["ang"][t0, 3] = math.radians(facing_deg)
    trace["ang"][t0 + 1:, 3] = math.radians(facing_deg + 90.0)


class TestAimReach:
    """Grid of the kickable arc: facing + {-35, -34, ..., +35} deg (71 angles); an angle is covered when an open teammate
    lies within 15 deg of it. A kick is reachable when an open teammate lies within 50 deg of the facing."""

    def test_reachable_aimed_and_arc_baseline_on_two_kicks(self):
        """[RED] Kick at 5: facing 0, the only open teammate A4 at +20.5 deg, kick direction +20.5 deg -> reachable and
        aimed; covered grid angles are those within [5.5, 35.5] deg -> 6..35 = 30 of 71. Kick at 40: the only open teammate
        at +80 deg -> not reachable (arc cover: [65, 95] holds no grid angle -> 0). Hand values:
        open_reachable_share 1/2, aim_open_reachable_share 1/1, aim_open_arc_baseline 30/71."""
        tr = make_trace(80)
        aim_scene(tr, 5, {4: 20.5}, dir_deg=20.5)
        aim_scene(tr, 40, {4: 80.0}, dir_deg=0.0)
        recs = census.pass_options(tr)
        assert [(r["step"], r["open"], r["aimed_open"], r["reachable"]) for r in recs] == [
            (5, [4], True, True), (40, [4], False, False)]
        assert recs[0]["arc_cover"] == pytest.approx(30 / 71)
        assert recs[1]["arc_cover"] == pytest.approx(0.0)
        met = census.summarize([census.match_metrics(tr)])["metrics"]
        assert met["open_reachable_share"]["mean"] == pytest.approx(1 / 2)
        assert met["aim_open_reachable_share"]["mean"] == pytest.approx(1.0)
        assert met["aim_open_arc_baseline"]["mean"] == pytest.approx(30 / 71)
        assert met["open_reachable_share"]["n"] == 1

    def test_facing_wraps_around_pi(self):
        """[RED] Facing +170 deg, open teammate at -175 deg: the difference -345 deg wraps to +15 deg -> reachable.
        Facing +170, teammate at -172.5 deg: relative +17.5 deg -> covered grid angles [2.5, 32.5] -> 3..32 = 30 of 71."""
        tr = make_trace(80)
        aim_scene(tr, 5, {4: -175.0}, facing_deg=170.0, dir_deg=170.0)
        aim_scene(tr, 40, {4: -172.5}, facing_deg=170.0, dir_deg=170.0)
        recs = census.pass_options(tr)
        assert [r["reachable"] for r in recs] == [True, True]
        assert recs[1]["arc_cover"] == pytest.approx(30 / 71)

    def test_reach_limit_is_50_degrees(self):
        """[RED] Open teammate at +49.5 deg -> reachable (arc cover [34.5, 64.5] -> grid angle 35 only -> 1/71); at +50.5 deg
        -> not reachable; at -49.5 deg -> reachable."""
        tr = make_trace(120)
        aim_scene(tr, 5, {4: 49.5})
        aim_scene(tr, 40, {4: 50.5})
        aim_scene(tr, 80, {4: -49.5})
        recs = census.pass_options(tr)
        assert [r["reachable"] for r in recs] == [True, False, True]
        assert recs[0]["arc_cover"] == pytest.approx(1 / 71)
        assert recs[2]["arc_cover"] == pytest.approx(1 / 71)

    def test_arc_cover_is_the_union_over_open_teammates_and_excluded_kicks(self):
        """[RED] Two open teammates at +10.5 and -0.5 deg cover [-4.5, 25.5] and [-15.5, 14.5]: the union -15..25 is 41 of 71
        (not 30 + 30). A blocked teammate covers nothing: A4 at +10.5 blocked by B1 on its lane, A5 at +60 open -> not
        reachable. A kick with an open teammate and no direction (|bvel| 40) and a kick without an open teammate enter no
        reach key."""
        tr = make_trace(200)
        aim_scene(tr, 5, {4: 10.5, 5: -0.5})
        aim_scene(tr, 40, {4: 10.5, 5: 60.0})
        tr["pos"][40, 7] = (300.0 + 100.0 * math.cos(math.radians(10.5)), 420.0 + 100.0 * math.sin(math.radians(10.5)))
        aim_scene(tr, 80, {4: 10.5}, speed=40.0)
        aim_scene(tr, 120, {})
        recs = census.pass_options(tr)
        assert [(r["step"], r["open"]) for r in recs] == [(5, [4, 5]), (40, [5]), (80, [4]), (120, [])]
        assert recs[0]["arc_cover"] == pytest.approx(41 / 71)
        assert [r["reachable"] for r in recs] == [True, False, None, None]
        counts = census.aim_reach_counts(recs)
        assert counts == {"open_reach_den": 2, "open_reach_n": 1, "aim_reach_n": 1, "arc_reach_sum": pytest.approx(41 / 71)}

    def test_trace_without_facing_enters_no_reach_key(self):
        """[RED] A trace without `ang` keeps every E7 pass-option key and has no reach key (n = 0)."""
        tr = make_trace(80)
        pass_scene(tr, 5)
        tr["pos"][5, 4] = (500.0, 420.0)
        recs = census.pass_options(tr)
        assert recs[0]["reachable"] is None and recs[0]["arc_cover"] is None
        met = census.summarize([census.match_metrics(tr)])["metrics"]
        assert met["aim_open_share"]["mean"] == pytest.approx(1.0)
        for key in ("open_reachable_share", "aim_open_reachable_share", "aim_open_arc_baseline"):
            assert met[key] == {"mean": None, "se": None, "n": 0}

    @staticmethod
    def _two_matches():
        """Match 1: the two kicks of `test_reachable_aimed_and_arc_baseline_on_two_kicks` (reach 1 of 2, aimed 1, cover
        30/71). Match 2: one kick with the open teammate at +49.5 deg, direction 0 (reach 1 of 1, aimed 0, cover 1/71)."""
        m1 = make_trace(80)
        aim_scene(m1, 5, {4: 20.5}, dir_deg=20.5)
        aim_scene(m1, 40, {4: 80.0})
        m2 = make_trace(80)
        aim_scene(m2, 5, {4: 49.5})
        return [m1, m2]

    def test_pooled_ratios_and_report_line_on_two_matches(self):
        """[RED] open_reachable_share: (reach, den) = (1, 2), (1, 1) -> r = 2/3; residuals -1/3, 1/3; se = sqrt(2 * 2/9) / 3
        = 2/9. aim_open_reachable_share: (1, 1), (0, 1) -> r = 1/2; residuals 1/2, -1/2; se = sqrt(2 * 1/2) / 2 = 1/2.
        aim_open_arc_baseline: (30/71, 1), (1/71, 1) -> r = 31/142; residuals +-29/142; se = sqrt(2 * 2 * (29/142)^2) / 2
        = 29/142. The report's pass options line ends with reachable 66.7%, aim|reach 50.0%, arc baseline 21.8%."""
        per_match = [census.match_metrics(t) for t in self._two_matches()]
        met = census.summarize(per_match)["metrics"]
        for key, mean, se in (("open_reachable_share", 2 / 3, 2 / 9), ("aim_open_reachable_share", 1 / 2, 1 / 2),
                              ("aim_open_arc_baseline", 31 / 142, 29 / 142)):
            assert met[key]["mean"] == pytest.approx(mean), key
            assert met[key]["se"] == pytest.approx(se), key
            assert met[key]["n"] == 2, key
        line = next(ln for ln in census.report(per_match, title="hand").splitlines() if ln.startswith("pass options"))
        assert " | reachable 66.7% +- 22.2% (n=2) | aim|reach 50.0% +- 50.0% (n=2) | arc baseline 21.8% +- 20.4% (n=2)" in line
        assert line.index("no_option_opponent") < line.index(" | reachable ")


# ----------------------------------------------------------------------------
# Basin keys: reverse motion, facing the ball, tackle-ready opponents
# ----------------------------------------------------------------------------
BASIN_BALL = (700.0, 420.0)


def basin_trace(T: int) -> dict:
    """`make_trace` with a zero body angle `ang` (T, 12) and a zero velocity `vel` (T, 12, 2) for every player."""
    tr = make_trace(T)
    tr["ang"] = np.zeros((T, 12), dtype=np.float32)
    tr["vel"] = np.zeros((T, 12, 2), dtype=np.float32)
    tr["ball"][:] = BASIN_BALL
    return tr


def place_seat(tr: dict, t: int, seat: int, *, bearing_deg: float, facing_deg: float, speed: float = 0.0,
               vel_off_deg: float = 180.0, dist: float = 200.0) -> None:
    """Row `t`: `seat` stands `dist` u from the ball so that the bearing seat -> ball is `bearing_deg`, faces `facing_deg` and
    moves at `speed` u/s along `facing_deg + vel_off_deg` (180 = straight backwards)."""
    b = math.radians(bearing_deg)
    tr["pos"][t, seat] = (BASIN_BALL[0] - dist * math.cos(b), BASIN_BALL[1] - dist * math.sin(b))
    tr["ang"][t, seat] = math.radians(facing_deg)
    v = math.radians(facing_deg + vel_off_deg)
    tr["vel"][t, seat] = (speed * math.cos(v), speed * math.sin(v))


def basin_scene(reverse_rows=(0, 1, 2), away_rows=(2, 3), T: int = 4) -> dict:
    """Ball loose at BASIN_BALL. Team A seat 3 faces within 15 deg of the ball in the rows not in `away_rows` and with the ball
    at +90 deg of its facing in `away_rows`; it moves backwards at 100 u/s in `reverse_rows`. Every other outfield seat stands
    still and faces within 30 deg of the ball. Both keepers move backwards at 100 u/s and face away from the ball."""
    tr = basin_trace(T)
    offsets = {1: 0.0, 2: 10.0, 4: -20.0, 5: 25.0, 7: -5.0, 8: 15.0, 9: -30.0, 10: 30.0, 11: 0.0}
    for t in range(T):
        for i, seat in enumerate(range(12)):
            bearing = 36.0 * i + 7.0 * t
            if seat in (0, 6):
                place_seat(tr, t, seat, bearing_deg=bearing, facing_deg=bearing + 180.0, speed=100.0)
            elif seat == 3:
                away = t in away_rows
                place_seat(tr, t, seat, bearing_deg=bearing, facing_deg=bearing - 90.0 if away else bearing - 15.0,
                           speed=100.0 if t in reverse_rows else 0.0)
            else:
                place_seat(tr, t, seat, bearing_deg=bearing, facing_deg=bearing - offsets[seat])
    return tr


class TestBasinKeys:
    def test_reverse_and_facing_on_a_four_row_trace(self):
        """[RED] Four rows, ball loose. Team A seat 3 moves backwards at 100 u/s in rows 0-2 and has the ball at +90 deg of its
        facing in rows 2-3; every other outfield seat stands still and faces within 30 deg of the ball. Both keepers (excluded)
        move backwards facing away from the ball. Outfield player-decisions: 5 seats x 4 rows = 20 per team.
        reverse_motion: team A 3/20, team B 0/20, pooled 3/40. facing_ball: team A 20 - 2 = 18/20, team B 20/20, pooled 38/40.
        No holder: tackle_ready has no decision, so every denominator is 0 and the share is None."""
        c = census.basin_counts(basin_scene())
        assert (c["reverse_motion_n_A"], c["reverse_motion_den_A"]) == (3, 20)
        assert (c["reverse_motion_n_B"], c["reverse_motion_den_B"]) == (0, 20)
        assert (c["reverse_motion_n_pooled"], c["reverse_motion_den_pooled"]) == (3, 40)
        assert (c["facing_ball_n_A"], c["facing_ball_den_A"]) == (18, 20)
        assert (c["facing_ball_n_B"], c["facing_ball_den_B"]) == (20, 20)
        assert (c["facing_ball_n_pooled"], c["facing_ball_den_pooled"]) == (38, 40)
        assert all(c[f"tackle_ready_{p}_{t}"] == 0 for p in ("n", "den") for t in ("A", "B", "pooled"))
        met = census.summarize([census.match_metrics(basin_scene())])["metrics"]
        assert met["reverse_motion_share_A"]["mean"] == pytest.approx(3 / 20)
        assert met["reverse_motion_share_B"]["mean"] == pytest.approx(0.0)
        assert met["reverse_motion_share_pooled"]["mean"] == pytest.approx(3 / 40)
        assert met["facing_ball_share_A"]["mean"] == pytest.approx(18 / 20)
        assert met["facing_ball_share_pooled"]["mean"] == pytest.approx(38 / 40)
        assert met["tackle_ready_share_pooled"] == {"mean": None, "se": None, "n": 0}

    def test_one_tackle_ready_of_two_near_opponents(self):
        """[RED] One row. Team B outfield seat 8 holds the ball at (700, 420) (the ball is parked at (700, 120), 300 u away, so
        that facing the ball differs from facing the carrier). Team A seats near it: A1 at (650, 420) facing east (dot with the
        direction to the carrier = 1, ready), A2 at (700, 480) facing north (dot = -1, not ready); A3 at (820, 420) facing the
        carrier but 120 u away (not near). Decoys that must not count: team A's keeper at (730, 420) facing the carrier, and team
        B's seat 9 at (700, 460) facing it. Team A: 1 of 2; team B holds the ball, so it has no tackler: None; pooled 1/2."""
        tr = basin_trace(1)
        tr["holder"][0] = 8
        tr["pos"][0, 8] = (700.0, 420.0)
        tr["ball"][0] = (700.0, 120.0)
        tr["pos"][0, 1], tr["ang"][0, 1] = (650.0, 420.0), 0.0
        tr["pos"][0, 2], tr["ang"][0, 2] = (700.0, 480.0), math.pi / 2
        tr["pos"][0, 3], tr["ang"][0, 3] = (820.0, 420.0), math.pi
        tr["pos"][0, 0], tr["ang"][0, 0] = (730.0, 420.0), math.pi
        tr["pos"][0, 9], tr["ang"][0, 9] = (700.0, 460.0), -math.pi / 2
        c = census.basin_counts(tr)
        assert (c["tackle_ready_n_A"], c["tackle_ready_den_A"]) == (1, 2)
        assert (c["tackle_ready_n_B"], c["tackle_ready_den_B"]) == (0, 0)
        assert (c["tackle_ready_n_pooled"], c["tackle_ready_den_pooled"]) == (1, 2)
        met = census.summarize([census.match_metrics(tr)])["metrics"]
        assert met["tackle_ready_share_pooled"]["mean"] == pytest.approx(0.5)
        assert met["tackle_ready_share_A"]["mean"] == pytest.approx(0.5)
        assert met["tackle_ready_share_B"] == {"mean": None, "se": None, "n": 0}

    def test_tackle_ready_boundaries_and_excluded_decisions(self):
        """[RED] Seven rows, carrier at (700, 420) with the ball on it (ready = the tackler's facing within cos >= 0.25 of the
        direction to the carrier, within 85 u inclusive):
          r0 team A keeper holds -> no decision          r1 ball loose -> no decision
          r2 B7 holds, A1 exactly 85 u away facing it    -> A 1/1
          r3 B7 holds, A1 85.5 u away                    -> not near: A 0/0
          r4 B7 holds, A1 faces 75 deg off the carrier (cos 0.2588, ready), A2 76 deg off (cos 0.2419, not) -> A 1/2
          r5 A3 holds, B8 at (700, 480) facing the carrier -> B 1/1
          r6 B7 holds, A1 near with a NaN facing (skipped), A2 near and facing it -> A 1/1
          r7 B7 holds at (0, 0), A1 at (5e-5, 0) facing away from it (the env's gate reads facing_dot = 1 below 1e-4 u) -> A 1/1
        Team A 4/5, team B 1/1, pooled 5/6."""
        tr = basin_trace(8)
        far = tr["pos"].copy()
        tr["holder"][0] = 0
        tr["holder"][2:5] = 7
        tr["holder"][5] = 3
        tr["holder"][6:8] = 7
        tr["pos"][:, 7] = (700.0, 420.0)
        tr["pos"][5, 3], tr["pos"][5, 7] = (700.0, 420.0), far[5, 7]
        tr["pos"][2, 1], tr["ang"][2, 1] = (785.0, 420.0), math.pi
        tr["pos"][3, 1], tr["ang"][3, 1] = (785.5, 420.0), math.pi
        tr["pos"][4, 1], tr["ang"][4, 1] = (650.0, 420.0), math.radians(75.0)
        tr["pos"][4, 2], tr["ang"][4, 2] = (750.0, 420.0), math.pi + math.radians(76.0)
        tr["pos"][5, 8], tr["ang"][5, 8] = (700.0, 480.0), -math.pi / 2
        tr["pos"][6, 1], tr["ang"][6, 1] = (650.0, 420.0), np.nan
        tr["pos"][6, 2], tr["ang"][6, 2] = (750.0, 420.0), math.pi
        tr["pos"][7, 7], tr["pos"][7, 1], tr["ang"][7, 1] = (0.0, 0.0), (5e-5, 0.0), 0.0
        c = census.basin_counts(tr)
        assert (c["tackle_ready_n_A"], c["tackle_ready_den_A"]) == (4, 5)
        assert (c["tackle_ready_n_B"], c["tackle_ready_den_B"]) == (1, 1)
        assert (c["tackle_ready_n_pooled"], c["tackle_ready_den_pooled"]) == (5, 6)

    def test_reverse_and_facing_thresholds_and_non_finite_rows(self):
        """[RED] Team A seat 1 over eight rows (ball at (700, 420); every other outfield seat faces the ball and stands still):
          speed 40.0 straight back -> not reverse (needs > 40)   speed 40.5 straight back -> reverse
          speed 100 at 119 deg from the facing -> not (cos -0.485)   speed 100 at 121 deg -> reverse (cos -0.515)
          speed 100 forwards -> not   speed 0 -> not   NaN velocity -> skipped (no reverse decision)   row 7 as row 5
        Facing the ball, same seat and rows: |bearing - facing| of 0, 59, -59, 61, -61, 179 deg count 1, 1, 1, 0, 0, 0; row 6 has
        facing 170 deg and bearing -175 deg, whose difference wraps to 15 deg -> 1; the NaN-velocity row 7 still counts for
        facing (0 deg) -> 1. Reverse: A 2 of 39 (the NaN decision leaves the denominator), B 0/40, pooled 2/79.
        Facing: A 32 + 5 = 37 of 40, pooled 77/80."""
        tr = basin_trace(8)
        for t in range(8):
            for seat in (2, 3, 4, 5, 7, 8, 9, 10, 11):
                place_seat(tr, t, seat, bearing_deg=40.0 * seat, facing_deg=40.0 * seat)
        specs = [
            dict(bearing_deg=0.0, facing_deg=0.0, speed=40.0),
            dict(bearing_deg=0.0, facing_deg=0.0, speed=40.5),
            dict(bearing_deg=0.0, facing_deg=0.0, speed=100.0, vel_off_deg=119.0),
            dict(bearing_deg=0.0, facing_deg=0.0, speed=100.0, vel_off_deg=121.0),
            dict(bearing_deg=0.0, facing_deg=0.0, speed=100.0, vel_off_deg=0.0),
            dict(bearing_deg=0.0, facing_deg=0.0, speed=0.0),
            dict(bearing_deg=-175.0, facing_deg=170.0, speed=0.0),
            dict(bearing_deg=0.0, facing_deg=0.0, speed=0.0),
        ]
        for t, kw in enumerate(specs):
            place_seat(tr, t, 1, **kw)
        for t, (b, f) in enumerate([(0, 0), (59, 0), (-59, 0), (61, 0), (-61, 0), (179, 0)]):
            place_seat(tr, t, 1, bearing_deg=float(b), facing_deg=float(f), speed=specs[t].get("speed", 0.0),
                       vel_off_deg=specs[t].get("vel_off_deg", 180.0))
        tr["vel"][7, 1] = (np.nan, np.nan)
        c = census.basin_counts(tr)
        assert (c["reverse_motion_n_A"], c["reverse_motion_den_A"]) == (2, 39)
        assert (c["reverse_motion_n_B"], c["reverse_motion_den_B"]) == (0, 40)
        assert (c["reverse_motion_n_pooled"], c["reverse_motion_den_pooled"]) == (2, 79)
        assert (c["facing_ball_n_A"], c["facing_ball_den_A"]) == (37, 40)
        assert (c["facing_ball_n_pooled"], c["facing_ball_den_pooled"]) == (77, 80)

    def test_empty_trace_has_no_basin_decisions(self):
        """[RED] A trace with no rows enters no basin key: every count is 0 and no key is a share."""
        c = census.basin_counts(basin_trace(0))
        assert set(c.values()) == {0}
        assert len(c) == 18

    def test_trace_without_facing_or_velocity_enters_no_basin_key(self):
        """[RED] A trace without `ang`, and one without `vel`, enters none of the three keys (n = 0), and the pass-option and
        aim keys of a kick scene do not change when `vel` is added to the trace."""
        base = basin_scene()
        for dropped in ("ang", "vel"):
            tr = {k: v for k, v in base.items() if k != dropped}
            assert set(census.basin_counts(tr).values()) == {0}, dropped
            met = census.summarize([census.match_metrics(tr)])["metrics"]
            for key in ("reverse_motion_share", "facing_ball_share", "tackle_ready_share"):
                for t in ("A", "B", "pooled"):
                    assert met[f"{key}_{t}"] == {"mean": None, "se": None, "n": 0}, (dropped, key, t)
        kick_keys = ("pass_option_share", "aim_open_share", "aim_open_completion", "no_option_opponent_share",
                     "open_reachable_share", "aim_open_reachable_share", "aim_open_arc_baseline")
        with_ang = make_trace(80)
        aim_scene(with_ang, 5, {4: 20.5}, dir_deg=20.5)
        aim_scene(with_ang, 40, {4: 80.0})
        without_vel = census.summarize([census.match_metrics(with_ang)])["metrics"]
        with_ang["vel"] = np.zeros((80, 12, 2), dtype=np.float32)
        with_vel = census.summarize([census.match_metrics(with_ang)])["metrics"]
        assert without_vel["facing_ball_share_pooled"]["n"] == 0 and with_vel["facing_ball_share_pooled"]["n"] == 1
        for key in kick_keys:
            assert with_vel[key] == without_vel[key], key
        assert without_vel["open_reachable_share"]["mean"] == pytest.approx(1 / 2)
        assert without_vel["aim_open_arc_baseline"]["mean"] == pytest.approx(30 / 71)

    def test_pooled_ratios_and_report_line_on_two_matches(self):
        """[RED] Match 1 is `basin_scene()`, match 2 the same scene with no reversing and everybody facing the ball. reverse_motion
        pooled: (3, 40), (0, 40) -> r = 3/80, residuals +1.5 and -1.5, se = sqrt(2 * 4.5) / 80 = 0.0375. Team A: (3, 20), (0, 20)
        -> r = 3/40, se = sqrt(9) / 40 = 0.075. facing_ball pooled: (38, 40), (40, 40) -> r = 0.975, residuals -1 and +1,
        se = sqrt(2 * 2) / 80 = 0.025. The report line of match 1 alone reads
        `basin: reverse 0.075 +- n/a (n=1) | facing ball 0.950 +- n/a (n=1) | tackle-ready -  (n=0)`."""
        calm = basin_scene(reverse_rows=(), away_rows=())
        for t in range(4):
            place_seat(calm, t, 3, bearing_deg=36.0 * 3 + 7.0 * t, facing_deg=36.0 * 3 + 7.0 * t)
        per_match = [census.match_metrics(basin_scene()), census.match_metrics(calm)]
        met = census.summarize(per_match)["metrics"]
        for key, mean, se in (("reverse_motion_share_pooled", 3 / 80, 0.0375), ("reverse_motion_share_A", 3 / 40, 0.075),
                              ("facing_ball_share_pooled", 0.975, 0.025)):
            assert met[key]["mean"] == pytest.approx(mean), key
            assert met[key]["se"] == pytest.approx(se), key
            assert met[key]["n"] == 2, key
        line = next(ln for ln in census.report(per_match[:1], title="hand").splitlines() if ln.startswith("basin:"))
        assert line == "basin: reverse 0.075 +- n/a (n=1) | facing ball 0.950 +- n/a (n=1) | tackle-ready -  (n=0)"


# ----------------------------------------------------------------------------
# Defence keys: goal-side defenders at goals, the nearest defender at shots, regains, tackle spells, deep block
# ----------------------------------------------------------------------------
DEFENCE_RATIO_KEYS = (
    "goal_side_at_goal", "shot_nearest_def_facing", "shot_nearest_def_closing", "regain_goalside_at_shot",
    "regain_goalside_retention", "tackle_in_range_rate", "tackle_win_rate", "deep_block_ge4", "outfield_depth_oop",
)


def defence_trace(T: int) -> dict:
    """`make_trace` with a zero body angle, zero velocity, zero actions and zero shot counters for every player."""
    tr = make_trace(T)
    tr["ang"] = np.zeros((T, 12), dtype=np.float32)
    tr["vel"] = np.zeros((T, 12, 2), dtype=np.float32)
    tr["act"] = np.zeros((T, 12, 4), dtype=np.float32)
    tr["s_shots"] = np.zeros((T, 2), dtype=np.int32)
    return tr


def place_outfield(tr: dict, team: int, xs, rows=slice(None), y: float = 420.0) -> None:
    """World x of `team`'s five outfielders (slots 1-5) on `rows`; slot s stands at y + 10 s."""
    for slot, x in zip(range(1, 6), xs):
        tr["pos"][rows, 6 * team + slot] = (x, y + 10.0 * slot)


def tile_trace(tr: dict, times: int) -> dict:
    """`tr` repeated `times` times along the decision axis (only for traces whose cumulative counters stay zero)."""
    out = {k: np.tile(v, (times,) + (1,) * (v.ndim - 1)) for k, v in tr.items() if k != "n_steps"}
    out["n_steps"] = np.array(int(tr["n_steps"]) * times)
    return out


class TestDeepBlock:
    # Appendix table: rows 0-3, the ball and the defending team's five outfielders in the team's own frame (own goal line at 0).
    BALL = (700.0, 699.0, 1000.0, 1000.0)
    XS = ((100.0, 200.0, 280.0, 280.0, 900.0), (100.0,) * 5, (100.0, 200.0, 281.0, 280.0, 900.0), (100.0,) * 5)

    @classmethod
    def scene(cls, team: int, sides=(0, 1)) -> dict:
        """The table in `team`'s frame; the other team's outfielders stand at world x = 700 (canonical 700, never deep)."""
        tr = defence_trace(4)
        tr["side"][:] = sides
        world = (lambda cx: cx) if sides[team] == 0 else (lambda cx: 1400.0 - cx)
        for t in range(4):
            tr["ball"][t] = (world(cls.BALL[t]), 420.0)
            place_outfield(tr, team, [world(x) for x in cls.XS[t]], rows=t)
            place_outfield(tr, 1 - team, [700.0] * 5, rows=t)
        return tr

    @pytest.mark.parametrize("team, sides", [(0, (0, 1)), (1, (0, 1)), (0, (1, 0)), (1, (1, 0))])
    def test_hand_table_counts_in_each_teams_own_frame(self, team, sides):
        """[RED] Qualifying decisions are those with the ball at canonical x >= 700 (exactly 700 counts, 699 does not); a decision is
        deep when at least 4 outfielders stand at canonical x <= 280 (exactly 280 counts, 281 does not; exactly 4 counts).
          row 0 ball 700, x 100 200 280 280 900  qualifies, 4 deep  -> deep    row 1 ball 699              -> not counted
          row 2 ball 1000, x 100 200 281 280 900 qualifies, 3 deep  -> not     row 3 ball 1000, five at 100 -> deep
        The team itself: 2 deep of 3 qualifying, in either colour and with either side assignment (team B's world x is 1400 - x
        on the default sides). The other team (at world x 700) has the ball at canonical 700, 701, 400, 400: 2 qualifying, 0 deep.
        Pooled: 2 of 5."""
        c = census.defence_counts(self.scene(team, sides))
        own, other = "AB"[team], "AB"[1 - team]
        assert (c[f"deep_block_ge4_n_{own}"], c[f"deep_block_ge4_den_{own}"]) == (2, 3)
        assert (c[f"deep_block_ge4_n_{other}"], c[f"deep_block_ge4_den_{other}"]) == (0, 2)
        assert (c["deep_block_ge4_n_pooled"], c["deep_block_ge4_den_pooled"]) == (2, 5)

    def test_outfield_depth_counts_only_decisions_with_the_opponent_holding_the_ball(self):
        """[RED] Four rows. Team A's outfielders stand at canonical x 100 200 300 400 500 (mean 300) on row 0 and 50 50 50 50 150
        (mean 70) on row 1; team B holds the ball on both. On row 2 team A holds (not counted for A) and on row 3 the ball is loose
        (counted for nobody). Team B's outfielders stand at world x 1300 1200 1100 1000 900 on every row (canonical 100 ... 500,
        mean 300). Team A: (300 + 70) / 2 = 185. Team B: row 2 only, 300. Pooled: 670 / 3."""
        tr = defence_trace(4)
        place_outfield(tr, 0, [100.0, 200.0, 300.0, 400.0, 500.0], rows=0)
        place_outfield(tr, 0, [50.0, 50.0, 50.0, 50.0, 150.0], rows=1)
        place_outfield(tr, 0, [700.0] * 5, rows=slice(2, 4))
        place_outfield(tr, 1, [1300.0, 1200.0, 1100.0, 1000.0, 900.0])
        tr["holder"][0:2] = 9
        tr["holder"][2] = 2
        c = census.defence_counts(tr)
        assert (c["outfield_depth_oop_sum_A"], c["outfield_depth_oop_den_A"]) == (370.0, 2)
        assert (c["outfield_depth_oop_sum_B"], c["outfield_depth_oop_den_B"]) == (300.0, 1)
        assert (c["outfield_depth_oop_sum_pooled"], c["outfield_depth_oop_den_pooled"]) == (670.0, 3)

    def test_the_default_floor_is_100_qualifying_decisions(self):
        """[RED] 100 rows with the ball at x = 1000 and team A's five outfielders at x = 100: 100 qualifying decisions, all deep -> 1.0;
        99 rows -> None, still counted in n. Team B's ball is at canonical 400 on every row: no qualifying decision, None, n = 0."""
        def metrics(rows: int) -> dict:
            tr = defence_trace(rows)
            tr["ball"][:] = (1000.0, 420.0)
            place_outfield(tr, 0, [100.0] * 5)
            return census.summarize([census.match_metrics(tr)])["metrics"]

        at, below = metrics(100), metrics(99)
        assert at["deep_block_ge4_A"] == {"mean": pytest.approx(1.0), "se": None, "n": 1}
        assert below["deep_block_ge4_A"] == {"mean": None, "se": None, "n": 1}
        assert at["deep_block_ge4_B"] == {"mean": None, "se": None, "n": 0}

    def test_tiled_hand_table_reaches_the_floor_for_the_team_with_three_qualifying_rows(self):
        """[RED] The table of team A tiled 34 times: 102 qualifying decisions, 68 deep -> 2/3. Team B has 2 x 34 = 68 qualifying
        decisions, below 100 -> None. Pooled: 68 deep of 170 qualifying = 0.4. With `min_decisions` 3 the untiled table gives 2/3
        and with 4 it gives None."""
        met = census.summarize([census.match_metrics(tile_trace(self.scene(0), 34))])["metrics"]
        assert met["deep_block_ge4_A"]["mean"] == pytest.approx(2 / 3)
        assert met["deep_block_ge4_B"]["mean"] is None
        assert met["deep_block_ge4_pooled"]["mean"] == pytest.approx(0.4)
        one = [census.match_metrics(self.scene(0))]
        assert census.defence_metrics(one, min_decisions=3)["deep_block_ge4_A"]["mean"] == pytest.approx(2 / 3)
        assert census.defence_metrics(one, min_decisions=4)["deep_block_ge4_A"]["mean"] is None


def goal_scene(goal_row: int = 30, late_shot: bool = True) -> dict:
    """70 rows, team A defends the left goal. Team A (striker holding) shoots on row 10 and scores on `goal_row`; team B shoots on row
    20 and scores on row 65 (45 decisions later); with `late_shot`, team A shoots again on row 40 and does not score within 45 decisions
    (for a `goal_row` of 30 or less).
    Row 10: ball (1000, 420), team B's outfielders at world x 1200 1000 1001 900 1300 (canonical 200 400 399 500 100): goal-side
    (canonical x strictly below the ball's 400) are 200, 399, 100 -> 3 (the 400 is level, not goal-side).
    Row 20: ball (300, 420), team A's outfielders at x 100 299 300 350 400: goal-side are 100 and 299 -> 2."""
    tr = defence_trace(70)
    tr["holder"][10], tr["holder"][20] = 5, 11
    tr["s_shots"][10:, 0], tr["s_shots"][20:, 1] = 1, 1
    if late_shot:
        tr["holder"][40] = 5
        tr["s_shots"][40:, 0] = 2
    tr["s_score"][goal_row:, 0], tr["s_score"][65:, 1] = 1, 1
    tr["ball"][10], tr["ball"][20] = (1000.0, 420.0), (300.0, 420.0)
    place_outfield(tr, 1, [1200.0, 1000.0, 1001.0, 900.0, 1300.0], rows=10)
    place_outfield(tr, 0, [100.0, 299.0, 300.0, 350.0, 400.0], rows=20)
    return tr


class TestGoalSideAtGoal:
    def test_goal_side_counts_of_the_conceding_team_at_the_scoring_shot(self):
        """[RED] See `goal_scene`: team B concedes the row-10 shot (3 goal-side), team A the row-20 shot (2 goal-side); the row-40 shot
        is followed by no goal and enters nothing. Team A: 2 over 1 goal, team B: 3 over 1, pooled 5 over 2."""
        c = census.defence_counts(goal_scene())
        assert (c["goal_side_at_goal_sum_A"], c["goal_side_at_goal_den_A"]) == (2, 1)
        assert (c["goal_side_at_goal_sum_B"], c["goal_side_at_goal_den_B"]) == (3, 1)
        assert (c["goal_side_at_goal_sum_pooled"], c["goal_side_at_goal_den_pooled"]) == (5, 2)
        met = census.defence_metrics([census.match_metrics(goal_scene())] * 10)
        assert [met[f"goal_side_at_goal_{g}"]["mean"] for g in ("A", "B", "pooled")] == pytest.approx([2.0, 3.0, 2.5])

    @pytest.mark.parametrize("goal_row, counted", [(55, 1), (56, 0)])
    def test_the_goal_must_come_within_45_decisions_of_the_shot(self, goal_row, counted):
        """[RED] The row-10 shot of team A is a goal shot when team A's score rises on a row up to 10 + 45 = 55 (inclusive), not on 56
        (the scene without its row-40 shot, which a goal on row 55 would also follow)."""
        c = census.defence_counts(goal_scene(goal_row, late_shot=False))
        assert c["goal_side_at_goal_den_B"] == counted

    def test_a_trace_without_shot_counters_enters_no_goal_shot(self):
        """[RED] Goals without `s_shots` (a hand trace of the older layout) give zero counts and no error."""
        tr = goal_scene()
        del tr["s_shots"]
        c = census.defence_counts(tr)
        assert c["goal_side_at_goal_den_pooled"] == 0 and c["regain_goalside_at_shot_den_pooled"] == 0


def shot_scene(*, vel_x=-30.0, facing: float = math.pi, k: int = 12, T: int = 20) -> dict:
    """Team A's striker (seat 5) holds the ball at (300, 420) and shoots on row `k`. Team B's keeper stands at (340, 420), closer than
    any outfielder, facing away and backing off (a decoy: only outfielders count). Team B's seat 8 stands at (380, 420), 80 u from the
    shooter, with body angle `facing` and x velocity `vel_x` (a scalar or one value per row). Every other player stands more than 400 u
    from the shooter."""
    tr = defence_trace(T)
    tr["pos"][:, 5] = (300.0, 420.0)
    tr["holder"][k] = 5
    tr["s_shots"][k:, 0] = 1
    set_kick(tr, k, 5)
    tr["pos"][:, 6] = (340.0, 420.0)
    tr["vel"][:, 6] = (50.0, 0.0)
    tr["pos"][:, 8] = (380.0, 420.0)
    tr["ang"][:, 8] = facing
    tr["vel"][:, 8, 0] = vel_x
    return tr


class TestShotNearestDefender:
    @staticmethod
    def flags(tr: dict, team: str = "B"):
        c = census.defence_counts(tr)
        assert (c[f"shot_nearest_def_facing_den_{team}"], c[f"shot_nearest_def_closing_den_{team}"]) == (1, 1)
        return c[f"shot_nearest_def_facing_n_{team}"], c[f"shot_nearest_def_closing_n_{team}"]

    def test_defender_facing_the_shooter_and_closing_at_30_u_per_s(self):
        """[RED] The shooter is at (300, 420); the nearest conceding outfielder (80 u away, the keeper at 40 u is ignored) faces angle pi, the
        bearing to the shooter, so cos = 1 >= 0.25 and facing = 1; its velocity (-30, 0) is 30 u/s toward the shooter on each of the
        8 decisions before the shot, above 20, so closing = 1. Team A took the shot, so these are team B's keys; team A's have no
        shot against it. Over the shots: facing 1/1, closing 1/1."""
        c = census.defence_counts(shot_scene())
        assert [c[f"shot_nearest_def_{k}_{g}"] for k in ("facing_n", "facing_den", "closing_n", "closing_den") for g in ("A", "B", "pooled")] == [
            0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]

    @pytest.mark.parametrize("vel_x, closing", [(-20.0, 0), (-20.5, 1)])
    def test_closing_must_exceed_20_u_per_s(self, vel_x, closing):
        """[RED] A velocity of (-20, 0) is 20 u/s toward the shooter: not above 20, closing 0; (-20.5, 0): closing 1."""
        assert self.flags(shot_scene(vel_x=vel_x)) == (1, closing)

    @pytest.mark.parametrize("off_deg, facing", [(0, 1), (75, 1), (76, 0), (90, 0), (180, 0)])
    def test_facing_gate_is_cos_0_25_at_the_shot_row(self, off_deg, facing):
        """[RED] The bearing to the shooter is pi. A body angle `off_deg` away from it faces the shooter when cos(off_deg) >= 0.25:
        75 deg (cos 0.2588) yes, 76 deg (cos 0.2419), 90 deg and 180 deg no. Closing is unaffected (velocity -30)."""
        assert self.flags(shot_scene(facing=math.pi - math.radians(off_deg))) == (facing, 1)

    def test_closing_is_the_mean_over_exactly_the_8_decisions_before_the_shot(self):
        """[RED] Shot on row 12: rows 4-11 are the window. Velocity -30 on rows 4-11 and +100 on rows 3 and 12 -> mean 30 -> closing 1
        (neither neighbour enters). Velocity -10 on rows 4-7 and -30 on rows 8-11: mean 20, not above 20 -> 0; -10 on rows 4-6 and -30 on
        rows 7-11: mean (30 + 150) / 8 = 22.5 -> 1."""
        vel = np.zeros(20)
        vel[4:12], vel[3], vel[12] = -30.0, 100.0, 100.0
        assert self.flags(shot_scene(vel_x=vel))[1] == 1
        vel = np.zeros(20)
        vel[4:8], vel[8:12] = -10.0, -30.0
        assert self.flags(shot_scene(vel_x=vel))[1] == 0
        vel[7] = -30.0
        vel[4:7] = -10.0
        assert self.flags(shot_scene(vel_x=vel))[1] == 1

    def test_the_nearest_outfielder_is_chosen_not_a_farther_one_that_faces_and_closes(self):
        """[RED] Seat 8 at 80 u faces away (angle 0) and backs off (+30); seat 9 at 120 u faces the shooter and closes at 30 u/s. The nearest
        is seat 8: facing 0, closing 0."""
        tr = shot_scene(vel_x=30.0, facing=0.0)
        tr["pos"][:, 9], tr["ang"][:, 9], tr["vel"][:, 9] = (420.0, 420.0), math.pi, (-30.0, 0.0)
        assert self.flags(tr) == (0, 0)

    def test_both_teams_shoot_and_each_conceding_team_has_its_own_key(self):
        """[RED] Team A's shot as in `shot_scene` (team B concedes: 1/1, 1/1). Team B's striker (seat 11) holds the ball at (1100, 420) and
        shoots on row 5; team A's seat 2 at (1020, 420), 80 u away, faces angle pi (away from the shooter) and moves at (-10, 0) (away):
        team A concedes with facing 0/1, closing 0/1. Pooled 1/2 and 1/2."""
        tr = shot_scene()
        tr["pos"][:, 11], tr["holder"][5], tr["s_shots"][5:, 1] = (1100.0, 420.0), 11, 1
        tr["pos"][:, 2], tr["ang"][:, 2], tr["vel"][:, 2] = (1020.0, 420.0), math.pi, (-10.0, 0.0)
        c = census.defence_counts(tr)
        assert (c["shot_nearest_def_facing_n_A"], c["shot_nearest_def_facing_den_A"]) == (0, 1)
        assert (c["shot_nearest_def_closing_n_A"], c["shot_nearest_def_closing_den_A"]) == (0, 1)
        assert (c["shot_nearest_def_facing_n_B"], c["shot_nearest_def_facing_den_B"]) == (1, 1)
        assert (c["shot_nearest_def_closing_n_B"], c["shot_nearest_def_closing_den_B"]) == (1, 1)
        assert (c["shot_nearest_def_facing_n_pooled"], c["shot_nearest_def_facing_den_pooled"]) == (1, 2)
        assert (c["shot_nearest_def_closing_n_pooled"], c["shot_nearest_def_closing_den_pooled"]) == (1, 2)

    def test_a_trace_without_facing_or_velocity_enters_only_the_key_that_needs_it(self):
        """[RED] Without `ang` the facing key has no shot but the closing key keeps its shot; without `vel` the reverse."""
        no_ang = {k: v for k, v in shot_scene().items() if k != "ang"}
        no_vel = {k: v for k, v in shot_scene().items() if k != "vel"}
        c = census.defence_counts(no_ang)
        assert (c["shot_nearest_def_facing_den_B"], c["shot_nearest_def_closing_den_B"]) == (0, 1)
        c = census.defence_counts(no_vel)
        assert (c["shot_nearest_def_facing_den_B"], c["shot_nearest_def_closing_den_B"]) == (1, 0)

    def test_a_non_finite_input_removes_the_shot_from_the_key_that_reads_it(self):
        """[RED] A NaN body angle of the nearest defender on the shot row leaves the shot out of the facing key (den 0) and in the closing key;
        a NaN velocity inside the 8-decision window leaves it out of the closing key and in the facing key."""
        tr = shot_scene()
        tr["ang"][12, 8] = np.nan
        c = census.defence_counts(tr)
        assert (c["shot_nearest_def_facing_den_B"], c["shot_nearest_def_closing_den_B"]) == (0, 1)
        tr = shot_scene()
        tr["vel"][7, 8] = (np.nan, np.nan)
        c = census.defence_counts(tr)
        assert (c["shot_nearest_def_facing_den_B"], c["shot_nearest_def_closing_den_B"]) == (1, 0)

    @pytest.mark.parametrize("holder", [-1, 9])
    def test_a_shot_that_the_shooting_team_did_not_hold_raises(self, holder):
        """[RED] A shot (a rise of `s_shots`) is a kick by the holder, so a row without a holder of the shooting team is a corrupt
        trace: the loose ball (-1) and a holder of the other team (seat 9) both raise instead of dropping the shot."""
        tr = shot_scene()
        tr["holder"][12] = holder
        with pytest.raises(ValueError, match="shot"):
            census.defence_counts(tr)


def regain_scene(shot_rows=(14, 24)) -> dict:
    """25 rows. Team A holds on rows 0-4 (seat 3) and 20-21 (seat 2), team B on rows 8-16 (seat 9) and 23-24 (seat 10), the ball is loose
    otherwise: team B regains on rows 8 and 23, team A on row 20. Team B shoots on each row of `shot_rows` (14 and 24 hold the ball).
      row 8  ball (600, 420), team A outfielders at x 100 400 599 600 900  -> goal-side (x < 600): seats 1, 2, 3
      row 14 ball (300, 420), team A outfielders at x 100 400 250 300 500  -> goal-side (x < 300): seats 1, 3
      rows 23 and 24 ball (100, 420), team A outfielders at x 500 ... 900  -> goal-side: nobody"""
    tr = defence_trace(25)
    tr["holder"][0:5], tr["holder"][8:17], tr["holder"][20:22], tr["holder"][23:25] = 3, 9, 2, 10
    for i, row in enumerate(shot_rows):
        tr["s_shots"][row:, 1] = i + 1
    for row, bx, xs in ((8, 600.0, [100, 400, 599, 600, 900]), (14, 300.0, [100, 400, 250, 300, 500]),
                        (23, 100.0, [500, 600, 700, 800, 900]), (24, 100.0, [500, 600, 700, 800, 900])):
        tr["ball"][row] = (bx, 420.0)
        place_outfield(tr, 0, xs, rows=row)
    return tr


class TestRegainKeys:
    def test_goal_side_at_the_shot_and_retention_since_the_regain(self):
        """[RED] See `regain_scene`. Team A loses the ball on row 8 and team B shoots on row 14 inside that spell: 2 of A's outfielders are
        goal-side at the shot (seats 1 and 3; seat 4 stands level with the ball, 300, which is not goal-side), and of the 3 goal-side
        at the regain (seats 1, 2, 3) two (1 and 3) still are: retention 2/3. Team A loses it again on row 23 and team B shoots on row 24: 0
        goal-side at the shot; nobody was goal-side at the regain, so that spell has no retention. Over both spells:
        at the shot (2 + 0) / 2 spells = 1.0; retention (2/3) / 1 spell. Team A's regain on row 20 has no team-A shot: team B's counts are 0."""
        c = census.defence_counts(regain_scene())
        assert (c["regain_goalside_at_shot_sum_A"], c["regain_goalside_at_shot_den_A"]) == (2, 2)
        assert c["regain_goalside_retention_sum_A"] == pytest.approx(2 / 3)
        assert c["regain_goalside_retention_den_A"] == 1
        assert [c[f"regain_goalside_{k}_B"] for k in ("at_shot_sum", "at_shot_den", "retention_sum", "retention_den")] == [0, 0, 0, 0]
        assert (c["regain_goalside_at_shot_sum_pooled"], c["regain_goalside_at_shot_den_pooled"]) == (2, 2)
        met = census.defence_metrics([census.match_metrics(regain_scene())] * 5)
        assert met["regain_goalside_at_shot_A"]["mean"] == pytest.approx(1.0)
        assert met["regain_goalside_retention_A"]["mean"] == pytest.approx(2 / 3)
        assert met["regain_goalside_at_shot_B"]["mean"] is None

    def test_only_a_shot_inside_the_spell_counts(self):
        """[RED] The only shot is on row 24, in team B's second possession. The first spell (regain row 8, ending when team A holds on
        row 20) has no shot even though one comes later in the match: one spell reaches a shot, not two. A match without any shot has none."""
        c = census.defence_counts(regain_scene(shot_rows=(24,)))
        assert (c["regain_goalside_at_shot_sum_A"], c["regain_goalside_at_shot_den_A"]) == (0, 1)
        assert c["regain_goalside_retention_den_A"] == 0
        assert census.defence_counts(regain_scene(shot_rows=()))["regain_goalside_at_shot_den_A"] == 0

    def test_the_spell_shot_is_a_rise_after_the_regain_row(self):
        """[RED] A rise of team B's shot counter on row 8, the regain row itself, is not a shot of the spell (the spell is the rows after
        the regain): no regain reaches a shot."""
        assert census.defence_counts(regain_scene(shot_rows=(8,)))["regain_goalside_at_shot_den_A"] == 0

    def test_a_match_without_a_regain_has_no_spell(self):
        """[RED] Ball loose or held by one team for the whole match: no regain, every regain count 0."""
        tr = defence_trace(30)
        c = census.defence_counts(tr)
        assert all(c[f"regain_goalside_{k}_{g}"] == 0 for k in ("at_shot_sum", "at_shot_den", "retention_sum", "retention_den") for g in ("A", "B", "pooled"))
        tr["holder"][:] = 4
        assert census.defence_counts(tr)["regain_goalside_at_shot_den_pooled"] == 0
        met = census.summarize([census.match_metrics(tr)] * 10)["metrics"]
        assert met["regain_goalside_at_shot_A"] == {"mean": None, "se": None, "n": 0}

    def test_team_b_regain_frame_is_mirrored(self):
        """[RED] The scene reflected through the pitch centre with the colours swapped (seats 0-5 and 6-11 exchanged, every world x
        becomes 1400 - x, holders and shot counters follow): team B now defends the right goal exactly as team A defended the left
        one, so every team-A value of the original is the team-B value of the reflection and the other way round."""
        tr = regain_scene()
        mirror = dict(tr)
        for key in ("pos", "ball"):
            mirror[key] = tr[key].copy()
            mirror[key][..., 0] = 1400.0 - mirror[key][..., 0]
        mirror["pos"] = np.concatenate([mirror["pos"][:, 6:], mirror["pos"][:, :6]], axis=1)
        mirror["holder"] = np.where(tr["holder"] < 0, tr["holder"], (tr["holder"] + 6) % 12).astype(np.int8)
        mirror["s_shots"] = tr["s_shots"][:, ::-1].copy()
        a, b = census.defence_counts(tr), census.defence_counts(mirror)
        for key in ("regain_goalside_at_shot_sum", "regain_goalside_at_shot_den", "regain_goalside_retention_sum", "regain_goalside_retention_den"):
            assert b[f"{key}_B"] == pytest.approx(a[f"{key}_A"]), key
            assert b[f"{key}_A"] == pytest.approx(a[f"{key}_B"]), key


def spell_scene(defending_team: int = 0) -> dict:
    """12 rows. The defending team's attacker of slot 3 on the other team holds the ball at (600, 420) on every row. The defending team's
    outfielders stand far away at x = 50 except as listed (slot s: seat 6 d + s; each stands on the carrier's row y = 420 facing +x, towards it).
    Reach per slot is 85, 85, 75, 75, 65 u beyond the 28 u of body contact, so the chance ranges are 113, 113, 103, 103, 93 u.
      slot 1 at 110 u on rows 2-4, fires on row 3        -> chance spell 1 (fired)
      slot 3 at 100 u on rows 6-7, never fires           -> chance spell 2 (not fired)
      slot 2 at 150 u on row 0, fires                    -> out of reach, within the 180 u whiff range: counts
      slot 4 at 200 u on row 9, fires                    -> beyond the whiff range: counts for nothing
      slot 5 at 100 u on row 10, fires                   -> out of reach for slot 5 (93 u): counts
      slot 4 at 100 u on row 11 facing away, fires       -> in reach but not facing: counts"""
    d = defending_team
    tr = defence_trace(12)
    carrier = 6 * (1 - d) + 3
    tr["holder"][:] = carrier
    tr["pos"][:, carrier] = (600.0, 420.0)
    for slot in range(1, 6):
        tr["pos"][:, 6 * d + slot] = (50.0, 420.0 + 10.0 * slot)

    def put(slot, rows, x, *, ang=0.0, fire_rows=()):
        seat = 6 * d + slot
        tr["pos"][rows, seat] = (x, 420.0)
        tr["ang"][rows, seat] = ang
        for row in fire_rows:
            tr["act"][row, seat, 3] = 1.0

    put(1, slice(2, 5), 490.0, fire_rows=(3,))
    put(3, slice(6, 8), 500.0)
    put(2, 0, 450.0, fire_rows=(0,))
    put(4, 9, 400.0, fire_rows=(9,))
    put(5, 10, 500.0, fire_rows=(10,))
    put(4, 11, 500.0, ang=math.pi, fire_rows=(11,))
    return tr


class TestTackleSpells:
    @pytest.mark.parametrize("defending_team", [0, 1])
    def test_spells_fires_and_out_of_reach_fires_by_slot(self, defending_team):
        """[RED] See `spell_scene`: 2 chance spells (slot 1 on rows 2-4, slot 3 on rows 6-7), one with a tackle fired and none won, and 3
        tackle decisions within 180 u of the carrier outside every own spell row (slot 2 on row 0, slot 5 on row 10, slot 4 on row 11; slot 4's
        fire at 200 u does not count). Team B defending gives the same numbers on its own keys. The out-of-reach count pools as a sum."""
        c = census.defence_counts(spell_scene(defending_team))
        own, other = "AB"[defending_team], "AB"[1 - defending_team]
        assert (c[f"tackle_in_range_rate_n_{own}"], c[f"tackle_in_range_rate_den_{own}"]) == (1, 2)
        assert (c[f"tackle_win_rate_n_{own}"], c[f"tackle_win_rate_den_{own}"]) == (0, 1)
        assert c[f"tackle_out_of_range_per_match_{own}"] == 3
        assert all(c[f"{k}_{other}"] == 0 for k in ("tackle_in_range_rate_n", "tackle_in_range_rate_den", "tackle_win_rate_den", "tackle_out_of_range_per_match"))
        assert (c["tackle_in_range_rate_n_pooled"], c["tackle_in_range_rate_den_pooled"]) == (1, 2)
        assert c["tackle_out_of_range_per_match_pooled"] == 3

    def test_floors_and_rates_over_five_matches(self):
        """[RED] Five copies of the scene: 10 chance spells (the floor is exactly 10), 5 fired, none won -> in-range rate 0.5, win rate
        0, and 3 out-of-reach decisions per match. Four copies (8 spells) give None for all three."""
        per_match = [census.match_metrics(spell_scene())] * 5
        met = census.defence_metrics(per_match)
        assert met["tackle_in_range_rate_A"]["mean"] == pytest.approx(0.5)
        assert met["tackle_win_rate_A"]["mean"] == pytest.approx(0.0)
        assert met["tackle_out_of_range_per_match_A"]["mean"] == pytest.approx(3.0)
        assert met["tackle_in_range_rate_B"]["mean"] is None
        below = census.defence_metrics(per_match[:4])
        assert [below[k]["mean"] for k in ("tackle_in_range_rate_A", "tackle_win_rate_A", "tackle_out_of_range_per_match_A")] == [None, None, None]

    @pytest.mark.parametrize("win_row, won", [(5, 1), (6, 0)])
    def test_a_tackle_wins_when_the_defender_holds_within_3_decisions_of_the_fire(self, win_row, won):
        """[RED] Seat 2 is in a chance spell on rows 0-2 (110 u from the carrier, seat 9) and fires on row 2; the ball is loose from row 3 and
        seat 2 holds it from `win_row`. Row 5 is 3 decisions after the fire: won; row 6 is 4 decisions after: not won."""
        tr = defence_trace(10)
        tr["holder"][0:3], tr["holder"][win_row:] = 9, 2
        tr["holder"][3:win_row] = -1
        tr["pos"][:, 9] = (600.0, 420.0)
        for seat in range(1, 6):
            tr["pos"][:, seat] = (50.0, 420.0 + 10.0 * seat)
        tr["pos"][0:3, 2] = (490.0, 420.0)
        tr["act"][2, 2, 3] = 1.0
        c = census.defence_counts(tr)
        assert (c["tackle_in_range_rate_n_A"], c["tackle_in_range_rate_den_A"]) == (1, 1)
        assert (c["tackle_win_rate_n_A"], c["tackle_win_rate_den_A"]) == (won, 1)

    @pytest.mark.parametrize("carriers, rows, spells", [
        ([9] * 5, (0, 1, 2, 3, 4), 1),
        ([9] * 5, (0, 1, 3, 4), 2),
        ([9, 9, 10, 10, 10], (0, 1, 2, 3, 4), 2),
        ([9, 9, -1, 9, 9], (0, 1, 2, 3, 4), 2),
    ])
    def test_a_spell_is_a_consecutive_run_against_one_carrier(self, carriers, rows, spells):
        """[RED] Seat 1 stands 110 u from the carrier, facing it, on `rows` (far away on the others). One run against one carrier is one
        spell; a row out of reach splits it in two; a change of carrier (seat 9 to seat 10, both at (600, 420) when they hold) does too,
        and so does a loose-ball row between two decisions of the same carrier (-1: nobody holds)."""
        tr = defence_trace(5)
        for t, c in enumerate(carriers):
            tr["holder"][t] = c
            if c >= 0:
                tr["pos"][t, c] = (600.0, 420.0)
            tr["pos"][t, 1] = (490.0, 420.0) if t in rows else (50.0, 430.0)
        assert census.defence_counts(tr)["tackle_in_range_rate_den_A"] == spells

    def test_a_keeper_carrier_and_an_own_team_holder_give_no_spell_and_no_out_of_reach_fire(self):
        """[RED] The carrier is the other team's keeper (seat 6) or a player of the defending team itself: seat 1 stands 110 u away
        facing it and fires, and nothing is counted."""
        for holder in (6, 3):
            tr = defence_trace(4)
            tr["holder"][:] = holder
            tr["pos"][:, holder] = (600.0, 420.0)
            tr["pos"][:, 1] = (490.0, 420.0)
            tr["act"][:, 1, 3] = 1.0
            c = census.defence_counts(tr)
            assert (c["tackle_in_range_rate_den_A"], c["tackle_out_of_range_per_match_A"]) == (0, 0), holder

    def test_a_non_finite_body_angle_removes_that_defender_decision(self):
        """[RED] In `spell_scene` slot 1's body angle is NaN on row 3, its fire row inside its chance rows 2-4: row 3 is no chance row, so the
        run is cut in two (rows 2 and 4) and the spells are 3 (with slot 3's), none fired; the fire on row 3 is neither a spell fire nor an
        out-of-reach fire, so the out-of-reach count stays 3."""
        tr = spell_scene()
        tr["ang"][3, 1] = np.nan
        c = census.defence_counts(tr)
        assert (c["tackle_in_range_rate_n_A"], c["tackle_in_range_rate_den_A"], c["tackle_win_rate_den_A"]) == (0, 3, 0)
        assert c["tackle_out_of_range_per_match_A"] == 3

    def test_a_trace_without_actions_or_facing_enters_no_spell(self):
        """[RED] Without `act` or without `ang` there is nothing to count: zero spells and zero fires."""
        for dropped in ("act", "ang"):
            tr = {k: v for k, v in spell_scene().items() if k != dropped}
            c = census.defence_counts(tr)
            assert c["tackle_in_range_rate_den_pooled"] == 0 and c["tackle_out_of_range_per_match_pooled"] == 0, dropped


class TestDefenceAggregation:
    def test_shot_keys_floor_at_10_shots_and_report_shares_over_shots(self):
        """[RED] One conceded shot per match (facing 1, closing 1): ten matches give 1.0 for both, nine matches give None."""
        m = census.match_metrics(shot_scene())
        met = census.defence_metrics([m] * 10)
        assert (met["shot_nearest_def_facing_B"]["mean"], met["shot_nearest_def_closing_B"]["mean"]) == (pytest.approx(1.0), pytest.approx(1.0))
        assert met["shot_nearest_def_facing_A"]["mean"] is None
        below = census.defence_metrics([m] * 9)
        assert below["shot_nearest_def_facing_B"] == {"mean": None, "se": None, "n": 9}
        assert below["shot_nearest_def_closing_B"]["mean"] is None

    def test_a_floor_of_one_returns_the_single_match_value(self):
        """[RED] With `min_events` 1 the regain scene gives its hand values from one match: at the shot (2 + 0) / 2 = 1, retention 2/3."""
        met = census.defence_metrics([census.match_metrics(regain_scene())], min_events=1)
        assert met["regain_goalside_at_shot_A"]["mean"] == pytest.approx(1.0)
        assert met["regain_goalside_retention_A"]["mean"] == pytest.approx(2 / 3)

    def test_summary_carries_every_defence_key_for_each_group(self):
        """[RED] `summarize` has the ten defence keys with the `_A`, `_B` and `_pooled` suffixes (the per-match count included), each a
        `{mean, se, n}` entry, on a trace that has none of the events."""
        tr = defence_trace(30)
        met = census.summarize([census.match_metrics(tr)])["metrics"]
        keys = [f"{k}_{g}" for k in (*DEFENCE_RATIO_KEYS, "tackle_out_of_range_per_match") for g in ("A", "B", "pooled")]
        assert all(set(met[k]) == {"mean", "se", "n"} for k in keys)

    def test_harness_traces_run_through_the_defence_counts(self, ckpt):
        """[RED] Two real 120-step matches: every count is a finite non-negative number and the summary has no error."""
        traces = harness.run_matches(quick_spec(ckpt, n=2, max_steps=120))
        for t in traces:
            counts = census.defence_counts(t)
            assert counts and all(np.isfinite(v) and v >= 0 for v in counts.values())
        census.summarize([census.match_metrics(t) for t in traces])


class TestDefenceCli:
    def test_the_json_of_the_command_line_carries_the_defence_keys_and_the_counts(self, ckpt, tmp_path):
        """[RED] `census.py DIR --json OUT` on two saved harness traces writes the thirty defence keys to the summary (each `{mean, se, n}`) and
        the 57 counts to every per-match record."""
        traces = harness.run_matches(quick_spec(ckpt, n=2, max_steps=60))
        out = harness.save_traces(traces, tmp_path / "traces", [Path(ckpt)])
        assert census.main([str(out), "--json", str(tmp_path / "census.json")]) == 0
        (payload,) = json.loads((tmp_path / "census.json").read_text(encoding="utf-8")).values()
        metrics = payload["summary"]["metrics"]
        keys = [f"{k}_{g}" for k in (*DEFENCE_RATIO_KEYS, "tackle_out_of_range_per_match") for g in ("A", "B", "pooled")]
        assert len(keys) == 30 and all(set(metrics[k]) == {"mean", "se", "n"} for k in keys)
        assert all(len([k for k in m if k in census.DEFENCE_COUNT_KEYS]) == 57 for m in payload["per_match"])


class TestTrapFlag:
    @staticmethod
    def payload(facing, goals):
        return {"summary": {"n_matches": 96, "metrics": {
            "facing_ball_share_pooled": {"mean": facing, "se": 0.01, "n": 96}, "goals_total": {"mean": goals, "se": 0.1, "n": 96}}}}

    @pytest.mark.parametrize("facing, goals, flag", [(0.35, 0.49, True), (0.351, 0.49, False), (0.35, 0.5, False), (0.0, 0.0, True), (0.33, 0.44, True), (0.47, 1.47, False)])
    def test_rule_is_facing_at_most_035_and_goals_below_05(self, facing, goals, flag):
        """[RED] facing_ball_share_pooled <= 0.35 and goals_total < 0.5: (0.35, 0.49) flags, (0.351, 0.49) and (0.35, 0.5) do not;
        ft43 at generation 50 (0.33, 0.44) flags and ft42 (0.47, 1.47) does not."""
        assert census.trap_flag_g50(self.payload(facing, goals)) is flag

    def test_the_summary_and_the_metrics_dict_are_accepted_too(self):
        """[RED] The census payload, its `summary`, the `summarize` result and the bare metrics mapping all give the same flag."""
        payload = self.payload(0.3, 0.2)
        assert census.trap_flag_g50(payload["summary"]) is True
        assert census.trap_flag_g50(payload["summary"]["metrics"]) is True

    @pytest.mark.parametrize("missing", ["facing_ball_share_pooled", "goals_total"])
    def test_a_missing_or_none_input_raises(self, missing):
        """[RED] A census without one of the two inputs (absent, or a mean of None) cannot be flagged: ValueError, not False."""
        payload = self.payload(0.3, 0.2)
        del payload["summary"]["metrics"][missing]
        with pytest.raises(ValueError, match=missing):
            census.trap_flag_g50(payload)
        payload = self.payload(0.3, 0.2)
        payload["summary"]["metrics"][missing]["mean"] = None
        with pytest.raises(ValueError, match=missing):
            census.trap_flag_g50(payload)


# ----------------------------------------------------------------------------
# Gates: parking guard P1 and the held-out bot target T1h
# ----------------------------------------------------------------------------
def _p1_arms(values: dict) -> dict:
    """`{seed: {gen: deep_block_ge4_pooled}}` as gates input (`self_sampled` censuses with the defaults of `_fresh_arm_seed`)."""
    return {s: _fresh_arm_seed({g: {"deep_block_ge4_pooled": (v, 0.01)} for g, v in gens.items()}) for s, gens in values.items()}


def _evaluate_p1(arm_values: dict, comp_values: dict | None, **kw):
    from tools.audit import gates

    comp = None if comp_values is None else _p1_arms(comp_values)
    return gates.evaluate_gates(_p1_arms(arm_values), comparator=comp, gens=["100", "200", "300"], calibration=_make_fake_calibration(), **kw)


class TestParkingGuardP1:
    # Comparator maximum per generation: 0.10 at 200 (mean 0.07), 0.30 at 300. The limit is the maximum plus 0.05.
    COMP = {"s42": {"100": 0.40, "200": 0.10, "300": 0.30}, "s43": {"100": 0.40, "200": 0.04, "300": 0.20}}

    def test_an_arm_seed_at_the_comparator_maximum_plus_005_passes_and_just_above_fails(self):
        """[RED] Gen 200: comparator seeds 0.10 and 0.04, limit 0.10 + 0.05 = 0.15 (the mean 0.07 would give 0.12). An arm seed at 0.14
        and one at 0.15 pass (a mean-based guard would fail both); 0.1501 fails, with the seed, generation, value and limit named.
        Gen 100 is not evaluated: an arm seed at 0.90 there (comparator limit 0.45) still passes."""
        passed = _evaluate_p1({"s42": {"100": 0.90, "200": 0.14, "300": 0.12}, "s43": {"100": 0.0, "200": 0.15, "300": 0.12}}, self.COMP)
        assert passed["gates"]["P1"]["passed"] is True
        assert passed["gates"]["P1"]["violations"] == []
        assert sorted(passed["gates"]["P1"]["per_gen"]) == ["200", "300"]
        failed = _evaluate_p1({"s42": {"100": 0.0, "200": 0.1501, "300": 0.12}, "s43": {"100": 0.0, "200": 0.0, "300": 0.0}}, self.COMP)
        p1 = failed["gates"]["P1"]
        assert p1["passed"] is False
        assert len(p1["violations"]) == 1
        v = p1["violations"][0]
        assert (v["seed"], v["gen"], v["value"]) == ("s42", "200", pytest.approx(0.1501))
        assert v["limit"] == pytest.approx(0.15)

    def test_any_arm_seed_above_the_limit_at_any_generation_from_200_fails(self):
        """[RED] Gen 300: comparator maximum 0.30 -> limit 0.35. Arm seed s43 at 0.36 fails there while s42 stays inside; only s43 is named."""
        res = _evaluate_p1({"s42": {"100": 0.0, "200": 0.0, "300": 0.35}, "s43": {"100": 0.0, "200": 0.0, "300": 0.36}}, self.COMP)
        p1 = res["gates"]["P1"]
        assert p1["passed"] is False
        assert [(v["seed"], v["gen"]) for v in p1["violations"]] == [("s43", "300")]

    def test_a_failing_parking_guard_has_a_report_section_and_does_not_set_the_exit_status(self, tmp_path):
        """[RED] Four-seed CLI run (gens 200 and 300, `--exit-on-c0`) in which every arm seed parks (0.60 against the comparator's 0.10):
        P1 fails, C0 passes, the exit status stays 0 (only C0, C0L and G6 set it), and the Markdown has a P1 row in the gate table
        and a `## Parking (P1)` section."""
        from tools.audit import gates

        cal_path = tmp_path / "cal.json"
        cal_path.write_text(json.dumps(_make_fake_calibration(gen="300")), encoding="utf-8")
        args = {}
        for name, deep in (("arm", 0.60), ("comp", 0.10)):
            args[name] = []
            for s in ("42", "43", "44", "45"):
                root = tmp_path / name / s
                for g in (200, 300):
                    _write_e6_eval_dir(root, g, extra_census={"deep_block_ge4_pooled": deep})
                args[name].append(f"{s}={root}")
        rc = gates.main(["--arm", *args["arm"], "--comparator", *args["comp"], "--gens", "200", "300", "--protocol", "fresh",
                         "--calibration", str(cal_path), "--exit-on-c0", "--json", str(tmp_path / "out.json"), "--md", str(tmp_path / "out.md")])
        res = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
        assert (res["gates"]["P1"]["passed"], res["gates"]["C0"]["passed"]) == (False, True)
        assert rc == 0
        md = (tmp_path / "out.md").read_text(encoding="utf-8")
        assert "| P1 |" in md and "## Parking (P1)" in md

    def test_without_a_comparator_or_without_the_key_the_guard_is_skipped_and_reported(self):
        """[RED] No comparator: no P1 entry and a note. A comparator but arm censuses without `deep_block_ge4_pooled` (older census files)
        and one that has it only at gen 100: the same."""
        from tools.audit import gates

        for comp in (None, self.COMP):
            notes: list = []
            arm = {s: _fresh_arm_seed({"100": {}, "200": {}, "300": {}}) for s in ("s42", "s43")}
            res = gates.evaluate_gates(arm, comparator=None if comp is None else _p1_arms(comp), gens=["100", "200", "300"],
                                       calibration=_make_fake_calibration(), notes=notes)
            assert "P1" not in res["gates"]
            assert [n for n in notes if n.startswith("P1: not evaluated")]
        notes = []
        only_100 = {"s42": {"100": 0.1, "200": 0.1, "300": 0.1}, "s43": {"100": 0.1, "200": 0.1, "300": 0.1}}
        arm = {s: _fresh_arm_seed({"100": {"deep_block_ge4_pooled": (0.9, 0.01)}, "200": {}, "300": {}}) for s in only_100}
        res = gates.evaluate_gates(arm, comparator=_p1_arms(only_100), gens=["100", "200", "300"], calibration=_make_fake_calibration(), notes=notes)
        assert "P1" not in res["gates"] and [n for n in notes if n.startswith("P1: not evaluated")]

    def test_a_missing_value_in_one_seed_or_in_the_comparator_raises(self):
        """[RED] The key present for seed s42 but absent for s43 at gen 200 raises naming s43; the arm having it and the comparator lacking it
        raises naming the comparator; a None mean (below the census floor) raises too. Nothing is skipped silently."""
        arm = _p1_arms({"s42": {"100": 0.0, "200": 0.1, "300": 0.1}, "s43": {"100": 0.0, "200": 0.1, "300": 0.1}})
        arm["s43"]["200"]["self_sampled"]["summary"]["metrics"].pop("deep_block_ge4_pooled")
        from tools.audit import gates

        comp = _p1_arms(self.COMP)
        with pytest.raises(ValueError, match="s43"):
            gates.evaluate_gates(arm, comparator=comp, gens=["100", "200", "300"], calibration=_make_fake_calibration())
        arm = _p1_arms({"s42": {"100": 0.0, "200": 0.1, "300": 0.1}, "s43": {"100": 0.0, "200": 0.1, "300": 0.1}})
        bare = {s: _fresh_arm_seed({"100": {}, "200": {}, "300": {}}) for s in ("s42", "s43")}
        with pytest.raises(ValueError, match="comparator"):
            gates.evaluate_gates(arm, comparator=bare, gens=["100", "200", "300"], calibration=_make_fake_calibration())
        arm["s42"]["300"]["self_sampled"]["summary"]["metrics"]["deep_block_ge4_pooled"]["mean"] = None
        with pytest.raises(ValueError, match="s42"):
            gates.evaluate_gates(arm, comparator=comp, gens=["100", "200", "300"], calibration=_make_fake_calibration())


def _t1h_arms(values: dict, goals_a: float = 20.0) -> dict:
    """`{seed: ((goals_B, goal_diff) on vs_heldout_sampled, (goals_B, goal_diff) on vs_heldout_sampled_red)}` at gen 300 on top of the
    defaults of `_fresh_arm_seed`; `goals_A` is `goals_a` on both tags."""
    out = {}
    for seed, tags in values.items():
        out[seed] = _fresh_arm_seed({"300": {}})
        for tag, (goals_b, diff) in zip(("vs_heldout_sampled", "vs_heldout_sampled_red"), tags):
            out[seed]["300"][tag] = {"summary": {"n_matches": 48, "metrics": {
                "goals_A": {"mean": goals_a, "se": 0.5, "n": 48}, "goals_B": {"mean": goals_b, "se": 0.3, "n": 48},
                "goal_diff": {"mean": diff, "se": 0.4, "n": 48}}}}
    return out


def _t1h(arm: dict, comp: dict | None, **kw) -> dict:
    from tools.audit import gates

    res = gates.evaluate_gates(arm, comparator=comp, gens=["300"], calibration=_make_fake_calibration(), **kw)
    found = [t for t in res["targets"] if t["id"] == "T1h"]
    return found[0] if found else None


class TestHeldOutTargetT1h:
    COMP = {"s42": ((6.0, -4.0), (4.0, -2.0)), "s43": ((7.0, -5.0), (5.0, -3.0))}   # goals_B per seed 5.0 and 6.0, mean 5.5

    def test_beats_neutral_and_loses_on_goals_conceded(self):
        """[RED] Lower is better, delta 1.0 goal, the seed value being the mean of the two colours' `goals_B`. Comparator seeds 5.0 and 6.0
        (mean 5.5). Arm 3.0 and 3.5 (mean 3.25): separated and 2.25 better -> BEATS. Arm 4.6 and 4.7 (mean 4.65): separated but only 0.85
        better -> NEUTRAL. Arm 7.0 and 7.5: every arm seed above every comparator seed and 1.75 worse -> LOSES. Arm 5.0 and 5.0 against
        comparator 6.0 and 6.0: exactly 1.0 better and separated -> BEATS. The record names the key, tags, direction and delta."""
        comp = _t1h_arms(self.COMP)
        beats = _t1h(_t1h_arms({"s42": ((2.0, 0.0), (4.0, 0.0)), "s43": ((4.0, 0.0), (3.0, 0.0))}), comp)
        assert beats["verdict"] == "BEATS"
        assert (beats["key"], beats["tag"], beats["direction"], beats["delta"], beats["gen"]) == (
            "goals_B", "vs_heldout_sampled+vs_heldout_sampled_red", "lower", 1.0, "300")
        assert beats["arm_per_seed"] == {"s42": 3.0, "s43": 3.5}
        assert beats["comp_per_seed"] == {"s42": 5.0, "s43": 6.0}
        assert beats["mean_diff"] == pytest.approx(3.25 - 5.5)
        assert _t1h(_t1h_arms({"s42": ((4.6, 0.0), (4.6, 0.0)), "s43": ((4.7, 0.0), (4.7, 0.0))}), comp)["verdict"] == "NEUTRAL"
        assert _t1h(_t1h_arms({"s42": ((7.0, 0.0), (7.0, 0.0)), "s43": ((7.5, 0.0), (7.5, 0.0))}), comp)["verdict"] == "LOSES"
        same6 = _t1h_arms({"s42": ((6.0, 0.0), (6.0, 0.0)), "s43": ((6.0, 0.0), (6.0, 0.0))})
        assert _t1h(_t1h_arms({"s42": ((5.0, 0.0), (5.0, 0.0)), "s43": ((5.0, 0.0), (5.0, 0.0))}), same6)["verdict"] == "BEATS"

    def test_the_target_reads_goals_b_not_goals_a(self):
        """[RED] The arm concedes fewer goals (BEATS on `goals_B`) while its `goals_A` (40) is far above the comparator's (10): a verdict taken
        from `goals_A` with the same direction would be LOSES."""
        arm = _t1h_arms({"s42": ((3.0, 0.0), (3.0, 0.0)), "s43": ((3.5, 0.0), (3.5, 0.0))}, goals_a=40.0)
        comp = _t1h_arms(self.COMP, goals_a=10.0)
        assert _t1h(arm, comp)["verdict"] == "BEATS"

    def test_goal_difference_is_reported_beside_the_verdict(self):
        """[RED] The record carries the goal difference of the same tags: per-seed means of the two colours (arm -1.0 and +1.0 -> seed 0.0;
        -2.0 and -4.0 -> -3.0; comparator -4.0 and -2.0 -> -3.0; -5.0 and -3.0 -> -4.0) and the mean difference (-1.5 - (-3.5)) = +2.0."""
        arm = _t1h_arms({"s42": ((3.0, -1.0), (3.0, 1.0)), "s43": ((3.0, -2.0), (3.0, -4.0))})
        rec = _t1h(arm, _t1h_arms(self.COMP))
        gd = rec["goal_diff"]
        assert (gd["arm_per_seed"], gd["comp_per_seed"]) == ({"s42": 0.0, "s43": -3.0}, {"s42": -3.0, "s43": -4.0})
        assert gd["mean_diff"] == pytest.approx(2.0)

    def test_skipped_and_reported_when_the_tags_or_the_comparator_are_absent(self):
        """[RED] Without a comparator, with arm censuses lacking the held-out tags, and with a comparator lacking them: no T1h target and a
        note that starts `T1h: not evaluated`; the other targets are still there."""
        arm = _t1h_arms({"s42": ((3.0, 0.0), (3.0, 0.0)), "s43": ((3.0, 0.0), (3.0, 0.0))})
        bare = {s: _fresh_arm_seed({"300": {}}) for s in ("s42", "s43")}
        from tools.audit import gates

        for a, c in ((arm, None), (bare, _t1h_arms(self.COMP)), (arm, bare)):
            notes: list = []
            res = gates.evaluate_gates(a, comparator=c, gens=["300"], calibration=_make_fake_calibration(), notes=notes)
            assert [t["id"] for t in res["targets"]] == ["T1", "T2", "T3"]
            assert [n for n in notes if n.startswith("T1h: not evaluated")]

    def test_tags_present_for_some_arm_seeds_only_raise(self):
        """[RED] Held-out tags for s42 but not for s43 are an error naming s43, not a skip."""
        from tools.audit import gates

        arm = _t1h_arms({"s42": ((3.0, 0.0), (3.0, 0.0))})
        arm["s43"] = _fresh_arm_seed({"300": {}})
        with pytest.raises(ValueError, match="s43"):
            gates.evaluate_gates(arm, comparator=_t1h_arms(self.COMP), gens=["300"], calibration=_make_fake_calibration())

    def test_the_markdown_lists_the_target_and_its_goal_difference(self):
        """[RED] The Targets table has a T1h row and a line with the goal difference."""
        from tools.audit import gates

        arm = _t1h_arms({"s42": ((3.0, -1.0), (3.0, 1.0)), "s43": ((3.0, -2.0), (3.0, -4.0))})
        res = gates.evaluate_gates(arm, comparator=_t1h_arms(self.COMP), gens=["300"], calibration=_make_fake_calibration())
        md = gates.format_markdown(res)
        assert "T1h (goals_B@vs_heldout_sampled+vs_heldout_sampled_red)" in md
        assert "T1h goal_diff" in md

    def test_the_cli_keeps_stdout_and_prints_skip_notes_on_stderr(self, tmp_path, capsys):
        """[RED] Four seeds without held-out tags and without the parking key: stdout is exactly the Markdown (no new line), stderr names
        both skipped checks."""
        from tools.audit import gates

        cal_path = tmp_path / "cal.json"
        cal_path.write_text(json.dumps(_make_fake_calibration(gen="300")), encoding="utf-8")
        args = {name: [f"{s}={_write_e6_eval_dir(tmp_path / name / s, 300)}" for s in ("42", "43", "44", "45")] for name in ("arm", "comp")}
        assert gates.main(["--arm", *args["arm"], "--comparator", *args["comp"], "--gens", "300", "--protocol", "fresh",
                           "--calibration", str(cal_path), "--md", str(tmp_path / "out.md")]) == 0
        out, err = capsys.readouterr()
        assert out == (tmp_path / "out.md").read_text(encoding="utf-8") + "\n"
        assert "T1h: not evaluated" in err and "P1: not evaluated" in err


# ----------------------------------------------------------------------------
# Kick-conditional aim head (`hybrid6`) in the harness and the aim-control internals
# ----------------------------------------------------------------------------
def write_hybrid6_checkpoint(directory: Path, seed: int = 0, spread: float = 0.3) -> Path:
    """hybrid6 checkpoint (both teams) whose outputs depend on the observation."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    for prefix in ("blue", "red"):
        for kind in ("field", "keeper"):
            net = ActorCritic(policy_head="hybrid6")
            with torch.no_grad():
                net.actor_mean[4].weight.normal_(0.0, spread)
            torch.save(net.state_dict(), directory / f"{prefix}_{kind}_policy.pt")
    return directory


def write_constant_hybrid6_checkpoint(directory: Path, z_kick: float) -> Path:
    """hybrid6 checkpoint with observation-independent outputs: throttle 0.7, steer -0.2, power 0.4, aim 0.6 (pre-tanh),
    special -3."""
    directory.mkdir(parents=True, exist_ok=True)
    for prefix in ("blue", "red"):
        for kind in ("field", "keeper"):
            net = ActorCritic(policy_head="hybrid6")
            with torch.no_grad():
                net.actor_mean[4].weight.zero_()
                net.actor_mean[4].bias.copy_(torch.tensor([0.7, -0.2, 0.4, 0.6, z_kick, -3.0]))
            torch.save(net.state_dict(), directory / f"{prefix}_{kind}_policy.pt")
    return directory


TANH_AIM = 0.5370496  # tanh(0.6)


class TestHybrid6Harness:
    def test_a_hybrid6_team_plays_a_scripted_team_and_records_its_aim_means(self, tmp_path):
        """[RED] CLI, one match of 30 decisions: `head` is [2, 0]; `aim_mean` (30, 12) float32 is finite on team A's seats
        and NaN on the scripted team's; a match without a hybrid6 team has no `aim_mean`."""
        d = write_hybrid6_checkpoint(tmp_path / "c", seed=2)
        out = tmp_path / "out"
        argv = ["--checkpoint-dir", str(d), "--opp", "scripted", "--n", "1", "--seed", "3", "--workers", "1",
                "--max-steps", "30", "--out-dir", str(out)]
        assert harness.main(argv) == 0
        with np.load(out / "m000.npz") as trace:
            assert trace["head"].tolist() == [2, 0]
            aim = trace["aim_mean"]
            assert aim.shape == (30, 12) and aim.dtype == np.float32
            assert np.isfinite(aim[:, :6]).all() and np.isnan(aim[:, 6:]).all()
            assert trace["act"].shape == (30, 12, 4) and np.isfinite(trace["act"]).all()
        h5 = write_head_checkpoint(tmp_path / "h5", {"blue": "hybrid5", "red": "hybrid5"})
        assert "aim_mean" not in harness.play_match(quick_spec(h5, max_steps=5), 0)

    def test_a_firing_holder_kicks_along_the_aim_mean_and_every_other_seat_steers(self, tmp_path):
        """[RED] Constant head, kick logit +3 (p 0.937): greedy fires on every seat; the seat whose observation index 7 is 1
        (slot 2) gets env steer tanh(0.6) = 0.5370496, every other seat tanh(-0.2) = -0.1973753. The recorded steer mean is
        tanh(-0.2) on every seat and the aim mean tanh(0.6). Mixed uses the same two means. With kick logit -3 nobody fires
        and slot 2 steers too."""
        obs, cen = _obs_cen()
        obs[:, 7] = 0.0
        obs[2, 7] = 1.0
        fires = harness.TeamPolicy(str(write_constant_hybrid6_checkpoint(tmp_path / "on", 3.0)), "blue")
        assert fires.head == 2
        expected_steer = [CONST_STEER, CONST_STEER, TANH_AIM, CONST_STEER, CONST_STEER, CONST_STEER]
        aim = np.full(6, np.nan, dtype=np.float32)
        a, _, m = fires.act(obs, cen, "greedy", 1.0, torch.Generator().manual_seed(0), aim_out=aim)
        np.testing.assert_allclose(a[:, 1], expected_steer, atol=1e-6)
        np.testing.assert_allclose(a[:, 2], CONST_KICK_ACTION, atol=1e-6)
        np.testing.assert_allclose(m[:, 1], CONST_STEER, atol=1e-6)
        np.testing.assert_allclose(aim, TANH_AIM, atol=1e-6)
        mixed, _, _ = fires.act(obs, cen, "mixed", 1.0, torch.Generator().manual_seed(1))
        fired = mixed[:, 2] != -1.0
        np.testing.assert_allclose(mixed[fired, 1], np.array(expected_steer)[fired], atol=1e-6)
        silent = harness.TeamPolicy(str(write_constant_hybrid6_checkpoint(tmp_path / "off", -3.0)), "blue")
        a, _, _ = silent.act(obs, cen, "greedy", 1.0, torch.Generator().manual_seed(0))
        np.testing.assert_allclose(a[:, 1], CONST_STEER, atol=1e-6)
        assert (a[:, 2] == -1.0).all()

    def test_sigma_scale_moves_the_four_gaussian_heads(self, tmp_path):
        """[RED] Sampled, kick logit +3: the fired holder's env steer is the aim sample, which sits at the aim mean at scale 0
        and moves away from it by twice as much at scale 2 as at scale 1 (same generator state)."""
        obs, cen = _obs_cen()
        obs[:, 7] = 1.0
        team = harness.TeamPolicy(str(write_constant_hybrid6_checkpoint(tmp_path / "c", 3.0)), "blue")
        zero, one, two = (team.act(obs, cen, "sampled", s, torch.Generator().manual_seed(4))[0] for s in (0.0, 1.0, 2.0))
        fired = (zero[:, 2] != -1.0) & (one[:, 2] != -1.0)
        assert fired.sum() >= 3
        np.testing.assert_allclose(zero[fired, 1], TANH_AIM, atol=1e-6)
        np.testing.assert_allclose((two - zero)[fired][:, [0, 1]], 2.0 * (one - zero)[fired][:, [0, 1]], rtol=1e-4, atol=1e-6)


def aim_control_trace(head, x_from_aim: bool, n_points: int = 12) -> dict:
    """`n_points` aim points of kicker A3 at rows 5, 55, 105, ... (50 apart). Point i: teammate A4 at +17.5 deg when i is even
    and -17.5 deg when odd (need +0.5 / -0.5); the steer mean one row earlier is +0.5 for i // 2 even, -0.5 otherwise; the
    kick's intended steer x_k equals need, stored in `aim_mean` (hybrid6 seat) or in `mean[..., 1]`; the sampled steer is
    x_k + 0.1 for even i and x_k - 0.1 for odd i."""
    T = 50 * n_points + 10
    tr = make_internals_trace(T)
    tr["mean"][:] = 0.0
    tr["act"][:] = 0.0
    tr["head"] = np.array(head, dtype=np.int8)
    if x_from_aim:
        tr["aim_mean"] = np.full((T, 12), np.nan, dtype=np.float32)
        tr["aim_mean"][:, :6] = 0.9                       # a value no point reads
    for i in range(n_points):
        k = 5 + 50 * i
        need = 0.5 if i % 2 == 0 else -0.5
        aim_scene(tr, k, {4: 35.0 * need})
        tr["mean"][k - 1, 3, 1] = 0.5 if (i // 2) % 2 == 0 else -0.5
        if x_from_aim:
            tr["aim_mean"][k, 3] = need
            tr["mean"][k, 3, 1] = -0.7                    # the steer mean, never read for a hybrid6 seat
        else:
            tr["mean"][k, 3, 1] = need
        tr["act"][k, 3, 1] = need + (0.1 if i % 2 == 0 else -0.1)
    return tr


class TestAimControl:
    @staticmethod
    def _hand(n_rep: int = 6):
        m_prev = np.tile([0.5, 0.5, -0.5, -0.5], n_rep)
        need = np.tile([0.5, -0.5, 0.5, -0.5], n_rep)
        return m_prev, need

    def test_stats_on_orthogonal_hand_arrays(self):
        """[RED] 24 points, m_prev and need zero-mean and orthogonal: x_k = need -> r2_prev 0, gain 1; x_k = 0.5 m_prev ->
        r2_prev 1, gain 0. With an intercept, x_k = m_prev on m_prev' = m_prev + 1 (mean 1) still gives r2_prev 1 and gain 0
        (a fit through the origin would give r2_prev 0.36)."""
        from tools.audit import internals

        m_prev, need = self._hand()
        s = internals.aim_control_stats(need, m_prev, need)
        assert s["aim_control_r2_prev"] == pytest.approx(0.0, abs=1e-12)
        assert s["aim_control_r2_gain"] == pytest.approx(1.0, abs=1e-12)
        s = internals.aim_control_stats(0.5 * m_prev, m_prev, need)
        assert s["aim_control_r2_prev"] == pytest.approx(1.0, abs=1e-12)
        assert s["aim_control_r2_gain"] == pytest.approx(0.0, abs=1e-12)
        s = internals.aim_control_stats(m_prev, m_prev + 1.0, need)
        assert s["aim_control_r2_prev"] == pytest.approx(1.0, abs=1e-12)
        assert s["aim_control_r2_gain"] == pytest.approx(0.0, abs=1e-12)

    def test_stats_are_none_below_twenty_points(self):
        """[RED] 19 points give None for both keys; 20 points give numbers."""
        from tools.audit import internals

        m_prev, need = self._hand(n_rep=5)
        s = internals.aim_control_stats(need[:19], m_prev[:19], need[:19])
        assert s == {"aim_control_r2_prev": None, "aim_control_r2_gain": None}
        s = internals.aim_control_stats(need, m_prev, need)
        assert s["aim_control_r2_gain"] == pytest.approx(1.0, abs=1e-12)

    def test_point_extraction_reads_the_aim_mean_of_a_hybrid6_seat(self):
        """[RED] One point at row 5 (kicker A3, A4 at +17.5 deg -> need 0.5); steer mean -0.6 at row 5 and 0.1 at row 4;
        aim mean 0.4 at row 5; sampled steer 0.7. Head [2, 0]: (x_k, m_prev, need, noise) = (0.4, 0.1, 0.5, 0.3).
        Head [1, 0]: x_k is the steer mean -0.6 and the noise 1.3."""
        from tools.audit import internals

        tr = make_internals_trace(20)
        tr["mean"][:] = 0.0
        tr["act"][:] = 0.0
        aim_scene(tr, 5, {4: 17.5})
        tr["mean"][5, 3, 1], tr["mean"][4, 3, 1], tr["act"][5, 3, 1] = -0.6, 0.1, 0.7
        tr["aim_mean"] = np.full((20, 12), np.nan, dtype=np.float32)
        tr["aim_mean"][5, 3] = 0.4
        tr["head"] = np.array([2, 0], dtype=np.int8)
        np.testing.assert_allclose(internals.aim_control_points(tr), [[0.4, 0.1, 0.5, 0.3]], atol=1e-6)
        tr["head"] = np.array([1, 0], dtype=np.int8)
        np.testing.assert_allclose(internals.aim_control_points(tr), [[-0.6, 0.1, 0.5, 1.3]], atol=1e-6)
        del tr["aim_mean"]
        np.testing.assert_allclose(internals.aim_control_points(tr), [[-0.6, 0.1, 0.5, 1.3]], atol=1e-6)

    def test_points_pool_over_matches_in_the_summary_and_the_cli(self, tmp_path, capsys):
        """[RED] Two matches of 12 points each (one hybrid6 team reading `aim_mean`, one hybrid5 team reading the steer mean),
        x_k = need: pooled r2_prev 0, gain 1, noise SD 0.1 over 24 points. Each match alone (12 points) gives None. The CLI
        writes them under `aim_control` beside the summary; the per-match records keep their keys."""
        from tools.audit import internals

        traces = [aim_control_trace([2, 0], True), aim_control_trace([1, 0], False)]
        met = internals.aim_control([internals.aim_control_points(t) for t in traces])
        assert met["aim_control_r2_prev"] == {"mean": pytest.approx(0.0, abs=1e-12), "se": None, "n": 24}
        assert met["aim_control_r2_gain"] == {"mean": pytest.approx(1.0, abs=1e-12), "se": None, "n": 24}
        assert met["kick_steer_noise_sd"] == {"mean": pytest.approx(0.1, abs=1e-6), "se": None, "n": 24}
        alone = internals.aim_control([internals.aim_control_points(traces[0])])
        assert alone["aim_control_r2_gain"]["mean"] is None and alone["aim_control_r2_gain"]["n"] == 12
        assert alone["kick_steer_noise_sd"]["mean"] is None
        d = tmp_path / "t"
        d.mkdir()
        for i, tr in enumerate(traces):
            np.savez_compressed(d / f"m{i:03d}.npz", **tr)
        out_json = tmp_path / "internals.json"
        assert internals.main([str(d), "--json", str(out_json)]) == 0
        assert "aim control" in capsys.readouterr().out
        payload = json.loads(out_json.read_text(encoding="utf-8"))[str(d)]
        assert payload["aim_control"]["aim_control_r2_gain"]["mean"] == pytest.approx(1.0, abs=1e-12)
        assert payload["aim_control"]["aim_control_r2_gain"]["n"] == 24
        assert all("aim_control_r2_gain" not in m for m in payload["per_match"])

    def test_head_code_2_reads_the_fire_columns_like_hybrid5(self):
        """[RED] Kick and special intent, saturation and gradient scale of a head-2 team equal those of a head-1 team."""
        from tools.audit import internals

        tr = TestInternalsHybridHead._trace([1, 0])
        ref = internals.match_internals(tr)
        tr["head"] = np.array([2, 0], dtype=np.int8)
        got = internals.match_internals(tr)
        for key in internals.METRIC_KEYS:
            assert got[key] == ref[key], key


# ----------------------------------------------------------------------------
# Kick-target head (`hybrid7`) in the harness and internals
# ----------------------------------------------------------------------------
def write_hybrid7_checkpoint(directory: Path, seed: int = 0, spread: float = 0.3) -> Path:
    """hybrid7 checkpoint (both teams) whose outputs depend on the observation."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    for prefix in ("blue", "red"):
        for kind in ("field", "keeper"):
            net = ActorCritic(policy_head="hybrid7")
            with torch.no_grad():
                net.actor_mean[4].weight.normal_(0.0, spread)
            torch.save(net.state_dict(), directory / f"{prefix}_{kind}_policy.pt")
    return directory


def write_constant_hybrid7_checkpoint(directory: Path, z_kick: float, target_logits) -> Path:
    """hybrid7 checkpoint with observation-independent outputs: thr 0.7, steer -0.2, power 0.4, aim 0.6,
    kick z_kick, special -3, and 6 target logits."""
    directory.mkdir(parents=True, exist_ok=True)
    bias = [0.7, -0.2, 0.4, 0.6, z_kick, -3.0, *target_logits]
    for prefix in ("blue", "red"):
        for kind in ("field", "keeper"):
            net = ActorCritic(policy_head="hybrid7")
            with torch.no_grad():
                net.actor_mean[4].weight.zero_()
                net.actor_mean[4].bias.copy_(torch.tensor(bias, dtype=torch.float32))
            torch.save(net.state_dict(), directory / f"{prefix}_{kind}_policy.pt")
    return directory


class TestHybrid7Harness:
    def test_a_hybrid7_team_plays_a_scripted_team_and_records_target_fields(self, tmp_path):
        """[RED] CLI, one match of 30 decisions: head is [3, 0]; kick_target (30, 12) int8 is -1 wherever K*c != 1
        and in 0..5 on K*c == 1; free_p (30, 12) float32 is NaN off c rows and in [0, 1] on them; scripted seats
        are -1 and NaN; aim_mean is finite on team A and NaN on scripted team B."""
        d = write_hybrid7_checkpoint(tmp_path / "c", seed=2)
        out = tmp_path / "out"
        argv = ["--checkpoint-dir", str(d), "--opp", "scripted", "--n", "1", "--seed", "3", "--workers", "1",
                "--max-steps", "30", "--out-dir", str(out)]
        assert harness.main(argv) == 0
        with np.load(out / "m000.npz") as trace:
            assert trace["head"].tolist() == [3, 0]
            kt = trace["kick_target"]
            fp = trace["free_p"]
            aim = trace["aim_mean"]
            assert kt.shape == (30, 12) and kt.dtype == np.int8
            assert fp.shape == (30, 12) and fp.dtype == np.float32
            assert aim.shape == (30, 12) and aim.dtype == np.float32
            assert np.isfinite(aim[:, :6]).all() and np.isnan(aim[:, 6:]).all()
            assert (kt[:, 6:] == -1).all() and np.isnan(fp[:, 6:]).all()
            holder = trace["holder"]
            hold_t = trace["hold_t"]
            act_k = trace["act"][:, :6, 2]
            for t in range(30):
                for s in range(6):
                    ready = (int(holder[t]) == s) and (float(hold_t[t]) / 5.0 > 1.0 / 600.0)
                    fired = float(act_k[t, s]) > 0.35
                    if ready:
                        assert np.isfinite(fp[t, s]) and 0.0 <= float(fp[t, s]) <= 1.0
                    else:
                        assert np.isnan(fp[t, s])
                    if ready and fired:
                        assert 0 <= int(kt[t, s]) <= 5
                    else:
                        assert int(kt[t, s]) == -1
        h6 = write_hybrid6_checkpoint(tmp_path / "h6", seed=1)
        tr6 = harness.play_match(quick_spec(h6, max_steps=5), 0)
        assert "aim_mean" in tr6 and "kick_target" not in tr6 and "free_p" not in tr6

    def test_hybrid7_modes_and_sigma_scale_do_not_touch_target(self, tmp_path):
        """[RED] Constant hybrid7: greedy picks masked argmax; mixed samples T and fires with mean Gaussians;
        sigma_scale scales the four Gaussians and leaves T unchanged."""
        obs, cen = _obs_cen()
        obs[:, 7] = 0.0
        obs[:, 8] = 0.0
        # Seat 2 is ready with TM1 (idx 23) = 0.10, TM2 (28) = 0.19, TM3 (33) = 0.20 (out), TM4 (38) = -0.05, TM5 (43) = -0.50 (out)
        obs[2, 7] = 1.0
        obs[2, 8] = 1.0 / 300.0
        obs[2, 23], obs[2, 28], obs[2, 33], obs[2, 38], obs[2, 43] = 0.10, 0.19, 0.20, -0.05, -0.50
        # Seat 3 is ready with all teammates out of arc -> only FREE valid
        obs[3, 7] = 1.0
        obs[3, 8] = 1.0 / 300.0
        obs[3, 23], obs[3, 28], obs[3, 33], obs[3, 38], obs[3, 43] = 0.5, 0.5, 0.5, -0.5, -0.5
        # Seat 4 is holding at hold_t = 0 -> not ready
        obs[4, 7] = 1.0
        obs[4, 8] = 0.0

        d = write_constant_hybrid7_checkpoint(tmp_path / "c", 3.0, [1.0, 0.0, 5.0, 0.0, 5.0, 0.0])
        team = harness.TeamPolicy(str(d), "blue")
        assert team.head == 3
        aim = np.full(6, np.nan, dtype=np.float32)
        kt = np.full(6, -1, dtype=np.int8)
        fp = np.full(6, np.nan, dtype=np.float32)
        a, _, m = team.act(obs, cen, "greedy", 1.0, torch.Generator().manual_seed(0),
                           aim_out=aim, kick_target_out=kt, free_p_out=fp)
        assert kt.tolist() == [-1, -1, 0, 5, -1, -1]
        assert np.isnan(fp[[0, 1, 4, 5]]).all()
        assert float(fp[2]) == pytest.approx(1.0 / (math.e + 3.0), abs=1e-6)
        assert float(fp[3]) == pytest.approx(1.0, abs=1e-6)
        np.testing.assert_allclose(a[2, 1], 0.10 * 180.0 / 35.0, atol=1e-6)
        np.testing.assert_allclose(a[3, 1], TANH_AIM, atol=1e-6)
        np.testing.assert_allclose(a[4, 1], CONST_STEER, atol=1e-6)
        np.testing.assert_allclose(aim, TANH_AIM, atol=1e-6)
        np.testing.assert_allclose(m[:, 1], CONST_STEER, atol=1e-6)

        # sigma_scale does not touch T (same generator seed gives identical kick_target across scales 0, 1, 2).
        kts = []
        acts = []
        for s in (0.0, 1.0, 2.0):
            kt_s = np.full(6, -1, dtype=np.int8)
            act_s, _, _ = team.act(obs, cen, "sampled", s, torch.Generator().manual_seed(7), kick_target_out=kt_s)
            kts.append(kt_s.tolist())
            acts.append(act_s)
        assert kts[0] == kts[1] == kts[2]
        assert (acts[2][3, 1] - acts[0][3, 1]) == pytest.approx(2.0 * (acts[1][3, 1] - acts[0][3, 1]), abs=1e-5)

        # mixed samples T (draws vary across seeds on seat 2) while keeping throttle at its mean.
        mixed_ts = set()
        for seed in range(20):
            kt_m = np.full(6, -1, dtype=np.int8)
            act_m, _, _ = team.act(obs, cen, "mixed", 1.0, torch.Generator().manual_seed(seed), kick_target_out=kt_m)
            np.testing.assert_allclose(act_m[:, 0], math.tanh(0.7), atol=1e-6)
            if kt_m[2] >= 0:
                mixed_ts.add(int(kt_m[2]))
        assert len(mixed_ts) >= 2 and 2 not in mixed_ts and 4 not in mixed_ts


def _hand_hybrid7_trace(n_tm_hit: int, n_tm_other_mate: int, n_tm_opp: int, n_free_with_tm: int, n_free_no_tm: int) -> dict:
    """Hand-built hybrid7 trace (team A = head 3, team B = head 0) with known field kicks by seat 3 (A def/mid)
    plus one keeper kick (seat 0, ignored) and one non-ready kick (kick_target = -1, ignored).
    For seat 3, env._entity_player_slots[3][:5] == [0, 1, 2, 4, 5], so target T = 3 is seat 4."""
    events = (
        [("tm_hit", 3)] * n_tm_hit
        + [("tm_other", 3)] * n_tm_other_mate
        + [("tm_opp", 3)] * n_tm_opp
        + [("free_with_tm", 5)] * n_free_with_tm
        + [("free_no_tm", 5)] * n_free_no_tm
        + [("keeper", 3), ("not_ready", -1)]
    )
    stride = 10
    T = stride * len(events) + 60
    tr = make_internals_trace(T)
    tr["head"] = np.array([3, 0], dtype=np.int8)
    tr["kick_target"] = np.full((T, 12), -1, dtype=np.int8)
    tr["lastkick"] = np.full((T, 12), -999.0, dtype=np.float32)
    tr["s_score"] = np.zeros((T, 2), dtype=np.int32)
    # Default positions: everyone behind kicker (facing 0), so no teammate is in arc unless placed ahead.
    tr["pos"][:] = np.array([200.0, 420.0], dtype=np.float32)
    tr["ang"][:] = 0.0
    tr["holder"][:] = -1

    for idx, (kind, target) in enumerate(events):
        k = 5 + idx * stride
        kicker = 0 if kind == "keeper" else 3
        tr["pos"][k, kicker] = [500.0, 420.0]
        tr["ang"][k, kicker] = 0.0
        tr["holder"][k] = kicker
        tr["lastkick"][k:, kicker] = float(k) / 15.0
        if kind != "not_ready":
            tr["kick_target"][k, kicker] = target
        if kind != "free_no_tm":
            # Put seat 4 (slot 3 for kicker 3; slot 3 for keeper 0 is seat 4 too) directly ahead (0 deg, in arc).
            tr["pos"][k, 4] = [700.0, 420.0]
        if kind == "tm_hit":
            tr["holder"][k + 2] = 4  # targeted teammate receives first
        elif kind == "tm_other":
            tr["holder"][k + 2] = 5  # different teammate receives first
        elif kind == "tm_opp":
            tr["holder"][k + 2] = 9  # opponent receives first
    return tr


class TestHybrid7Internals:
    def test_kick_target_keys_hand_counted_and_threshold(self, tmp_path, capsys):
        """[RED] Hand-counted kick_target_n, free_share and tm_reach:
        12 tm_hit + 4 tm_other + 4 tm_opp (= 20 tm kicks, 12 reached -> 0.60) + 5 free_with_tm (= 25 choice kicks,
        free share 5/25 = 0.20), plus 3 free_no_tm, 1 keeper kick and 1 non-ready kick (all excluded).
        Below 20 kicks (19 total): free_share and tm_reach are None."""
        from tools.audit import internals

        tr_full = _hand_hybrid7_trace(n_tm_hit=12, n_tm_other_mate=4, n_tm_opp=4, n_free_with_tm=5, n_free_no_tm=3)
        res = internals.kick_target_summary([internals.kick_target_counts(tr_full)])
        assert res["kick_target_n"] == 25
        assert res["kick_target_free_share"] == pytest.approx(5.0 / 25.0, abs=1e-12)
        assert res["kick_target_tm_reach"] == pytest.approx(12.0 / 20.0, abs=1e-12)

        tr_small = _hand_hybrid7_trace(n_tm_hit=10, n_tm_other_mate=2, n_tm_opp=2, n_free_with_tm=5, n_free_no_tm=2)
        small = internals.kick_target_summary([internals.kick_target_counts(tr_small)])
        assert small["kick_target_n"] == 19
        assert small["kick_target_free_share"] is None
        assert small["kick_target_tm_reach"] is None

        # CLI writes kick_target in JSON and prints its line; old traces without kick_target work unchanged.
        d = tmp_path / "t"
        d.mkdir()
        np.savez_compressed(d / "m000.npz", **tr_full)
        out_json = tmp_path / "internals.json"
        assert internals.main([str(d), "--json", str(out_json)]) == 0
        assert "kick target" in capsys.readouterr().out
        payload = json.loads(out_json.read_text(encoding="utf-8"))[str(d)]
        assert payload["kick_target"]["kick_target_n"] == 25
        assert payload["kick_target"]["kick_target_free_share"] == pytest.approx(0.20, abs=1e-12)
        assert payload["kick_target"]["kick_target_tm_reach"] == pytest.approx(0.60, abs=1e-12)

        old = make_internals_trace(20)
        assert internals.kick_target_summary([internals.kick_target_counts(old)]) == {
            "kick_target_n": 0, "kick_target_free_share": None, "kick_target_tm_reach": None,
        }

    def test_head_code_3_reads_fire_and_aim_columns_like_hybrid6(self):
        """[RED] Head code 3 matches head code 2 on match_internals and aim_control_points."""
        from tools.audit import internals

        tr = TestInternalsHybridHead._trace([2, 0])
        ref = internals.match_internals(tr)
        tr["head"] = np.array([3, 0], dtype=np.int8)
        got = internals.match_internals(tr)
        for key in internals.METRIC_KEYS:
            assert got[key] == ref[key], key

        tr_aim = make_internals_trace(20)
        tr_aim["mean"][:] = 0.0
        tr_aim["act"][:] = 0.0
        aim_scene(tr_aim, 5, {4: 17.5})
        tr_aim["mean"][5, 3, 1], tr_aim["mean"][4, 3, 1], tr_aim["act"][5, 3, 1] = -0.6, 0.1, 0.7
        tr_aim["aim_mean"] = np.full((20, 12), np.nan, dtype=np.float32)
        tr_aim["aim_mean"][5, 3] = 0.4
        tr_aim["head"] = np.array([3, 0], dtype=np.int8)
        np.testing.assert_allclose(internals.aim_control_points(tr_aim), [[0.4, 0.1, 0.5, 0.3]], atol=1e-6)

