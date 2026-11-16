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
        self_ov = {"kick_outcome_teammate": (kot, 0.01), "gk_in_box_pooled": (0.99, 0.002), "gk_goal_line_pooled": (0.175, 0.01), **ov}
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

    def test_resume150_and_fresh300_unchanged_with_missing_data(self):
        """resume150 and fresh300 keep their missing-data behaviour unchanged."""
        from tools.audit import gates

        arm = {"s42": {"300": {"self_sampled": _fake_census_point(hold_share_pooled=(None, 0.0))}}}
        res_f300 = gates.evaluate_gates(arm, gens=["200", "300"], protocol="fresh300", targets=["goal_diff@vs_frozen:higher:0.5"])
        assert res_f300["gates"]["G1"]["metrics"]["hold_share_pooled"]["passed"] is None
        assert res_f300["targets"][0]["verdict"] == "NEUTRAL"

        res_r150 = gates.evaluate_gates(arm, gens=["300", "1200"], protocol="resume150", targets=["goal_diff@vs_frozen:higher:0.5"])
        assert res_r150["gates"]["G1"]["metrics"]["hold_share_pooled"]["passed"] is None
        assert res_r150["targets"][0]["verdict"] == "NEUTRAL"

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

    def test_fresh300_and_resume150_have_no_learning_floor(self):
        """[GUARD] The older protocols carry neither C0L, G6 nor a primary target."""
        from tools.audit import gates

        arm = {"42": {"300": {"self_sampled": _fake_census_point()}}}
        for protocol in ("fresh300", "resume150"):
            res = gates.evaluate_gates(arm, gens=["300"], protocol=protocol)
            assert "C0L" not in res["gates"] and "G6" not in res["gates"] and "primary_target" not in res


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
        assert "    fire_sat.kick  blue_field=0.030 blue_keeper=0.130 red_field=0.230 red_keeper=0.330" in lines
        assert "    fire_sat.spec  blue_field=0.530 blue_keeper=0.630 red_field=0.730 red_keeper=0.830" in lines
        assert "    p_kick         blue_field=0.215 blue_keeper=0.315 red_field=0.415 red_keeper=0.515" in lines
        assert "    p_spec         blue_field=0.585 blue_keeper=0.485 red_field=0.385 red_keeper=0.285" in lines
        assert "  fire gens     3-    4 (n=2)" in lines
        assert "    fire_sat.kick  blue_field=0.070 blue_keeper=0.170 red_field=0.270 red_keeper=0.370" in lines
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

    def test_histories_with_other_protocols_are_an_error(self):
        from tools.audit import gates

        arm = {"42": {"300": {"self_sampled": _fake_census_point()}}}
        with pytest.raises(ValueError, match="history|histories"):
            gates.evaluate_gates(arm, gens=["300"], protocol="fresh300", histories={"42": _takeoff_history(37)})

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

    @pytest.mark.parametrize("protocol", ["fresh300", "resume150"])
    def test_flag_with_other_protocols_is_an_error(self, tmp_path, protocol):
        """[RED] --exit-on-c0 belongs to the fresh protocol only."""
        from tools.audit import gates

        _write_e3_anchor_dir(tmp_path / "arm" / "42", "300", self_overrides=_FRESH_SELF_BASE)
        with pytest.raises(ValueError, match="exit-on-c0"):
            gates.main(["--arm", f"42={tmp_path}/arm/42", "--protocol", protocol, "--gens", "300", "--exit-on-c0"])


class TestFreshCliFlags:
    @pytest.mark.parametrize("protocol", ["fresh300", "resume150"])
    def test_calibration_with_other_protocols_is_an_error(self, tmp_path, protocol):
        """[RED] --calibration is only read by the fresh protocol, so giving it with another one is an error."""
        from tools.audit import gates

        cal_path = tmp_path / "cal.json"
        cal_path.write_text(json.dumps(_make_fake_calibration()), encoding="utf-8")
        _write_e3_anchor_dir(tmp_path / "arm" / "42", "300", self_overrides=_FRESH_SELF_BASE)
        with pytest.raises(ValueError, match="calibration"):
            gates.main(["--arm", f"42={tmp_path}/arm/42", "--protocol", protocol, "--gens", "300", "--calibration", str(cal_path)])

    @pytest.mark.parametrize("protocol", ["fresh300", "resume150"])
    def test_history_with_other_protocols_is_an_error(self, tmp_path, protocol):
        """[RED] --history is only read by the fresh protocol."""
        from tools.audit import gates

        _write_e3_anchor_dir(tmp_path / "arm" / "42", "300", self_overrides=_FRESH_SELF_BASE)
        h = _write_history(tmp_path / "h.json", _takeoff_history(37))
        with pytest.raises(ValueError, match="history"):
            gates.main(["--arm", f"42={tmp_path}/arm/42", "--protocol", protocol, "--gens", "300", "--history", f"42={h}"])


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
