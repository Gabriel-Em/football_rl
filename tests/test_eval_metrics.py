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
