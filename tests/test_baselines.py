"""Specification tests for the scripted baseline opponent (`football_rl/baselines.py`).

Expected actions are hand-solved from the documented bot rules on tiny hand-placed states:
the two outfielders nearest the ball chase it, the other three hold role anchors at
depth {0.20, 0.20, 0.42, 0.42, 0.62} x 1400 + 0.4 x (ball distance from own goal) with
y {300, 540, 200, 640, 420}, the carrier runs at the goal and shoots inside 380 u when it
faces the goal within 0.15 rad, and the keeper holds the goal-line bisector.
Steering is clip(2 x heading error, -1, 1); throttle is 1.0 when farther than 25 u from the
target and within 1.2 rad of facing it, 0.2 when farther than 25 u but facing away, else 0.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from football_rl.baselines import scripted_actions  # noqa: E402
from football_rl.config import DEFAULT_CONFIG  # noqa: E402
from football_rl.env import FootballParallelEnv  # noqa: E402

FIELD_W = 1400.0
FIELD_H = 840.0

THROTTLE, STEER, KICK, SPECIAL = 0, 1, 2, 3


@pytest.fixture(scope="module")
def env():
    e = FootballParallelEnv(config=DEFAULT_CONFIG)
    e.reset(seed=0)
    return e


def far_team(team):
    """Six players parked on a column far from every scenario below (never the nearest to the ball)."""
    return [(1390.0 if team == 1 else 10.0, 40.0 + 120.0 * i) for i in range(6)]


def place(env, a_pos, b_pos=None, a_ang=None, ball=(700.0, 420.0), holder=None, side=(0, 1), b_ang=None):
    """Write an explicit state into the environment arrays the bot reads."""
    b_pos = b_pos if b_pos is not None else far_team(1)
    env.player_pos[:6] = np.asarray(a_pos, dtype=np.float64)
    env.player_pos[6:] = np.asarray(b_pos, dtype=np.float64)
    env.player_angle[:6] = np.zeros(6) if a_ang is None else np.asarray(a_ang, dtype=np.float64)
    env.player_angle[6:] = np.zeros(6) if b_ang is None else np.asarray(b_ang, dtype=np.float64)
    env.ball_pos[:] = ball
    env.ball_holder = holder
    env.team_side[:] = side


# Team A defends the left goal. GK at its bisector point; outfield slots D1, D2, M1, M2, ST.
BASE_A = [
    (45.0, 420.0),    # GK: target (45, clip(420, 360, 480)) -> already there
    (560.0, 300.0),   # D1: anchor x = 280 + 0.4 * 700 = 560, y = 300 -> already there
    (460.0, 540.0),   # D2: anchor (560, 540) is 100 u dead ahead
    (868.0, 250.0),   # M1: anchor x = 588 + 280 = 868, y = 200 is 50 u straight down
    (650.0, 420.0),   # M2: 50 u from the ball (nearest)
    (820.0, 420.0),   # ST: 120 u from the ball (second nearest)
]
BASE_A_ANG = [0.0, 0.0, 0.0, -math.pi / 2, 0.0, math.pi / 2]


class TestShapeAndPurity:
    def test_output_contract(self, env):
        place(env, BASE_A, a_ang=BASE_A_ANG)
        out = scripted_actions(env, 0)
        assert out.shape == (6, 4)
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))
        assert out.min() >= -1.0 and out.max() <= 1.0

    def test_no_rng_and_no_state_mutation(self, env):
        place(env, BASE_A, a_ang=BASE_A_ANG)
        before = (env.player_pos.copy(), env.player_angle.copy(), env.ball_pos.copy(), env.team_side.copy())
        first = scripted_actions(env, 0)
        second = scripted_actions(env, 0)
        assert np.array_equal(first, second)
        assert np.array_equal(before[0], env.player_pos)
        assert np.array_equal(before[1], env.player_angle)
        assert np.array_equal(before[2], env.ball_pos)
        assert np.array_equal(before[3], env.team_side)

    @pytest.mark.parametrize("team", [0, 1])
    def test_bounds_and_finiteness_on_random_states(self, env, team):
        rng = np.random.default_rng(3)
        for _ in range(40):
            pos = np.column_stack([rng.uniform(0, FIELD_W, 12), rng.uniform(0, FIELD_H, 12)])
            ang = rng.uniform(-math.pi, math.pi, 12)
            holder = None if rng.random() < 0.3 else int(rng.integers(0, 12))
            ball = pos[holder] if holder is not None else rng.uniform([0, 0], [FIELD_W, FIELD_H])
            side = (0, 1) if rng.random() < 0.5 else (1, 0)
            place(env, pos[:6], pos[6:], ang[:6], ball=ball, holder=holder, side=side, b_ang=ang[6:])
            out = scripted_actions(env, team)
            assert out.shape == (6, 4)
            assert np.all(np.isfinite(out))
            assert out.min() >= -1.0 and out.max() <= 1.0


class TestHandSolvedActions:
    def test_chasers_anchors_and_keeper(self, env):
        place(env, BASE_A, a_ang=BASE_A_ANG)
        out = scripted_actions(env, 0)
        # [throttle, steer, kick, special] per slot GK, D1, D2, M1, M2, ST
        expected = np.array(
            [
                [0.0, 0.0, -1.0, -1.0],   # GK on its target
                [0.0, 0.0, -1.0, -1.0],   # D1 on its anchor (distance 0 -> stops)
                [1.0, 0.0, -1.0, -1.0],   # D2: 100 u from anchor, facing it
                [1.0, 0.0, -1.0, -1.0],   # M1: 50 u from anchor, facing it
                [1.0, 0.0, -1.0, -1.0],   # M2 chaser: ball 50 u dead ahead
                [0.2, 1.0, -1.0, -1.0],   # ST chaser: ball behind (error pi/2 > 1.2 -> throttle 0.2, steer saturates)
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_steering_is_twice_heading_error_below_saturation(self, env):
        # M2 chaser faces 0.3 rad to the left of the ball bearing (bearing 0, angle +0.3): error -0.3 -> steer -0.6
        ang = list(BASE_A_ANG)
        ang[4] = 0.3
        place(env, BASE_A, a_ang=ang)
        out = scripted_actions(env, 0)
        assert out[4, STEER] == pytest.approx(-0.6, abs=1e-6)
        assert out[4, THROTTLE] == pytest.approx(1.0)

    def test_carrier_shoots_inside_380_when_aligned(self, env):
        a = list(BASE_A)
        a[5] = (1100.0, 420.0)           # 300 u from the right goal centre (1400, 420)
        place(env, a, a_ang=[0.0] * 6, ball=a[5], holder=5)
        out = scripted_actions(env, 0)
        np.testing.assert_allclose(out[5], [1.0, 0.0, 1.0, -1.0], atol=1e-6)

    @pytest.mark.parametrize(
        "x, ang, kicks",
        [
            (1020.0, 0.0, False),   # exactly 380 u from the goal centre: not inside
            (1021.0, 0.0, True),    # 379 u
            (900.0, 0.0, False),    # 500 u
            (1100.0, 0.16, False),  # error 0.16 rad -> outside 0.15
            (1100.0, 0.14, True),
            (1100.0, -0.14, True),
        ],
    )
    def test_carrier_kick_thresholds(self, env, x, ang, kicks):
        a = list(BASE_A)
        a[5] = (x, 420.0)
        place(env, a, a_ang=[0.0, 0.0, 0.0, 0.0, 0.0, ang], ball=a[5], holder=5)
        out = scripted_actions(env, 0)
        assert (out[5, KICK] == 1.0) == kicks
        assert out[5, KICK] in (-1.0, 1.0)

    def test_misaligned_carrier_steers_without_kicking(self, env):
        a = list(BASE_A)
        a[5] = (1100.0, 420.0)
        place(env, a, a_ang=[0.0, 0.0, 0.0, 0.0, 0.0, 0.3], ball=a[5], holder=5)
        out = scripted_actions(env, 0)
        np.testing.assert_allclose(out[5], [1.0, -0.6, -1.0, -1.0], atol=1e-6)

    def test_keeper_goes_to_a_loose_ball_inside_120_of_its_goal(self, env):
        place(env, BASE_A, a_ang=BASE_A_ANG, ball=(100.0, 420.0))
        out = scripted_actions(env, 0)
        # GK target is the ball: 55 u dead ahead at angle 0
        np.testing.assert_allclose(out[0], [1.0, 0.0, -1.0, -1.0], atol=1e-6)

    def test_keeper_ignores_a_loose_ball_beyond_120_of_its_goal(self, env):
        place(env, BASE_A, a_ang=BASE_A_ANG, ball=(121.0, 420.0))
        out = scripted_actions(env, 0)
        np.testing.assert_allclose(out[0], [0.0, 0.0, -1.0, -1.0], atol=1e-6)

    def test_keeper_bisector_follows_ball_y_clamped(self, env):
        # Ball y = 700 is clamped to 480: GK at (45, 420) must walk +y (heading error +pi/2 -> steer 1, throttle 0.2 for 60 u)
        place(env, BASE_A, a_ang=BASE_A_ANG, ball=(700.0, 700.0))
        out = scripted_actions(env, 0)
        np.testing.assert_allclose(out[0], [0.2, 1.0, -1.0, -1.0], atol=1e-6)

    def test_keeper_with_ball_kicks_only_when_facing_downfield(self, env):
        a = list(BASE_A)
        place(env, a, a_ang=[0.0] * 6, ball=a[0], holder=0)
        assert scripted_actions(env, 0)[0, KICK] == 1.0
        place(env, a, a_ang=[0.5] + [0.0] * 5, ball=a[0], holder=0)
        assert scripted_actions(env, 0)[0, KICK] == -1.0

    def test_special_only_for_chasers_within_45_of_an_opponent_holder(self, env):
        b = far_team(1)
        b[0] = (700.0, 420.0)             # opponent keeper slot used as the holder (global index 6)
        a = list(BASE_A)
        a[4] = (656.0, 420.0)             # 44 u from the holder, a chaser
        place(env, a, b, a_ang=BASE_A_ANG, ball=(700.0, 420.0), holder=6)
        assert scripted_actions(env, 0)[4, SPECIAL] == 1.0
        a[4] = (655.0, 420.0)             # exactly 45 u: not inside
        place(env, a, b, a_ang=BASE_A_ANG, ball=(700.0, 420.0), holder=6)
        assert scripted_actions(env, 0)[4, SPECIAL] == -1.0

    def test_non_chaser_never_uses_special(self, env):
        b = far_team(1)
        b[0] = (560.0, 330.0)            # opponent holder (global 6) 30 u from D1
        a = list(BASE_A)
        a[4] = (555.0, 335.0)            # M2: 7.1 u from the ball -> chaser
        a[5] = (565.0, 322.0)            # ST: 9.4 u from the ball -> chaser
        place(env, a, b, a_ang=[0.0] * 6, ball=(560.0, 330.0), holder=6)
        out = scripted_actions(env, 0)
        assert out[4, SPECIAL] == 1.0 and out[5, SPECIAL] == 1.0
        assert out[1, SPECIAL] == -1.0   # D1 is within 45 u but is not a chaser
        assert out[0, SPECIAL] == -1.0


class TestMirrorSymmetry:
    @staticmethod
    def mirrored_state(pos, ang, ball, holder):
        """Point-reflect the pitch and swap the two teams' index blocks."""
        pos2 = np.empty_like(pos)
        ang2 = np.empty_like(ang)
        for i in range(12):
            j = (i + 6) % 12
            pos2[j] = (FIELD_W - pos[i, 0], FIELD_H - pos[i, 1])
            ang2[j] = ang[i] + math.pi
        ball2 = np.array([FIELD_W - ball[0], FIELD_H - ball[1]])
        holder2 = None if holder is None else (holder + 6) % 12
        return pos2, ang2, ball2, holder2

    @pytest.mark.parametrize("side", [(0, 1), (1, 0)])
    def test_team_one_on_the_mirrored_state_equals_team_zero(self, env, side):
        rng = np.random.default_rng(11)
        for _ in range(30):
            pos = np.column_stack([rng.uniform(20, FIELD_W - 20, 12), rng.uniform(20, FIELD_H - 20, 12)])
            ang = rng.uniform(-math.pi, math.pi, 12)
            holder = None if rng.random() < 0.4 else int(rng.integers(0, 12))
            ball = pos[holder].copy() if holder is not None else rng.uniform([20, 20], [FIELD_W - 20, FIELD_H - 20])
            place(env, pos[:6], pos[6:], ang[:6], ball=ball, holder=holder, side=side, b_ang=ang[6:])
            reference = scripted_actions(env, 0)
            pos2, ang2, ball2, holder2 = self.mirrored_state(pos, ang, ball, holder)
            place(env, pos2[:6], pos2[6:], ang2[:6], ball=ball2, holder=holder2, side=side, b_ang=ang2[6:])
            mirrored = scripted_actions(env, 1)
            np.testing.assert_allclose(mirrored, reference, atol=1e-5)

    def test_second_half_flip_changes_attack_direction(self, env):
        # With sides swapped, team A attacks the left goal: a carrier at x = 300 facing pi shoots.
        a = list(BASE_A)
        a[5] = (300.0, 420.0)
        place(env, a, a_ang=[0.0, 0.0, 0.0, 0.0, 0.0, math.pi], ball=a[5], holder=5, side=(1, 0))
        out = scripted_actions(env, 0)
        assert out[5, KICK] == 1.0
        place(env, a, a_ang=[0.0, 0.0, 0.0, 0.0, 0.0, math.pi], ball=a[5], holder=5, side=(0, 1))
        assert scripted_actions(env, 0)[5, KICK] == -1.0


class TestBotVersusBot:
    def test_two_bots_are_statistically_symmetric(self):
        from tools.audit import harness

        spec = harness.EvalSpec(opponent="scripted2", mode="sampled", n=8, seed=21, workers=4)
        traces = harness.run_matches(spec)
        a = np.array([float(t["final_score"][0]) for t in traces])
        b = np.array([float(t["final_score"][1]) for t in traces])
        se_diff = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        assert abs(a.mean() - b.mean()) < 3.0 * se_diff
        assert a.mean() + b.mean() > 0.0
