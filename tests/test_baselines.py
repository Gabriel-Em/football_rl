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

from football_rl.baselines import passing_actions, scripted_actions  # noqa: E402
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


# ----------------------------------------------------------------------------
# Held-out passing bot
# ----------------------------------------------------------------------------
# Hand-solved reference values (team A, canonical frame):
#   bearing from (500, 420) to (800, 620) is atan(200 / 300) = 0.5880026 rad; the pass is sqrt(300^2 + 200^2) = 360.5551 u
#   long, so its kick value is 0.35 + 0.65 * (360.5551 / 600) = 0.7406014.
BEARING_TO_OPEN = 0.5880026
KICK_360 = 0.7406014


def pass_scene(env, carrier=(500.0, 420.0), ang=0.0, slot=3, receivers=None, opponents=None):
    """Team A's `slot` holds the ball at `carrier`. `receivers` maps teammate slots, `opponents` team B slots, to positions;
    everybody else is parked on a far column (teammates behind the carrier, opponents beyond x = 1390)."""
    a = far_team(0)
    a[slot] = carrier
    for s, p in (receivers or {}).items():
        a[s] = p
    b = far_team(1)
    for s, p in (opponents or {}).items():
        b[s] = p
    angles = [0.0] * 6
    angles[slot] = ang
    place(env, a, b, a_ang=angles, ball=carrier, holder=slot)
    return slot


def pass_rich_state(rng):
    """Random state in which an outfielder of team A holds the ball with an opposing outfielder 40-150 u away."""
    pos = np.column_stack([rng.uniform(20, FIELD_W - 20, 12), rng.uniform(20, FIELD_H - 20, 12)])
    ang = rng.uniform(-math.pi, math.pi, 12)
    holder = int(rng.integers(1, 6))
    heading = rng.uniform(0.0, 2.0 * math.pi)
    radius = rng.uniform(40.0, 150.0)
    presser = pos[holder] + radius * np.array([math.cos(heading), math.sin(heading)])
    pos[int(rng.integers(7, 12))] = np.clip(presser, [20, 20], [FIELD_W - 20, FIELD_H - 20])
    return pos, ang, holder


class TestPassingActions:
    def test_output_contract_and_purity(self, env):
        rng = np.random.default_rng(5)
        for _ in range(30):
            pos, ang, holder = pass_rich_state(rng)
            place(env, pos[:6], pos[6:], ang[:6], ball=pos[holder], holder=holder, b_ang=ang[6:])
            before = (env.player_pos.copy(), env.player_angle.copy(), env.ball_pos.copy(), env.team_side.copy())
            first = passing_actions(env, 0)
            second = passing_actions(env, 0)
            assert first.shape == (6, 4) and first.dtype == np.float32
            assert np.all(np.isfinite(first)) and first.min() >= -1.0 and first.max() <= 1.0
            assert np.array_equal(first, second)
            for b, now in zip(before, (env.player_pos, env.player_angle, env.ball_pos, env.team_side)):
                assert np.array_equal(b, now)

    @pytest.mark.parametrize(
        "ang, steer, kick",
        [
            (0.3, 0.5760052, -1.0),                       # error 0.288: turning, no kick
            (BEARING_TO_OPEN - 0.14, 0.28, KICK_360),     # aligned within 0.15 on the right: kicks
            (BEARING_TO_OPEN + 0.14, -0.28, KICK_360),    # aligned on the left: kicks
            (BEARING_TO_OPEN - 0.16, 0.32, -1.0),         # 0.16 rad off: no kick
            (BEARING_TO_OPEN + 0.16, -0.32, -1.0),
        ],
    )
    def test_pressed_carrier_steers_to_the_open_teammate_and_kicks_once_aligned(self, env, ang, steer, kick):
        # Presser B1 100 u ahead of the carrier. ST (slot 5) at (800, 620) is open; M2 (slot 4) at (900, 420) has the larger
        # gain (400 against 300) but B2 stands 30 u from it (and B1 on its pass line). The ST pass line passes 55.5 u from B1.
        slot = pass_scene(env, ang=ang, receivers={5: (800.0, 620.0), 4: (900.0, 420.0)}, opponents={1: (600.0, 420.0), 2: (900.0, 450.0)})
        np.testing.assert_allclose(passing_actions(env, 0)[slot], [1.0, steer, kick, -1.0], atol=1e-6)

    @pytest.mark.parametrize(
        "carrier_x, receiver_x, kick",
        [
            (500.0, 600.0, 0.545),     # 100 u: 100 / 600 is below 0.3, so 0.35 + 0.65 * 0.3
            (500.0, 800.0, 0.675),     # 300 u: 0.35 + 0.65 * 0.5
            (500.0, 1040.0, 0.935),    # 540 u: 0.35 + 0.65 * 0.9
            (200.0, 1100.0, 1.0),      # 900 u: 900 / 600 is above 1, so 0.35 + 0.65 * 1
        ],
    )
    def test_pass_power_follows_the_pass_length_between_the_clip_bounds(self, env, carrier_x, receiver_x, kick):
        assert DEFAULT_CONFIG.mechanics.kick_trigger_threshold == 0.35
        slot = pass_scene(env, carrier=(carrier_x, 420.0), receivers={5: (receiver_x, 420.0)}, opponents={1: (carrier_x, 520.0)})
        np.testing.assert_allclose(passing_actions(env, 0)[slot], [1.0, 0.0, kick, -1.0], atol=1e-6)

    @pytest.mark.parametrize(
        "name, opponents, expected",
        [
            ("both open: the larger gain wins", {}, "far"),
            ("receiver marker exactly 40 u away is not inside", {2: (900.0, 460.0)}, "far"),
            ("receiver marker 39 u away marks it", {2: (900.0, 459.0)}, "alt"),
            ("line blocker exactly 25 u from the pass line is not inside", {2: (700.0, 445.0)}, "far"),
            ("line blocker 24 u from the pass line blocks it", {2: (700.0, 444.0)}, "alt"),
            ("both receivers marked: dribble", {2: (900.0, 459.0), 3: (800.0, 339.0)}, "dribble"),
            ("opponent behind the carrier on the pass line does not block", {2: (400.0, 420.0)}, "far"),
            ("opponent 100 u beyond the receiver on the pass line does not block", {2: (1000.0, 420.0)}, "far"),
        ],
    )
    def test_the_target_must_be_open(self, env, name, opponents, expected):
        # Carrier (500, 420) facing 0, pressed by B1 100 u to its side. FAR = M2 at (900, 420): gain 400, bearing 0, 400 u
        # (kick 0.35 + 0.65 * 400 / 600 = 0.7833333). ALT = ST at (800, 300): gain 300, bearing -atan(0.4) = -0.3805064.
        # The dribble steers at the goal (bearing 0) and cannot kick from 900 u.
        rows = {
            "far": [1.0, 0.0, 0.7833333, -1.0],
            "alt": [1.0, -0.7610128, -1.0, -1.0],
            "dribble": [1.0, 0.0, -1.0, -1.0],
        }
        slot = pass_scene(env, receivers={4: (900.0, 420.0), 5: (800.0, 300.0)}, opponents={1: (500.0, 520.0), **opponents})
        np.testing.assert_allclose(passing_actions(env, 0)[slot], rows[expected], atol=1e-6, err_msg=name)

    @pytest.mark.parametrize("receiver", [(400.0, 300.0), (500.0, 300.0)])
    def test_a_teammate_level_with_or_behind_the_carrier_is_not_a_target(self, env, receiver):
        slot = pass_scene(env, receivers={4: receiver}, opponents={1: (500.0, 520.0)})
        np.testing.assert_allclose(passing_actions(env, 0)[slot], [1.0, 0.0, -1.0, -1.0], atol=1e-6)

    @pytest.mark.parametrize(
        "presser_slot, presser, passes",
        [
            (1, (500.0, 660.0), False),    # exactly 240 u, level with the carrier
            (1, (500.0, 659.0), True),     # 239 u
            (1, (692.0, 676.0), False),    # exactly 320 u (192-256-320), goal-side
            (1, (692.0, 675.0), True),     # 319.2 u, goal-side
            (1, (308.0, 675.0), False),    # 319.2 u but behind the carrier
            (0, (500.0, 500.0), False),    # the opposing keeper (80 u) does not press
        ],
    )
    def test_the_carrier_passes_only_when_pressed(self, env, presser_slot, presser, passes):
        # The open target is ST at (800, 620): bearing 0.5880026, 0.3 rad off the carrier's facing (steer +0.5760052).
        # Dribbling at the goal steers -0.6.
        slot = pass_scene(env, ang=0.3, receivers={5: (800.0, 620.0)}, opponents={presser_slot: presser})
        expected = [1.0, 0.5760052, -1.0, -1.0] if passes else [1.0, -0.6, -1.0, -1.0]
        np.testing.assert_allclose(passing_actions(env, 0)[slot], expected, atol=1e-6)

    def test_an_unpressed_carrier_plays_the_training_bot_row(self, env):
        slot = pass_scene(env, ang=0.3, receivers={5: (800.0, 620.0)})
        row = passing_actions(env, 0)[slot]
        np.testing.assert_allclose(row, [1.0, -0.6, -1.0, -1.0], atol=1e-6)
        assert np.array_equal(row, scripted_actions(env, 0)[slot])

    def test_a_carrier_inside_shooting_range_shoots_when_aligned_and_passes_when_not(self, env):
        # Carrier (1100, 420), presser B1 100 u beside it, open ST at (1200, 520): bearing pi / 4 = 0.7853982.
        scene = dict(carrier=(1100.0, 420.0), receivers={5: (1200.0, 520.0)}, opponents={1: (1100.0, 520.0)})
        slot = pass_scene(env, ang=0.0, **scene)
        np.testing.assert_allclose(passing_actions(env, 0)[slot], [1.0, 0.0, 1.0, -1.0], atol=1e-6)
        pass_scene(env, ang=0.3, **scene)
        np.testing.assert_allclose(passing_actions(env, 0)[slot], [1.0, 0.9707963, -1.0, -1.0], atol=1e-6)

    @pytest.mark.parametrize("ang, kick", [(0.0, 1.0), (0.5, -1.0)])
    def test_a_keeper_holding_the_ball_plays_the_training_bot_row(self, env, ang, kick):
        pass_scene(env, carrier=(45.0, 420.0), ang=ang, slot=0, receivers={4: (600.0, 420.0)}, opponents={1: (45.0, 520.0)})
        row = passing_actions(env, 0)[0]
        assert row[2] == kick
        assert np.array_equal(row, scripted_actions(env, 0)[0])

    def test_defence_and_keeper_rows_without_the_ball(self, env):
        # Same hand-solved block as the training bot's chaser / anchor / keeper test.
        place(env, BASE_A, a_ang=BASE_A_ANG)
        expected = np.array(
            [
                [0.0, 0.0, -1.0, -1.0],
                [0.0, 0.0, -1.0, -1.0],
                [1.0, 0.0, -1.0, -1.0],
                [1.0, 0.0, -1.0, -1.0],
                [1.0, 0.0, -1.0, -1.0],
                [0.2, 1.0, -1.0, -1.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(passing_actions(env, 0), expected, atol=1e-6)

    def test_non_chasing_attackers_push_up_and_wide_while_the_team_holds_the_ball(self, env):
        # M2 (slot 4) holds at (650, 420); the chasers are M2 and D1 (150 u from the ball); D2, M1 and ST hold anchors.
        # Ball x 650 gives a follow term of 0.4 * 650 = 260. D2: (540, 540), unchanged. M1: base (848, 200) moves +168 forward and
        # 120 away from y = 420 to (1016, 80). ST: base (1128, 420) moves forward to (1296, 420) and, being on the centre line,
        # not sideways.
        a = [(45.0, 420.0), (560.0, 300.0), (460.0, 540.0), (868.0, 228.0), (650.0, 420.0), (1290.0, 420.0)]
        ang = [0.0, 0.0, 0.0, -math.pi / 4 + 0.2, 0.0, 0.0]
        place(env, a, far_team(1), a_ang=ang, ball=a[4], holder=4)
        expected = np.array(
            [
                [0.0, 0.0, -1.0, -1.0],      # GK on its bisector point
                [1.0, 1.0, -1.0, -1.0],      # D1 chases: bearing atan(4/3) = 0.927 -> steer saturates
                [1.0, 0.0, -1.0, -1.0],      # D2: 80 u dead ahead
                [1.0, -0.4, -1.0, -1.0],     # M1: target dx 148, dy -148 -> bearing -pi/4, error -0.2
                [1.0, 0.0, -1.0, -1.0],      # M2 carries toward the goal, 750 u away
                [0.0, 0.0, -1.0, -1.0],      # ST: 6 u from its shifted anchor
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(passing_actions(env, 0), expected, atol=1e-6)

    def test_a_chasing_attacker_keeps_chasing_the_ball_while_the_team_holds_it(self, env):
        # M2 holds at (650, 420); ST (700, 420) is the second-nearest outfielder (50 u), so it chases the ball 50 u behind it:
        # error -pi, throttle 0.2, steer -1. A forward-shifted target (x 818) would be dead ahead.
        a = [(45.0, 420.0), (560.0, 300.0), (460.0, 540.0), (868.0, 228.0), (650.0, 420.0), (700.0, 420.0)]
        place(env, a, far_team(1), a_ang=[0.0] * 6, ball=a[4], holder=4)
        np.testing.assert_allclose(passing_actions(env, 0)[5], [0.2, -1.0, -1.0, -1.0], atol=1e-6)

    def test_the_attackers_also_push_up_while_the_keeper_holds_the_ball(self, env):
        # Keeper holds at (45, 420); the chasers are D1 and D2 (196 u from the ball). ST's anchor: 0.62 * 1400 + 0.4 * 45 = 886,
        # moved forward to 1054, where ST stands (throttle 0). Unmoved, the anchor would be 168 u behind it.
        a = [(45.0, 420.0), (200.0, 300.0), (200.0, 540.0), (868.0, 228.0), (650.0, 420.0), (1054.0, 420.0)]
        place(env, a, far_team(1), a_ang=[0.0] * 6, ball=a[0], holder=0)
        np.testing.assert_allclose(passing_actions(env, 0)[5], [0.0, 0.0, -1.0, -1.0], atol=1e-6)

    def test_the_same_attackers_hold_the_training_anchors_when_the_ball_is_loose(self, env):
        a = [(45.0, 420.0), (560.0, 300.0), (460.0, 540.0), (868.0, 228.0), (650.0, 420.0), (1290.0, 420.0)]
        ang = [0.0, 0.0, 0.0, -math.pi / 4 + 0.2, 0.0, 0.0]
        place(env, a, far_team(1), a_ang=ang, ball=(650.0, 420.0), holder=None)
        out = passing_actions(env, 0)
        # M1 anchor (848, 200) is behind and above it (error -1.6056 -> throttle 0.2, steer -1); ST anchor (1128, 420) is
        # directly behind it (error -pi).
        np.testing.assert_allclose(out[3], [0.2, -1.0, -1.0, -1.0], atol=1e-6)
        np.testing.assert_allclose(out[5], [0.2, -1.0, -1.0, -1.0], atol=1e-6)

    @pytest.mark.parametrize("team", [0, 1])
    def test_rows_equal_the_training_bots_whenever_the_team_does_not_hold_the_ball(self, env, team):
        rng = np.random.default_rng(17 + team)
        checked = 0
        for _ in range(120):
            pos = np.column_stack([rng.uniform(0, FIELD_W, 12), rng.uniform(0, FIELD_H, 12)])
            ang = rng.uniform(-math.pi, math.pi, 12)
            others = [i for i in range(12) if i // 6 != team]
            holder = None if rng.random() < 0.4 else int(rng.choice(others))
            ball = pos[holder] if holder is not None else rng.uniform([0, 0], [FIELD_W, FIELD_H])
            side = (0, 1) if rng.random() < 0.5 else (1, 0)
            place(env, pos[:6], pos[6:], ang[:6], ball=ball, holder=holder, side=side, b_ang=ang[6:])
            assert np.array_equal(passing_actions(env, team), scripted_actions(env, team))
            checked += 1
        assert checked == 120

    def test_keeper_and_defender_rows_equal_the_training_bots_even_with_the_ball(self, env):
        rng = np.random.default_rng(23)
        checked = 0
        for _ in range(200):
            pos, ang, holder = pass_rich_state(rng)
            if holder in (1, 2):
                continue
            place(env, pos[:6], pos[6:], ang[:6], ball=pos[holder], holder=holder, b_ang=ang[6:])
            assert np.array_equal(passing_actions(env, 0)[:3], scripted_actions(env, 0)[:3])
            checked += 1
        assert checked >= 100

    @pytest.mark.parametrize("side", [(0, 1), (1, 0)])
    def test_team_one_on_the_point_reflected_state_equals_team_zero(self, env, side):
        rng = np.random.default_rng(31)
        differing = 0
        for _ in range(60):
            pos, ang, holder = pass_rich_state(rng)
            ang[holder] = (0.0 if side == (0, 1) else math.pi) + rng.uniform(-0.5, 0.5)   # roughly toward the attacked goal
            ball = pos[holder].copy()
            place(env, pos[:6], pos[6:], ang[:6], ball=ball, holder=holder, side=side, b_ang=ang[6:])
            reference = passing_actions(env, 0)
            differing += int(not np.array_equal(reference[holder], scripted_actions(env, 0)[holder]))
            pos2, ang2, ball2, holder2 = TestMirrorSymmetry.mirrored_state(pos, ang, ball, holder)
            place(env, pos2[:6], pos2[6:], ang2[:6], ball=ball2, holder=holder2, side=side, b_ang=ang2[6:])
            np.testing.assert_allclose(passing_actions(env, 1), reference, atol=1e-5)
        # The comparison must exercise the passing branch, not only the shared rows.
        assert differing >= 15


# ----------------------------------------------------------------------------
# Frozen behaviour of the held-out bot
# ----------------------------------------------------------------------------
# Team B (defending the right goal, so every row passes through the point reflection) with its M2 (global player 10) holding the ball.
# Each case pins the whole 6 x 4 action block. The carrier rows are also solved by hand in the canonical frame:
#   pass kick:  carrier (685.04, 98.09) facing 1.1816, open teammate M1 at (760.18, 314.40): bearing 1.2365, error 0.0549 rad
#               (< 0.15), steer 0.109744, length 228.99 u, kick 0.35 + 0.65 * 228.99 / 600 = 0.598072.
#   pass turn:  same carrier at (631.19, 134.44) facing 4.7376, teammate at (648.65, 419.84): error 3.0553 rad, so it turns hard
#               with throttle 0.2 and steer 1 and does not kick.
#   dribble:    pressed, no open forward teammate; steers at the goal (1400, 420): error -0.265273, steer -0.530545, 819.8 u out.
GOLDEN_STATES = {
    "pass_kick": dict(
        pos=[[26.77, 422.21], [382.72, 546.88], [395.55, 696.17], [553.54, 535.23], [643.97, 749.97], [746.21, 684.91],
             [1344.94, 474.99], [864.23, 538.11], [864.23, 301.89], [639.82, 525.6], [714.96, 741.91], [718.9, 706.13]],
        ang=[-1.009, 1.219, 1.432, 1.078, 1.192, 1.669, 0.462, 3.034, -3.034, 2.426, -1.96, 2.278],
        ball=[705.85, 719.71],
        rows=[[0.0, 0.000112, -1.0, -1.0], [0.0, 0.042931, -1.0, -1.0], [0.0, -0.042931, -1.0, -1.0],
              [1.0, 0.013981, -1.0, -1.0], [1.0, 0.109744, 0.598072, -1.0], [0.0, 0.116590, -1.0, -1.0]],
    ),
    "pass_turn": dict(
        pos=[[47.06, 415.32], [395.83, 466.08], [393.04, 585.21], [540.44, 437.83], [613.95, 702.13], [707.15, 599.23],
             [1344.94, 474.99], [873.69, 536.86], [873.69, 303.14], [751.35, 420.16], [768.81, 705.56], [763.04, 614.07]],
        ang=[-0.116, 0.745, 1.397, 1.264, 0.986, 0.74, 0.462, 2.709, -2.709, 2.289, 1.596, 1.528],
        ball=[768.22, 729.55],
        rows=[[0.0, 0.000112, -1.0, -1.0], [0.0, -0.046807, -1.0, -1.0], [0.0, 0.046807, -1.0, -1.0],
              [1.0, 0.192158, -1.0, -1.0], [0.2, 1.0, -1.0, -1.0], [1.0, -0.004060, -1.0, -1.0]],
    ),
    "dribble_pressed": dict(
        pos=[[18.0, 434.93], [397.52, 495.55], [400.58, 645.15], [552.77, 500.5], [631.13, 732.94], [749.61, 655.46],
             [1344.94, 474.99], [873.03, 537.0], [873.03, 303.0], [687.5, 482.92], [759.06, 729.64], [753.92, 694.01]],
        ang=[-1.096, 2.133, 1.515, 1.492, 1.092, 1.617, 0.462, 2.962, -2.962, 2.407, -2.489, 2.073],
        ball=[740.0, 715.06],
        rows=[[0.0, 0.000112, -1.0, -1.0], [0.0, 0.010444, -1.0, -1.0], [0.0, -0.010444, -1.0, -1.0],
              [1.0, 0.002370, -1.0, -1.0], [1.0, -0.530545, -1.0, -1.0], [1.0, 0.164125, -1.0, -1.0]],
    ),
}


class TestPassingGolden:
    @pytest.mark.parametrize("name", list(GOLDEN_STATES))
    def test_the_frozen_bot_plays_the_saved_state(self, env, name):
        s = GOLDEN_STATES[name]
        pos, ang = np.asarray(s["pos"]), np.asarray(s["ang"])
        place(env, pos[:6], pos[6:], ang[:6], ball=s["ball"], holder=10, side=(0, 1), b_ang=ang[6:])
        np.testing.assert_allclose(passing_actions(env, 1), np.asarray(s["rows"], dtype=np.float32), atol=1e-5)
