"""Unit tests for the pure, vectorised potential field in `football_rl.potentials`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from football_rl.config import DEFAULT_CONFIG, SIDE_LEFT, SIDE_RIGHT
from football_rl.potentials import (
    ROLE_BALL,
    ROLE_CHASE,
    ROLE_COVER,
    ROLE_GK,
    ROLE_PRESS,
    ROLE_SECOND_PRESS,
    PotentialState,
    compute_potentials,
    potential_gradient_arrows,
)

REW = DEFAULT_CONFIG.rewards
PHYS = DEFAULT_CONFIG.physics
W = PHYS.field_width
H = PHYS.field_height

# Per-slot tactical biases (keeper, def_1, def_2, mid_1, mid_2, atk_1), identical for both teams.
DEPTH = np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 1.0] * 2)
FLANK = np.array([0.0, -1.0, 1.0, -1.0, 1.0, 0.0] * 2)

# Team A (left side, indices 0..5) formation; team B is its point reflection (indices 6..11).
FORMATION_A = np.array(
    [[40.0, 420.0], [250.0, 250.0], [250.0, 590.0], [450.0, 200.0], [450.0, 640.0], [600.0, 420.0]]
)
TEAM_PERM = np.r_[6:12, 0:6]


def make_state(**overrides) -> PotentialState:
    """Kickoff-like state: both teams spread out (no crowding), ball loose at the centre spot."""
    pos = np.vstack([FORMATION_A, np.array([W, H]) - FORMATION_A])
    angle = np.array([0.0] * 6 + [math.pi] * 6)
    fields = dict(
        pos=pos,
        angle=angle,
        ball=np.array([W / 2, H / 2]),
        holder=-1,
        ball_in_flight=False,
        last_ball_team=-1,
        team_side=np.array([0, 1], dtype=np.int32),
        disabled=np.zeros(12, dtype=bool),
        stunned=np.zeros(12, dtype=bool),
        post_kick_timer=np.zeros(12),
        depth_bias=DEPTH.copy(),
        flank_bias=FLANK.copy(),
        active_box_radius=np.array([PHYS.keeper_box_radius, PHYS.keeper_box_radius]),
    )
    fields.update(overrides)
    return PotentialState(**fields)


def random_state(rng: np.random.Generator) -> PotentialState:
    holder = int(rng.integers(-1, 12))
    in_flight = holder < 0 and bool(rng.integers(0, 2))
    disabled = rng.random(12) < 0.15
    return make_state(
        pos=np.stack([rng.uniform(20.0, W - 20.0, 12), rng.uniform(20.0, H - 20.0, 12)], axis=1),
        angle=rng.uniform(-math.pi, math.pi, 12),
        ball=np.array([rng.uniform(0.0, W), rng.uniform(0.0, H)]),
        holder=holder,
        ball_in_flight=in_flight,
        last_ball_team=int(rng.integers(0, 2)) if (in_flight or rng.random() < 0.3) else -1,
        disabled=disabled,
        stunned=disabled & (rng.random(12) < 0.7),
        post_kick_timer=np.where(rng.random(12) < 0.2, rng.uniform(0.0, 0.5, 12), 0.0),
    )


def mirrored(state: PotentialState) -> PotentialState:
    """Point reflection of the pitch with the team indices swapped.

    The reflected players of team B (right side) stand on the left and take the team A indices.
    """
    reflect = np.array([W, H])
    return state._replace(
        pos=(reflect - state.pos)[TEAM_PERM],
        angle=(state.angle + math.pi)[TEAM_PERM],
        ball=reflect - state.ball,
        holder=-1 if state.holder < 0 else (state.holder + 6) % 12,
        last_ball_team=-1 if state.last_ball_team < 0 else 1 - state.last_ball_team,
        team_side=np.where(state.team_side[::-1] == SIDE_LEFT, SIDE_RIGHT, SIDE_LEFT).astype(np.int32),
        disabled=state.disabled[TEAM_PERM],
        stunned=state.stunned[TEAM_PERM],
        post_kick_timer=state.post_kick_timer[TEAM_PERM],
        depth_bias=state.depth_bias[TEAM_PERM],
        flank_bias=state.flank_bias[TEAM_PERM],
        active_box_radius=state.active_box_radius[::-1].copy(),
    )


def evaluate(state: PotentialState):
    return compute_potentials(state, REW, PHYS)


def test_result_is_deterministic_and_inputs_are_untouched():
    rng = np.random.default_rng(7)
    for _ in range(25):
        state = random_state(rng)
        snapshot = [np.array(f, copy=True) if isinstance(f, np.ndarray) else f for f in state]
        first = evaluate(state)
        second = evaluate(state)
        for name in first._fields:
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name), err_msg=name)
        for before, after in zip(snapshot, state):
            np.testing.assert_array_equal(before, after)
        for name in first._fields:
            for field in state:
                if isinstance(field, np.ndarray):
                    assert not np.shares_memory(getattr(first, name), field), name


def test_result_shapes_dtypes_and_value_ranges():
    rng = np.random.default_rng(11)
    for _ in range(25):
        res = evaluate(random_state(rng))
        assert res.phi.shape == (12,) and res.phi.dtype == np.float64
        assert res.crowd.shape == (12,) and res.crowd.dtype == np.float64
        assert res.role_code.shape == (12,) and res.role_code.dtype == np.int8
        assert res.target.shape == (12, 2) and res.target.dtype == np.float64
        assert res.cover_mark.shape == (12,) and np.issubdtype(res.cover_mark.dtype, np.integer)
        assert set(res.role_code.tolist()) <= {0, 1, 2, 3, 4, 5}
        assert np.all(np.isfinite(res.phi)) and np.all(np.isfinite(res.target))
        assert np.all(res.crowd >= 0.0)


def test_role_code_values_are_stable():
    assert (ROLE_GK, ROLE_BALL, ROLE_CHASE, ROLE_PRESS, ROLE_SECOND_PRESS, ROLE_COVER) == (0, 1, 2, 3, 4, 5)


def test_keepers_are_goalkeeper_role_and_never_crowded_or_marking():
    rng = np.random.default_rng(3)
    for _ in range(25):
        res = evaluate(random_state(rng))
        assert res.role_code[0] == ROLE_GK and res.role_code[6] == ROLE_GK
        assert res.crowd[0] == 0.0 and res.crowd[6] == 0.0
        assert res.cover_mark[0] == -1 and res.cover_mark[6] == -1
        assert np.count_nonzero(res.role_code == ROLE_GK) == 2


def test_phi_is_bounded_by_largest_role_weight_and_by_crowd_penalty():
    weight_cap = max(
        REW.support_potential_weight,
        REW.def_press_potential_weight,
        REW.def_cover_potential_weight,
        REW.keeper_bisect_potential_weight,
    )
    rng = np.random.default_rng(5)
    for _ in range(50):
        res = evaluate(random_state(rng))
        assert np.all(res.phi <= weight_cap + 1e-12)
        assert np.all(res.phi >= -res.crowd - 1e-12)


def test_ball_carrier_holds_baseline_and_teammates_are_never_pressers():
    state = make_state(holder=5, ball=FORMATION_A[5] + np.array([12.0, 0.0]))
    res = evaluate(state)
    assert res.role_code[5] == ROLE_BALL
    assert res.phi[5] == pytest.approx(REW.def_press_potential_weight, abs=1e-12)
    np.testing.assert_allclose(res.target[5], state.ball, atol=1e-12)
    assert set(res.role_code[1:5].tolist()) == {ROLE_COVER}
    assert np.count_nonzero(res.role_code[7:12] == ROLE_PRESS) == 1
    assert ROLE_BALL not in res.role_code[6:].tolist()


def test_keeper_holding_ball_makes_its_team_the_possessing_side():
    state = make_state(holder=0, ball=np.array([60.0, 420.0]))
    res = evaluate(state)
    assert res.role_code[0] == ROLE_GK
    assert set(res.role_code[1:6].tolist()) == {ROLE_COVER}
    assert np.count_nonzero(res.role_code[7:12] == ROLE_PRESS) == 1
    assert np.count_nonzero(res.role_code[7:12] == ROLE_SECOND_PRESS) == 0


def test_loose_ball_gives_each_team_one_presser_and_one_second_presser():
    res = evaluate(make_state())
    for outfield in (slice(1, 6), slice(7, 12)):
        codes = res.role_code[outfield].tolist()
        assert codes.count(ROLE_PRESS) == 1
        assert codes.count(ROLE_SECOND_PRESS) == 1
        assert codes.count(ROLE_COVER) == 3


def test_second_presser_requires_an_enabled_runner_up():
    disabled = np.zeros(12, dtype=bool)
    disabled[[1, 2, 3, 4]] = True  # only the striker (index 5) of team A is enabled
    res = evaluate(make_state(disabled=disabled, stunned=disabled))
    codes = res.role_code[1:6].tolist()
    assert codes.count(ROLE_PRESS) == 1 and ROLE_SECOND_PRESS not in codes
    assert res.role_code[5] == ROLE_PRESS


def test_in_flight_pass_picks_the_receiver_nearest_the_ball_as_chaser():
    ball = np.array([520.0, 300.0])
    state = make_state(ball=ball, ball_in_flight=True, last_ball_team=0)
    res = evaluate(state)
    dists = np.hypot(*(state.pos[1:6] - ball).T)
    expected_chaser = 1 + int(np.argmin(dists))
    assert res.role_code[expected_chaser] == ROLE_CHASE
    assert np.count_nonzero(res.role_code[1:6] == ROLE_CHASE) == 1
    assert np.count_nonzero(res.role_code[7:12] == ROLE_PRESS) == 1
    assert ROLE_BALL not in res.role_code.tolist()


def test_in_flight_ball_of_other_team_leaves_team_out_of_possession():
    res = evaluate(make_state(ball_in_flight=True, last_ball_team=1))
    assert ROLE_CHASE in res.role_code[7:12].tolist()
    assert ROLE_CHASE not in res.role_code[1:6].tolist()
    assert np.count_nonzero(res.role_code[1:6] == ROLE_PRESS) == 1


def test_covers_mark_distinct_opponents_and_never_the_ball_carrier():
    state = make_state(holder=8)
    state = state._replace(ball=state.pos[8] + np.array([-12.0, 0.0]))
    res = evaluate(state)
    covering = [i for i in range(1, 6) if res.role_code[i] == ROLE_COVER]
    marks = [int(res.cover_mark[i]) for i in covering]
    assert len(covering) == 4
    assert sorted(marks) == [7, 9, 10, 11]
    assert all(res.cover_mark[i] == -1 for i in (0, 6, 7, 8, 9, 10, 11) if res.role_code[i] != ROLE_COVER)


def test_loose_ball_covers_leave_the_opponent_nearest_the_ball_unmarked():
    state = make_state(ball=np.array([760.0, 430.0]))
    res = evaluate(state)
    nearest_opp = 7 + int(np.argmin(np.hypot(*(state.pos[7:12] - state.ball).T)))
    covering = [i for i in range(1, 6) if res.role_code[i] == ROLE_COVER or res.role_code[i] == ROLE_SECOND_PRESS]
    marks = {int(res.cover_mark[i]) for i in range(1, 6) if res.role_code[i] in (ROLE_COVER, ROLE_SECOND_PRESS)}
    assert nearest_opp not in marks
    assert len(marks) == 4 and marks <= {7, 8, 9, 10, 11}
    assert len(covering) == 4
    assert res.cover_mark[int(np.flatnonzero(res.role_code[1:6] == ROLE_PRESS)[0]) + 1] == -1


@pytest.mark.parametrize("gap, expected_fraction", [(0.0, 1.0), (55.0, 0.25), (110.0, 0.0), (200.0, 0.0)])
def test_crowd_penalty_follows_the_squared_overlap_of_the_crowd_radius(gap, expected_fraction):
    pos = make_state().pos.copy()
    pos[1] = [250.0, 250.0]
    pos[2] = [250.0, 250.0 + gap]
    res = evaluate(make_state(pos=pos))
    # Hand-solved: radius 110 u, overlap fraction (110 - gap) / 110, penalty = weight * fraction^2.
    assert res.crowd[1] == pytest.approx(0.35 * expected_fraction, abs=1e-12)
    assert res.crowd[2] == pytest.approx(0.35 * expected_fraction, abs=1e-12)
    assert np.count_nonzero(res.crowd) == (2 if expected_fraction > 0.0 else 0)


def test_crowd_penalty_ignores_disabled_neighbours_but_not_disabled_victims():
    pos = make_state().pos.copy()
    pos[2] = pos[1]
    disabled = np.zeros(12, dtype=bool)
    disabled[2] = True
    res = evaluate(make_state(pos=pos, disabled=disabled, stunned=disabled))
    assert res.crowd[1] == 0.0
    assert res.crowd[2] == pytest.approx(0.35, abs=1e-12)


def test_crowd_penalty_accumulates_over_several_neighbours():
    pos = make_state().pos.copy()
    pos[2] = pos[1]
    pos[3] = pos[1]
    res = evaluate(make_state(pos=pos))
    # Three teammates stacked on one point: each sees two neighbours at overlap 1.0.
    for i in (1, 2, 3):
        assert res.crowd[i] == pytest.approx(0.35 * 2.0, abs=1e-12)


def test_crowding_in_one_team_does_not_touch_the_other_team():
    pos = make_state().pos.copy()
    pos[8] = pos[7]
    res = evaluate(make_state(pos=pos))
    assert np.count_nonzero(res.crowd[:6]) == 0
    assert res.crowd[7] > 0.0 and res.crowd[8] > 0.0


def test_goalkeeper_on_its_bisector_target_facing_the_ball_earns_the_full_weight():
    state = make_state()  # ball at the centre spot, 700 u from the left goal centre
    # Hand-solved bisector: depth = 38 + 47 * (1400 - 700) / 980 = 71.5714..., on the goal-to-ball ray (y = 420).
    gk_target = np.array([38.0 + 47.0 * 700.0 / 980.0, 420.0])
    pos = state.pos.copy()
    pos[0] = gk_target
    res = evaluate(state._replace(pos=pos))
    np.testing.assert_allclose(res.target[0], gk_target, atol=1e-9)
    assert res.phi[0] == pytest.approx(REW.keeper_bisect_potential_weight, abs=1e-12)


@pytest.mark.parametrize("heading, facing_factor", [(0.0, 1.0), (math.pi / 2, 1.0), (2 * math.pi / 3, 0.5), (math.pi, 0.0)])
def test_goalkeeper_potential_scales_with_how_squarely_it_faces_the_ball(heading, facing_factor):
    state = make_state()
    gk_target = np.array([38.0 + 47.0 * 700.0 / 980.0, 420.0])
    pos = state.pos.copy()
    pos[0] = gk_target
    angle = state.angle.copy()
    angle[0] = heading
    res = evaluate(state._replace(pos=pos, angle=angle))
    # Facing factor is clip(1 + cos(angle between heading and ball direction), 0, 1); the ball lies along +x.
    assert res.phi[0] == pytest.approx(REW.keeper_bisect_potential_weight * facing_factor, abs=1e-9)


def test_goalkeeper_falls_off_with_distance_from_its_target():
    state = make_state()
    gk_target = np.array([38.0 + 47.0 * 700.0 / 980.0, 420.0])
    values = []
    for offset in (0.0, 40.0, 95.0, 190.0):
        pos = state.pos.copy()
        pos[0] = gk_target + np.array([0.0, offset])
        values.append(evaluate(state._replace(pos=pos)).phi[0])
    # Gaussian with sigma 95 u: exp(-d^2 / (2 * 95^2)); at d = 95 -> exp(-0.5), d = 190 -> exp(-2).
    base = REW.keeper_bisect_potential_weight
    assert values[0] == pytest.approx(base, abs=1e-12)
    assert values[2] == pytest.approx(base * math.exp(-0.5), rel=1e-9)
    assert values[3] == pytest.approx(base * math.exp(-2.0), rel=1e-9)
    assert values[0] > values[1] > values[2] > values[3] > 0.0


def test_goalkeeper_claims_a_loose_ball_inside_its_box_and_covers_the_goal_otherwise():
    inside = evaluate(make_state(ball=np.array([120.0, 380.0])))
    np.testing.assert_allclose(inside.target[0], [120.0, 380.0], atol=1e-12)
    outside = evaluate(make_state(ball=np.array([600.0, 380.0])))
    assert outside.target[0][0] < 100.0 and 328.0 <= outside.target[0][1] <= 512.0
    held = evaluate(make_state(holder=0, ball=np.array([60.0, 420.0])))
    # Possession by the keeper aims at the front outlet arc of its box: box radius - 25 u in front of the goal.
    np.testing.assert_allclose(held.target[0], [PHYS.keeper_box_radius - 25.0, 420.0], atol=1e-12)


def test_goalkeeper_targets_mirror_for_the_right_hand_side_team():
    left = evaluate(make_state(ball=np.array([300.0, 500.0])))
    right = evaluate(make_state(ball=np.array([W - 300.0, H - 500.0])))
    np.testing.assert_allclose(right.target[6], [W - left.target[0][0], H - left.target[0][1]], atol=1e-9)


def test_potentials_are_invariant_under_swapping_the_two_teams_across_the_pitch():
    rng = np.random.default_rng(21)
    for _ in range(40):
        state = random_state(rng)
        res = evaluate(state)
        res_m = evaluate(mirrored(state))
        np.testing.assert_allclose(res_m.phi[TEAM_PERM], res.phi, rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(res_m.crowd[TEAM_PERM], res.crowd, rtol=0.0, atol=1e-9)
        np.testing.assert_array_equal(res_m.role_code[TEAM_PERM], res.role_code)
        np.testing.assert_allclose(
            res_m.target[TEAM_PERM], np.array([W, H]) - res.target, rtol=0.0, atol=1e-9
        )


def test_swapped_team_sides_move_a_defenders_support_target_to_the_other_side_of_the_ball():
    holder_state = make_state(holder=5, ball=FORMATION_A[5] + np.array([12.0, 0.0]))
    # Defender def_1 (index 1) supports 210 u behind the ball relative to its own goal.
    left_side = evaluate(holder_state)
    assert left_side.target[1][0] < holder_state.ball[0]
    right_side = evaluate(holder_state._replace(team_side=np.array([1, 0], dtype=np.int32)))
    assert right_side.target[1][0] > holder_state.ball[0]


@pytest.mark.parametrize("holder", [-1, 0, 3, 6, 9])
def test_degenerate_geometry_stays_finite(holder):
    # Everyone and the ball on one point: zero-length lanes, maximal crowding, zero ball distances.
    point = np.array([700.0, 420.0])
    res = evaluate(make_state(pos=np.tile(point, (12, 1)), ball=point.copy(), holder=holder))
    assert np.all(np.isfinite(res.phi)) and np.all(np.isfinite(res.target)) and np.all(np.isfinite(res.crowd))
    # Each outfielder overlaps its four teammates completely: weight * 4 neighbours * 1.0^2.
    assert res.crowd[1] == pytest.approx(0.35 * 4.0, abs=1e-12)


def test_ball_exactly_on_a_receiver_has_a_finite_open_lane():
    state = make_state(holder=5, ball=FORMATION_A[5] + np.array([15.0, 0.0]))
    pos = state.pos.copy()
    pos[3] = state.ball  # receiver coincides with the ball: zero-length passing segment
    res = evaluate(state._replace(pos=pos))
    assert np.all(np.isfinite(res.phi))
    assert res.role_code[3] == ROLE_COVER


def test_all_players_disabled_still_yields_a_valid_assignment():
    disabled = np.ones(12, dtype=bool)
    res = evaluate(make_state(disabled=disabled, stunned=disabled))
    assert np.all(np.isfinite(res.phi))
    assert np.count_nonzero(res.role_code[1:6] == ROLE_PRESS) == 1
    assert np.count_nonzero(res.role_code[7:12] == ROLE_PRESS) == 1
    assert ROLE_SECOND_PRESS not in res.role_code.tolist()


def test_post_kick_timer_lifts_a_supporting_receivers_potential_toward_the_carrier_baseline():
    state = make_state(holder=5, ball=FORMATION_A[5] + np.array([12.0, 0.0]))
    base = evaluate(state)
    timer = np.zeros(12)
    timer[3] = 0.50
    lifted = evaluate(state._replace(post_kick_timer=timer))
    assert lifted.phi[3] == pytest.approx(max(base.phi[3], REW.def_press_potential_weight), abs=1e-12)
    half = np.zeros(12)
    half[3] = 0.25
    mid = evaluate(state._replace(post_kick_timer=half)).phi[3]
    assert base.phi[3] <= mid <= lifted.phi[3] + 1e-12
    np.testing.assert_array_equal(lifted.phi[[1, 2, 4, 5]], base.phi[[1, 2, 4, 5]])


def arrows_of(state: PotentialState):
    return potential_gradient_arrows(state, evaluate(state), REW)


def test_gradient_arrows_have_one_triple_per_player():
    rng = np.random.default_rng(13)
    for _ in range(20):
        arrows = arrows_of(random_state(rng))
        assert len(arrows) == 12
        assert all(len(a) == 3 and all(math.isfinite(v) for v in a) for a in arrows)


def test_gradient_arrows_are_unit_directions_or_zero():
    rng = np.random.default_rng(17)
    for _ in range(60):
        for ux, uy, mag in arrows_of(random_state(rng)):
            if (ux, uy, mag) == (0.0, 0.0, 0.0):
                continue
            assert math.hypot(ux, uy) == pytest.approx(1.0, abs=2e-3)
            assert mag >= 0.0


def test_carrier_arrow_points_in_the_attacking_direction_of_its_side():
    state = make_state(holder=5, ball=FORMATION_A[5] + np.array([12.0, 0.0]))
    assert arrows_of(state)[5] == pytest.approx([1.0, 0.0, 0.02], abs=1e-12)
    right_side = state._replace(team_side=np.array([1, 0], dtype=np.int32))
    assert arrows_of(right_side)[5] == pytest.approx([-1.0, 0.0, 0.02], abs=1e-12)


def test_presser_arrow_points_at_the_ball_with_the_chase_gradient_magnitude():
    # Presser 100 u from the ball: phi = 0.65 * exp(-(100 - 28) / 220) = 0.46858, magnitude = phi / 220 = 0.00213.
    arrows = arrows_of(make_state())
    assert arrows[5] == pytest.approx([1.0, 0.0, 0.0021], abs=1e-12)
    assert arrows[11] == pytest.approx([-1.0, 0.0, 0.0021], abs=1e-12)


def test_chaser_arrow_points_at_the_ball_in_flight():
    # Chaser mid_1 at (450, 200), ball at (520, 300): offset (70, 100), distance 122.066 u.
    state = make_state(ball=np.array([520.0, 300.0]), ball_in_flight=True, last_ball_team=0)
    assert arrows_of(state)[3] == pytest.approx([0.573, 0.819, 0.0019], abs=1e-12)


def test_keeper_arrow_points_at_its_target_with_a_gaussian_gradient_magnitude():
    state = make_state()
    pos = state.pos.copy()
    pos[0] = [38.0 + 47.0 * 700.0 / 980.0, 380.0]  # 40 u below the bisector target, facing the ball
    # phi = 0.6 * exp(-40^2 / (2 * 95^2)) = 0.54910, magnitude = phi * 2 * 40 / (2 * 95^2) = 0.002434.
    assert arrows_of(state._replace(pos=pos))[0] == pytest.approx([0.0, 1.0, 0.0024], abs=1e-12)


def push_scenario(neighbour_offset, disabled_neighbour=False):
    """Keeper holds the ball; defender 1 sits 40 u from it (zero ring pull) with defender 2 beside it."""
    state = make_state(holder=0, ball=np.array([60.0, 420.0]))
    pos = state.pos.copy()
    pos[1] = [60.0, 460.0]
    pos[2] = pos[1] + np.asarray(neighbour_offset)
    disabled = np.zeros(12, dtype=bool)
    disabled[2] = disabled_neighbour
    return state._replace(pos=pos, disabled=disabled, stunned=disabled)


def test_supporter_inside_the_ring_radius_is_pushed_directly_away_from_a_crowding_teammate():
    # No ring pull inside 45 u of the ball, so the arrow is the push alone. Neighbour offset (30, 40) is 50 u away:
    # overlap (110 - 50) / 110, push = 2 * 0.35 * overlap / 110 / 50 * (-30, -40), magnitude 0.003471.
    assert arrows_of(push_scenario((30.0, 40.0)))[1] == pytest.approx([-0.6, -0.8, 0.0035], abs=1e-12)
    assert arrows_of(push_scenario((-30.0, -40.0)))[1] == pytest.approx([0.6, 0.8, 0.0035], abs=1e-12)
    # Still inside the 110 u crowd radius at 80 u: overlap 30 / 110, magnitude 0.001736.
    assert arrows_of(push_scenario((48.0, 64.0)))[1] == pytest.approx([-0.6, -0.8, 0.0017], abs=1e-12)


def test_disabled_or_coincident_teammates_exert_no_push():
    assert arrows_of(push_scenario((30.0, 40.0), disabled_neighbour=True))[1] == [0.0, 0.0, 0.0]
    assert arrows_of(push_scenario((0.0, 0.0)))[1] == [0.0, 0.0, 0.0]
    assert arrows_of(push_scenario((300.0, 40.0)))[1] == [0.0, 0.0, 0.0]


def test_keeper_bisector_stays_at_least_38_units_off_the_goal_line():
    # Ball level with the goal line: the unclamped bisector would sit 4.5 u from it.
    res = evaluate(make_state(ball=np.array([20.0, 800.0])))
    assert res.target[0][0] == pytest.approx(38.0, abs=1e-12)
    assert res.target[0][1] == pytest.approx(504.8825, abs=1e-4)
