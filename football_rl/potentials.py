"""Pure per-player potential field for potential-based reward shaping.

`compute_potentials` maps a `PotentialState` snapshot to a `PotentialResult` without reading or
writing anything else: the same state always yields the same arrays and the inputs are never
modified. Player indices 0..5 are team A (keeper 0) and 6..11 are team B (keeper 6); the five
outfielders of a team follow the order defender, defender, midfielder, midfielder, striker.

Each team is either in possession (carrier, or the passing team while a kick is in flight) or
defending. Role potentials, before the teammate-crowding penalty, are:

* ball carrier: the carrier baseline (`def_press_potential_weight`);
* pass chaser (nearest outfielder of the passing team to a ball in flight) and primary presser:
  a Gaussian-in-distance pull toward the ball. The presser of a defending team is the outfielder
  with the lowest cost, the distance to the ball plus `RewardConfig.press_trail_weight` times how
  far the player stands upfield of the ball (measured from the team's own goal line);
* possessing-team supporters: a support ring around the ball, shaded by passing-lane openness and
  by the distance to a tactical channel target;
* defenders: a cover term from the distance to the assigned opponent's goal-side marking point and
  to a tactical anchor, with the second-closest defender blending in a ball pull when the ball is
  loose. Marks are assigned to defenders by minimising the total distance weighted by how close each
  opponent stands to the defended goal. When the keeper is nearer to a loose ball inside its box than
  the cost of every outfielder, no outfielder presses and all five cover the five opposing outfielders;
* keepers: a Gaussian in the distance to a goal-line bisector target (or the ball, or the front
  outlet arc when holding it) scaled by how squarely the keeper faces the ball.

The teammate-crowding penalty is additive and subtracted from supporters and defenders, so a
potential can be negative.

A covering defender (role `ROLE_COVER`, either possession phase) that stands ahead of its own striker,
measured along the attacking direction, loses `line_order_weight * tanh(lead / line_order_sigma)` from
its potential, where `lead` is how far the defender is ahead of the striker. The term is zero level
with or behind the striker, saturates at `line_order_weight`, and leaves the role potential, the
crowding penalty, the role labels and the targets untouched.

The field is evaluated with scalar arithmetic over Python floats taken from the state arrays. With
five players per team the arrays are too small for NumPy dispatch to pay for itself: element-wise
array code measures several times slower than these loops. The marking assignment is the one NumPy
step; it reads fixed-size scratch arrays that every call overwrites before use, so a result never
depends on earlier calls, but evaluations must not run concurrently in one process. Pairwise
distances use `math.hypot` / `math.dist`, which are correctly rounded and agree with each other bit
for bit; the player-to-ball distances use `np.hypot`, which differs from them in the last bit for a
small share of inputs.
"""

from __future__ import annotations

import itertools
import math
from math import dist, exp
from typing import NamedTuple

import numpy as np

from football_rl.config import SIDE_LEFT, PhysicsConfig, RewardConfig

ROLE_GK = 0
ROLE_BALL = 1
ROLE_CHASE = 2
ROLE_PRESS = 3
ROLE_SECOND_PRESS = 4
ROLE_COVER = 5
ROLE_LABELS = ("GK", "BALL", "CHSE", "1stP", "2ndP", "COVR")


class PotentialState(NamedTuple):
    """Everything the potential field reads.

    Attributes:
        pos: Player positions, shape (12, 2).
        angle: Player headings in radians, shape (12,).
        ball: Ball position, shape (2,).
        holder: Index of the player holding the ball, or -1 when the ball is loose.
        ball_in_flight: True while a kicked ball is travelling.
        last_ball_team: Team (0 or 1) that last touched the ball, or -1 when none.
        team_side: Side of the pitch per team (`SIDE_LEFT` / `SIDE_RIGHT`), shape (2,).
        disabled: Players that are stunned or ghosted, shape (12,) bool.
        stunned: Players that are stunned (a subset of `disabled`), shape (12,) bool. Not read by the
            evaluation; replay telemetry uses it to label stunned players.
        post_kick_timer: Seconds left of the post-kick shaping window per player, shape (12,).
        depth_bias: Tactical depth per player (-1 defensive, 0 middle, +1 attacking), shape (12,).
        flank_bias: Tactical lateral position per player (-1, 0, +1), shape (12,).
        active_box_radius: Current keeper-box radius per pitch side (indexed by the `team_side` value),
            shape (2,). A keeper claims a loose ball only inside its team's current radius; the keeper's
            own potential uses `PhysicsConfig.keeper_box_radius`.
    """

    pos: np.ndarray
    angle: np.ndarray
    ball: np.ndarray
    holder: int
    ball_in_flight: bool
    last_ball_team: int
    team_side: np.ndarray
    disabled: np.ndarray
    stunned: np.ndarray
    post_kick_timer: np.ndarray
    depth_bias: np.ndarray
    flank_bias: np.ndarray
    active_box_radius: np.ndarray


class PotentialResult(NamedTuple):
    """Per-player output of `compute_potentials`.

    Attributes:
        phi: Potential Phi per player, shape (12,).
        crowd: Teammate-crowding penalty per player (>= 0, keepers 0), shape (12,).
        role_code: Role per player (`ROLE_*`), shape (12,) int8.
        target: Point each player's potential pulls toward, shape (12, 2).
        cover_mark: Index of the opponent each defender marks, -1 when none, shape (12,).
        phi_role: Role potential before the crowding penalty, shape (12,).
    """

    phi: np.ndarray
    crowd: np.ndarray
    role_code: np.ndarray
    target: np.ndarray
    cover_mark: np.ndarray
    phi_role: np.ndarray


_INV_75 = 1.0 / 75.0
_INV_980 = 1.0 / 980.0
_INV_LAMBDA_CHASE = 1.0 / 220.0
_INV_2_95_SQ = 1.0 / (2.0 * 95.0 * 95.0)
_INV_2_240_SQ = 1.0 / (2.0 * 240.0 * 240.0)
_INV_2_280_SQ = 1.0 / (2.0 * 280.0 * 280.0)
_NEG_INV_2_95_SQ = -_INV_2_95_SQ
_NEG_INV_2_165_SQ = -(1.0 / (2.0 * 165.0 * 165.0))
_NEG_INV_2_240_SQ = -_INV_2_240_SQ
_NEG_INV_2_260_SQ = -(1.0 / (2.0 * 260.0 * 260.0))
_NEG_INV_2_280_SQ = -_INV_2_280_SQ
_PICKUP_REACH = 28.0
_POST_KICK_WINDOW = 0.50
_GOAL_CENTER_Y = 420.0
_DISABLED_RANK_PENALTY = 10000.0

_ROLE_TEMPLATE = [ROLE_GK, *([ROLE_COVER] * 5), ROLE_GK, *([ROLE_COVER] * 5)]

# Distance charged per unit of danger weight to a mark left without a coverer (the distance the potential
# assigns to a defender that has no mark).
_UNCOVERED_MARK_DIST = 300.0

# Every ordered choice of k distinct cover points out of m, as rows of point indices, for 1 <= k <= m <= 5.
_PERMS = {
    (m, k): np.array(list(itertools.permutations(range(m), k)), dtype=np.intp).reshape(-1, k)
    for m in range(1, 6)
    for k in range(1, m + 1)
}
_ROWS = tuple(np.arange(k)[None, :] for k in range(6))

# Scratch inputs of `assign_cover`: coverer positions, cover points and danger weights, filled per call.
_COVER_XY = np.zeros((5, 2))
_COVER_PTS = np.zeros((5, 2))
_COVER_WEIGHT = np.zeros(5)


def assign_cover(cover_xy: np.ndarray, cov_pts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Assign K coverers to K distinct cover points out of M (0 <= K <= M <= 5).

    Returns `perm`, shape (K,), such that coverer `c` marks cover point `perm[c]`. The assignment minimises

        sum_c weights[perm[c]] * |cover_xy[c] - cov_pts[perm[c]]|
        + _UNCOVERED_MARK_DIST * sum_{m not in perm} weights[m]

    so that when K < M the points left uncovered are the least dangerous ones; for K == M the second term
    is constant and the objective is the plain weighted distance sum. Ties resolve to the first permutation
    in `itertools.permutations` order.

    Raises:
        ValueError: on K > M, M > 5, a shape mismatch, or a non-finite cost.
    """
    k = cover_xy.shape[0]
    m = cov_pts.shape[0]
    if (
        cover_xy.ndim != 2
        or cov_pts.ndim != 2
        or cover_xy.shape[1] != 2
        or cov_pts.shape[1] != 2
        or weights.shape != (m,)
        or k > m
        or m > 5
    ):
        raise ValueError(f"assign_cover: invalid shapes cover={cover_xy.shape} pts={cov_pts.shape} w={weights.shape}")
    if k == 0:
        return np.zeros(0, dtype=np.intp)
    d = np.hypot(cover_xy[:, None, 0] - cov_pts[None, :, 0], cover_xy[:, None, 1] - cov_pts[None, :, 1])
    # Charging each uncovered mark `weights[m] * _UNCOVERED_MARK_DIST` is the same as subtracting that distance
    # from every entry: the sum over a permutation then differs from the full objective by a constant.
    cost = (d - _UNCOVERED_MARK_DIST) * weights[None, :]
    if not np.isfinite(cost).all():
        raise ValueError("assign_cover: non-finite cost")
    perms = _PERMS[(m, k)]
    totals = cost[_ROWS[k], perms].sum(axis=1)
    return perms[int(np.argmin(totals))]


def compute_potentials(state: PotentialState, rew: RewardConfig, phys: PhysicsConfig) -> PotentialResult:
    """Evaluate the potential field for one state."""
    pos_arr = state.pos
    bx = float(state.ball[0])
    by = float(state.ball[1])
    holder = int(state.holder)
    pos = pos_arr.tolist()
    ball_dist = np.hypot(pos_arr[:, 0] - bx, pos_arr[:, 1] - by).tolist()
    disabled = state.disabled.tolist()
    side = state.team_side.tolist()
    depth = state.depth_bias.tolist()
    flank = state.flank_bias.tolist()
    kick = state.post_kick_timer.tolist()
    box_radius = state.active_box_radius.tolist()

    holder_team = -1 if holder < 0 else (0 if holder < 6 else 1)
    flight_team = int(state.last_ball_team) if (holder < 0 and state.ball_in_flight) else -1
    w_press = rew.def_press_potential_weight

    phi = [0.0] * 12
    phi_role = [0.0] * 12
    role = list(_ROLE_TEMPLATE)
    target = [0.0] * 24
    cover_mark = [-1] * 12
    crowd = [0.0] * 12

    # Teammate crowding: additive penalty from the squared overlap with enabled teammates, summed in
    # ascending teammate order (accumulated in `crowd`, then scaled by the weight).
    r_crowd = rew.teammate_crowd_radius
    w_crowd = rew.teammate_crowd_weight
    inv_r_crowd = 1.0 / max(1e-6, r_crowd)
    for f0 in (1, 7):
        for a in range(f0, f0 + 4):
            pa = pos[a]
            a_enabled = not disabled[a]
            for b in range(a + 1, f0 + 5):
                d_ab = dist(pa, pos[b])
                if d_ab < r_crowd:
                    viol = (r_crowd - d_ab) * inv_r_crowd
                    sq = viol * viol
                    if not disabled[b]:
                        crowd[a] += sq
                    if a_enabled:
                        crowd[b] += sq
    for g in (1, 2, 3, 4, 5, 7, 8, 9, 10, 11):
        crowd[g] = w_crowd * crowd[g]

    defending = []
    for team in (0, 1):
        in_possession = holder_team == team or flight_team == team
        if in_possession:
            _support_possession(
                team, side[team] == SIDE_LEFT, rew, holder, bx, by, pos, ball_dist, depth, flank, kick, crowd,
                phi, phi_role, role, target,
            )
        else:
            defending.append(team)
    for team in defending:
        _cover_defence(
            team, side[team] == SIDE_LEFT, rew, phys, holder, bx, by, pos, ball_dist, depth, flank, disabled,
            box_radius[side[team]], crowd, phi, phi_role, role, target, cover_mark,
        )

    for team in (0, 1):
        _keeper_value(team, side[team] == SIDE_LEFT, state, rew, phys, holder, bx, by, pos, ball_dist, phi, phi_role, target)

    w_line = rew.line_order_weight
    if w_line > 0.0:
        inv_sigma = 1.0 / rew.line_order_sigma
        for team in (0, 1):
            f0 = 1 + 6 * team
            sgn = 1.0 if side[team] == SIDE_LEFT else -1.0
            striker_cx = pos[f0 + 4][0] * sgn
            for g in (f0, f0 + 1):
                if role[g] != ROLE_COVER:
                    continue
                lead = pos[g][0] * sgn - striker_cx
                if lead > 0.0:
                    phi[g] -= w_line * math.tanh(lead * inv_sigma)

    return PotentialResult(
        np.array(phi),
        np.array(crowd),
        np.array(role, dtype=np.int8),
        np.array(target).reshape(12, 2),
        np.array(cover_mark),
        np.array(phi_role),
    )


def _reference_point(f0, bx, by, pos):
    """Tactical reference: a blend of the ball and the team's outfield centroid."""
    p0, p1, p2, p3, p4 = pos[f0:f0 + 5]
    centroid_x = (p0[0] + p1[0] + p2[0] + p3[0] + p4[0]) * 0.2
    centroid_y = (p0[1] + p1[1] + p2[1] + p3[1] + p4[1]) * 0.2
    return 0.65 * bx + 0.35 * centroid_x, 0.50 * by + 0.50 * centroid_y


def _support_possession(
    team, is_left, rew, holder, bx, by, pos, ball_dist, depth, flank, kick, crowd, phi, phi_role, role, target
):
    """Carrier baseline, pass chaser, and support ring for the team in possession."""
    f0 = 1 + 6 * team
    o0 = 7 - 6 * team
    sign = 1.0 if is_left else -1.0
    ref_x, ref_y = _reference_point(f0, bx, by, pos)
    carrier_baseline = rew.def_press_potential_weight
    w_supp = rew.support_potential_weight

    chaser = -1
    if holder < 0:
        nearest = 1e9
        for g in range(f0, f0 + 5):
            if ball_dist[g] < nearest:
                nearest = ball_dist[g]
                chaser = g

    rivals = pos[o0:o0 + 5]
    rel = [(r[0] - bx, r[1] - by) for r in rivals]
    ball_pt = (bx, by)

    for g in range(f0, f0 + 5):
        if g == holder:
            phi[g] = phi_role[g] = carrier_baseline
            role[g] = ROLE_BALL
            target[2 * g] = bx
            target[2 * g + 1] = by
            continue
        d_ib = ball_dist[g]
        if g == chaser:
            pull = _press_pull(carrier_baseline, d_ib)
            phi[g] = phi_role[g] = pull
            role[g] = ROLE_CHASE
            target[2 * g] = bx
            target[2 * g + 1] = by
            continue
        px, py = pos[g]

        # Channel target and shading.
        target_x = ref_x + sign * (depth[g] * 210.0)
        target_x = 90.0 if target_x < 90.0 else (1310.0 if target_x > 1310.0 else target_x)
        target_y = ref_y + sign * (flank[g] * 195.0)
        target_y = 75.0 if target_y < 75.0 else (765.0 if target_y > 765.0 else target_y)
        dx = px - target_x
        dy = py - target_y
        psi_channel = 0.30 + 0.70 * exp((dx * dx + dy * dy) * _NEG_INV_2_240_SQ)

        # Support ring around the ball.
        if d_ib <= 45.0:
            ramp = 0.0
        elif d_ib >= 95.0:
            ramp = 1.0
        else:
            ramp = (d_ib - 45.0) * 0.02
        d_diff = d_ib - 265.0
        psi_ring = ramp * exp((d_diff * d_diff) * _NEG_INV_2_165_SQ)

        # Passing lane: smallest distance from an opponent to the segment from the ball to the receiver.
        abx = px - bx
        aby = py - by
        d2 = abx * abx + aby * aby
        if d2 < 1e-12:
            lane = min([dist(r, ball_pt) for r in rivals])
        else:
            inv_d2 = 1.0 / d2
            lane = 1e18
            for k in range(5):
                rx, ry = rel[k]
                t = (rx * abx + ry * aby) * inv_d2
                t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                d_seg = dist(rivals[k], (bx + t * abx, by + t * aby))
                if d_seg < lane:
                    lane = d_seg
        psi_lane = lane * _INV_75
        psi_lane = 0.15 if psi_lane < 0.15 else (1.0 if psi_lane > 1.0 else psi_lane)

        ring = w_supp * psi_ring * psi_lane * psi_channel
        value = ring - crowd[g]
        if kick[g] > 0.0:
            alpha = kick[g] / _POST_KICK_WINDOW
            blended = alpha * carrier_baseline + (1.0 - alpha) * value
            if blended > value:
                value = blended
        phi[g] = value
        phi_role[g] = ring
        target[2 * g] = target_x
        target[2 * g + 1] = target_y


def _press_pull(w_press, d_ball):
    """Gaussian-in-distance pull toward the ball, measured beyond the pickup reach."""
    reach = d_ball - _PICKUP_REACH
    return w_press * exp(-(reach if reach > 0.0 else 0.0) * _INV_LAMBDA_CHASE)


def _cover_defence(
    team, is_left, rew, phys, holder, bx, by, pos, ball_dist, depth, flank, disabled, box_radius, crowd,
    phi, phi_role, role, target, cover_mark,
):
    """Presser selection, keeper claim of a loose ball, optimal marking assignment, and cover potentials."""
    f0 = 1 + 6 * team
    o0 = 7 - 6 * team
    sign = 1.0 if is_left else -1.0
    own_gx = 0.0 if is_left else phys.field_width
    w_press = rew.def_press_potential_weight
    ref_x, ref_y = _reference_point(f0, bx, by, pos)

    # Presser: lowest-cost outfielder, the distance to the ball plus a penalty for standing upfield of it
    # (disabled players rank behind every enabled one; ties go to the nearer, then the lower index).
    trail_weight = rew.press_trail_weight
    ball_depth = (bx - own_gx) * sign
    scores = []
    for loc in range(5):
        g = f0 + loc
        trail = (pos[g][0] - own_gx) * sign - ball_depth
        cost = ball_dist[g] + trail_weight * (trail if trail > 0.0 else 0.0)
        scores.append((cost + (_DISABLED_RANK_PENALTY if disabled[g] else 0.0), ball_dist[g], loc))
    scores.sort()

    # A keeper that is enabled and nearer to a loose ball inside its box than the best press cost claims the
    # ball: no outfielder presses and all five mark.
    keeper = 6 * team
    keeper_claims = (
        holder < 0
        and not disabled[keeper]
        and math.hypot(bx - own_gx, by - _GOAL_CENTER_Y) <= box_radius
        and ball_dist[keeper] < scores[0][0]
    )
    presser = -1 if keeper_claims else scores[0][2]
    second = -1
    second_pot = 0.0
    if holder < 0 and not keeper_claims and not disabled[f0 + scores[1][2]]:
        second = scores[1][2]
        second_pot = rew.loose_ball_secondary_press_blend * w_press * exp(
            -max(0.0, ball_dist[f0 + second] - _PICKUP_REACH) * _INV_LAMBDA_CHASE
        )

    # Opponents to mark: every outfield opponent except the carrier, or, when the ball is loose, except the one
    # nearest the ball (unless the keeper claims the ball, which leaves all five to be marked).
    if o0 <= holder < o0 + 5:
        marks = [o for o in range(o0, o0 + 5) if o != holder]
    else:
        marks = sorted(range(o0, o0 + 5), key=ball_dist.__getitem__)
        if not keeper_claims:
            marks = marks[1:]
    cover_locs = [loc for loc in range(5) if loc != presser]
    n_cover = len(cover_locs)
    n_mark = len(marks)
    assert n_cover <= n_mark

    # Marking points sit between each opponent and the defended goal; each is weighted by how close its
    # opponent stands to that goal (2 on the goal line, 1 on the far one).
    points = []
    for j, o in enumerate(marks):
        ox, oy = pos[o]
        point_x = ox + 0.35 * (own_gx - ox)
        point_y = oy + 0.35 * (_GOAL_CENTER_Y - oy)
        points.append((point_x, point_y))
        _COVER_PTS[j, 0] = point_x
        _COVER_PTS[j, 1] = point_y
        advance = (ox - own_gx) * sign / phys.field_width
        _COVER_WEIGHT[j] = 2.0 - (0.0 if advance < 0.0 else (1.0 if advance > 1.0 else advance))
    for c, loc in enumerate(cover_locs):
        mate = pos[f0 + loc]
        _COVER_XY[c, 0] = mate[0]
        _COVER_XY[c, 1] = mate[1]
    assignment = assign_cover(_COVER_XY[:n_cover], _COVER_PTS[:n_mark], _COVER_WEIGHT[:n_mark]).tolist()
    assigned_gap = [300.0] * 5
    for c, loc in enumerate(cover_locs):
        m = assignment[c]
        assigned_gap[loc] = dist(pos[f0 + loc], points[m])
        cover_mark[f0 + loc] = marks[m]

    w_cover = rew.def_cover_potential_weight
    for loc in range(5):
        g = f0 + loc
        if loc == presser:
            pull = _press_pull(w_press, ball_dist[g])
            phi[g] = phi_role[g] = pull
            role[g] = ROLE_PRESS
            target[2 * g] = bx
            target[2 * g + 1] = by
            continue
        px, py = pos[g]
        gap = assigned_gap[loc]
        psi_cover = exp((gap * gap) * _NEG_INV_2_260_SQ)
        anchor_x = ref_x + sign * (depth[g] * 190.0)
        anchor_x = 80.0 if anchor_x < 80.0 else (1320.0 if anchor_x > 1320.0 else anchor_x)
        anchor_y = ref_y + sign * (flank[g] * 180.0)
        anchor_y = 75.0 if anchor_y < 75.0 else (765.0 if anchor_y > 765.0 else anchor_y)
        dx = px - anchor_x
        dy = py - anchor_y
        psi_anchor = 0.40 + 0.60 * exp((dx * dx + dy * dy) * _NEG_INV_2_280_SQ)
        base = w_cover * psi_cover * psi_anchor
        if loc == second:
            if second_pot > base:
                base = second_pot
            role[g] = ROLE_SECOND_PRESS
            target[2 * g] = bx
            target[2 * g + 1] = by
        else:
            target[2 * g] = anchor_x
            target[2 * g + 1] = anchor_y
        phi[g] = base - crowd[g]
        phi_role[g] = base


def _keeper_value(team, is_left, state, rew, phys, holder, bx, by, pos, ball_dist, phi, phi_role, target):
    """Score a keeper against its bisector target, the loose ball in its box, or its outlet arc."""
    keeper = 6 * team
    width = phys.field_width
    own_gx = 0.0 if is_left else width
    own_gy = _GOAL_CENTER_Y
    ray_x = bx - own_gx
    ray_y = by - own_gy
    ray_norm = max(1e-6, math.hypot(ray_x, ray_y))
    ray_inv = 1.0 / ray_norm
    depth_t = (1400.0 - ray_norm) * _INV_980
    depth_t = 0.0 if depth_t < 0.0 else (1.0 if depth_t > 1.0 else depth_t)
    depth = 38.0 + 47.0 * depth_t
    bis_x = own_gx + (ray_x * ray_inv) * depth
    bis_y = own_gy + (ray_y * ray_inv) * depth
    low = phys.goal_y_min + 8.0
    high = phys.goal_y_max - 8.0
    bis_y = low if bis_y < low else (high if bis_y > high else bis_y)
    if is_left:
        low_x, high_x = 38.0, 95.0
    else:
        low_x, high_x = width - 95.0, width - 38.0
    bis_x = low_x if bis_x < low_x else (high_x if bis_x > high_x else bis_x)

    gk_x, gk_y = pos[keeper]
    if holder == keeper:
        arc = phys.keeper_box_radius - 25.0
        tgt_x = own_gx + arc if is_left else own_gx - arc
        tgt_y = min(max(gk_y, own_gy - 110.0), own_gy + 110.0)
        gap = math.hypot(gk_x - tgt_x, gk_y - tgt_y) * 0.5
        facing = 1.0
    else:
        if holder < 0 and ray_norm <= phys.keeper_box_radius + 50.0:
            tgt_x = bx
            tgt_y = by
            gap = ball_dist[keeper]
        else:
            tgt_x = bis_x
            tgt_y = bis_y
            gap = math.hypot(gk_x - bis_x, gk_y - bis_y)
        heading = float(state.angle[keeper])
        to_ball_x = bx - gk_x
        to_ball_y = by - gk_y
        inv_norm = 1.0 / max(1e-6, math.hypot(to_ball_x, to_ball_y))
        cos_phi = math.cos(heading) * (to_ball_x * inv_norm) + math.sin(heading) * (to_ball_y * inv_norm)
        facing = 1.0 + cos_phi
        facing = 0.0 if facing < 0.0 else (1.0 if facing > 1.0 else facing)
    value = rew.keeper_bisect_potential_weight * exp((gap * gap) * _NEG_INV_2_95_SQ) * facing
    phi[keeper] = phi_role[keeper] = value
    target[2 * keeper] = tgt_x
    target[2 * keeper + 1] = tgt_y


def potential_gradient_arrows(state: PotentialState, res: PotentialResult, rew: RewardConfig) -> list:
    """Per-player ``[direction x, direction y, magnitude]`` of the analytic pull toward the role target.

    Supporters and defenders add the push away from crowding teammates to the pull toward their
    target; the carrier, pass chaser, presser and keepers use their own pulls. Values are rounded
    (direction to 3 decimals, magnitude to 4) for replay telemetry and are ``[0.0, 0.0, 0.0]`` when
    the pull vanishes.
    """
    pos = state.pos.tolist()
    bx = float(state.ball[0])
    by = float(state.ball[1])
    holder = int(state.holder)
    ball_dist = np.hypot(state.pos[:, 0] - bx, state.pos[:, 1] - by).tolist()
    disabled = state.disabled.tolist()
    side = state.team_side.tolist()
    role = res.role_code.tolist()
    value = res.phi_role.tolist()
    target = res.target.tolist()
    holder_team = -1 if holder < 0 else (0 if holder < 6 else 1)
    flight_team = int(state.last_ball_team) if (holder < 0 and state.ball_in_flight) else -1
    r_crowd = rew.teammate_crowd_radius
    w_crowd = rew.teammate_crowd_weight
    inv_r_crowd = 1.0 / max(1e-6, r_crowd)

    arrows = [[0.0, 0.0, 0.0] for _ in range(12)]
    for g in range(12):
        r = role[g]
        team = 0 if g < 6 else 1
        px, py = pos[g]
        if r == ROLE_BALL:
            arrows[g] = [round(1.0 if side[team] == SIDE_LEFT else -1.0, 3), 0.0, 0.02]
        elif r == ROLE_CHASE:
            d_safe = max(1e-4, ball_dist[g])
            arrows[g] = [
                round((bx - px) / d_safe, 3),
                round((by - py) / d_safe, 3),
                round(value[g] * _INV_LAMBDA_CHASE, 4),
            ]
        elif r == ROLE_PRESS:
            d_safe = max(1e-4, ball_dist[g])
            gx = (bx - px) / d_safe * (value[g] * _INV_LAMBDA_CHASE)
            gy = (by - py) / d_safe * (value[g] * _INV_LAMBDA_CHASE)
            g_mag = math.hypot(gx, gy)
            if g_mag > 1e-6:
                arrows[g] = [round(gx / g_mag, 3), round(gy / g_mag, 3), round(g_mag, 4)]
        elif r == ROLE_GK:
            dx = target[g][0] - px
            dy = target[g][1] - py
            d_gk = math.hypot(dx, dy)
            g_mag = value[g] * 2.0 * _INV_2_95_SQ * d_gk
            if d_gk > 1e-4 and g_mag > 1e-6:
                arrows[g] = [round(dx / d_gk, 3), round(dy / d_gk, 3), round(g_mag, 4)]
        else:
            if holder_team == team or flight_team == team:
                tgt_x, tgt_y = target[g]
                inv_sigma = _INV_2_240_SQ
            else:
                tgt_x = round(target[g][0], 1)
                tgt_y = round(target[g][1], 1)
                inv_sigma = _INV_2_280_SQ
            dx = tgt_x - px
            dy = tgt_y - py
            d_t = math.hypot(dx, dy)
            pull_mag = value[g] * 2.0 * inv_sigma * d_t
            ux = (dx / d_t) if d_t > 1e-4 else 0.0
            uy = (dy / d_t) if d_t > 1e-4 else 0.0
            push_x = 0.0
            push_y = 0.0
            f0 = 1 + 6 * team
            for j in range(f0, f0 + 5):
                if j == g or disabled[j]:
                    continue
                d_ij = math.hypot(px - pos[j][0], py - pos[j][1])
                if d_ij < r_crowd and d_ij > 1e-4:
                    viol = (r_crowd - d_ij) * inv_r_crowd
                    coeff = 2.0 * w_crowd * viol * inv_r_crowd / d_ij
                    push_x += coeff * (px - pos[j][0])
                    push_y += coeff * (py - pos[j][1])
            gx = ux * pull_mag + push_x
            gy = uy * pull_mag + push_y
            g_mag = math.hypot(gx, gy)
            if g_mag > 1e-6:
                arrows[g] = [round(gx / g_mag, 3), round(gy / g_mag, 3), round(g_mag, 4)]
    return arrows
