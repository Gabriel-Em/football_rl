"""Deterministic scripted opponents. `scripted_actions` is an evaluation benchmark and also the training opponent
of the trainer's `--bot-envs`; `passing_actions` is an evaluation-only opponent.

`scripted_actions(env, team)` reads the true engine state and returns one `(6, 4)` action block
`[throttle, steer, kick, special]` per slot (GK, D1, D2, M1, M2, ST). It has no random component and
does not modify the environment.

Behaviour, evaluated in the team's canonical frame (own goal at x = 0, attacking toward +x; a team that
defends the right goal is point-reflected, so both teams play identically):
  * The two outfielders nearest the ball chase it; the other three hold role anchors at depths
    {0.20, 0.20, 0.42, 0.42, 0.62} x field width plus 0.4 x the ball's distance from the own goal line,
    at y {300, 540, 200, 640, 420}.
  * An outfield ball carrier runs at the centre of the opposing goal and kicks at full power once it is
    inside 380 u of that point and faces it within 0.15 rad.
  * The goalkeeper stands 45 u off its line at the ball's y clamped to [360, 480] (the goal-line
    bisector) and walks to a loose ball that comes within 120 u of its goal centre. A keeper holding the
    ball kicks once it faces its bisector target within 0.2 rad (the engine redirects goalkeeper kicks
    downfield).
  * A chaser within 45 u of an opponent ball holder presses the special (tackle) action.
Steering is the heading error to the current target times 2, clipped to [-1, 1]; throttle is 1.0 when the
target is farther than 25 u and within 1.2 rad of dead ahead, 0.2 when it is farther than 25 u but
behind, and 0 inside 25 u.

`passing_actions(env, team)` is a second opponent that is held out of every training run; it must never be
trained against. It defends, chases, tackles and keeps goal exactly like `scripted_actions` (the same code)
and differs only in how its team attacks:
  * While its team holds the ball, the non-chasing midfielders and the striker move their anchors
    0.12 x field width forward and 120 u away from the centre line y = 420 (a striker anchored on the
    centre line only moves forward).
  * An outfield carrier that is not shooting under the rule above passes when pressed: an opposing
    outfielder within 240 u, or within 320 u and between the carrier and the goal it attacks. The target
    is the teammate with the largest gain in x over the carrier (more than 0) who is open: no opponent
    within 40 u of the receiver and none within 25 u of the segment from the carrier to the receiver.
    The carrier steers to face the target and kicks once it faces it within 0.15 rad, at kick value
    trigger + (1 - trigger) x clip(pass length / 600 u, 0.3, 1.0), where trigger is the engine's kick
    threshold. A pressed carrier with no open target dribbles at the goal as `scripted_actions` does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["scripted_actions", "passing_actions"]

NUM_OUTFIELD = 5
TEAM_SIZE = 6
CHASERS = 2

ANCHOR_DEPTH = np.array([0.20, 0.20, 0.42, 0.42, 0.62])
ANCHOR_Y = np.array([300.0, 540.0, 200.0, 640.0, 420.0])
ANCHOR_BALL_FOLLOW = 0.4
ANCHOR_X_MARGIN = 80.0

KEEPER_LINE_OFFSET = 45.0
KEEPER_Y_MIN = 360.0
KEEPER_Y_MAX = 480.0
KEEPER_BALL_REACH = 120.0

SHOOT_RANGE = 380.0
SHOOT_ALIGNMENT = 0.15
KEEPER_KICK_ALIGNMENT = 0.2
TACKLE_RANGE = 45.0

ARRIVE_RADIUS = 25.0
FACING_LIMIT = 1.2
STEER_GAIN = 2.0
TURNING_THROTTLE = 0.2

THROTTLE, STEER, KICK, SPECIAL = 0, 1, 2, 3
ATTACKER_OUTFIELD_SLOTS = slice(2, NUM_OUTFIELD)    # M1, M2, ST within the five outfielders

ATTACK_ADVANCE = 0.12             # anchor shift toward the opposing goal, in field widths
ATTACK_WIDENING = 120.0           # anchor shift away from the centre line, in u
PRESS_NEAR = 240.0
PRESS_GOAL_SIDE = 320.0
OPEN_RECEIVER = 40.0
OPEN_LINE = 25.0
PASS_ALIGNMENT = 0.15
PASS_FULL_POWER_LENGTH = 600.0
PASS_MIN_POWER_SHARE = 0.3


@dataclass(frozen=True)
class _View:
    """One team's side of the engine state in its canonical frame (own goal at x = 0, attacking toward +x)."""

    width: float
    height: float
    goal_y: float
    kick_trigger: float
    pos: np.ndarray              # (6, 2) own players
    ang: np.ndarray              # (6,)
    ball: np.ndarray             # (2,)
    opponents: np.ndarray        # (6, 2) opposing players
    carrier: int                 # own slot holding the ball, -1 if the team does not hold it
    loose: bool                  # nobody holds the ball
    opp_holder_pos: Optional[np.ndarray]


def _view(env, team: int) -> _View:
    phys = env.cfg.physics
    width = float(phys.field_width)
    height = float(phys.field_height)
    goal_y = 0.5 * (float(phys.goal_y_min) + float(phys.goal_y_max))

    base = TEAM_SIZE * team
    opp_base = TEAM_SIZE * (1 - team)
    flip = int(env.team_side[team]) != 0          # defends the right goal: reflect into the canonical frame
    corner = np.array([width, height])

    def canon_pos(p):
        return corner - p if flip else p

    holder = env.ball_holder
    carrier = -1
    opp_holder_pos = None
    if holder is not None:
        if base <= holder < base + TEAM_SIZE:
            carrier = holder - base
        else:
            opp_holder_pos = canon_pos(env.player_pos[holder])

    return _View(
        width=width,
        height=height,
        goal_y=goal_y,
        kick_trigger=float(env.cfg.mechanics.kick_trigger_threshold),
        pos=canon_pos(env.player_pos[base:base + TEAM_SIZE]),
        ang=env.player_angle[base:base + TEAM_SIZE] + (np.pi if flip else 0.0),
        ball=canon_pos(env.ball_pos),
        opponents=canon_pos(env.player_pos[opp_base:opp_base + TEAM_SIZE]),
        carrier=carrier,
        loose=holder is None,
        opp_holder_pos=opp_holder_pos,
    )


def _plan(v: _View):
    """Targets `(tx, ty)` of the six slots (keeper, chasers, anchors) and the chaser mask over the five outfielders."""
    outfield = v.pos[1:]
    chaser = np.zeros(NUM_OUTFIELD, dtype=bool)
    chaser[np.argsort(np.hypot(outfield[:, 0] - v.ball[0], outfield[:, 1] - v.ball[1]), kind="stable")[:CHASERS]] = True

    tx = np.empty(TEAM_SIZE)
    ty = np.empty(TEAM_SIZE)
    tx[0] = KEEPER_LINE_OFFSET
    ty[0] = min(max(v.ball[1], KEEPER_Y_MIN), KEEPER_Y_MAX)
    if v.loose and np.hypot(v.ball[0], v.ball[1] - v.goal_y) < KEEPER_BALL_REACH:
        tx[0], ty[0] = v.ball
    tx[1:] = np.clip(v.width * ANCHOR_DEPTH + ANCHOR_BALL_FOLLOW * v.ball[0], ANCHOR_X_MARGIN, v.width - ANCHOR_X_MARGIN)
    ty[1:] = ANCHOR_Y
    tx[1:][chaser] = v.ball[0]
    ty[1:][chaser] = v.ball[1]
    return tx, ty, chaser


def _drive(pos: np.ndarray, ang: np.ndarray, tx: np.ndarray, ty: np.ndarray):
    """Throttle and steer toward `(tx, ty)` for each row, and the heading errors; kick and special are -1."""
    dx = tx - pos[:, 0]
    dy = ty - pos[:, 1]
    heading_error = (np.arctan2(dy, dx) - ang + np.pi) % (2.0 * np.pi) - np.pi
    dist = np.hypot(dx, dy)

    out = np.empty((len(tx), 4), dtype=np.float32)
    out[:, THROTTLE] = np.where(dist > ARRIVE_RADIUS, np.where(np.abs(heading_error) < FACING_LIMIT, 1.0, TURNING_THROTTLE), 0.0)
    out[:, STEER] = np.clip(STEER_GAIN * heading_error, -1.0, 1.0)
    out[:, KICK] = -1.0
    out[:, SPECIAL] = -1.0
    return out, heading_error


def _act(v: _View, tx: np.ndarray, ty: np.ndarray, chaser: np.ndarray) -> np.ndarray:
    """Drive every slot to its target, then apply the keeper kick, the shot and the tackle rules."""
    out, heading_error = _drive(v.pos, v.ang, tx, ty)
    carrier = v.carrier
    if carrier == 0:
        if abs(heading_error[0]) < KEEPER_KICK_ALIGNMENT:
            out[0, KICK] = 1.0
    elif carrier > 0:
        to_goal = np.hypot(v.width - v.pos[carrier, 0], v.goal_y - v.pos[carrier, 1])
        if to_goal < SHOOT_RANGE and abs(heading_error[carrier]) < SHOOT_ALIGNMENT:
            out[carrier, KICK] = 1.0

    if v.opp_holder_pos is not None:
        outfield = v.pos[1:]
        near = np.hypot(outfield[:, 0] - v.opp_holder_pos[0], outfield[:, 1] - v.opp_holder_pos[1]) < TACKLE_RANGE
        out[1:, SPECIAL][chaser & near] = 1.0
    return out


def scripted_actions(env, team: int) -> np.ndarray:
    """Return the `(6, 4)` float32 action block of `team` (0 = players 0..5, 1 = players 6..11)."""
    v = _view(env, team)
    tx, ty, chaser = _plan(v)
    if v.carrier > 0:
        tx[v.carrier] = v.width
        ty[v.carrier] = v.goal_y
    return _act(v, tx, ty, chaser)


def _push_up(v: _View, tx: np.ndarray, ty: np.ndarray, chaser: np.ndarray) -> None:
    """Move the anchors of the non-chasing midfielders and striker forward and away from the centre line."""
    free = np.zeros(NUM_OUTFIELD, dtype=bool)
    free[ATTACKER_OUTFIELD_SLOTS] = True
    free &= ~chaser
    tx[1:][free] = np.clip(tx[1:][free] + ATTACK_ADVANCE * v.width, ANCHOR_X_MARGIN, v.width - ANCHOR_X_MARGIN)
    ty[1:][free] += ATTACK_WIDENING * np.sign(ty[1:][free] - 0.5 * v.height)


def _is_pressed(v: _View) -> bool:
    """An opposing outfielder is within 240 u of the carrier, or within 320 u and goal-side of it."""
    carrier_pos = v.pos[v.carrier]
    rel = v.opponents[1:] - carrier_pos
    dist = np.hypot(rel[:, 0], rel[:, 1])
    goal_side = v.opponents[1:, 0] > carrier_pos[0]
    return bool(np.any((dist < PRESS_NEAR) | ((dist < PRESS_GOAL_SIDE) & goal_side)))


def _segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Distance of each of `points` (n, 2) to the segment `start` -> `end`."""
    segment = end - start
    along = np.clip(((points - start) @ segment) / (segment @ segment), 0.0, 1.0)
    nearest = start + along[:, None] * segment
    return np.hypot(points[:, 0] - nearest[:, 0], points[:, 1] - nearest[:, 1])


def _open_receiver(v: _View) -> int:
    """Slot of the open teammate with the largest gain in x over the carrier (lowest slot on a tie), or -1."""
    origin = v.pos[v.carrier]
    best, best_gain = -1, 0.0
    for slot in range(1, TEAM_SIZE):
        gain = v.pos[slot, 0] - origin[0]
        if slot == v.carrier or gain <= best_gain:
            continue
        to_receiver = np.hypot(v.opponents[:, 0] - v.pos[slot, 0], v.opponents[:, 1] - v.pos[slot, 1])
        if np.all(to_receiver >= OPEN_RECEIVER) and np.all(_segment_distance(v.opponents, origin, v.pos[slot]) >= OPEN_LINE):
            best, best_gain = slot, gain
    return best


def _pass_row(v: _View, slot: int) -> np.ndarray:
    """The carrier's action block when it passes to `slot`: face the receiver, kick once aligned."""
    c = v.carrier
    out, heading_error = _drive(v.pos[c:c + 1], v.ang[c:c + 1], v.pos[slot:slot + 1, 0], v.pos[slot:slot + 1, 1])
    if abs(heading_error[0]) < PASS_ALIGNMENT:
        length = np.hypot(*(v.pos[slot] - v.pos[c]))
        share = np.clip(length / PASS_FULL_POWER_LENGTH, PASS_MIN_POWER_SHARE, 1.0)
        out[0, KICK] = v.kick_trigger + (1.0 - v.kick_trigger) * share
    return out[0]


def passing_actions(env, team: int) -> np.ndarray:
    """Return the `(6, 4)` float32 action block of `team`; see the module docstring. Used for evaluation only."""
    v = _view(env, team)
    tx, ty, chaser = _plan(v)
    if v.carrier >= 0:
        _push_up(v, tx, ty, chaser)
    if v.carrier > 0:
        tx[v.carrier] = v.width
        ty[v.carrier] = v.goal_y
    out = _act(v, tx, ty, chaser)
    shooting = v.carrier > 0 and out[v.carrier, KICK] > 0.0
    if v.carrier > 0 and not shooting and _is_pressed(v):
        receiver = _open_receiver(v)
        if receiver >= 0:
            out[v.carrier] = _pass_row(v, receiver)
    return out
