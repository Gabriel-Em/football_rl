"""Deterministic scripted opponent used as an evaluation benchmark (never as a training signal).

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
"""

from __future__ import annotations

import numpy as np

__all__ = ["scripted_actions"]

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


def scripted_actions(env, team: int) -> np.ndarray:
    """Return the `(6, 4)` float32 action block of `team` (0 = players 0..5, 1 = players 6..11)."""
    phys = env.cfg.physics
    width = float(phys.field_width)
    height = float(phys.field_height)
    goal_y = 0.5 * (float(phys.goal_y_min) + float(phys.goal_y_max))

    base = TEAM_SIZE * team
    flip = int(env.team_side[team]) != 0          # defends the right goal: reflect into the canonical frame
    corner = np.array([width, height])

    def canon_pos(p):
        return corner - p if flip else p

    pos = canon_pos(env.player_pos[base:base + TEAM_SIZE])
    ang = env.player_angle[base:base + TEAM_SIZE] + (np.pi if flip else 0.0)
    ball = canon_pos(env.ball_pos)
    holder = env.ball_holder

    carrier = -1
    opp_holder_pos = None
    if holder is not None:
        if base <= holder < base + TEAM_SIZE:
            carrier = holder - base
        else:
            opp_holder_pos = canon_pos(env.player_pos[holder])

    outfield = pos[1:]
    chaser = np.zeros(NUM_OUTFIELD, dtype=bool)
    chaser[np.argsort(np.hypot(outfield[:, 0] - ball[0], outfield[:, 1] - ball[1]), kind="stable")[:CHASERS]] = True

    tx = np.empty(TEAM_SIZE)
    ty = np.empty(TEAM_SIZE)
    tx[0] = KEEPER_LINE_OFFSET
    ty[0] = min(max(ball[1], KEEPER_Y_MIN), KEEPER_Y_MAX)
    if holder is None and np.hypot(ball[0], ball[1] - goal_y) < KEEPER_BALL_REACH:
        tx[0], ty[0] = ball
    tx[1:] = np.clip(width * ANCHOR_DEPTH + ANCHOR_BALL_FOLLOW * ball[0], ANCHOR_X_MARGIN, width - ANCHOR_X_MARGIN)
    ty[1:] = ANCHOR_Y
    tx[1:][chaser] = ball[0]
    ty[1:][chaser] = ball[1]
    if carrier > 0:
        tx[carrier] = width
        ty[carrier] = goal_y

    dx = tx - pos[:, 0]
    dy = ty - pos[:, 1]
    heading_error = (np.arctan2(dy, dx) - ang + np.pi) % (2.0 * np.pi) - np.pi
    dist = np.hypot(dx, dy)

    out = np.empty((TEAM_SIZE, 4), dtype=np.float32)
    out[:, 0] = np.where(dist > ARRIVE_RADIUS, np.where(np.abs(heading_error) < FACING_LIMIT, 1.0, TURNING_THROTTLE), 0.0)
    out[:, 1] = np.clip(STEER_GAIN * heading_error, -1.0, 1.0)
    out[:, 2] = -1.0
    out[:, 3] = -1.0

    if carrier == 0:
        if abs(heading_error[0]) < KEEPER_KICK_ALIGNMENT:
            out[0, 2] = 1.0
    elif carrier > 0:
        to_goal = np.hypot(width - pos[carrier, 0], goal_y - pos[carrier, 1])
        if to_goal < SHOOT_RANGE and abs(heading_error[carrier]) < SHOOT_ALIGNMENT:
            out[carrier, 2] = 1.0

    if opp_holder_pos is not None:
        near = np.hypot(outfield[:, 0] - opp_holder_pos[0], outfield[:, 1] - opp_holder_pos[1]) < TACKLE_RANGE
        out[1:, 3][chaser & near] = 1.0
    return out
