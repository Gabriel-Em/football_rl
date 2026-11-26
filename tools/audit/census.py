"""Behaviour census over harness traces (`tools/audit/harness.py`).

Every metric is computed per match from one trace and aggregated across matches. A "decision" is one
trace row (1/15 s). Slots are GK, D1, D2, M1, M2, ST (player index % 6); players 0..5 are team A,
6..11 team B; "outfield" means slots 1..5. Distances are in field units (u), times in seconds.

Aggregation (`report`): per-match scalars are summarised as mean +- SE with n matches, where
SE = sample sd / sqrt(n). Event-level shares (kick outcomes, keeper pickups, goal types, mean holder
spell) are pooled ratios of per-match counts, with the cluster-robust ratio-estimator SE
sqrt(n / (n - 1) * sum_i (y_i - r x_i)^2) / sum_i x_i, which treats a match as the sampling unit.

Metrics
  hold_share        fraction of decisions in which a player of the team holds the ball (per team).
  loose_share       fraction of decisions in which nobody holds the ball (a ball in flight is loose).
  holder_spell_s    duration in seconds of an uninterrupted spell of the same holder (mean pooled over
                    spells, median; a spell ends when the holder changes or the ball is released).
  kick_outcome      for every kick (a change of a player's last-kick time) the first event within 60
                    decisions (4 s) after the kick step: `goal` (the score changed, either team, counted
                    from the kick step on), `teammate` (another player of the kicker's team holds the
                    ball), `opponent` (a player of the other team holds it), `self` (the kicker holds it
                    again after more than 8 decisions; earlier holds are the release itself), otherwise
                    `none`. A kick whose window runs past the end of the trace without an event is
                    censored (omitted). Reported for outfield kicks.
  pass options      for every outfield kick (both teams pooled) at step k by player i, from `pos[k]`, `ball[k]` and
                    `bvel[k + 1]` (a kick on the last row has no k + 1 and enters no key):
                    a candidate is an outfield teammate j != i with 80 <= |pos_j - ball| <= 700 u; its lane is open
                    when every opponent (all six, keeper included) projects onto the segment ball -> pos_j outside the
                    open interval (0, |pos_j - ball|) or lies strictly more than 50 u from the line; the kick direction
                    is bvel[k + 1] / |bvel[k + 1]|, and a kick with |bvel[k + 1]| < 50 u/s has no direction; the kick is
                    aimed at j when the angle between its direction and pos_j - ball is <= 15 deg.
                    `pass_option_share`: kicks with at least one open candidate, over kicks.
                    `aim_open_share`: kicks aimed at an open candidate, over kicks that have one and a direction.
                    `aim_open_completion`: kicks aimed at an open candidate whose `kick_outcome` is `teammate`, over
                    kicks aimed at an open candidate that have an outcome.
                    `no_option_opponent_share`: kicks without an open candidate whose `kick_outcome` is `opponent`,
                    over kicks without one that have an outcome.
                    The outcome is the `kick_outcome` above; a censored kick enters only the first two keys.
                    Chance baselines for the aim, over kicks with an open candidate and a direction, with facing =
                    `ang[k, i]` (the kicker's body angle at the kick row), theta_j = the angle of pos_j - ball for each open
                    candidate j, and angle differences wrapped to (-pi, pi]:
                    `open_reachable_share`: kicks with an open candidate within 50 deg of the facing (35 deg of kick
                    deflection plus the 15 deg aim cone), over kicks with an open candidate and a direction.
                    `aim_open_reachable_share`: kicks aimed at an open candidate among those reachable kicks, over them.
                    `aim_open_arc_baseline`: over the reachable kicks, the mean share of the 71 angles facing + {-35, -34,
                    ..., +35} deg lying within 15 deg (inclusive) of some open theta_j: the aim share of a kick sent in a
                    uniformly random direction of the kickable arc. A trace without `ang` enters none of the three keys.
  basin keys        outfield seats only (slots 1..5), per team A / B and pooled, as per-match counts pooled across matches
                    like the shares above; a trace without `ang` or without `vel` enters none of the three keys, and a
                    player-decision with a non-finite value among the quantities a key reads enters neither its numerator
                    nor its denominator. A seat's team is its own team.
                    `reverse_motion_share`: outfield player-decisions with speed > 40 u/s and cos(velocity, facing) < -0.5
                    (the body moves within 60 deg of straight backwards; facing = `ang`, velocity = `vel`), over all
                    outfield player-decisions.
                    `facing_ball_share`: outfield player-decisions whose bearing to the ball is within 60 deg (inclusive) of
                    the facing, over all outfield player-decisions; a player exactly on the ball has no bearing and counts
                    as facing it.
                    `tackle_ready_share`: over decisions in which an outfield player holds the ball, the opposing outfield
                    players within 85 u (inclusive) of the carrier that face it, i.e. cos(facing, carrier - tackler) >= 0.25
                    (the facing gate of a tackle; a tackler standing on the carrier reads 1), over all opposing outfield
                    players within 85 u of the carrier. The team is the tacklers' team. Decisions with a loose ball or a
                    keeper holding it enter nothing.
                    The report line is `basin: reverse X | facing ball Y | tackle-ready Z` (pooled shares).
  defence keys      per defending team: `<key>_A` and `<key>_B` are the values of the team that defends (A defends against B's
                    attacks), `<key>_pooled` pools both. Canonical x is the distance from the defending team's own goal line
                    (0..1400, `side` gives the team's goal); a goal-side outfielder has canonical x strictly below the ball's. Defenders
                    are the five outfield seats. A shot is a rise of the team's `s_shots` counter on row k; its shooter is the holder on
                    row k (a shot on a row without a holder of the shooting team raises ValueError). The per-match records carry the
                    counts (`defence_counts`), `summarize` the shares (`defence_metrics`): ratios of per-match counts with the cluster-robust
                    SE, except `tackle_out_of_range_per_match` (mean per match, pooled = A + B). A key is None (mean and SE) when its event
                    count over the matches is below `DEFENCE_MIN_EVENTS` (10 shots, regains reaching a shot or chance spells) or, for the
                    two decision-based keys, `DEFENCE_MIN_DECISIONS` (100); `n` still counts the matches that had events.
                    `goal_side_at_goal`: over the shots after which the shooting team scores within 45 decisions (3 s; the goal shots),
                    the mean number of the conceding team's outfielders goal-side of the ball on the shot row.
                    `shot_nearest_def_facing` / `shot_nearest_def_closing`: over every shot, the conceding outfielder nearest the shooter on
                    the shot row faces it (cos(body angle, bearing to the shooter) >= 0.25, the tackle gate) / has a mean velocity component
                    toward the shooter above 20 u/s over the 8 decisions before the shot (rows k - 8 .. k - 1, those that exist; the
                    shooter and defender positions of each row). Shares over shots; a shot with a non-finite input enters neither.
                    `regain_goalside_at_shot` / `regain_goalside_retention`: a regain is the first row held by the opponent after the last
                    held row was the defending team's (the ball may be loose in between); its spell runs until the defending team holds
                    again or the match ends. Over the regains whose spell holds a shot of the opponent (the first rise of its `s_shots`
                    after the regain row), the mean number of the defending team's outfielders goal-side on the shot row, and the mean over
                    those regains that had a goal-side outfielder on the regain row of the share of them still goal-side on the shot row.
                    `tackle_in_range_rate`, `tackle_win_rate`, `tackle_out_of_range_per_match`: on rows where an opposing outfielder holds the
                    ball (a keeper carrier enters nothing), a chance row of defender i is one with the carrier within the tackle reach of
                    its slot (distance - 28 u of body contact <= 85, 85, 75, 75, 65 u for slots 1..5) and i facing it (the tackle gate);
                    a chance spell is a maximal run of consecutive chance rows of one defender against one carrier. In-range rate:
                    spells in which i fired a tackle (`act[..., 3] > 0.5`) over spells. Win rate: spells in which a fired tackle was
                    followed by i holding the ball on the same row or within the next 3 decisions, over spells with a fire. Out of range per
                    match: tackle decisions by a defender against a carrier within 180 u (the whiff range) on rows outside that defender's
                    own chance rows. Traces without `act` or `ang` enter none of them.
                    `deep_block_ge4`: over the team-decisions with the ball at canonical x >= 700 (the opponent's half), the share in which
                    at least 4 of the 5 outfielders stand at canonical x <= 280 (a fifth of the pitch). `outfield_depth_oop`: the mean over
                    the decisions in which the opponent holds the ball of the mean canonical x of the five outfielders.
  trap_flag_g50     `trap_flag_g50(census)` on a generation-50 `self_sampled` census: `facing_ball_share_pooled <= 0.35` and
                    `goals_total < 0.5`.
  n_within(R)       mean number of a team's outfield players strictly closer than R u to the ball per
                    decision (per team). Uniform random placement gives 0.123 (R=100), 0.266 (R=150).
  cluster_ge4       fraction of decisions in which at least 4 of a team's 5 outfield players form one
                    connected group when two players are linked if strictly closer than 90 u (per team).
  role_inversion    fraction of decisions in which D1 or D2 is strictly more than 30 u ahead of the same
                    team's striker along the attack axis (per team).
  gk_opportunity    an episode of consecutive decisions in which the ball is loose, inside the keeper's
                    box (distance from the own goal centre <= the active box radius), slower than
                    80 u/s, and at most 120 u from the keeper. `pickup`: the keeper holds the ball
                    during the episode or within 2 decisions after it; `within30`: the keeper came
                    strictly closer than 30 u to the ball (the decision that closes the episode
                    included). Reported as episodes per keeper-match, pickup rate, within-30 rate and
                    pickup rate given within-30.
  goal_types        per goal: own goal or not; for open-play goals the scoring team's last kick, whether
                    the goal was `direct` (that kick was at most 60 decisions earlier and nobody held the
                    ball afterwards), seconds since that kick, kicker distance to the attacked goal
                    centre at the kick, kicker slot, and completed passes by the scoring team in the
                    preceding 150 decisions (10 s).

Usage:
  python tools/audit/census.py TRACE_DIR [TRACE_DIR ...] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from football_rl.config import DEFAULT_CONFIG  # noqa: E402

__all__ = [
    "hold_share", "loose_share", "holder_spell_intervals", "holder_spells", "kick_events", "kick_outcomes",
    "outfield_outcome_counts", "pass_options", "pass_option_counts", "aim_reach_counts", "basin_counts", "defence_counts", "defence_metrics",
    "trap_flag_g50", "kick_propensity_counts", "kick_propensity", "n_within", "max_component_size",
    "cluster_ge4", "role_inversion", "gk_geometry", "gk_opportunities", "goal_events", "goal_type_counts",
    "time_binned_counts", "time_binned_metrics", "match_metrics", "mean_se", "ratio_se", "summarize", "report",
    "load_traces", "main",
]

STEP_SECONDS = 1.0 / float(DEFAULT_CONFIG.physics.rl_hz)
FIELD_WIDTH = float(DEFAULT_CONFIG.physics.field_width)
GOAL_Y = 0.5 * (float(DEFAULT_CONFIG.physics.goal_y_min) + float(DEFAULT_CONFIG.physics.goal_y_max))

OUTCOMES = ("teammate", "opponent", "self", "none", "goal")
KICK_WINDOW_STEPS = 60
SAME_PLAYER_MIN_STEPS = 8
NEVER_KICKED = -999.0
KICK_TIME_VALID = -998.0
PASS_WINDOW_STEPS = 150
DIRECT_KICK_STEPS = 60
LINK_RADIUS = 90.0
CLUSTER_MIN = 4
INVERSION_MARGIN = 30.0
GK_REACH = 120.0
GK_MAX_BALL_SPEED = 80.0
GK_PICKUP_GRACE_STEPS = 2
GK_CLOSE = 30.0
GK_BOX_RADIUS = 195.0
GK_GOAL_LINE_X = 19.5
GK_QUICK_KICK_S = 0.5
PASS_MIN_DIST = 80.0
PASS_MAX_DIST = 700.0
LANE_HALF_WIDTH = 50.0
AIM_MIN_SPEED = 50.0
AIM_CONE_DEG = 15.0
REACH_DEG = 50.0
ARC_GRID_DEG = np.arange(-35.0, 36.0, 1.0)  # kickable arc: facing + 35 deg of kick deflection either side, 1 deg apart
AIM_REACH_COUNT_KEYS = ("open_reach_den", "open_reach_n", "aim_reach_n", "arc_reach_sum")
PASS_OPTION_COUNT_KEYS = (
    "pass_opt_n", "pass_opt_with", "aim_open_den", "aim_open_n",
    "aim_comp_den", "aim_comp_mate", "noopt_den", "noopt_opp",
)
REVERSE_MIN_SPEED = 40.0
REVERSE_COS_MAX = -0.5
FACING_BALL_DEG = 60.0
FACING_BALL_COS = math.cos(math.radians(FACING_BALL_DEG))
TACKLE_RADIUS = 85.0
TACKLE_FACING_DOT = 0.25
TACKLE_ON_CARRIER = 1e-4
BASIN_KEYS = ("reverse_motion", "facing_ball", "tackle_ready")
BASIN_PARTS = ("n", "den")
BASIN_GROUPS = ("A", "B", "pooled")
BASIN_COUNT_KEYS = tuple(f"{k}_{p}_{g}" for k in BASIN_KEYS for p in BASIN_PARTS for g in BASIN_GROUPS)
BASIN_BLOCK_ROWS = 768
LATE_WINDOW_STEPS = 225
TEAM_SIZE = 6
TEAMS = ("A", "B")
DEFENCE_MIN_EVENTS = 10
DEFENCE_MIN_DECISIONS = 100
GOAL_AFTER_SHOT_STEPS = 45
SHOT_APPROACH_STEPS = 8
CLOSING_MIN_SPEED = 20.0
DEEP_BLOCK_MIN_BALL_X = 0.5 * FIELD_WIDTH
DEEP_BLOCK_MAX_X = FIELD_WIDTH / 5.0
DEEP_BLOCK_MIN_PLAYERS = 4
TACKLE_WIN_STEPS = 3
TACKLE_CONTACT_OFFSET = 2.0 * float(DEFAULT_CONFIG.physics.player_radius) - 8.0
TACKLE_WHIFF_RANGE = float(DEFAULT_CONFIG.mechanics.tackle_whiff_max_ball_dist)
# tackle reach beyond body contact by slot 1..5 (D1, D2, M1, M2, ST)
TACKLE_REACH = np.array([
    DEFAULT_CONFIG.mechanics.tackle_radius_def, DEFAULT_CONFIG.mechanics.tackle_radius_def,
    DEFAULT_CONFIG.mechanics.tackle_radius_mid, DEFAULT_CONFIG.mechanics.tackle_radius_mid,
    DEFAULT_CONFIG.mechanics.tackle_radius_atk,
], dtype=np.float64)
TRAP_FACING_MAX = 0.35
TRAP_GOALS_BELOW = 0.5
# key -> (numerator count, denominator count, count that must reach the minimum, kind of minimum); each count has the suffixes _A, _B, _pooled
DEFENCE_SHARES = {
    "goal_side_at_goal": ("goal_side_at_goal_sum", "goal_side_at_goal_den", "goal_side_at_goal_den", "events"),
    "shot_nearest_def_facing": ("shot_nearest_def_facing_n", "shot_nearest_def_facing_den", "shot_nearest_def_facing_den", "events"),
    "shot_nearest_def_closing": ("shot_nearest_def_closing_n", "shot_nearest_def_closing_den", "shot_nearest_def_facing_den", "events"),
    "regain_goalside_at_shot": ("regain_goalside_at_shot_sum", "regain_goalside_at_shot_den", "regain_goalside_at_shot_den", "events"),
    "regain_goalside_retention": ("regain_goalside_retention_sum", "regain_goalside_retention_den", "regain_goalside_at_shot_den", "events"),
    "tackle_in_range_rate": ("tackle_in_range_rate_n", "tackle_in_range_rate_den", "tackle_in_range_rate_den", "events"),
    "tackle_win_rate": ("tackle_win_rate_n", "tackle_win_rate_den", "tackle_in_range_rate_den", "events"),
    "deep_block_ge4": ("deep_block_ge4_n", "deep_block_ge4_den", "deep_block_ge4_den", "decisions"),
    "outfield_depth_oop": ("outfield_depth_oop_sum", "outfield_depth_oop_den", "outfield_depth_oop_den", "decisions"),
}
DEFENCE_COUNT_KEYS = tuple(
    f"{part}_{group}" for num, den, _, _ in DEFENCE_SHARES.values() for part in (num, den) for group in BASIN_GROUPS
) + tuple(f"tackle_out_of_range_per_match_{group}" for group in BASIN_GROUPS)
# row = the carrier's team; True at the outfield seats of the other team (the players that can tackle the carrier)
_OPPOSING_OUTFIELD = np.zeros((2, 2 * TEAM_SIZE), dtype=bool)
for _carrier_team in (0, 1):
    _OPPOSING_OUTFIELD[_carrier_team, TEAM_SIZE * (1 - _carrier_team) + 1:TEAM_SIZE * (2 - _carrier_team)] = True
ROLE_GROUPS = {"GK": (0,), "D": (1, 2), "M": (3, 4), "ST": (5,)}


# ----------------------------------------------------------------------------
# Possession
# ----------------------------------------------------------------------------
def _holder(trace: Dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(trace["holder"]).astype(np.int64)


def hold_share(trace: Dict[str, np.ndarray]) -> np.ndarray:
    """Fraction of decisions in which team A / team B holds the ball, shape (2,)."""
    h = _holder(trace)
    if h.size == 0:
        return np.full(2, np.nan)
    return np.array([((h >= 0) & (h < TEAM_SIZE)).mean(), (h >= TEAM_SIZE).mean()])


def loose_share(trace: Dict[str, np.ndarray]) -> float:
    h = _holder(trace)
    return float("nan") if h.size == 0 else float((h < 0).mean())


def holder_spell_intervals(trace: Dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(players, starts, ends)` for every uninterrupted holding spell (`[start, end)`)."""
    h = _holder(trace)
    if h.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    changes = np.flatnonzero(np.diff(h) != 0) + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [h.size]])
    keep = h[starts] >= 0
    return h[starts][keep], starts[keep], ends[keep]


def holder_spells(trace: Dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Holder (player index) and duration in seconds of every uninterrupted holding spell."""
    players, starts, ends = holder_spell_intervals(trace)
    return players, (ends - starts) * STEP_SECONDS


# ----------------------------------------------------------------------------
# Kicks
# ----------------------------------------------------------------------------
def kick_events(trace: Dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Steps and players of every kick: the player's last-kick time changed to a valid time at that step."""
    lk = np.asarray(trace["lastkick"], dtype=np.float64)
    if lk.shape[0] == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    previous = np.vstack([np.full((1, lk.shape[1]), NEVER_KICKED), lk[:-1]])
    steps, players = np.nonzero((lk != previous) & (lk > KICK_TIME_VALID))
    return steps.astype(np.int64), players.astype(np.int64)


def kick_outcomes(
    trace: Dict[str, np.ndarray], kicks: Optional[tuple[np.ndarray, np.ndarray]] = None
) -> List[Dict[str, Any]]:
    """Outcome of every kick whose window is complete or already resolved (see the module docstring)."""
    holder = _holder(trace)
    score = np.asarray(trace["s_score"]).astype(np.int64)
    n = holder.size
    steps, players = kick_events(trace) if kicks is None else kicks
    out: List[Dict[str, Any]] = []
    for t0, p in zip(steps.tolist(), players.tolist()):
        last = min(t0 + KICK_WINDOW_STEPS, n - 1)
        base = score[t0 - 1] if t0 > 0 else np.zeros(2, dtype=np.int64)
        candidates = []
        changed = np.flatnonzero((score[t0:last + 1] != base).any(axis=1))
        if changed.size:
            candidates.append((t0 + int(changed[0]), 0, "goal"))
        window = holder[t0 + 1:last + 1]
        arrival = np.flatnonzero((window >= 0) & (window != p))
        if arrival.size:
            first = int(arrival[0])
            team = "teammate" if int(window[first]) // TEAM_SIZE == p // TEAM_SIZE else "opponent"
            candidates.append((t0 + 1 + first, 1, team))
        again = np.flatnonzero((window == p) & (np.arange(t0 + 1, last + 1) > t0 + SAME_PLAYER_MIN_STEPS))
        if again.size:
            candidates.append((t0 + 1 + int(again[0]), 1, "self"))
        if candidates:
            outcome = min(candidates)[2]
        elif t0 + KICK_WINDOW_STEPS <= n - 1:
            outcome = "none"
        else:
            continue
        out.append({"step": t0, "player": p, "team": p // TEAM_SIZE, "slot": p % TEAM_SIZE, "outcome": outcome})
    return out


def outfield_outcome_counts(kicks: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = {k: 0 for k in OUTCOMES}
    n = 0
    for k in kicks:
        if k["slot"] != 0:
            counts[k["outcome"]] += 1
            n += 1
    counts["n"] = n
    return counts


def pass_options(
    trace: Dict[str, np.ndarray],
    kicks: Optional[tuple[np.ndarray, np.ndarray]] = None,
    outcomes: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Passing options of every outfield kick that has a next row (see `pass options` in the module docstring).

    Each record: `step`, `player`, `candidates` and `open` (sorted player indices), `has_direction`, `aimed_open`,
    `outcome` (the `kick_outcomes` result, or None for a censored kick), `reachable` (an open candidate within `REACH_DEG`
    of the facing) and `arc_cover` (the share of the kickable arc grid within `AIM_CONE_DEG` of an open candidate); the
    last two are None for a kick without an open candidate or a direction, and for a trace without `ang`.
    """
    steps, players = kick_events(trace) if kicks is None else kicks
    n = int(np.asarray(trace["holder"]).shape[0])
    keep = (players % TEAM_SIZE != 0) & (steps + 1 < n)
    k, i = steps[keep], players[keep]
    if k.size == 0:
        return []
    by_kick = {
        (o["step"], o["player"]): o["outcome"]
        for o in (kick_outcomes(trace, kicks=(steps, players)) if outcomes is None else outcomes)
    }
    pos = np.asarray(trace["pos"], dtype=np.float64)
    ball = np.asarray(trace["ball"], dtype=np.float64)[k]                        # (K, 2)
    bvel = np.asarray(trace["bvel"], dtype=np.float64)[k + 1]                    # (K, 2)
    team = i // TEAM_SIZE
    mates = team[:, None] * TEAM_SIZE + np.arange(1, TEAM_SIZE)[None, :]          # (K, 5)
    opps = (1 - team)[:, None] * TEAM_SIZE + np.arange(TEAM_SIZE)[None, :]        # (K, 6)
    seg = pos[k[:, None], mates] - ball[:, None, :]                              # (K, 5, 2) ball -> teammate
    length = np.hypot(seg[..., 0], seg[..., 1])                                  # (K, 5)
    cand = (mates != i[:, None]) & (length >= PASS_MIN_DIST) & (length <= PASS_MAX_DIST)
    rel = pos[k[:, None], opps] - ball[:, None, :]                               # (K, 6, 2) ball -> opponent
    safe = np.where(length > 0.0, length, 1.0)[:, :, None]
    proj = (seg[:, :, None, 0] * rel[:, None, :, 0] + seg[:, :, None, 1] * rel[:, None, :, 1]) / safe
    perp = np.abs(seg[:, :, None, 0] * rel[:, None, :, 1] - seg[:, :, None, 1] * rel[:, None, :, 0]) / safe
    blocked = ((proj > 0.0) & (proj < length[:, :, None]) & (perp <= LANE_HALF_WIDTH)).any(axis=2)
    open_ = cand & ~blocked
    speed = np.hypot(bvel[:, 0], bvel[:, 1])
    has_dir = speed >= AIM_MIN_SPEED
    angle = np.degrees(np.abs(np.arctan2(
        bvel[:, None, 0] * seg[..., 1] - bvel[:, None, 1] * seg[..., 0],
        bvel[:, None, 0] * seg[..., 0] + bvel[:, None, 1] * seg[..., 1],
    )))
    aimed_open = (open_ & (angle <= AIM_CONE_DEG)).any(axis=1) & has_dir
    scored = open_.any(axis=1) & has_dir
    reachable = np.zeros(k.size, dtype=bool)
    arc_cover = np.zeros(k.size)
    has_facing = "ang" in trace
    if has_facing:
        facing = np.asarray(trace["ang"], dtype=np.float64)[k, i]                # (K,) body angle at the kick row
        theta = np.arctan2(seg[..., 1], seg[..., 0])                              # (K, 5) ball -> teammate
        rel = _wrap_pi(theta - facing[:, None])
        reachable = (open_ & (np.abs(rel) <= math.radians(REACH_DEG))).any(axis=1)
        grid = facing[:, None] + np.radians(ARC_GRID_DEG)[None, :]                # (K, 71)
        near = np.abs(_wrap_pi(grid[:, :, None] - theta[:, None, :])) <= math.radians(AIM_CONE_DEG)
        arc_cover = (near & open_[:, None, :]).any(axis=2).mean(axis=1)
    return [
        {
            "step": int(k[r]),
            "player": int(i[r]),
            "candidates": mates[r][cand[r]].tolist(),
            "open": mates[r][open_[r]].tolist(),
            "has_direction": bool(has_dir[r]),
            "aimed_open": bool(aimed_open[r]),
            "outcome": by_kick.get((int(k[r]), int(i[r]))),
            "reachable": bool(reachable[r]) if has_facing and scored[r] else None,
            "arc_cover": float(arc_cover[r]) if has_facing and scored[r] else None,
        }
        for r in range(k.size)
    ]


def _wrap_pi(a: np.ndarray) -> np.ndarray:
    """Angles wrapped to (-pi, pi]."""
    return math.pi - np.mod(math.pi - a, 2.0 * math.pi)


def pass_option_counts(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """Per-match numerators and denominators of the four pass-option shares."""
    c = {key: 0 for key in PASS_OPTION_COUNT_KEYS}
    for rec in records:
        option = bool(rec["open"])
        c["pass_opt_n"] += 1
        c["pass_opt_with"] += int(option)
        if option and rec["has_direction"]:
            c["aim_open_den"] += 1
            c["aim_open_n"] += int(rec["aimed_open"])
        if rec["outcome"] is None:
            continue
        if rec["aimed_open"]:
            c["aim_comp_den"] += 1
            c["aim_comp_mate"] += int(rec["outcome"] == "teammate")
        if not option:
            c["noopt_den"] += 1
            c["noopt_opp"] += int(rec["outcome"] == "opponent")
    return c


def aim_reach_counts(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-match numerators and denominators of the three aim chance-baseline shares."""
    c: Dict[str, Any] = {key: 0 for key in AIM_REACH_COUNT_KEYS}
    c["arc_reach_sum"] = 0.0
    for rec in records:
        if rec["reachable"] is None:
            continue
        c["open_reach_den"] += 1
        if rec["reachable"]:
            c["open_reach_n"] += 1
            c["aim_reach_n"] += int(rec["aimed_open"])
            c["arc_reach_sum"] += float(rec["arc_cover"])
    return c


def _floats(a: Any) -> np.ndarray:
    """`a` as an array of its own floating dtype (a trace stores float32), integers promoted to float64."""
    arr = np.asarray(a)
    return arr if arr.dtype.kind == "f" else arr.astype(np.float64)


def basin_counts(trace: Dict[str, np.ndarray]) -> Dict[str, int]:
    """Per-match numerators and denominators of the three basin shares, per team and pooled (see `basin keys`).

    The tests use the facing vector (cos `ang`, sin `ang`) and the dot product of a vector with it: cos(velocity, facing) < -0.5 is
    `dot < -0.5 |v|`, a bearing within 60 deg is `dot >= 0.5 |d|` and the tackle gate is `dot >= 0.25 |d|`. Decisions are processed
    `BASIN_BLOCK_ROWS` at a time, which keeps the temporaries of a block in cache.
    """
    c = {key: 0 for key in BASIN_COUNT_KEYS}
    if "ang" not in trace or "vel" not in trace:
        return c

    def add(key: str, group: str, num: int, den: int) -> None:
        c[f"{key}_n_{group}"] += num
        c[f"{key}_den_{group}"] += den
        c[f"{key}_n_pooled"] += num
        c[f"{key}_den_pooled"] += den

    pos, vel, ang, ball = (_floats(trace[k]) for k in ("pos", "vel", "ang", "ball"))
    clean = bool(np.isfinite(sum(float(x.sum()) for x in (pos, vel, ang, ball))))      # a NaN or an infinity anywhere makes the sum non-finite
    for lo in range(0, int(ang.shape[0]), BASIN_BLOCK_ROWS):
        rows = slice(lo, lo + BASIN_BLOCK_ROWS)
        p, v, a, b = pos[rows], vel[rows], ang[rows], ball[rows]
        fx, fy = np.cos(a), np.sin(a)
        speed = np.sqrt(v[..., 0] * v[..., 0] + v[..., 1] * v[..., 1])
        reverse = (speed > REVERSE_MIN_SPEED) & (v[..., 0] * fx + v[..., 1] * fy < REVERSE_COS_MAX * speed)
        to_ball_x, to_ball_y = b[:, 0:1] - p[..., 0], b[:, 1:2] - p[..., 1]
        facing = to_ball_x * fx + to_ball_y * fy >= FACING_BALL_COS * np.sqrt(to_ball_x * to_ball_x + to_ball_y * to_ball_y)
        if clean:
            moving = seen = None
        else:
            moving = np.isfinite(v).all(axis=-1) & np.isfinite(a)
            seen = np.isfinite(p).all(axis=-1) & np.isfinite(a) & np.isfinite(b).all(axis=-1)[:, None]
        for team, label in enumerate(TEAMS):
            seats = slice(TEAM_SIZE * team + 1, TEAM_SIZE * team + TEAM_SIZE)
            if clean:
                add("reverse_motion", label, int(np.count_nonzero(reverse[:, seats])), int(a.shape[0]) * (TEAM_SIZE - 1))
                add("facing_ball", label, int(np.count_nonzero(facing[:, seats])), int(a.shape[0]) * (TEAM_SIZE - 1))
            else:
                add("reverse_motion", label, int(np.count_nonzero(reverse[:, seats] & moving[:, seats])),
                    int(np.count_nonzero(moving[:, seats])))
                add("facing_ball", label, int(np.count_nonzero(facing[:, seats] & seen[:, seats])),
                    int(np.count_nonzero(seen[:, seats])))

    holder = _holder(trace)
    held = np.flatnonzero((holder >= 0) & (holder % TEAM_SIZE != 0))
    if held.size:
        carrier = holder[held]
        where = pos[held]
        at = np.arange(held.size)
        to_x = where[at, carrier, 0][:, None] - where[..., 0]
        to_y = where[at, carrier, 1][:, None] - where[..., 1]
        dist = np.sqrt(to_x * to_x + to_y * to_y)
        row, seat = np.nonzero((dist <= TACKLE_RADIUS) & _OPPOSING_OUTFIELD[carrier // TEAM_SIZE])       # opposing outfield players near the carrier
        facing = ang[held[row], seat]
        d = dist[row, seat]
        toward = np.cos(facing) * to_x[row, seat] + np.sin(facing) * to_y[row, seat]
        valid = np.isfinite(toward) if not clean else np.ones(row.size, dtype=bool)
        ready = valid & ((d < TACKLE_ON_CARRIER) | (toward >= TACKLE_FACING_DOT * d))
        for team, label in enumerate(TEAMS):
            mine = seat // TEAM_SIZE == team
            add("tackle_ready", label, int(np.count_nonzero(ready & mine)), int(np.count_nonzero(valid & mine)))
    return c


# ----------------------------------------------------------------------------
# Defence keys
# ----------------------------------------------------------------------------
def _canonical_x(x: np.ndarray, side_col: np.ndarray) -> np.ndarray:
    """`x` (K, ...) as the distance from the own goal line of the team whose `side` entry on each of the K rows is `side_col`."""
    own_left = (side_col == 0).reshape((-1,) + (1,) * (x.ndim - 1))
    return np.where(own_left, x, FIELD_WIDTH - x)


def _rise_rows(counter: np.ndarray) -> np.ndarray:
    """Rows on which the cumulative `counter` (T,) rises."""
    return np.flatnonzero(np.diff(counter, prepend=0) > 0)


def _outfield_seats(team: int) -> np.ndarray:
    return TEAM_SIZE * team + np.arange(1, TEAM_SIZE)


def _add_share(c: Dict[str, Any], key: str, team: int, num: float, den: int) -> None:
    num_name, den_name = DEFENCE_SHARES[key][:2]
    for group in (TEAMS[team], "pooled"):
        c[f"{num_name}_{group}"] += num
        c[f"{den_name}_{group}"] += den


def _goal_side(pos: np.ndarray, ball: np.ndarray, side: np.ndarray, rows: np.ndarray, team: int) -> np.ndarray:
    """(K, 5): which of `team`'s outfielders stand strictly closer to their own goal line than the ball on each of `rows`."""
    side_rows = side[rows, team]
    return _canonical_x(pos[rows][:, _outfield_seats(team), 0], side_rows) < _canonical_x(ball[rows, 0], side_rows)[:, None]


def _shot_counts(c: Dict[str, Any], trace: Dict[str, np.ndarray], pos: np.ndarray, ball: np.ndarray, side: np.ndarray, holder: np.ndarray) -> None:
    shots = np.asarray(trace["s_shots"]).astype(np.int64)
    score = np.asarray(trace["s_score"]).astype(np.int64)
    last = holder.size - 1
    for shooting in (0, 1):
        defending = 1 - shooting
        ks = _rise_rows(shots[:, shooting])
        if ks.size == 0:
            continue
        shooter = holder[ks]
        wrong = (shooter < 0) | (shooter // TEAM_SIZE != shooting)
        if wrong.any():
            k = int(ks[wrong][0])
            raise ValueError(f"shot of team {TEAMS[shooting]} on row {k}: the holder on that row is {int(holder[k])}, not a player of that team")
        goal_side = _goal_side(pos, ball, side, ks, defending).sum(axis=1)
        before = np.where(ks > 0, score[np.maximum(ks - 1, 0), shooting], 0)
        scored = score[np.minimum(ks + GOAL_AFTER_SHOT_STEPS, last), shooting] > before
        _add_share(c, "goal_side_at_goal", defending, float(goal_side[scored].sum()), int(scored.sum()))

        seats = _outfield_seats(defending)
        gap = pos[ks, shooter][:, None, :] - pos[ks][:, seats]                       # defender -> shooter
        dist = np.hypot(gap[..., 0], gap[..., 1])
        at = np.arange(ks.size)
        nearest = np.argmin(dist, axis=1)
        seat = seats[nearest]
        if "ang" in trace:
            angle = np.asarray(trace["ang"], dtype=np.float64)[ks, seat]
            toward = np.cos(angle) * gap[at, nearest, 0] + np.sin(angle) * gap[at, nearest, 1]
            seen = np.isfinite(toward)
            facing = (dist[at, nearest] < TACKLE_ON_CARRIER) | (toward >= TACKLE_FACING_DOT * dist[at, nearest])
            _add_share(c, "shot_nearest_def_facing", defending, int((facing & seen).sum()), int(seen.sum()))
        if "vel" in trace:
            vel = np.asarray(trace["vel"], dtype=np.float64)
            window = ks[:, None] + np.arange(-SHOT_APPROACH_STEPS, 0)[None, :]       # the decisions before the shot
            inside = window >= 0
            rows = np.maximum(window, 0)
            gap_w = pos[rows, shooter[:, None]] - pos[rows, seat[:, None]]
            speed = (vel[rows, seat[:, None], 0] * gap_w[..., 0] + vel[rows, seat[:, None], 1] * gap_w[..., 1]) / np.maximum(
                np.hypot(gap_w[..., 0], gap_w[..., 1]), 1e-6)
            seen = inside.any(axis=1) & np.where(inside, np.isfinite(speed), True).all(axis=1)
            mean = np.where(inside, speed, 0.0).sum(axis=1) / np.maximum(inside.sum(axis=1), 1)
            _add_share(c, "shot_nearest_def_closing", defending, int(((mean > CLOSING_MIN_SPEED) & seen).sum()), int(seen.sum()))


def _regain_counts(c: Dict[str, Any], trace: Dict[str, np.ndarray], pos: np.ndarray, ball: np.ndarray, side: np.ndarray, holder: np.ndarray) -> None:
    shots = np.asarray(trace["s_shots"]).astype(np.int64)
    team = np.where(holder >= 0, holder // TEAM_SIZE, -1)
    held = np.flatnonzero(team >= 0)
    for defending in (0, 1):
        opponent = 1 - defending
        regain = held[1:][(team[held[:-1]] == defending) & (team[held[1:]] == opponent)]
        rises = _rise_rows(shots[:, opponent])
        if regain.size == 0 or rises.size == 0:
            continue
        own = held[team[held] == defending]
        after = np.searchsorted(own, regain, side="right")
        end = np.where(after < own.size, own[np.minimum(after, own.size - 1)], holder.size - 1)
        first = np.searchsorted(rises, regain + 1)
        shot = rises[np.minimum(first, rises.size - 1)]
        reaches = (first < rises.size) & (shot <= end)
        regain, shot = regain[reaches], shot[reaches]
        if regain.size == 0:
            continue
        goal_side_at_regain = _goal_side(pos, ball, side, regain, defending)
        goal_side_at_shot = _goal_side(pos, ball, side, shot, defending)
        had = goal_side_at_regain.sum(axis=1)
        kept = (goal_side_at_regain & goal_side_at_shot).sum(axis=1)[had > 0] / had[had > 0]
        _add_share(c, "regain_goalside_at_shot", defending, float(goal_side_at_shot.sum()), int(regain.size))
        _add_share(c, "regain_goalside_retention", defending, float(kept.sum()), int(kept.size))


def _tackle_counts(c: Dict[str, Any], trace: Dict[str, np.ndarray], pos: np.ndarray, holder: np.ndarray) -> None:
    ang = np.asarray(trace["ang"], dtype=np.float64)
    fires = np.asarray(trace["act"])[..., 3] > 0.5
    last = holder.size - 1
    for defending in (0, 1):
        seats = _outfield_seats(defending)
        rows = np.flatnonzero((holder >= 0) & (holder // TEAM_SIZE == 1 - defending) & (holder % TEAM_SIZE != 0))
        if rows.size == 0:
            continue
        carrier = holder[rows]
        gap = pos[rows, carrier][:, None, :] - pos[rows][:, seats]                    # defender -> carrier, (R, 5, 2)
        dist = np.hypot(gap[..., 0], gap[..., 1])
        toward = np.cos(ang[rows][:, seats]) * gap[..., 0] + np.sin(ang[rows][:, seats]) * gap[..., 1]
        seen = np.isfinite(dist) & np.isfinite(toward)
        facing = (dist < TACKLE_ON_CARRIER) | (toward >= TACKLE_FACING_DOT * dist)
        chance = facing & (np.maximum(0.0, dist - TACKLE_CONTACT_OFFSET) <= TACKLE_REACH[None, :])
        fired = fires[rows][:, seats]
        out_of_reach = int((fired & seen & ~chance & (dist <= TACKLE_WHIFF_RANGE)).sum())
        for group in (TEAMS[defending], "pooled"):
            c[f"tackle_out_of_range_per_match_{group}"] += out_of_reach
        # a spell starts at a chance row that does not continue the previous chance row of the same defender against the same carrier
        continues = (rows[1:] == rows[:-1] + 1) & (carrier[1:] == carrier[:-1])
        start = chance.copy()
        start[1:] &= ~(chance[:-1] & continues[:, None])
        spell = np.cumsum(start, axis=0)
        # the defender holds the ball on the fire row or within the next TACKLE_WIN_STEPS decisions
        holds = np.cumsum(holder[:, None] == seats[None, :], axis=0)
        upto = np.minimum(rows + TACKLE_WIN_STEPS, last)
        from_row = np.where(rows[:, None] > 0, holds[np.maximum(rows - 1, 0)], 0)
        won = (holds[upto] - from_row) > 0
        n_fired = n_won = 0
        for j in range(seats.size):
            n_fired += np.unique(spell[chance[:, j] & fired[:, j], j]).size
            n_won += np.unique(spell[chance[:, j] & fired[:, j] & won[:, j], j]).size
        _add_share(c, "tackle_in_range_rate", defending, n_fired, int(start.sum()))
        _add_share(c, "tackle_win_rate", defending, n_won, n_fired)


def _deep_block_counts(c: Dict[str, Any], pos: np.ndarray, ball: np.ndarray, side: np.ndarray, holder: np.ndarray) -> None:
    rows = np.arange(holder.size)
    for defending in (0, 1):
        x = _canonical_x(pos[:, _outfield_seats(defending), 0], side[:, defending])
        in_opponent_half = _canonical_x(ball[:, 0], side[:, defending]) >= DEEP_BLOCK_MIN_BALL_X
        deep = (x <= DEEP_BLOCK_MAX_X).sum(axis=1) >= DEEP_BLOCK_MIN_PLAYERS
        _add_share(c, "deep_block_ge4", defending, int((in_opponent_half & deep).sum()), int(in_opponent_half.sum()))
        out_of_possession = rows[(holder >= 0) & (holder // TEAM_SIZE == 1 - defending)]
        _add_share(c, "outfield_depth_oop", defending, float(x[out_of_possession].mean(axis=1).sum()), int(out_of_possession.size))


def defence_counts(trace: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Per-match numerators and denominators of the defence keys, per defending team and pooled (see `defence keys`).

    A trace without `s_shots` enters no shot or regain key, one without `act` or `ang` no tackle key, one without `ang` no facing key
    and one without `vel` no closing key.
    """
    c: Dict[str, Any] = {key: 0 for key in DEFENCE_COUNT_KEYS}
    holder = _holder(trace)
    if holder.size == 0:
        return c
    pos = np.asarray(trace["pos"], dtype=np.float64)
    ball = np.asarray(trace["ball"], dtype=np.float64)
    side = np.asarray(trace["side"]).astype(np.int64)
    _deep_block_counts(c, pos, ball, side, holder)
    if "s_shots" in trace:
        _shot_counts(c, trace, pos, ball, side, holder)
        _regain_counts(c, trace, pos, ball, side, holder)
    if "act" in trace and "ang" in trace:
        _tackle_counts(c, trace, pos, holder)
    return c


def defence_metrics(
    per_match: Sequence[Dict[str, Any]], *, min_events: int = DEFENCE_MIN_EVENTS, min_decisions: int = DEFENCE_MIN_DECISIONS
) -> Dict[str, Dict[str, Any]]:
    """`{mean, se, n}` of every defence key from per-match records; a key whose event count over the matches is below the minimum is None."""
    col = lambda key: [m[key] for m in per_match]  # noqa: E731
    minimum = {"events": min_events, "decisions": min_decisions}
    out: Dict[str, Dict[str, Any]] = {}
    for key, (num, den, floor, kind) in DEFENCE_SHARES.items():
        for group in BASIN_GROUPS:
            stat = ratio_se(col(f"{num}_{group}"), col(f"{den}_{group}"))
            if sum(col(f"{floor}_{group}")) < minimum[kind]:
                stat = (None, None, stat[2])
            out[f"{key}_{group}"] = _stat_dict(stat)
    for group in BASIN_GROUPS:
        key = f"tackle_out_of_range_per_match_{group}"
        stat = mean_se(col(key))
        if sum(col(f"tackle_in_range_rate_den_{group}")) < min_events:
            stat = (None, None, stat[2])
        out[key] = _stat_dict(stat)
    return out


def trap_flag_g50(census_json: Dict[str, Any]) -> bool:
    """The early-trap flag of a generation-50 `self_sampled` census: pooled facing-the-ball share <= 0.35 and goals per match < 0.5.

    Accepts the census payload (`summary`), the `summarize` result or its `metrics` mapping; raises ValueError when either input is
    missing, None or not finite.
    """
    metrics = census_json
    if "summary" in metrics:
        metrics = metrics["summary"]
    if "metrics" in metrics:
        metrics = metrics["metrics"]
    values = []
    for key in ("facing_ball_share_pooled", "goals_total"):
        entry = metrics.get(key)
        mean = entry.get("mean") if isinstance(entry, dict) else None
        if mean is None or not math.isfinite(mean):
            raise ValueError(f"trap flag needs a finite mean of {key!r}, got {mean!r}")
        values.append(float(mean))
    return bool(values[0] <= TRAP_FACING_MAX and values[1] < TRAP_GOALS_BELOW)


def kick_propensity_counts(
    trace: Dict[str, np.ndarray],
    holder: Optional[np.ndarray] = None,
    kicks: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> Dict[str, tuple[int, int]]:
    """Return `(kicks_by_holder, holding_decisions)` per team, pooled, and per role group."""
    h = _holder(trace) if holder is None else holder
    held = h >= 0
    hold_by_player = np.bincount(h[held], minlength=2 * TEAM_SIZE) if held.any() else np.zeros(2 * TEAM_SIZE, dtype=np.int64)
    steps, players = kick_events(trace) if kicks is None else kicks
    if steps.size:
        valid = h[steps] == players
        kick_by_player = np.bincount(players[valid], minlength=2 * TEAM_SIZE) if valid.any() else np.zeros(2 * TEAM_SIZE, dtype=np.int64)
    else:
        kick_by_player = np.zeros(2 * TEAM_SIZE, dtype=np.int64)

    out: Dict[str, tuple[int, int]] = {}
    for t_idx, t_name in enumerate(TEAMS):
        sl = slice(TEAM_SIZE * t_idx, TEAM_SIZE * (t_idx + 1))
        out[t_name] = (int(kick_by_player[sl].sum()), int(hold_by_player[sl].sum()))
    out["pooled"] = (int(kick_by_player.sum()), int(hold_by_player.sum()))
    for group, slots in ROLE_GROUPS.items():
        idxs_a = [s for s in slots]
        idxs_b = [TEAM_SIZE + s for s in slots]
        ka, ha = int(kick_by_player[idxs_a].sum()), int(hold_by_player[idxs_a].sum())
        kb, hb = int(kick_by_player[idxs_b].sum()), int(hold_by_player[idxs_b].sum())
        out[f"A_{group}"] = (ka, ha)
        out[f"B_{group}"] = (kb, hb)
        out[group] = (ka + kb, ha + hb)
    return out


def kick_propensity(trace: Dict[str, np.ndarray]) -> Dict[str, Optional[float]]:
    """Kicks by the holder divided by decisions on which that player holds the ball; None if 0 holds."""
    return {
        k: (float(kn) / hn if hn > 0 else None)
        for k, (kn, hn) in kick_propensity_counts(trace).items()
    }


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def _outfield(trace: Dict[str, np.ndarray], team: int) -> np.ndarray:
    """(T, 5, 2) outfield positions of `team` as float64."""
    return np.asarray(trace["pos"], dtype=np.float64)[:, TEAM_SIZE * team + 1:TEAM_SIZE * team + TEAM_SIZE]


def n_within(trace: Dict[str, np.ndarray], radius: float) -> np.ndarray:
    """Mean number of outfield players strictly within `radius` of the ball per decision, shape (2,)."""
    ball = np.asarray(trace["ball"], dtype=np.float64)
    if ball.shape[0] == 0:
        return np.full(2, np.nan)
    out = np.empty(2)
    for team in (0, 1):
        p = _outfield(trace, team)
        d = np.hypot(p[:, :, 0] - ball[:, None, 0], p[:, :, 1] - ball[:, None, 1])
        out[team] = (d < radius).sum(axis=1).mean()
    return out


def max_component_size(trace: Dict[str, np.ndarray], link: float = LINK_RADIUS) -> np.ndarray:
    """Per decision, size of the largest connected group of outfield players, shape (2, T)."""
    sizes = []
    for team in (0, 1):
        p = _outfield(trace, team)
        d = np.hypot(p[:, :, None, 0] - p[:, None, :, 0], p[:, :, None, 1] - p[:, None, :, 1])
        reach = (d < link).astype(np.float32)
        for _ in range(2):                           # path lengths up to 4 cover a 5-player group
            reach = (np.matmul(reach, reach) > 0).astype(np.float32)
        sizes.append(reach.sum(axis=2).max(axis=1).astype(np.int64))
    return np.stack(sizes)


def cluster_ge4(trace: Dict[str, np.ndarray], link: float = LINK_RADIUS) -> np.ndarray:
    """Fraction of decisions with a connected group of at least 4 outfield players, shape (2,)."""
    sizes = max_component_size(trace, link)
    if sizes.shape[1] == 0:
        return np.full(2, np.nan)
    return (sizes >= CLUSTER_MIN).mean(axis=1)


def _role_inversion_arrays(pos: np.ndarray, side: np.ndarray, margin: float = INVERSION_MARGIN) -> np.ndarray:
    if pos.shape[0] == 0:
        return np.full(2, np.nan)
    out = np.empty(2)
    for team in (0, 1):
        sign = np.where(side[:, team] == 0, 1.0, -1.0)
        base = TEAM_SIZE * team
        striker = pos[:, base + 5, 0]
        lead = np.maximum(sign * (pos[:, base + 1, 0] - striker), sign * (pos[:, base + 2, 0] - striker))
        out[team] = (lead > margin).mean()
    return out


def role_inversion(trace: Dict[str, np.ndarray], margin: float = INVERSION_MARGIN) -> np.ndarray:
    """Fraction of decisions with D1 or D2 more than `margin` ahead of the striker, shape (2,)."""
    pos = np.asarray(trace["pos"], dtype=np.float64)
    side = np.asarray(trace["side"])
    return _role_inversion_arrays(pos, side, margin=margin)


def gk_geometry(trace: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Per-keeper geometry: mean distance to own goal centre, fixed 195 u in-box share (inclusive),
    and goal-line share (canonical x_c <= 19.5 u, inclusive).
    """
    pos = np.asarray(trace["pos"], dtype=np.float64)
    side = np.asarray(trace["side"]).astype(np.int64)
    n = pos.shape[0]
    if n == 0:
        return {
            f"gk_{m}_{t}": float("nan")
            for m in ("dist_goal", "in_box", "goal_line")
            for t in ("A", "B", "pooled")
        }
    out: Dict[str, float] = {}
    for team, label in enumerate(TEAMS):
        keeper = TEAM_SIZE * team
        goal_x = np.where(side[:, team] == 0, 0.0, FIELD_WIDTH)
        sign = np.where(side[:, team] == 0, 1.0, -1.0)
        x_c = (pos[:, keeper, 0] - goal_x) * sign
        dist = np.hypot(pos[:, keeper, 0] - goal_x, pos[:, keeper, 1] - GOAL_Y)
        out[f"gk_dist_goal_{label}"] = float(dist.mean())
        out[f"gk_in_box_{label}"] = float((dist <= GK_BOX_RADIUS).mean())
        out[f"gk_goal_line_{label}"] = float((x_c <= GK_GOAL_LINE_X).mean())
    for m in ("dist_goal", "in_box", "goal_line"):
        out[f"gk_{m}_pooled"] = 0.5 * (out[f"gk_{m}_A"] + out[f"gk_{m}_B"])
    return out


# ----------------------------------------------------------------------------
# Goalkeeper opportunities
# ----------------------------------------------------------------------------
def gk_opportunities(
    trace: Dict[str, np.ndarray], reach: float = GK_REACH, max_ball_speed: float = GK_MAX_BALL_SPEED
) -> List[Dict[str, Any]]:
    """Episodes of a slow loose ball inside a keeper's box with the keeper within `reach` of it."""
    holder = _holder(trace)
    n = holder.size
    pos = np.asarray(trace["pos"], dtype=np.float64)
    ball = np.asarray(trace["ball"], dtype=np.float64)
    speed = np.hypot(*np.asarray(trace["bvel"], dtype=np.float64).T) if n else np.zeros(0)
    side = np.asarray(trace["side"]).astype(np.int64)
    box = np.asarray(trace["box"], dtype=np.float64)
    episodes: List[Dict[str, Any]] = []
    for team in (0, 1):
        keeper = TEAM_SIZE * team
        goal_x = np.where(side[:, team] == 0, 0.0, FIELD_WIDTH)
        radius = box[np.arange(n), side[:, team]]
        ball_goal = np.hypot(ball[:, 0] - goal_x, ball[:, 1] - GOAL_Y)
        keeper_ball = np.hypot(pos[:, keeper, 0] - ball[:, 0], pos[:, keeper, 1] - ball[:, 1])
        active = (holder < 0) & (ball_goal <= radius) & (keeper_ball <= reach) & (speed < max_ball_speed)
        padded = np.concatenate([[False], active, [False]]).astype(np.int8)
        edges = np.flatnonzero(np.diff(padded))
        for start, end in zip(edges[0::2].tolist(), edges[1::2].tolist()):
            closing = min(end, n - 1)
            episodes.append({
                "team": team,
                "start": start,
                "end": end,
                "pickup": bool((holder[start:min(n, closing + GK_PICKUP_GRACE_STEPS + 1)] == keeper).any()),
                "within30": bool(keeper_ball[start:closing + 1].min() < GK_CLOSE),
            })
    episodes.sort(key=lambda e: (e["start"], e["team"]))
    return episodes


# ----------------------------------------------------------------------------
# Goals
# ----------------------------------------------------------------------------
def goal_events(
    trace: Dict[str, np.ndarray],
    holder: Optional[np.ndarray] = None,
    kicks: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> List[Dict[str, Any]]:
    """One record per goal, in match order (see `goal_types` in the module docstring)."""
    score = np.asarray(trace["s_score"]).astype(np.int64)
    n = score.shape[0]
    if n == 0:
        return []
    holder_arr = _holder(trace) if holder is None else holder
    passes = np.asarray(trace["s_team_passes"]).astype(np.int64)
    own_goals = np.asarray(trace["s_own_goals"]).astype(np.int64)
    pos = np.asarray(trace["pos"], dtype=np.float64)
    side = np.asarray(trace["side"])
    kick_steps, kick_players = kick_events(trace) if kicks is None else kicks

    def cumulative(arr, index, team):
        return int(arr[index, team]) if index >= 0 else 0

    delta = np.diff(np.vstack([np.zeros((1, 2), dtype=np.int64), score]), axis=0)
    events = []
    for step, team in sorted(zip(*(a.tolist() for a in np.nonzero(delta > 0)))):
        previous_own = int(own_goals[step - 1]) if step > 0 else 0
        own = int(own_goals[step]) > previous_own
        record: Dict[str, Any] = {
            "step": step,
            "team": team,
            "own_goal": own,
            "direct": False,
            "seconds_since_kick": None,
            "kick_distance": None,
            "kicker_slot": None,
            "passes_prior_10s": cumulative(passes, step - 1, team) - cumulative(passes, step - 1 - PASS_WINDOW_STEPS, team),
        }
        mine = (kick_players // TEAM_SIZE == team) & (kick_steps <= step)
        if not own and mine.any():
            t_kick = int(kick_steps[mine][-1])
            player = int(kick_players[mine][-1])
            goal_x = FIELD_WIDTH if side[t_kick, team] == 0 else 0.0
            record["seconds_since_kick"] = (step - t_kick) * STEP_SECONDS
            record["kick_distance"] = float(math.hypot(goal_x - pos[t_kick, player, 0], GOAL_Y - pos[t_kick, player, 1]))
            record["kicker_slot"] = player % TEAM_SIZE
            record["direct"] = bool(step - t_kick <= DIRECT_KICK_STEPS and not (holder_arr[t_kick + 1:step + 1] >= 0).any())
        events.append(record)
    return events


def goal_type_counts(goals: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Counts behind the goal-type shares: open-play goals exclude own goals."""
    out = {"n": 0, "own_goals": 0, "n_open_play": 0, "n_with_kick": 0, "direct": 0, "no_pass_10s": 0, "kick_distance_sum": 0.0}
    for g in goals:
        out["n"] += 1
        if g["own_goal"]:
            out["own_goals"] += 1
            continue
        out["n_open_play"] += 1
        out["direct"] += int(g["direct"])
        out["no_pass_10s"] += int(g["passes_prior_10s"] == 0)
        if g["kick_distance"] is not None:
            out["n_with_kick"] += 1
            out["kick_distance_sum"] += g["kick_distance"]
    return out


PASS_OPTION_SHARES = {
    "pass_option_share": ("pass_opt_with", "pass_opt_n"),
    "aim_open_share": ("aim_open_n", "aim_open_den"),
    "aim_open_completion": ("aim_comp_mate", "aim_comp_den"),
    "no_option_opponent_share": ("noopt_opp", "noopt_den"),
    "open_reachable_share": ("open_reach_n", "open_reach_den"),
    "aim_open_reachable_share": ("aim_reach_n", "open_reach_n"),
    "aim_open_arc_baseline": ("arc_reach_sum", "open_reach_n"),
}

BASIN_SHARES = {
    f"{key}_share_{group}": (f"{key}_n_{group}", f"{key}_den_{group}") for key in BASIN_KEYS for group in BASIN_GROUPS
}

KICK_PROP_KEYS = (
    "A", "B", "pooled", "GK", "D", "M", "ST",
    "A_GK", "A_D", "A_M", "A_ST", "B_GK", "B_D", "B_M", "B_ST",
)


def _safe_ratio(num: float, den: float) -> Optional[float]:
    return float(num) / float(den) if den > 0 else None


def time_binned_counts(
    trace: Dict[str, np.ndarray],
    late_steps: int = LATE_WINDOW_STEPS,
    *,
    holder: Optional[np.ndarray] = None,
    spell_intervals: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    kicks: Optional[tuple[np.ndarray, np.ndarray]] = None,
    outcomes: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Raw counts and per-bin scalars comparing the last `late_steps` decisions with the rest."""
    h = _holder(trace) if holder is None else holder
    n = int(h.size)
    split = max(0, n - int(late_steps))
    pos = np.asarray(trace["pos"], dtype=np.float64)
    side = np.asarray(trace["side"]).astype(np.int64)
    sp_players, sp_starts, sp_ends = holder_spell_intervals(trace) if spell_intervals is None else spell_intervals
    k_steps, k_players = kick_events(trace) if kicks is None else kicks
    k_out = kick_outcomes(trace, kicks=(k_steps, k_players)) if outcomes is None else outcomes

    has_shots = "s_shots" in trace and n > 0
    s_shots = np.asarray(trace["s_shots"], dtype=np.int64) if has_shots else None
    has_viol = "s_field_5s_violations" in trace and "s_keeper_5s_violations" in trace and n > 0
    s_f5 = np.asarray(trace["s_field_5s_violations"], dtype=np.int64) if has_viol else None
    s_k5 = np.asarray(trace["s_keeper_5s_violations"], dtype=np.int64) if has_viol else None

    # Precompute keeper quick kicks (within 0.5 s of spell start) using sorted spell starts.
    gk_kick_steps: Dict[int, np.ndarray] = {}
    gk_kick_quick: Dict[int, np.ndarray] = {}
    for team in (0, 1):
        keeper = TEAM_SIZE * team
        m_sp = sp_players == keeper
        starts_k = sp_starts[m_sp]
        m_k = (k_players == keeper) & (h[k_steps] == keeper) if k_steps.size else np.zeros(0, dtype=bool)
        steps_k = k_steps[m_k]
        if steps_k.size and starts_k.size:
            idx = np.clip(np.searchsorted(starts_k, steps_k, side="right") - 1, 0, starts_k.size - 1)
            dt_s = (steps_k - starts_k[idx] + 1) * STEP_SECONDS
            gk_kick_steps[team] = steps_k
            gk_kick_quick[team] = dt_s <= GK_QUICK_KICK_S + 1e-9
        else:
            gk_kick_steps[team] = np.zeros(0, dtype=np.int64)
            gk_kick_quick[team] = np.zeros(0, dtype=bool)

    sp_dur = (sp_ends - sp_starts) * STEP_SECONDS
    outfield_steps = np.array([k["step"] for k in k_out if k["slot"] != 0], dtype=np.int64)
    outfield_teams = np.array([k["team"] for k in k_out if k["slot"] != 0], dtype=np.int64)
    outfield_self = np.array([k["outcome"] == "self" for k in k_out if k["slot"] != 0], dtype=bool)

    rec: Dict[str, Any] = {}
    for b_name, a, b in (("early", 0, split), ("late", split, n)):
        m = b - a
        minutes = m * STEP_SECONDS / 60.0
        rec[f"{b_name}_minutes"] = float(minutes)
        rec[f"{b_name}_shots_minutes"] = float(minutes) if has_shots and m > 0 else 0.0
        rec[f"{b_name}_viol_minutes"] = float(minutes) if has_viol and m > 0 else 0.0

        # Keeper spells & quick kicks
        sp_in_bin = (sp_starts >= a) & (sp_starts < b)
        for team, lbl in enumerate(TEAMS):
            keeper = TEAM_SIZE * team
            mask_sp = sp_in_bin & (sp_players == keeper)
            rec[f"{b_name}_gk_spells_n_{lbl}"] = int(mask_sp.sum())
            rec[f"{b_name}_gk_spells_sum_{lbl}"] = float(sp_dur[mask_sp].sum())

            ks = gk_kick_steps[team]
            kq = gk_kick_quick[team]
            mask_k = (ks >= a) & (ks < b)
            rec[f"{b_name}_gk_kicks_{lbl}"] = int(mask_k.sum())
            rec[f"{b_name}_gk_quick_{lbl}"] = int(kq[mask_k].sum()) if mask_k.any() else 0

        rec[f"{b_name}_gk_spells_n_pooled"] = rec[f"{b_name}_gk_spells_n_A"] + rec[f"{b_name}_gk_spells_n_B"]
        rec[f"{b_name}_gk_spells_sum_pooled"] = rec[f"{b_name}_gk_spells_sum_A"] + rec[f"{b_name}_gk_spells_sum_B"]
        rec[f"{b_name}_gk_kicks_pooled"] = rec[f"{b_name}_gk_kicks_A"] + rec[f"{b_name}_gk_kicks_B"]
        rec[f"{b_name}_gk_quick_pooled"] = rec[f"{b_name}_gk_quick_A"] + rec[f"{b_name}_gk_quick_B"]

        # Defender mean canonical x & role inversion
        if m > 0:
            inv = _role_inversion_arrays(pos[a:b], side[a:b])
            for team, lbl in enumerate(TEAMS):
                base = TEAM_SIZE * team
                goal_x = np.where(side[a:b, team] == 0, 0.0, FIELD_WIDTH)[:, None]
                sign = np.where(side[a:b, team] == 0, 1.0, -1.0)[:, None]
                def_xc = (pos[a:b, base + 1:base + 3, 0] - goal_x) * sign
                rec[f"{b_name}_def_x_{lbl}"] = float(def_xc.mean())
                rec[f"{b_name}_role_inversion_{lbl}"] = float(inv[team])
            rec[f"{b_name}_def_x_pooled"] = 0.5 * (rec[f"{b_name}_def_x_A"] + rec[f"{b_name}_def_x_B"])
            rec[f"{b_name}_role_inversion_pooled"] = 0.5 * (
                rec[f"{b_name}_role_inversion_A"] + rec[f"{b_name}_role_inversion_B"]
            )
        else:
            for lbl in ("A", "B", "pooled"):
                rec[f"{b_name}_def_x_{lbl}"] = None
                rec[f"{b_name}_role_inversion_{lbl}"] = None

        # Shots
        if has_shots and m > 0:
            base_s = s_shots[a - 1] if a > 0 else np.zeros(2, dtype=np.int64)
            d_shots = s_shots[b - 1] - base_s
            rec[f"{b_name}_shots_A"] = int(d_shots[0])
            rec[f"{b_name}_shots_B"] = int(d_shots[1])
            rec[f"{b_name}_shots_both"] = int(d_shots[0] + d_shots[1])
        else:
            rec[f"{b_name}_shots_A"] = 0
            rec[f"{b_name}_shots_B"] = 0
            rec[f"{b_name}_shots_both"] = 0

        # Outfield kick-to-self
        in_bin_k = (outfield_steps >= a) & (outfield_steps < b) if outfield_steps.size else np.zeros(0, dtype=bool)
        for team, lbl in enumerate(TEAMS):
            mk = in_bin_k & (outfield_teams == team)
            rec[f"{b_name}_kick_outfield_n_{lbl}"] = int(mk.sum())
            rec[f"{b_name}_kick_self_{lbl}"] = int((mk & outfield_self).sum())
        rec[f"{b_name}_kick_outfield_n_pooled"] = rec[f"{b_name}_kick_outfield_n_A"] + rec[f"{b_name}_kick_outfield_n_B"]
        rec[f"{b_name}_kick_self_pooled"] = rec[f"{b_name}_kick_self_A"] + rec[f"{b_name}_kick_self_B"]

        # 5s violations (pooled _both)
        if has_viol and m > 0:
            df = int(s_f5[b - 1]) - (int(s_f5[a - 1]) if a > 0 else 0)
            dk = int(s_k5[b - 1]) - (int(s_k5[a - 1]) if a > 0 else 0)
            rec[f"{b_name}_field_5s_both"] = df
            rec[f"{b_name}_keeper_5s_both"] = dk
            rec[f"{b_name}_violations_5s_both"] = df + dk
        else:
            rec[f"{b_name}_field_5s_both"] = 0
            rec[f"{b_name}_keeper_5s_both"] = 0
            rec[f"{b_name}_violations_5s_both"] = 0
    return rec


def time_binned_metrics(trace: Dict[str, np.ndarray], late_steps: int = LATE_WINDOW_STEPS) -> Dict[str, Optional[float]]:
    """Compute early vs late (last `late_steps` decisions) rate/mean metrics for one trace."""
    c = time_binned_counts(trace, late_steps=late_steps)
    out: Dict[str, Optional[float]] = {}
    for b_name in ("early", "late"):
        for lbl in ("A", "B", "pooled"):
            out[f"{b_name}_gk_hold_s_{lbl}"] = _safe_ratio(c[f"{b_name}_gk_spells_sum_{lbl}"], c[f"{b_name}_gk_spells_n_{lbl}"])
            out[f"{b_name}_gk_quick_kick_share_{lbl}"] = _safe_ratio(c[f"{b_name}_gk_quick_{lbl}"], c[f"{b_name}_gk_kicks_{lbl}"])
            out[f"{b_name}_def_x_{lbl}"] = c[f"{b_name}_def_x_{lbl}"]
            out[f"{b_name}_role_inversion_{lbl}"] = c[f"{b_name}_role_inversion_{lbl}"]
            out[f"{b_name}_kick_self_share_{lbl}"] = _safe_ratio(c[f"{b_name}_kick_self_{lbl}"], c[f"{b_name}_kick_outfield_n_{lbl}"])
        for lbl in ("A", "B", "both"):
            out[f"{b_name}_shots_per_min_{lbl}"] = _safe_ratio(c[f"{b_name}_shots_{lbl}"], c[f"{b_name}_shots_minutes"])
        out[f"{b_name}_field_5s_per_min_both"] = _safe_ratio(c[f"{b_name}_field_5s_both"], c[f"{b_name}_viol_minutes"])
        out[f"{b_name}_keeper_5s_per_min_both"] = _safe_ratio(c[f"{b_name}_keeper_5s_both"], c[f"{b_name}_viol_minutes"])
        out[f"{b_name}_violations_5s_per_min_both"] = _safe_ratio(c[f"{b_name}_violations_5s_both"], c[f"{b_name}_viol_minutes"])
    return out


# ----------------------------------------------------------------------------
# Per-match record and aggregation
# ----------------------------------------------------------------------------
def match_metrics(trace: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Flat record of one match: scalars, event counts and the raw holder-spell durations."""
    holder = _holder(trace)
    n_steps = int(holder.shape[0])
    m: Dict[str, Any] = {"n_steps": n_steps}
    final = np.asarray(trace["final_score"]) if "final_score" in trace else np.asarray(trace["s_score"])[-1]
    m["goals_A"], m["goals_B"] = float(final[0]), float(final[1])
    m["goals_total"] = m["goals_A"] + m["goals_B"]
    m["goal_diff"] = m["goals_A"] - m["goals_B"]
    m["scoreless"] = 1.0 if m["goals_total"] == 0 else 0.0
    hs = hold_share(trace)
    m["hold_share_A"], m["hold_share_B"] = float(hs[0]), float(hs[1])
    m["hold_share_pooled"] = m["hold_share_A"] + m["hold_share_B"]
    m["loose_share"] = loose_share(trace)

    sp_players, sp_starts, sp_ends = holder_spell_intervals(trace)
    spells = (sp_ends - sp_starts) * STEP_SECONDS
    m["spell_seconds"] = spells
    m["spells_n"], m["spells_sum"] = int(spells.size), float(spells.sum())
    mask_a = sp_players < TEAM_SIZE
    m["spells_n_A"], m["spells_sum_A"] = int(mask_a.sum()), float(spells[mask_a].sum())
    m["spells_n_B"], m["spells_sum_B"] = int((~mask_a).sum()), float(spells[~mask_a].sum())

    kicks = kick_events(trace)
    outcomes = kick_outcomes(trace, kicks=kicks)
    counts = outfield_outcome_counts(outcomes)
    m["kicks_n"] = counts["n"]
    for name in OUTCOMES:
        m[f"kick_{name}"] = counts[name]
    options = pass_options(trace, kicks=kicks, outcomes=outcomes)
    m.update(pass_option_counts(options))
    m.update(aim_reach_counts(options))
    m.update(basin_counts(trace))
    m.update(defence_counts(trace))

    kp_counts = kick_propensity_counts(trace, holder=holder, kicks=kicks)
    for k, (kn, hn) in kp_counts.items():
        m[f"kick_prop_kicks_{k}"] = kn
        m[f"kick_prop_holds_{k}"] = hn

    for radius in (100, 150):
        w = n_within(trace, float(radius))
        m[f"n_within_{radius}_A"], m[f"n_within_{radius}_B"] = float(w[0]), float(w[1])
        m[f"n_within_{radius}_pooled"] = 0.5 * (m[f"n_within_{radius}_A"] + m[f"n_within_{radius}_B"])
    c = cluster_ge4(trace)
    m["cluster_ge4_A"], m["cluster_ge4_B"] = float(c[0]), float(c[1])
    m["cluster_ge4_pooled"] = 0.5 * (m["cluster_ge4_A"] + m["cluster_ge4_B"])
    r = role_inversion(trace)
    m["role_inversion_A"], m["role_inversion_B"] = float(r[0]), float(r[1])
    m["role_inversion_pooled"] = 0.5 * (m["role_inversion_A"] + m["role_inversion_B"])

    m.update(gk_geometry(trace))

    eps = gk_opportunities(trace)
    m["gk_episodes"] = len(eps)
    m["gk_pickups"] = sum(e["pickup"] for e in eps)
    m["gk_within30"] = sum(e["within30"] for e in eps)
    m["gk_within30_pickups"] = sum(e["within30"] and e["pickup"] for e in eps)

    g_events = goal_events(trace, holder=holder, kicks=kicks)
    gt = goal_type_counts(g_events)
    m.update({f"goals_{k}": v for k, v in gt.items()})
    m["goals_distances"] = np.array([g["kick_distance"] for g in g_events if g["kick_distance"] is not None])

    m.update(
        time_binned_counts(
            trace,
            holder=holder,
            spell_intervals=(sp_players, sp_starts, sp_ends),
            kicks=kicks,
            outcomes=outcomes,
        )
    )
    return m


def mean_se(values: Iterable[Optional[float]]) -> tuple[Optional[float], Optional[float], int]:
    """Mean, standard error of the mean (sample sd / sqrt(n)) and n, skipping None and NaN."""
    vals = np.array([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=np.float64)
    n = int(vals.size)
    if n == 0:
        return None, None, 0
    if n == 1:
        return float(vals[0]), None, 1
    return float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(n)), n


def ratio_se(num: Sequence[float], den: Sequence[float]) -> tuple[Optional[float], Optional[float], int]:
    """Pooled ratio sum(num)/sum(den) with the cluster-robust SE over the matches that have den > 0."""
    y = np.asarray(num, dtype=np.float64)
    x = np.asarray(den, dtype=np.float64)
    keep = x > 0
    y, x = y[keep], x[keep]
    n = int(x.size)
    if n == 0:
        return None, None, 0
    r = float(y.sum() / x.sum())
    if n == 1:
        return r, None, 1
    resid = y - r * x
    return r, float(math.sqrt(n / (n - 1) * float((resid ** 2).sum())) / x.sum()), n


def _stat_dict(stat: tuple[Optional[float], Optional[float], int]) -> Dict[str, Any]:
    return {"mean": stat[0], "se": stat[1], "n": stat[2]}


def summarize(per_match: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-match records into `{n_matches, metrics, medians}` with mean/se/n per metric."""
    n = len(per_match)
    col = lambda key: [m[key] for m in per_match]  # noqa: E731
    metrics: Dict[str, Dict[str, Any]] = {}

    scalar_keys = (
        "goals_A", "goals_B", "goals_total", "goal_diff", "scoreless",
        "hold_share_A", "hold_share_B", "hold_share_pooled", "loose_share",
        "n_within_100_A", "n_within_100_B", "n_within_100_pooled",
        "n_within_150_A", "n_within_150_B", "n_within_150_pooled",
        "cluster_ge4_A", "cluster_ge4_B", "cluster_ge4_pooled",
        "role_inversion_A", "role_inversion_B", "role_inversion_pooled",
        "gk_dist_goal_A", "gk_dist_goal_B", "gk_dist_goal_pooled",
        "gk_in_box_A", "gk_in_box_B", "gk_in_box_pooled",
        "gk_goal_line_A", "gk_goal_line_B", "gk_goal_line_pooled",
    )
    for k in scalar_keys:
        metrics[k] = _stat_dict(mean_se(col(k)))

    metrics["holder_spell_s_pooled"] = _stat_dict(ratio_se(col("spells_sum"), col("spells_n")))
    metrics["holder_spell_s_A"] = _stat_dict(ratio_se(col("spells_sum_A"), col("spells_n_A")))
    metrics["holder_spell_s_B"] = _stat_dict(ratio_se(col("spells_sum_B"), col("spells_n_B")))

    for name in OUTCOMES:
        metrics[f"kick_outcome_{name}"] = _stat_dict(ratio_se(col(f"kick_{name}"), col("kicks_n")))

    for key, (num, den) in PASS_OPTION_SHARES.items():
        metrics[key] = _stat_dict(ratio_se(col(num), col(den)))

    for key, (num, den) in BASIN_SHARES.items():
        metrics[key] = _stat_dict(ratio_se(col(num), col(den)))

    metrics.update(defence_metrics(per_match))

    for k in KICK_PROP_KEYS:
        metrics[f"kick_propensity_{k}"] = _stat_dict(ratio_se(col(f"kick_prop_kicks_{k}"), col(f"kick_prop_holds_{k}")))

    metrics["gk_episodes_per_keeper_match"] = _stat_dict(mean_se([e / 2.0 for e in col("gk_episodes")]))
    metrics["gk_pickup_rate"] = _stat_dict(ratio_se(col("gk_pickups"), col("gk_episodes")))
    metrics["gk_within30_rate"] = _stat_dict(ratio_se(col("gk_within30"), col("gk_episodes")))
    metrics["gk_pickup_given_within30"] = _stat_dict(ratio_se(col("gk_within30_pickups"), col("gk_within30")))

    metrics["goal_own_share"] = _stat_dict(ratio_se(col("goals_own_goals"), col("goals_n")))
    metrics["goal_direct_share"] = _stat_dict(ratio_se(col("goals_direct"), col("goals_n_open_play")))
    metrics["goal_no_pass_10s_share"] = _stat_dict(ratio_se(col("goals_no_pass_10s"), col("goals_n_open_play")))

    for b_name in ("early", "late"):
        for lbl in ("A", "B", "pooled"):
            metrics[f"{b_name}_gk_hold_s_{lbl}"] = _stat_dict(
                ratio_se(col(f"{b_name}_gk_spells_sum_{lbl}"), col(f"{b_name}_gk_spells_n_{lbl}"))
            )
            metrics[f"{b_name}_gk_quick_kick_share_{lbl}"] = _stat_dict(
                ratio_se(col(f"{b_name}_gk_quick_{lbl}"), col(f"{b_name}_gk_kicks_{lbl}"))
            )
            metrics[f"{b_name}_def_x_{lbl}"] = _stat_dict(mean_se(col(f"{b_name}_def_x_{lbl}")))
            metrics[f"{b_name}_role_inversion_{lbl}"] = _stat_dict(mean_se(col(f"{b_name}_role_inversion_{lbl}")))
            metrics[f"{b_name}_kick_self_share_{lbl}"] = _stat_dict(
                ratio_se(col(f"{b_name}_kick_self_{lbl}"), col(f"{b_name}_kick_outfield_n_{lbl}"))
            )
        for lbl in ("A", "B", "both"):
            metrics[f"{b_name}_shots_per_min_{lbl}"] = _stat_dict(
                ratio_se(col(f"{b_name}_shots_{lbl}"), col(f"{b_name}_shots_minutes"))
            )
        for v_name in ("field_5s", "keeper_5s", "violations_5s"):
            metrics[f"{b_name}_{v_name}_per_min_both"] = _stat_dict(
                ratio_se(col(f"{b_name}_{v_name}_both"), col(f"{b_name}_viol_minutes"))
            )

    spells = np.concatenate([m["spell_seconds"] for m in per_match]) if n else np.zeros(0)
    dist = np.concatenate([m["goals_distances"] for m in per_match]) if n else np.zeros(0)
    medians = {
        "holder_spell_s": float(np.median(spells)) if spells.size else None,
        "goal_kick_distance": float(np.median(dist)) if dist.size else None,
    }
    return {"n_matches": n, "metrics": metrics, "medians": medians}


def _fmt(stat: tuple, nd: int = 3, scale: float = 1.0, unit: str = "") -> str:
    mean, se, n = stat
    if mean is None:
        return f"-  (n={n})"
    se_text = "n/a" if se is None else f"{se * scale:.{nd}f}"
    return f"{mean * scale:.{nd}f}{unit} +- {se_text}{unit} (n={n})"


def _fmt_d(entry: Dict[str, Any], nd: int = 3, scale: float = 1.0, unit: str = "") -> str:
    return _fmt((entry["mean"], entry["se"], entry["n"]), nd=nd, scale=scale, unit=unit)


def report(per_match: Sequence[Dict[str, Any]], title: str = "") -> str:
    """Printable summary of the per-match records (mean +- SE with n; see the module docstring)."""
    n = len(per_match)
    col = lambda key: [m[key] for m in per_match]  # noqa: E731
    s = summarize(per_match)
    met = s["metrics"]
    lines = [f"===== {title}  n={n} matches"]
    lines.append(
        "goals A / B / total / diff / scoreless: "
        + " / ".join(_fmt_d(met[k], 3) for k in ("goals_A", "goals_B", "goals_total", "goal_diff", "scoreless"))
    )
    lines.append(
        "hold_share A / B / pooled, loose: "
        + " / ".join(_fmt_d(met[k], 4) for k in ("hold_share_A", "hold_share_B", "hold_share_pooled", "loose_share"))
    )
    spells = np.concatenate([m["spell_seconds"] for m in per_match]) if n else np.zeros(0)
    lines.append(
        f"holder_spell_s             : pooled {_fmt_d(met['holder_spell_s_pooled'], 3)} "
        f"(A {_fmt_d(met['holder_spell_s_A'], 3)}, B {_fmt_d(met['holder_spell_s_B'], 3)}), "
        f"median {np.median(spells) if spells.size else float('nan'):.3f} s over {spells.size} spells"
    )
    total_kicks = sum(col("kicks_n"))
    shares = " | ".join(f"{name} {_fmt_d(met[f'kick_outcome_{name}'], 1, 100.0, '%')}" for name in OUTCOMES)
    lines.append(f"kick outcome (outfield, {total_kicks} kicks): {shares}")
    lines.append(
        f"pass options ({sum(col('pass_opt_n'))} kicks)    : "
        f"option {_fmt_d(met['pass_option_share'], 1, 100.0, '%')} | "
        f"aim_open {_fmt_d(met['aim_open_share'], 1, 100.0, '%')} | "
        f"completion {_fmt_d(met['aim_open_completion'], 1, 100.0, '%')} | "
        f"no_option_opponent {_fmt_d(met['no_option_opponent_share'], 1, 100.0, '%')} | "
        f"reachable {_fmt_d(met['open_reachable_share'], 1, 100.0, '%')} | "
        f"aim|reach {_fmt_d(met['aim_open_reachable_share'], 1, 100.0, '%')} | "
        f"arc baseline {_fmt_d(met['aim_open_arc_baseline'], 1, 100.0, '%')}"
    )
    lines.append(
        "basin: reverse " + _fmt_d(met["reverse_motion_share_pooled"], 3)
        + " | facing ball " + _fmt_d(met["facing_ball_share_pooled"], 3)
        + " | tackle-ready " + _fmt_d(met["tackle_ready_share_pooled"], 3)
    )
    lines.append(
        "kick_propensity A/B/pooled : "
        + " / ".join(_fmt_d(met[f"kick_propensity_{k}"], 4) for k in ("A", "B", "pooled"))
        + " | roles GK/D/M/ST: "
        + " / ".join(_fmt_d(met[f"kick_propensity_{k}"], 4) for k in ("GK", "D", "M", "ST"))
    )
    for key, label in (
        ("n_within_100", "n_within(100) A / B / pool"),
        ("n_within_150", "n_within(150) A / B / pool"),
        ("cluster_ge4", "cluster_ge4 A / B / pool"),
        ("role_inversion", "role_inversion A / B / pool"),
        ("gk_dist_goal", "gk_dist_goal A / B / pool"),
        ("gk_in_box", "gk_in_box(195) A / B / pool"),
        ("gk_goal_line", "gk_goal_line A / B / pool"),
    ):
        lines.append(f"{label:27s}: " + " / ".join(_fmt_d(met[f"{key}_{t}"], 3) for t in ("A", "B", "pooled")))
    lines.append(
        "gk_opportunities           : "
        f"{_fmt_d(met['gk_episodes_per_keeper_match'], 2)} episodes per keeper-match, "
        f"pickup {_fmt_d(met['gk_pickup_rate'], 3)}, "
        f"within30 {_fmt_d(met['gk_within30_rate'], 3)}, "
        f"pickup | within30 {_fmt_d(met['gk_pickup_given_within30'], 3)}"
    )
    n_goals = sum(col("goals_n"))
    dist = np.concatenate([m["goals_distances"] for m in per_match]) if n else np.zeros(0)
    lines.append(
        f"goal_types ({n_goals} goals)      : "
        f"own {_fmt_d(met['goal_own_share'], 3)}, "
        f"direct {_fmt_d(met['goal_direct_share'], 3)}, "
        f"no pass in 10 s {_fmt_d(met['goal_no_pass_10s_share'], 3)}, "
        f"kicker distance median {np.median(dist) if dist.size else float('nan'):.0f} u"
    )
    lines.append(
        "time_binned (early vs late): "
        f"def_x pool {_fmt_d(met['early_def_x_pooled'], 1)} -> {_fmt_d(met['late_def_x_pooled'], 1)} | "
        f"inv pool {_fmt_d(met['early_role_inversion_pooled'], 3)} -> {_fmt_d(met['late_role_inversion_pooled'], 3)} | "
        f"shots/m both {_fmt_d(met['early_shots_per_min_both'], 2)} -> {_fmt_d(met['late_shots_per_min_both'], 2)} | "
        f"5s/m both {_fmt_d(met['early_violations_5s_per_min_both'], 3)} -> {_fmt_d(met['late_violations_5s_per_min_both'], 3)}"
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Loading and command line
# ----------------------------------------------------------------------------
def _trace_paths(directory: Path | str) -> List[Path]:
    paths = sorted(Path(directory).glob("m*.npz"))
    if not paths:
        raise FileNotFoundError(f"no m*.npz traces in {directory}")
    return paths


def iter_traces(directory: Path | str) -> Iterator[Dict[str, np.ndarray]]:
    for path in _trace_paths(directory):
        with np.load(path) as z:
            yield {k: z[k] for k in z.files}


def load_traces(directory: Path | str) -> List[Dict[str, np.ndarray]]:
    return list(iter_traces(directory))


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def census_payload(per_match: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the JSON-serializable census payload (`summary` + `per_match`) without path metadata."""
    return {
        "summary": summarize(per_match),
        "per_match": [{k: _json_safe(v) for k, v in m.items()} for m in per_match],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace_dirs", nargs="+", help="directories of m*.npz traces written by harness.py")
    ap.add_argument("--json", default=None, help="write the summary and per-match records of every directory here")
    args = ap.parse_args(argv)
    dump: Dict[str, Any] = {}
    for directory in args.trace_dirs:
        per_match = [match_metrics(t) for t in iter_traces(directory)]
        print(report(per_match, title=str(directory)))
        dump[str(directory)] = census_payload(per_match)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(dump, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
