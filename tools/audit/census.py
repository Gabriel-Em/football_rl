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
    "outfield_outcome_counts", "pass_options", "pass_option_counts", "kick_propensity_counts", "kick_propensity", "n_within", "max_component_size",
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
PASS_OPTION_COUNT_KEYS = (
    "pass_opt_n", "pass_opt_with", "aim_open_den", "aim_open_n",
    "aim_comp_den", "aim_comp_mate", "noopt_den", "noopt_opp",
)
LATE_WINDOW_STEPS = 225
TEAM_SIZE = 6
TEAMS = ("A", "B")
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

    Each record: `step`, `player`, `candidates` and `open` (sorted player indices), `has_direction`, `aimed_open`, and
    `outcome` (the `kick_outcomes` result, or None for a censored kick).
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
    return [
        {
            "step": int(k[r]),
            "player": int(i[r]),
            "candidates": mates[r][cand[r]].tolist(),
            "open": mates[r][open_[r]].tolist(),
            "has_direction": bool(has_dir[r]),
            "aimed_open": bool(aimed_open[r]),
            "outcome": by_kick.get((int(k[r]), int(i[r]))),
        }
        for r in range(k.size)
    ]


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
    m.update(pass_option_counts(pass_options(trace, kicks=kicks, outcomes=outcomes)))

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
        f"no_option_opponent {_fmt_d(met['no_option_opponent_share'], 1, 100.0, '%')}"
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
