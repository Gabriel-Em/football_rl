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
    "hold_share", "loose_share", "holder_spells", "kick_events", "kick_outcomes", "outfield_outcome_counts",
    "n_within", "max_component_size", "cluster_ge4", "role_inversion", "gk_opportunities", "goal_events",
    "goal_type_counts", "match_metrics", "mean_se", "ratio_se", "report", "load_traces", "main",
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
TEAM_SIZE = 6
TEAMS = ("A", "B")


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


def holder_spells(trace: Dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Holder (player index) and duration in seconds of every uninterrupted holding spell."""
    h = _holder(trace)
    if h.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    changes = np.flatnonzero(np.diff(h) != 0) + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [h.size]])
    keep = h[starts] >= 0
    return h[starts][keep], (ends - starts)[keep] * STEP_SECONDS


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


def kick_outcomes(trace: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """Outcome of every kick whose window is complete or already resolved (see the module docstring)."""
    holder = _holder(trace)
    score = np.asarray(trace["s_score"]).astype(np.int64)
    n = holder.size
    steps, players = kick_events(trace)
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


def role_inversion(trace: Dict[str, np.ndarray], margin: float = INVERSION_MARGIN) -> np.ndarray:
    """Fraction of decisions with D1 or D2 more than `margin` ahead of the striker, shape (2,)."""
    pos = np.asarray(trace["pos"], dtype=np.float64)
    side = np.asarray(trace["side"])
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
def goal_events(trace: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """One record per goal, in match order (see `goal_types` in the module docstring)."""
    score = np.asarray(trace["s_score"]).astype(np.int64)
    n = score.shape[0]
    if n == 0:
        return []
    holder = _holder(trace)
    passes = np.asarray(trace["s_team_passes"]).astype(np.int64)
    own_goals = np.asarray(trace["s_own_goals"]).astype(np.int64)
    pos = np.asarray(trace["pos"], dtype=np.float64)
    side = np.asarray(trace["side"])
    kick_steps, kick_players = kick_events(trace)

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
            record["direct"] = bool(step - t_kick <= DIRECT_KICK_STEPS and not (holder[t_kick + 1:step + 1] >= 0).any())
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


# ----------------------------------------------------------------------------
# Per-match record and aggregation
# ----------------------------------------------------------------------------
def match_metrics(trace: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Flat record of one match: scalars, event counts and the raw holder-spell durations."""
    m: Dict[str, Any] = {"n_steps": int(np.asarray(trace["holder"]).shape[0])}
    final = np.asarray(trace["final_score"]) if "final_score" in trace else np.asarray(trace["s_score"])[-1]
    m["goals_A"], m["goals_B"] = float(final[0]), float(final[1])
    m["goals_total"] = m["goals_A"] + m["goals_B"]
    hs = hold_share(trace)
    m["hold_share_A"], m["hold_share_B"], m["loose_share"] = float(hs[0]), float(hs[1]), loose_share(trace)
    _, spells = holder_spells(trace)
    m["spell_seconds"] = spells
    m["spells_n"], m["spells_sum"] = int(spells.size), float(spells.sum())
    counts = outfield_outcome_counts(kick_outcomes(trace))
    m["kicks_n"] = counts["n"]
    for name in OUTCOMES:
        m[f"kick_{name}"] = counts[name]
    for radius in (100, 150):
        w = n_within(trace, float(radius))
        m[f"n_within_{radius}_A"], m[f"n_within_{radius}_B"] = float(w[0]), float(w[1])
    c = cluster_ge4(trace)
    m["cluster_ge4_A"], m["cluster_ge4_B"] = float(c[0]), float(c[1])
    r = role_inversion(trace)
    m["role_inversion_A"], m["role_inversion_B"] = float(r[0]), float(r[1])
    eps = gk_opportunities(trace)
    m["gk_episodes"] = len(eps)
    m["gk_pickups"] = sum(e["pickup"] for e in eps)
    m["gk_within30"] = sum(e["within30"] for e in eps)
    m["gk_within30_pickups"] = sum(e["within30"] and e["pickup"] for e in eps)
    gt = goal_type_counts(goal_events(trace))
    m.update({f"goals_{k}": v for k, v in gt.items()})
    m["goals_distances"] = np.array([g["kick_distance"] for g in goal_events(trace) if g["kick_distance"] is not None])
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


def _fmt(stat: tuple, nd: int = 3, scale: float = 1.0, unit: str = "") -> str:
    mean, se, n = stat
    if mean is None:
        return f"-  (n={n})"
    se_text = "n/a" if se is None else f"{se * scale:.{nd}f}"
    return f"{mean * scale:.{nd}f}{unit} +- {se_text}{unit} (n={n})"


def report(per_match: Sequence[Dict[str, Any]], title: str = "") -> str:
    """Printable summary of the per-match records (mean +- SE with n; see the module docstring)."""
    n = len(per_match)
    col = lambda key: [m[key] for m in per_match]  # noqa: E731
    lines = [f"===== {title}  n={n} matches"]
    lines.append(
        "goals A / B / total        : "
        + " / ".join(_fmt(mean_se(col(k)), 3) for k in ("goals_A", "goals_B", "goals_total"))
    )
    lines.append(
        "hold_share A / B, loose    : "
        + " / ".join(_fmt(mean_se(col(k)), 4) for k in ("hold_share_A", "hold_share_B", "loose_share"))
    )
    spells = np.concatenate([m["spell_seconds"] for m in per_match]) if n else np.zeros(0)
    spell_mean = ratio_se(col("spells_sum"), col("spells_n"))
    lines.append(
        f"holder_spell_s             : mean {_fmt(spell_mean, 3)}, median "
        f"{np.median(spells) if spells.size else float('nan'):.3f} s over {spells.size} spells"
    )
    total_kicks = sum(col("kicks_n"))
    shares = " | ".join(f"{name} {_fmt(ratio_se(col(f'kick_{name}'), col('kicks_n')), 1, 100.0, '%')}" for name in OUTCOMES)
    lines.append(f"kick outcome (outfield, {total_kicks} kicks): {shares}")
    for key, label in (
        ("n_within_100", "n_within(100) A / B"),
        ("n_within_150", "n_within(150) A / B"),
        ("cluster_ge4", "cluster_ge4 A / B"),
        ("role_inversion", "role_inversion A / B"),
    ):
        lines.append(f"{label:27s}: " + " / ".join(_fmt(mean_se(col(f"{key}_{t}")), 3) for t in TEAMS))
    per_keeper = [e / 2.0 for e in col("gk_episodes")]
    lines.append(
        "gk_opportunities           : "
        f"{_fmt(mean_se(per_keeper), 2)} episodes per keeper-match, "
        f"pickup {_fmt(ratio_se(col('gk_pickups'), col('gk_episodes')), 3)}, "
        f"within30 {_fmt(ratio_se(col('gk_within30'), col('gk_episodes')), 3)}, "
        f"pickup | within30 {_fmt(ratio_se(col('gk_within30_pickups'), col('gk_within30')), 3)}"
    )
    n_goals = sum(col("goals_n"))
    dist = np.concatenate([m["goals_distances"] for m in per_match]) if n else np.zeros(0)
    lines.append(
        f"goal_types ({n_goals} goals)      : "
        f"own {_fmt(ratio_se(col('goals_own_goals'), col('goals_n')), 3)}, "
        f"direct {_fmt(ratio_se(col('goals_direct'), col('goals_n_open_play')), 3)}, "
        f"no pass in 10 s {_fmt(ratio_se(col('goals_no_pass_10s'), col('goals_n_open_play')), 3)}, "
        f"kicker distance median {np.median(dist) if dist.size else float('nan'):.0f} u"
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace_dirs", nargs="+", help="directories of m*.npz traces written by harness.py")
    ap.add_argument("--json", default=None, help="write the per-match records of every directory here")
    args = ap.parse_args(argv)
    dump: Dict[str, Any] = {}
    for directory in args.trace_dirs:
        per_match = [match_metrics(t) for t in iter_traces(directory)]
        print(report(per_match, title=str(directory)))
        dump[str(directory)] = [{k: _json_safe(v) for k, v in m.items()} for m in per_match]
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(dump))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
