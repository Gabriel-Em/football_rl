"""Per-match behavioural metrics from recorded replays (read-only).

Metrics per team (index 0 = team A / red, 1 = team B / blue):
possession share, passes, outfield width/depth, players near the ball, ball-centred
clustering, role inversion (a defender ahead of the striker in the opposing half),
presser-role share, and goalkeeper distance / pickup / hold / near-miss statistics.

Usage:
  python tools/audit/replay_metrics.py [REPLAY ...] [--replays-dir DIR] [--match ID]
         [--gen G] [--last N] [--json OUT]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.audit._common import (  # noqa: E402
    FLAG_GHOST,
    FLAG_STUN,
    PLAYERS_PER_TEAM,
    REPO_ROOT,
    SLOT_NAMES,
    TEAM_LABELS,
    fmt,
    load_json,
    replay_arrays,
    safe_mean,
    select_replays,
    write_json,
)

__all__ = ["compute_match_metrics", "summarize", "replay_guards", "REPLAY_GUARD_KEYS", "select_replays", "main"]

PASS_EVENT_TYPES = ("PASS!", "THROUGH BALL!", "WALL PASS!")

# Guard metrics derived purely from replays (the remaining run_all guards come from history / perf).
REPLAY_GUARD_KEYS = (
    "ball_hold_rate", "gk_dist_goal_line", "gk_pickups_per_match", "role_inversion_rate", "mean_cluster_size",
    "gk_wall_pinned_rate", "gk_near_miss_frames_per_match", "gk_facing_blocked_frames_per_match",
    "gk_dives_per_match", "presser_trailing_rate", "presser_trailing_avoidable_rate", "longest_hold_spell_s",
    "loose_share",
)


def _empty_metrics(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "match_id": meta.get("match_id"),
        "generation": meta.get("generation"),
        "n_frames": 0,
        "ball_hold_rate": None,
        "possession_share": [None, None],
        "loose_share": None,
        "passes": [0, 0],
        "width_mean": [None, None],
        "depth_mean": [None, None],
        "near_ball_mean": [None, None],
        "cluster_frames": 0,
        "cluster_frame_rate": None,
        "mean_cluster_size": None,
        "inversion_rate": [None, None],
        "inversion_rate_oop": [None, None],
        "gk_dist_ball_mean": [None, None],
        "gk_dist_goal_line_mean": [None, None],
        "gk_dist_goal_center_mean": [None, None],
        "gk_wall_pinned_rate": [None, None],
        "gk_pickups": [0, 0],
        "gk_hold_mean_s": [None, None],
        "gk_near_miss_frames": [0, 0],
        "presser_share": [{}, {}],
        "presser_trailing_rate": [None, None],
        "presser_trailing_avoidable_rate": [None, None],
        "gk_facing_blocked_frames": [0, 0],
        "gk_dives": [0, 0],
        "longest_hold_spell_s": None,
    }


def _passes(frames: List[Dict[str, Any]]) -> List[int]:
    last_stats = (frames[-1].get("stats") or {}) if frames else {}
    if isinstance(last_stats.get("passes"), list) and len(last_stats["passes"]) == 2:
        return [int(last_stats["passes"][0]), int(last_stats["passes"][1])]
    counts = [0, 0]
    for fr in frames:
        for ev in fr.get("events") or []:
            team = ev.get("team", -1)
            if ev.get("type") in PASS_EVENT_TYPES and team in (0, 1):
                counts[team] += 1
    return counts


def _presser_share(frames: List[Dict[str, Any]], team: int) -> Dict[str, float]:
    base = team * PLAYERS_PER_TEAM
    counts = np.zeros(PLAYERS_PER_TEAM, dtype=np.int64)
    for fr in frames:
        rs = fr.get("role_state")
        if not rs:
            continue
        for k in range(PLAYERS_PER_TEAM):
            if rs[base + k] == "1stP":
                counts[k] += 1
    total = int(counts.sum())
    if total == 0:
        return {}
    return {SLOT_NAMES[k]: float(counts[k]) / total for k in range(PLAYERS_PER_TEAM)}


def _presser_trailing(frames: List[Dict[str, Any]], cx_team: np.ndarray, ball_cx: np.ndarray,
                      flags: np.ndarray, team: int, margin: float) -> Tuple[Optional[float], Optional[float]]:
    """Presser-trailing rates over frames with a `1stP` presser (None when there are none).

    Trailing: the presser's canonical x exceeds the ball's by more than `margin`.
    Avoidable trailing: additionally, another outfielder of the team who is neither stunned nor ghosted
    stands at canonical x <= ball x + `margin`, i.e. a goal-side player could have been the presser.
    """
    base = team * PLAYERS_PER_TEAM
    n_press = 0
    n_trail = 0
    n_avoid = 0
    for f_i, fr in enumerate(frames):
        rs = fr.get("role_state")
        if not rs:
            continue
        for k in range(1, PLAYERS_PER_TEAM):
            if rs[base + k] == "1stP":
                n_press += 1
                if cx_team[f_i, base + k] - ball_cx[f_i] > margin:
                    n_trail += 1
                    # The trailing presser itself is upfield of ball + margin, so it never qualifies as cover.
                    mates_cx = cx_team[f_i, base + 1:base + PLAYERS_PER_TEAM]
                    active = (flags[f_i, base + 1:base + PLAYERS_PER_TEAM] & (FLAG_STUN | FLAG_GHOST)) == 0
                    if bool((active & (mates_cx <= ball_cx[f_i] + margin)).any()):
                        n_avoid += 1
                break
    if not n_press:
        return None, None
    return n_trail / n_press, n_avoid / n_press


def _longest_spell_frames(holder: np.ndarray, max_gap: int) -> int:
    """Longest run held by one player, bridging loose gaps of at most `max_gap` frames."""
    best = 0
    cur_player = -1
    start = 0
    last = -1
    for f_i, h in enumerate(holder.tolist()):
        if h < 0:
            continue
        if h == cur_player and f_i - last - 1 <= max_gap:
            last = f_i
        else:
            cur_player, start, last = h, f_i, f_i
        best = max(best, last - start + 1)
    return best


def compute_match_metrics(
    replay: Dict[str, Any],
    *,
    cluster_radius: float = 90.0,
    cluster_n: int = 6,
    near_radius: float = 100.0,
    inversion_margin: float = 30.0,
    pickup_distance: float = 28.0,
    near_miss_window: float = 10.0,
    trail_margin: float = 50.0,
    spell_gap_frames: int = 2,
) -> Dict[str, Any]:
    """Compute behavioural metrics for one replay dict (raw 60 Hz frames, no cinematics)."""
    ra = replay_arrays(replay)
    out = _empty_metrics(ra.meta)
    n = ra.n_frames
    if n == 0:
        return out

    hz = float(ra.meta.get("physics_hz", 60))
    half_w = 0.5 * ra.field_width()
    gcy = ra.goal_center_y()
    box_r = float(ra.meta.get("keeper_box_radius", 195.0))
    p_rad = float(ra.meta.get("player_radius", 18.0))

    holder = ra.holder
    held = holder >= 0
    holder_team = np.where(held, holder // PLAYERS_PER_TEAM, -1)
    out["n_frames"] = n
    out["ball_hold_rate"] = float(held.mean())
    out["possession_share"] = [float((holder_team == t).mean()) for t in (0, 1)]
    out["loose_share"] = float((~held).mean())
    out["passes"] = _passes(ra.frames)
    out["longest_hold_spell_s"] = _longest_spell_frames(holder, spell_gap_frames) / hz

    d_ball = np.hypot(ra.pos[:, :, 0] - ra.ball[:, None, 0], ra.pos[:, :, 1] - ra.ball[:, None, 1])  # (F, 12)
    cluster = (d_ball <= cluster_radius).sum(axis=1)
    out["cluster_frames"] = int((cluster >= cluster_n).sum())
    out["cluster_frame_rate"] = float((cluster >= cluster_n).mean())
    out["mean_cluster_size"] = float(cluster.mean())

    for t in (0, 1):
        base = t * PLAYERS_PER_TEAM
        outfield = slice(base + 1, base + PLAYERS_PER_TEAM)
        cx = ra.canonical_x(t)
        ys = ra.pos[:, outfield, 1]
        out["width_mean"][t] = float((ys.max(axis=1) - ys.min(axis=1)).mean())
        cxo = cx[:, outfield]
        out["depth_mean"][t] = float((cxo.max(axis=1) - cxo.min(axis=1)).mean())
        out["near_ball_mean"][t] = float((d_ball[:, base:base + PLAYERS_PER_TEAM] <= near_radius).sum(axis=1).mean())

        st_x = cx[:, base + 5]
        d_x = cx[:, base + 1:base + 3]
        inverted = ((d_x > st_x[:, None] + inversion_margin) & (d_x > half_w)).any(axis=1)
        out["inversion_rate"][t] = float(inverted.mean())
        oop = holder_team != t
        out["inversion_rate_oop"][t] = float(inverted[oop].mean()) if oop.any() else None

        gk = base
        gx = ra.own_goal_x(t)
        gk_x = ra.pos[:, gk, 0]
        gk_y = ra.pos[:, gk, 1]
        d_line = np.abs(gk_x - gx)
        out["gk_dist_ball_mean"][t] = float(d_ball[:, gk].mean())
        out["gk_dist_goal_line_mean"][t] = float(d_line.mean())
        out["gk_dist_goal_center_mean"][t] = float(np.hypot(gk_x - gx, gk_y - gcy).mean())
        out["gk_wall_pinned_rate"][t] = float((d_line <= p_rad + 0.5).mean())

        gk_hold = holder == gk
        prev = np.concatenate(([False], gk_hold[:-1]))
        starts = int((gk_hold & ~prev).sum())
        out["gk_pickups"][t] = starts
        out["gk_hold_mean_s"][t] = (float(gk_hold.sum()) / starts / hz) if starts else None

        ball_in_box = np.hypot(ra.ball[:, 0] - gx, ra.ball[:, 1] - gcy) <= box_r
        dg = d_ball[:, gk]
        near_miss = (~held) & ball_in_box & (dg > pickup_distance) & (dg <= pickup_distance + near_miss_window)
        out["gk_near_miss_frames"][t] = int(near_miss.sum())

        out["presser_share"][t] = _presser_share(ra.frames, t)
        ball_cx = (ra.ball[:, 0] - gx) * np.where(ra.team_side[:, t] == 0, 1.0, -1.0)
        (out["presser_trailing_rate"][t],
         out["presser_trailing_avoidable_rate"][t]) = _presser_trailing(ra.frames, cx, ball_cx, ra.flags, t,
                                                                         trail_margin)

        dxg = ra.ball[:, 0] - gk_x
        dyg = ra.ball[:, 1] - gk_y
        cos_b = (np.cos(ra.angle[:, gk]) * dxg + np.sin(ra.angle[:, gk]) * dyg) / np.maximum(dg, 1e-9)
        out["gk_facing_blocked_frames"][t] = int(((~held) & (dg <= pickup_distance) & (cos_b < -0.15)).sum())
        lunge = (ra.flags[:, gk] & 8) != 0
        out["gk_dives"][t] = int((lunge & ~np.concatenate(([False], lunge[:-1]))).sum())
    return out


def summarize(per_match: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean of every numeric metric across matches (None values skipped)."""
    summary: Dict[str, Any] = {"n_matches": len(per_match)}
    if not per_match:
        return summary
    skip = {"match_id", "generation", "path"}
    for key, sample in per_match[0].items():
        if key in skip:
            continue
        vals = [m.get(key) for m in per_match]
        if isinstance(sample, list) and sample and isinstance(sample[0], dict):
            merged = []
            for t in range(len(sample)):
                slots = sorted({s for v in vals for s in (v[t] if v else {})})
                merged.append({s: safe_mean([v[t][s] for v in vals if v and s in v[t]]) for s in slots})
            summary[key] = merged
        elif isinstance(sample, list):
            summary[key] = [safe_mean([v[t] for v in vals if v is not None and v[t] is not None])
                            for t in range(len(sample))]
        else:
            summary[key] = safe_mean([v for v in vals if v is not None])
    return summary


def _pair_mean(v: Any) -> Optional[float]:
    if not isinstance(v, list):
        return None
    return safe_mean([x for x in v if x is not None]) if any(x is not None for x in v) else None


def _pair_sum(v: Any) -> Optional[float]:
    if not isinstance(v, list) or not any(x is not None for x in v):
        return None
    return float(sum(x for x in v if x is not None))


def replay_guards(summary: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Scalar replay guard metrics from a `summarize` result (per-team pairs collapsed to one number)."""
    s = summary or {}
    return {
        "ball_hold_rate": s.get("ball_hold_rate"),
        "gk_dist_goal_line": _pair_mean(s.get("gk_dist_goal_line_mean")),
        "gk_pickups_per_match": _pair_sum(s.get("gk_pickups")),
        "role_inversion_rate": _pair_mean(s.get("inversion_rate")),
        "mean_cluster_size": s.get("mean_cluster_size"),
        "gk_wall_pinned_rate": _pair_mean(s.get("gk_wall_pinned_rate")),
        "gk_near_miss_frames_per_match": _pair_sum(s.get("gk_near_miss_frames")),
        "gk_facing_blocked_frames_per_match": _pair_sum(s.get("gk_facing_blocked_frames")),
        "gk_dives_per_match": _pair_sum(s.get("gk_dives")),
        "presser_trailing_rate": _pair_mean(s.get("presser_trailing_rate")),
        "presser_trailing_avoidable_rate": _pair_mean(s.get("presser_trailing_avoidable_rate")),
        "longest_hold_spell_s": s.get("longest_hold_spell_s"),
        "loose_share": s.get("loose_share"),
    }


def _summary_lines(per_match: Sequence[Dict[str, Any]], summary: Dict[str, Any], max_rows: int = 8) -> List[str]:
    lines = [f"replay_metrics: {summary['n_matches']} match(es)  (teams: {TEAM_LABELS[0]}, {TEAM_LABELS[1]})"]
    for m in list(per_match)[-max_rows:]:
        lines.append(
            f"  match {m['match_id']} gen {m['generation']}: frames={m['n_frames']} hold={fmt(m['ball_hold_rate'])} "
            f"inv={fmt(m['inversion_rate'], 2)} clus={fmt(m['mean_cluster_size'], 2)} "
            f"gk_pick={m['gk_pickups']} gk_line={fmt(m['gk_dist_goal_line_mean'], 1)}"
        )
    if summary["n_matches"]:
        keys = ["ball_hold_rate", "possession_share", "loose_share", "passes", "width_mean", "depth_mean",
                "near_ball_mean", "cluster_frame_rate", "mean_cluster_size", "inversion_rate", "inversion_rate_oop",
                "gk_dist_ball_mean", "gk_dist_goal_line_mean", "gk_wall_pinned_rate", "gk_pickups", "gk_hold_mean_s",
                "gk_near_miss_frames", "gk_facing_blocked_frames", "gk_dives", "presser_trailing_rate",
                "presser_trailing_avoidable_rate", "longest_hold_spell_s"]
        lines.append("  mean over matches:")
        for k in keys:
            lines.append(f"    {k:24s} {fmt(summary.get(k))}")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replays", nargs="*", help="explicit replay JSON paths")
    ap.add_argument("--replays-dir", default=str(REPO_ROOT / "replays"))
    ap.add_argument("--match", type=int, default=None)
    ap.add_argument("--gen", type=int, default=None)
    ap.add_argument("--last", type=int, default=None, help="newest N replays (default 1 without other filters)")
    ap.add_argument("--cluster-radius", type=float, default=90.0)
    ap.add_argument("--cluster-n", type=int, default=6)
    ap.add_argument("--near-radius", type=float, default=100.0)
    ap.add_argument("--inversion-margin", type=float, default=30.0)
    ap.add_argument("--pickup-distance", type=float, default=28.0)
    ap.add_argument("--json", default=None, help="write full per-match detail here")
    args = ap.parse_args(argv)

    if args.replays:
        paths = [Path(p) for p in args.replays]
    else:
        last = args.last if args.last is not None else (None if (args.match or args.gen) else 1)
        paths = select_replays(args.replays_dir, match_id=args.match, generation=args.gen, last=last)
    per_match = []
    for p in paths:
        m = compute_match_metrics(
            load_json(p),
            cluster_radius=args.cluster_radius,
            cluster_n=args.cluster_n,
            near_radius=args.near_radius,
            inversion_margin=args.inversion_margin,
            pickup_distance=args.pickup_distance,
        )
        m["path"] = str(p)
        per_match.append(m)
    summary = summarize(per_match)
    print("\n".join(_summary_lines(per_match, summary)))
    if args.json:
        write_json(args.json, {"matches": per_match, "summary": summary})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
