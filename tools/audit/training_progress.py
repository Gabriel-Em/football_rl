"""Per-generation-bucket training progress from checkpoints/training_history.json (read-only).

Reports mean interest (with median / max), goals, completed passes, shots, tackles,
keeper saves, loose-ball % and stationary-ball frames per bucket of generations.

Every statistic, and the takeoff generation, reads self-play matches only. Pool matches (a team played by
a frozen snapshot from the opponent pool; match record `pool` set) face weaker opponents and would inflate
them; they are counted per bucket (`n_pooled`, printed as `pooled=K` when present) and excluded. A bucket
whose matches are all pool matches has n 0 and no statistics.

With `--takeoff`, also prints the takeoff generation: the smallest generation g such that the mean,
over generations g-window+1 ... g, of the per-generation mean self-play training-match `goals` reaches the
threshold (defaults: window 25, goals 1.5). The window starts at the first generation of the history,
so the earliest possible value is first generation + window - 1. It uses the whole history, ignoring
`--gen-min` and `--gen-max`. Every generation needs a self-play match.

With `--fire`, also prints per bucket the mean of the generation records' `fire_sat` (kick and
special saturation), `p_kick` and `p_spec` for the four roles, labelled by team and drawn colour
(`A/red field`, `A/red keeper`, `B/blue field`, `B/blue keeper`; the JSON keeps the stored `blue_*` and
`red_*` keys); histories of a trainer without fire telemetry print `fire: n/a (no hybrid5 telemetry)`.

Usage:
  python tools/audit/training_progress.py [--history PATH] [--bucket 100]
         [--gen-min G] [--gen-max G] [--last-gens K] [--json OUT]
         [--takeoff [--takeoff-goals 1.5] [--takeoff-window 25]] [--fire]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.audit._common import REPO_ROOT, fmt, load_json, write_json  # noqa: E402

__all__ = ["bucket_progress", "per_generation", "takeoff_summary", "takeoff_generation", "fire_progress", "main"]

FIRE_ROLES = ("blue_field", "blue_keeper", "red_field", "red_keeper")
# Team A (seats 0-5) is stored under `blue_*` and drawn red; team B is stored under `red_*` and drawn blue.
ROLE_LABELS = {
    "blue_field": "A/red field",
    "blue_keeper": "A/red keeper",
    "red_field": "B/blue field",
    "red_keeper": "B/blue keeper",
}
TAKEOFF_GOALS = 1.5
TAKEOFF_WINDOW = 25


def _shots(m: Dict[str, Any]) -> float:
    s = m.get("shots", 0)
    if isinstance(s, (list, tuple)):
        return float(sum(s))
    return float(s or 0)


def _loose(m: Dict[str, Any]) -> Optional[float]:
    tp = m.get("true_possession_pct")
    if isinstance(tp, (list, tuple)) and len(tp) >= 3:
        return float(tp[2])
    return None


def _is_pool(m: Dict[str, Any]) -> bool:
    """True for a match played with frozen opponent-pool seats (record `pool` set)."""
    return m.get("pool") is not None


_STAT_KEYS = ("interest_mean", "interest_median", "interest_max", "goals", "passes", "shots", "tackles", "saves",
              "loose_pct", "stationary_frames")


def _row(gen_start: int, gen_end: int, ms: List[Dict[str, Any]], n_pooled: int = 0) -> Dict[str, Any]:
    """Bucket statistics over the self-play matches `ms`; `n_pooled` pool matches of the bucket were left out."""
    if not ms:
        return {"gen_start": gen_start, "gen_end": gen_end, "n": 0, **{k: None for k in _STAT_KEYS}, "n_pooled": n_pooled}
    interest = np.array([float(m.get("interest_score", 0.0)) for m in ms])
    loose = [v for v in (_loose(m) for m in ms) if v is not None]
    return {
        "gen_start": gen_start,
        "gen_end": gen_end,
        "n": len(ms),
        "interest_mean": float(interest.mean()),
        "interest_median": float(np.median(interest)),
        "interest_max": float(interest.max()),
        "goals": float(np.mean([float(m.get("goals", 0)) for m in ms])),
        "passes": float(np.mean([float(m.get("completed_passes", 0)) for m in ms])),
        "shots": float(np.mean([_shots(m) for m in ms])),
        "tackles": float(np.mean([float(m.get("successful_tackles", 0)) for m in ms])),
        "saves": float(np.mean([float(m.get("keeper_saves", 0)) for m in ms])),
        "loose_pct": float(np.mean(loose)) if loose else None,
        "stationary_frames": float(np.mean([float(m.get("stationary_ball_frames", 0)) for m in ms])),
        "n_pooled": n_pooled,
    }


def _filtered(history: Dict[str, Any], gen_min: Optional[int], gen_max: Optional[int]) -> List[Dict[str, Any]]:
    out = []
    for m in history.get("matches") or []:
        g = int(m.get("generation", 0))
        if gen_min is not None and g < gen_min:
            continue
        if gen_max is not None and g > gen_max:
            continue
        out.append(m)
    return out


def bucket_progress(
    history: Dict[str, Any],
    bucket: int = 100,
    gen_min: Optional[int] = None,
    gen_max: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Group matches into consecutive generation buckets starting at the first generation present.

    Bucket bounds come from all matches; the statistics from the self-play ones (`n`), with the pool matches
    counted in `n_pooled`.
    """
    if bucket <= 0:
        raise ValueError("bucket must be a positive integer")
    ms = _filtered(history, gen_min, gen_max)
    if not ms:
        return []
    first = min(int(m.get("generation", 0)) for m in ms)
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for m in ms:
        key = (int(m.get("generation", 0)) - first) // bucket
        groups.setdefault(key, []).append(m)
    rows = []
    for key in sorted(groups):
        g = groups[key]
        own = [m for m in g if not _is_pool(m)]
        rows.append(_row(first + key * bucket, max(int(m.get("generation", 0)) for m in g), own, len(g) - len(own)))
    return rows


def per_generation(history: Dict[str, Any], last_gens: int) -> List[Dict[str, Any]]:
    """One row per generation for the newest `last_gens` generations."""
    ms = history.get("matches") or []
    if not ms or last_gens <= 0:
        return []
    top = max(int(m.get("generation", 0)) for m in ms)
    return bucket_progress(history, bucket=1, gen_min=top - last_gens + 1)


def takeoff_summary(
    history: Dict[str, Any],
    goals: float = TAKEOFF_GOALS,
    window: int = TAKEOFF_WINDOW,
) -> Dict[str, Any]:
    """Takeoff generation plus the parameters and generation span it was computed over.

    Returns `{"gen", "goals", "window", "first_gen", "last_gen"}` where `gen` is the smallest generation
    whose trailing `window`-generation mean of the per-generation mean self-play training-match `goals`
    is at least `goals` (None if no generation qualifies). Pool matches are left out. Raises ValueError for
    a history without matches, a match without a `generation`, a self-play match without a finite `goals`,
    a generation between the first and the last that has no match, or one that has only pool matches.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError(f"takeoff window must be a positive integer, got {window!r}")
    if not math.isfinite(goals):
        raise ValueError(f"takeoff goals threshold must be finite, got {goals!r}")
    matches = history.get("matches") or []
    if not matches:
        raise ValueError("history has no matches")
    per_gen: Dict[int, List[float]] = {}
    pool_gens = set()
    for m in matches:
        if m.get("generation") is None:
            raise ValueError("history match without a generation")
        g = int(m["generation"])
        if _is_pool(m):
            pool_gens.add(g)
            continue
        value = m.get("goals")
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"history match in generation {g} has no finite goals ({value!r})")
        per_gen.setdefault(g, []).append(float(value))
    if not per_gen:
        raise ValueError("history has no self-play match (every match is a pool match)")
    all_gens = set(per_gen) | pool_gens
    first, last = min(all_gens), max(all_gens)
    for g in range(first, last + 1):
        if g not in per_gen and g in pool_gens:
            raise ValueError(f"generation {g} has only pool matches (takeoff needs a self-play match in every generation)")
        if g not in per_gen:
            raise ValueError(
                f"generation {g} has no matches (every generation from {first} to {last} needs at least one)"
            )
    means = [math.fsum(per_gen[g]) / len(per_gen[g]) for g in range(first, last + 1)]
    gen: Optional[int] = None
    for i in range(window - 1, len(means)):
        if math.fsum(means[i - window + 1:i + 1]) / window >= goals:
            gen = first + i
            break
    return {"gen": gen, "goals": goals, "window": window, "first_gen": first, "last_gen": last}


def takeoff_generation(
    history: Dict[str, Any],
    goals: float = TAKEOFF_GOALS,
    window: int = TAKEOFF_WINDOW,
) -> Optional[int]:
    """Smallest generation whose trailing window mean of per-generation mean `goals` is >= `goals`, else None."""
    return takeoff_summary(history, goals, window)["gen"]


def _fire_record(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Fire telemetry of one generation record, None if it has none; ValueError if it is incomplete."""
    if "fire_sat" not in rec and "p_kick" not in rec and "p_spec" not in rec:
        return None
    gen = rec.get("generation")
    try:
        return {
            "fire_sat": {r: {"kick": float(rec["fire_sat"][r]["kick"]), "spec": float(rec["fire_sat"][r]["spec"])} for r in FIRE_ROLES},
            "p_kick": {r: float(rec["p_kick"][r]) for r in FIRE_ROLES},
            "p_spec": {r: float(rec["p_spec"][r]) for r in FIRE_ROLES},
        }
    except (KeyError, TypeError) as exc:
        raise ValueError(f"generation {gen} has incomplete fire telemetry ({exc!r})") from exc


def fire_progress(
    history: Dict[str, Any],
    bucket: int = 100,
    gen_min: Optional[int] = None,
    gen_max: Optional[int] = None,
    rows: Optional[List[Dict[str, Any]]] = None,
) -> List[Optional[Dict[str, Any]]]:
    """Mean fire telemetry per `bucket_progress` bucket (same order), None where a bucket has no records.

    `rows` is the result of `bucket_progress` with the same arguments, when the caller already has it.

    Each entry is `{"n", "fire_sat": {role: {"kick", "spec"}}, "p_kick": {role}, "p_spec": {role}}` with the
    mean over the bucket's generation records that carry fire telemetry.
    """
    if rows is None:
        rows = bucket_progress(history, bucket, gen_min, gen_max)
    records: List[tuple[int, Dict[str, Any]]] = []
    for rec in history.get("generations") or []:
        g = int(rec.get("generation", 0))
        if (gen_min is not None and g < gen_min) or (gen_max is not None and g > gen_max):
            continue
        fire = _fire_record(rec)
        if fire is not None:
            records.append((g, fire))
    out: List[Optional[Dict[str, Any]]] = []
    for row in rows:
        lo, hi = row["gen_start"], row["gen_start"] + bucket - 1
        group = [f for g, f in records if lo <= g <= hi]
        if not group:
            out.append(None)
            continue
        n = len(group)
        out.append({
            "n": n,
            "fire_sat": {
                r: {k: math.fsum(f["fire_sat"][r][k] for f in group) / n for k in ("kick", "spec")} for r in FIRE_ROLES
            },
            "p_kick": {r: math.fsum(f["p_kick"][r] for f in group) / n for r in FIRE_ROLES},
            "p_spec": {r: math.fsum(f["p_spec"][r] for f in group) / n for r in FIRE_ROLES},
        })
    return out


def _fire_lines(row: Dict[str, Any], fire: Optional[Dict[str, Any]]) -> List[str]:
    head = f"  fire gens {row['gen_start']:5d}-{row['gen_end']:5d}"
    if fire is None:
        return [f"{head}: n/a"]

    def fields(values: Dict[str, float]) -> str:
        return " ".join(f"{ROLE_LABELS[r]}={values[r]:.3f}" for r in FIRE_ROLES)

    return [
        f"{head} (n={fire['n']})",
        f"    {'fire_sat.kick':15s}{fields({r: fire['fire_sat'][r]['kick'] for r in FIRE_ROLES})}",
        f"    {'fire_sat.spec':15s}{fields({r: fire['fire_sat'][r]['spec'] for r in FIRE_ROLES})}",
        f"    {'p_kick':15s}{fields(fire['p_kick'])}",
        f"    {'p_spec':15s}{fields(fire['p_spec'])}",
    ]


def _line(r: Dict[str, Any]) -> str:
    pooled = f" pooled={r['n_pooled']}" if r["n_pooled"] else ""
    if r["n"] == 0:
        return f"  gens {r['gen_start']:5d}-{r['gen_end']:5d} n={r['n']:4d} (no self-play match){pooled}"
    return (
        f"  gens {r['gen_start']:5d}-{r['gen_end']:5d} n={r['n']:4d} interest={r['interest_mean']:5.1f} "
        f"(med {r['interest_median']:5.1f} max {r['interest_max']:5.1f}) goals={r['goals']:.2f} "
        f"passes={r['passes']:.1f} shots={r['shots']:.1f} tackles={r['tackles']:.1f} saves={r['saves']:.2f} "
        f"loose%={fmt(r['loose_pct'], 1)} stat={r['stationary_frames']:.0f}{pooled}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", default=str(REPO_ROOT / "checkpoints" / "training_history.json"))
    ap.add_argument("--bucket", type=int, default=100)
    ap.add_argument("--gen-min", type=int, default=None)
    ap.add_argument("--gen-max", type=int, default=None)
    ap.add_argument("--last-gens", type=int, default=0, help="also list the newest K generations individually")
    ap.add_argument("--max-rows", type=int, default=20)
    ap.add_argument("--json", default=None)
    ap.add_argument("--takeoff", action="store_true", help="also print the takeoff generation (whole history)")
    ap.add_argument("--takeoff-goals", type=float, default=TAKEOFF_GOALS, help="takeoff threshold on trailing mean goals")
    ap.add_argument("--takeoff-window", type=int, default=TAKEOFF_WINDOW, help="takeoff trailing window in generations")
    ap.add_argument("--fire", action="store_true", help="also print per-bucket mean fire telemetry (hybrid5 histories)")
    args = ap.parse_args(argv)

    hist = load_json(args.history)
    rows = bucket_progress(hist, args.bucket, args.gen_min, args.gen_max)
    last = per_generation(hist, args.last_gens)
    n_pooled = sum(r["n_pooled"] for r in rows)
    excluded = f" ({n_pooled} pool matches excluded)" if n_pooled else ""
    print(f"training_progress: {sum(r['n'] for r in rows)} matches in {len(rows)} bucket(s) of {args.bucket} gens{excluded}")
    for r in rows[-args.max_rows:]:
        print(_line(r))
    if last:
        print(f"  newest {len(last)} generation(s):")
        for r in last[-args.max_rows:]:
            print(_line(r))
    payload: Dict[str, Any] = {"buckets": rows, "last_generations": last}
    if args.takeoff:
        summary = takeoff_summary(hist, args.takeoff_goals, args.takeoff_window)
        if summary["gen"] is None:
            print(f"takeoff_gen=none (not reached by gen {summary['last_gen']})")
        else:
            print(f"takeoff_gen={summary['gen']} (trailing {args.takeoff_window}-gen mean training goals >= {args.takeoff_goals})")
        pool_total = sum(1 for m in hist.get("matches") or [] if _is_pool(m))
        if pool_total:
            print(f"takeoff: self-play matches only ({pool_total} pool matches excluded)")
        payload["takeoff"] = summary
    if args.fire:
        fires = fire_progress(hist, args.bucket, args.gen_min, args.gen_max, rows=rows)
        for row, fire in zip(rows, fires):
            row["fire"] = fire
        if all(f is None for f in fires):
            print("fire: n/a (no hybrid5 telemetry)")
        else:
            for row, fire in list(zip(rows, fires))[-args.max_rows:]:
                for line in _fire_lines(row, fire):
                    print(line)
    if args.json:
        write_json(args.json, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
