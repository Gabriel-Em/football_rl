"""Per-generation-bucket training progress from checkpoints/training_history.json (read-only).

Reports mean interest (with median / max), goals, completed passes, shots, tackles,
keeper saves, loose-ball % and stationary-ball frames per bucket of generations.

Usage:
  python tools/audit/training_progress.py [--history PATH] [--bucket 100]
         [--gen-min G] [--gen-max G] [--last-gens K] [--json OUT]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.audit._common import REPO_ROOT, fmt, load_json, write_json  # noqa: E402

__all__ = ["bucket_progress", "per_generation", "main"]


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


def _row(gen_start: int, gen_end: int, ms: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    """Group matches into consecutive generation buckets starting at the first generation present."""
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
        rows.append(_row(first + key * bucket, max(int(m.get("generation", 0)) for m in g), g))
    return rows


def per_generation(history: Dict[str, Any], last_gens: int) -> List[Dict[str, Any]]:
    """One row per generation for the newest `last_gens` generations."""
    ms = history.get("matches") or []
    if not ms or last_gens <= 0:
        return []
    top = max(int(m.get("generation", 0)) for m in ms)
    return bucket_progress(history, bucket=1, gen_min=top - last_gens + 1)


def _line(r: Dict[str, Any]) -> str:
    return (
        f"  gens {r['gen_start']:5d}-{r['gen_end']:5d} n={r['n']:4d} interest={r['interest_mean']:5.1f} "
        f"(med {r['interest_median']:5.1f} max {r['interest_max']:5.1f}) goals={r['goals']:.2f} "
        f"passes={r['passes']:.1f} shots={r['shots']:.1f} tackles={r['tackles']:.1f} saves={r['saves']:.2f} "
        f"loose%={fmt(r['loose_pct'], 1)} stat={r['stationary_frames']:.0f}"
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
    args = ap.parse_args(argv)

    hist = load_json(args.history)
    rows = bucket_progress(hist, args.bucket, args.gen_min, args.gen_max)
    last = per_generation(hist, args.last_gens)
    print(f"training_progress: {sum(r['n'] for r in rows)} matches in {len(rows)} bucket(s) of {args.bucket} gens")
    for r in rows[-args.max_rows:]:
        print(_line(r))
    if last:
        print(f"  newest {len(last)} generation(s):")
        for r in last[-args.max_rows:]:
            print(_line(r))
    if args.json:
        write_json(args.json, {"buckets": rows, "last_generations": last})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
