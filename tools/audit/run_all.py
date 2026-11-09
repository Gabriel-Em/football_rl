"""Run every audit tool and write context/metrics/latest.{md,json}, with deltas vs a saved baseline.

Guard metrics (compared before/after every change):
  ball_hold_rate, gk_dist_goal_line, gk_pickups_per_match, role_inversion_rate,
  mean_cluster_size (+ GK wall-pinned / near-miss / facing-blocked / dives, presser trailing
  rate, longest hold spell, loose_share, interest_last_bucket, ms_per_step when available).

Usage:
  python tools/audit/run_all.py [--last 8] [--save-baseline] [--skip-perf] [--perf-steps 300]
         [--out-dir context/metrics] [--replays-dir replays] [--history PATH] [--source-dir football_rl]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.audit import orphan_scan, perf_profile, replay_metrics, training_progress  # noqa: E402
from tools.audit._common import REPO_ROOT, fmt, load_json, safe_mean, write_json  # noqa: E402

__all__ = ["guard_metrics", "compute_deltas", "run", "main"]

GUARD_ORDER = (
    "ball_hold_rate", "gk_dist_goal_line", "gk_pickups_per_match", "role_inversion_rate", "mean_cluster_size",
    "gk_wall_pinned_rate", "gk_near_miss_frames_per_match", "gk_facing_blocked_frames_per_match", "gk_dives_per_match",
    "presser_trailing_rate", "longest_hold_spell_s", "loose_share", "interest_last_bucket", "ms_per_step",
)


def _pair_mean(v: Any) -> Optional[float]:
    if not isinstance(v, list):
        return None
    return safe_mean([x for x in v if x is not None]) if any(x is not None for x in v) else None


def _pair_sum(v: Any) -> Optional[float]:
    if not isinstance(v, list) or not any(x is not None for x in v):
        return None
    return float(sum(x for x in v if x is not None))


def guard_metrics(report: Dict[str, Any]) -> Dict[str, Optional[float]]:
    s = ((report.get("replays") or {}).get("summary")) or {}
    hist = report.get("history") or {}
    buckets = hist.get("buckets") or []
    perf = report.get("perf") or {}
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
        "longest_hold_spell_s": s.get("longest_hold_spell_s"),
        "loose_share": s.get("loose_share"),
        "interest_last_bucket": buckets[-1]["interest_mean"] if buckets else None,
        "ms_per_step": perf.get("total_ms_per_step"),
    }


def compute_deltas(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Dict[str, Optional[float]]]:
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for k in sorted(set(before) | set(after)):
        b, a = before.get(k), after.get(k)
        delta = (float(a) - float(b)) if (a is not None and b is not None) else None
        out[k] = {"before": b, "after": a, "delta": delta}
    return out


def run(args: argparse.Namespace) -> Dict[str, Any]:
    errors: List[str] = []
    report: Dict[str, Any] = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "inputs": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "history": None, "replays": {"n_matches": 0, "summary": {"n_matches": 0}, "matches": []},
        "orphans": None, "perf": None, "guards": {}, "deltas": None, "errors": errors,
    }

    hist_path = Path(args.history)
    if hist_path.is_file():
        hist = load_json(hist_path)
        report["history"] = {
            "buckets": training_progress.bucket_progress(hist, args.bucket),
            "last_generations": training_progress.per_generation(hist, args.last_gens),
        }
    else:
        errors.append(f"history not found: {hist_path}")

    paths = replay_metrics.select_replays(args.replays_dir, last=args.last)
    if not paths:
        errors.append(f"no replays found in {args.replays_dir}")
    per_match = []
    for p in paths:
        m = replay_metrics.compute_match_metrics(load_json(p))
        m["path"] = str(p)
        per_match.append(m)
    report["replays"] = {"n_matches": len(per_match), "summary": replay_metrics.summarize(per_match),
                         "matches": per_match}

    try:
        report["orphans"] = orphan_scan.scan(args.source_dir)
    except FileNotFoundError as exc:
        errors.append(str(exc))

    if not args.skip_perf:
        report["perf"] = perf_profile.profile_env(steps=args.perf_steps, seed=args.seed, record=True)

    report["guards"] = guard_metrics(report)
    out_dir = Path(args.out_dir)
    base_path = out_dir / "baseline.json"
    if base_path.is_file():
        base = load_json(base_path)
        report["deltas"] = compute_deltas(base.get("guards") or {}, report["guards"])
        report["baseline_generated_at"] = base.get("generated_at")
    write_json(out_dir / "latest.json", report)
    if args.save_baseline:
        write_json(base_path, report)
    (out_dir / "latest.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(r: Dict[str, Any]) -> str:
    L = [f"# Audit metrics ({r['generated_at']})", ""]
    deltas = r.get("deltas") or {}
    L += ["## Guard metrics", "", "| metric | baseline | latest | delta |", "|---|---|---|---|"]
    for k in GUARD_ORDER:
        d = deltas.get(k) or {}
        L.append(f"| {k} | {fmt(d.get('before'))} | {fmt(r['guards'].get(k))} | {fmt(d.get('delta'))} |")
    if r.get("baseline_generated_at"):
        L.append(f"\nBaseline from {r['baseline_generated_at']}.")
    hist = r.get("history")
    if hist and hist["buckets"]:
        L += ["", "## Training progress (last buckets)", "",
              "| gens | n | interest | goals | passes | shots | tackles | saves | loose % | stationary |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for b in hist["buckets"][-6:]:
            L.append(f"| {b['gen_start']}-{b['gen_end']} | {b['n']} | {b['interest_mean']:.1f} | {b['goals']:.2f} | "
                     f"{b['passes']:.1f} | {b['shots']:.1f} | {b['tackles']:.1f} | {b['saves']:.2f} | "
                     f"{fmt(b['loose_pct'], 1)} | {b['stationary_frames']:.0f} |")
    s = r["replays"]["summary"]
    L += ["", f"## Replays ({r['replays']['n_matches']} matches, per-team [A red, B blue])", ""]
    for k in ("possession_share", "passes", "width_mean", "depth_mean", "near_ball_mean", "cluster_frame_rate",
              "inversion_rate", "inversion_rate_oop", "gk_dist_ball_mean", "gk_hold_mean_s", "presser_share"):
        if k in s:
            L.append(f"- {k}: {fmt(s[k], 3) if k != 'presser_share' else s[k]}")
    o = r.get("orphans")
    if o:
        L += ["", "## Orphan scan", ""]
        for k in ("unread_config_fields", "write_only_attributes", "unused_class_aliases",
                  "config_fields_read_only_in_tests"):
            L.append(f"- {k} ({len(o[k])}): {', '.join(o[k][:20])}")
        sim = [g for g in o["duplicate_value_groups"] if g["similarity"] > 0][:10]
        L.append("- name-similar duplicate values: " + "; ".join(
            f"{g['class']} {g['value']}: {', '.join(g['fields'])}" for g in sim))
    p = r.get("perf")
    if p:
        L += ["", f"## Step performance ({p['total_ms_per_step']:.3f} ms/step, record={p['record']})", "",
              "| component | calls | self ms/step | self % | numpy calls/step |", "|---|---|---|---|---|"]
        npc = p.get("numpy_calls_by_component") or {}
        for name, c in sorted(p["components"].items(), key=lambda kv: -kv[1]["self_ms"])[:10]:
            L.append(f"| {name} | {c['calls']} | {c['self_ms_per_step']:.4f} | {c['self_pct']:.1f} | "
                     f"{npc.get(name, 0.0):.1f} |")
    if r["errors"]:
        L += ["", "## Errors", ""] + [f"- {e}" for e in r["errors"]]
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "context" / "metrics"))
    ap.add_argument("--replays-dir", default=str(REPO_ROOT / "replays"))
    ap.add_argument("--history", default=str(REPO_ROOT / "checkpoints" / "training_history.json"))
    ap.add_argument("--source-dir", default=str(REPO_ROOT / "football_rl"))
    ap.add_argument("--last", type=int, default=8, help="newest N replays to analyse")
    ap.add_argument("--bucket", type=int, default=100)
    ap.add_argument("--last-gens", type=int, default=10)
    ap.add_argument("--skip-perf", action="store_true")
    ap.add_argument("--perf-steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-baseline", action="store_true", help="also store this run as baseline.json")
    args = ap.parse_args(argv)
    report = run(args)
    md = (Path(args.out_dir) / "latest.md").read_text(encoding="utf-8").splitlines()
    print("\n".join(md[:16]))
    print(f"... full report: {Path(args.out_dir) / 'latest.md'} / latest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
