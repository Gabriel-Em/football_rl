"""Calibrate relative gate and target thresholds from multi-seed anchor evaluations.

Reads the census JSON outputs of at least three anchor seeds at one target generation and computes
per-seed values, anchor mean, sample SD `s` (ddof=1), and `delta = max(2*s, floor)` for every
fresh-protocol guard and target metric.

Guard metrics (all read from `self_sampled`):
  Gate  Metric                 Worse   Floor
  G1    hold_share_pooled      lower   0.010
  G1    holder_spell_s_pooled  lower   0.020
  G2    gk_in_box_pooled       lower   0.010
  G2    gk_goal_line_pooled    higher  0.030
  G3    n_within_100_pooled    higher  0.030
  G3    cluster_ge4_pooled     higher  0.010
  G4    role_inversion_pooled  higher  0.030
  G5    goals_total            lower   0.30
  G5    scoreless              higher  0.05

Target metrics (two-tag targets average the two tag means per seed):
  ID    Key                    Tags                                              Dir     Floor
  T1    goal_diff              vs_scripted_sampled, vs_scripted_sampled_red        higher  2.0
  T2    goal_diff              vs_frozen, vs_frozen_red                            higher  0.5
  T3    kick_outcome_teammate  self_sampled                                      higher  0.03

Usage:
  python -m tools.audit.calibrate --anchor SEED=DIR [SEED=DIR ...] --gen 300 --json OUT [--md OUT]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.audit import gates  # noqa: E402
from tools.audit._common import REPO_ROOT  # noqa: E402

__all__ = [
    "MIN_ANCHOR_SEEDS",
    "GUARD_TAG",
    "GUARD_CALIBRATION_SPECS",
    "TARGET_CALIBRATION_SPECS",
    "current_git_commit",
    "calibrate_anchor",
    "calibrate_from_dirs",
    "format_markdown",
    "main",
]

MIN_ANCHOR_SEEDS = 3
GUARD_TAG = "self_sampled"

GUARD_CALIBRATION_SPECS: Tuple[Tuple[str, str, str, float], ...] = (
    ("G1", "hold_share_pooled", "lower", 0.010),
    ("G1", "holder_spell_s_pooled", "lower", 0.020),
    ("G2", "gk_in_box_pooled", "lower", 0.010),
    ("G2", "gk_goal_line_pooled", "higher", 0.030),
    ("G3", "n_within_100_pooled", "higher", 0.030),
    ("G3", "cluster_ge4_pooled", "higher", 0.010),
    ("G4", "role_inversion_pooled", "higher", 0.030),
    ("G5", "goals_total", "lower", 0.30),
    ("G5", "scoreless", "higher", 0.05),
)

TARGET_CALIBRATION_SPECS: Tuple[Tuple[str, str, Tuple[str, ...], str, float], ...] = (
    ("T1", "goal_diff", ("vs_scripted_sampled", "vs_scripted_sampled_red"), "higher", 2.0),
    ("T2", "goal_diff", ("vs_frozen", "vs_frozen_red"), "higher", 0.5),
    ("T3", "kick_outcome_teammate", ("self_sampled",), "higher", 0.03),
)


def _run_git(args: Sequence[str]) -> Optional[str]:
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def current_git_commit() -> str:
    """Short HEAD hash of the checkout, suffixed `-dirty` if modified, or `unknown`."""
    rev = _run_git(["rev-parse", "--short", "HEAD"])
    if rev is None or not rev.strip():
        return "unknown"
    status = _run_git(["status", "--porcelain"])
    if status is None:
        return "unknown"
    return rev.strip() + ("-dirty" if status.strip() else "")


def _extract_seed_metric(
    arm_norm: Mapping[str, Mapping[str, Mapping[str, Any]]],
    seed: str,
    gen: str,
    tag: str,
    metric: str,
) -> float:
    gen_map = arm_norm.get(seed, {}).get(gen)
    if not gen_map or tag not in gen_map:
        raise ValueError(f"anchor seed {seed!r} gen {gen!r} tag {tag!r} metric {metric!r} is missing")
    metrics = gates._extract_metrics_dict(gen_map[tag])
    entry = metrics.get(metric)
    if not isinstance(entry, Mapping) or entry.get("mean") is None:
        raise ValueError(f"anchor seed {seed!r} gen {gen!r} tag {tag!r} metric {metric!r} is missing or None")
    val = float(entry["mean"])
    if not math.isfinite(val):
        raise ValueError(f"anchor seed {seed!r} gen {gen!r} tag {tag!r} metric {metric!r} is non-finite ({val!r})")
    return val


def _summarize_series(per_seed: Dict[str, float], floor: float) -> Dict[str, Any]:
    vals = list(per_seed.values())
    k = len(vals)
    mean = sum(vals) / k
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (k - 1))
    fl = float(floor)
    return {
        "per_seed": per_seed,
        "mean": mean,
        "sd": sd,
        "floor": fl,
        "delta": max(2.0 * sd, fl),
    }


def _seed_sort_key(seed: str) -> Tuple[int, int, str]:
    s = str(seed).strip()
    return (0, int(s), s) if s.isdigit() else (1, 0, s)


def calibrate_anchor(
    anchor_data: Mapping[str, Mapping[str, Mapping[str, Any]]],
    gen: str | int,
    *,
    anchor_dirs: Optional[Mapping[str, str]] = None,
    git_commit: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute guard and target calibration statistics from loaded anchor census data."""
    if len(anchor_data) < MIN_ANCHOR_SEEDS:
        raise ValueError(f"calibration requires at least {MIN_ANCHOR_SEEDS} anchor seeds, got {len(anchor_data)}")
    gen_norm = gates.normalize_gen(gen)
    arm_norm = gates._normalize_arm(dict(anchor_data))  # type: ignore[arg-type]
    seeds = sorted(arm_norm, key=_seed_sort_key)
    commit = git_commit if git_commit is not None else current_git_commit()

    guards_out: Dict[str, Any] = {}
    for gate_id, metric, worse, floor in GUARD_CALIBRATION_SPECS:
        per_seed = {
            seed: _extract_seed_metric(arm_norm, seed, gen_norm, GUARD_TAG, metric)
            for seed in seeds
        }
        guards_out[metric] = {
            "gate": gate_id,
            "tag": GUARD_TAG,
            "worse": worse,
            **_summarize_series(per_seed, floor),
        }

    targets_out: Dict[str, Any] = {}
    for tid, key, tags, direction, floor in TARGET_CALIBRATION_SPECS:
        per_seed: Dict[str, float] = {}
        for seed in seeds:
            vals = [_extract_seed_metric(arm_norm, seed, gen_norm, tag, key) for tag in tags]
            per_seed[seed] = sum(vals) / len(vals)
        targets_out[tid] = {
            "id": tid,
            "key": key,
            "tags": list(tags),
            "direction": direction,
            **_summarize_series(per_seed, floor),
        }

    dirs_map = {str(s): str(anchor_dirs[s]) for s in seeds} if anchor_dirs else {s: "" for s in seeds}
    return {
        "gen": gen_norm,
        "git_commit": commit,
        "anchor_seeds": seeds,
        "anchor_dirs": dirs_map,
        "guards": guards_out,
        "targets": targets_out,
    }


def calibrate_from_dirs(
    anchor_dirs: Mapping[str, Path | str],
    gen: str | int,
    *,
    git_commit: Optional[str] = None,
) -> Dict[str, Any]:
    """Load anchor `evaluate_arm.py` output directories and compute calibration JSON."""
    if len(anchor_dirs) < MIN_ANCHOR_SEEDS:
        raise ValueError(f"calibration requires at least {MIN_ANCHOR_SEEDS} anchor seeds, got {len(anchor_dirs)}")
    loaded = {str(seed): gates.load_arm_dir(path) for seed, path in anchor_dirs.items()}
    dirs_str = {str(seed): str(path) for seed, path in anchor_dirs.items()}
    return calibrate_anchor(loaded, gen, anchor_dirs=dirs_str, git_commit=git_commit)


def _fmt_seed_map(per_seed: Mapping[str, float], nd: int = 3) -> str:
    return ", ".join(f"{s}={per_seed[s]:.{nd}f}" for s in sorted(per_seed, key=_seed_sort_key))


def format_markdown(cal: Mapping[str, Any]) -> str:
    """Render calibration payload as Markdown tables."""
    seeds_str = ", ".join(sorted(cal.get("anchor_seeds", []), key=_seed_sort_key))
    lines = [
        f"# Anchor Calibration (gen={cal['gen']}, seeds=[{seeds_str}], commit={cal.get('git_commit', 'unknown')})",
        "",
        "## Guards (`self_sampled`)",
        "",
        "| Gate | Metric | Worse | Anchor seeds | Mean | SD (s) | Floor | Delta |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for metric, g in cal["guards"].items():
        lines.append(
            f"| {g['gate']} | {metric} | {g['worse']} | {_fmt_seed_map(g['per_seed'])} | "
            f"{g['mean']:.4f} | {g['sd']:.4f} | {g['floor']:.3f} | {g['delta']:.4f} |"
        )
    lines.extend([
        "",
        "## Targets",
        "",
        "| Target | Key | Tags | Dir | Anchor seeds | Mean | SD (s) | Floor | Delta |",
        "|---|---|---|---|---|---|---|---|---|",
    ])
    for tid, t in cal["targets"].items():
        tags_str = "+".join(t["tags"])
        lines.append(
            f"| {tid} | {t['key']} | {tags_str} | {t['direction']} | {_fmt_seed_map(t['per_seed'])} | "
            f"{t['mean']:.4f} | {t['sd']:.4f} | {t['floor']:.3f} | {t['delta']:.4f} |"
        )
    return "\n".join(lines)


def _parse_anchor_args(items: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--anchor entries must match SEED=DIR, got {item!r}")
        seed, path = item.split("=", 1)
        seed = seed.strip()
        if not seed or not path:
            raise ValueError(f"--anchor entries must have non-empty SEED and DIR, got {item!r}")
        if seed in out:
            raise ValueError(f"duplicate anchor seed {seed!r}")
        out[seed] = path
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchor", nargs="+", required=True, help="SEED=DIR entries from evaluate_arm.py (min 3)")
    ap.add_argument("--gen", required=True, help="calibration generation (e.g. 300)")
    ap.add_argument("--json", required=True, help="write calibration JSON here")
    ap.add_argument("--md", default=None, help="write calibration Markdown here")
    args = ap.parse_args(argv)

    anchor_dirs = _parse_anchor_args(args.anchor)
    cal = calibrate_from_dirs(anchor_dirs, args.gen)
    md = format_markdown(cal)
    print(md)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(cal, indent=2, sort_keys=True), encoding="utf-8")
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(md + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
