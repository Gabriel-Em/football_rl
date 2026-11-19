"""Paired readout across evaluation directories and training histories for long runs.

Resolves each `(run, gen)` to the first supplied `EVAL_DIR` containing `gen<g:04d>/`,
extracts target, style, census and internals metrics (mean +- SE), and averages
`fire_sat`, `p_kick`, `p_spec` and `v_corr_mc` per role over `[g - window + 1, g]`
from each run's `training_history.json`. Missing generations or files render as `-`. The history
table labels roles `A/red field`, `A/red keeper`, `B/blue field`, `B/blue keeper` (the stored
`blue_*` and `red_*` keys, team A drawn red); the JSON keeps the stored keys.

Usage:
  python -m tools.audit.run_compare --run NAME=EVAL_DIR[,EVAL_DIR...] [...]
         [--history NAME=PATH [...]] --gens 300 500 700 1000 [--window 100]
         [--md OUT] [--json OUT]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.audit import gates, training_progress  # noqa: E402
from tools.audit._common import load_json, write_json  # noqa: E402

__all__ = [
    "EVAL_METRIC_SPECS",
    "compare_runs",
    "format_markdown",
    "main",
]

# (display_name, tag_or_target_id, metric_key, is_pair, decimal_places)
EVAL_METRIC_SPECS: Tuple[Tuple[str, str, str, bool, int], ...] = (
    ("T1_blue", "vs_scripted_sampled", "goal_diff", False, 3),
    ("T1_red", "vs_scripted_sampled_red", "goal_diff", False, 3),
    ("T1", "T1", "goal_diff", True, 3),
    ("T2_blue", "vs_frozen", "goal_diff", False, 3),
    ("T2_red", "vs_frozen_red", "goal_diff", False, 3),
    ("T2", "T2", "goal_diff", True, 3),
    ("T3", "self_sampled", "kick_outcome_teammate", False, 4),
    ("S1", "self_sampled", "hold_share_pooled", False, 4),
    ("S2", "self_sampled", "aim_teammate_share_sampled", False, 4),
    ("S3", "self_sampled", "kick_outcome_opponent", False, 4),
    ("S4", "self_sampled", "goal_no_pass_10s_share", False, 4),
    ("role_inversion_pooled", "self_sampled", "role_inversion_pooled", False, 4),
    ("gk_in_box_pooled", "self_sampled", "gk_in_box_pooled", False, 4),
    ("gk_pickup_rate", "self_sampled", "gk_pickup_rate", False, 4),
    ("holder_spell_s_pooled", "self_sampled", "holder_spell_s_pooled", False, 4),
    ("goals_total", "self_sampled", "goals_total", False, 3),
    ("power_mean", "self_sampled", "power_mean", False, 4),
    ("full_power_share", "self_sampled", "full_power_share", False, 4),
    ("keeper_special_rate", "self_sampled", "keeper_special_rate", False, 4),
    ("special_rate", "self_sampled", "special_rate", False, 4),
    ("successful_tackles_pm", "self_sampled", "successful_tackles_pm", False, 3),
    ("failed_tackles_pm", "self_sampled", "failed_tackles_pm", False, 3),
)


def _single_metric(tag_map: Dict[str, Any], tag: str, key: str) -> Optional[Dict[str, Any]]:
    payload = tag_map.get(tag)
    if payload is None:
        return None
    entry = gates._extract_metrics_dict(payload).get(key)
    if not isinstance(entry, dict) or entry.get("mean") is None:
        return None
    mean = float(entry["mean"])
    if not math.isfinite(mean):
        return None
    se_raw = entry.get("se")
    se = float(se_raw) if se_raw is not None and math.isfinite(float(se_raw)) else None
    n = int(entry["n"]) if entry.get("n") is not None else None
    return {"mean": mean, "se": se, "n": n}


def _pair_metric(tag_map: Dict[str, Any], tags: Sequence[str], key: str) -> Optional[Dict[str, Any]]:
    parts = [_single_metric(tag_map, t, key) for t in tags]
    if any(p is None for p in parts):
        return None
    k = len(parts)
    mean = sum(p["mean"] for p in parts if p is not None) / k
    ses = [p["se"] for p in parts if p is not None and p["se"] is not None]
    se = (math.sqrt(sum(s ** 2 for s in ses)) / k) if len(ses) == k else None
    return {"mean": mean, "se": se, "n": None}


def _extract_eval_row(tag_map: Dict[str, Any]) -> Dict[str, Optional[Dict[str, Any]]]:
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for name, src, key, is_pair, _ in EVAL_METRIC_SPECS:
        if is_pair:
            out[name] = _pair_metric(tag_map, gates.FRESH_TARGET_TAGS[src], key)
        else:
            out[name] = _single_metric(tag_map, src, key)
    return out


def _v_corr_mc_window(history: Dict[str, Any], lo: int, hi: int) -> Tuple[Optional[Dict[str, float]], int]:
    roles = training_progress.FIRE_ROLES
    records: List[Dict[str, float]] = []
    for rec in history.get("generations") or []:
        g = int(rec.get("generation", 0))
        if g < lo or g > hi:
            continue
        vc = rec.get("v_corr_mc")
        if not isinstance(vc, dict):
            continue
        row: Dict[str, float] = {}
        ok = True
        for r in roles:
            val = vc.get(r)
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                ok = False
                break
            row[r] = float(val)
        if ok:
            records.append(row)
    if not records:
        return None, 0
    n = len(records)
    return {r: math.fsum(rec[r] for rec in records) / n for r in roles}, n


def _window_bounds(gen_int: int, window: int) -> Tuple[int, int]:
    """Clamp the trailing window `[g - window + 1, g]` to generation >= 1 (or `g` if `g < 1`)."""
    lo = min(gen_int, max(1, gen_int - window + 1))
    return lo, gen_int


def _history_window(history: Optional[Dict[str, Any]], gen_int: int, window: int) -> Optional[Dict[str, Any]]:
    if history is None:
        return None
    lo, hi = _window_bounds(gen_int, window)
    span = max(1, hi - lo + 1)
    fire = training_progress.fire_progress(
        history,
        bucket=span,
        gen_min=lo,
        gen_max=hi,
        rows=[{"gen_start": lo, "gen_end": hi}],
    )[0]
    v_corr_mc, v_n = _v_corr_mc_window(history, lo, hi)
    return {
        "gen_start": lo,
        "gen_end": hi,
        "fire": fire,
        "v_corr_mc": v_corr_mc,
        "v_corr_mc_n": v_n,
    }


def compare_runs(
    run_eval_dirs: Dict[str, Sequence[str | Path]],
    gens: Sequence[str | int],
    *,
    histories: Optional[Dict[str, str | Path]] = None,
    window: int = 100,
) -> Dict[str, Any]:
    """Collect paired eval metrics and trailing-window training history means across runs."""
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError(f"window must be a positive integer, got {window!r}")
    norm_gens = [gates.normalize_gen(g) for g in gens]
    if not norm_gens or any(not g.isdigit() for g in norm_gens):
        raise ValueError(f"gens must be non-empty numeric generations, got {gens!r}")

    hist_map = histories or {}
    unknown_hist = sorted(name for name in hist_map if name not in run_eval_dirs)
    if unknown_hist:
        raise ValueError(f"history run name(s) not in --run: {', '.join(unknown_hist)}")

    loaded_dirs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    loaded_hists: Dict[str, Optional[Dict[str, Any]]] = {}
    for name, path in hist_map.items():
        p = Path(path)
        if p.is_file():
            loaded_hists[name] = load_json(p)
        else:
            loaded_hists[name] = None
            print(f"warning: history file for {name!r} not found: {p}", file=sys.stderr)

    eval_out: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = {}
    eval_dirs_out: Dict[str, Dict[str, Optional[str]]] = {}
    hist_out: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = {}

    for run_name, dirs in run_eval_dirs.items():
        eval_out[run_name] = {}
        eval_dirs_out[run_name] = {}
        hist_out[run_name] = {}
        hist_obj = loaded_hists.get(run_name)
        for gen_str in norm_gens:
            gen_int = int(gen_str)
            subdir = f"gen{gen_int:04d}"
            chosen: Optional[Path] = None
            for d in dirs:
                dp = Path(d)
                if dp.is_dir() and (dp / subdir).is_dir():
                    chosen = dp
                    break
            if chosen is None:
                eval_dirs_out[run_name][gen_str] = None
                eval_out[run_name][gen_str] = None
            else:
                key = str(chosen)
                if key not in loaded_dirs:
                    loaded_dirs[key] = gates.load_arm_dir(chosen)
                tag_map = loaded_dirs[key].get(gen_str)
                eval_dirs_out[run_name][gen_str] = key
                eval_out[run_name][gen_str] = _extract_eval_row(tag_map) if tag_map else None
            hist_out[run_name][gen_str] = _history_window(hist_obj, gen_int, window)

    return {
        "runs": list(run_eval_dirs),
        "gens": norm_gens,
        "window": window,
        "eval_dirs": eval_dirs_out,
        "eval": eval_out,
        "history": hist_out,
    }


def _fmt_stat(entry: Optional[Dict[str, Any]], nd: int) -> str:
    if entry is None or entry.get("mean") is None:
        return "-"
    s = f"{entry['mean']:.{nd}f}"
    if entry.get("se") is not None:
        s += f" +- {entry['se']:.{nd}f}"
    return s


def _hist_value(h_entry: Optional[Dict[str, Any]], group: str, role: str) -> str:
    if h_entry is None:
        return "-"
    if group == "v_corr_mc":
        vc = h_entry.get("v_corr_mc")
        return "-" if not vc or role not in vc else f"{vc[role]:.3f}"
    fire = h_entry.get("fire")
    if not fire:
        return "-"
    if group == "fire_sat.kick":
        val = fire["fire_sat"][role]["kick"]
    elif group == "fire_sat.spec":
        val = fire["fire_sat"][role]["spec"]
    elif group == "p_kick":
        val = fire["p_kick"][role]
    else:
        val = fire["p_spec"][role]
    return f"{val:.3f}"


def format_markdown(result: Dict[str, Any]) -> str:
    """Render paired comparison `result` as Markdown tables."""
    runs: List[str] = result["runs"]
    gens: List[str] = result["gens"]
    window: int = result["window"]
    lines = [
        f"# Run Comparison (gens={','.join(gens)}, window={window})",
        "",
        "## Eval Metrics",
        "",
        f"| Metric | Gen | {' | '.join(runs)} |",
        "|---|---|" + "|".join(["---"] * len(runs)) + "|",
    ]
    for name, _, _, _, nd in EVAL_METRIC_SPECS:
        for g in gens:
            cells = []
            for r in runs:
                ev = result["eval"][r].get(g)
                cells.append(_fmt_stat(ev.get(name) if ev else None, nd))
            lines.append(f"| {name} | {g} | {' | '.join(cells)} |")

    lines.extend([
        "",
        f"## Training History (window={window})",
        "",
        f"| Metric | Role | Window | {' | '.join(runs)} |",
        "|---|---|---|" + "|".join(["---"] * len(runs)) + "|",
    ])
    for group in ("fire_sat.kick", "fire_sat.spec", "p_kick", "p_spec", "v_corr_mc"):
        for role in training_progress.FIRE_ROLES:
            for g in gens:
                gi = int(g)
                lo, hi = _window_bounds(gi, window)
                w_label = f"{lo}-{hi}"
                cells = [_hist_value(result["history"][r].get(g), group, role) for r in runs]
                lines.append(f"| {group} | {training_progress.ROLE_LABELS[role]} | {w_label} | {' | '.join(cells)} |")

    return "\n".join(lines)


def _parse_run_args(items: Sequence[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for item in items:
        name, sep, dirs_str = item.partition("=")
        name = name.strip()
        dirs = [d.strip() for d in dirs_str.split(",") if d.strip()]
        if not sep or not name or not dirs:
            raise ValueError(f"--run entries must match NAME=EVAL_DIR[,EVAL_DIR...], got {item!r}")
        if name in out:
            raise ValueError(f"duplicate --run name {name!r}")
        out[name] = dirs
    return out


def _parse_history_args(items: Optional[Sequence[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items or ():
        name, sep, path = item.partition("=")
        name = name.strip()
        path = path.strip()
        if not sep or not name or not path:
            raise ValueError(f"--history entries must match NAME=PATH, got {item!r}")
        if name in out:
            raise ValueError(f"duplicate --history name {name!r}")
        out[name] = path
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="extend", nargs="+", required=True, help="NAME=EVAL_DIR[,EVAL_DIR...]")
    ap.add_argument("--history", action="extend", nargs="+", default=None, help="NAME=PATH training_history.json")
    ap.add_argument("--gens", nargs="+", required=True, help="generations to report")
    ap.add_argument("--window", type=int, default=100, help="trailing generation window for history averages")
    ap.add_argument("--md", default=None, help="write Markdown report here")
    ap.add_argument("--json", default=None, help="write JSON report here")
    args = ap.parse_args(argv)

    run_dirs = _parse_run_args(args.run)
    hist_paths = _parse_history_args(args.history)
    result = compare_runs(run_dirs, args.gens, histories=hist_paths, window=args.window)
    md = format_markdown(result)
    print(md)
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(md, encoding="utf-8")
    if args.json:
        write_json(args.json, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
