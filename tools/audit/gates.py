"""Gate evaluator for arm comparisons over census JSON directories.

Inputs map `{seed: {gen: {tag: census_dict}}}` for an arm and an optional comparator, where each
`census_dict` is the JSON output of `evaluate_arm.py` or `census.py`. Evaluates gates G1-G5
(per-team and pooled, including two-consecutive-points rules) and G9'(b)/(c) at gen 1200,
computing `delta = arm - comparator`, `SE(delta) = sqrt(SE_a^2 + SE_c^2)`, `*` when
`|delta| > 2 * SE(delta)`, and between-seed ranges when >= 2 seeds are supplied.

Usage:
  python tools/audit/gates.py --arm [SEED=]DIR ... [--comparator [SEED=]DIR ...]
         [--tag self_sampled] [--gens 1100 1150 1200] [--json OUT] [--md OUT]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.audit import census  # noqa: E402

__all__ = [
    "PROTOCOLS",
    "TargetSpec",
    "parse_target_spec",
    "delta_stat",
    "check_consecutive_failures",
    "normalize_gen",
    "load_arm_dir",
    "evaluate_gates",
    "format_markdown",
    "main",
]

PROTOCOLS = ("resume150", "fresh300")
_GEN_RE = re.compile(r"(\d+)$")


@dataclass(frozen=True)
class TargetSpec:
    """One EBE target comparison: `KEY@TAG:higher|lower:DELTA`."""

    key: str
    tag: str
    direction: str
    delta: float


def parse_target_spec(spec: str | TargetSpec) -> TargetSpec:
    """Parse `KEY@TAG:higher|lower:DELTA` into a `TargetSpec`."""
    if isinstance(spec, TargetSpec):
        return spec
    if "@" not in spec:
        raise ValueError(f"target must match KEY@TAG:higher|lower:DELTA, got {spec!r}")
    key, rest = spec.split("@", 1)
    parts = rest.rsplit(":", 2)
    if len(parts) != 3:
        raise ValueError(f"target must match KEY@TAG:higher|lower:DELTA, got {spec!r}")
    tag, direction, delta_str = parts
    if not key or not tag:
        raise ValueError(f"target key and tag must be non-empty, got {spec!r}")
    if direction not in ("higher", "lower"):
        raise ValueError(f"target direction must be 'higher' or 'lower', got {direction!r}")
    delta = float(delta_str)
    if not math.isfinite(delta) or delta < 0.0:
        raise ValueError(f"target delta must be finite and non-negative, got {delta_str!r}")
    return TargetSpec(key=key, tag=tag, direction=direction, delta=delta)


def normalize_gen(gen: str | int) -> str:
    """Normalize generation identifiers like 'gen1100' or '1100' to '1100'."""
    s = str(gen).strip()
    m = _GEN_RE.search(s)
    return str(int(m.group(1))) if m else s


def _gen_sort_key(gen: str) -> Tuple[int, str]:
    n = normalize_gen(gen)
    return (int(n), gen) if n.isdigit() else (10 ** 9, gen)


def delta_stat(
    mean_a: Optional[float],
    se_a: Optional[float],
    mean_c: Optional[float],
    se_c: Optional[float],
) -> Dict[str, Any]:
    """Return `delta = mean_a - mean_c`, `se = sqrt(se_a^2 + se_c^2)`, and `|delta| > 2*se`."""
    if mean_a is None or mean_c is None:
        return {"delta": None, "se": None, "significant": False}
    delta = float(mean_a) - float(mean_c)
    if se_a is None or se_c is None:
        return {"delta": delta, "se": None, "significant": False}
    se = math.sqrt(float(se_a) ** 2 + float(se_c) ** 2)
    return {"delta": delta, "se": se, "significant": abs(delta) > 2.0 * se}


def check_consecutive_failures(flags: Sequence[bool], consecutive: int = 2) -> bool:
    """Return True if at least `consecutive` adjacent elements of `flags` are True."""
    if consecutive <= 1:
        return any(flags)
    run = 0
    for f in flags:
        run = run + 1 if f else 0
        if run >= consecutive:
            return True
    return False


def _extract_metrics_dict(obj: Any) -> Dict[str, Dict[str, Any]]:
    """Accept `{summary: {metrics: ...}}`, `{metrics: ...}`, or `{dir: {summary: ...}}`."""
    if not isinstance(obj, dict):
        raise ValueError(f"expected census JSON dict, got {type(obj).__name__}")
    if "metrics" in obj and isinstance(obj["metrics"], dict):
        return obj["metrics"]
    if "summary" in obj and isinstance(obj["summary"], dict):
        return obj["summary"]["metrics"]
    if len(obj) == 1:
        val = next(iter(obj.values()))
        if isinstance(val, list):
            return census.summarize(val)["metrics"]
        return _extract_metrics_dict(val)
    raise ValueError("unrecognized census JSON layout")


def load_arm_dir(root: Path | str, tag: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Load `{gen: {tag: census_dict}}` from an `evaluate_arm.py` output directory."""
    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"arm output directory not found: {base}")
    out: Dict[str, Dict[str, Any]] = {}
    for gen_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        gen_key = normalize_gen(gen_dir.name)
        tags_map: Dict[str, Any] = {}
        for tag_dir in sorted(p for p in gen_dir.iterdir() if p.is_dir()):
            if tag is not None and tag_dir.name != tag:
                continue
            cj = tag_dir / "census.json"
            if cj.is_file():
                payload = json.loads(cj.read_text(encoding="utf-8"))
                ij = tag_dir / "internals.json"
                if ij.is_file():
                    int_payload = json.loads(ij.read_text(encoding="utf-8"))
                    c_metrics = _extract_metrics_dict(payload)
                    for k, v in _extract_metrics_dict(int_payload).items():
                        c_metrics.setdefault(k, v)
                tags_map[tag_dir.name] = payload
        if tags_map:
            out[gen_key] = tags_map
    return out


def _normalize_arm(raw_arm: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    return {
        str(seed): {normalize_gen(gen): tags_map for gen, tags_map in gens_map.items()}
        for seed, gens_map in raw_arm.items()
    }


def _aggregate_seeds(
    arm_norm: Dict[str, Dict[str, Dict[str, Any]]],
    gen: str,
    tag: str,
    metric: str,
) -> Dict[str, Any]:
    means: List[float] = []
    ses: List[float] = []
    per_seed: Dict[str, Optional[float]] = {}
    for seed in sorted(arm_norm):
        tag_map = arm_norm[seed].get(gen) or {}
        if tag not in tag_map:
            continue
        metrics = _extract_metrics_dict(tag_map[tag])
        entry = metrics.get(metric)
        if not entry or entry.get("mean") is None:
            per_seed[seed] = None
            continue
        m = float(entry["mean"])
        per_seed[seed] = m
        means.append(m)
        if entry.get("se") is not None:
            ses.append(float(entry["se"]))
    if not means:
        return {"mean": None, "se": None, "sd": None, "range": None, "per_seed": per_seed}
    k = len(means)
    mean = sum(means) / k
    se = (math.sqrt(sum(s ** 2 for s in ses)) / k) if len(ses) == k else None
    sd = math.sqrt(sum((m - mean) ** 2 for m in means) / (k - 1)) if k >= 2 else None
    rng = [min(means), max(means)] if k >= 2 else None
    return {"mean": mean, "se": se, "sd": sd, "range": rng, "per_seed": per_seed}


def _seed_series(
    arm_norm: Dict[str, Dict[str, Dict[str, Any]]],
    seed: str,
    gens: Sequence[str],
    tag: str,
    metric: str,
) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for g in gens:
        tag_map = (arm_norm.get(seed) or {}).get(g) or {}
        if tag not in tag_map:
            out.append(None)
            continue
        entry = _extract_metrics_dict(tag_map[tag]).get(metric)
        out.append(float(entry["mean"]) if entry and entry.get("mean") is not None else None)
    return out


GateCheck = Tuple[str, Callable[[float], bool]]

GUARD_SPECS: List[Tuple[str, str, int, List[GateCheck]]] = [
    (
        "G1",
        "ball holding (guard)",
        1,
        [
            ("hold_share_pooled", lambda v: v >= 0.170),
            ("hold_share_A", lambda v: v >= 0.075),
            ("hold_share_B", lambda v: v >= 0.075),
            ("holder_spell_s_pooled", lambda v: v >= 0.23),
        ],
    ),
    (
        "G2",
        "keeper (guard)",
        1,
        [
            ("gk_in_box_A", lambda v: v >= 0.95),
            ("gk_in_box_B", lambda v: v >= 0.95),
            ("gk_dist_goal_A", lambda v: v >= 60.0),
            ("gk_dist_goal_B", lambda v: v >= 60.0),
            ("gk_goal_line_A", lambda v: v <= 0.30),
            ("gk_goal_line_B", lambda v: v <= 0.30),
        ],
    ),
    (
        "G3",
        "clumping (guard)",
        2,
        [
            ("n_within_100_pooled", lambda v: v <= 0.70),
            ("cluster_ge4_pooled", lambda v: v <= 0.10),
            ("cluster_ge4_A", lambda v: v <= 0.15),
            ("cluster_ge4_B", lambda v: v <= 0.15),
        ],
    ),
    (
        "G4",
        "role order (guard)",
        2,
        [
            ("role_inversion_pooled", lambda v: v <= 0.22),
            ("role_inversion_A", lambda v: v <= 0.30),
            ("role_inversion_B", lambda v: v <= 0.30),
        ],
    ),
    (
        "G5",
        "scoring band (guard)",
        2,
        [
            ("goals_total", lambda v: 1.5 <= v <= 3.0),
            ("scoreless", lambda v: v <= 0.25),
        ],
    ),
]


def _point_record(
    arm_norm: Dict[str, Dict[str, Dict[str, Any]]],
    comp_norm: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
    gen: str,
    tag: str,
    metric: str,
) -> Dict[str, Any]:
    a = _aggregate_seeds(arm_norm, gen, tag, metric)
    c = (
        _aggregate_seeds(comp_norm, gen, tag, metric)
        if comp_norm
        else {"mean": None, "se": None, "sd": None, "range": None, "per_seed": {}}
    )
    d = delta_stat(a["mean"], a["se"], c["mean"], c["se"])
    return {
        "arm_mean": a["mean"],
        "arm_se": a["se"],
        "arm_sd": a["sd"],
        "arm_range": a["range"],
        "arm_per_seed": a["per_seed"],
        "comp_mean": c["mean"],
        "comp_se": c["se"],
        "comp_sd": c["sd"],
        "comp_range": c["range"],
        "comp_per_seed": c["per_seed"],
        "delta": d["delta"],
        "se_delta": d["se"],
        "significant": d["significant"],
    }


def _evaluate_ebe_target(
    arm_norm: Dict[str, Dict[str, Dict[str, Any]]],
    comp_norm: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
    gen: Optional[str],
    spec: TargetSpec,
) -> Dict[str, Any]:
    if gen is None:
        return {
            "key": spec.key,
            "tag": spec.tag,
            "direction": spec.direction,
            "delta": spec.delta,
            "gen": None,
            "arm_per_seed": {},
            "comp_per_seed": {},
            "arm_mean": None,
            "comp_mean": None,
            "mean_diff": None,
            "verdict": "NEUTRAL",
            "note": "no generation available",
        }
    a = _aggregate_seeds(arm_norm, gen, spec.tag, spec.key)
    c = (
        _aggregate_seeds(comp_norm, gen, spec.tag, spec.key)
        if comp_norm
        else {"mean": None, "per_seed": {}}
    )
    a_vals = [v for v in a["per_seed"].values() if v is not None]
    c_vals = [v for v in c["per_seed"].values() if v is not None]
    arm_mean = a["mean"]
    comp_mean = c["mean"]
    mean_diff = (arm_mean - comp_mean) if (arm_mean is not None and comp_mean is not None) else None

    note = ""
    if len(a_vals) < 2 or len(c_vals) < 2:
        verdict = "NEUTRAL"
        note = f"fewer than 2 seeds (arm={len(a_vals)}, comparator={len(c_vals)})"
    elif spec.direction == "higher":
        if min(a_vals) > max(c_vals) and (arm_mean - comp_mean) >= spec.delta:
            verdict = "BEATS"
        elif min(c_vals) > max(a_vals) and (comp_mean - arm_mean) >= spec.delta:
            verdict = "LOSES"
        else:
            verdict = "NEUTRAL"
    else:
        if max(a_vals) < min(c_vals) and (comp_mean - arm_mean) >= spec.delta:
            verdict = "BEATS"
        elif max(c_vals) < min(a_vals) and (arm_mean - comp_mean) >= spec.delta:
            verdict = "LOSES"
        else:
            verdict = "NEUTRAL"

    return {
        "key": spec.key,
        "tag": spec.tag,
        "direction": spec.direction,
        "delta": spec.delta,
        "gen": gen,
        "arm_per_seed": a["per_seed"],
        "comp_per_seed": c["per_seed"],
        "arm_mean": arm_mean,
        "comp_mean": comp_mean,
        "mean_diff": mean_diff,
        "verdict": verdict,
        "note": note,
    }


def evaluate_gates(
    arm: Dict[str, Dict[str, Dict[str, Any]]],
    comparator: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    tag: str = "self_sampled",
    gens: Optional[Sequence[str]] = None,
    protocol: str = "resume150",
    targets: Optional[Sequence[str | TargetSpec]] = None,
) -> Dict[str, Any]:
    """Evaluate guard and target gates over `arm` (and optional `comparator`)."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")
    parsed_targets = [parse_target_spec(t) for t in (targets or ())]
    arm_norm = _normalize_arm(arm)
    comp_norm = _normalize_arm(comparator) if comparator else None

    if gens is None:
        all_gens = {g for seed_map in arm_norm.values() for g in seed_map}
        eval_gens = sorted(all_gens, key=_gen_sort_key)
    else:
        eval_gens = [normalize_gen(g) for g in gens]

    gates_out: Dict[str, Any] = {}
    for gate_id, desc, consec, checks in GUARD_SPECS:
        passed_all: Optional[bool] = True
        metrics_out: Dict[str, Any] = {}
        for metric, pred in checks:
            points = {g: _point_record(arm_norm, comp_norm, g, tag, metric) for g in eval_gens}
            metric_pass = True
            has_any = False
            for seed in sorted(arm_norm):
                vals = _seed_series(arm_norm, seed, eval_gens, tag, metric)
                window = vals if protocol == "resume150" else vals[-consec:]
                flags = [False if v is None else (not pred(v)) for v in window]
                if any(v is not None for v in window):
                    has_any = True
                if check_consecutive_failures(flags, consecutive=consec):
                    metric_pass = False
            if not has_any:
                metric_pass_val: Optional[bool] = None
            else:
                metric_pass_val = metric_pass
                if not metric_pass:
                    passed_all = False
            metrics_out[metric] = {"passed": metric_pass_val, "points": points}
        if all(m["passed"] is None for m in metrics_out.values()):
            passed_all = None
        gates_out[gate_id] = {"description": desc, "consecutive": consec, "passed": passed_all, "metrics": metrics_out}

    if protocol == "resume150":
        # G9'(b) and G9'(c) at generation 1200
        target_gen = "1200"
        kp_pt = _point_record(arm_norm, comp_norm, target_gen, tag, "kick_propensity_pooled")
        hs_pt = _point_record(arm_norm, comp_norm, target_gen, tag, "hold_share_pooled")

        g9b_pass: Optional[bool] = None
        if kp_pt["arm_mean"] is not None and kp_pt["comp_mean"] is not None:
            a_vals = [v for v in kp_pt["arm_per_seed"].values() if v is not None]
            c_vals = [v for v in kp_pt["comp_per_seed"].values() if v is not None]
            g9b_pass = bool(
                kp_pt["arm_mean"] <= kp_pt["comp_mean"] - 0.03
                and a_vals
                and c_vals
                and max(a_vals) < min(c_vals)
            )
        gates_out["G9_prime_b"] = {
            "description": "kick_propensity_pooled at 1200: arm <= comp - 0.03 and each arm seed < each comp seed",
            "passed": g9b_pass,
            "points": {target_gen: kp_pt},
        }

        g9c_pass: Optional[bool] = None
        if hs_pt["arm_mean"] is not None and hs_pt["comp_mean"] is not None:
            g9c_pass = bool(hs_pt["arm_mean"] >= hs_pt["comp_mean"])
        gates_out["G9_prime_c"] = {
            "description": "hold_share_pooled at 1200: arm >= comp",
            "passed": g9c_pass,
            "points": {target_gen: hs_pt},
        }

    out: Dict[str, Any] = {
        "protocol": protocol,
        "tag": tag,
        "gens": eval_gens,
        "arm_seeds": sorted(arm_norm),
        "comparator_seeds": sorted(comp_norm) if comp_norm else [],
        "gates": gates_out,
    }
    if parsed_targets:
        last_gen = eval_gens[-1] if eval_gens else None
        out["targets"] = [_evaluate_ebe_target(arm_norm, comp_norm, last_gen, spec) for spec in parsed_targets]
    return out


def _fmt_val(mean: Optional[float], se: Optional[float], rng: Optional[Sequence[float]], nd: int = 3) -> str:
    if mean is None:
        return "-"
    s = f"{mean:.{nd}f}" + (f" +- {se:.{nd}f}" if se is not None else "")
    if rng is not None:
        s += f" [{rng[0]:.{nd}f}..{rng[1]:.{nd}f}]"
    return s


def _fmt_delta(pt: Dict[str, Any], nd: int = 3) -> str:
    d, se = pt.get("delta"), pt.get("se_delta")
    if d is None:
        return "-"
    star = " *" if pt.get("significant") else ""
    se_str = f" +- {se:.{nd + 1}f}" if se is not None else ""
    return f"{d:+.{nd}f}{se_str}{star}"


def _status_str(passed: Optional[bool]) -> str:
    if passed is None:
        return "N/A"
    return "PASS" if passed else "FAIL"


def _fmt_seeds(per_seed: Dict[str, Optional[float]], nd: int = 3) -> str:
    if not per_seed:
        return "-"
    parts = [f"{s}={('-' if v is None else f'{v:.{nd}f}')}" for s, v in sorted(per_seed.items())]
    return ", ".join(parts)


def format_markdown(result: Dict[str, Any]) -> str:
    """Render gate evaluation `result` as Markdown summary + per-point tables."""
    proto = result.get("protocol", "resume150")
    proto_suffix = f", protocol={proto}" if proto != "resume150" else ""
    lines = [f"# Gate Report (tag={result['tag']}, gens={','.join(result['gens'])}{proto_suffix})", ""]
    lines.append("| Gate | Description | Status |")
    lines.append("|---|---|---|")
    for gid, g in result["gates"].items():
        lines.append(f"| {gid} | {g['description']} | {_status_str(g['passed'])} |")
    lines.append("")
    lines.append("| Gate | Metric | Gen | Arm | Comparator | Delta |")
    lines.append("|---|---|---|---|---|---|")
    for gid, g in result["gates"].items():
        if "metrics" in g:
            for metric, mdata in g["metrics"].items():
                for gen, pt in mdata["points"].items():
                    lines.append(
                        f"| {gid} ({_status_str(mdata['passed'])}) | {metric} | {gen} | "
                        f"{_fmt_val(pt['arm_mean'], pt['arm_se'], pt['arm_range'])} | "
                        f"{_fmt_val(pt['comp_mean'], pt['comp_se'], pt['comp_range'])} | "
                        f"{_fmt_delta(pt)} |"
                    )
        else:
            for gen, pt in g["points"].items():
                lines.append(
                    f"| {gid} ({_status_str(g['passed'])}) | {gid} | {gen} | "
                    f"{_fmt_val(pt['arm_mean'], pt['arm_se'], pt['arm_range'])} | "
                    f"{_fmt_val(pt['comp_mean'], pt['comp_se'], pt['comp_range'])} | "
                    f"{_fmt_delta(pt)} |"
                )

    if len(result.get("arm_seeds", [])) >= 2 and result.get("gens"):
        last_gen = result["gens"][-1]
        lines.extend(["", f"## Between-seed SD at gen {last_gen}", "", "| Gate | Metric | Arm SD | Comp SD |", "|---|---|---|---|"])
        for gid, g in result["gates"].items():
            if "metrics" not in g:
                continue
            for metric, mdata in g["metrics"].items():
                pt = mdata["points"].get(last_gen, {})
                asd = "-" if pt.get("arm_sd") is None else f"{pt['arm_sd']:.4f}"
                csd = "-" if pt.get("comp_sd") is None else f"{pt['comp_sd']:.4f}"
                lines.append(f"| {gid} | {metric} | {asd} | {csd} |")

    if result.get("targets"):
        lines.extend(["", "## Targets", "", "| Target | Gen | Dir | Delta min | Arm seeds | Comp seeds | Mean diff | Verdict | Note |", "|---|---|---|---|---|---|---|---|---|"])
        for t in result["targets"]:
            mdiff = "-" if t["mean_diff"] is None else f"{t['mean_diff']:+.3f}"
            lines.append(
                f"| {t['key']}@{t['tag']} | {t['gen'] or '-'} | {t['direction']} | {t['delta']:.3f} | "
                f"{_fmt_seeds(t['arm_per_seed'])} | {_fmt_seeds(t['comp_per_seed'])} | "
                f"{mdiff} | {t['verdict']} | {t['note']} |"
            )

    return "\n".join(lines)


def _parse_seed_dirs(items: Sequence[str], tag: Optional[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for idx, item in enumerate(items):
        if "=" in item:
            seed, path = item.split("=", 1)
        else:
            seed, path = f"s{idx}", item
        out[seed] = load_arm_dir(path, tag=tag)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs="+", required=True, help="[SEED=]DIR entries from evaluate_arm.py")
    ap.add_argument("--comparator", nargs="*", default=None, help="optional [SEED=]DIR comparator entries")
    ap.add_argument("--tag", default="self_sampled")
    ap.add_argument("--gens", nargs="*", default=None, help="generations to evaluate (e.g. 1100 1150 1200)")
    ap.add_argument("--protocol", choices=PROTOCOLS, default="resume150", help="gate protocol")
    ap.add_argument("--target", action="append", default=None, help="EBE target KEY@TAG:higher|lower:DELTA (repeatable)")
    ap.add_argument("--json", default=None, help="write JSON gate report here")
    ap.add_argument("--md", default=None, help="write Markdown gate report here")
    args = ap.parse_args(argv)

    load_tag = None if args.target else args.tag
    arm_data = _parse_seed_dirs(args.arm, load_tag)
    comp_data = _parse_seed_dirs(args.comparator, load_tag) if args.comparator else None
    result = evaluate_gates(
        arm_data,
        comparator=comp_data,
        tag=args.tag,
        gens=args.gens,
        protocol=args.protocol,
        targets=args.target,
    )
    md = format_markdown(result)
    print(md)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
