"""Gate evaluator for arm comparisons over census JSON directories.

Inputs map `{seed: {gen: {tag: census_dict}}}` for an arm and an optional comparator, where each
`census_dict` is the JSON output of `evaluate_arm.py` or `census.py`.

Supports `--protocol`:
  resume150  guards G1-G5 (consecutive-point rules) and G9'(b)/(c) at gen 1200
  fresh300   guards G1-G5 over the final points (superseded by `fresh`)
  fresh      requires `--calibration PATH`: C0 catastrophe floors (gens >= 200), C0L learning
             floor (T2 gain from gen 100 to the last generation), G1-G5 relative guards vs
             calibration anchor, G6 takeoff (with `--history`), and targets T1-T3 vs
             `--comparator` or anchor; T2 is the primary target

C0 floors (per arm seed, gens >= 200, strict; `gk_*` rows apply to each keeper):
  hold_share_pooled < 0.05      holder_spell_s_pooled < 0.15
  gk_in_box_A/B < 0.80          gk_goal_line_A/B > 0.60
  n_within_100_pooled > 1.00    cluster_ge4_pooled > 0.30
  role_inversion_pooled > 0.55  goals_total < 0.40 or > 5.0      scoreless > 0.70
C0L: T2(last) - T2(100) < 0.50 fails, where T2 is the mean `goal_diff` of `vs_frozen` and
`vs_frozen_red`; not evaluated unless generation 100 is in `--gens` and is not the last one.
G6: fails if more than half of the arm seeds have no takeoff (`training_progress.takeoff_generation`
with its default threshold and window, self-play matches only) in their `--history`; not evaluated
unless every arm seed has a history.

Computes `delta = arm - comparator`, `SE(delta) = sqrt(SE_a^2 + SE_c^2)`, `*` when
`|delta| > 2 * SE(delta)`, and between-seed SDs / ranges when >= 2 seeds are supplied.

Usage:
  python tools/audit/gates.py --arm [SEED=]DIR ... [--comparator [SEED=]DIR ...]
         [--tag self_sampled] [--gens 1100 1150 1200]
         [--protocol {resume150,fresh300,fresh}] [--calibration PATH]
         [--history SEED=PATH ...] [--exit-on-c0]
         [--target KEY@TAG:higher|lower:DELTA] [--json OUT] [--md OUT]

`--history`, `--calibration` and `--exit-on-c0` belong to `--protocol fresh`; with another protocol
they are an error. With `--exit-on-c0` the exit status is 2 when C0, C0L or G6 failed (a gate that
was not evaluated does not count) and 0 otherwise.
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

from tools.audit import census, training_progress  # noqa: E402

__all__ = [
    "PROTOCOLS",
    "LEARNING_FLOOR_GAIN",
    "TAKEOFF_FAIL_SHARE",
    "PRIMARY_TARGET",
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

PROTOCOLS = ("resume150", "fresh300", "fresh")
_GEN_RE = re.compile(r"(\d+)$")

CATASTROPHE_SPECS: List[Tuple[str, str, Callable[[float], bool]]] = [
    ("hold_share_pooled", "< 0.05", lambda v: v < 0.05),
    ("holder_spell_s_pooled", "< 0.15", lambda v: v < 0.15),
    ("gk_in_box_A", "< 0.80", lambda v: v < 0.80),
    ("gk_in_box_B", "< 0.80", lambda v: v < 0.80),
    ("gk_goal_line_A", "> 0.60", lambda v: v > 0.60),
    ("gk_goal_line_B", "> 0.60", lambda v: v > 0.60),
    ("n_within_100_pooled", "> 1.00", lambda v: v > 1.00),
    ("cluster_ge4_pooled", "> 0.30", lambda v: v > 0.30),
    ("role_inversion_pooled", "> 0.55", lambda v: v > 0.55),
    ("goals_total", "< 0.40 or > 5.0", lambda v: v < 0.40 or v > 5.0),
    ("scoreless", "> 0.70", lambda v: v > 0.70),
]

LEARNING_FLOOR_GAIN = 0.50
LEARNING_FLOOR_METRIC = "goal_diff"
LEARNING_FLOOR_FIRST_GEN = "100"
TAKEOFF_FAIL_SHARE = 0.5
PRIMARY_TARGET = "T2"
INFORMATIONAL_TARGETS: Dict[str, str] = {"T1": "informational at gen 300"}

FRESH_GUARD_GROUPS: List[Tuple[str, str, List[str]]] = [
    ("G1", "ball holding (relative)", ["hold_share_pooled", "holder_spell_s_pooled"]),
    ("G2", "keeper (relative)", ["gk_in_box_pooled", "gk_goal_line_pooled"]),
    ("G3", "clumping (relative)", ["n_within_100_pooled", "cluster_ge4_pooled"]),
    ("G4", "role order (relative)", ["role_inversion_pooled"]),
    ("G5", "scoring (relative)", ["goals_total", "scoreless"]),
]

FRESH_GUARD_WORSE: Dict[str, str] = {
    "hold_share_pooled": "lower",
    "holder_spell_s_pooled": "lower",
    "gk_in_box_pooled": "lower",
    "gk_goal_line_pooled": "higher",
    "n_within_100_pooled": "higher",
    "cluster_ge4_pooled": "higher",
    "role_inversion_pooled": "higher",
    "goals_total": "lower",
    "scoreless": "higher",
}

FRESH_TARGET_TAGS: Dict[str, Tuple[str, ...]] = {
    "T1": ("vs_scripted_sampled", "vs_scripted_sampled_red"),
    "T2": ("vs_frozen", "vs_frozen_red"),
    "T3": ("self_sampled",),
}

FRESH_TARGET_DIRECTION: Dict[str, str] = {"T1": "higher", "T2": "higher", "T3": "higher"}

FRESH_INFO_METRICS: List[str] = [
    "hold_share_A", "hold_share_B",
    "gk_in_box_A", "gk_in_box_B",
    "gk_goal_line_A", "gk_goal_line_B",
    "cluster_ge4_A", "cluster_ge4_B",
    "role_inversion_A", "role_inversion_B",
]


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


def _ebe_verdict(
    arm_per_seed: Dict[str, Optional[float]],
    comp_per_seed: Dict[str, Optional[float]],
    direction: str,
    delta: float,
) -> Dict[str, Any]:
    a_vals = [v for v in arm_per_seed.values() if v is not None]
    c_vals = [v for v in comp_per_seed.values() if v is not None]
    arm_mean = (sum(a_vals) / len(a_vals)) if a_vals else None
    comp_mean = (sum(c_vals) / len(c_vals)) if c_vals else None
    mean_diff = (arm_mean - comp_mean) if (arm_mean is not None and comp_mean is not None) else None

    note = ""
    if len(a_vals) < 2 or len(c_vals) < 2:
        verdict = "NEUTRAL"
        note = f"fewer than 2 seeds (arm={len(a_vals)}, comparator={len(c_vals)})"
    elif direction == "higher":
        if min(a_vals) > max(c_vals) and (arm_mean - comp_mean) >= delta:
            verdict = "BEATS"
        elif min(c_vals) > max(a_vals) and (comp_mean - arm_mean) >= delta:
            verdict = "LOSES"
        else:
            verdict = "NEUTRAL"
    else:
        if max(a_vals) < min(c_vals) and (comp_mean - arm_mean) >= delta:
            verdict = "BEATS"
        elif max(c_vals) < min(a_vals) and (arm_mean - comp_mean) >= delta:
            verdict = "LOSES"
        else:
            verdict = "NEUTRAL"

    return {
        "arm_per_seed": arm_per_seed,
        "comp_per_seed": comp_per_seed,
        "arm_mean": arm_mean,
        "comp_mean": comp_mean,
        "mean_diff": mean_diff,
        "verdict": verdict,
        "note": note,
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
    res = _ebe_verdict(a["per_seed"], c["per_seed"], spec.direction, spec.delta)
    return {
        "key": spec.key,
        "tag": spec.tag,
        "direction": spec.direction,
        "delta": spec.delta,
        "gen": gen,
        **res,
    }


def _require_seed_metric(
    norm_data: Dict[str, Dict[str, Dict[str, Any]]],
    seed: str,
    gen: str,
    tag: str,
    metric: str,
    *,
    role: str = "arm",
) -> float:
    gen_map = norm_data.get(seed, {}).get(gen)
    if not gen_map or tag not in gen_map:
        raise ValueError(f"{role} seed {seed!r} gen {gen!r} tag {tag!r} metric {metric!r} is missing")
    metrics = _extract_metrics_dict(gen_map[tag])
    entry = metrics.get(metric)
    if not isinstance(entry, dict) or entry.get("mean") is None:
        raise ValueError(f"{role} seed {seed!r} gen {gen!r} tag {tag!r} metric {metric!r} is missing or None")
    val = float(entry["mean"])
    if not math.isfinite(val):
        raise ValueError(f"{role} seed {seed!r} gen {gen!r} tag {tag!r} metric {metric!r} is non-finite ({val!r})")
    return val


def _check_calibration_entry(kind: str, name: str, entry: Dict[str, Any], min_seeds: int) -> None:
    """Reject an entry with fewer than `min_seeds` anchor seeds or a delta that is not finite, positive and >= its floor."""
    per_seed = entry.get("per_seed")
    n_seeds = len(per_seed) if isinstance(per_seed, dict) else 0
    if n_seeds < min_seeds:
        raise ValueError(f"calibration {kind} {name!r} has {n_seeds} anchor seeds; at least {min_seeds} are required")
    delta, floor = entry.get("delta"), entry.get("floor")
    for label, value in (("delta", delta), ("floor", floor)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"calibration {kind} {name!r} {label} must be a number, got {value!r}")
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError(f"calibration {kind} {name!r} delta must be finite and > 0, got {delta!r}")
    if delta < floor:
        raise ValueError(f"calibration {kind} {name!r} delta {delta!r} is below its floor {floor!r}")


def _validate_calibration(calibration: Dict[str, Any], tag: str) -> None:
    from tools.audit.calibrate import MIN_ANCHOR_SEEDS  # deferred: calibrate imports this module

    guards = calibration.get("guards")
    if not isinstance(guards, dict) or set(guards) != set(FRESH_GUARD_WORSE):
        got = sorted(guards) if isinstance(guards, dict) else type(guards).__name__
        raise ValueError(f"calibration guards mismatch: expected {sorted(FRESH_GUARD_WORSE)}, got {got}")
    for metric, expected_worse in FRESH_GUARD_WORSE.items():
        g_entry = guards[metric]
        actual_worse = g_entry.get("worse") if isinstance(g_entry, dict) else None
        if actual_worse != expected_worse:
            raise ValueError(
                f"calibration guard {metric!r} worse mismatch: expected {expected_worse!r}, got {actual_worse!r}"
            )
        if g_entry.get("tag") != tag:
            raise ValueError(
                f"calibration guard {metric!r} tag {g_entry.get('tag')!r} does not match the --tag in use {tag!r}"
            )
        _check_calibration_entry("guard", metric, g_entry, MIN_ANCHOR_SEEDS)
    targets = calibration.get("targets")
    if not isinstance(targets, dict) or set(targets) != set(FRESH_TARGET_TAGS):
        got_t = sorted(targets) if isinstance(targets, dict) else type(targets).__name__
        raise ValueError(f"calibration targets mismatch: expected {sorted(FRESH_TARGET_TAGS)}, got {got_t}")
    for tid, expected_tags in FRESH_TARGET_TAGS.items():
        t_entry = targets[tid]
        actual_tags = tuple(t_entry.get("tags", ())) if isinstance(t_entry, dict) else ()
        if actual_tags != expected_tags:
            raise ValueError(
                f"calibration target {tid!r} tags mismatch: expected {expected_tags!r}, got {actual_tags!r}"
            )
        if t_entry.get("direction") != FRESH_TARGET_DIRECTION[tid]:
            raise ValueError(
                f"calibration target {tid!r} direction mismatch: expected {FRESH_TARGET_DIRECTION[tid]!r}, "
                f"got {t_entry.get('direction')!r}"
            )
        _check_calibration_entry("target", tid, t_entry, MIN_ANCHOR_SEEDS)


def _multi_tag_per_seed(
    norm_data: Dict[str, Dict[str, Dict[str, Any]]],
    gen: str,
    tags: Sequence[str],
    metric: str,
    *,
    role: str = "arm",
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for seed in sorted(norm_data):
        vals = [_require_seed_metric(norm_data, seed, gen, t, metric, role=role) for t in tags]
        out[seed] = sum(vals) / len(vals)
    return out


def _evaluate_learning_floor(
    arm_norm: Dict[str, Dict[str, Dict[str, Any]]],
    eval_gens: List[str],
) -> Dict[str, Any]:
    """C0L: per arm seed, T2 at the last generation minus T2 at generation 100 must be >= `LEARNING_FLOOR_GAIN`."""
    last_gen = eval_gens[-1]
    out: Dict[str, Any] = {
        "description": "learning floor (T2 gain from gen 100 to the last gen)",
        "rule": f"T2(last) - T2({LEARNING_FLOOR_FIRST_GEN}) < {LEARNING_FLOOR_GAIN:.2f}",
        "passed": None,
        "per_seed": {},
        "note": "",
    }
    if LEARNING_FLOOR_FIRST_GEN not in eval_gens:
        out["note"] = f"not evaluated: gen {LEARNING_FLOOR_FIRST_GEN} not in --gens"
        return out
    if last_gen == LEARNING_FLOOR_FIRST_GEN:
        out["note"] = f"not evaluated: the last generation is {LEARNING_FLOOR_FIRST_GEN}"
        return out
    tags = FRESH_TARGET_TAGS["T2"]
    first = _multi_tag_per_seed(arm_norm, LEARNING_FLOOR_FIRST_GEN, tags, LEARNING_FLOOR_METRIC, role="arm")
    last = _multi_tag_per_seed(arm_norm, last_gen, tags, LEARNING_FLOOR_METRIC, role="arm")
    for seed in sorted(arm_norm):
        gain = last[seed] - first[seed]
        out["per_seed"][seed] = {
            "t_first": first[seed],
            "t_last": last[seed],
            "gain": gain,
            "passed": not (gain < LEARNING_FLOOR_GAIN),
        }
    out["passed"] = all(v["passed"] for v in out["per_seed"].values())
    return out


def _evaluate_takeoff_gate(
    arm_norm: Dict[str, Dict[str, Dict[str, Any]]],
    histories: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """G6: fails if more than `TAKEOFF_FAIL_SHARE` of the arm seeds have no takeoff in their training history."""
    seeds = sorted(arm_norm)
    out: Dict[str, Any] = {
        "description": "takeoff (share of seeds without takeoff)",
        "rule": f"FAIL if the share of seeds without takeoff > {TAKEOFF_FAIL_SHARE}",
        "passed": None,
        "per_seed": {},
        "history_gens": {},
        "no_takeoff_seeds": [],
        "no_takeoff_share": None,
        "note": "",
    }
    have = {str(k): v for k, v in (histories or {}).items()}
    lacking = [s for s in seeds if s not in have]
    if lacking:
        out["note"] = f"not evaluated: no training history for seed(s) {', '.join(lacking)}"
        return out
    for seed in seeds:
        info = training_progress.takeoff_summary(have[seed])
        out["per_seed"][seed] = info["gen"]
        out["history_gens"][seed] = [info["first_gen"], info["last_gen"]]
    out["no_takeoff_seeds"] = [s for s in seeds if out["per_seed"][s] is None]
    out["no_takeoff_share"] = len(out["no_takeoff_seeds"]) / len(seeds)
    out["passed"] = not (out["no_takeoff_share"] > TAKEOFF_FAIL_SHARE)
    return out


def _evaluate_fresh(
    arm_norm: Dict[str, Dict[str, Dict[str, Any]]],
    comp_norm: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
    tag: str,
    eval_gens: List[str],
    parsed_targets: List[TargetSpec],
    calibration: Optional[Dict[str, Any]],
    calibration_path: Optional[str],
    histories: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if calibration is None:
        raise ValueError("--calibration is required when protocol='fresh'")
    if not eval_gens:
        raise ValueError("at least one generation is required when protocol='fresh'")
    if not arm_norm:
        raise ValueError("at least one arm seed is required when protocol='fresh'")
    _validate_calibration(calibration, tag)
    extra = sorted(str(k) for k in (histories or {}) if str(k) not in arm_norm)
    if extra:
        raise ValueError(f"history seed(s) {', '.join(extra)} are not arm seeds {sorted(arm_norm)}")
    cal_gen = normalize_gen(calibration["gen"])
    last_gen = eval_gens[-1]
    if last_gen != cal_gen:
        raise ValueError(f"last generation {last_gen!r} does not equal calibration gen {cal_gen!r}")

    gates_out: Dict[str, Any] = {}

    # C0 catastrophe floors across gens >= 200
    c0_gens = [g for g in eval_gens if g.isdigit() and int(g) >= 200]
    violations: List[Dict[str, Any]] = []
    c0_has_any = False
    for g in c0_gens:
        for seed in sorted(arm_norm):
            for metric, rule_str, is_bad in CATASTROPHE_SPECS:
                val = _require_seed_metric(arm_norm, seed, g, tag, metric, role="arm")
                c0_has_any = True
                if is_bad(val):
                    violations.append({
                        "seed": seed,
                        "gen": g,
                        "metric": metric,
                        "rule": rule_str,
                        "value": val,
                    })
    gates_out["C0"] = {
        "description": "catastrophe floors (gens >= 200)",
        "gens_checked": c0_gens,
        "passed": None if not c0_has_any else (len(violations) == 0),
        "violations": violations,
    }
    gates_out["C0L"] = _evaluate_learning_floor(arm_norm, eval_gens)

    # G1-G5 relative guards at last_gen against anchor values in calibration
    cal_guards = calibration["guards"]
    for gid, desc, metric_list in FRESH_GUARD_GROUPS:
        passed_all: Optional[bool] = True
        metrics_out: Dict[str, Any] = {}
        for metric in metric_list:
            gcal = cal_guards[metric]
            worse = str(gcal["worse"])
            delta_req = float(gcal["delta"])
            anchor_per_seed: Dict[str, Optional[float]] = {
                str(s): float(v) for s, v in gcal["per_seed"].items()
            }
            anchor_vals = [v for v in anchor_per_seed.values() if v is not None]
            anchor_mean = sum(anchor_vals) / len(anchor_vals)
            anchor_sd = float(gcal["sd"]) if gcal.get("sd") is not None else None

            for seed in sorted(arm_norm):
                _require_seed_metric(arm_norm, seed, last_gen, tag, metric, role="arm")

            arm_agg = _aggregate_seeds(arm_norm, last_gen, tag, metric)
            arm_per_seed = arm_agg["per_seed"]
            arm_vals = [v for v in arm_per_seed.values() if v is not None]
            arm_mean = arm_agg["mean"]
            mean_diff = (arm_mean - anchor_mean) if arm_mean is not None else None

            if worse == "lower":
                separated = max(arm_vals) < min(anchor_vals)
                metric_pass = not (separated and (anchor_mean - arm_mean) >= delta_req)
            else:
                separated = min(arm_vals) > max(anchor_vals)
                metric_pass = not (separated and (arm_mean - anchor_mean) >= delta_req)

            if metric_pass is False:
                passed_all = False

            points = {g: _point_record(arm_norm, None, g, tag, metric) for g in eval_gens}
            points[last_gen]["comp_sd"] = anchor_sd
            points[last_gen]["comp_mean"] = anchor_mean
            points[last_gen]["comp_per_seed"] = anchor_per_seed

            metrics_out[metric] = {
                "passed": metric_pass,
                "worse": worse,
                "delta_req": delta_req,
                "arm_per_seed": arm_per_seed,
                "anchor_per_seed": anchor_per_seed,
                "arm_mean": arm_mean,
                "anchor_mean": anchor_mean,
                "mean_diff": mean_diff,
                "points": points,
            }
        gates_out[gid] = {
            "description": desc,
            "passed": passed_all,
            "metrics": metrics_out,
        }
    gates_out["G6"] = _evaluate_takeoff_gate(arm_norm, histories)

    # Targets T1-T3 against --comparator if given, else calibration anchor
    targets_out: List[Dict[str, Any]] = []
    for tid, tcal in calibration["targets"].items():
        t_key = str(tcal["key"])
        t_tags = list(tcal["tags"])
        t_dir = str(tcal["direction"])
        t_delta = float(tcal["delta"])
        arm_seed_vals = _multi_tag_per_seed(arm_norm, last_gen, t_tags, t_key, role="arm")
        if comp_norm is not None:
            comp_seed_vals = _multi_tag_per_seed(comp_norm, last_gen, t_tags, t_key, role="comparator")
        else:
            comp_seed_vals = {str(s): float(v) for s, v in tcal["per_seed"].items()}
        verdict = _ebe_verdict(arm_seed_vals, comp_seed_vals, t_dir, t_delta)
        targets_out.append({
            "id": tid,
            "key": t_key,
            "tag": "+".join(t_tags),
            "direction": t_dir,
            "delta": t_delta,
            "gen": last_gen,
            **verdict,
        })
    for spec in parsed_targets:
        targets_out.append(_evaluate_ebe_target(arm_norm, comp_norm, last_gen, spec))

    info_metrics = {
        metric: _aggregate_seeds(arm_norm, last_gen, tag, metric)
        for metric in FRESH_INFO_METRICS
    }

    return {
        "protocol": "fresh",
        "primary_target": PRIMARY_TARGET,
        "tag": tag,
        "gens": eval_gens,
        "calibration_path": calibration_path or "<in-memory>",
        "calibration_gen": cal_gen,
        "arm_seeds": sorted(arm_norm),
        "comparator_seeds": sorted(comp_norm) if comp_norm else list(calibration.get("anchor_seeds", [])),
        "gates": gates_out,
        "targets": targets_out,
        "info_metrics": info_metrics,
    }


def evaluate_gates(
    arm: Dict[str, Dict[str, Dict[str, Any]]],
    comparator: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    tag: str = "self_sampled",
    gens: Optional[Sequence[str]] = None,
    protocol: str = "resume150",
    targets: Optional[Sequence[str | TargetSpec]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    calibration_path: Optional[str] = None,
    histories: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Evaluate guard and target gates over `arm` (and optional `comparator`).

    `histories` maps arm seeds to loaded `training_history.json` dicts and feeds G6 (fresh protocol only).
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")
    if histories and protocol != "fresh":
        raise ValueError(f"training histories are only used with protocol='fresh', got {protocol!r}")
    parsed_targets = [parse_target_spec(t) for t in (targets or ())]
    arm_norm = _normalize_arm(arm)
    comp_norm = _normalize_arm(comparator) if comparator else None

    if gens is None:
        all_gens = {g for seed_map in arm_norm.values() for g in seed_map}
        eval_gens = sorted(all_gens, key=_gen_sort_key)
    else:
        eval_gens = [normalize_gen(g) for g in gens]

    if protocol == "fresh":
        return _evaluate_fresh(
            arm_norm,
            comp_norm,
            tag,
            eval_gens,
            parsed_targets,
            calibration,
            calibration_path,
            histories,
        )

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


def _append_between_seed_sd(lines: List[str], result: Dict[str, Any]) -> None:
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


def _target_label(t: Dict[str, Any]) -> str:
    if not t.get("id"):
        return f"{t['key']}@{t['tag']}"
    if t["id"] == PRIMARY_TARGET:
        return f"{t['id']} (primary)"
    if t["id"] in INFORMATIONAL_TARGETS:
        return f"{t['id']} ({INFORMATIONAL_TARGETS[t['id']]})"
    return f"{t['id']} ({t['key']}@{t['tag']})"


def _append_targets_table(lines: List[str], result: Dict[str, Any]) -> None:
    if result.get("targets"):
        lines.extend(["", "## Targets", "", "| Target | Gen | Dir | Delta min | Arm seeds | Comp seeds | Mean diff | Verdict | Note |", "|---|---|---|---|---|---|---|---|---|"])
        for t in result["targets"]:
            mdiff = "-" if t["mean_diff"] is None else f"{t['mean_diff']:+.3f}"
            label = _target_label(t)
            lines.append(
                f"| {label} | {t['gen'] or '-'} | {t['direction']} | {t['delta']:.3f} | "
                f"{_fmt_seeds(t['arm_per_seed'])} | {_fmt_seeds(t['comp_per_seed'])} | "
                f"{mdiff} | {t['verdict']} | {t['note']} |"
            )


def _format_fresh_markdown(result: Dict[str, Any]) -> str:
    cal_path = result.get("calibration_path", "<in-memory>")
    cal_gen = result.get("calibration_gen", "-")
    last_gen = result["gens"][-1] if result.get("gens") else "-"
    lines = [
        f"# Gate Report (tag={result['tag']}, gens={','.join(result['gens'])}, "
        f"protocol=fresh, calibration={cal_path}, calibration_gen={cal_gen})",
        "",
        "| Gate | Description | Status |",
        "|---|---|---|",
    ]
    for gid, g in result["gates"].items():
        lines.append(f"| {gid} | {g['description']} | {_status_str(g['passed'])} |")

    c0 = result["gates"]["C0"]
    lines.extend(["", "## C0 Catastrophe Check", ""])
    if c0["violations"]:
        lines.extend(["| Seed | Gen | Metric | Rule | Value |", "|---|---|---|---|---|"])
        for v in c0["violations"]:
            lines.append(f"| {v['seed']} | {v['gen']} | {v['metric']} | {v['rule']} | {v['value']:.4f} |")
    else:
        gens_str = ", ".join(c0["gens_checked"]) if c0["gens_checked"] else "none"
        lines.append(f"No catastrophe floor violations across gens [{gens_str}].")

    c0l = result["gates"]["C0L"]
    lines.extend(["", "## C0L Learning Floor", ""])
    if c0l["passed"] is None:
        lines.append(c0l["note"])
    else:
        lines.extend([
            f"FAIL if T2({last_gen}) - T2({LEARNING_FLOOR_FIRST_GEN}) < {LEARNING_FLOOR_GAIN:.2f}.",
            "",
            f"| Seed | T2 at {LEARNING_FLOOR_FIRST_GEN} | T2 at {last_gen} | Gain | Status |",
            "|---|---|---|---|---|",
        ])
        for seed, v in c0l["per_seed"].items():
            lines.append(
                f"| {seed} | {v['t_first']:.3f} | {v['t_last']:.3f} | {v['gain']:+.3f} | {_status_str(v['passed'])} |"
            )

    g6 = result["gates"]["G6"]
    lines.extend(["", "## Takeoff (G6)", ""])
    if g6["passed"] is None:
        lines.append(g6["note"])
    else:
        lines.extend(["| Seed | Takeoff gen | History gens |", "|---|---|---|"])
        for seed, gen in g6["per_seed"].items():
            first, last = g6["history_gens"][seed]
            lines.append(f"| {seed} | {'none' if gen is None else gen} | {first}-{last} |")
        lines.extend([
            "",
            f"Seeds without takeoff: {len(g6['no_takeoff_seeds'])} of {len(g6['per_seed'])} "
            f"(share {g6['no_takeoff_share']:.3f}; FAIL if > {TAKEOFF_FAIL_SHARE}).",
        ])

    lines.extend([
        "",
        f"## Relative Guards (G1-G5 at gen {last_gen})",
        "",
        "| Gate | Metric | Worse | Arm seeds | Anchor seeds | Mean diff | Delta | Status |",
        "|---|---|---|---|---|---|---|---|",
    ])
    for gid, g in result["gates"].items():
        if "metrics" not in g:
            continue
        for metric, mdata in g["metrics"].items():
            mdiff = "-" if mdata["mean_diff"] is None else f"{mdata['mean_diff']:+.4f}"
            lines.append(
                f"| {gid} | {metric} | {mdata['worse']} | "
                f"{_fmt_seeds(mdata['arm_per_seed'])} | {_fmt_seeds(mdata['anchor_per_seed'])} | "
                f"{mdiff} | {mdata['delta_req']:.4f} | {_status_str(mdata['passed'])} |"
            )

    _append_between_seed_sd(lines, result)
    _append_targets_table(lines, result)

    info = result.get("info_metrics") or {}
    if info:
        lines.extend([
            "",
            f"## Per-team Informational (gen {last_gen})",
            "",
            "| Metric | Arm mean | Per seed |",
            "|---|---|---|",
        ])
        for metric, agg in info.items():
            mval = "-" if agg["mean"] is None else f"{agg['mean']:.4f}"
            lines.append(f"| {metric} | {mval} | {_fmt_seeds(agg['per_seed'])} |")

    return "\n".join(lines)


def format_markdown(result: Dict[str, Any]) -> str:
    """Render gate evaluation `result` as Markdown summary + per-point tables."""
    proto = result.get("protocol", "resume150")
    if proto == "fresh":
        return _format_fresh_markdown(result)
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

    _append_between_seed_sd(lines, result)
    _append_targets_table(lines, result)

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


def _parse_history_args(items: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in items:
        seed, sep, path = item.partition("=")
        seed = seed.strip()
        if not sep or not seed or not path:
            raise ValueError(f"--history entries must match SEED=PATH, got {item!r}")
        if seed in out:
            raise ValueError(f"duplicate --history seed {seed!r}")
        out[seed] = json.loads(Path(path).read_text(encoding="utf-8"))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs="+", required=True, help="[SEED=]DIR entries from evaluate_arm.py")
    ap.add_argument("--comparator", nargs="*", default=None, help="optional [SEED=]DIR comparator entries")
    ap.add_argument("--tag", default="self_sampled")
    ap.add_argument("--gens", nargs="*", default=None, help="generations to evaluate (e.g. 1100 1150 1200)")
    ap.add_argument("--protocol", choices=PROTOCOLS, default="resume150", help="gate protocol")
    ap.add_argument("--calibration", default=None, help="calibration JSON path (required for --protocol fresh)")
    ap.add_argument("--history", nargs="+", default=None, help="SEED=PATH training_history.json per arm seed, for G6 (fresh only)")
    ap.add_argument("--exit-on-c0", action="store_true", help="exit 2 if C0, C0L or G6 failed (fresh only)")
    ap.add_argument("--target", action="append", default=None, help="EBE target KEY@TAG:higher|lower:DELTA (repeatable)")
    ap.add_argument("--json", default=None, help="write JSON gate report here")
    ap.add_argument("--md", default=None, help="write Markdown gate report here")
    args = ap.parse_args(argv)

    if args.protocol != "fresh":
        for flag, given in (("--calibration", args.calibration), ("--history", args.history), ("--exit-on-c0", args.exit_on_c0)):
            if given:
                raise ValueError(f"{flag} requires --protocol fresh, got {args.protocol!r}")

    load_tag = None if (args.target or args.protocol == "fresh") else args.tag
    arm_data = _parse_seed_dirs(args.arm, load_tag)
    comp_data = _parse_seed_dirs(args.comparator, load_tag) if args.comparator else None
    cal_data = json.loads(Path(args.calibration).read_text(encoding="utf-8")) if args.calibration else None
    history_data = _parse_history_args(args.history) if args.history else None
    result = evaluate_gates(
        arm_data,
        comparator=comp_data,
        tag=args.tag,
        gens=args.gens,
        protocol=args.protocol,
        targets=args.target,
        calibration=cal_data,
        calibration_path=args.calibration,
        histories=history_data,
    )
    md = format_markdown(result)
    print(md)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(md, encoding="utf-8")
    if args.exit_on_c0 and any(result["gates"][g]["passed"] is False for g in ("C0", "C0L", "G6")):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
