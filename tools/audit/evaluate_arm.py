"""Evaluate snapshot checkpoints across the harness evaluation tags and write census outputs.

Runs `self_sampled` (n=96, seed 11), `self_mixed` (n=48, seed 12), `vs_scripted_sampled` and
`vs_scripted_sampled_red` (n=48, seed 13), `vs_heldout_sampled` and `vs_heldout_sampled_red` (n=48,
seed 14; team B is the held-out passing bot), and `vs_frozen` and `vs_frozen_red` (n=48, seed 15,
requires `--frozen-dir`) via `tools/audit/harness.py`, `tools/audit/census.py`, and
`tools/audit/internals.py`, writing traces (`m*.npz`), `census.json`, `report.txt`, and
`internals.json` (for `self_sampled`) under `OUT/<gen>/<tag>/`.

With `--recensus`, recomputes `census.json`, `report.txt`, and (for `self_sampled`) `internals.json`
from existing `m*.npz` traces without running the harness.

Usage:
  python tools/audit/evaluate_arm.py --snapshots DIR [DIR ...] --out OUT
         [--frozen-dir DIR] [--workers 10] [--tags ...] [--recensus]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.audit import census, harness, internals  # noqa: E402

__all__ = ["TAG_SPECS", "evaluate_snapshot", "evaluate_arm", "main"]

TAG_SPECS: Dict[str, Dict[str, Any]] = {
    "self_sampled": {"opp": "self", "mode": "sampled", "n": 96, "seed": 11, "swap": False},
    "self_mixed": {"opp": "self", "mode": "mixed", "n": 48, "seed": 12, "swap": False},
    "vs_scripted_sampled": {"opp": "scripted", "mode": "sampled", "n": 48, "seed": 13, "swap": False},
    "vs_scripted_sampled_red": {"opp": "scripted", "mode": "sampled", "n": 48, "seed": 13, "swap": True},
    "vs_heldout_sampled": {"opp": "heldout", "mode": "sampled", "n": 48, "seed": 14, "swap": False},
    "vs_heldout_sampled_red": {"opp": "heldout", "mode": "sampled", "n": 48, "seed": 14, "swap": True},
    "vs_frozen": {"opp": "frozen", "mode": "sampled", "n": 48, "seed": 15, "swap": False},
    "vs_frozen_red": {"opp": "frozen", "mode": "sampled", "n": 48, "seed": 15, "swap": True},
}


def _build_spec(
    snapshot_dir: Path,
    tag: str,
    *,
    frozen_dir: Optional[Path] = None,
    workers: int = 10,
    n_override: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> harness.EvalSpec:
    if tag not in TAG_SPECS:
        raise ValueError(f"unknown tag {tag!r}; expected one of {tuple(TAG_SPECS)}")
    cfg = TAG_SPECS[tag]
    if cfg["opp"] == "frozen":
        if frozen_dir is None:
            raise ValueError(f"tag {tag!r} requires --frozen-dir / frozen_dir")
        opp = f"frozen:{frozen_dir}"
    else:
        opp = cfg["opp"]
    return harness.EvalSpec(
        checkpoint_dir=str(snapshot_dir),
        opponent=opp,
        mode=cfg["mode"],
        n=int(n_override) if n_override is not None else int(cfg["n"]),
        seed=int(cfg["seed"]),
        swap=bool(cfg.get("swap", False)),
        workers=int(workers),
        max_steps=max_steps,
    )


def _write_tag_outputs(
    tag_dir: Path,
    *,
    gen: str,
    tag: str,
    mode: str,
    n: int,
    seed: int,
    per_match: List[Dict[str, Any]],
    per_match_internals: List[Dict[str, Any]],
) -> Dict[str, Any]:
    payload = {
        "gen": gen,
        "tag": tag,
        "mode": mode,
        "n": n,
        "seed": seed,
        **census.census_payload(per_match),
    }
    (tag_dir / "census.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    rep = census.report(per_match, title=f"{gen}/{tag}")
    (tag_dir / "report.txt").write_text(rep + "\n", encoding="utf-8")
    print(rep)
    if tag == "self_sampled":
        int_payload = {
            "gen": gen,
            "tag": tag,
            "mode": mode,
            "n": n,
            "seed": seed,
            **internals.internals_payload(per_match_internals),
        }
        (tag_dir / "internals.json").write_text(
            json.dumps(int_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(internals.report(per_match_internals, title=f"{gen}/{tag}"))
    return payload


def evaluate_snapshot(
    snapshot_dir: Path | str,
    out_dir: Path | str,
    *,
    gen_name: Optional[str] = None,
    workers: int = 10,
    frozen_dir: Optional[Path | str] = None,
    tags: Optional[Sequence[str]] = None,
    n_override: Optional[int] = None,
    max_steps: Optional[int] = None,
    recensus: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Run all requested tags for one snapshot and write `OUT/<gen>/<tag>/{m*.npz,census.json}`."""
    snap = Path(snapshot_dir)
    frozen = Path(frozen_dir) if frozen_dir is not None else None
    gen = gen_name or snap.name
    if recensus:
        if tags is None:
            selected = [t for t in TAG_SPECS if (Path(out_dir) / gen / t).is_dir()]
            if not selected:
                raise FileNotFoundError(f"no tag directories found under {Path(out_dir) / gen}")
        else:
            selected = list(tags)
            for tag in selected:
                if tag not in TAG_SPECS:
                    raise ValueError(f"unknown tag {tag!r}; expected one of {tuple(TAG_SPECS)}")
    elif tags is None:
        selected = [t for t in TAG_SPECS if TAG_SPECS[t]["opp"] != "frozen" or frozen is not None]
    else:
        selected = list(tags)

    results: Dict[str, Dict[str, Any]] = {}
    if recensus:
        staged: List[tuple[str, Path, Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]] = []
        for tag in selected:
            cfg = TAG_SPECS[tag]
            tag_dir = harness.check_output_dir(Path(out_dir) / gen / tag, ())
            per_match = []
            per_match_internals = []
            for trace in census.iter_traces(tag_dir):
                per_match.append(census.match_metrics(trace))
                if tag == "self_sampled":
                    per_match_internals.append(internals.match_internals(trace))
            staged.append((tag, tag_dir, cfg, per_match, per_match_internals))
        for tag, tag_dir, cfg, per_match, per_match_internals in staged:
            results[tag] = _write_tag_outputs(
                tag_dir,
                gen=gen,
                tag=tag,
                mode=cfg["mode"],
                n=len(per_match),
                seed=int(cfg["seed"]),
                per_match=per_match,
                per_match_internals=per_match_internals,
            )
        return results

    for tag in selected:
        spec = _build_spec(
            snap,
            tag,
            frozen_dir=frozen,
            workers=workers,
            n_override=n_override,
            max_steps=max_steps,
        )
        tag_dir = harness.check_output_dir(Path(out_dir) / gen / tag, spec.policy_dirs())
        per_match = []
        per_match_internals = []
        for idx, trace, _ in harness.iter_matches(spec):
            harness.save_trace(trace, tag_dir, idx)
            per_match.append(census.match_metrics(trace))
            if tag == "self_sampled":
                per_match_internals.append(internals.match_internals(trace))

        results[tag] = _write_tag_outputs(
            tag_dir,
            gen=gen,
            tag=tag,
            mode=spec.mode,
            n=spec.n,
            seed=spec.seed,
            per_match=per_match,
            per_match_internals=per_match_internals,
        )
    return results


def evaluate_arm(
    snapshot_dirs: Sequence[Path | str],
    out_dir: Path | str,
    *,
    workers: int = 10,
    frozen_dir: Optional[Path | str] = None,
    tags: Optional[Sequence[str]] = None,
    n_override: Optional[int] = None,
    max_steps: Optional[int] = None,
    recensus: bool = False,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Evaluate a sequence of snapshot directories and return `{gen: {tag: payload}}`."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    seen: set[str] = set()
    for s in snapshot_dirs:
        gen = Path(s).name
        if gen in seen:
            raise ValueError(f"duplicate snapshot directory name {gen!r}")
        seen.add(gen)
        out[gen] = evaluate_snapshot(
            s,
            out_dir,
            gen_name=gen,
            workers=workers,
            frozen_dir=frozen_dir,
            tags=tags,
            n_override=n_override,
            max_steps=max_steps,
            recensus=recensus,
        )
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshots", nargs="+", required=True, help="snapshot directories (e.g. .../gen1100 .../gen1200)")
    ap.add_argument("--out", required=True, help="output root receiving <gen>/<tag>/")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--frozen-dir", default=None, help="frozen checkpoint dir for vs_frozen tags")
    ap.add_argument("--tags", nargs="*", default=None, choices=tuple(TAG_SPECS), help="subset of tags to run")
    ap.add_argument("--n-override", type=int, default=None, help="override match count per tag (for quick checks)")
    ap.add_argument("--max-steps", type=int, default=None, help="cap steps per match")
    ap.add_argument("--recensus", action="store_true", help="recompute census/internals from existing m*.npz traces")
    args = ap.parse_args(argv)
    evaluate_arm(
        args.snapshots,
        args.out,
        workers=args.workers,
        frozen_dir=args.frozen_dir,
        tags=args.tags,
        n_override=args.n_override,
        max_steps=args.max_steps,
        recensus=args.recensus,
    )
    return 0


if __name__ == "__main__":  # required: spawned workers re-import harness
    raise SystemExit(main())
