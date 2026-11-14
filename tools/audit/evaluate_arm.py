"""Evaluate snapshot checkpoints across the harness evaluation tags and write census outputs.

Runs `self_sampled` (n=96, seed 11), `self_mixed` (n=48, seed 12), `vs_scripted_sampled` (n=48,
seed 13), and `vs_frozen` (n=48, seed 15, requires `--frozen-dir`) via `tools/audit/harness.py`
and `tools/audit/census.py`, writing traces (`m*.npz`), `census.json`, and `report.txt` under
`OUT/<gen>/<tag>/`.

Usage:
  python tools/audit/evaluate_arm.py --snapshots DIR [DIR ...] --out OUT
         [--frozen-dir DIR] [--workers 10] [--tags ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.audit import census, harness  # noqa: E402

__all__ = ["TAG_SPECS", "evaluate_snapshot", "evaluate_arm", "main"]

TAG_SPECS: Dict[str, Dict[str, Any]] = {
    "self_sampled": {"opp": "self", "mode": "sampled", "n": 96, "seed": 11},
    "self_mixed": {"opp": "self", "mode": "mixed", "n": 48, "seed": 12},
    "vs_scripted_sampled": {"opp": "scripted", "mode": "sampled", "n": 48, "seed": 13},
    "vs_frozen": {"opp": "frozen", "mode": "sampled", "n": 48, "seed": 15},
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
            raise ValueError("tag 'vs_frozen' requires --frozen-dir / frozen_dir")
        opp = f"frozen:{frozen_dir}"
    else:
        opp = cfg["opp"]
    return harness.EvalSpec(
        checkpoint_dir=str(snapshot_dir),
        opponent=opp,
        mode=cfg["mode"],
        n=int(n_override) if n_override is not None else int(cfg["n"]),
        seed=int(cfg["seed"]),
        workers=int(workers),
        max_steps=max_steps,
    )


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
) -> Dict[str, Dict[str, Any]]:
    """Run all requested tags for one snapshot and write `OUT/<gen>/<tag>/{m*.npz,census.json}`."""
    snap = Path(snapshot_dir)
    frozen = Path(frozen_dir) if frozen_dir is not None else None
    gen = gen_name or snap.name
    if tags is None:
        selected = [t for t in TAG_SPECS if t != "vs_frozen" or frozen is not None]
    else:
        selected = list(tags)

    results: Dict[str, Dict[str, Any]] = {}
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
        per_match: List[Dict[str, Any]] = []
        for idx, trace, _ in harness.iter_matches(spec):
            harness.save_trace(trace, tag_dir, idx)
            per_match.append(census.match_metrics(trace))

        payload = {
            "gen": gen,
            "tag": tag,
            "mode": spec.mode,
            "n": spec.n,
            "seed": spec.seed,
            **census.census_payload(per_match),
        }
        (tag_dir / "census.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        rep = census.report(per_match, title=f"{gen}/{tag}")
        (tag_dir / "report.txt").write_text(rep + "\n", encoding="utf-8")
        print(rep)
        results[tag] = payload
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
        )
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshots", nargs="+", required=True, help="snapshot directories (e.g. .../gen1100 .../gen1200)")
    ap.add_argument("--out", required=True, help="output root receiving <gen>/<tag>/")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--frozen-dir", default=None, help="frozen checkpoint dir for vs_frozen tag")
    ap.add_argument("--tags", nargs="*", default=None, choices=tuple(TAG_SPECS), help="subset of tags to run")
    ap.add_argument("--n-override", type=int, default=None, help="override match count per tag (for quick checks)")
    ap.add_argument("--max-steps", type=int, default=None, help="cap steps per match")
    args = ap.parse_args(argv)
    evaluate_arm(
        args.snapshots,
        args.out,
        workers=args.workers,
        frozen_dir=args.frozen_dir,
        tags=args.tags,
        n_override=args.n_override,
        max_steps=args.max_steps,
    )
    return 0


if __name__ == "__main__":  # required: spawned workers re-import harness
    raise SystemExit(main())
