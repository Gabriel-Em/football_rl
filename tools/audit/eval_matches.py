"""Fresh, unselected evaluation matches from a checkpoint directory (checkpoints are never written).

Training replays are biased samples (`--save-best-only` keeps the best of 8; unselected runs differ in
volume), so behavioural guards compared across them are confounded. This tool instead loads a checkpoint,
plays N brand-new matches (8 parallel environments, one full match each per rollout), exports *every*
replay to a scratch directory, and reports per-match metrics plus guard mean / standard error / n.

Nothing is written to the checkpoint directory, `./checkpoints` or `./replays`; the scratch replay
directory is deleted afterwards unless `--keep-replays` is given.

Usage:
  python tools/audit/eval_matches.py --checkpoint-dir DIR [--code-root PATH] [--matches 32] [--seed 1234]
         [--out JSON] [--keep-replays DIR]

`--code-root PATH` evaluates the checkpoint under another source tree (a directory that contains a
`football_rl` package, e.g. from `git archive <commit> football_rl | tar -x -C PATH`); it is prepended to
`sys.path` before `football_rl` is imported. Metrics are always computed by the current audit code.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.audit import replay_metrics  # noqa: E402
from tools.audit._common import fmt, load_json, select_replays, write_json  # noqa: E402

__all__ = ["run_eval", "guard_stats", "match_guard_values", "main"]

ENVS_PER_ROLLOUT = 8  # one full match per environment per rollout
DEFAULT_MATCHES = 32
DEFAULT_SEED = 1234


def guard_stats(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    """Mean, standard error of the mean (sample sd / sqrt(n)) and sample count, skipping None values."""
    vals = [float(v) for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return {"mean": None, "se": None, "n": 0}
    mean = sum(vals) / n
    if n == 1:
        return {"mean": mean, "se": None, "n": 1}
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return {"mean": mean, "se": math.sqrt(var / n), "n": n}


def match_guard_values(match_metrics: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Replay guard scalars of one match, defined exactly as `run_all` defines them on a summary."""
    return replay_metrics.replay_guards(replay_metrics.summarize([match_metrics]))


def _prepend_code_root(code_root: str) -> Path:
    root = Path(code_root).resolve()
    if not (root / "football_rl").is_dir():
        raise FileNotFoundError(f"--code-root {root} has no football_rl package")
    if not sys.path or sys.path[0] != str(root):
        sys.path.insert(0, str(root))
    return root


def run_eval(
    checkpoint_dir: str,
    matches: int = DEFAULT_MATCHES,
    seed: int = DEFAULT_SEED,
    keep_replays: Optional[str] = None,
    code_root: Optional[str] = None,
    config: Any = None,
    rollout_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Play `matches` fresh matches from `checkpoint_dir` and return per-match metrics and guard statistics.

    `config` (default: `football_rl.config.DEFAULT_CONFIG`) supplies everything except the replay settings,
    which are forced to export every match to the scratch directory. `rollout_steps` defaults to one full
    match (`physics.max_rl_steps`).
    """
    if matches <= 0 or matches % ENVS_PER_ROLLOUT != 0:
        raise ValueError(f"--matches must be a positive multiple of {ENVS_PER_ROLLOUT}, got {matches}")
    ckpt = Path(checkpoint_dir)
    if not ckpt.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {ckpt}")

    root = _prepend_code_root(code_root) if code_root else None
    import numpy as np
    import torch

    import football_rl
    from football_rl.config import DEFAULT_CONFIG
    from football_rl.train_ppo import ParameterSharingPPOTrainer

    package_dir = Path(football_rl.__file__).resolve().parent
    if root is not None and root not in package_dir.parents:
        raise RuntimeError(f"football_rl was already imported from {package_dir}, not from --code-root {root}")

    if keep_replays:
        replay_dir = Path(keep_replays)
        replay_dir.mkdir(parents=True, exist_ok=True)
        if select_replays(replay_dir):
            raise ValueError(f"--keep-replays directory already holds replays: {replay_dir}")
    else:
        replay_dir = Path(tempfile.mkdtemp(prefix="eval_matches_"))

    trainer = None
    try:
        base = config if config is not None else DEFAULT_CONFIG
        cfg = replace(
            base,
            replay=replace(
                base.replay,
                save_dir=str(replay_dir),
                save_best_only=False,
                enable_recording=True,
                interest_threshold=0.0,
            ),
        )
        steps = int(rollout_steps) if rollout_steps else int(cfg.physics.max_rl_steps)
        trainer = ParameterSharingPPOTrainer(
            config=cfg,
            num_envs=ENVS_PER_ROLLOUT,
            rollout_steps=steps,
            device="cpu",
            checkpoint_dir=str(ckpt),
            seed=seed,
            save_best_only=False,
        )
        if not trainer.load_checkpoints():
            raise RuntimeError(f"no loadable checkpoint in {ckpt}")
        generation = int(trainer.current_generation)
        torch.manual_seed(seed)
        np.random.seed(seed)
        for _ in range(matches // ENVS_PER_ROLLOUT):
            trainer.collect_rollouts(active_generation=generation + 1)

        paths = select_replays(replay_dir)
        if len(paths) != matches:
            raise RuntimeError(f"expected {matches} exported replays in {replay_dir}, found {len(paths)}")
        per_match = [replay_metrics.compute_match_metrics(load_json(p)) for p in paths]
    finally:
        if trainer is not None:
            trainer.close()
        if not keep_replays:
            shutil.rmtree(replay_dir, ignore_errors=True)

    values = [match_guard_values(m) for m in per_match]
    guards = {name: guard_stats([v[name] for v in values]) for name in replay_metrics.REPLAY_GUARD_KEYS}
    return {
        "per_match": per_match,
        "guards": guards,
        "meta": {
            "checkpoint_dir": str(ckpt),
            "checkpoint_generation": generation,
            "matches": matches,
            "seed": seed,
            "football_rl": str(package_dir),
        },
    }


def _summary_lines(result: Dict[str, Any]) -> list[str]:
    meta = result.get("meta") or {}
    lines = [f"eval_matches: {len(result['per_match'])} matches from gen {meta.get('checkpoint_generation')} "
             f"(seed {meta.get('seed')}, code {meta.get('football_rl')})"]
    for name, g in (result.get("guards") or {}).items():
        lines.append(f"  {name:36s} mean {fmt(g['mean'], 4):>9s}  se {fmt(g['se'], 4):>8s}  n {g['n']}")
    return lines


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", required=True, help="directory holding training_state.pt / policies")
    ap.add_argument("--code-root", default=None, help="source tree containing football_rl to evaluate under")
    ap.add_argument("--matches", type=int, default=DEFAULT_MATCHES, help="matches to play (multiple of 8)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default=None, help="write the full JSON result here")
    ap.add_argument("--keep-replays", default=None, metavar="DIR",
                    help="export replays here and keep them (default: scratch directory, deleted afterwards)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_eval(
        args.checkpoint_dir,
        matches=args.matches,
        seed=args.seed,
        keep_replays=args.keep_replays,
        code_root=args.code_root,
    )
    if args.out:
        write_json(args.out, result)
    print("\n".join(_summary_lines(result)))
    return 0


if __name__ == "__main__":  # required: SubprocVecFootballEnv spawns workers that re-import this module
    raise SystemExit(main())
