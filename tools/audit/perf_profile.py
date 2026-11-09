"""Seeded per-component timing and numpy call counts for `FootballParallelEnv.step_array`.

Pass 1 (timing): selected env methods are wrapped on the instance and timed
(inclusive and self time). Pass 2 (counting, Python >= 3.12): an identical seeded
run counts calls into numpy (functions, ufuncs, ndarray methods) made from
non-numpy code, attributed to the innermost wrapped component. Array operators
(`a + b`) allocate too but are invisible to call monitoring, so the count is a
lower bound on allocations; `numpy_constructor_calls` isolates explicit
constructors (array/zeros/copy/astype/...). Replay export is disabled; nothing
is written to disk.

Usage:
  python tools/audit/perf_profile.py [--steps 300] [--seed 0] [--no-record]
         [--no-count] [--json OUT]
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.audit._common import ensure_repo_on_path, write_json  # noqa: E402

__all__ = ["profile_env", "DEFAULT_COMPONENTS", "main"]

DEFAULT_COMPONENTS: tuple[str, ...] = (
    "_physics_substep",
    "_resolve_player_collisions",
    "_clamp_players_to_boundaries",
    "_check_free_ball_pickup",
    "_check_keeper_perimeter_strip",
    "_execute_kick",
    "_attempt_tackle",
    "_bounce_ball_off_walls",
    "_check_goal_scored",
    "_compute_ball_potential",
    "_compute_potential_shaping",
    "_record_replay_frame",
    "_get_observations_array",
    "get_centralized_state",
)
CONSTRUCTOR_NAMES = frozenset({
    "array", "asarray", "zeros", "zeros_like", "ones", "ones_like", "empty", "empty_like", "full", "full_like",
    "copy", "astype", "concatenate", "stack", "hstack", "vstack", "tolist",
})
_NUMPY_DIR = os.path.dirname(np.__file__)


def _make_env(record: bool, seed: int) -> Any:
    ensure_repo_on_path()
    from football_rl.config import DEFAULT_CONFIG
    from football_rl.env import FootballParallelEnv

    cfg = dataclasses.replace(
        DEFAULT_CONFIG,
        replay=dataclasses.replace(DEFAULT_CONFIG.replay, enable_recording=record, interest_threshold=float("inf")),
    )
    env = FootballParallelEnv(config=cfg)
    env.reset(seed=seed)
    return env


def _is_numpy_callable(c: Any) -> bool:
    if isinstance(c, np.ufunc):
        return True
    owner = getattr(c, "__self__", None)
    if isinstance(owner, (np.ndarray, np.generic)):
        return True
    mod = getattr(c, "__module__", None) or ""
    return mod == "numpy" or mod.startswith("numpy.")


class _Tracker:
    def __init__(self) -> None:
        self.stack: List[List[Any]] = []          # [name, start, child_time]
        self.calls: Counter = Counter()
        self.incl: Counter = Counter()
        self.self_t: Counter = Counter()

    def wrap(self, name: str, fn: Callable[..., Any], timed: bool) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            frame = [name, time.perf_counter() if timed else 0.0, 0.0]
            self.stack.append(frame)
            try:
                return fn(*args, **kwargs)
            finally:
                self.stack.pop()
                self.calls[name] += 1
                if timed:
                    elapsed = time.perf_counter() - frame[1]
                    self.incl[name] += elapsed
                    self.self_t[name] += elapsed - frame[2]
                    if self.stack:
                        self.stack[-1][2] += elapsed

        return wrapper

    @property
    def current(self) -> str:
        return self.stack[-1][0] if self.stack else "(step_array other)"


def _run(steps: int, seed: int, record: bool, components: Sequence[str], timed: bool,
         on_start: Optional[Callable[[_Tracker], None]] = None,
         on_stop: Optional[Callable[[], None]] = None) -> tuple[_Tracker, float]:
    random.seed(seed)
    np.random.seed(seed)
    env = _make_env(record, seed)
    actions = np.random.default_rng(seed).uniform(-1.0, 1.0, size=(steps, env.num_players, 4))
    tr = _Tracker()
    for name in components:
        fn = getattr(env, name, None)
        if callable(fn):
            setattr(env, name, tr.wrap(name, fn, timed))
    if on_start:
        on_start(tr)
    t0 = time.perf_counter()
    try:
        for k in range(steps):
            _, _, _, done, _ = env.step_array(actions[k])
            if done:
                env.reset(seed=seed + k + 1)
    finally:
        total = time.perf_counter() - t0
        if on_stop:
            on_stop()
    return tr, total


def _count_numpy(steps: int, seed: int, record: bool, components: Sequence[str]) -> Optional[Dict[str, Any]]:
    mon = getattr(sys, "monitoring", None)
    if mon is None:
        return None
    tool = next((t for t in (mon.PROFILER_ID, 3, 4) if mon.get_tool(t) is None), None)
    if tool is None:
        return None
    by_comp: Counter = Counter()
    ctor_by_comp: Counter = Counter()
    state: Dict[str, Optional[_Tracker]] = {"tr": None}

    def on_call(code: Any, _offset: int, callable_: Any, _arg0: Any) -> Any:
        if code.co_filename.startswith(_NUMPY_DIR) or not _is_numpy_callable(callable_):
            return None
        tr = state["tr"]
        comp = tr.current if tr is not None else "(outside)"
        by_comp[comp] += 1
        if getattr(callable_, "__name__", "") in CONSTRUCTOR_NAMES:
            ctor_by_comp[comp] += 1
        return None

    def start(tr: _Tracker) -> None:
        state["tr"] = tr
        mon.use_tool_id(tool, "audit.perf_profile")
        mon.register_callback(tool, mon.events.CALL, on_call)
        mon.set_events(tool, mon.events.CALL)

    def stop() -> None:
        mon.set_events(tool, 0)
        mon.register_callback(tool, mon.events.CALL, None)
        mon.free_tool_id(tool)

    _run(steps, seed, record, components, timed=False, on_start=start, on_stop=stop)
    by_comp.pop("(outside)", None)
    ctor_by_comp.pop("(outside)", None)
    return {
        "numpy_calls_per_step": sum(by_comp.values()) / steps,
        "numpy_constructor_calls_per_step": sum(ctor_by_comp.values()) / steps,
        "numpy_calls_by_component": {k: v / steps for k, v in sorted(by_comp.items())},
        "numpy_constructor_calls_by_component": {k: v / steps for k, v in sorted(ctor_by_comp.items())},
    }


def profile_env(
    steps: int = 300,
    seed: int = 0,
    record: bool = True,
    count_numpy: bool = True,
    components: Sequence[str] = DEFAULT_COMPONENTS,
) -> Dict[str, Any]:
    """Profile `steps` seeded random-action RL steps; times are wall-clock and machine dependent."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    tr, total = _run(steps, seed, record, components, timed=True)
    total_ms = total * 1000.0
    comps = {}
    for name in components:
        comps[name] = {
            "calls": int(tr.calls.get(name, 0)),
            "total_ms": tr.incl.get(name, 0.0) * 1000.0,
            "self_ms": tr.self_t.get(name, 0.0) * 1000.0,
            "self_ms_per_step": tr.self_t.get(name, 0.0) * 1000.0 / steps,
            "self_pct": (100.0 * tr.self_t.get(name, 0.0) / total) if total > 0 else 0.0,
        }
    out: Dict[str, Any] = {
        "steps": steps,
        "seed": seed,
        "record": record,
        "total_ms_per_step": total_ms / steps,
        "steps_per_sec": steps / total if total > 0 else None,
        "components": comps,
        "numpy_calls_per_step": None,
        "numpy_constructor_calls_per_step": None,
        "numpy_calls_by_component": None,
        "numpy_constructor_calls_by_component": None,
    }
    if count_numpy:
        counted = _count_numpy(steps, seed, record, components)
        if counted:
            out.update(counted)
    return out


def _summary(res: Dict[str, Any]) -> List[str]:
    lines = [
        f"perf_profile: steps={res['steps']} seed={res['seed']} record={res['record']} "
        f"{res['total_ms_per_step']:.3f} ms/step ({res['steps_per_sec']:.0f} steps/s single env)"
    ]
    rows = sorted(res["components"].items(), key=lambda kv: -kv[1]["self_ms"])
    npc = res.get("numpy_calls_by_component") or {}
    ctor = res.get("numpy_constructor_calls_by_component") or {}
    lines.append(f"  {'component':32s} {'calls':>7s} {'self ms/step':>12s} {'self %':>7s} {'np/step':>8s} {'ctor/step':>9s}")
    for name, c in rows:
        lines.append(f"  {name:32s} {c['calls']:7d} {c['self_ms_per_step']:12.4f} {c['self_pct']:7.1f} "
                     f"{npc.get(name, 0.0):8.1f} {ctor.get(name, 0.0):9.1f}")
    if res.get("numpy_calls_per_step") is not None:
        lines.append(f"  numpy calls/step={res['numpy_calls_per_step']:.1f} "
                     f"(constructors {res['numpy_constructor_calls_per_step']:.1f}; "
                     f"outside components {npc.get('(step_array other)', 0.0):.1f})")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-record", action="store_true", help="disable 60 Hz replay recording")
    ap.add_argument("--no-count", action="store_true", help="skip the numpy call-count pass")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    res = profile_env(steps=args.steps, seed=args.seed, record=not args.no_record, count_numpy=not args.no_count)
    print("\n".join(_summary(res)))
    if args.json:
        write_json(args.json, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
