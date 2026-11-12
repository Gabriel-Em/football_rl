"""Unbiased, seeded evaluation matches with step-resolution traces.

Plays full matches with the training environment settings and records one `.npz` trace per match.
Nothing is written to `checkpoints/` or `replays/`; traces go to an explicit output directory that
must lie outside every checkpoint directory and outside the repository's `checkpoints/` and
`replays/` folders.

Reproducibility: match `i` of a run with seed `s` resets the environment with seed `s * 1000 + i` and
draws policy noise from a `torch.Generator` seeded with the same value, so the traces of a match do
not depend on `n`, on the worker count or on the other matches.

Policies: team A plays the `blue_*` state dicts of the checkpoint (team B the `red_*` ones, or with
`--swap` the other way round). Opponents:
  self          team B plays the checkpoint's own other state dicts
  scripted      team B is the scripted baseline (`football_rl.baselines`)
  frozen:DIR    team B plays the `red_*` state dicts of DIR
  scripted2     both teams are the scripted baseline (no checkpoint needed)
Modes: `sampled` draws all four action components from the policy distribution, `mixed` uses the mean
for throttle and steer and samples kick and special, `greedy` uses the mean. `--sigma-scale` multiplies
the standard deviation of the sampled components.

Trace layout (row t = decision t, T = steps played): `pos vel ang ball bvel holder inflight lastteam box
half side hold_t act mean val` hold the state before the step-t action (act = clipped actions applied,
mean = deterministic action means, val = critic values; scripted teams record mean = act and val = 0);
`rew`, `lastkick` and every `s_*` counter hold the value after step t. `holder` is -1 while the ball is
loose, `side` is 0 for the team defending the left goal, `box` is the active keeper box radius per side.
`final_score`, `n_steps` are scalars. `--store-obs` adds `obs_s` and `cen_s` (observations of every 5th
decision); `--breakdown` enables replay recording in the environment and adds `bd` (ball advance,
individual potential, role potential, crowd potential, sparse), `role` (label index) and `phi`.

Usage:
  python tools/audit/harness.py --checkpoint-dir DIR --opp self --mode sampled --n 96 --seed 11
         --workers 10 --out-dir OUT [--sigma-scale 0.5] [--swap] [--store-obs] [--breakdown]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.audit._common import REPO_ROOT  # noqa: E402

__all__ = ["EvalSpec", "Session", "run_matches", "iter_matches", "play_match", "save_traces", "check_output_dir", "main"]

MODES = ("sampled", "mixed", "greedy")
FIXED_OPPONENTS = ("self", "scripted", "scripted2")
POLICY_FILES = ("{prefix}_field_policy.pt", "{prefix}_keeper_policy.pt")
OBS_STRIDE = 5
ROLE_LABELS = ("GK", "BALL", "CHSE", "1stP", "2ndP", "COVR", "STUN", "LNGE")
VECTOR_STATS = ("score", "team_passes", "team_tackles", "team_saves", "shots", "team_assists", "possession_frames")
SCALAR_STATS = (
    "goals", "own_goals", "completed_passes", "through_balls", "wall_passes", "interceptions",
    "successful_tackles", "failed_tackles", "keeper_saves", "field_5s_violations", "keeper_5s_violations",
    "stationary_ball_frames", "assists",
)
REPLAY_KEEP_THRESHOLD = 1e9


@dataclass(frozen=True)
class EvalSpec:
    """One evaluation: which policies play, how they act and which seeded matches are run."""

    checkpoint_dir: Optional[str] = None
    opponent: str = "self"
    mode: str = "sampled"
    n: int = 8
    seed: int = 1
    sigma_scale: float = 1.0
    swap: bool = False
    breakdown: bool = False
    store_obs: bool = False
    workers: int = 1
    max_steps: Optional[int] = None

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.opponent not in FIXED_OPPONENTS and not self.opponent.startswith("frozen:"):
            raise ValueError(f"opponent must be one of {FIXED_OPPONENTS} or frozen:DIR, got {self.opponent!r}")
        if self.n <= 0:
            raise ValueError(f"n must be positive, got {self.n}")
        if self.sigma_scale < 0.0:
            raise ValueError(f"sigma_scale must be non-negative, got {self.sigma_scale}")
        if self.workers <= 0:
            raise ValueError(f"workers must be positive, got {self.workers}")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        for path in self.required_files():
            if not path.is_file():
                raise FileNotFoundError(f"missing policy file: {path}")

    @property
    def opponent_dir(self) -> Optional[str]:
        return self.opponent.split(":", 1)[1] if self.opponent.startswith("frozen:") else None

    def policy_dirs(self) -> List[str]:
        """Directories the evaluation reads policies from (empty when both teams are scripted)."""
        dirs: List[str] = []
        if self.opponent != "scripted2":
            if not self.checkpoint_dir:
                raise ValueError("checkpoint_dir is required unless the opponent is scripted2")
            dirs.append(self.checkpoint_dir)
        if self.opponent_dir:
            dirs.append(self.opponent_dir)
        return dirs

    def team_prefixes(self) -> tuple[str, str]:
        """Checkpoint file prefixes of team A and of the `self` opponent."""
        return ("red", "blue") if self.swap else ("blue", "red")

    def required_files(self) -> List[Path]:
        prefixes = {"scripted2": (), "scripted": self.team_prefixes()[:1], "self": self.team_prefixes()}.get(
            self.opponent, self.team_prefixes()[:1]
        )
        files = [Path(d) / pat.format(prefix=pre) for d in self.policy_dirs()[:1] for pre in prefixes for pat in POLICY_FILES]
        if self.opponent_dir:
            files += [Path(self.opponent_dir) / pat.format(prefix="red") for pat in POLICY_FILES]
        return files


def check_output_dir(out_dir: Path | str, protected: Sequence[Path | str]) -> Path:
    """Return the resolved `out_dir`, refusing a location inside a checkpoint or replay directory."""
    out = Path(out_dir).resolve()
    guarded = [Path(p).resolve() for p in protected]
    for root in (REPO_ROOT, Path.cwd()):
        guarded += [(root / "checkpoints").resolve(), (root / "replays").resolve()]
    for g in guarded:
        if out == g or g in out.parents:
            raise ValueError(f"refusing to write traces inside {g}")
    return out


def _linear_layers(module) -> List[tuple]:
    layers = [(m.weight, m.bias) for m in module if isinstance(m, torch.nn.Linear)]
    if len(layers) != 3:
        raise ValueError(f"expected three linear layers, found {len(layers)}")
    return layers


def _mlp(x, layers, squash_output: bool):
    """Forward pass of a 3-layer tanh network by direct functional calls (bit-identical to the module)."""
    for i, (weight, bias) in enumerate(layers):
        x = F.linear(x, weight, bias)
        if i < 2 or squash_output:
            x = torch.tanh(x)
    return x


class TeamPolicy:
    """A team's shared field and keeper networks with batched forward passes."""

    def __init__(self, directory: str, prefix: str) -> None:
        from football_rl.train_ppo import ActorCritic

        base = Path(directory)
        self.heads = []
        for pattern, slot in ((POLICY_FILES[1], slice(0, 1)), (POLICY_FILES[0], slice(1, 6))):
            net = ActorCritic()
            net.load_state_dict(torch.load(base / pattern.format(prefix=prefix), map_location="cpu", weights_only=True))
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            self.heads.append((slot, _linear_layers(net.actor_mean), _linear_layers(net.critic), net.actor_logstd.exp()))

    def forward(self, obs, cen, slot, actor, critic):
        """Return `(mean, value)` of one network for its slice of the team."""
        o = torch.as_tensor(obs[slot])
        c = torch.as_tensor(cen[slot])
        return _mlp(o, actor, True), _mlp(torch.cat([o, c], dim=-1), critic, False).squeeze(-1)

    def act(self, obs, cen, mode: str, sigma_scale: float, gen):
        """Return `(actions (6,4), values (6,), means (6,4))` for the team's six slots."""
        actions = np.empty((6, 4), dtype=np.float32)
        values = np.empty(6, dtype=np.float32)
        means = np.empty((6, 4), dtype=np.float32)
        with torch.inference_mode():
            for slot, actor, critic, std in self.heads:
                mean, value = self.forward(obs, cen, slot, actor, critic)
                noise = torch.randn(mean.shape, generator=gen)
                sampled = mean + sigma_scale * std * noise
                if mode == "greedy":
                    chosen = mean
                elif mode == "mixed":
                    chosen = torch.cat([mean[:, :2], sampled[:, 2:]], dim=1)
                else:
                    chosen = sampled
                actions[slot] = chosen.numpy()
                means[slot] = mean.numpy()
                values[slot] = value.numpy()
        return actions, values, means


def _build_env(spec: EvalSpec, replay_dir: str):
    from football_rl.config import DEFAULT_CONFIG
    from football_rl.env import FootballParallelEnv

    cfg = replace(
        DEFAULT_CONFIG,
        replay=replace(
            DEFAULT_CONFIG.replay,
            enable_recording=bool(spec.breakdown),
            interest_threshold=REPLAY_KEEP_THRESHOLD,
            save_dir=replay_dir,
        ),
    )
    return FootballParallelEnv(config=cfg)


def _load_policies(spec: EvalSpec):
    if spec.opponent == "scripted2":
        return None, None
    first, second = spec.team_prefixes()
    team_a = TeamPolicy(spec.checkpoint_dir, first)
    if spec.opponent == "self":
        team_b = TeamPolicy(spec.checkpoint_dir, second)
    elif spec.opponent.startswith("frozen:"):
        team_b = TeamPolicy(spec.opponent_dir, "red")
    else:
        team_b = None
    return team_a, team_b


def _alloc(steps: int, with_obs: bool, with_breakdown: bool) -> Dict[str, np.ndarray]:
    f32, i8, i32 = np.float32, np.int8, np.int32
    shapes = {
        "pos": ((12, 2), f32), "vel": ((12, 2), f32), "ang": ((12,), f32), "ball": ((2,), f32), "bvel": ((2,), f32),
        "holder": ((), i8), "inflight": ((), i8), "lastteam": ((), i8), "half": ((), i8), "side": ((2,), i8),
        "box": ((2,), f32), "hold_t": ((), f32),
        "act": ((12, 4), f32), "mean": ((12, 4), f32), "val": ((12,), f32), "rew": ((12,), f32), "lastkick": ((12,), f32),
    }
    for name in VECTOR_STATS:
        shapes["s_" + name] = ((2,), i32)
    for name in SCALAR_STATS:
        shapes["s_" + name] = ((), i32)
    if with_breakdown:
        shapes.update({"bd": ((12, 5), f32), "role": ((12,), i8), "phi": ((12,), f32)})
    out = {k: np.zeros((steps,) + shp, dt) for k, (shp, dt) in shapes.items()}
    if with_obs:
        n_obs = (steps + OBS_STRIDE - 1) // OBS_STRIDE
        out["obs_s"] = np.zeros((n_obs, 12, 76), f32)
        out["cen_s"] = np.zeros((n_obs, 12, 54), f32)
    return out


class Session:
    """Policies of one evaluation, reusable for any number of matches of its spec.

    Every match runs in a freshly constructed environment, so it never depends on the matches played
    before it. Torch runs single-threaded while a session is open: the intra-op thread count changes the
    floating-point summation order of the matrix products and with it the sampled trajectories.
    """

    def __init__(self, spec: EvalSpec) -> None:
        spec.validate()
        self.spec = spec
        self._threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.replay_dir = tempfile.mkdtemp(prefix="harness_replays_") if spec.breakdown else None
        self.policies = _load_policies(spec)

    def close(self) -> None:
        torch.set_num_threads(self._threads)
        if self.replay_dir:
            shutil.rmtree(self.replay_dir, ignore_errors=True)
            self.replay_dir = None

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def play(self, idx: int) -> Dict[str, np.ndarray]:
        env = _build_env(self.spec, self.replay_dir or tempfile.gettempdir())
        return _play(self.spec, idx, env, self.policies)


def play_match(spec: EvalSpec, idx: int) -> Dict[str, np.ndarray]:
    """Play match `idx` of `spec` in a fresh environment and return its trace."""
    with Session(spec) as session:
        return session.play(idx)


def _play(spec: EvalSpec, idx: int, env, policies) -> Dict[str, np.ndarray]:
    from football_rl.baselines import scripted_actions

    team_a, team_b = policies
    match_seed = spec.seed * 1000 + idx
    gen = torch.Generator().manual_seed(match_seed)
    env.reset(seed=match_seed)
    obs = env._get_observations_array()
    cen = env.get_centralized_state()
    horizon = int(env.cfg.physics.max_rl_steps)
    if spec.max_steps is not None:
        horizon = min(horizon, int(spec.max_steps))
    tr = _alloc(horizon, spec.store_obs, spec.breakdown)
    steps = 0
    for t in range(horizon):
        if team_a is None:
            act_a, val_a = scripted_actions(env, 0), np.zeros(6, np.float32)
            mean_a = act_a
        else:
            act_a, val_a, mean_a = team_a.act(obs[:6], cen[:6], spec.mode, spec.sigma_scale, gen)
        if team_b is None:
            act_b, val_b = scripted_actions(env, 1), np.zeros(6, np.float32)
            mean_b = act_b
        else:
            act_b, val_b, mean_b = team_b.act(obs[6:], cen[6:], spec.mode, spec.sigma_scale, gen)
        act = np.concatenate([act_a, act_b])
        val = np.concatenate([val_a, val_b])
        if spec.store_obs and t % OBS_STRIDE == 0:
            tr["obs_s"][t // OBS_STRIDE] = obs
            tr["cen_s"][t // OBS_STRIDE] = cen
        tr["pos"][t] = env.player_pos
        tr["vel"][t] = env.player_vel
        tr["ang"][t] = env.player_angle
        tr["ball"][t] = env.ball_pos
        tr["bvel"][t] = env.ball_vel
        tr["holder"][t] = -1 if env.ball_holder is None else env.ball_holder
        tr["inflight"][t] = env.ball_kicked_in_flight
        tr["lastteam"][t] = -1 if env.last_ball_team is None else env.last_ball_team
        tr["box"][t] = env.active_box_radius
        tr["half"][t] = env.current_half
        tr["side"][t] = env.team_side
        tr["hold_t"][t] = env.ball_hold_timer
        tr["act"][t] = np.clip(act, -1.0, 1.0)
        tr["mean"][t] = np.clip(np.concatenate([mean_a, mean_b]), -1.0, 1.0)
        tr["val"][t] = val
        obs, cen, rew, done, _ = env.step_array(act, val)
        tr["rew"][t] = rew
        tr["lastkick"][t] = env.last_kick_time
        stats = env.stats
        for name in VECTOR_STATS:
            tr["s_" + name][t] = stats[name]
        for name in SCALAR_STATS:
            tr["s_" + name][t] = stats.get(name, 0)
        if spec.breakdown:
            tr["bd"][t] = [
                [d["r_ball_adv"], d["r_pot_i"], d["phi_role"], d["phi_crowd"], d["r_sparse"]]
                for d in env.last_rew_breakdown
            ]
            tr["role"][t] = [ROLE_LABELS.index(s) for s in env.last_role_states]
            tr["phi"][t] = env._phi_curr
        steps = t + 1
        if done:
            break
    n_obs = (steps + OBS_STRIDE - 1) // OBS_STRIDE
    trace = {k: (v[:n_obs] if k in ("obs_s", "cen_s") else v[:steps]) for k, v in tr.items()}
    trace["final_score"] = np.array(env.stats["score"], dtype=np.int32)
    trace["n_steps"] = np.array(steps)
    return trace


# ----------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------
_WORKER_SESSION: Optional[Session] = None


def _init_worker(spec: EvalSpec) -> None:
    global _WORKER_SESSION
    _WORKER_SESSION = Session(spec)


def _play_in_worker(idx: int):
    t0 = time.time()
    trace = _WORKER_SESSION.play(idx)
    return idx, trace, time.time() - t0


def iter_matches(spec: EvalSpec) -> Iterator[tuple[int, Dict[str, np.ndarray], float]]:
    """Yield `(match_index, trace, seconds)` in match order; matches run in `spec.workers` processes."""
    spec.validate()
    if spec.workers == 1 or spec.n == 1:
        with Session(spec) as session:
            for idx in range(spec.n):
                t0 = time.time()
                trace = session.play(idx)
                yield idx, trace, time.time() - t0
        return
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(spec.workers, spec.n), initializer=_init_worker, initargs=(spec,)) as pool:
        yield from pool.imap(_play_in_worker, range(spec.n), chunksize=1)


def run_matches(spec: EvalSpec) -> List[Dict[str, np.ndarray]]:
    """Play all matches of `spec` and return their traces in match order."""
    return [trace for _, trace, _ in iter_matches(spec)]


def save_trace(trace: Dict[str, np.ndarray], out_dir: Path, idx: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"m{idx:03d}.npz"
    np.savez_compressed(path, **trace)
    return path


def save_traces(traces: Sequence[Dict[str, np.ndarray]], out_dir: Path | str, protected: Sequence[Path | str] = ()) -> Path:
    """Write `m000.npz, m001.npz, ...` into `out_dir` after checking that it is not a protected location."""
    out = check_output_dir(out_dir, protected)
    for idx, trace in enumerate(traces):
        save_trace(trace, out, idx)
    return out


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", default=str(REPO_ROOT / "checkpoints"), help="directory with blue_*/red_* policies (read only)")
    ap.add_argument("--opp", default="self", help="self | scripted | scripted2 | frozen:DIR")
    ap.add_argument("--mode", default="sampled", choices=MODES)
    ap.add_argument("--n", type=int, default=8, help="number of matches")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sigma-scale", type=float, default=1.0, help="multiplier on the policy standard deviation")
    ap.add_argument("--swap", action="store_true", help="team A plays the red_* policies, team B the blue_* ones")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--store-obs", action="store_true", help="also store observations of every 5th decision")
    ap.add_argument("--breakdown", action="store_true", help="record the reward breakdown (enables environment recording)")
    ap.add_argument("--max-steps", type=int, default=None, help="stop each match after this many decisions")
    ap.add_argument("--out-dir", required=True, help="directory receiving m000.npz, m001.npz, ...")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    spec = EvalSpec(
        checkpoint_dir=args.checkpoint_dir,
        opponent=args.opp,
        mode=args.mode,
        n=args.n,
        seed=args.seed,
        sigma_scale=args.sigma_scale,
        swap=args.swap,
        breakdown=args.breakdown,
        store_obs=args.store_obs,
        workers=args.workers,
        max_steps=args.max_steps,
    )
    out = check_output_dir(args.out_dir, spec.policy_dirs())
    started = time.time()
    scores, elapsed = [], []
    for idx, trace, seconds in iter_matches(spec):
        save_trace(trace, out, idx)
        scores.append(trace["final_score"])
        elapsed.append(seconds)
    wall = time.time() - started
    scores_arr = np.array(scores, dtype=np.float64)
    print(
        f"harness: n={len(scores)} opp={spec.opponent} mode={spec.mode} seed={spec.seed} "
        f"mean goals A={scores_arr[:, 0].mean():.3f} B={scores_arr[:, 1].mean():.3f} "
        f"total={scores_arr.sum(1).mean():.3f} | {np.mean(elapsed):.1f} s/match, wall {wall:.0f} s -> {out}"
    )
    return 0


if __name__ == "__main__":  # required: spawned workers re-import this module
    raise SystemExit(main())
