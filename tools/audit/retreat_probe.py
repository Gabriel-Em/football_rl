"""Line-order probe: do covering defenders ahead of their own striker gain by retreating? (read-only)

Plays one fresh match per seed from a checkpoint with a single environment, snapshots the potential-field
state every few steps, and evaluates two properties of the potential field on those snapshots:

* `retreat_penalty_share`: over defenders 1 and 2 that carry the covering role, stand more than 30 u ahead
  of their own striker (canonical x) and stand in the opponent half, the fraction for which moving 20 u
  straight back (toward their own goal) lowers their own potential. Reported overall and split by whether
  the defender's team is in possession (the covering label is also the in-possession supporter template).
* cluster proxy: on 300 sampled snapshots, every outfielder of a team that is out of possession takes
  10 simultaneous best-response rounds on its own potential (best of staying or 8 compass steps of 20 u,
  clamped to the pitch), and the resulting formation is summarised: mean players within 90 u of the ball,
  out-of-possession teammate pairs closer than 110 u, mean nearest-teammate distance and the share of
  out-of-possession teams with a defender still ahead of its striker.

`--rewards-override key=value` (repeatable) replaces `RewardConfig` fields for the evaluation only; the
matches are played with the unmodified configuration, so every override sees the same snapshots.

Nothing is written to the checkpoint directory, `./checkpoints` or `./replays`; the scratch replay
directory is deleted afterwards.

Usage:
  python tools/audit/retreat_probe.py --checkpoint-dir DIR [--seeds 11 12 13] [--steps N] [--every 3]
         [--cluster-states 300] [--rewards-override line_order_weight=0 ...] [--json OUT]
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.audit._common import fmt, write_json  # noqa: E402

__all__ = [
    "collect_states",
    "retreat_penalty_share",
    "cluster_proxy",
    "parse_rewards_override",
    "run_probe",
    "main",
]

DEFAULT_SEEDS = (11, 12, 13)
DEFAULT_EVERY = 3
DEFAULT_CLUSTER_STATES = 300
LEAD_MARGIN = 30.0  # a defender counts as out of line when it stands this far ahead of its striker
RETREAT_STEP = 20.0
ASCENT_ROUNDS = 10
ASCENT_MARGIN = 18.0  # closest approach to a pitch boundary during the ascent
BALL_CLUSTER_RADIUS = 90.0
CLOSE_PAIR_RADIUS = 110.0
SAMPLE_SEED = 0
_TIE_EPS = 1e-12


def parse_rewards_override(items: Optional[Sequence[str]], base: Any = None) -> Any:
    """`RewardConfig` with each `key=value` item applied to `base` (default: `DEFAULT_CONFIG.rewards`).

    Raises ValueError for an item without `=`, an unknown field, a non-numeric value, or a value the
    reward configuration rejects.
    """
    from football_rl.config import DEFAULT_CONFIG

    rewards = DEFAULT_CONFIG.rewards if base is None else base
    known = {f.name for f in fields(rewards)}
    changes: Dict[str, float] = {}
    for item in items or ():
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep:
            raise ValueError(f"--rewards-override expects key=value, got {item!r}")
        if key not in known:
            raise ValueError(f"unknown RewardConfig field {key!r}")
        try:
            changes[key] = float(raw)
        except ValueError:
            raise ValueError(f"--rewards-override {key}: {raw!r} is not a number") from None
    return replace(rewards, **changes) if changes else rewards


def collect_states(
    checkpoint_dir: str,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    steps: Optional[int] = None,
    every: int = DEFAULT_EVERY,
    config: Any = None,
) -> Dict[str, Any]:
    """Play one match per seed (single environment, sampled actions) and snapshot every `every`-th step.

    Returns `{"states": [PotentialState, ...], "generation": int, "per_seed": [count, ...]}`. `steps`
    defaults to one step short of the match length: a match that runs out re-exports its replay on every
    further step. Snapshots are array copies taken after each environment step.
    """
    import numpy as np
    import torch

    from football_rl.config import DEFAULT_CONFIG
    from football_rl.potentials import PotentialState
    from football_rl.train_ppo import ParameterSharingPPOTrainer

    ckpt = Path(checkpoint_dir)
    if not ckpt.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {ckpt}")
    if every <= 0:
        raise ValueError(f"every must be positive, got {every}")
    base = config if config is not None else DEFAULT_CONFIG
    match_steps = int(base.physics.max_rl_steps)
    steps = match_steps - 1 if steps is None else int(steps)
    if not 0 < steps < match_steps:
        raise ValueError(f"steps must be in [1, {match_steps - 1}] (one match without a reset), got {steps}")

    states: List[Any] = []
    per_seed: List[int] = []
    generation = -1
    scratch = Path(tempfile.mkdtemp(prefix="retreat_probe_"))
    try:
        cfg = replace(base, replay=replace(base.replay, save_dir=str(scratch)))
        for seed in seeds:
            torch.manual_seed(int(seed))
            trainer = ParameterSharingPPOTrainer(
                config=cfg, num_envs=1, rollout_steps=steps, device="cpu", checkpoint_dir=str(ckpt),
                seed=int(seed), save_best_only=False,
            )
            try:
                if not trainer.load_checkpoints():
                    raise RuntimeError(f"no loadable checkpoint in {ckpt}")
                generation = int(trainer.current_generation)
                env = trainer.vec_env.envs[0]
                taken = 0
                for t in range(steps):
                    actions, _, values = trainer._select_actions(trainer.current_obs, trainer.current_central_state)
                    obs, central, _, _, _ = trainer.vec_env.step_array(np.clip(actions, -1, 1), values)
                    trainer.current_obs, trainer.current_central_state = obs, central
                    if t % every:
                        continue
                    stunned = env.stun_timer > 0.0
                    holder = env.ball_holder
                    last_team = env.last_ball_team
                    states.append(
                        PotentialState(
                            pos=env.player_pos.copy(),
                            angle=env.player_angle.copy(),
                            ball=env.ball_pos.copy(),
                            holder=-1 if holder is None else int(holder),
                            ball_in_flight=bool(env.ball_kicked_in_flight),
                            last_ball_team=-1 if last_team is None else int(last_team),
                            team_side=env.team_side.copy(),
                            disabled=(stunned | env.is_ghost).copy(),
                            stunned=stunned.copy(),
                            post_kick_timer=env.post_kick_pbrs_timer.copy(),
                            depth_bias=env._tactical_depth_bias.copy(),
                            flank_bias=env._tactical_flank_bias.copy(),
                            active_box_radius=env.active_box_radius.copy(),
                        )
                    )
                    taken += 1
                per_seed.append(taken)
            finally:
                trainer.close()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return {"states": states, "generation": generation, "per_seed": per_seed}


def _out_of_possession_teams(state: Any) -> List[int]:
    """Teams that neither hold the ball nor have a kick of theirs in flight."""
    holder_team = -1 if state.holder < 0 else int(state.holder) // 6
    flight_team = int(state.last_ball_team) if (state.holder < 0 and state.ball_in_flight) else -1
    return [t for t in (0, 1) if t != holder_team and t != flight_team]


def _team_direction(state: Any, team: int) -> float:
    """+1 when the team defends the left goal (attacks toward +x), -1 otherwise."""
    from football_rl.config import SIDE_LEFT

    return 1.0 if int(state.team_side[team]) == SIDE_LEFT else -1.0


def _canonical_x(state: Any, team: int, pos: Any, width: float) -> Any:
    """x of every player measured from the team's own goal line along its attacking direction."""
    sign = _team_direction(state, team)
    return (pos[:, 0] - (0.0 if sign > 0.0 else width)) * sign


_MOVES = [(0.0, 0.0)] + [
    (RETREAT_STEP * math.cos(k * math.pi / 4.0), RETREAT_STEP * math.sin(k * math.pi / 4.0)) for k in range(8)
]


def _candidate_points(x: float, y: float, width: float, height: float) -> List[tuple]:
    """Staying put and the 8 compass steps of `RETREAT_STEP`, each clamped to the pitch margin."""
    return [
        (min(width - ASCENT_MARGIN, max(ASCENT_MARGIN, x + dx)), min(height - ASCENT_MARGIN, max(ASCENT_MARGIN, y + dy)))
        for dx, dy in _MOVES
    ]


def _ascend(state: Any, rew: Any, phys: Any, movers: Sequence[int], rounds: int) -> Any:
    """Positions after `rounds` simultaneous best responses of `movers` on their own potential.

    Every mover evaluates its candidates against the positions at the start of the round, so the result does
    not depend on the order of `movers`. Ties keep the earlier candidate (staying put first).
    """
    from football_rl.potentials import compute_potentials

    pos = state.pos.copy()
    for _ in range(rounds):
        nxt = pos.copy()
        for g in movers:
            best, best_value = pos[g].copy(), -math.inf
            for cx_g, cy_g in _candidate_points(float(pos[g, 0]), float(pos[g, 1]), phys.field_width, phys.field_height):
                cand = pos.copy()
                cand[g, 0] = cx_g
                cand[g, 1] = cy_g
                value = float(compute_potentials(state._replace(pos=cand), rew, phys).phi[g])
                if value > best_value + _TIE_EPS:
                    best_value, best = value, cand[g].copy()
            nxt[g] = best
        pos = nxt
    return pos


def _share(counts: Sequence[int]) -> Dict[str, Any]:
    n, hits = counts
    return {"n": int(n), "share": (hits / n) if n else None}


def retreat_penalty_share(states: Sequence[Any], rew: Any, phys: Any) -> Dict[str, Any]:
    """Share of out-of-line covering defenders whose own potential falls when they step 20 u back."""
    from football_rl.potentials import ROLE_COVER, compute_potentials

    width = phys.field_width
    overall = [0, 0]
    out_of_possession = [0, 0]
    in_possession = [0, 0]
    for state in states:
        base = compute_potentials(state, rew, phys)
        oop = _out_of_possession_teams(state)
        for team in (0, 1):
            cx = _canonical_x(state, team, state.pos, width)
            sign = _team_direction(state, team)
            first = 6 * team
            for g in (first + 1, first + 2):
                if int(base.role_code[g]) != ROLE_COVER:
                    continue
                if not (cx[g] > cx[first + 5] + LEAD_MARGIN and cx[g] > 0.5 * width):
                    continue
                pos = state.pos.copy()
                pos[g, 0] -= sign * RETREAT_STEP
                lowered = float(compute_potentials(state._replace(pos=pos), rew, phys).phi[g]) < float(base.phi[g])
                bucket = out_of_possession if team in oop else in_possession
                for counts in (overall, bucket):
                    counts[0] += 1
                    counts[1] += lowered
    return {
        "overall": _share(overall),
        "out_of_possession": _share(out_of_possession),
        "in_possession": _share(in_possession),
    }


def cluster_proxy(
    states: Sequence[Any],
    rew: Any,
    phys: Any,
    n_states: int = DEFAULT_CLUSTER_STATES,
    rounds: int = ASCENT_ROUNDS,
    seed: int = SAMPLE_SEED,
) -> Dict[str, Any]:
    """Formation statistics after simultaneous best-response ascent of out-of-possession outfielders."""
    import numpy as np

    width = phys.field_width
    if not states:
        return {"n_states": 0}
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(states), size=min(int(n_states), len(states)), replace=False)
    near_ball_after = near_ball_before = 0.0
    pairs = nearest = 0.0
    inverted = 0
    team_count = 0
    for i in chosen:
        state = states[int(i)]
        oop = _out_of_possession_teams(state)
        movers = [g for t in oop for g in range(6 * t + 1, 6 * t + 6)]
        pos = _ascend(state, rew, phys, movers, rounds)
        near_ball_after += float((np.hypot(pos[:, 0] - state.ball[0], pos[:, 1] - state.ball[1]) <= BALL_CLUSTER_RADIUS).sum())
        near_ball_before += float(
            (np.hypot(state.pos[:, 0] - state.ball[0], state.pos[:, 1] - state.ball[1]) <= BALL_CLUSTER_RADIUS).sum()
        )
        for team in oop:
            outfield = pos[6 * team + 1: 6 * team + 6]
            gaps = np.hypot(outfield[:, None, 0] - outfield[None, :, 0], outfield[:, None, 1] - outfield[None, :, 1])
            np.fill_diagonal(gaps, np.inf)
            pairs += float((gaps[np.triu_indices(5, 1)] < CLOSE_PAIR_RADIUS).sum())
            nearest += float(gaps.min(axis=1).mean())
            cx = _canonical_x(state, team, pos, width)
            defenders = cx[6 * team + 1: 6 * team + 3]
            inverted += bool(((defenders > cx[6 * team + 5] + LEAD_MARGIN) & (defenders > 0.5 * width)).any())
            team_count += 1
    n = len(chosen)
    return {
        "n_states": int(n),
        "oop_teams": int(team_count),
        "mean_cluster_size": near_ball_after / n,
        "mean_cluster_size_start": near_ball_before / n,
        "oop_pairs_lt_110": pairs / team_count if team_count else None,
        "oop_nn_dist": nearest / team_count if team_count else None,
        "inversion_after_ascent": inverted / team_count if team_count else None,
    }


def run_probe(
    checkpoint_dir: str,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    steps: Optional[int] = None,
    every: int = DEFAULT_EVERY,
    cluster_states: int = DEFAULT_CLUSTER_STATES,
    rewards_override: Optional[Sequence[str]] = None,
    config: Any = None,
) -> Dict[str, Any]:
    """Collect snapshots from `checkpoint_dir` and evaluate both properties under the overridden rewards."""
    from football_rl.config import DEFAULT_CONFIG

    base = config if config is not None else DEFAULT_CONFIG
    rewards = parse_rewards_override(rewards_override, base.rewards)
    collected = collect_states(checkpoint_dir, seeds=seeds, steps=steps, every=every, config=base)
    states = collected["states"]
    return {
        "retreat_penalty_share": retreat_penalty_share(states, rewards, base.physics),
        "cluster_proxy": cluster_proxy(states, rewards, base.physics, n_states=cluster_states),
        "meta": {
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_generation": collected["generation"],
            "seeds": [int(s) for s in seeds],
            "snapshots": len(states),
            "snapshots_per_seed": collected["per_seed"],
            "every": int(every),
            "rewards_override": list(rewards_override or ()),
        },
    }


def _summary_lines(result: Dict[str, Any]) -> List[str]:
    meta, share, proxy = result["meta"], result["retreat_penalty_share"], result["cluster_proxy"]
    override = " ".join(meta["rewards_override"]) or "none"
    lines = [
        f"retreat_probe: gen {meta['checkpoint_generation']}, {meta['snapshots']} snapshots "
        f"(seeds {meta['seeds']}, every {meta['every']}), rewards override: {override}",
        "  retreat_penalty_share  "
        + "  ".join(f"{k} {fmt(share[k]['share'], 3)} (n={share[k]['n']})" for k in ("overall", "out_of_possession", "in_possession")),
    ]
    if proxy.get("n_states"):
        lines.append(
            f"  cluster proxy ({proxy['n_states']} snapshots)  mean_cluster_size {fmt(proxy['mean_cluster_size'], 3)} "
            f"(start {fmt(proxy['mean_cluster_size_start'], 3)})  oop_pairs<110 {fmt(proxy['oop_pairs_lt_110'], 3)}  "
            f"oop_nn_dist {fmt(proxy['oop_nn_dist'], 1)}  inversion_after_ascent {fmt(proxy['inversion_after_ascent'], 3)}"
        )
    return lines


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-dir", required=True, help="directory holding training_state.pt / policies")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS), help="one match per seed")
    ap.add_argument("--steps", type=int, default=None, help="RL steps per match (default: match length - 1)")
    ap.add_argument("--every", type=int, default=DEFAULT_EVERY, help="snapshot every N steps")
    ap.add_argument("--cluster-states", type=int, default=DEFAULT_CLUSTER_STATES, help="snapshots in the cluster proxy")
    ap.add_argument("--rewards-override", action="append", default=[], metavar="KEY=VALUE",
                    help="RewardConfig field for the evaluation (repeatable)")
    ap.add_argument("--json", default=None, metavar="OUT", help="write the full JSON result here")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_probe(
        args.checkpoint_dir,
        seeds=args.seeds,
        steps=args.steps,
        every=args.every,
        cluster_states=args.cluster_states,
        rewards_override=args.rewards_override,
    )
    if args.json:
        write_json(args.json, result)
    print("\n".join(_summary_lines(result)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
