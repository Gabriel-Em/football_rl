"""Counterfactual rollouts over harness traces: keeper reach (`gk`), hold-vs-kick (`hold`), kick aim (`aim`) and the
credit a decision's benefit receives in the trainer's advantage (`credit`).

From a harness trace directory (`m*.npz`) and (for `hold` and `aim`) the checkpoint that produced it,
re-simulates each match from its harness seed (`trace_seed * 1000 + match_idx`), verifies position
determinism (< 1e-3 u for players and ball across every recorded decision, skipping and counting
any match that fails), clones the engine at target decision points, and rolls counterfactual
branches:

  gk    At loose-ball keeper opportunities (loose ball inside the keeper's box, 40-140 u from the
        keeper, ball speed < 80 u/s, >= 3 s after that keeper's previous opportunity, and >= 15
        decisions remaining in the match), rolls 15 decisions (1 s) while all other players
        replay recorded actions and the keeper plays: (a) recorded actions, (b) scripted
        go-to-ball, (c) inert. Reports per-variant pickup within 1 s, undiscounted keeper step
        reward (including shaping), and undiscounted team step reward (sum over 6 players), plus
        paired (b) - (a). `--checkpoint-dir` is accepted for CLI symmetry but unused by `gk`.

  hold  At eligible outfield holding decisions (outfield holder, hold_t >= 0.08 s, >= 3 s after
        the previous point in the match), clones the env twice with identical generator seeds and
        loads policies via `tools/audit/harness.py` (supporting `gaussian4`, `hybrid5` and `hybrid6`).
        Branch KICK forces the holder's env-space kick column to +1 on decision 0; branch HOLD
        forces it to -1 for the first `--hold-steps` decisions (default 15). Rolls 75 decisions (5 s)
        and 225 decisions (15 s), reporting paired HOLD - KICK differences (overall, by pressure
        < 70 u / >= 120 u, by canonical third, and by the holder's kick intent) for discounted
        non-shaping return, full return with gamma^H team-mean critic bootstrap, P(team holds at
        1 s), P(goal for), P(goal against), and the non-shaping component split.
        Kick intent is read from the trace at the decision (`mean` kick column of the holder; the
        team's head from `head`, absent = gaussian4): hybrid5 or hybrid6, p = (mean + 1) / 2, `kick` if
        p > 0.5, `hold` if p < 0.05, else `mid`; gaussian4 (or a scripted team), `kick` if
        mean > 0.35, else `hold`. Buckets `intent_kick`, `intent_mid`, `intent_hold`, and in the
        defensive third `def_intent_kick`, `def_intent_hold`.

  aim   At real outfield kicks with an open teammate in the kickable arc: `tools/audit/census.py` pass-option records
        with an open candidate and a direction, where the kicker holds the ball at the kick row k, k + 2 < the trace
        length, and k is >= 45 decisions after the previous point of the match. The target is the open candidate with
        the smallest |theta_j - facing| (theta_j = angle of pos_j - ball, facing = `ang[k, kicker]`, wrapped to
        (-pi, pi]); it must be within 35 deg (`kick_steer_max_deflection_deg`), else the kick is no point. Points carry
        the canonical third of the ball, the kicker's intent (`decision_intent`), `already` (census `aimed_open`), the
        target distance and the signed deflection theta_target - facing. Three branches with one generator seed
        `(seed * 1_000_003 + match_idx * 10_000 + k) & 0x7FFFFFFF`, rolled for 75 decisions (5 s):
          POLICY  step 0 plays the recorded clipped actions `act[k]` (the real kick);
          AIM     as POLICY, with the kicker's steer set to clip(deflection / 35 deg, -1, 1), so the kick points at the
                  target with the recorded power;
          AIMP    as AIM, with the kicker's kick column set to threshold + pn * (1 - threshold) + 1e-4, where
                  pn = clip((v0 - v_min) / (v_max - v_min), 0, 1), v0 = min(v_max, sqrt(200^2 + 2 * friction * dist))
                  (the ball reaches the target at about 200 u/s), v_max the kicker's role kick speed and v_min
                  `ball_min_kick_speed`.
        From step 1 on both teams sample from the checkpoint (`mode="sampled"`); every branch draws the policies at
        step 0 too, so the generator stays aligned. Per branch: discounted team-mean non-shaping return and full return
        with the gamma^h team-mean critic bootstrap at h = 15, 45, 75 (no bootstrap after a terminal step); the kicking
        team in possession after decisions 15 and 30; the first holder other than the kicker after decisions 2..45
        (`target`, `teammate`, `opponent` or `none`); goals for and against by the end; and the angle between the ball's
        direction after step 0 and the ball -> target direction at the snapshot (None while a player holds the ball).
        Buckets: `not_aimed`, `already_aimed`, and the not-aimed points by third (`not_aimed_third_<third>`) and by
        intent (`not_aimed_intent_<intent>`); each gives per-branch shares and the paired AIM - POLICY and
        AIMP - POLICY differences (mean +- SE). The loaded policies must reproduce `val` and `mean` of the trace at
        decision 0 of every match within 1e-5, as in `hold`.

  credit How much of a decision's true counterfactual benefit reaches the trainer's GAE advantage, and with how much noise.
        `--decision aim` (default): the points of `aim` that are not already aimed (`find_aim_points`), with the branches POLICY
        (the recorded actions at decision 0) and AIM (as POLICY, the kicker's steer turned at the target). `--decision attack`:
        outfield kicks where the kicker holds the ball at kick row k (k + 2 < the trace length), the ball is in the kicker's
        attacking third (canonical x > 2/3 of the pitch) and k is >= 45 decisions after the previous point, with the branches
        KICK (the recorded actions at decision 0) and HOLD (as KICK, the kicker's env kick column forced to -1 for decisions 0-14).
        Seed index j = 0..K-1 (`--seeds`, default 4) gives every pair one generator seed
        `((seed + 7919 j) * 1_000_003 + match_idx * 10_000 + k) & 0x7FFFFFFF` (j = 0 is the seed of `aim`). From decision 1 on both
        teams sample from the checkpoint, for 75 decisions (5 s), then the kicker's critic value V_75 is read (none after a
        terminal step).
        Per branch the kicker's own reward per decision is split into `goal` = 10 x (goals for - goals against in the decision),
        `sparse` = the non-shaping reward - goal, `ball` = `team_advance_pbrs_weight` x the decision's change of the kicker team's
        ball potential (the full change, also in a decision that teleports the ball, where the env pays no shaping) and `role` =
        the shaping - ball; the critic value V_t of the kicker is recorded for t = 0..75.
        The estimator is the trainer's GAE at decision 0, A(lambda) = sum_l (gamma lambda)^l delta_l with delta_l = r_l +
        gamma V_{l+1} - V_l (gamma from the config), cut at 75 decisions with V_75 and without any bootstrap after a terminal
        step, for every lambda in `--lambdas` (default 0 0.5 0.8 0.9 0.95 0.99 1). A(lambda) splits into the four reward parts and
        a critic part, which add up to it. Per lambda:
          S     signal: the mean over points of the seed-averaged benefit of the decision, A_AIM - A_POLICY for aim, A_KICK - A_HOLD
                for attack (the recorded kick against holding), with its standard error over points;
          N     noise: sqrt of the mean over points of the variance across seeds of the recorded branch's A (same state, same
                action, different futures); SNR = S / N;
          paired SD  the standard deviation over all (point, seed) pairs of the same difference as S;
          variance share  of each part (goal, sparse, ball, role, critic) in the variance across seeds of the recorded branch;
          signal parts    the same parts of S (they add up to S).
        `attack` also reports the goals for and against per branch within the 75 decisions (mean per point and seed).
        A point set that is empty gives an empty result with every statistic None. The loaded policies must reproduce `val` and
        `mean` of the trace at decision 0 of every match within 1e-5, as in `hold`.

Nothing is written to `checkpoints/`, `replays/`, checkpoint directories, or the trace directory.

Usage:
  python tools/audit/rollouts.py gk --trace-dir DIR [--max-matches 48] [--workers 4] --out OUT
  python tools/audit/rollouts.py hold --trace-dir DIR --checkpoint-dir DIR [--max-matches 48]
         [--workers 4] [--seed 1] [--hold-steps 15] --out OUT
  python tools/audit/rollouts.py aim --trace-dir DIR --checkpoint-dir DIR [--max-matches 32]
         [--workers 4] [--seed 1] [--shaping-mode legacy] --out OUT
  python tools/audit/rollouts.py credit --trace-dir DIR --checkpoint-dir DIR [--decision aim|attack] [--seeds 4]
         [--lambdas 0 0.5 0.8 0.9 0.95 0.99 1] [--max-matches 32] [--workers 4] [--seed 1] [--shaping-mode legacy] --out OUT
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import re
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from football_rl.config import DEFAULT_CONFIG  # noqa: E402
from football_rl.env import FootballParallelEnv  # noqa: E402
from tools.audit import census, harness  # noqa: E402

__all__ = [
    "AIM_BRANCHES",
    "AIM_HORIZONS",
    "AIM_MAX_DEFLECTION_DEG",
    "ATTACK_THIRD",
    "COMPONENT_KEYS",
    "CREDIT_GOAL_REWARD",
    "CREDIT_LAMBDAS",
    "CREDIT_PARTS",
    "CREDIT_SEEDS",
    "CREDIT_SEED_STRIDE",
    "CREDIT_STEPS",
    "GK_HORIZON_STEPS",
    "GK_INERT_ACTION",
    "GK_MAX_BALL_SPEED",
    "GK_MAX_DIST",
    "GK_MIN_DIST",
    "HOLD_HORIZONS",
    "HOLD_MIN_HOLD_S",
    "HOLD_OVERRIDE_STEPS",
    "KICK_OVERRIDE_STEPS",
    "POS_ERR_LIMIT",
    "SPACING_STEPS",
    "aim_steer",
    "aimp_power",
    "credit_branch_seed",
    "credit_gae",
    "credit_statistics",
    "credit_step_components",
    "decision_intent",
    "eval_gk_opportunity",
    "find_aim_points",
    "find_attack_points",
    "find_credit_points",
    "find_gk_opportunities",
    "find_hold_decisions",
    "gk_scripted_action",
    "main",
    "resimulate_match",
    "roll_aim_branches",
    "roll_credit_branch",
    "roll_credit_point",
    "roll_hold_branches",
    "run_aim",
    "run_credit",
    "run_gk",
    "run_hold",
    "summarize_aim",
    "summarize_credit",
    "summarize_gk",
    "summarize_hold",
]

POS_ERR_LIMIT = 1e-3
CHECKPOINT_MATCH_TOL = 1e-5
STEP_SECONDS = census.STEP_SECONDS
RL_HZ = float(DEFAULT_CONFIG.physics.rl_hz)
FIELD_WIDTH = census.FIELD_WIDTH
GOAL_Y = census.GOAL_Y
TEAM_SIZE = census.TEAM_SIZE

SPACING_SECONDS = 3.0
SPACING_STEPS = int(round(SPACING_SECONDS * RL_HZ))  # 45 decisions at 15 Hz

GK_HORIZON_STEPS = 15
GK_MIN_DIST = 40.0
GK_MAX_DIST = 140.0
GK_MAX_BALL_SPEED = 80.0
GK_INERT_ACTION = np.array([0.0, 0.0, -1.0, -1.0], dtype=np.float32)

HOLD_MIN_HOLD_S = float(DEFAULT_CONFIG.mechanics.pickup_kick_grace)
KICK_OVERRIDE_STEPS = 1
HOLD_OVERRIDE_STEPS = 15
HOLD_HORIZONS = (75, 225)
ONE_SECOND_STEPS = 15

KICK_INTENT_THR = float(DEFAULT_CONFIG.mechanics.kick_trigger_threshold)
FIRE_PROB_KICK = 0.5
FIRE_PROB_HOLD = 0.05
INTENT_BUCKETS = (
    ("intent_kick", "intent=kick"),
    ("intent_mid", "intent=mid"),
    ("intent_hold", "intent=hold"),
    ("def_intent_kick", "def&intent=kick"),
    ("def_intent_hold", "def&intent=hold"),
)

PRESSURE_NEAR = 70.0
PRESSURE_FAR = 120.0
PITCH_THIRD = FIELD_WIDTH / 3.0

AIM_MAX_DEFLECTION_DEG = float(DEFAULT_CONFIG.mechanics.kick_steer_max_deflection_deg)
AIM_HORIZONS = (15, 45, 75)
AIM_POSSESSION_STEPS = (15, 30)
AIM_FIRST_RECEIVER_STEPS = 45
AIM_ARRIVE_SPEED = 200.0
AIM_POWER_MARGIN = 1e-4
AIM_BRANCHES = ("policy", "aim", "aimp")
AIM_RECEIVERS = ("target", "teammate", "opponent", "none")
AIM_THIRDS = ("defensive", "middle", "attacking")
AIM_INTENTS = ("kick", "mid", "hold")
AIM_SCALAR_KEYS = (
    *(f"{m}_{h}" for h in AIM_HORIZONS for m in ("non_shaping_return", "full_return_bootstrap")),
    *(f"possession_{s}" for s in AIM_POSSESSION_STEPS),
    "goal_for",
    "goal_against",
)

COMPONENT_KEYS = (
    "turnover",
    "pass_bonus",
    "shot_on_target",
    "goal",
    "failed_tackle",
    "five_s_rule",
    "save",
    "other",
)

_MATCH_INDEX_RE = re.compile(r"^m(\d+)\.npz$")


# ----------------------------------------------------------------------------
# Trace, environment, and output helpers
# ----------------------------------------------------------------------------
def _trace_paths_with_indices(
    trace_dir: Path | str, max_matches: Optional[int] = None
) -> List[Tuple[int, Path]]:
    paths = census._trace_paths(trace_dir)
    if max_matches is not None:
        if max_matches <= 0:
            raise ValueError(f"max_matches must be positive, got {max_matches}")
        paths = paths[:max_matches]
    out: List[Tuple[int, Path]] = []
    for i, path in enumerate(paths):
        m = _MATCH_INDEX_RE.match(path.name)
        out.append((int(m.group(1)) if m else i, path))
    return out


def _resolve_trace_seed(trace_dir: Path | str, trace_seed: Optional[int]) -> int:
    if trace_seed is not None:
        return int(trace_seed)
    census_path = Path(trace_dir) / "census.json"
    if census_path.is_file():
        data = json.loads(census_path.read_text(encoding="utf-8"))
        if "seed" in data and data["seed"] is not None:
            return int(data["seed"])
    return 11


def _load_trace(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _build_env(shaping_mode: Optional[str] = None) -> FootballParallelEnv:
    cfg = DEFAULT_CONFIG
    if shaping_mode is not None:
        cfg = replace(cfg, rewards=replace(cfg.rewards, shaping_mode=shaping_mode))
    cfg = replace(
        cfg,
        replay=replace(
            cfg.replay,
            enable_recording=False,
            interest_threshold=harness.REPLAY_KEEP_THRESHOLD,
            save_dir=tempfile.gettempdir(),
        ),
    )
    return FootballParallelEnv(config=cfg)


def _validate_out_path(out: Path | str, protected: Sequence[Path | str]) -> Path:
    resolved = Path(out).resolve()
    harness.check_output_dir(resolved, protected)
    harness.check_output_dir(resolved.parent, protected)
    return resolved


def _write_outputs(out_path: Path, payload: Dict[str, Any], report_text: str) -> None:
    if out_path.suffix.lower() == ".json":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "rollouts.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        (out_path / "report.txt").write_text(report_text + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------
# Re-simulation
# ----------------------------------------------------------------------------
def resimulate_match(
    trace: Dict[str, np.ndarray],
    match_seed: int,
    *,
    target_steps: Iterable[int] = (),
    shaping_mode: Optional[str] = None,
) -> Tuple[bool, float, Dict[int, FootballParallelEnv]]:
    """Re-simulate a trace from `match_seed`, checking player and ball error < 1e-3 u at every step.

    Returns `(ok, max_pos_err, snapshots)` where `snapshots[t]` is a `copy.deepcopy(env)` taken
    immediately before step `t` for each `t` in `target_steps`. If any step exceeds `POS_ERR_LIMIT`,
    returns `(False, err, {})`.
    """
    act = np.asarray(trace["act"], dtype=np.float32)
    pos = np.asarray(trace["pos"], dtype=np.float64)
    ball = np.asarray(trace["ball"], dtype=np.float64)
    n_steps = int(act.shape[0])
    if n_steps <= 0 or pos.shape[0] < n_steps or ball.shape[0] < n_steps:
        return False, float("inf"), {}

    wanted = set(int(s) for s in target_steps)
    snapshots: Dict[int, FootballParallelEnv] = {}
    env = _build_env(shaping_mode=shaping_mode)
    env.reset(seed=int(match_seed))
    max_err = 0.0

    for t in range(n_steps):
        p_err = float(
            np.max(np.hypot(env.player_pos[:, 0] - pos[t, :, 0], env.player_pos[:, 1] - pos[t, :, 1]))
        )
        b_err = float(math.hypot(env.ball_pos[0] - ball[t, 0], env.ball_pos[1] - ball[t, 1]))
        err = max(p_err, b_err)
        if err > max_err:
            max_err = err
        if not math.isfinite(err) or err >= POS_ERR_LIMIT:
            return False, max_err, {}
        if t in wanted:
            snapshots[t] = copy.deepcopy(env)
        env.step_array(act[t])

    return True, max_err, snapshots


# ----------------------------------------------------------------------------
# GK opportunity rollouts
# ----------------------------------------------------------------------------
def find_gk_opportunities(trace: Dict[str, np.ndarray]) -> List[Tuple[int, int]]:
    """Find `(step, team)` opportunities: loose ball in own box, 40-140 u from GK, speed < 80 u/s, >= 3 s apart."""
    holder = np.asarray(trace["holder"], dtype=np.int64)
    n = int(holder.size)
    if n < GK_HORIZON_STEPS:
        return []
    pos = np.asarray(trace["pos"], dtype=np.float64)
    ball = np.asarray(trace["ball"], dtype=np.float64)
    bvel = np.asarray(trace["bvel"], dtype=np.float64)
    speed = np.hypot(bvel[:, 0], bvel[:, 1])
    side = np.asarray(trace["side"], dtype=np.int64)
    box = np.asarray(trace["box"], dtype=np.float64)

    max_start = n - GK_HORIZON_STEPS
    out: List[Tuple[int, int]] = []
    for team in (0, 1):
        keeper = TEAM_SIZE * team
        goal_x = np.where(side[:, team] == 0, 0.0, FIELD_WIDTH)
        radius = box[np.arange(n), side[:, team]]
        ball_goal = np.hypot(ball[:, 0] - goal_x, ball[:, 1] - GOAL_Y)
        keeper_ball = np.hypot(pos[:, keeper, 0] - ball[:, 0], pos[:, keeper, 1] - ball[:, 1])
        eligible = (
            (holder < 0)
            & (ball_goal <= radius)
            & (keeper_ball >= GK_MIN_DIST)
            & (keeper_ball <= GK_MAX_DIST)
            & (speed < GK_MAX_BALL_SPEED)
        )
        last_step = -SPACING_STEPS - 1
        for step in np.flatnonzero(eligible[: max_start + 1]).tolist():
            if step - last_step >= SPACING_STEPS:
                out.append((step, team))
                last_step = step
    out.sort()
    return out


def gk_scripted_action(env: FootballParallelEnv, keeper_idx: int) -> np.ndarray:
    """Scripted go-to-ball action: throttle 1, steer = clip(2 * bearing_error, -1, 1), kick -1, special -1."""
    dx = float(env.ball_pos[0] - env.player_pos[keeper_idx, 0])
    dy = float(env.ball_pos[1] - env.player_pos[keeper_idx, 1])
    bearing_err = (math.atan2(dy, dx) - float(env.player_angle[keeper_idx]) + math.pi) % (2.0 * math.pi) - math.pi
    steer = max(-1.0, min(1.0, 2.0 * bearing_err))
    return np.array([1.0, steer, -1.0, -1.0], dtype=np.float32)


def eval_gk_opportunity(
    snap: FootballParallelEnv,
    team: int,
    recorded_actions: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Run variants (a) recorded, (b) scripted go-to-ball, (c) inert for 15 decisions from `snap`."""
    keeper = TEAM_SIZE * team
    team_slice = slice(keeper, keeper + TEAM_SIZE)
    results: Dict[str, Dict[str, float]] = {}

    for variant in ("recorded", "scripted", "inert"):
        env = copy.deepcopy(snap)
        picked_up = False
        gk_rew = 0.0
        team_rew = 0.0
        for k in range(GK_HORIZON_STEPS):
            act = np.array(recorded_actions[k], dtype=np.float32, copy=True)
            if variant == "scripted":
                act[keeper] = gk_scripted_action(env, keeper)
            elif variant == "inert":
                act[keeper] = GK_INERT_ACTION
            _, _, rew, done, _ = env.step_array(act)
            if env.ball_holder == keeper:
                picked_up = True
            gk_rew += float(rew[keeper])
            team_rew += float(np.sum(rew[team_slice]))
            if done:
                break
        results[variant] = {
            "pickup": 1.0 if picked_up else 0.0,
            "gk_reward": gk_rew,
            "team_reward": team_rew,
        }

    results["diff_scripted_minus_recorded"] = {
        k: results["scripted"][k] - results["recorded"][k]
        for k in ("pickup", "gk_reward", "team_reward")
    }
    return results


def _gk_worker(task: Tuple[int, str, int, Optional[str]]) -> Dict[str, Any]:
    match_idx, path_str, trace_seed, shaping_mode = task
    trace = _load_trace(Path(path_str))
    opps = find_gk_opportunities(trace)
    target_steps = sorted({step for step, _ in opps})
    ok, max_err, snapshots = resimulate_match(
        trace,
        match_seed=trace_seed * 1000 + match_idx,
        target_steps=target_steps,
        shaping_mode=shaping_mode,
    )
    if not ok:
        return {"ok": False, "max_err": max_err, "records": []}
    act = np.asarray(trace["act"], dtype=np.float32)
    records = []
    for step, team in opps:
        rec = eval_gk_opportunity(snapshots[step], team, act[step : step + GK_HORIZON_STEPS])
        rec["match_idx"] = match_idx
        rec["step"] = step
        rec["team"] = team
        records.append(rec)
    return {"ok": True, "max_err": max_err, "records": records}


def _stat_dict(values: Iterable[float]) -> Dict[str, Any]:
    mean, se, n = census.mean_se(values)
    return {"mean": mean, "se": se, "n": n}


def _resolved_shaping_mode(shaping_mode: Optional[str]) -> str:
    return str(shaping_mode) if shaping_mode is not None else str(DEFAULT_CONFIG.rewards.shaping_mode)


def summarize_gk(
    records: Sequence[Dict[str, Any]],
    n_matches_total: int,
    n_matches_ok: int,
    n_matches_skipped: int,
    max_pos_err: Optional[float],
    *,
    shaping_mode: str = str(DEFAULT_CONFIG.rewards.shaping_mode),
    gamma: float = float(DEFAULT_CONFIG.ppo.gamma),
) -> Dict[str, Any]:
    variants = {}
    for group in ("recorded", "scripted", "inert", "diff_scripted_minus_recorded"):
        variants[group] = {
            metric: _stat_dict(r[group][metric] for r in records)
            for metric in ("pickup", "gk_reward", "team_reward")
        }
    return {
        "mode": "gk",
        "shaping_mode": shaping_mode,
        "gamma": float(gamma),
        "n_matches_total": n_matches_total,
        "n_matches_ok": n_matches_ok,
        "n_matches_skipped": n_matches_skipped,
        "max_pos_err": max_pos_err,
        "n_opportunities": len(records),
        "variants": variants,
    }


def run_gk(
    *,
    trace_dir: Path | str,
    trace_seed: Optional[int] = None,
    max_matches: Optional[int] = None,
    workers: int = 1,
    shaping_mode: Optional[str] = None,
) -> Dict[str, Any]:
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    seed_val = _resolve_trace_seed(trace_dir, trace_seed)
    items = _trace_paths_with_indices(trace_dir, max_matches=max_matches)
    tasks = [(idx, str(path), seed_val, shaping_mode) for idx, path in items]

    if workers == 1 or len(tasks) == 1:
        outputs = [_gk_worker(t) for t in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(workers, len(tasks))) as pool:
            outputs = list(pool.map(_gk_worker, tasks, chunksize=1))

    records: List[Dict[str, Any]] = []
    n_ok = 0
    n_skipped = 0
    max_err: Optional[float] = None
    for res in outputs:
        if res["ok"]:
            n_ok += 1
            records.extend(res["records"])
            err = float(res["max_err"])
            max_err = err if max_err is None else max(max_err, err)
        else:
            n_skipped += 1
    return summarize_gk(
        records,
        len(tasks),
        n_ok,
        n_skipped,
        max_err,
        shaping_mode=_resolved_shaping_mode(shaping_mode),
        gamma=float(DEFAULT_CONFIG.ppo.gamma),
    )


def format_gk_report(summary: Dict[str, Any], title: str = "") -> str:
    header = f"===== rollouts gk {title}".strip()
    lines = [
        f"{header}  matches ok={summary['n_matches_ok']}/{summary['n_matches_total']} "
        f"skipped={summary['n_matches_skipped']} n_opp={summary['n_opportunities']} "
        f"max_pos_err={summary['max_pos_err'] if summary['max_pos_err'] is not None else '-'}"
    ]
    labels = (
        ("recorded", "recorded (a)"),
        ("scripted", "scripted (b)"),
        ("inert", "inert    (c)"),
        ("diff_scripted_minus_recorded", "diff (b)-(a)"),
    )
    for key, label in labels:
        v = summary["variants"][key]
        lines.append(
            f"  {label}: pickup={census._fmt_d(v['pickup'], 3)} | "
            f"gk_rew={census._fmt_d(v['gk_reward'], 3)} | "
            f"team_rew={census._fmt_d(v['team_reward'], 3)}"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Hold-vs-kick rollouts
# ----------------------------------------------------------------------------
def find_hold_decisions(trace: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """Eligible outfield holding decisions (`hold_t >= 0.08`, >= 3 s apart) with pressure and third."""
    holder = np.asarray(trace["holder"], dtype=np.int64)
    hold_t = np.asarray(trace["hold_t"], dtype=np.float64)
    pos = np.asarray(trace["pos"], dtype=np.float64)
    ball = np.asarray(trace["ball"], dtype=np.float64)
    side = np.asarray(trace["side"], dtype=np.int64)
    n = int(holder.size)

    eligible = (holder >= 0) & (holder % TEAM_SIZE != 0) & (hold_t >= HOLD_MIN_HOLD_S)
    points: List[Dict[str, Any]] = []
    last_step = -SPACING_STEPS - 1
    for step in np.flatnonzero(eligible).tolist():
        if step - last_step < SPACING_STEPS:
            continue
        last_step = step
        h = int(holder[step])
        team = h // TEAM_SIZE
        opp_slice = slice(TEAM_SIZE, 2 * TEAM_SIZE) if team == 0 else slice(0, TEAM_SIZE)
        d_opp = float(
            np.min(np.hypot(pos[step, opp_slice, 0] - pos[step, h, 0], pos[step, opp_slice, 1] - pos[step, h, 1]))
        )
        if d_opp < PRESSURE_NEAR:
            pressure = "lt70"
        elif d_opp >= PRESSURE_FAR:
            pressure = "ge120"
        else:
            pressure = "mid"
        canon_x = float(ball[step, 0]) if int(side[step, team]) == 0 else (FIELD_WIDTH - float(ball[step, 0]))
        if canon_x < PITCH_THIRD:
            third = "defensive"
        elif canon_x < 2.0 * PITCH_THIRD:
            third = "middle"
        else:
            third = "attacking"
        points.append({
            "step": step,
            "holder": h,
            "team": team,
            "nearest_opp": d_opp,
            "pressure": pressure,
            "third": third,
        })
    return points


def decision_intent(trace: Dict[str, np.ndarray], step: int, holder: int) -> str:
    """Kick intent of `holder` at `step` from its recorded kick mean: `kick`, `mid` (hybrid5 and hybrid6 only) or `hold`."""
    mean_kick = float(np.asarray(trace["mean"])[step, holder, 2])
    head = np.asarray(trace["head"], dtype=np.int64) if "head" in trace else np.zeros(2, dtype=np.int64)
    if int(head[holder // TEAM_SIZE]) in (1, 2):
        p = (mean_kick + 1.0) / 2.0
        if p > FIRE_PROB_KICK:
            return "kick"
        return "hold" if p < FIRE_PROB_HOLD else "mid"
    return "kick" if mean_kick > KICK_INTENT_THR else "hold"


def _non_shaping_rewards(env: Any, rew: np.ndarray) -> np.ndarray:
    """Step reward minus `last_shaping_rewards` when exposed, else breakdown `r_sparse`."""
    if hasattr(env, "last_shaping_rewards"):
        return np.asarray(rew, dtype=np.float64) - np.asarray(env.last_shaping_rewards, dtype=np.float64)
    bd = getattr(env, "last_rew_breakdown", None)
    if bd is not None:
        return np.array([float(row["r_sparse"]) for row in bd], dtype=np.float64)
    raise AttributeError("environment exposes neither last_shaping_rewards nor last_rew_breakdown")


def _snapshot_step_state(env: Any) -> Dict[str, Any]:
    stats = env.stats
    return {
        "score": (int(stats["score"][0]), int(stats["score"][1])),
        "own_goals": int(stats.get("own_goals", 0)),
        "team_assists": (
            int(stats.get("team_assists", (0, 0))[0]),
            int(stats.get("team_assists", (0, 0))[1]),
        ),
        "field_5s": int(stats.get("field_5s_violations", 0)),
        "keeper_5s": int(stats.get("keeper_5s_violations", 0)),
        "failed_tackles": int(stats.get("failed_tackles", 0)),
        "shots": (int(stats.get("shots", (0, 0))[0]), int(stats.get("shots", (0, 0))[1])),
        "team_passes": (int(stats.get("team_passes", (0, 0))[0]), int(stats.get("team_passes", (0, 0))[1])),
        "team_saves": (int(stats.get("team_saves", (0, 0))[0]), int(stats.get("team_saves", (0, 0))[1])),
        "holder": env.ball_holder,
        "turnover_t": float(env.last_turnover_reward_time),
        "tackle_cd": np.asarray(env.tackle_cooldown, dtype=np.float64).copy(),
        "stun": np.asarray(env.stun_timer, dtype=np.float64).copy(),
    }


def _attribute_sparse_components(
    pre: Dict[str, Any],
    env: Any,
    act: np.ndarray,
    r_ns: np.ndarray,
    team: int,
) -> Dict[str, float]:
    """Decompose `team`'s mean non-shaping reward for one step across sparse categories."""
    opp = 1 - team
    team_mean_ns = float(np.mean(r_ns[team * TEAM_SIZE : (team + 1) * TEAM_SIZE]))
    stats = env.stats
    rew_cfg = env.cfg.rewards
    mech_cfg = env.cfg.mechanics
    dt = float(env.cfg.physics.dt)

    # 1. Goal
    goal = 0.0
    d_score_for = int(stats["score"][team]) - pre["score"][team]
    d_score_opp = int(stats["score"][opp]) - pre["score"][opp]
    d_own = int(stats.get("own_goals", 0)) - pre["own_goals"]
    d_assist_for = int(stats.get("team_assists", (0, 0))[team]) - pre["team_assists"][team]
    if d_score_for > 0:
        goal += d_score_for * float(rew_cfg.goal_team_shared)
        if d_own == 0:
            goal += (float(rew_cfg.goal_scorer_ind) / TEAM_SIZE) + d_assist_for * (
                float(rew_cfg.goal_assist_ind) / TEAM_SIZE
            )
    if d_score_opp > 0:
        goal += d_score_opp * float(rew_cfg.goal_concede_team_shared)
        if d_own > 0:
            goal += float(rew_cfg.own_goal_ind) / TEAM_SIZE

    # 2. 5-second violation
    five_s = 0.0
    d_f5 = int(stats.get("field_5s_violations", 0)) - pre["field_5s"]
    d_k5 = int(stats.get("keeper_5s_violations", 0)) - pre["keeper_5s"]
    if d_f5 > 0 and pre["holder"] is not None and (pre["holder"] // TEAM_SIZE) == team:
        five_s += (float(rew_cfg.field_5s_violation_ind) / TEAM_SIZE) + float(rew_cfg.field_5s_violation_team)
    if d_k5 > 0 and pre["holder"] is not None and (pre["holder"] // TEAM_SIZE) == team:
        five_s += float(rew_cfg.keeper_5s_violation_ind) / TEAM_SIZE

    # 3. Failed tackle
    failed_tackle = 0.0
    d_ft = int(stats.get("failed_tackles", 0)) - pre["failed_tackles"]
    if d_ft > 0:
        tackle_cd = np.asarray(env.tackle_cooldown, dtype=np.float64)
        stun = np.asarray(env.stun_timer, dtype=np.float64)
        n_ft_team = 0
        for i in range(team * TEAM_SIZE, (team + 1) * TEAM_SIZE):
            if (
                i % TEAM_SIZE != 0
                and act[i, 3] > mech_cfg.special_trigger_threshold
                and pre["tackle_cd"][i] <= 4.0 * dt + 1e-9
                and tackle_cd[i] > 0.5
                and 0.5 < (stun[i] - pre["stun"][i]) <= mech_cfg.tackle_miss_stun_time + 1e-6
            ):
                n_ft_team += 1
        failed_tackle = n_ft_team * (float(rew_cfg.failed_tackle_ind) / TEAM_SIZE)

    # 4. Shot on target (if not immediately clawed back by a save in the same RL step)
    sot = 0.0
    d_shots_for = int(stats.get("shots", (0, 0))[team]) - pre["shots"][team]
    d_shots_opp = int(stats.get("shots", (0, 0))[opp]) - pre["shots"][opp]
    d_saves_for = int(stats.get("team_saves", (0, 0))[team]) - pre["team_saves"][team]
    d_saves_opp = int(stats.get("team_saves", (0, 0))[opp]) - pre["team_saves"][opp]
    if d_shots_for == 1 and d_shots_opp == 0 and d_saves_opp == 0:
        sot = (float(getattr(env, "last_shot_sot_reward", 0.0)) / TEAM_SIZE) + float(
            getattr(env, "last_shot_team_reward", 0.0)
        )

    # 5. Remainder: turnover, pass_bonus, save, other
    rem = team_mean_ns - (goal + five_s + failed_tackle + sot)
    pass_active = int(stats.get("team_passes", (0, 0))[team]) > pre["team_passes"][team]
    save_active = (d_saves_for > 0) or (d_saves_opp > 0)
    stun = np.asarray(env.stun_timer, dtype=np.float64)
    strip_stun = float(mech_cfg.keeper_strip_stun_duration)
    gk_strip_victim = next(
        (
            i
            for i in range(2 * TEAM_SIZE)
            if (i % TEAM_SIZE != 0)
            and 0.25 * strip_stun < (stun[i] - pre["stun"][i])
            and stun[i] <= strip_stun + 1e-6
        ),
        None,
    )
    gk_strip = (
        (
            pre["holder"] is not None
            and env.ball_holder in (0, TEAM_SIZE)
            and (env.ball_holder // TEAM_SIZE) != (pre["holder"] // TEAM_SIZE)
        )
        or gk_strip_victim is not None
    )
    turnover_active = (float(env.last_turnover_reward_time) != pre["turnover_t"]) or gk_strip

    turnover = 0.0
    pass_bonus = 0.0
    save = 0.0
    other = 0.0
    if abs(rem) > 1e-9:
        n_active = int(pass_active) + int(save_active) + int(turnover_active)
        if n_active == 1:
            if pass_active:
                pass_bonus = rem
            elif save_active:
                save = rem
            else:
                turnover = rem
        elif turnover_active and pass_active and not save_active:
            # Turnover magnitude is fixed zero-sum (+-0.316667); attribute remainder to pass_bonus.
            unit_to = float(rew_cfg.successful_tackle_tackler_ind) / TEAM_SIZE + float(rew_cfg.successful_tackle_team)
            if env.ball_holder is not None:
                won = (env.ball_holder // TEAM_SIZE) == team
            elif gk_strip_victim is not None:
                won = (gk_strip_victim // TEAM_SIZE) != team
            else:
                won = env.last_ball_team == team
            sign = 1.0 if won else -1.0
            turnover = sign * unit_to
            pass_bonus = rem - turnover
        elif save_active:
            save = rem
        elif pass_active:
            pass_bonus = rem
        elif turnover_active:
            turnover = rem
        else:
            other = rem

    return {
        "turnover": turnover,
        "pass_bonus": pass_bonus,
        "shot_on_target": sot,
        "goal": goal,
        "failed_tackle": failed_tackle,
        "five_s_rule": five_s,
        "save": save,
        "other": other,
    }


def _team_action_and_value(
    policy: Optional[harness.TeamPolicy],
    env: Any,
    team: int,
    obs: np.ndarray,
    cen: np.ndarray,
    mode: str,
    sigma_scale: float,
    gen: torch.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if policy is None:
        from football_rl.baselines import scripted_actions

        return scripted_actions(env, team), np.zeros(TEAM_SIZE, dtype=np.float32)
    sl = slice(team * TEAM_SIZE, (team + 1) * TEAM_SIZE)
    act, val, _ = policy.act(obs[sl], cen[sl], mode, sigma_scale, gen)
    return act, val


def _roll_single_branch(
    snap: Any,
    *,
    holder_player: int,
    team_a: Optional[harness.TeamPolicy],
    team_b: Optional[harness.TeamPolicy],
    branch_seed: int,
    mode: str,
    sigma_scale: float,
    horizons: Sequence[int],
    override_steps: int,
    override_value: Optional[float],
    after_step: Optional[Callable[[int, Any], None]] = None,
) -> Dict[int, Dict[str, Any]]:
    env = copy.deepcopy(snap)
    holder_team = holder_player // TEAM_SIZE
    opp_team = 1 - holder_team
    sl = slice(holder_team * TEAM_SIZE, (holder_team + 1) * TEAM_SIZE)

    gamma = float(env.cfg.ppo.gamma)
    max_h = max(horizons)
    sorted_h = sorted(set(int(h) for h in horizons))

    gen = torch.Generator().manual_seed(int(branch_seed))
    obs = env._get_observations_array()
    cen = env.get_centralized_state()
    start_score = (int(env.stats["score"][0]), int(env.stats["score"][1]))

    disc_ns = 0.0
    disc_full = 0.0
    comp_sums = {k: 0.0 for k in COMPONENT_KEYS}
    team_ball_1s: Optional[float] = None
    horizon_out: Dict[int, Dict[str, Any]] = {}
    done_at: Optional[int] = None

    for k in range(max_h):
        act_a, val_a = _team_action_and_value(team_a, env, 0, obs, cen, mode, sigma_scale, gen)
        act_b, val_b = _team_action_and_value(team_b, env, 1, obs, cen, mode, sigma_scale, gen)
        val = np.concatenate([val_a, val_b])

        # Record critic bootstrap at any completed horizon k before applying step k's override
        if k in horizon_out:
            raise RuntimeError(f"duplicate horizon {k}")
        if k in sorted_h:
            boot = (gamma ** k) * float(np.mean(val[sl]))
            horizon_out[k] = {
                "non_shaping_return": disc_ns,
                "full_return_bootstrap": disc_full + boot,
                "p_team_ball_1s": float(team_ball_1s) if team_ball_1s is not None else 0.0,
                "p_goal_for": 1.0 if int(env.stats["score"][holder_team]) > start_score[holder_team] else 0.0,
                "p_goal_against": 1.0 if int(env.stats["score"][opp_team]) > start_score[opp_team] else 0.0,
                "components": dict(comp_sums),
            }

        act = np.concatenate([act_a, act_b])
        if override_value is not None and k < override_steps:
            act[holder_player, 2] = float(override_value)

        pre = _snapshot_step_state(env)
        obs, cen, rew, done, _ = env.step_array(act, val)
        if after_step is not None:
            after_step(k, env)

        r_ns = _non_shaping_rewards(env, rew)
        w = gamma ** k
        disc_ns += w * float(np.mean(r_ns[sl]))
        disc_full += w * float(np.mean(np.asarray(rew, dtype=np.float64)[sl]))

        step_comps = _attribute_sparse_components(pre, env, act, r_ns, holder_team)
        for ck, cv in step_comps.items():
            comp_sums[ck] += w * cv

        if k == ONE_SECOND_STEPS - 1 or (done and team_ball_1s is None):
            team_ball_1s = 1.0 if (env.ball_holder is not None and (env.ball_holder // TEAM_SIZE) == holder_team) else 0.0

        if done:
            done_at = k + 1
            break

    # Finalise horizons >= steps actually run
    final_steps = done_at if done_at is not None else max_h
    if done_at is None:
        _, val_a_end = _team_action_and_value(team_a, env, 0, obs, cen, mode, sigma_scale, gen)
        _, val_b_end = _team_action_and_value(team_b, env, 1, obs, cen, mode, sigma_scale, gen)
        val_end = np.concatenate([val_a_end, val_b_end])
        end_boot_val = float(np.mean(val_end[sl]))
    else:
        end_boot_val = 0.0

    goal_for_final = 1.0 if int(env.stats["score"][holder_team]) > start_score[holder_team] else 0.0
    goal_against_final = 1.0 if int(env.stats["score"][opp_team]) > start_score[opp_team] else 0.0
    ball_1s_final = float(team_ball_1s) if team_ball_1s is not None else 0.0

    for h in sorted_h:
        if h not in horizon_out:
            boot = (gamma ** final_steps) * end_boot_val if done_at is None else 0.0
            horizon_out[h] = {
                "non_shaping_return": disc_ns,
                "full_return_bootstrap": disc_full + boot,
                "p_team_ball_1s": ball_1s_final,
                "p_goal_for": goal_for_final,
                "p_goal_against": goal_against_final,
                "components": dict(comp_sums),
            }
    return horizon_out


def roll_hold_branches(
    snap: Any,
    *,
    holder_player: int,
    team_a: Optional[harness.TeamPolicy],
    team_b: Optional[harness.TeamPolicy],
    branch_seed: int,
    mode: str = "sampled",
    sigma_scale: float = 1.0,
    horizons: Sequence[int] = HOLD_HORIZONS,
    kick_steps: int = KICK_OVERRIDE_STEPS,
    hold_steps: int = HOLD_OVERRIDE_STEPS,
    identical_override: bool = False,
    after_kick_step: Optional[Callable[[int, Any], None]] = None,
    after_hold_step: Optional[Callable[[int, Any], None]] = None,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Roll paired KICK and HOLD branches from `snap` with identical generator seeds."""
    kick_val = None if identical_override else 1.0
    kick_n = 0 if identical_override else kick_steps
    hold_val = None if identical_override else -1.0
    hold_n = 0 if identical_override else hold_steps

    kick_res = _roll_single_branch(
        snap,
        holder_player=holder_player,
        team_a=team_a,
        team_b=team_b,
        branch_seed=branch_seed,
        mode=mode,
        sigma_scale=sigma_scale,
        horizons=horizons,
        override_steps=kick_n,
        override_value=kick_val,
        after_step=after_kick_step,
    )
    hold_res = _roll_single_branch(
        snap,
        holder_player=holder_player,
        team_a=team_a,
        team_b=team_b,
        branch_seed=branch_seed,
        mode=mode,
        sigma_scale=sigma_scale,
        horizons=horizons,
        override_steps=hold_n,
        override_value=hold_val,
        after_step=after_hold_step,
    )

    diff_res: Dict[int, Dict[str, Any]] = {}
    for h in sorted(set(int(x) for x in horizons)):
        kh, hh = kick_res[h], hold_res[h]
        diff_res[h] = {
            metric: hh[metric] - kh[metric]
            for metric in (
                "non_shaping_return",
                "full_return_bootstrap",
                "p_team_ball_1s",
                "p_goal_for",
                "p_goal_against",
            )
        }
        diff_res[h]["components"] = {
            ck: hh["components"][ck] - kh["components"][ck] for ck in COMPONENT_KEYS
        }

    return {"kick": kick_res, "hold": hold_res, "diff": diff_res}


_HOLD_POLICIES: Optional[Tuple[Optional[harness.TeamPolicy], Optional[harness.TeamPolicy]]] = None
_HOLD_SPEC_CFG: Optional[Tuple[str, float, Optional[str]]] = None


def _init_hold_worker(spec: harness.EvalSpec, shaping_mode: Optional[str]) -> None:
    global _HOLD_POLICIES, _HOLD_SPEC_CFG
    torch.set_num_threads(1)
    spec.validate()
    _HOLD_POLICIES = harness._load_policies(spec)
    _HOLD_SPEC_CFG = (spec.mode, float(spec.sigma_scale), shaping_mode)


def _team_mean_and_value(
    policy: Optional[harness.TeamPolicy],
    env: Any,
    team: int,
    obs: np.ndarray,
    cen: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if policy is None:
        from football_rl.baselines import scripted_actions

        return scripted_actions(env, team), np.zeros(TEAM_SIZE, dtype=np.float32)
    sl = slice(team * TEAM_SIZE, (team + 1) * TEAM_SIZE)
    _, val, mean = policy.act(obs[sl], cen[sl], "greedy", 0.0, torch.Generator())
    return mean, val


def _verify_checkpoint_matches_trace(
    match_idx: int,
    step: int,
    snap: FootballParallelEnv,
    trace: Dict[str, np.ndarray],
    team_a: Optional[harness.TeamPolicy],
    team_b: Optional[harness.TeamPolicy],
) -> None:
    obs = snap._get_observations_array()
    cen = snap.get_centralized_state()
    mean_a, val_a = _team_mean_and_value(team_a, snap, 0, obs, cen)
    mean_b, val_b = _team_mean_and_value(team_b, snap, 1, obs, cen)
    pred_mean = np.clip(np.concatenate([mean_a, mean_b]), -1.0, 1.0).astype(np.float64)
    pred_val = np.concatenate([val_a, val_b]).astype(np.float64)
    trace_mean = np.asarray(trace["mean"][step], dtype=np.float64)
    trace_val = np.asarray(trace["val"][step], dtype=np.float64)
    val_diff = np.abs(pred_val - trace_val)
    mean_diff = np.max(np.abs(pred_mean - trace_mean), axis=1)
    seat_diff = np.maximum(val_diff, mean_diff)
    if not np.all(np.isfinite(seat_diff)):
        bad = int(np.flatnonzero(~np.isfinite(seat_diff))[0])
        raise ValueError(
            f"checkpoint mismatch at match {match_idx} step {step} seat {bad}: "
            f"max_diff=non-finite > {CHECKPOINT_MATCH_TOL}"
        )
    max_seat = int(np.argmax(seat_diff))
    max_diff = float(seat_diff[max_seat])
    if max_diff > CHECKPOINT_MATCH_TOL:
        raise ValueError(
            f"checkpoint mismatch at match {match_idx} step {step} seat {max_seat}: "
            f"max_diff={max_diff:.6g} > {CHECKPOINT_MATCH_TOL}"
        )


def _hold_worker(task: Tuple[int, str, int, int, int]) -> Dict[str, Any]:
    match_idx, path_str, trace_seed, rollout_seed, hold_steps = task
    assert _HOLD_POLICIES is not None and _HOLD_SPEC_CFG is not None
    team_a, team_b = _HOLD_POLICIES
    mode, sigma_scale, shaping_mode = _HOLD_SPEC_CFG

    trace = _load_trace(Path(path_str))
    pts = find_hold_decisions(trace)
    target_steps = sorted({0, *(p["step"] for p in pts)})
    ok, max_err, snapshots = resimulate_match(
        trace,
        match_seed=trace_seed * 1000 + match_idx,
        target_steps=target_steps,
        shaping_mode=shaping_mode,
    )
    if not ok:
        return {"ok": False, "max_err": max_err, "records": []}

    _verify_checkpoint_matches_trace(match_idx, 0, snapshots[0], trace, team_a, team_b)

    records = []
    for p in pts:
        step = p["step"]
        branch_seed = (rollout_seed * 1_000_003 + match_idx * 10_000 + step) & 0x7FFF_FFFF
        res = roll_hold_branches(
            snapshots[step],
            holder_player=p["holder"],
            team_a=team_a,
            team_b=team_b,
            branch_seed=branch_seed,
            mode=mode,
            sigma_scale=sigma_scale,
            horizons=HOLD_HORIZONS,
            hold_steps=hold_steps,
        )
        records.append({
            "match_idx": match_idx,
            "step": step,
            "holder": p["holder"],
            "team": p["team"],
            "nearest_opp": p["nearest_opp"],
            "pressure": p["pressure"],
            "third": p["third"],
            "intent": decision_intent(trace, step, p["holder"]),
            "rollout": res,
        })
    return {"ok": True, "max_err": max_err, "records": records}


def _summarize_hold_bucket(records: Sequence[Dict[str, Any]], h: int) -> Dict[str, Any]:
    scalar_metrics = (
        "non_shaping_return",
        "full_return_bootstrap",
        "p_team_ball_1s",
        "p_goal_for",
        "p_goal_against",
    )
    out: Dict[str, Any] = {"n": len(records)}
    for m in scalar_metrics:
        out[m] = _stat_dict(r["rollout"]["diff"][h][m] for r in records)
    out["components"] = {
        ck: _stat_dict(r["rollout"]["diff"][h]["components"][ck] for r in records)
        for ck in COMPONENT_KEYS
    }
    return out


def summarize_hold(
    records: Sequence[Dict[str, Any]],
    n_matches_total: int,
    n_matches_ok: int,
    n_matches_skipped: int,
    max_pos_err: Optional[float],
    *,
    seed: int,
    shaping_mode: str = str(DEFAULT_CONFIG.rewards.shaping_mode),
    gamma: float = float(DEFAULT_CONFIG.ppo.gamma),
    hold_steps: int = HOLD_OVERRIDE_STEPS,
) -> Dict[str, Any]:
    buckets = {
        "overall": lambda r: True,
        "pressure_lt70": lambda r: r["pressure"] == "lt70",
        "pressure_ge120": lambda r: r["pressure"] == "ge120",
        "pressure_mid": lambda r: r["pressure"] == "mid",
        "third_defensive": lambda r: r["third"] == "defensive",
        "third_middle": lambda r: r["third"] == "middle",
        "third_attacking": lambda r: r["third"] == "attacking",
        "intent_kick": lambda r: r.get("intent") == "kick",
        "intent_mid": lambda r: r.get("intent") == "mid",
        "intent_hold": lambda r: r.get("intent") == "hold",
        "def_intent_kick": lambda r: r["third"] == "defensive" and r.get("intent") == "kick",
        "def_intent_hold": lambda r: r["third"] == "defensive" and r.get("intent") == "hold",
    }
    horizons_out: Dict[str, Any] = {}
    for h in HOLD_HORIZONS:
        horizons_out[str(h)] = {
            b_name: _summarize_hold_bucket([r for r in records if pred(r)], h)
            for b_name, pred in buckets.items()
        }
    return {
        "mode": "hold",
        "seed": seed,
        "shaping_mode": shaping_mode,
        "gamma": float(gamma),
        "n_matches_total": n_matches_total,
        "n_matches_ok": n_matches_ok,
        "n_matches_skipped": n_matches_skipped,
        "max_pos_err": max_pos_err,
        "n_decisions": len(records),
        "hold_steps": int(hold_steps),
        "horizons": horizons_out,
    }


def run_hold(
    *,
    trace_dir: Path | str,
    checkpoint_dir: Optional[Path | str],
    opponent: str = "self",
    swap: bool = False,
    mode: str = "sampled",
    sigma_scale: float = 1.0,
    trace_seed: Optional[int] = None,
    seed: int = 1,
    max_matches: Optional[int] = None,
    workers: int = 1,
    shaping_mode: Optional[str] = None,
    hold_steps: int = HOLD_OVERRIDE_STEPS,
) -> Dict[str, Any]:
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    if int(hold_steps) < 1:
        raise ValueError(f"hold_steps must be >= 1, got {hold_steps}")
    spec = harness.EvalSpec(
        checkpoint_dir=str(checkpoint_dir) if checkpoint_dir is not None else None,
        opponent=opponent,
        mode=mode,
        n=1,
        seed=int(seed),
        sigma_scale=float(sigma_scale),
        swap=bool(swap),
        workers=int(workers),
    )
    spec.validate()
    t_seed = _resolve_trace_seed(trace_dir, trace_seed)
    items = _trace_paths_with_indices(trace_dir, max_matches=max_matches)
    tasks = [(idx, str(path), t_seed, int(seed), int(hold_steps)) for idx, path in items]

    if workers == 1 or len(tasks) == 1:
        prev_threads = torch.get_num_threads()
        try:
            _init_hold_worker(spec, shaping_mode)
            outputs = [_hold_worker(t) for t in tasks]
        finally:
            torch.set_num_threads(prev_threads)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(workers, len(tasks)), initializer=_init_hold_worker, initargs=(spec, shaping_mode)) as pool:
            outputs = list(pool.map(_hold_worker, tasks, chunksize=1))

    records: List[Dict[str, Any]] = []
    n_ok = 0
    n_skipped = 0
    max_err: Optional[float] = None
    for res in outputs:
        if res["ok"]:
            n_ok += 1
            records.extend(res["records"])
            err = float(res["max_err"])
            max_err = err if max_err is None else max(max_err, err)
        else:
            n_skipped += 1
    return summarize_hold(
        records,
        len(tasks),
        n_ok,
        n_skipped,
        max_err,
        seed=int(seed),
        shaping_mode=_resolved_shaping_mode(shaping_mode),
        gamma=float(DEFAULT_CONFIG.ppo.gamma),
        hold_steps=int(hold_steps),
    )


def format_hold_report(summary: Dict[str, Any], title: str = "") -> str:
    header = f"===== rollouts hold {title}".strip()
    lines = [
        f"{header}  matches ok={summary['n_matches_ok']}/{summary['n_matches_total']} "
        f"skipped={summary['n_matches_skipped']} n_decisions={summary['n_decisions']} "
        f"seed={summary['seed']} max_pos_err={summary['max_pos_err'] if summary['max_pos_err'] is not None else '-'}"
    ]
    bucket_order = (
        ("overall", "overall"),
        ("pressure_lt70", "opp < 70 u"),
        ("pressure_ge120", "opp >= 120 u"),
        ("third_defensive", "defensive 1/3"),
        ("third_middle", "middle 1/3"),
        ("third_attacking", "attacking 1/3"),
        *INTENT_BUCKETS,
    )
    for h in HOLD_HORIZONS:
        h_data = summary["horizons"][str(h)]
        lines.append(f"  -- horizon H={h} ({h * STEP_SECONDS:.0f} s), paired HOLD - KICK --")
        for b_key, b_label in bucket_order:
            b = h_data[b_key]
            comps = ", ".join(f"{ck}={census._fmt_d(b['components'][ck], 4)}" for ck in COMPONENT_KEYS)
            lines.append(
                f"    {b_label:15s}: non_shaping={census._fmt_d(b['non_shaping_return'], 4)} | "
                f"full_boot={census._fmt_d(b['full_return_bootstrap'], 4)} | "
                f"ball_1s={census._fmt_d(b['p_team_ball_1s'], 3)} | "
                f"gf={census._fmt_d(b['p_goal_for'], 4)} | ga={census._fmt_d(b['p_goal_against'], 4)}"
            )
            lines.append(f"      components: {comps}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Aim counterfactual at real kicks
# ----------------------------------------------------------------------------
def _wrap_angle(a: float) -> float:
    """Angle wrapped to (-pi, pi]."""
    return math.pi - (math.pi - a) % (2.0 * math.pi)


def _third(trace: Dict[str, np.ndarray], step: int, team: int) -> str:
    x = float(np.asarray(trace["ball"])[step, 0])
    canon_x = x if int(np.asarray(trace["side"])[step, team]) == 0 else FIELD_WIDTH - x
    if canon_x < PITCH_THIRD:
        return "defensive"
    return "middle" if canon_x < 2.0 * PITCH_THIRD else "attacking"


def find_aim_points(trace: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """Kicks with an open teammate in the kickable arc, >= 45 decisions apart (see `aim` in the module docstring)."""
    holder = np.asarray(trace["holder"], dtype=np.int64)
    n = int(holder.size)
    pos = np.asarray(trace["pos"], dtype=np.float64)
    ball = np.asarray(trace["ball"], dtype=np.float64)
    ang = np.asarray(trace["ang"], dtype=np.float64)
    limit = math.radians(AIM_MAX_DEFLECTION_DEG)
    points: List[Dict[str, Any]] = []
    last_step = -SPACING_STEPS - 1
    for rec in census.pass_options(trace):
        if not rec["open"] or not rec["has_direction"]:
            continue
        step, kicker = rec["step"], rec["player"]
        if int(holder[step]) != kicker or step - last_step < SPACING_STEPS or step + 2 >= n:
            continue
        facing = float(ang[step, kicker])
        best: Optional[Tuple[int, float, float]] = None
        for mate in rec["open"]:
            dx, dy = pos[step, mate] - ball[step]
            deflection = _wrap_angle(math.atan2(dy, dx) - facing)
            if abs(deflection) <= limit and (best is None or abs(deflection) < abs(best[1])):
                best = (int(mate), deflection, math.hypot(dx, dy))
        if best is None:
            continue
        last_step = step
        team = kicker // TEAM_SIZE
        points.append({
            "step": step,
            "kicker": kicker,
            "team": team,
            "target": best[0],
            "deflection": best[1],
            "target_dist": best[2],
            "third": _third(trace, step, team),
            "intent": decision_intent(trace, step, kicker),
            "already": bool(rec["aimed_open"]),
        })
    return points


def aim_steer(deflection: float, max_deflection_deg: float = AIM_MAX_DEFLECTION_DEG) -> float:
    """Steer that turns a kick by `deflection` rad from the facing: clip(deflection / max deflection, -1, 1)."""
    return float(min(1.0, max(-1.0, deflection / math.radians(max_deflection_deg))))


def aimp_power(
    dist: float,
    v_max: float,
    *,
    v_min: float = float(DEFAULT_CONFIG.physics.ball_min_kick_speed),
    decel: float = float(DEFAULT_CONFIG.physics.ball_friction_decel),
    threshold: float = KICK_INTENT_THR,
    arrive_speed: float = AIM_ARRIVE_SPEED,
) -> Tuple[float, float, float]:
    """`(v0, pn, power)`: the launch speed reaching `dist` at `arrive_speed` under linear friction (capped at `v_max`),
    its normalised power and the env kick column that produces it."""
    v0 = min(float(v_max), math.sqrt(arrive_speed ** 2 + 2.0 * decel * float(dist)))
    pn = min(1.0, max(0.0, (v0 - v_min) / (v_max - v_min)))
    return v0, pn, threshold + pn * (1.0 - threshold) + AIM_POWER_MARGIN


def _team_holds(env: Any, team: int) -> float:
    return 1.0 if (env.ball_holder is not None and env.ball_holder // TEAM_SIZE == team) else 0.0


def _roll_aim_branch(
    snap: Any,
    *,
    branch: str,
    recorded_actions: np.ndarray,
    kicker: int,
    target: int,
    branch_seed: int,
    team_a: Optional[harness.TeamPolicy],
    team_b: Optional[harness.TeamPolicy],
    steer: Optional[float],
    power: Optional[float],
    mode: str,
    sigma_scale: float,
    after_step: Optional[Callable[[str, int, Any, np.ndarray], None]],
) -> Dict[str, Any]:
    env = copy.deepcopy(snap)
    team = kicker // TEAM_SIZE
    sl = slice(team * TEAM_SIZE, (team + 1) * TEAM_SIZE)
    gamma = float(env.cfg.ppo.gamma)
    max_h = max(AIM_HORIZONS)
    gen = torch.Generator().manual_seed(int(branch_seed))
    obs = env._get_observations_array()
    cen = env.get_centralized_state()
    start = (int(env.stats["score"][team]), int(env.stats["score"][1 - team]))
    to_target = np.asarray(env.player_pos[target], dtype=np.float64) - np.asarray(env.ball_pos, dtype=np.float64)

    disc_ns = 0.0
    disc_full = 0.0
    returns: Dict[int, Dict[str, float]] = {}
    possession: Dict[int, float] = {}
    first: Optional[str] = None
    kick_err: Optional[float] = None
    done = False
    for k in range(max_h):
        act_a, val_a = _team_action_and_value(team_a, env, 0, obs, cen, mode, sigma_scale, gen)
        act_b, val_b = _team_action_and_value(team_b, env, 1, obs, cen, mode, sigma_scale, gen)
        val = np.concatenate([val_a, val_b])
        if k in AIM_HORIZONS:
            returns[k] = {
                "non_shaping_return": disc_ns,
                "full_return_bootstrap": disc_full + gamma ** k * float(np.mean(val[sl])),
            }
        act = np.concatenate([act_a, act_b])
        if k == 0:
            act = np.array(recorded_actions, dtype=np.float32, copy=True)
            if steer is not None:
                act[kicker, 1] = steer
            if power is not None:
                act[kicker, 2] = power
        obs, cen, rew, done, _ = env.step_array(act, val)
        if after_step is not None:
            after_step(branch, k, env, act)
        if k == 0 and env.ball_holder is None:
            bv = env.ball_vel
            kick_err = abs(math.degrees(_wrap_angle(math.atan2(bv[1], bv[0]) - math.atan2(to_target[1], to_target[0]))))
        r_ns = _non_shaping_rewards(env, rew)
        disc_ns += gamma ** k * float(np.mean(r_ns[sl]))
        disc_full += gamma ** k * float(np.mean(np.asarray(rew, dtype=np.float64)[sl]))
        h = env.ball_holder
        if first is None and 1 <= k < AIM_FIRST_RECEIVER_STEPS and h is not None and h != kicker:
            first = "target" if h == target else ("teammate" if h // TEAM_SIZE == team else "opponent")
        if k + 1 in AIM_POSSESSION_STEPS:
            possession[k + 1] = _team_holds(env, team)
        if done:
            break
    if not done:
        _, val_a = _team_action_and_value(team_a, env, 0, obs, cen, mode, sigma_scale, gen)
        _, val_b = _team_action_and_value(team_b, env, 1, obs, cen, mode, sigma_scale, gen)
        returns[max_h] = {
            "non_shaping_return": disc_ns,
            "full_return_bootstrap": disc_full + gamma ** max_h * float(np.mean(np.concatenate([val_a, val_b])[sl])),
        }
    for h in AIM_HORIZONS:
        returns.setdefault(h, {"non_shaping_return": disc_ns, "full_return_bootstrap": disc_full})
    for s in AIM_POSSESSION_STEPS:
        possession.setdefault(s, _team_holds(env, team))
    return {
        "returns": returns,
        "possession": possession,
        "first_receiver": first or "none",
        "goal_for": 1.0 if int(env.stats["score"][team]) > start[0] else 0.0,
        "goal_against": 1.0 if int(env.stats["score"][1 - team]) > start[1] else 0.0,
        "kick_err_deg": kick_err,
    }


def roll_aim_branches(
    snap: Any,
    *,
    recorded_actions: np.ndarray,
    kicker: int,
    target: int,
    branch_seed: int,
    team_a: Optional[harness.TeamPolicy],
    team_b: Optional[harness.TeamPolicy],
    steer: float,
    power: float,
    mode: str = "sampled",
    sigma_scale: float = 1.0,
    after_step: Optional[Callable[[str, int, Any, np.ndarray], None]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Roll POLICY, AIM (kicker steer `steer` at step 0) and AIMP (steer `steer`, kick column `power`) from `snap`,
    all three from one generator seed. `after_step(branch, k, env, act)` sees every applied action."""
    overrides = {"policy": (None, None), "aim": (steer, None), "aimp": (steer, power)}
    return {
        branch: _roll_aim_branch(
            snap,
            branch=branch,
            recorded_actions=recorded_actions,
            kicker=kicker,
            target=target,
            branch_seed=branch_seed,
            team_a=team_a,
            team_b=team_b,
            steer=overrides[branch][0],
            power=overrides[branch][1],
            mode=mode,
            sigma_scale=sigma_scale,
            after_step=after_step,
        )
        for branch in AIM_BRANCHES
    }


def _aim_worker(task: Tuple[int, str, int, int]) -> Dict[str, Any]:
    match_idx, path_str, trace_seed, rollout_seed = task
    assert _HOLD_POLICIES is not None and _HOLD_SPEC_CFG is not None
    team_a, team_b = _HOLD_POLICIES
    mode, sigma_scale, shaping_mode = _HOLD_SPEC_CFG

    trace = _load_trace(Path(path_str))
    pts = find_aim_points(trace)
    ok, max_err, snapshots = resimulate_match(
        trace,
        match_seed=trace_seed * 1000 + match_idx,
        target_steps=sorted({0, *(p["step"] for p in pts)}),
        shaping_mode=shaping_mode,
    )
    if not ok:
        return {"ok": False, "max_err": max_err, "records": []}
    _verify_checkpoint_matches_trace(match_idx, 0, snapshots[0], trace, team_a, team_b)

    act = np.asarray(trace["act"], dtype=np.float32)
    records = []
    for p in pts:
        step, kicker = p["step"], p["kicker"]
        snap = snapshots[step]
        phys, mech = snap.cfg.physics, snap.cfg.mechanics
        steer = aim_steer(p["deflection"], float(mech.kick_steer_max_deflection_deg))
        v_max, _ = snap._get_role_kick_params(kicker)
        _, _, power = aimp_power(
            p["target_dist"],
            float(v_max),
            v_min=float(phys.ball_min_kick_speed),
            decel=float(phys.ball_friction_decel),
            threshold=float(mech.kick_trigger_threshold),
        )
        branches = roll_aim_branches(
            snap,
            recorded_actions=act[step],
            kicker=kicker,
            target=p["target"],
            branch_seed=(rollout_seed * 1_000_003 + match_idx * 10_000 + step) & 0x7FFF_FFFF,
            team_a=team_a,
            team_b=team_b,
            steer=steer,
            power=power,
            mode=mode,
            sigma_scale=sigma_scale,
        )
        records.append({
            "match_idx": match_idx,
            **p,
            "recorded_steer": float(act[step, kicker, 1]),
            "recorded_power": float(act[step, kicker, 2]),
            "aim_steer": steer,
            "aimp_power": power,
            "branches": branches,
        })
    return {"ok": True, "max_err": max_err, "records": records}


def _branch_scalars(res: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for h in AIM_HORIZONS:
        for metric in ("non_shaping_return", "full_return_bootstrap"):
            out[f"{metric}_{h}"] = float(res["returns"][h][metric])
    for s in AIM_POSSESSION_STEPS:
        out[f"possession_{s}"] = float(res["possession"][s])
    out["goal_for"] = float(res["goal_for"])
    out["goal_against"] = float(res["goal_against"])
    return out


def _summarize_aim_bucket(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    branches: Dict[str, Any] = {}
    for b in AIM_BRANCHES:
        res = [r["branches"][b] for r in records]
        errs = [x["kick_err_deg"] for x in res]
        loose = [e for e in errs if e is not None]
        entry: Dict[str, Any] = {
            f"first_receiver_{o}": _stat_dict(1.0 if x["first_receiver"] == o else 0.0 for x in res) for o in AIM_RECEIVERS
        }
        for s in AIM_POSSESSION_STEPS:
            entry[f"possession_{s}"] = _stat_dict(float(x["possession"][s]) for x in res)
        entry["goal_for"] = _stat_dict(float(x["goal_for"]) for x in res)
        entry["goal_against"] = _stat_dict(float(x["goal_against"]) for x in res)
        entry["ball_held_after_kick"] = _stat_dict(1.0 if e is None else 0.0 for e in errs)
        entry["kick_within_15deg"] = _stat_dict(1.0 if (e is not None and e <= census.AIM_CONE_DEG) else 0.0 for e in errs)
        entry["kick_err_median_deg"] = float(np.median(loose)) if loose else None
        branches[b] = entry
    paired: Dict[str, Any] = {}
    for b in ("aim", "aimp"):
        diffs = []
        for r in records:
            base, alt = _branch_scalars(r["branches"]["policy"]), _branch_scalars(r["branches"][b])
            diffs.append({key: alt[key] - base[key] for key in AIM_SCALAR_KEYS})
        paired[f"{b}_minus_policy"] = {key: _stat_dict(d[key] for d in diffs) for key in AIM_SCALAR_KEYS}
    return {"n": len(records), "branches": branches, "paired": paired}


def summarize_aim(
    records: Sequence[Dict[str, Any]],
    n_matches_total: int,
    n_matches_ok: int,
    n_matches_skipped: int,
    max_pos_err: Optional[float],
    *,
    seed: int,
    shaping_mode: str = str(DEFAULT_CONFIG.rewards.shaping_mode),
    gamma: float = float(DEFAULT_CONFIG.ppo.gamma),
) -> Dict[str, Any]:
    buckets: Dict[str, Callable[[Dict[str, Any]], bool]] = {
        "not_aimed": lambda r: not r["already"],
        "already_aimed": lambda r: bool(r["already"]),
    }
    for t in AIM_THIRDS:
        buckets[f"not_aimed_third_{t}"] = lambda r, t=t: (not r["already"]) and r["third"] == t
    for i in AIM_INTENTS:
        buckets[f"not_aimed_intent_{i}"] = lambda r, i=i: (not r["already"]) and r["intent"] == i
    return {
        "mode": "aim",
        "seed": seed,
        "shaping_mode": shaping_mode,
        "gamma": float(gamma),
        "n_matches_total": n_matches_total,
        "n_matches_ok": n_matches_ok,
        "n_matches_skipped": n_matches_skipped,
        "max_pos_err": max_pos_err,
        "n_points": len(records),
        "buckets": {name: _summarize_aim_bucket([r for r in records if pred(r)]) for name, pred in buckets.items()},
    }


def run_aim(
    *,
    trace_dir: Path | str,
    checkpoint_dir: Path | str,
    trace_seed: Optional[int] = None,
    seed: int = 1,
    max_matches: Optional[int] = None,
    workers: int = 1,
    shaping_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Summary of `aim` over the traces, with the per-point records under `records`."""
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    spec = harness.EvalSpec(
        checkpoint_dir=str(checkpoint_dir), opponent="self", mode="sampled", n=1, seed=int(seed), workers=int(workers)
    )
    spec.validate()
    t_seed = _resolve_trace_seed(trace_dir, trace_seed)
    items = _trace_paths_with_indices(trace_dir, max_matches=max_matches)
    tasks = [(idx, str(path), t_seed, int(seed)) for idx, path in items]

    if workers == 1 or len(tasks) == 1:
        prev_threads = torch.get_num_threads()
        try:
            _init_hold_worker(spec, shaping_mode)
            outputs = [_aim_worker(t) for t in tasks]
        finally:
            torch.set_num_threads(prev_threads)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(workers, len(tasks)), initializer=_init_hold_worker, initargs=(spec, shaping_mode)) as pool:
            outputs = list(pool.map(_aim_worker, tasks, chunksize=1))

    records: List[Dict[str, Any]] = []
    n_ok = 0
    max_err: Optional[float] = None
    for res in outputs:
        if res["ok"]:
            n_ok += 1
            records.extend(res["records"])
            err = float(res["max_err"])
            max_err = err if max_err is None else max(max_err, err)
    summary = summarize_aim(
        records,
        len(tasks),
        n_ok,
        len(tasks) - n_ok,
        max_err,
        seed=int(seed),
        shaping_mode=_resolved_shaping_mode(shaping_mode),
        gamma=float(DEFAULT_CONFIG.ppo.gamma),
    )
    summary["records"] = records
    return summary


def format_aim_report(summary: Dict[str, Any], title: str = "") -> str:
    header = f"===== rollouts aim {title}".strip()
    lines = [
        f"{header}  matches ok={summary['n_matches_ok']}/{summary['n_matches_total']} "
        f"skipped={summary['n_matches_skipped']} points={summary['n_points']} seed={summary['seed']} "
        f"shaping={summary['shaping_mode']} max_pos_err={summary['max_pos_err'] if summary['max_pos_err'] is not None else '-'}"
    ]
    labels = {"not_aimed": "not already aimed", "already_aimed": "already aimed"}
    labels.update({f"not_aimed_third_{t}": f"not already aimed, {t} third" for t in AIM_THIRDS})
    labels.update({f"not_aimed_intent_{i}": f"not already aimed, intent={i}" for i in AIM_INTENTS})

    def mean(stat: Dict[str, Any]) -> str:
        return "-" if stat["mean"] is None else f"{stat['mean']:.3f}"

    for key, label in labels.items():
        b = summary["buckets"][key]
        lines.append(f"  -- {label} (n={b['n']}) --")
        if b["n"] == 0:
            continue
        for br in AIM_BRANCHES:
            e = b["branches"][br]
            med = "-" if e["kick_err_median_deg"] is None else f"{e['kick_err_median_deg']:.1f}"
            lines.append(
                f"    {br:6s}: first receiver target {mean(e['first_receiver_target'])} teammate "
                f"{mean(e['first_receiver_teammate'])} opponent {mean(e['first_receiver_opponent'])} none "
                f"{mean(e['first_receiver_none'])} | possession 1 s {mean(e['possession_15'])} 2 s {mean(e['possession_30'])}"
                f" | gf {mean(e['goal_for'])} ga {mean(e['goal_against'])} | kick dir vs target: median {med} deg, "
                f"<= 15 deg {mean(e['kick_within_15deg'])}, ball held {mean(e['ball_held_after_kick'])}"
            )
        for pk, plabel in (("aim_minus_policy", "AIM - POLICY"), ("aimp_minus_policy", "AIMP - POLICY")):
            d = b["paired"][pk]
            for h in AIM_HORIZONS:
                lines.append(
                    f"    {plabel:13s} H={h:2d}: non_shaping {census._fmt_d(d[f'non_shaping_return_{h}'], 4)} | "
                    f"full_boot {census._fmt_d(d[f'full_return_bootstrap_{h}'], 4)}"
                )
            lines.append(
                f"    {plabel:13s} possession 1 s {census._fmt_d(d['possession_15'], 3)} | "
                f"2 s {census._fmt_d(d['possession_30'], 3)} | gf {census._fmt_d(d['goal_for'], 3)} | "
                f"ga {census._fmt_d(d['goal_against'], 3)}"
            )
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Credit: how much of a decision's counterfactual benefit reaches the trainer's advantage
# ----------------------------------------------------------------------------
CREDIT_STEPS = 75
CREDIT_SEEDS = 4
CREDIT_SEED_STRIDE = 7919
CREDIT_GOAL_REWARD = 10.0
CREDIT_LAMBDAS = (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0)
CREDIT_PARTS = ("goal", "sparse", "ball", "role")
CREDIT_SHARE_PARTS = (*CREDIT_PARTS, "critic")
ATTACK_THIRD = 2.0 * FIELD_WIDTH / 3.0
# decision -> (recorded branch, alternative branch, sign of the benefit: +1 = alternative - recorded, -1 = recorded - alternative)
CREDIT_DECISIONS = {"aim": ("policy", "aim", 1.0), "attack": ("kick", "hold", -1.0)}


def credit_branch_seed(seed: int, j: int, match_idx: int, step: int) -> int:
    """Generator seed shared by the two branches of seed index `j` at decision `step` of match `match_idx`."""
    return ((int(seed) + CREDIT_SEED_STRIDE * int(j)) * 1_000_003 + int(match_idx) * 10_000 + int(step)) & 0x7FFF_FFFF


def credit_step_components(
    reward: float,
    shaping: float,
    goals_for: int,
    goals_against: int,
    phi_before: float,
    phi_after: float,
    advance_weight: float,
) -> np.ndarray:
    """One decision of the kicker's reward as `(goal, sparse, ball, role)`; the four add up to `reward`."""
    goal = CREDIT_GOAL_REWARD * (int(goals_for) - int(goals_against))
    ball = float(advance_weight) * (float(phi_after) - float(phi_before))
    return np.array([goal, (float(reward) - float(shaping)) - goal, ball, float(shaping) - ball], dtype=np.float64)


def credit_gae(
    components: np.ndarray, values: np.ndarray, gamma: float, lam: float, *, terminal: bool
) -> Tuple[np.ndarray, float]:
    """The trainer's GAE at decision 0 of a branch as `(reward parts, critic part)`; their sum is the advantage.

    `components` is `(n, parts)` (the reward of every decision, split in any way), `values` holds V_0..V_n (V_n is the bootstrap
    after the last decision and is not read when `terminal`, where the last decision ends the episode). The advantage is
    sum_l (gamma lam)^l delta_l with delta_l = r_l + gamma V_{l+1} - V_l; the reward part is sum_l (gamma lam)^l r_l per part and the
    critic part sum_l (gamma lam)^l (gamma V_{l+1} - V_l).
    """
    comps = np.asarray(components, dtype=np.float64)
    n = int(comps.shape[0])
    if n < 1:
        raise ValueError("a branch needs at least one decision")
    if not (0.0 <= float(lam) <= 1.0):
        raise ValueError(f"lambda must be in [0, 1], got {lam}")
    v = np.asarray(values, dtype=np.float64)
    if v.shape[0] < (n if terminal else n + 1):
        raise ValueError(f"{n} decisions need {n if terminal else n + 1} critic values, got {v.shape[0]}")
    weights = (float(gamma) * float(lam)) ** np.arange(n)
    next_value = np.zeros(n)
    next_value[: n - 1] = v[1:n]
    if not terminal:
        next_value[n - 1] = v[n]
    return (weights[:, None] * comps).sum(axis=0), float((weights * (float(gamma) * next_value - v[:n])).sum())


def find_attack_points(trace: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """Outfield kicks in the kicker's attacking third, >= 45 decisions apart (see `credit` in the module docstring)."""
    holder = np.asarray(trace["holder"], dtype=np.int64)
    ball = np.asarray(trace["ball"], dtype=np.float64)
    side = np.asarray(trace["side"], dtype=np.int64)
    n = int(holder.size)
    steps, players = census.kick_events(trace)
    points: List[Dict[str, Any]] = []
    last_step = -SPACING_STEPS - 1
    for step, kicker in sorted(zip(steps.tolist(), players.tolist())):
        if kicker % TEAM_SIZE == 0 or int(holder[step]) != kicker or step + 2 >= n or step - last_step < SPACING_STEPS:
            continue
        team = kicker // TEAM_SIZE
        canon_x = float(ball[step, 0]) if int(side[step, team]) == 0 else FIELD_WIDTH - float(ball[step, 0])
        if canon_x <= ATTACK_THIRD:
            continue
        last_step = step
        points.append({"step": int(step), "kicker": int(kicker), "team": team})
    return points


def find_credit_points(trace: Dict[str, np.ndarray], decision: str) -> List[Dict[str, Any]]:
    """Points of a credit decision: `aim` points not already aimed, or attacking-third kicks."""
    if decision == "aim":
        return [p for p in find_aim_points(trace) if not p["already"]]
    if decision == "attack":
        return find_attack_points(trace)
    raise ValueError(f"decision must be one of {tuple(CREDIT_DECISIONS)}, got {decision!r}")


def roll_credit_branch(
    snap: Any,
    *,
    recorded_actions: np.ndarray,
    kicker: int,
    branch_seed: int,
    team_a: Optional[harness.TeamPolicy],
    team_b: Optional[harness.TeamPolicy],
    steer: Optional[float] = None,
    hold_steps: int = 0,
    mode: str = "sampled",
    sigma_scale: float = 1.0,
    steps: int = CREDIT_STEPS,
    after_step: Optional[Callable[[int, Any, np.ndarray], None]] = None,
) -> Dict[str, Any]:
    """Roll `steps` decisions from `snap`: decision 0 plays `recorded_actions` (the kicker's steer replaced by `steer` when given),
    later decisions sample both teams; the kicker's kick column is -1 for the first `hold_steps` decisions.

    Returns the kicker's reward split per decision `components` `(steps, 4)` (goal, sparse, ball, role; zero after the last played
    decision), its critic values `values` `(steps + 1,)` (V_t before decision t, V_steps after the last one unless the branch ended),
    `n_steps`, `terminal`, and the goals for and against of the kicker's team within the branch.
    """
    env = copy.deepcopy(snap)
    team = kicker // TEAM_SIZE
    opp = 1 - team
    gen = torch.Generator().manual_seed(int(branch_seed))
    obs = env._get_observations_array()
    cen = env.get_centralized_state()
    advance_weight = float(env.cfg.rewards.team_advance_pbrs_weight)
    components = np.zeros((steps, len(CREDIT_PARTS)))
    values = np.zeros(steps + 1)
    n_steps, terminal, goals_for, goals_against = steps, False, 0, 0
    for k in range(steps):
        act_a, val_a = _team_action_and_value(team_a, env, 0, obs, cen, mode, sigma_scale, gen)
        act_b, val_b = _team_action_and_value(team_b, env, 1, obs, cen, mode, sigma_scale, gen)
        val = np.concatenate([val_a, val_b])
        values[k] = float(val[kicker])
        act = np.concatenate([act_a, act_b])
        if k == 0:
            act = np.array(recorded_actions, dtype=np.float32, copy=True)
            if steer is not None:
                act[kicker, 1] = steer
        if k < hold_steps:
            act[kicker, 2] = -1.0
        score_for, score_against = int(env.stats["score"][team]), int(env.stats["score"][opp])
        phi_before = float(env._compute_ball_potential(team))
        obs, cen, rew, done, _ = env.step_array(act, val)
        if after_step is not None:
            after_step(k, env, act)
        d_for, d_against = int(env.stats["score"][team]) - score_for, int(env.stats["score"][opp]) - score_against
        goals_for += d_for
        goals_against += d_against
        components[k] = credit_step_components(
            float(rew[kicker]), float(env.last_shaping_rewards[kicker]), d_for, d_against,
            phi_before, float(env._compute_ball_potential(team)), advance_weight,
        )
        if done:
            n_steps, terminal = k + 1, True
            break
    if not terminal:
        _, val_a = _team_action_and_value(team_a, env, 0, obs, cen, mode, sigma_scale, gen)
        _, val_b = _team_action_and_value(team_b, env, 1, obs, cen, mode, sigma_scale, gen)
        values[steps] = float(np.concatenate([val_a, val_b])[kicker])
    return {
        "components": components,
        "values": values,
        "n_steps": n_steps,
        "terminal": terminal,
        "goals_for": goals_for,
        "goals_against": goals_against,
    }


def roll_credit_point(
    snap: Any,
    *,
    decision: str,
    recorded_actions: np.ndarray,
    kicker: int,
    branch_seeds: Sequence[int],
    team_a: Optional[harness.TeamPolicy],
    team_b: Optional[harness.TeamPolicy],
    steer: Optional[float] = None,
    hold_steps: int = HOLD_OVERRIDE_STEPS,
    mode: str = "sampled",
    sigma_scale: float = 1.0,
    after_step: Optional[Callable[[str, int, int, Any, np.ndarray], None]] = None,
) -> Dict[str, np.ndarray]:
    """Both branches of a decision for every seed in `branch_seeds` (one generator seed per index j for the pair).

    `aim`: POLICY and AIM (`steer` required, the kicker's steer at decision 0). `attack`: KICK and HOLD (`hold_steps` decisions with the
    kicker's kick column at -1). Arrays are indexed `[j, branch, ...]` with the recorded branch first.
    `after_step(branch, j, k, env, act)` sees every applied action.
    """
    if decision not in CREDIT_DECISIONS:
        raise ValueError(f"decision must be one of {tuple(CREDIT_DECISIONS)}, got {decision!r}")
    if decision == "aim" and steer is None:
        raise ValueError("the aim decision needs a steer")
    labels = CREDIT_DECISIONS[decision][:2]
    n_seeds = len(branch_seeds)
    components = np.zeros((n_seeds, 2, CREDIT_STEPS, len(CREDIT_PARTS)))
    values = np.zeros((n_seeds, 2, CREDIT_STEPS + 1))
    n_steps = np.zeros((n_seeds, 2), dtype=np.int64)
    terminal = np.zeros((n_seeds, 2), dtype=bool)
    goals = np.zeros((n_seeds, 2, 2), dtype=np.int64)
    for j, seed in enumerate(branch_seeds):
        for b, label in enumerate(labels):
            hook = None if after_step is None else (lambda k, env, act, label=label, j=j: after_step(label, j, k, env, act))
            res = roll_credit_branch(
                snap, recorded_actions=recorded_actions, kicker=kicker, branch_seed=int(seed), team_a=team_a, team_b=team_b,
                steer=steer if (decision == "aim" and b == 1) else None,
                hold_steps=hold_steps if (decision == "attack" and b == 1) else 0,
                mode=mode, sigma_scale=sigma_scale, after_step=hook,
            )
            components[j, b], values[j, b] = res["components"], res["values"]
            n_steps[j, b], terminal[j, b] = res["n_steps"], res["terminal"]
            goals[j, b] = (res["goals_for"], res["goals_against"])
    return {"components": components, "values": values, "n_steps": n_steps, "terminal": terminal, "goals": goals}


def _lambda_key(lam: float) -> str:
    return f"{float(lam):g}"


def credit_statistics(
    components: np.ndarray,
    values: np.ndarray,
    n_steps: np.ndarray,
    terminal: np.ndarray,
    *,
    lambdas: Sequence[float],
    gamma: float,
    sign: float,
) -> Dict[str, Dict[str, Any]]:
    """Signal, noise and variance shares of the GAE at decision 0 per lambda (see `credit` in the module docstring).

    Arrays are indexed `[point, seed, branch, ...]`, the recorded branch first: `components` `(P, K, 2, T, 4)`, `values`
    `(P, K, 2, T + 1)`, `n_steps` and `terminal` `(P, K, 2)`. `sign` is +1 when the benefit is alternative - recorded and -1 when it
    is recorded - alternative. Every statistic is None where it is undefined (no points, one point for an SE, zero noise for an SNR
    or variance share).
    """
    n_points, n_seeds = int(components.shape[0]), int(components.shape[1])
    if n_seeds < 2:
        raise ValueError(f"the noise needs at least two seeds, got {n_seeds}")
    out: Dict[str, Dict[str, Any]] = {}
    for lam in lambdas:
        adv = np.zeros((n_points, n_seeds, 2))
        parts = np.zeros((n_points, n_seeds, 2, len(CREDIT_SHARE_PARTS)))
        for p in range(n_points):
            for j in range(n_seeds):
                for b in range(2):
                    n = int(n_steps[p, j, b])
                    reward, critic = credit_gae(
                        components[p, j, b, :n], values[p, j, b, : n + 1], gamma, lam, terminal=bool(terminal[p, j, b])
                    )
                    parts[p, j, b, : len(CREDIT_PARTS)] = reward
                    parts[p, j, b, -1] = critic
                    adv[p, j, b] = reward.sum() + critic
        entry: Dict[str, Any] = {
            "S": None, "S_se": None, "N": None, "SNR": None, "paired_sd": None,
            "var_share": {k: None for k in CREDIT_SHARE_PARTS}, "signal_parts": {k: None for k in CREDIT_SHARE_PARTS},
        }
        if n_points > 0:
            benefit = sign * (adv[:, :, 1] - adv[:, :, 0])                                  # (P, K)
            per_point = benefit.mean(axis=1)
            entry["S"] = float(per_point.mean())
            if n_points > 1:
                entry["S_se"] = float(per_point.std(ddof=1) / math.sqrt(n_points))
            entry["paired_sd"] = float(benefit.std(ddof=1))
            variance = adv[:, :, 0].var(axis=1, ddof=1).mean()
            entry["N"] = float(math.sqrt(variance))
            if entry["N"] > 0.0:
                entry["SNR"] = entry["S"] / entry["N"]
                entry["var_share"] = {
                    name: float(parts[:, :, 0, c].var(axis=1, ddof=1).mean() / variance) for c, name in enumerate(CREDIT_SHARE_PARTS)
                }
            entry["signal_parts"] = {
                name: float(sign * (parts[:, :, 1, c] - parts[:, :, 0, c]).mean(axis=1).mean())
                for c, name in enumerate(CREDIT_SHARE_PARTS)
            }
        out[_lambda_key(lam)] = entry
    return out


def _credit_worker(task: Tuple[int, str, int, int, int, str]) -> Dict[str, Any]:
    match_idx, path_str, trace_seed, rollout_seed, n_seeds, decision = task
    assert _HOLD_POLICIES is not None and _HOLD_SPEC_CFG is not None
    team_a, team_b = _HOLD_POLICIES
    mode, sigma_scale, shaping_mode = _HOLD_SPEC_CFG

    trace = _load_trace(Path(path_str))
    pts = find_credit_points(trace, decision)
    ok, max_err, snapshots = resimulate_match(
        trace,
        match_seed=trace_seed * 1000 + match_idx,
        target_steps=sorted({0, *(p["step"] for p in pts)}),
        shaping_mode=shaping_mode,
    )
    if not ok:
        return {"ok": False, "max_err": max_err, "records": []}
    _verify_checkpoint_matches_trace(match_idx, 0, snapshots[0], trace, team_a, team_b)

    act = np.asarray(trace["act"], dtype=np.float32)
    records = []
    for p in pts:
        step, kicker = p["step"], p["kicker"]
        snap = snapshots[step]
        steer = (
            aim_steer(p["deflection"], float(snap.cfg.mechanics.kick_steer_max_deflection_deg)) if decision == "aim" else None
        )
        res = roll_credit_point(
            snap,
            decision=decision,
            recorded_actions=act[step],
            kicker=kicker,
            branch_seeds=[credit_branch_seed(rollout_seed, j, match_idx, step) for j in range(n_seeds)],
            team_a=team_a,
            team_b=team_b,
            steer=steer,
            mode=mode,
            sigma_scale=sigma_scale,
        )
        point = {"match_idx": match_idx, "step": step, "kicker": kicker}
        if decision == "aim":
            point["target"] = p["target"]
        records.append({"point": point, **res})
    return {"ok": True, "max_err": max_err, "records": records}


def summarize_credit(
    records: Sequence[Dict[str, Any]],
    n_matches_total: int,
    n_matches_ok: int,
    n_matches_skipped: int,
    max_pos_err: Optional[float],
    *,
    decision: str,
    seed: int,
    seeds: int,
    lambdas: Sequence[float],
    shaping_mode: str = str(DEFAULT_CONFIG.rewards.shaping_mode),
    gamma: float = float(DEFAULT_CONFIG.ppo.gamma),
) -> Dict[str, Any]:
    """Summary of `credit` over per-point records (`point` plus the arrays of `roll_credit_point`)."""
    n_points = len(records)

    def stack(key: str, shape: Tuple[int, ...], dtype: Any) -> np.ndarray:
        return np.stack([r[key] for r in records]) if records else np.zeros((0, *shape), dtype=dtype)

    comps = stack("components", (seeds, 2, CREDIT_STEPS, len(CREDIT_PARTS)), np.float64)
    vals = stack("values", (seeds, 2, CREDIT_STEPS + 1), np.float64)
    n_steps = stack("n_steps", (seeds, 2), np.int64)
    terminal = stack("terminal", (seeds, 2), bool)
    sign = CREDIT_DECISIONS[decision][2]
    summary: Dict[str, Any] = {
        "mode": "credit",
        "decision": decision,
        "seed": seed,
        "seeds": seeds,
        "steps": CREDIT_STEPS,
        "lambdas": [float(x) for x in lambdas],
        "shaping_mode": shaping_mode,
        "gamma": float(gamma),
        "n_matches_total": n_matches_total,
        "n_matches_ok": n_matches_ok,
        "n_matches_skipped": n_matches_skipped,
        "max_pos_err": max_pos_err,
        "n_points": n_points,
        "points": [r["point"] for r in records],
        "estimators": credit_statistics(comps, vals, n_steps, terminal, lambdas=lambdas, gamma=gamma, sign=sign),
    }
    if decision == "attack":
        goals = stack("goals", (seeds, 2, 2), np.int64)
        recorded, alternative = CREDIT_DECISIONS[decision][:2]
        summary["goals_5s"] = {
            label: {
                kind: (float(goals[:, :, b, c].mean()) if n_points else None) for c, kind in enumerate(("for", "against"))
            }
            for b, label in enumerate((recorded, alternative))
        }
    return summary


def run_credit(
    *,
    trace_dir: Path | str,
    checkpoint_dir: Path | str,
    decision: str = "aim",
    trace_seed: Optional[int] = None,
    seed: int = 1,
    seeds: int = CREDIT_SEEDS,
    lambdas: Sequence[float] = CREDIT_LAMBDAS,
    max_matches: Optional[int] = None,
    workers: int = 1,
    shaping_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """`credit` over the traces (see the module docstring); the summary carries the points under `points`."""
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    if decision not in CREDIT_DECISIONS:
        raise ValueError(f"decision must be one of {tuple(CREDIT_DECISIONS)}, got {decision!r}")
    if int(seeds) < 2:
        raise ValueError(f"seeds must be at least 2, got {seeds}")
    if not lambdas or not all(0.0 <= float(x) <= 1.0 for x in lambdas):
        raise ValueError(f"lambdas must be a non-empty list in [0, 1], got {list(lambdas)}")
    spec = harness.EvalSpec(
        checkpoint_dir=str(checkpoint_dir), opponent="self", mode="sampled", n=1, seed=int(seed), workers=int(workers)
    )
    spec.validate()
    t_seed = _resolve_trace_seed(trace_dir, trace_seed)
    items = _trace_paths_with_indices(trace_dir, max_matches=max_matches)
    tasks = [(idx, str(path), t_seed, int(seed), int(seeds), decision) for idx, path in items]

    if workers == 1 or len(tasks) == 1:
        prev_threads = torch.get_num_threads()
        try:
            _init_hold_worker(spec, shaping_mode)
            outputs = [_credit_worker(t) for t in tasks]
        finally:
            torch.set_num_threads(prev_threads)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(workers, len(tasks)), initializer=_init_hold_worker, initargs=(spec, shaping_mode)) as pool:
            outputs = list(pool.map(_credit_worker, tasks, chunksize=1))

    records: List[Dict[str, Any]] = []
    n_ok = 0
    max_err: Optional[float] = None
    for res in outputs:
        if res["ok"]:
            n_ok += 1
            records.extend(res["records"])
            err = float(res["max_err"])
            max_err = err if max_err is None else max(max_err, err)
    return summarize_credit(
        records,
        len(tasks),
        n_ok,
        len(tasks) - n_ok,
        max_err,
        decision=decision,
        seed=int(seed),
        seeds=int(seeds),
        lambdas=[float(x) for x in lambdas],
        shaping_mode=_resolved_shaping_mode(shaping_mode),
        gamma=float(DEFAULT_CONFIG.ppo.gamma),
    )


def format_credit_report(summary: Dict[str, Any], title: str = "") -> str:
    header = f"===== rollouts credit {title}".strip()
    decision = summary["decision"]
    recorded, alternative, _ = CREDIT_DECISIONS[decision]
    lines = [
        f"{header}  decision={decision} matches ok={summary['n_matches_ok']}/{summary['n_matches_total']} "
        f"skipped={summary['n_matches_skipped']} points={summary['n_points']} seeds={summary['seeds']} seed={summary['seed']} "
        f"shaping={summary['shaping_mode']} gamma={summary['gamma']:g} "
        f"max_pos_err={summary['max_pos_err'] if summary['max_pos_err'] is not None else '-'}"
    ]
    if summary["n_points"] == 0:
        lines.append("  no points: nothing to measure")
    benefit = f"A_{alternative.upper()} - A_{recorded.upper()}" if decision == "aim" else f"A_{recorded.upper()} - A_{alternative.upper()}"
    lines.append(f"signal S = mean over points of ({benefit}); noise N = SD across seeds of A_{recorded.upper()} (same state and action)")
    if "goals_5s" in summary:
        g = summary["goals_5s"]

        def rate(v: Optional[float]) -> str:
            return "-" if v is None else f"{v:.3f}"

        lines.append(
            "goals within 5 s: "
            + " | ".join(f"{label.upper()} for {rate(g[label]['for'])} against {rate(g[label]['against'])}" for label in g)
        )
    lines.append(
        "lambda | S +- se | N | SNR | paired SD | variance share goal/sparse/ball/role/critic | signal goal/sparse/ball/role/critic"
    )

    def num(v: Optional[float], fmt: str) -> str:
        return "-" if v is None else format(v, fmt)

    for key, r in summary["estimators"].items():
        shares = "/".join(num(r["var_share"][c], ".2f") for c in CREDIT_SHARE_PARTS)
        parts = "/".join(num(r["signal_parts"][c], "+.3f") for c in CREDIT_SHARE_PARTS)
        lines.append(
            f"{key} | {num(r['S'], '+.4f')} +- {num(r['S_se'], '.4f')} | {num(r['N'], '.4f')} | {num(r['SNR'], '+.3f')} | "
            f"{num(r['paired_sd'], '.3f')} | {shares} | {parts}"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer >= 1, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected an integer >= 1, got {value}")
    return value


def _seed_count(text: str) -> int:
    value = _positive_int(text)
    if value < 2:
        raise argparse.ArgumentTypeError(f"expected at least 2 seeds (the noise needs two), got {value}")
    return value


def _unit_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number in [0, 1], got {text!r}") from None
    if not (0.0 <= value <= 1.0):
        raise argparse.ArgumentTypeError(f"expected a number in [0, 1], got {text!r}")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--trace-dir", required=True, help="directory containing m*.npz harness traces (read only)")
        p.add_argument("--trace-seed", type=int, default=None, help="harness seed used for traces (default: census.json or 11)")
        p.add_argument("--max-matches", type=int, default=None, help="use only the first N trace files")
        p.add_argument("--workers", type=int, default=4, help="worker processes")
        p.add_argument("--shaping-mode", choices=("legacy", "ball", "all"), default=None, help="override env shaping mode")
        p.add_argument("--out", required=True, help="output .json file or directory (outside protected dirs)")

    p_gk = sub.add_parser("gk", help="counterfactual goalkeeper loose-ball opportunity rollouts")
    add_common(p_gk)
    p_gk.add_argument("--checkpoint-dir", default=None, help="optional checkpoint directory (read only)")

    p_hold = sub.add_parser("hold", help="counterfactual hold-vs-kick rollouts from eligible holding decisions")
    add_common(p_hold)
    p_hold.add_argument("--checkpoint-dir", required=True, help="checkpoint directory with blue_*/red_* policies (read only)")
    p_hold.add_argument("--opp", default="self", help="self | scripted | scripted2 | frozen:DIR")
    p_hold.add_argument("--mode", default="sampled", choices=harness.MODES)
    p_hold.add_argument("--sigma-scale", type=float, default=1.0)
    p_hold.add_argument("--swap", action="store_true", help="team A plays red_*, team B plays blue_*")
    p_hold.add_argument("--seed", type=int, default=1, help="rollout RNG seed")
    p_hold.add_argument(
        "--hold-steps",
        type=_positive_int,
        default=HOLD_OVERRIDE_STEPS,
        help=f"decisions the HOLD branch forces the holder not to kick (>= 1, default {HOLD_OVERRIDE_STEPS})",
    )

    p_aim = sub.add_parser("aim", help="counterfactual aim of real kicks at an open teammate in the kickable arc")
    add_common(p_aim)
    p_aim.add_argument("--checkpoint-dir", required=True, help="checkpoint directory with blue_*/red_* policies (read only)")
    p_aim.add_argument("--seed", type=int, default=1, help="rollout RNG seed")

    p_credit = sub.add_parser("credit", help="how much of a decision's counterfactual benefit reaches the trainer's GAE advantage")
    add_common(p_credit)
    p_credit.add_argument("--checkpoint-dir", required=True, help="checkpoint directory with blue_*/red_* policies (read only)")
    p_credit.add_argument("--decision", choices=tuple(CREDIT_DECISIONS), default="aim", help="aim: AIM vs POLICY; attack: KICK vs HOLD")
    p_credit.add_argument("--seeds", type=_seed_count, default=CREDIT_SEEDS, help=f"generator seeds per point (>= 2, default {CREDIT_SEEDS})")
    p_credit.add_argument(
        "--lambdas", type=_unit_float, nargs="+", default=list(CREDIT_LAMBDAS), help="GAE lambdas in [0, 1] (default 0 0.5 0.8 0.9 0.95 0.99 1)"
    )
    p_credit.add_argument("--seed", type=int, default=1, help="rollout RNG seed")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    protected: List[Path | str] = [args.trace_dir]
    if getattr(args, "checkpoint_dir", None):
        protected.append(args.checkpoint_dir)
    opp = getattr(args, "opp", "self")
    if isinstance(opp, str) and opp.startswith("frozen:"):
        protected.append(opp.split(":", 1)[1])

    out_path = _validate_out_path(args.out, protected)

    if args.command == "gk":
        summary = run_gk(
            trace_dir=args.trace_dir,
            trace_seed=args.trace_seed,
            max_matches=args.max_matches,
            workers=args.workers,
            shaping_mode=args.shaping_mode,
        )
        rep = format_gk_report(summary, title=str(args.trace_dir))
    elif args.command == "aim":
        summary = run_aim(
            trace_dir=args.trace_dir,
            checkpoint_dir=args.checkpoint_dir,
            trace_seed=args.trace_seed,
            seed=args.seed,
            max_matches=args.max_matches,
            workers=args.workers,
            shaping_mode=args.shaping_mode,
        )
        rep = format_aim_report(summary, title=str(args.trace_dir))
    elif args.command == "credit":
        summary = run_credit(
            trace_dir=args.trace_dir,
            checkpoint_dir=args.checkpoint_dir,
            decision=args.decision,
            trace_seed=args.trace_seed,
            seed=args.seed,
            seeds=args.seeds,
            lambdas=args.lambdas,
            max_matches=args.max_matches,
            workers=args.workers,
            shaping_mode=args.shaping_mode,
        )
        rep = format_credit_report(summary, title=str(args.trace_dir))
    else:
        summary = run_hold(
            trace_dir=args.trace_dir,
            checkpoint_dir=args.checkpoint_dir,
            opponent=args.opp,
            swap=args.swap,
            mode=args.mode,
            sigma_scale=args.sigma_scale,
            trace_seed=args.trace_seed,
            seed=args.seed,
            max_matches=args.max_matches,
            workers=args.workers,
            shaping_mode=args.shaping_mode,
            hold_steps=args.hold_steps,
        )
        rep = format_hold_report(summary, title=str(args.trace_dir))

    _write_outputs(out_path, summary, rep)
    print(rep)
    return 0


if __name__ == "__main__":  # required: spawned workers re-import this module
    raise SystemExit(main())
