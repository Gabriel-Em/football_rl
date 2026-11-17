"""Policy-internal metrics over harness traces (`tools/audit/harness.py`).

Every metric is computed per match from one trace and aggregated across matches as mean +- SE with n
(skipping matches where a denominator is zero). Keepers are slots 0 and 6; field players are slots 1..5
and 7..11.

Policy heads: the trace's optional `head` field gives each team's head (0 = gaussian4, the default when the field
is absent; 1 = hybrid5). A hybrid5 seat records `mean = 2 p - 1` in its kick and special columns, so its intent is
p = (mean + 1) / 2 > 0.5; a gaussian4 seat's intent is mean > 0.35 (kick) or mean > 0.5 (special). `sat99_*` and
`grad_scale_*` of the kick and special columns are computed over gaussian4 field players only (a probability
column has no tanh gradient scale) and are None when no such decision exists. The fire-behaviour metrics
(`keeper_special_rate`, `field_special_rate_far`, `keeper_kick_fire_holding`) read the env-space `act` columns,
so they mean the same thing for both policy heads.

Tackle counters: in `football_rl/env.py`, `self.stats["failed_tackles"]` and `self.stats["successful_tackles"]`
are incremented across the match and never reset per step, and `tools/audit/harness.py` records them after each
step; both `s_failed_tackles` and `s_successful_tackles` are therefore cumulative series, and the per-match total
is the final step's value.

Metrics
  sat99_<head>            share of field-player decisions with |mean| > 0.99 (throttle, steer, kick, special).
  grad_scale_<head>       mean of 1 - mean^2 over field-player decisions (local tanh gradient scale).
  p_kick_obs              share of eligible outfield holding decisions (hold_t >= pickup_kick_grace) with act_kick > 0.35.
  p_kick_intent           share of eligible holding decisions with kick intent (see above).
  kick_from_intent        share of eligible kicks taken on a decision with kick intent.
  p_kick_given_intent     kick rate on eligible holding decisions with kick intent.
  p_kick_given_no_intent  kick rate on eligible holding decisions without kick intent.
  power_mean, power_sd    mean and sd of clip((act_kick - 0.35) / 0.65, 0, 1) over eligible kicks.
  full_power_share        share of eligible kicks with act_kick >= 0.999.
  aim_teammate_share_*    share of eligible kicks whose direction (angle + steer * 35 deg) is within 12 deg of
                          the bearing to an outfield teammate strictly 85..500 u away (sampled and mean steer).
  special_rate            share of field-player decisions with act_special > 0.5.
  special_from_intent     share of field-player special fires taken on a decision with special intent.
  keeper_special_rate     share of keeper decisions where that keeper is not the holder, with act_special > 0.5.
  field_special_rate_far  share of field-player decisions with act_special > 0.5 while strictly more than
                          `tackle_whiff_max_ball_dist` u from the ball.
  keeper_kick_fire_holding share of keeper decisions where that keeper holds the ball and
                          hold_t >= pickup_kick_grace, with act_kick > 0.35.
  failed_tackles_pm       per-match failed tackle total (final value of cumulative `s_failed_tackles`).
  successful_tackles_pm   per-match successful tackle total (final value of cumulative `s_successful_tackles`).

Usage:
  python tools/audit/internals.py TRACE_DIR [TRACE_DIR ...] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from football_rl.config import DEFAULT_CONFIG  # noqa: E402
from tools.audit import census  # noqa: E402

__all__ = [
    "KEEPERS",
    "FIELD_PLAYERS",
    "HEAD_COLS",
    "METRIC_KEYS",
    "match_internals",
    "summarize",
    "internals_payload",
    "report",
    "main",
]

KEEPERS = (0, 6)
FIELD_PLAYERS = (1, 2, 3, 4, 5, 7, 8, 9, 10, 11)
TEAM_OUTFIELD_OFFSETS = np.arange(1, 6, dtype=np.int64)
HEAD_COLS = (("throttle", 0), ("steer", 1), ("kick", 2), ("special", 3))
TRACE_KEYS = ("mean", "act", "holder", "hold_t", "pos", "ang", "ball")
OPTIONAL_TRACE_KEYS = ("head", "s_failed_tackles", "s_successful_tackles")

KICK_THR = float(DEFAULT_CONFIG.mechanics.kick_trigger_threshold)
SPECIAL_THR = float(DEFAULT_CONFIG.mechanics.special_trigger_threshold)
PICKUP_GRACE_S = float(DEFAULT_CONFIG.mechanics.pickup_kick_grace)
TACKLE_WHIFF_MAX_BALL_DIST = float(
    getattr(
        DEFAULT_CONFIG.mechanics,
        "tackle_whiff_max_ball_dist",
        getattr(DEFAULT_CONFIG.rewards, "tackle_whiff_max_ball_dist", 180.0),
    )
)
DEFLECTION_RAD = math.radians(float(DEFAULT_CONFIG.mechanics.kick_steer_max_deflection_deg))
MIN_PASS_DIST = float(DEFAULT_CONFIG.mechanics.min_pass_distance)
AIM_MAX_DIST = 500.0
AIM_CONE_RAD = math.radians(12.0)
SAT_THRESHOLD = 0.99
FIRE_PROB_THR = 0.5
FULL_POWER_ACTION_THR = 0.999

METRIC_KEYS = (
    *(f"sat99_{h}" for h, _ in HEAD_COLS),
    *(f"grad_scale_{h}" for h, _ in HEAD_COLS),
    "p_kick_obs",
    "p_kick_intent",
    "kick_from_intent",
    "p_kick_given_intent",
    "p_kick_given_no_intent",
    "power_mean",
    "power_sd",
    "full_power_share",
    "aim_teammate_share_sampled",
    "aim_teammate_share_mean",
    "special_rate",
    "special_from_intent",
    "keeper_special_rate",
    "field_special_rate_far",
    "keeper_kick_fire_holding",
    "failed_tackles_pm",
    "successful_tackles_pm",
)


def _aim_share(
    pos: np.ndarray,
    ang: np.ndarray,
    steer: np.ndarray,
    ki: np.ndarray,
    kp: np.ndarray,
) -> float:
    """Fraction of kicks `(ki, kp)` aimed within 12 deg of an outfield teammate 85..500 u away."""
    kt = kp // census.TEAM_SIZE
    mates = kt[:, None] * census.TEAM_SIZE + TEAM_OUTFIELD_OFFSETS[None, :]
    is_other = mates != kp[:, None]
    delta = pos[ki[:, None], mates] - pos[ki, kp][:, None, :]
    dist = np.hypot(delta[:, :, 0], delta[:, :, 1])
    bearing = np.arctan2(delta[:, :, 1], delta[:, :, 0])
    kick_dir = ang[ki, kp] + steer * DEFLECTION_RAD
    err = np.abs((bearing - kick_dir[:, None] + math.pi) % (2.0 * math.pi) - math.pi)
    hit = is_other & (err < AIM_CONE_RAD) & (dist > MIN_PASS_DIST) & (dist < AIM_MAX_DIST)
    return float(hit.any(axis=1).mean())


def match_internals(trace: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Compute per-match policy-internal metrics from one harness trace."""
    holder = np.asarray(trace["holder"], dtype=np.int64)
    n_steps = int(holder.shape[0])
    if n_steps == 0:
        rec: Dict[str, Any] = {"n_steps": 0, "n_eligible": 0, "n_kicks": 0, "n_special_fires": 0}
        for key in METRIC_KEYS:
            rec[key] = None
        return rec

    mean = np.asarray(trace["mean"], dtype=np.float64)
    act = np.asarray(trace["act"], dtype=np.float64)
    hold_t = np.asarray(trace["hold_t"], dtype=np.float64)
    pos = np.asarray(trace["pos"], dtype=np.float64)
    ang = np.asarray(trace["ang"], dtype=np.float64)

    team_head = np.asarray(trace["head"], dtype=np.int64) if "head" in trace else np.zeros(2, dtype=np.int64)
    seat_hybrid = np.repeat(team_head == 1, census.TEAM_SIZE)
    field_hybrid = seat_hybrid[list(FIELD_PLAYERS)]

    rec = {"n_steps": n_steps}
    field_mean = mean[:, FIELD_PLAYERS, :]
    for head, col in HEAD_COLS:
        m_col = field_mean[:, :, col]
        if col >= 2:
            m_col = m_col[:, ~field_hybrid]
        if m_col.size == 0:
            rec[f"sat99_{head}"] = None
            rec[f"grad_scale_{head}"] = None
            continue
        rec[f"sat99_{head}"] = float((np.abs(m_col) > SAT_THRESHOLD).mean())
        rec[f"grad_scale_{head}"] = float((1.0 - m_col * m_col).mean())

    elig_mask = (holder >= 0) & (holder % census.TEAM_SIZE != 0) & (hold_t >= PICKUP_GRACE_S)
    n_eligible = int(elig_mask.sum())
    rec["n_eligible"] = n_eligible

    if n_eligible == 0:
        rec["n_kicks"] = 0
        for key in (
            "p_kick_obs",
            "p_kick_intent",
            "kick_from_intent",
            "p_kick_given_intent",
            "p_kick_given_no_intent",
            "power_mean",
            "power_sd",
            "full_power_share",
            "aim_teammate_share_sampled",
            "aim_teammate_share_mean",
        ):
            rec[key] = None
    else:
        idx = np.flatnonzero(elig_mask)
        pl = holder[idx]
        mk = mean[idx, pl, 2]
        ak = act[idx, pl, 2]
        kicked = ak > KICK_THR
        intent = np.where(seat_hybrid[pl], (mk + 1.0) / 2.0 > FIRE_PROB_THR, mk > KICK_THR)
        n_kicks = int(kicked.sum())
        n_intent = int(intent.sum())
        n_no_intent = n_eligible - n_intent
        kicked_intent = int((kicked & intent).sum())
        kicked_no_intent = n_kicks - kicked_intent

        rec["n_kicks"] = n_kicks
        rec["p_kick_obs"] = float(n_kicks / n_eligible)
        rec["p_kick_intent"] = float(n_intent / n_eligible)
        rec["kick_from_intent"] = float(kicked_intent / n_kicks) if n_kicks > 0 else None
        rec["p_kick_given_intent"] = float(kicked_intent / n_intent) if n_intent > 0 else None
        rec["p_kick_given_no_intent"] = float(kicked_no_intent / n_no_intent) if n_no_intent > 0 else None

        if n_kicks > 0:
            power = np.clip((ak[kicked] - KICK_THR) / (1.0 - KICK_THR), 0.0, 1.0)
            rec["power_mean"] = float(power.mean())
            rec["power_sd"] = float(power.std())
            rec["full_power_share"] = float((ak[kicked] >= FULL_POWER_ACTION_THR).mean())
            ki = idx[kicked]
            kp = pl[kicked]
            rec["aim_teammate_share_sampled"] = _aim_share(pos, ang, act[ki, kp, 1], ki, kp)
            rec["aim_teammate_share_mean"] = _aim_share(pos, ang, mean[ki, kp, 1], ki, kp)
        else:
            rec["power_mean"] = None
            rec["power_sd"] = None
            rec["full_power_share"] = None
            rec["aim_teammate_share_sampled"] = None
            rec["aim_teammate_share_mean"] = None

    sp_a = act[:, FIELD_PLAYERS, 3]
    sp_m = field_mean[:, :, 3]
    fire = sp_a > SPECIAL_THR
    n_fires = int(fire.sum())
    sp_intent = np.where(field_hybrid[None, :], (sp_m + 1.0) / 2.0 > FIRE_PROB_THR, sp_m > SPECIAL_THR)
    rec["n_special_fires"] = n_fires
    rec["special_rate"] = float(fire.mean())
    rec["special_from_intent"] = float(sp_intent[fire].mean()) if n_fires > 0 else None

    ball = np.asarray(trace["ball"], dtype=np.float64)
    field_delta = pos[:, FIELD_PLAYERS, :] - ball[:, None, :]
    field_dist = np.hypot(field_delta[:, :, 0], field_delta[:, :, 1])
    rec["field_special_rate_far"] = float((fire & (field_dist > TACKLE_WHIFF_MAX_BALL_DIST)).mean())

    keeper_not_holder = holder[:, None] != np.asarray(KEEPERS, dtype=np.int64)[None, :]
    if keeper_not_holder.any():
        keeper_fire = act[:, KEEPERS, 3] > SPECIAL_THR
        rec["keeper_special_rate"] = float(keeper_fire[keeper_not_holder].mean())
    else:
        rec["keeper_special_rate"] = None

    gk_elig_mask = (holder >= 0) & (holder % census.TEAM_SIZE == 0) & (hold_t >= PICKUP_GRACE_S)
    if gk_elig_mask.any():
        gk_idx = np.flatnonzero(gk_elig_mask)
        gk_pl = holder[gk_idx]
        rec["keeper_kick_fire_holding"] = float((act[gk_idx, gk_pl, 2] > KICK_THR).mean())
    else:
        rec["keeper_kick_fire_holding"] = None

    rec["failed_tackles_pm"] = (
        float(np.asarray(trace["s_failed_tackles"], dtype=np.float64)[-1])
        if "s_failed_tackles" in trace
        else None
    )
    rec["successful_tackles_pm"] = (
        float(np.asarray(trace["s_successful_tackles"], dtype=np.float64)[-1])
        if "s_successful_tackles" in trace
        else None
    )
    return rec


def summarize(per_match: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-match internal records into `{n_matches, metrics}` with mean/se/n."""
    metrics: Dict[str, Dict[str, Any]] = {}
    for key in METRIC_KEYS:
        mean, se, n = census.mean_se([m[key] for m in per_match])
        metrics[key] = {"mean": mean, "se": se, "n": n}
    return {"n_matches": len(per_match), "metrics": metrics}


def internals_payload(per_match: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the JSON-serializable payload (`summary` + `per_match`)."""
    return {
        "summary": summarize(per_match),
        "per_match": [{k: census._json_safe(v) for k, v in m.items()} for m in per_match],
    }


def report(per_match: Sequence[Dict[str, Any]], title: str = "") -> str:
    """Printable summary of per-match policy internals (mean +- SE with n)."""
    s = summarize(per_match)
    met = s["metrics"]
    lines = [f"===== {title} internals  n={s['n_matches']} matches"]
    for head, _ in HEAD_COLS:
        lines.append(
            f"{head:8s} sat99 / grad_scale : "
            f"{census._fmt_d(met[f'sat99_{head}'], 3)} / {census._fmt_d(met[f'grad_scale_{head}'], 3)}"
        )
    lines.append(
        "kick decomp (obs / intent / from_int): "
        + " / ".join(census._fmt_d(met[k], 3) for k in ("p_kick_obs", "p_kick_intent", "kick_from_intent"))
    )
    lines.append(
        "kick cond   (P|intent / P|no_intent) : "
        + " / ".join(census._fmt_d(met[k], 3) for k in ("p_kick_given_intent", "p_kick_given_no_intent"))
    )
    lines.append(
        "power       (mean / sd / full_share) : "
        + " / ".join(census._fmt_d(met[k], 3) for k in ("power_mean", "power_sd", "full_power_share"))
    )
    lines.append(
        "aim_teammate(sampled / mean)         : "
        + " / ".join(census._fmt_d(met[k], 3) for k in ("aim_teammate_share_sampled", "aim_teammate_share_mean"))
    )
    lines.append(
        "special     (rate / from_intent)     : "
        + " / ".join(census._fmt_d(met[k], 3) for k in ("special_rate", "special_from_intent"))
    )
    lines.append(
        "keeper      (spec / kick_holding)    : "
        + " / ".join(census._fmt_d(met[k], 3) for k in ("keeper_special_rate", "keeper_kick_fire_holding"))
    )
    lines.append(
        "field_spec  (far)                    : "
        + census._fmt_d(met["field_special_rate_far"], 3)
    )
    lines.append(
        "tackles/m   (failed / successful)    : "
        + " / ".join(census._fmt_d(met[k], 2) for k in ("failed_tackles_pm", "successful_tackles_pm"))
    )
    return "\n".join(lines)


def _iter_internal_traces(directory: Path | str) -> Iterator[Dict[str, np.ndarray]]:
    for path in census._trace_paths(directory):
        with np.load(path) as z:
            yield {k: z[k] for k in TRACE_KEYS + tuple(k for k in OPTIONAL_TRACE_KEYS if k in z.files)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace_dirs", nargs="+", help="directories of m*.npz traces written by harness.py")
    ap.add_argument("--json", default=None, help="write the summary and per-match records of every directory here")
    args = ap.parse_args(argv)
    dump: Dict[str, Any] = {}
    for directory in args.trace_dirs:
        per_match = [match_internals(t) for t in _iter_internal_traces(directory)]
        print(report(per_match, title=str(directory)))
        dump[str(directory)] = internals_payload(per_match)
    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(dump, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
