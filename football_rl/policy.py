"""Actor-critic network, its action heads (Gaussian and hybrid fire) and the return statistics it carries."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.distributions.normal import Normal
import torch.nn as nn
import torch.nn.functional as F

from football_rl.config import ACTION_DIM, CENTRAL_STATE_DIM, OBS_DIM, _gaussian4_fire_rate, _initial_fire_logit


def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal initialization for PPO linear layers."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


_HALF_LOG_TWO_PI = math.log(math.sqrt(2 * math.pi))

POLICY_HEADS = ("gaussian4", "hybrid5", "hybrid6", "hybrid7")
# Heads that fire kick and special as Bernoulli draws (they share the fire telemetry).
FIRE_HEADS = ("hybrid5", "hybrid6", "hybrid7")
# Observation column that is 1.0 on the seat holding the ball and 0.0 on every other seat.
HOLDING_OBS_INDEX = 7
# Observation column holding ball_hold_timer / 5.0 for the carrier.
HOLD_TIME_OBS_INDEX = 8
# Fire telemetry is read from every this-many-th decision of a role's rollout batch. The subsample adds sampling noise
# to the saturation shares: at most about 3 percentage points of standard error for a keeper batch (300 sampled rows
# at 8 environments x 300 decisions) and about 1 for a field batch (1500 sampled rows).
FIRE_STATS_STRIDE = 8
# hybrid5 policy action: [throttle, steer, power sample u, kick fire K, special fire S].
HYBRID_ACTION_DIM = 5
# hybrid6 policy action: [throttle, steer, power sample u, aim a, kick fire K, special fire S].
HYBRID6_ACTION_DIM = 6
# hybrid7 policy action: [throttle, steer, power sample u, aim a, target T, kick fire K, special fire S].
HYBRID7_ACTION_DIM = 7
# hybrid7 output logits: [z_thr, z_steer, z_pow, z_aim, z_kick, z_spec, z_tm1..z_tm5, z_free].
HYBRID7_LOGIT_DIM = 12
NUM_KICK_TARGETS = 6
KICK_TARGET_FREE = 5
# Half of one physics substep in normalized hold-time units (1/60 s / 5.0 s = 1/300 -> midpoint 1/600).
KICK_READY_OBS_MIN = 1.0 / 600.0
# Kickable deflection arc in normalized bearing units (rel_ang / pi for +-35 deg).
KICK_ARC_OBS = 35.0 / 180.0
# Observation indices of teammate relative bearings (entity slots 1..5: 16 + 5*s + 2 = 18 + 5*s -> 23, 28, 33, 38, 43).
TEAMMATE_REL_ANG_SLICE = slice(23, 44, 5)
# Initial hybrid5 power-head bias: the mean power (1 + tanh(bias)) / 2 is 0.4666.
HYBRID_POWER_BIAS = -0.067


def policy_head_of_weights(weights: Dict[str, torch.Tensor]) -> Optional[str]:
    """Head of an `ActorCritic` state dict from the width of its output layer (4 -> gaussian4, 5 -> hybrid5,
    6 -> hybrid6, 12 -> hybrid7), else None."""
    out = weights.get("actor_mean.4.weight")
    if out is None or out.dim() != 2:
        return None
    return {
        ACTION_DIM: "gaussian4",
        HYBRID_ACTION_DIM: "hybrid5",
        HYBRID6_ACTION_DIM: "hybrid6",
        HYBRID7_LOGIT_DIM: "hybrid7",
    }.get(int(out.shape[0]))


def validate_hybrid7_env_config(cfg: Any) -> None:
    """Verify the environment constants relied on by `kick_targets` for `hybrid7`, raising ValueError on mismatch."""
    phys = cfg.physics
    mech = cfg.mechanics
    # KICK_READY_OBS_MIN is half of one 1/60 s substep in obs[8] units; another substep length moves the k = 1 hold.
    if phys.dt != 1.0 / 60.0:
        raise ValueError(f"hybrid7 requires dt == 1/60 s, got {phys.dt}")
    min_substeps = math.ceil(mech.pickup_kick_grace / phys.dt - phys.frame_skip - 1e-12)
    if min_substeps != 1:
        raise ValueError(
            f"hybrid7 requires ceil(pickup_kick_grace / dt - frame_skip) == 1 substep, "
            f"got {min_substeps} from pickup_kick_grace={mech.pickup_kick_grace}, dt={phys.dt}, frame_skip={phys.frame_skip}"
        )
    if mech.field_max_hold_time != 5.0:
        raise ValueError(f"hybrid7 requires field_max_hold_time == 5.0, got {mech.field_max_hold_time}")
    if mech.keeper_max_hold_time != 5.0:
        raise ValueError(f"hybrid7 requires keeper_max_hold_time == 5.0, got {mech.keeper_max_hold_time}")
    if mech.kick_steer_max_deflection_deg != 35.0:
        raise ValueError(
            f"hybrid7 requires kick_steer_max_deflection_deg == 35.0, got {mech.kick_steer_max_deflection_deg}"
        )


def kick_targets(obs: Any) -> Tuple[Any, Any, Any]:
    """Compute `(ready, valid, tm_steer)` from raw observations `obs` (`np.ndarray` or `torch.Tensor`).

    Readiness derivation from the substep order (`physics.py` `_physics_substep`, 4 substeps per decision at dt = 1/60 s):
    each substep decrements `kick_grace_timer` and increments `ball_hold_timer` before checking whether a kick fires
    (`kick_grace_timer <= 0`). A free-ball pickup or successful tackle sets `ball_hold_timer = 0` and
    `kick_grace_timer = 0.08` together after the action pass, so at a decision row after k substeps held,
    `hold = k/60` and `grace = max(0, 0.08 - k/60)`. On substep j in 1..4 of the next decision `grace = 0.08 - (k + j)/60`,
    which reaches <= 0 iff `k + j >= 5`. Thus a kick can fire on some substep of the next decision iff `k >= 1`:
    k = 0 is blocked on all four substeps, k = 1 fires on substep 4, and k >= 4 fires on substep 1. Because `obs[..., 8]`
    stores `ball_hold_timer / 5.0`, k = 0 gives 0.0 and k = 1 gives 1/300; `KICK_READY_OBS_MIN = 1/600` separates them.
    Kickoffs (which clear both timers to 0) and keeper perimeter strips (which reset hold without resetting grace) are
    exceptions the observation cannot see and are treated as not ready (`ready = False`).

    Teammate slots 1..5 carry `rel_ang / pi` at columns 23, 28, 33, 38, 43 (`18 + 5*s`). A teammate slot is valid iff
    `|rel_ang / pi| <= 35/180` (`KICK_ARC_OBS`), FREE (index 5) is always valid, and `tm_steer = rel_ang * (180 / 35)`.
    """
    if torch.is_tensor(obs):
        if obs.shape[-1] <= 43:
            raise ValueError(f"obs last dimension must exceed 43 for kick_targets, got {obs.shape[-1]}")
        ready = (obs[..., HOLDING_OBS_INDEX] > 0.5) & (obs[..., HOLD_TIME_OBS_INDEX] > KICK_READY_OBS_MIN)
        rel = obs[..., TEAMMATE_REL_ANG_SLICE]
        valid = F.pad(rel.abs() <= KICK_ARC_OBS, (0, 1), value=True)
        tm_steer = rel * (180.0 / 35.0)
        return ready, valid, tm_steer
    arr = np.asarray(obs)
    if arr.shape[-1] <= 43:
        raise ValueError(f"obs last dimension must exceed 43 for kick_targets, got {arr.shape[-1]}")
    ready = (arr[..., HOLDING_OBS_INDEX] > 0.5) & (arr[..., HOLD_TIME_OBS_INDEX] > KICK_READY_OBS_MIN)
    rel = arr[..., TEAMMATE_REL_ANG_SLICE]
    valid = np.empty((*arr.shape[:-1], NUM_KICK_TARGETS), dtype=np.bool_)
    valid[..., :5] = np.abs(rel) <= KICK_ARC_OBS
    valid[..., KICK_TARGET_FREE] = True
    tm_steer = rel * (180.0 / 35.0)
    return ready, valid, tm_steer


def masked_target_log_probs_and_entropy(
    target_logits: torch.Tensor, valid: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """`(log_pi, H_pi)` of the masked categorical over `{TM_1..5, FREE}` without `-inf * 0` or NaN in values or gradients."""
    masked = target_logits.masked_fill(~valid, float("-inf"))
    log_pi = F.log_softmax(masked, dim=-1)
    zero = torch.zeros_like(log_pi)
    pi = torch.where(valid, log_pi.exp(), zero)
    log_pi_safe = torch.where(valid, log_pi, zero)
    h_pi = -torch.where(valid, pi * log_pi_safe, zero).sum(dim=-1)
    return log_pi, h_pi


def sample_kick_target(
    target_logits: torch.Tensor,
    valid: torch.Tensor,
    deterministic: bool = False,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample a kick target index in `0..5` from the masked categorical, or take the masked argmax when `deterministic`."""
    if target_logits.numel() == 0:
        return torch.empty(target_logits.shape[:-1], dtype=torch.int64, device=target_logits.device)
    masked = target_logits.masked_fill(~valid, float("-inf"))
    if deterministic:
        return masked.argmax(dim=-1)
    probs = F.softmax(masked, dim=-1)
    flat = probs.reshape(-1, NUM_KICK_TARGETS)
    drawn = torch.multinomial(flat, 1, generator=generator).squeeze(-1)
    return drawn.reshape(target_logits.shape[:-1])


def fire_probability(z: torch.Tensor, floor: Any, bound: Any) -> torch.Tensor:
    """Bounded Bernoulli probability `floor + (1 - 2*floor) * sigmoid(bound * tanh(z / bound))`, inside [floor, 1 - floor]."""
    return floor + (1.0 - 2.0 * floor) * torch.sigmoid(bound * torch.tanh(z / bound))


def to_env_actions(
    policy_actions: np.ndarray,
    kick_threshold: float,
    holding: Optional[np.ndarray] = None,
    obs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Decode hybrid5 `[thr, steer, u, K, S]`, hybrid6 `[thr, steer, u, a, K, S]` or hybrid7 `[thr, steer, u, a, T, K, S]`
    policy actions into the env's `[thr, steer, kick, special]`.

    Kick is `-1` unless K = 1, then `thr + (1 - thr) * max(power, 0.001)` with `thr = kick_threshold` and
    `power = clip((u + 1) / 2, 0, 1)`, so a fired kick is always strictly above the env threshold. Special is
    `+1` when S = 1 and `-1` otherwise. Throttle passes through unchanged. The env steer is `steer`, except:
    - hybrid6: on rows where the seat holds the ball (`holding`) and fires the kick, it is the aim `a`;
    - hybrid7: requires `obs` (matching leading shape), computes `(ready, valid, tm_steer) = kick_targets(obs)`, and on
      rows where `K > 0.5` and `ready`, sets the env steer to `tm_steer[T]` for `T < 5` or `a` for `T = 5`, raising
      ValueError if `T` is not valid on a ready kick row.
    """
    pol = np.asarray(policy_actions)
    width = pol.shape[-1]
    if width not in (HYBRID_ACTION_DIM, HYBRID6_ACTION_DIM, HYBRID7_ACTION_DIM):
        raise ValueError(
            f"expected {HYBRID_ACTION_DIM}, {HYBRID6_ACTION_DIM} or {HYBRID7_ACTION_DIM} policy action components, got {width}"
        )
    fire = 3 if width == HYBRID_ACTION_DIM else 4 if width == HYBRID6_ACTION_DIM else 5
    out = np.empty(pol.shape[:-1] + (ACTION_DIM,), dtype=pol.dtype)
    out[..., 0] = pol[..., 0]
    if width == HYBRID_ACTION_DIM:
        out[..., 1] = pol[..., 1]
    elif width == HYBRID6_ACTION_DIM:
        if holding is None:
            raise ValueError("hybrid6 policy actions need the holding flag of every seat to decode the steer")
        holding = np.asarray(holding)
        if holding.shape != pol.shape[:-1]:
            raise ValueError(f"holding has shape {holding.shape}, expected {pol.shape[:-1]}")
        if holding.dtype != np.bool_:
            holding = holding > 0.5
        out[..., 1] = np.where((pol[..., fire] > 0.5) & holding, pol[..., 3], pol[..., 1])
    else:
        if obs is None:
            raise ValueError("hybrid7 policy actions need obs to decode the kick target")
        obs_arr = np.asarray(obs)
        if obs_arr.shape[:-1] != pol.shape[:-1]:
            raise ValueError(f"obs has leading shape {obs_arr.shape[:-1]}, expected {pol.shape[:-1]}")
        ready, valid, tm_steer = kick_targets(obs_arr)
        ready_kick = (pol[..., fire] > 0.5) & ready
        out[..., 1] = pol[..., 1]
        if np.any(ready_kick):
            t_raw = pol[ready_kick, 4]
            t_int = np.rint(t_raw).astype(np.int64)
            v_act = valid[ready_kick]
            in_range = (
                np.isfinite(t_raw)
                & (t_int >= 0)
                & (t_int < NUM_KICK_TARGETS)
                & (np.abs(t_raw - t_int) <= 1e-4)
            )
            if not np.all(in_range) or not np.all(v_act[np.arange(t_int.size), t_int]):
                raise ValueError("hybrid7 ready kick selected an invalid target")
            tm_act = tm_steer[ready_kick]
            chosen_tm = tm_act[np.arange(t_int.size), np.minimum(t_int, KICK_TARGET_FREE - 1)]
            out[ready_kick, 1] = np.where(t_int == KICK_TARGET_FREE, pol[ready_kick, 3], chosen_tm)
    power = np.maximum(np.clip((pol[..., 2] + 1.0) / 2.0, 0.0, 1.0), 0.001)
    out[..., 2] = np.where(pol[..., fire] > 0.5, kick_threshold + (1.0 - kick_threshold) * power, -1.0)
    out[..., 3] = np.where(pol[..., fire + 1] > 0.5, 1.0, -1.0)
    return out


def _holding_mask(holding: Optional[torch.Tensor], shape: torch.Size) -> torch.Tensor:
    """Boolean holding flag of each row of a hybrid6 batch; ValueError when it is missing or has another shape."""
    if holding is None:
        raise ValueError("the hybrid6 head needs the holding flag of every row")
    if holding.shape != shape:
        raise ValueError(f"holding has shape {tuple(holding.shape)}, expected {tuple(shape)}")
    return holding if holding.dtype == torch.bool else holding > 0.5


def _resolve_kick_targets_torch(
    shape: torch.Size,
    obs: Optional[torch.Tensor],
    ready: Optional[torch.Tensor],
    valid: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resolve `(ready, valid)` for a hybrid7 batch from explicit tensors or `obs`, validating shapes."""
    if ready is None or valid is None:
        if obs is None:
            raise ValueError("the hybrid7 head needs obs or (ready, valid) of every row")
        if obs.shape[:-1] != shape:
            raise ValueError(f"obs has leading shape {tuple(obs.shape[:-1])}, expected {tuple(shape)}")
        ready, valid, _ = kick_targets(obs)
    else:
        if ready.shape != shape:
            raise ValueError(f"ready has shape {tuple(ready.shape)}, expected {tuple(shape)}")
        if valid.shape != (*shape, NUM_KICK_TARGETS):
            raise ValueError(f"valid has shape {tuple(valid.shape)}, expected {(*shape, NUM_KICK_TARGETS)}")
    return ready, valid


def _hybrid_log_prob(
    mean: torch.Tensor,
    logstd: torch.Tensor,
    p: torch.Tensor,
    action: torch.Tensor,
    holding: Optional[torch.Tensor] = None,
    *,
    target_logits: Optional[torch.Tensor] = None,
    obs: Optional[torch.Tensor] = None,
    ready: Optional[torch.Tensor] = None,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Log-probability of a hybrid5, hybrid6 or hybrid7 `action`; `p[..., 0]` and `p[..., 1]` are the kick and special
    fire probabilities.

    With `target_logits` (hybrid7), ready fired kicks (`K * c = 1`) use the masked target categorical plus the aim
    Gaussian when `T = FREE`; all other rows use the steer Gaussian. With four Gaussians and no `target_logits`
    (hybrid6), the aim Gaussian replaces steer on rows where a holder fired (`K * h = 1`). hybrid5 ignores `holding`.
    """
    if target_logits is not None:
        r, v = _resolve_kick_targets_torch(action.shape[:-1], obs, ready, valid)
        return _hybrid7_log_prob(mean, logstd, p, target_logits, action, r, v)
    if mean.shape[-1] == 4:
        return _hybrid6_log_prob(mean, logstd, p, action, _holding_mask(holding, action.shape[:-1]))
    kick = action[..., 3]
    log_n = -0.5 * ((action[..., :3] - mean) * torch.exp(-logstd)) ** 2 - logstd - _HALF_LOG_TWO_PI
    log_fire = torch.log(torch.where(action[..., 3:] > 0.5, p, 1.0 - p)).sum(-1)
    return log_n[..., 0] + log_n[..., 1] + kick * log_n[..., 2] + log_fire


def _hybrid6_log_prob(
    mean: torch.Tensor, logstd: torch.Tensor, p: torch.Tensor, action: torch.Tensor, holding: torch.Tensor
) -> torch.Tensor:
    """`logN(thr) + (1 - K h) logN(steer) + K h logN(a) + K logN(u) + log Bern(K) + log Bern(S)` of `[thr, steer, u, a, K, S]`."""
    kick = action[..., 4]
    log_n = -0.5 * ((action[..., :4] - mean) * torch.exp(-logstd)) ** 2 - logstd - _HALF_LOG_TWO_PI
    log_fire = torch.log(torch.where(action[..., 4:] > 0.5, p, 1.0 - p)).sum(-1)
    direction = torch.where((kick > 0.5) & holding, log_n[..., 3], log_n[..., 1])
    return log_n[..., 0] + direction + kick * log_n[..., 2] + log_fire


def _validate_hybrid7_targets(
    action: torch.Tensor, ready: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Validate stored target column `action[..., 4]` on ready-kick rows; return safe int64 indices."""
    kick_fired = action[..., 5] > 0.5
    active = kick_fired & ready
    target = action[..., 4]
    t_int = torch.round(target).to(torch.int64)
    in_range = (
        torch.isfinite(target)
        & (t_int >= 0)
        & (t_int < NUM_KICK_TARGETS)
        & ((target - t_int.to(target.dtype)).abs() <= 1e-4)
    )
    free_idx = torch.full_like(t_int, KICK_TARGET_FREE)
    t_safe = torch.where(in_range, t_int, free_idx)
    t_valid = in_range & valid.gather(-1, t_safe.unsqueeze(-1)).squeeze(-1)
    if bool((active & ~t_valid).any()):
        raise ValueError("hybrid7 ready kick has an invalid target")
    return t_safe


def _hybrid7_log_prob(
    mean: torch.Tensor,
    logstd: torch.Tensor,
    p: torch.Tensor,
    target_logits: torch.Tensor,
    action: torch.Tensor,
    ready: torch.Tensor,
    valid: torch.Tensor,
    validate_targets: bool = True,
) -> torch.Tensor:
    """`logN(thr) + (1 - K c) logN(steer) + K c (log pi(T|valid) + [T=5] logN(a)) + K logN(u) + log Bern(K) + log Bern(S)`."""
    kick_fired = action[..., 5] > 0.5
    active = kick_fired & ready
    if validate_targets:
        t_safe = _validate_hybrid7_targets(action, ready, valid)
    else:
        t_safe = torch.round(action[..., 4]).to(torch.int64).clamp(0, KICK_TARGET_FREE)

    masked = target_logits.masked_fill(~valid, float("-inf"))
    log_pi = F.log_softmax(masked, dim=-1)
    free_idx = torch.full_like(t_safe, KICK_TARGET_FREE)
    t_gather = torch.where(active, t_safe, free_idx)
    log_pi_t = log_pi.gather(-1, t_gather.unsqueeze(-1)).squeeze(-1)

    log_n = -0.5 * ((action[..., :4] - mean) * torch.exp(-logstd)) ** 2 - logstd - _HALF_LOG_TWO_PI
    log_fire = torch.log(torch.where(action[..., 5:] > 0.5, p, 1.0 - p)).sum(-1)
    zero = torch.zeros_like(log_n[..., 0])
    is_free = t_safe == KICK_TARGET_FREE
    kick_dir = torch.where(is_free, log_pi_t + log_n[..., 3], log_pi_t)
    direction = torch.where(active, kick_dir, log_n[..., 1])
    power_term = torch.where(kick_fired, log_n[..., 2], zero)
    return log_n[..., 0] + direction + power_term + log_fire


def _hybrid7_sample_and_log_prob(
    mean: torch.Tensor,
    logstd: torch.Tensor,
    p: torch.Tensor,
    target_logits: torch.Tensor,
    ready: torch.Tensor,
    valid: torch.Tensor,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a 7-wide hybrid7 action and its log-probability in one masked-target softmax pass.

    Draws in order: Gaussian `(thr, steer, u, a)`, Bernoulli `(K, S)`, then masked categorical `T` (or masked argmax
    when `deterministic`). Log-probability matches `_hybrid7_log_prob` identically.
    """
    if deterministic:
        gauss = mean
        fires = (p > 0.5).to(mean.dtype)
    else:
        gauss = torch.normal(mean, torch.exp(logstd))
        fires = torch.bernoulli(p)
    masked = target_logits.masked_fill(~valid, float("-inf"))
    log_pi = F.log_softmax(masked, dim=-1)
    if deterministic:
        t_int = masked.argmax(dim=-1)
    else:
        flat = log_pi.exp().reshape(-1, NUM_KICK_TARGETS)
        t_int = torch.multinomial(flat, 1).squeeze(-1).reshape(target_logits.shape[:-1])
    action = torch.cat([gauss, t_int.to(mean.dtype).unsqueeze(-1), fires], dim=-1)

    kick = fires[..., 0]
    log_n = -0.5 * ((gauss - mean) * torch.exp(-logstd)) ** 2 - logstd - _HALF_LOG_TWO_PI
    log_fire = torch.log(torch.where(fires > 0.5, p, 1.0 - p)).sum(-1)
    active = (kick > 0.5) & ready
    if bool(active.any()):
        log_pi_t = log_pi.gather(-1, t_int.unsqueeze(-1)).squeeze(-1)
        kick_dir = torch.where(t_int == KICK_TARGET_FREE, log_pi_t + log_n[..., 3], log_pi_t)
        direction = torch.where(active, kick_dir, log_n[..., 1])
    else:
        direction = log_n[..., 1]
    return action, log_n[..., 0] + direction + kick * log_n[..., 2] + log_fire


def _hybrid_entropy(
    logstd: torch.Tensor,
    p: torch.Tensor,
    holding: Optional[torch.Tensor] = None,
    *,
    ready: Optional[torch.Tensor] = None,
    is_free: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Entropy of the hybrid5 head distribution: two Gaussians, two Bernoullis and `p_kick` times the power Gaussian.

    hybrid6 (four Gaussians) adds `p_kick * h` times the aim Gaussian, with h the row's holding flag.
    hybrid7 (`ready` and `is_free` given) replaces `h` with `c * [T = FREE]` and omits the target categorical entropy.
    """
    h_n = 0.5 + _HALF_LOG_TWO_PI + logstd
    h_b = -(p * torch.log(p) + (1.0 - p) * torch.log1p(-p)).sum(-1)
    if ready is not None:
        if is_free is None:
            raise ValueError("hybrid7 entropy requires is_free alongside ready")
        aim_w = torch.where(ready & is_free, p[..., 0], torch.zeros_like(p[..., 0]))
        return h_n[..., 0] + h_n[..., 1] + h_b + p[..., 0] * h_n[..., 2] + aim_w * h_n[..., 3]
    if logstd.shape[-1] == 4:
        hold = _holding_mask(holding, p.shape[:-1]).to(p.dtype)
        return h_n[..., 0] + h_n[..., 1] + h_b + p[..., 0] * h_n[..., 2] + p[..., 0] * hold * h_n[..., 3]
    return h_n[..., 0] + h_n[..., 1] + h_b + p[..., 0] * h_n[..., 2]


def hybrid_log_prob_entropy(
    mean: torch.Tensor,
    logstd: torch.Tensor,
    p_kick: torch.Tensor,
    p_spec: torch.Tensor,
    action: torch.Tensor,
    holding: Optional[torch.Tensor] = None,
    *,
    target_logits: Optional[torch.Tensor] = None,
    obs: Optional[torch.Tensor] = None,
    ready: Optional[torch.Tensor] = None,
    valid: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Log-probability of a hybrid5, hybrid6 or hybrid7 `action` and the logged entropy of the head distribution."""
    p = torch.stack([p_kick, p_spec], dim=-1)
    if target_logits is not None:
        r, v = _resolve_kick_targets_torch(action.shape[:-1], obs, ready, valid)
        lp = _hybrid7_log_prob(mean, logstd, p, target_logits, action, r, v)
        is_free = torch.round(action[..., 4]).to(torch.int64) == KICK_TARGET_FREE
        return lp, _hybrid_entropy(logstd, p, ready=r, is_free=is_free)
    return _hybrid_log_prob(mean, logstd, p, action, holding), _hybrid_entropy(logstd, p, holding)


def hybrid_entropy_bonus(
    logstd: torch.Tensor,
    p: torch.Tensor,
    ent_coef: float,
    fire_entropy: str,
    fire_ent_coef: float,
    holding: Optional[torch.Tensor] = None,
    *,
    target_logits: Optional[torch.Tensor] = None,
    target_ent_coef: float = 0.0,
    target: Optional[torch.Tensor] = None,
    obs: Optional[torch.Tensor] = None,
    ready: Optional[torch.Tensor] = None,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-sample entropy bonus of the hybrid5, hybrid6 or hybrid7 loss; `p[..., 0]` is kick and `p[..., 1]` special.

    `joint` (hybrid5 / hybrid6): `ent_coef` times the full head entropy. `separate`: `ent_coef` times the Gaussian
    terms (power weighted by detached p_kick; hybrid6 aim by detached p_kick * h; hybrid7 aim by detached p_kick * c *
    [T = FREE] where T is the stored sample), plus `fire_ent_coef` times the two Bernoulli entropies, plus (for
    hybrid7) `target_ent_coef * p_kick.detach() * c * [n_valid > 1] * H_pi`. Callers reduce this with `.mean()` over
    every row of the minibatch.
    """
    if target_logits is not None:
        if fire_entropy != "separate":
            raise ValueError(f"hybrid7 requires fire_entropy='separate', got {fire_entropy!r}")
        if target is None:
            raise ValueError("hybrid7 entropy bonus requires the stored target T")
        r, v = _resolve_kick_targets_torch(p.shape[:-1], obs, ready, valid)
        h_n = 0.5 + _HALF_LOG_TWO_PI + logstd
        h_b = -(p * torch.log(p) + (1.0 - p) * torch.log1p(-p)).sum(-1)
        p_hat = p[..., 0].detach()
        zero = torch.zeros_like(p_hat)
        is_free = (torch.round(target).to(torch.int64) == KICK_TARGET_FREE).detach()
        aim_term = torch.where(r & is_free, p_hat * h_n[..., 3], zero)
        gaussian = h_n[..., 0] + h_n[..., 1] + p_hat * h_n[..., 2] + aim_term
        _, h_pi = masked_target_log_probs_and_entropy(target_logits, v)
        multi = r & (v.sum(dim=-1) > 1)
        target_term = torch.where(multi, p_hat * h_pi, zero)
        return ent_coef * gaussian + target_ent_coef * target_term + fire_ent_coef * h_b
    if fire_entropy == "joint":
        return ent_coef * _hybrid_entropy(logstd, p, holding)
    if fire_entropy != "separate":
        raise ValueError(f"fire_entropy must be 'joint' or 'separate', got {fire_entropy!r}")
    h_n = 0.5 + _HALF_LOG_TWO_PI + logstd
    h_b = -(p * torch.log(p) + (1.0 - p) * torch.log1p(-p)).sum(-1)
    gaussian = h_n[..., 0] + h_n[..., 1] + p[..., 0].detach() * h_n[..., 2]
    if logstd.shape[-1] == 4:
        hold = _holding_mask(holding, p.shape[:-1]).to(p.dtype)
        gaussian = gaussian + p[..., 0].detach() * hold * h_n[..., 3]
    return ent_coef * gaussian + fire_ent_coef * h_b


class RunningStat:
    """Exponential moving mean and variance of a scalar stream (the critic return targets of one policy).

    A fresh statistic is the identity transform (mean 0, variance 1, `count == 0`). The first batch sets the mean and
    the population variance; later batches blend their first and second moments in with weight `1 - momentum`.
    `std` never drops below `MIN_STD`.
    """

    MIN_STD = 1e-3

    def __init__(self, momentum: float = 0.95) -> None:
        self.momentum = float(momentum)
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    @property
    def std(self) -> float:
        return max(math.sqrt(self.var), self.MIN_STD)

    def update(self, x: Any) -> None:
        """Fold a batch into the statistic; an empty batch is ignored and a batch with a non-finite value raises ValueError."""
        x = torch.as_tensor(x).detach().reshape(-1)
        if x.numel() == 0:
            return
        x = x.double()
        batch_mean, batch_second = torch.stack([x.mean(), (x * x).mean()]).tolist()
        if not (math.isfinite(batch_mean) and math.isfinite(batch_second)):
            raise ValueError("RunningStat.update received non-finite values")
        if self.count == 0:
            mean, second = batch_mean, batch_second
        else:
            m = self.momentum
            mean = m * self.mean + (1.0 - m) * batch_mean
            second = m * (self.var + self.mean * self.mean) + (1.0 - m) * batch_second
        self.mean = mean
        self.var = max(second - mean * mean, 0.0)
        self.count += 1

    def normalize(self, x: Any) -> Any:
        return (x - self.mean) / self.std

    def denormalize(self, x: Any) -> Any:
        return x * self.std + self.mean


RETURN_STAT_KEYS = ("ret_mean", "ret_std", "ret_count")


def holds_return_statistics(state_dict: Dict[str, Any], prefix: str = "") -> bool:
    """True when `state_dict` carries the critic return statistics, False when it carries none of them.

    The statistics are one set: a state dict with only some of `RETURN_STAT_KEYS` raises ValueError naming the missing keys.
    """
    present = [key for key in RETURN_STAT_KEYS if prefix + key in state_dict]
    if present and len(present) < len(RETURN_STAT_KEYS):
        missing = [prefix + key for key in RETURN_STAT_KEYS if prefix + key not in state_dict]
        raise ValueError(
            f"state dict holds some of the critic return statistics {RETURN_STAT_KEYS} but is missing {missing}; "
            "they are loaded only as a complete set"
        )
    return bool(present)


class ActorCritic(nn.Module):
    """Actor-Critic network for shared role policies, with a Gaussian or a hybrid action head.

    Supports CTDE (Centralized Training with Decentralized Execution):
    - Actor policy network observes decentralized observations (obs_dim = 82).
    - Critic value network observes augmented centralized state (critic_obs_dim = 136 = 82 + 54).

    `policy_head="gaussian4"`: four Gaussian components (throttle, steer, kick, special) around `tanh` means.
    `policy_head="hybrid5"`: the output layer has five logits `[z_thr, z_steer, z_pow, z_kick, z_spec]`; throttle,
    steer and power are Gaussians around `tanh(z)`, kick and special fire as Bernoulli draws with probabilities
    `fire_probability(z)`. The policy action is `[thr, steer, u, K, S]` (see `to_env_actions`). Its fire bounds
    are stored in the state dict as buffers.
    `policy_head="hybrid6"`: hybrid5 with a fourth Gaussian, the aim, which has its own logstd. The output layer has six
    logits `[z_thr, z_steer, z_pow, z_aim, z_kick, z_spec]` and the policy action is `[thr, steer, u, a, K, S]`. The aim
    replaces the steer, in the env action and in the log-probability, on rows where the seat holds the ball
    (observation column `HOLDING_OBS_INDEX`) and the kick fires.
    `policy_head="hybrid7"`: hybrid6 with six kick-target logits `[z_tm1..z_tm5, z_free]` after the fire logits (twelve
    outputs) and the policy action `[thr, steer, u, a, T, K, S]`. On rows where the holder is ready (`kick_targets`) and
    the kick fires, T picks an in-arc teammate slot, whose bearing becomes the env steer, or FREE, which uses the aim.

    With `value_norm` the critic head is trained in standardised return units and the network holds the running
    return statistics as the buffers `ret_mean`, `ret_std` and `ret_count`. Every value it returns is in raw return units
    (`head * ret_std + ret_mean`) unless `normalized_value` asks for the head output itself. A network built without
    `value_norm` has no such buffers and returns the head output; one that loads a state dict holding them takes the
    buffers on.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dims: Tuple[int, int] = (256, 256),
        central_state_dim: int = CENTRAL_STATE_DIM,
        policy_head: str = "gaussian4",
        fire_prob_floor: float = 0.005,
        fire_logit_bound: float = 6.0,
        value_norm: bool = False,
    ) -> None:
        super().__init__()
        if policy_head not in POLICY_HEADS:
            raise ValueError(f"policy_head must be one of {POLICY_HEADS}, got {policy_head!r}")
        self.value_norm = False
        self.policy_head = policy_head
        self.hybrid = policy_head in FIRE_HEADS
        self.aim = policy_head in ("hybrid6", "hybrid7")
        self.kick_target = policy_head == "hybrid7"
        # Gaussian components ahead of the two fire logits: throttle, steer, power (and aim).
        self.n_gauss = 4 if self.aim else 3
        h1, h2 = hidden_dims
        self.obs_dim = obs_dim
        self.central_state_dim = central_state_dim
        critic_in_dim = obs_dim + central_state_dim

        self.critic = nn.Sequential(
            layer_init(nn.Linear(critic_in_dim, h1)),
            nn.Tanh(),
            layer_init(nn.Linear(h1, h2)),
            nn.Tanh(),
            layer_init(nn.Linear(h2, 1), std=1.0),
        )
        out_width = (
            HYBRID7_LOGIT_DIM
            if self.kick_target
            else HYBRID6_ACTION_DIM
            if self.aim
            else HYBRID_ACTION_DIM
            if self.hybrid
            else action_dim
        )
        out_layer = layer_init(nn.Linear(h2, out_width), std=0.01)
        if self.hybrid:
            # Initial priors reproduce the gaussian4 start: throttle logit 0.35, steer 0, mean power 0.4666, and
            # kick / special fire rates of 0.1289 / 0.0275 at a zero observation; the hybrid6/hybrid7 aim starts at 0
            # like the steer, and hybrid7 target logits start at 0 (uniform over valid options).
            kick_bias = _initial_fire_logit(
                _gaussian4_fire_rate(0.35, -0.35), fire_prob_floor, fire_logit_bound, "kick"
            )
            spec_bias = _initial_fire_logit(
                _gaussian4_fire_rate(0.5, -0.8), fire_prob_floor, fire_logit_bound, "special"
            )
            gaussian_bias = [0.35, 0.0, HYBRID_POWER_BIAS] + ([0.0] if self.aim else [])
            target_bias = [0.0] * NUM_KICK_TARGETS if self.kick_target else []
            with torch.no_grad():
                out_layer.bias.copy_(torch.tensor(gaussian_bias + [kick_bias, spec_bias] + target_bias, dtype=torch.float32))
            self.register_buffer("fire_prob_floor", torch.tensor(float(fire_prob_floor), dtype=torch.float32))
            self.register_buffer("fire_logit_bound", torch.tensor(float(fire_logit_bound), dtype=torch.float32))
        elif action_dim == 4:
            # Sensible initial action priors before Tanh:
            # [0] throttle = +0.35 (move forward), [1] steer = 0.0 (unbiased),
            # [2] kick = -0.35 (hold & dribble ball by default until intentionally kicking > 0.35),
            # [3] special/tackle = -0.8 (do not spam air-tackles by default until > 0.5)
            with torch.no_grad():
                out_layer.bias.copy_(torch.tensor([0.35, 0.0, -0.35, -0.8], dtype=torch.float32))

        actor_layers: List[nn.Module] = [
            layer_init(nn.Linear(obs_dim, h1)),
            nn.Tanh(),
            layer_init(nn.Linear(h1, h2)),
            nn.Tanh(),
            out_layer,
        ]
        if not self.hybrid:
            actor_layers.append(nn.Tanh())  # Bound mean to [-1.0, 1.0]
        self.actor_mean = nn.Sequential(*actor_layers)
        self.actor_logstd = nn.Parameter(torch.full((1, self.n_gauss if self.hybrid else action_dim), -0.5))
        self.action_dim = (
            HYBRID7_ACTION_DIM
            if self.kick_target
            else HYBRID6_ACTION_DIM
            if self.aim
            else HYBRID_ACTION_DIM
            if self.hybrid
            else action_dim
        )
        if value_norm:
            self.enable_value_norm()

    def enable_value_norm(self) -> None:
        """Give the network the identity return statistics `ret_mean = 0`, `ret_std = 1`, `ret_count = 0` (a no-op when it has them).

        `ret_count` is the number of batches folded into the statistics, so a first batch whose mean is 0 and whose
        standard deviation is 1 is told apart from a statistic that was never updated.
        """
        if self.value_norm:
            return
        device = self.actor_logstd.device
        self.register_buffer("ret_mean", torch.zeros((), dtype=torch.float32, device=device))
        self.register_buffer("ret_std", torch.ones((), dtype=torch.float32, device=device))
        self.register_buffer("ret_count", torch.zeros((), dtype=torch.int64, device=device))
        self.value_norm = True

    def _load_from_state_dict(self, state_dict: Dict[str, Any], prefix: str, *args: Any, **kwargs: Any) -> None:
        if holds_return_statistics(state_dict, prefix) and not self.value_norm:
            self.enable_value_norm()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def return_stats(self) -> Tuple[float, float]:
        """`(ret_mean, ret_std)` as floats; `(0.0, 1.0)` for a network without return statistics."""
        if not self.value_norm:
            return 0.0, 1.0
        return float(self.ret_mean), float(self.ret_std)

    def value_affine(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """`(ret_mean, ret_std)` buffers mapping head outputs to raw return units, or None without return statistics."""
        return (self.ret_mean, self.ret_std) if self.value_norm else None

    def return_stat(self, momentum: float) -> RunningStat:
        """The `RunningStat` the buffers hold: a zero `ret_count` is a fresh statistic, any other count resumes the blend."""
        mean, std = self.return_stats()
        stat = RunningStat(momentum)
        count = int(self.ret_count)
        if count > 0:
            stat.mean, stat.var, stat.count = mean, std * std, count
        return stat

    def set_return_stat(self, stat: RunningStat) -> None:
        """Store `stat`'s mean, standard deviation and batch count in the buffers."""
        with torch.no_grad():
            self.ret_mean.fill_(stat.mean)
            self.ret_std.fill_(stat.std)
            self.ret_count.fill_(stat.count)

    def _prepare_critic_input(
        self, obs: torch.Tensor, central_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if central_state is not None:
            return torch.cat([obs, central_state], dim=-1)
        if obs.shape[-1] == self.obs_dim + self.central_state_dim:
            return obs
        zeros = torch.zeros(
            (*obs.shape[:-1], self.central_state_dim),
            dtype=obs.dtype,
            device=obs.device,
        )
        return torch.cat([obs, zeros], dim=-1)

    def get_value(
        self, obs: torch.Tensor, central_state: Optional[torch.Tensor] = None, normalized_value: bool = False
    ) -> torch.Tensor:
        critic_in = self._prepare_critic_input(obs, central_state)
        value = self.critic(critic_in).squeeze(-1)
        if self.value_norm and not normalized_value:
            value = value * self.ret_std + self.ret_mean
        return value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        central_state: Optional[torch.Tensor] = None,
        return_mean: bool = False,
        return_heads: bool = False,
        normalized_value: bool = False,
        ready: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        validate_targets: bool = True,
    ) -> Tuple[torch.Tensor, ...]:
        """Return `(action, log_prob, entropy, value)`; with `return_mean` the policy mean action is appended.

        With `return_heads` (hybrid heads only) the `(logstd, p)` pair (or `(logstd, p, target_logits)` for hybrid7) of
        the same forward pass is appended last, where `p[..., 0]` and `p[..., 1]` are the kick and special fire
        probabilities. A hybrid6 network reads each row's holding flag from column `HOLDING_OBS_INDEX` of `obs`;
        hybrid7 reads `(ready, valid)` via `kick_targets(obs)` unless passed explicitly. `value` is in raw return
        units; `normalized_value` returns the critic head output instead (they coincide without return statistics).
        """
        actor_obs = obs[..., :self.obs_dim] if obs.shape[-1] > self.obs_dim else obs
        if self.hybrid:
            return self._hybrid_action_and_value(
                actor_obs,
                action,
                deterministic,
                central_state,
                return_mean,
                return_heads,
                normalized_value,
                ready=ready,
                valid=valid,
                validate_targets=validate_targets,
            )
        if return_heads:
            raise ValueError("return_heads requires the hybrid5, hybrid6 or hybrid7 policy head")
        action_mean = self.actor_mean(actor_obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = action_mean if deterministic else probs.sample()

        log_prob = probs.log_prob(action).sum(-1)
        entropy = probs.entropy().sum(-1)
        critic_in = self._prepare_critic_input(actor_obs, central_state)
        value = self.critic(critic_in).squeeze(-1)
        if self.value_norm and not normalized_value:
            value = value * self.ret_std + self.ret_mean
        if return_mean:
            return action, log_prob, entropy, value, action_mean
        return action, log_prob, entropy, value

    def _hybrid_split(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """`(Gaussian means (n, 3) or (n, 4), fire probabilities (n, 2))` from the five, six or twelve output logits."""
        n = self.n_gauss
        return torch.tanh(z[..., :n]), fire_probability(z[..., n : n + 2], self.fire_prob_floor, self.fire_logit_bound)

    def _hybrid_from_logits(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(Gaussian means (n, 3) or (n, 4), p_kick (n,), p_spec (n,))` from the output logits."""
        mean3, p = self._hybrid_split(z)
        return mean3, p[..., 0], p[..., 1]

    @staticmethod
    def _hybrid_mode_action(mean3: torch.Tensor, p_kick: torch.Tensor, p_spec: torch.Tensor) -> torch.Tensor:
        """Deterministic hybrid action: the Gaussian means, and a fire when its probability exceeds one half."""
        return torch.cat([mean3, (p_kick > 0.5).to(mean3.dtype).unsqueeze(-1), (p_spec > 0.5).to(mean3.dtype).unsqueeze(-1)], dim=-1)

    @staticmethod
    def _hybrid_env_mean(mean3: torch.Tensor, p_kick: torch.Tensor, p_spec: torch.Tensor) -> torch.Tensor:
        """Env-space mean action `[tanh(z_thr), tanh(z_steer), 2 p_kick - 1, 2 p_spec - 1]`."""
        return torch.cat([mean3[..., :2], (2.0 * p_kick - 1.0).unsqueeze(-1), (2.0 * p_spec - 1.0).unsqueeze(-1)], dim=-1)

    def hybrid_heads(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(Gaussian means (n, 3) or (n, 4), p_kick (n,), p_spec (n,))` of a hybrid network for observations `obs`."""
        actor_obs = obs[..., :self.obs_dim] if obs.shape[-1] > self.obs_dim else obs
        return self._hybrid_from_logits(self.actor_mean(actor_obs))

    @torch.no_grad()
    def fire_stats(self, obs: torch.Tensor) -> Tuple[float, float, float, float]:
        """`(kick saturation, special saturation, mean p_kick, mean p_spec)` over the rows of `obs` (hybrid heads only).

        Saturation is the share of rows whose bounded logit `L * tanh(z / L)` exceeds `0.9 * L` in magnitude.
        """
        actor_obs = obs[..., :self.obs_dim] if obs.shape[-1] > self.obs_dim else obs
        z = self.actor_mean(actor_obs)
        bound = self.fire_logit_bound
        fire = z[..., self.n_gauss:self.n_gauss + 2]
        sat = (torch.abs(bound * torch.tanh(fire / bound)) > 0.9 * bound).to(torch.float32).mean(dim=0)
        _, p_kick, p_spec = self._hybrid_from_logits(z)
        return float(sat[0]), float(sat[1]), float(p_kick.mean()), float(p_spec.mean())

    def _hybrid_action_and_value(
        self,
        actor_obs: torch.Tensor,
        action: Optional[torch.Tensor],
        deterministic: bool,
        central_state: Optional[torch.Tensor],
        return_mean: bool,
        return_heads: bool = False,
        normalized_value: bool = False,
        ready: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        validate_targets: bool = True,
    ) -> Tuple[torch.Tensor, ...]:
        z = self.actor_mean(actor_obs)
        mean3, p = self._hybrid_split(z)
        logstd = self.actor_logstd
        if self.kick_target:
            target_logits = z[..., 6:]
            if ready is None or valid is None:
                ready, valid, _ = kick_targets(actor_obs)
            if action is None:
                if deterministic:
                    t = sample_kick_target(target_logits, valid, deterministic=True).to(mean3.dtype).unsqueeze(-1)
                    action = torch.cat(
                        [
                            mean3,
                            t,
                            (p[..., 0] > 0.5).to(mean3.dtype).unsqueeze(-1),
                            (p[..., 1] > 0.5).to(mean3.dtype).unsqueeze(-1),
                        ],
                        dim=-1,
                    )
                else:
                    gauss = Normal(mean3, torch.exp(logstd).expand_as(mean3)).sample()
                    fires = torch.bernoulli(p)
                    t = sample_kick_target(target_logits, valid, deterministic=False).to(mean3.dtype).unsqueeze(-1)
                    action = torch.cat([gauss, t, fires], dim=-1)
            log_prob = _hybrid7_log_prob(
                mean3, logstd, p, target_logits, action, ready, valid, validate_targets=validate_targets
            )
            is_free = torch.round(action[..., 4]).to(torch.int64) == KICK_TARGET_FREE
            entropy = _hybrid_entropy(logstd, p, ready=ready, is_free=is_free)
            heads: Tuple[torch.Tensor, ...] = (logstd, p, target_logits)
        else:
            holding = actor_obs[..., HOLDING_OBS_INDEX] > 0.5 if self.aim else None
            if action is None:
                if deterministic:
                    action = self._hybrid_mode_action(mean3, p[..., 0], p[..., 1])
                else:
                    action = torch.cat(
                        [Normal(mean3, torch.exp(logstd).expand_as(mean3)).sample(), torch.bernoulli(p)], dim=-1
                    )
            log_prob = _hybrid_log_prob(mean3, logstd, p, action, holding)
            entropy = _hybrid_entropy(logstd, p, holding)
            heads = (logstd, p)
        critic_in = self._prepare_critic_input(actor_obs, central_state)
        value = self.critic(critic_in).squeeze(-1)
        if self.value_norm and not normalized_value:
            value = value * self.ret_std + self.ret_mean
        out: Tuple[Any, ...] = (action, log_prob, entropy, value)
        if return_mean:
            out += (self._hybrid_env_mean(mean3, p[..., 0], p[..., 1]),)
        if return_heads:
            out += (heads,)
        return out

    @torch.no_grad()
    def hybrid_logits_and_value(
        self, obs: torch.Tensor, central_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference-only `(output logits (n, 5) or (n, 6), value (n,))` of a hybrid network for 2-D `obs`."""
        layers = self.__dict__.get("_inference_layers")
        if layers is None:
            layers = (
                self.actor_mean[0], self.actor_mean[2], self.actor_mean[4],
                self.critic[0], self.critic[2], self.critic[4],
            )
            self.__dict__["_inference_layers"] = layers
        a0, a1, a2, c0, c1, c2 = layers
        h = torch.tanh(F.linear(obs, a0.weight, a0.bias))
        h = torch.tanh(F.linear(h, a1.weight, a1.bias))
        z = F.linear(h, a2.weight, a2.bias)
        critic_in = self._prepare_critic_input(obs, central_state)
        v = torch.tanh(F.linear(critic_in, c0.weight, c0.bias))
        v = torch.tanh(F.linear(v, c1.weight, c1.bias))
        value = F.linear(v, c2.weight, c2.bias).squeeze(-1)
        if self.value_norm:
            value = value * self.ret_std + self.ret_mean
        return z, value

    @torch.no_grad()
    def sample_action_and_value(
        self,
        obs: torch.Tensor,
        central_state: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        return_mean: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Inference-only `(action, log_prob, value)` for 2-D `obs`; with `return_mean` the mean action is appended.

        Returns exactly what `get_action_and_value` returns for the same inputs and RNG state (same sampling call,
        same arithmetic order for the Gaussian log-density) without building a distribution object, an entropy
        term or an autograd graph.
        """
        layers = self.__dict__.get("_inference_layers")
        if layers is None:
            layers = (
                self.actor_mean[0], self.actor_mean[2], self.actor_mean[4],
                self.critic[0], self.critic[2], self.critic[4],
            )
            self.__dict__["_inference_layers"] = layers
        a0, a1, a2, c0, c1, c2 = layers
        h = torch.tanh(F.linear(obs, a0.weight, a0.bias))
        h = torch.tanh(F.linear(h, a1.weight, a1.bias))
        if self.hybrid:
            z = F.linear(h, a2.weight, a2.bias)
            mean3, p = self._hybrid_split(z)
            logstd = self.actor_logstd
            if self.kick_target:
                target_logits = z[..., 6:]
                ready, valid, _ = kick_targets(obs)
                if deterministic:
                    t = sample_kick_target(target_logits, valid, deterministic=True).to(mean3.dtype).unsqueeze(-1)
                    action = torch.cat(
                        [
                            mean3,
                            t,
                            (p[..., 0] > 0.5).to(mean3.dtype).unsqueeze(-1),
                            (p[..., 1] > 0.5).to(mean3.dtype).unsqueeze(-1),
                        ],
                        dim=-1,
                    )
                else:
                    gauss = torch.normal(mean3, torch.exp(logstd).expand_as(mean3))
                    fires = torch.bernoulli(p)
                    t = sample_kick_target(target_logits, valid, deterministic=False).to(mean3.dtype).unsqueeze(-1)
                    action = torch.cat([gauss, t, fires], dim=-1)
                log_prob = _hybrid7_log_prob(mean3, logstd, p, target_logits, action, ready, valid)
            else:
                if deterministic:
                    action = self._hybrid_mode_action(mean3, p[..., 0], p[..., 1])
                else:
                    action = torch.cat([torch.normal(mean3, torch.exp(logstd).expand_as(mean3)), torch.bernoulli(p)], dim=-1)
                holding = obs[..., HOLDING_OBS_INDEX] > 0.5 if self.aim else None
                log_prob = _hybrid_log_prob(mean3, logstd, p, action, holding)
            critic_in = self._prepare_critic_input(obs, central_state)
            v = torch.tanh(F.linear(critic_in, c0.weight, c0.bias))
            v = torch.tanh(F.linear(v, c1.weight, c1.bias))
            value = F.linear(v, c2.weight, c2.bias).squeeze(-1)
            if self.value_norm:
                value = value * self.ret_std + self.ret_mean
            if return_mean:
                return action, log_prob, value, self._hybrid_env_mean(mean3, p[..., 0], p[..., 1])
            return action, log_prob, value
        action_mean = torch.tanh(F.linear(h, a2.weight, a2.bias))
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        action = action_mean if deterministic else torch.normal(action_mean, action_std)
        log_prob = (
            -((action - action_mean) ** 2) / (2 * action_std**2) - action_std.log() - _HALF_LOG_TWO_PI
        ).sum(-1)
        critic_in = self._prepare_critic_input(obs, central_state)
        v = torch.tanh(F.linear(critic_in, c0.weight, c0.bias))
        v = torch.tanh(F.linear(v, c1.weight, c1.bias))
        value = F.linear(v, c2.weight, c2.bias).squeeze(-1)
        if self.value_norm:
            value = value * self.ret_std + self.ret_mean
        if return_mean:
            return action, log_prob, value, action_mean
        return action, log_prob, value
