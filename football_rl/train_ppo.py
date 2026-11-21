"""Multi-Agent PPO with Per-Team Role Parameter Sharing (Red Team vs. Blue Team) for 6v6 MARL Football."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import fields, replace
from datetime import datetime
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.optim as optim

from football_rl.config import (
    ACTION_DIM,
    ALL_AGENTS,
    CENTRAL_STATE_DIM,
    CRITIC_STATE_DIM,
    DEFAULT_CONFIG,
    EnvConfig,
    OBS_DIM,
    PPOHyperparams,
    RewardConfig,
    _gaussian4_fire_rate,
    _initial_fire_logit,
)
from football_rl.env import FootballParallelEnv, parallel_env
from football_rl.vec_env import make_vec_football_env, SubprocVecFootballEnv, DummyVecFootballEnv

# Co-evolution soft-sync is evaluated over the last this-many completed matches, independent of how
# many parallel environments (and therefore matches per generation) a run uses.
COEVO_WINDOW_MATCHES = 24


def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal initialization for PPO linear layers."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


_HALF_LOG_TWO_PI = math.log(math.sqrt(2 * math.pi))

POLICY_HEADS = ("gaussian4", "hybrid5")
# Fire telemetry is read from every this-many-th decision of a role's rollout batch. The subsample adds sampling noise
# to the saturation shares: at most about 3 percentage points of standard error for a keeper batch (300 sampled rows
# at 8 environments x 300 decisions) and about 1 for a field batch (1500 sampled rows).
FIRE_STATS_STRIDE = 8
# hybrid5 policy action: [throttle, steer, power sample u, kick fire K, special fire S].
HYBRID_ACTION_DIM = 5
# Initial hybrid5 power-head bias: the mean power (1 + tanh(bias)) / 2 is 0.4666.
HYBRID_POWER_BIAS = -0.067


def policy_head_of_weights(weights: Dict[str, torch.Tensor]) -> Optional[str]:
    """Head of an `ActorCritic` state dict from the width of its output layer (4 -> gaussian4, 5 -> hybrid5), else None."""
    out = weights.get("actor_mean.4.weight")
    if out is None or out.dim() != 2:
        return None
    return {ACTION_DIM: "gaussian4", HYBRID_ACTION_DIM: "hybrid5"}.get(int(out.shape[0]))


def fire_probability(z: torch.Tensor, floor: Any, bound: Any) -> torch.Tensor:
    """Bounded Bernoulli probability `floor + (1 - 2*floor) * sigmoid(bound * tanh(z / bound))`, inside [floor, 1 - floor]."""
    return floor + (1.0 - 2.0 * floor) * torch.sigmoid(bound * torch.tanh(z / bound))


def to_env_actions(policy_actions: np.ndarray, kick_threshold: float) -> np.ndarray:
    """Decode hybrid5 policy actions `[thr, steer, u, K, S]` into the env's `[thr, steer, kick, special]`.

    Kick is `-1` unless K = 1, then `thr + (1 - thr) * max(power, 0.001)` with `thr = kick_threshold` and
    `power = clip((u + 1) / 2, 0, 1)`, so a fired kick is always strictly above the env threshold. Special is
    `+1` when S = 1 and `-1` otherwise. Throttle and steer pass through unchanged.
    """
    pol = np.asarray(policy_actions)
    if pol.shape[-1] != HYBRID_ACTION_DIM:
        raise ValueError(f"expected {HYBRID_ACTION_DIM} policy action components, got {pol.shape[-1]}")
    out = np.empty(pol.shape[:-1] + (ACTION_DIM,), dtype=pol.dtype)
    out[..., 0] = pol[..., 0]
    out[..., 1] = pol[..., 1]
    power = np.maximum(np.clip((pol[..., 2] + 1.0) / 2.0, 0.0, 1.0), 0.001)
    out[..., 2] = np.where(pol[..., 3] > 0.5, kick_threshold + (1.0 - kick_threshold) * power, -1.0)
    out[..., 3] = np.where(pol[..., 4] > 0.5, 1.0, -1.0)
    return out


def _hybrid_log_prob(mean: torch.Tensor, logstd: torch.Tensor, p: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Log-probability of a hybrid5 `action`; `p[..., 0]` and `p[..., 1]` are the kick and special fire probabilities."""
    kick = action[..., 3]
    log_n = -0.5 * ((action[..., :3] - mean) * torch.exp(-logstd)) ** 2 - logstd - _HALF_LOG_TWO_PI
    log_fire = torch.log(torch.where(action[..., 3:] > 0.5, p, 1.0 - p)).sum(-1)
    return log_n[..., 0] + log_n[..., 1] + kick * log_n[..., 2] + log_fire


def _hybrid_entropy(logstd: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Entropy of the hybrid5 head distribution: two Gaussians, two Bernoullis and `p_kick` times the power Gaussian."""
    h_n = 0.5 + _HALF_LOG_TWO_PI + logstd
    h_b = -(p * torch.log(p) + (1.0 - p) * torch.log1p(-p)).sum(-1)
    return h_n[..., 0] + h_n[..., 1] + h_b + p[..., 0] * h_n[..., 2]


def hybrid_log_prob_entropy(
    mean: torch.Tensor,
    logstd: torch.Tensor,
    p_kick: torch.Tensor,
    p_spec: torch.Tensor,
    action: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Log-probability of a hybrid5 `action` and the entropy of the head distribution.

    `mean` and `logstd` hold the throttle, steer and power Gaussians (last dimension 3); `p_kick` and `p_spec` are
    the fire probabilities. The power Gaussian counts only when a kick fired: log-probability adds `K * logN(u)`
    and entropy adds `p_kick * H_N(power)`.
    """
    p = torch.stack([p_kick, p_spec], dim=-1)
    return _hybrid_log_prob(mean, logstd, p, action), _hybrid_entropy(logstd, p)


def hybrid_entropy_bonus(
    logstd: torch.Tensor, p: torch.Tensor, ent_coef: float, fire_entropy: str, fire_ent_coef: float
) -> torch.Tensor:
    """Per-sample entropy bonus of the hybrid5 loss; `p[..., 0]` is the kick and `p[..., 1]` the special fire probability.

    `joint`: `ent_coef` times the full head entropy. `separate`: `ent_coef` times the three Gaussian terms, with the
    power term weighted by the detached kick probability, plus `fire_ent_coef` times the two Bernoulli entropies.
    """
    if fire_entropy == "joint":
        return ent_coef * _hybrid_entropy(logstd, p)
    if fire_entropy != "separate":
        raise ValueError(f"fire_entropy must be 'joint' or 'separate', got {fire_entropy!r}")
    h_n = 0.5 + _HALF_LOG_TWO_PI + logstd
    h_b = -(p * torch.log(p) + (1.0 - p) * torch.log1p(-p)).sum(-1)
    gaussian = h_n[..., 0] + h_n[..., 1] + p[..., 0].detach() * h_n[..., 2]
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
        self.hybrid = policy_head == "hybrid5"
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
        out_layer = layer_init(nn.Linear(h2, HYBRID_ACTION_DIM if self.hybrid else action_dim), std=0.01)
        if self.hybrid:
            # Initial priors reproduce the gaussian4 start: throttle logit 0.35, steer 0, mean power 0.4666, and
            # kick / special fire rates of 0.1289 / 0.0275 at a zero observation.
            kick_bias = _initial_fire_logit(
                _gaussian4_fire_rate(0.35, -0.35), fire_prob_floor, fire_logit_bound, "kick"
            )
            spec_bias = _initial_fire_logit(
                _gaussian4_fire_rate(0.5, -0.8), fire_prob_floor, fire_logit_bound, "special"
            )
            with torch.no_grad():
                out_layer.bias.copy_(
                    torch.tensor([0.35, 0.0, HYBRID_POWER_BIAS, kick_bias, spec_bias], dtype=torch.float32)
                )
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
        self.actor_logstd = nn.Parameter(torch.full((1, 3 if self.hybrid else action_dim), -0.5))
        self.action_dim = HYBRID_ACTION_DIM if self.hybrid else action_dim
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
    ) -> Tuple[torch.Tensor, ...]:
        """Return `(action, log_prob, entropy, value)`; with `return_mean` the policy mean action is appended.

        With `return_heads` (hybrid5 only) the `(logstd, p)` pair of the same forward pass is appended last, where
        `p[..., 0]` and `p[..., 1]` are the kick and special fire probabilities. `value` is in raw return units;
        `normalized_value` returns the critic head output instead (they coincide without return statistics).
        """
        actor_obs = obs[..., :self.obs_dim] if obs.shape[-1] > self.obs_dim else obs
        if self.hybrid:
            return self._hybrid_action_and_value(
                actor_obs, action, deterministic, central_state, return_mean, return_heads, normalized_value
            )
        if return_heads:
            raise ValueError("return_heads requires the hybrid5 policy head")
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
        """`(Gaussian means (n, 3), fire probabilities (n, 2))` from the five output logits."""
        return torch.tanh(z[..., :3]), fire_probability(z[..., 3:], self.fire_prob_floor, self.fire_logit_bound)

    def _hybrid_from_logits(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(Gaussian means (n, 3), p_kick (n,), p_spec (n,))` from the five output logits."""
        mean3, p = self._hybrid_split(z)
        return mean3, p[..., 0], p[..., 1]

    @staticmethod
    def _hybrid_mode_action(mean3: torch.Tensor, p_kick: torch.Tensor, p_spec: torch.Tensor) -> torch.Tensor:
        """Deterministic hybrid5 action: the Gaussian means, and a fire when its probability exceeds one half."""
        return torch.cat([mean3, (p_kick > 0.5).to(mean3.dtype).unsqueeze(-1), (p_spec > 0.5).to(mean3.dtype).unsqueeze(-1)], dim=-1)

    @staticmethod
    def _hybrid_env_mean(mean3: torch.Tensor, p_kick: torch.Tensor, p_spec: torch.Tensor) -> torch.Tensor:
        """Env-space mean action `[tanh(z_thr), tanh(z_steer), 2 p_kick - 1, 2 p_spec - 1]`."""
        return torch.cat([mean3[..., :2], (2.0 * p_kick - 1.0).unsqueeze(-1), (2.0 * p_spec - 1.0).unsqueeze(-1)], dim=-1)

    def hybrid_heads(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(Gaussian means (n, 3), p_kick (n,), p_spec (n,))` of a hybrid5 network for observations `obs`."""
        actor_obs = obs[..., :self.obs_dim] if obs.shape[-1] > self.obs_dim else obs
        return self._hybrid_from_logits(self.actor_mean(actor_obs))

    @torch.no_grad()
    def fire_stats(self, obs: torch.Tensor) -> Tuple[float, float, float, float]:
        """`(kick saturation, special saturation, mean p_kick, mean p_spec)` over the rows of `obs` (hybrid5 only).

        Saturation is the share of rows whose bounded logit `L * tanh(z / L)` exceeds `0.9 * L` in magnitude.
        """
        actor_obs = obs[..., :self.obs_dim] if obs.shape[-1] > self.obs_dim else obs
        z = self.actor_mean(actor_obs)
        bound = self.fire_logit_bound
        sat = (torch.abs(bound * torch.tanh(z[..., 3:5] / bound)) > 0.9 * bound).to(torch.float32).mean(dim=0)
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
    ) -> Tuple[torch.Tensor, ...]:
        mean3, p = self._hybrid_split(self.actor_mean(actor_obs))
        logstd = self.actor_logstd
        if action is None:
            if deterministic:
                action = self._hybrid_mode_action(mean3, p[..., 0], p[..., 1])
            else:
                action = torch.cat(
                    [Normal(mean3, torch.exp(logstd).expand_as(mean3)).sample(), torch.bernoulli(p)], dim=-1
                )
        log_prob = _hybrid_log_prob(mean3, logstd, p, action)
        entropy = _hybrid_entropy(logstd, p)
        critic_in = self._prepare_critic_input(actor_obs, central_state)
        value = self.critic(critic_in).squeeze(-1)
        if self.value_norm and not normalized_value:
            value = value * self.ret_std + self.ret_mean
        out: Tuple[Any, ...] = (action, log_prob, entropy, value)
        if return_mean:
            out += (self._hybrid_env_mean(mean3, p[..., 0], p[..., 1]),)
        if return_heads:
            out += ((logstd, p),)
        return out

    @torch.no_grad()
    def hybrid_logits_and_value(
        self, obs: torch.Tensor, central_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference-only `(output logits (n, 5), value (n,))` of a hybrid5 network for 2-D `obs`."""
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
            mean3, p = self._hybrid_split(F.linear(h, a2.weight, a2.bias))
            logstd = self.actor_logstd
            if deterministic:
                action = self._hybrid_mode_action(mean3, p[..., 0], p[..., 1])
            else:
                action = torch.cat([torch.normal(mean3, torch.exp(logstd).expand_as(mean3)), torch.bernoulli(p)], dim=-1)
            log_prob = _hybrid_log_prob(mean3, logstd, p, action)
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


SNAPSHOT_INTERVAL = 50
SNAPSHOT_POLICY_FILES = (
    "field_policy.pt",
    "keeper_policy.pt",
    "blue_field_policy.pt",
    "blue_keeper_policy.pt",
    "red_field_policy.pt",
    "red_keeper_policy.pt",
)


def uniform_env_index(seed: int, generation: int, num_envs: int) -> int:
    """Environment whose match is exported as the unbiased replay of `generation`.

    Drawn from a generator seeded with `seed + generation` before the rollout, so it is independent of
    how the matches turn out and reproducible across resumes.
    """
    return int(np.random.default_rng(int(seed) + int(generation)).integers(int(num_envs)))


# Opponent snapshot pool: the newest `pool_size` generation snapshots serve as frozen opponents in `pool_envs` of the
# parallel environments. Team A is stored under the `blue_*` file names and team B under `red_*`.
POOL_SCHEDULE_SALT = 0xB5
DEFAULT_POOL_SIZE = 10
POOL_TEAM_PREFIX = {"A": "blue", "B": "red"}
_SNAPSHOT_DIR_PATTERN = re.compile(r"^gen(\d{4,})$")


def validate_pool_settings(
    pool_envs: int, pool_size: int, num_envs: int, separate_teams: bool = True, seed: int = 0, shared_keepers: bool = False
) -> None:
    """Raise ValueError unless the opponent-pool and shared-keeper settings are usable; `pool_envs = 0` (pool off) accepts any `pool_size`.

    Shared keepers need separate teams (a shared-team run already plays one keeper network) and no pool.
    """
    if shared_keepers and not separate_teams:
        raise ValueError("shared_keepers needs separate teams; a shared-team run already plays one keeper network")
    if pool_envs < 0 or pool_envs > num_envs:
        raise ValueError(f"pool_envs must be between 0 and num_envs ({num_envs}), got {pool_envs}")
    if pool_envs == 0:
        return
    if pool_size < 1:
        raise ValueError(f"pool_size must be at least 1 when pool_envs > 0, got {pool_size}")
    if not separate_teams:
        raise ValueError("pool_envs > 0 needs separate teams; a shared-team run has no opponent snapshot to freeze")
    if shared_keepers:
        raise ValueError("pool_envs > 0 is not supported with shared_keepers; a frozen pool snapshot holds one keeper per team")
    if seed < 0:
        raise ValueError(f"seed must be non-negative when pool_envs > 0, got {seed}")


REPLAY_KINDS = ("both", "uniform", "best")


def validate_replay_settings(replay_every: Any, replay_kind: Any) -> Tuple[int, str]:
    """Validate `--replay-every` (integer >= 1) and `--replay-kind` (`both`, `uniform`, `best`)."""
    if isinstance(replay_every, bool) or not isinstance(replay_every, int) or replay_every < 1:
        raise ValueError(f"--replay-every / replay_every must be an integer >= 1, got {replay_every!r}")
    if not isinstance(replay_kind, str) or replay_kind not in REPLAY_KINDS:
        raise ValueError(f"--replay-kind / replay_kind must be one of {REPLAY_KINDS}, got {replay_kind!r}")
    return int(replay_every), replay_kind


def list_pool_generations(checkpoint_dir: str, last_completed: int, pool_size: int) -> List[int]:
    """Generations of the newest `pool_size` snapshots `<checkpoint_dir>/snapshots/genNNNN` with generation <= `last_completed`.

    Returned in ascending order. Staging directories (`genNNNN.tmp`), plain files and other names are not snapshots.
    """
    root = os.path.join(checkpoint_dir, "snapshots")
    try:
        names = os.listdir(root)
    except OSError:
        return []
    found = sorted(
        int(match.group(1))
        for match in (_SNAPSHOT_DIR_PATTERN.match(name) for name in names)
        if match is not None and os.path.isdir(os.path.join(root, match.group(0))) and int(match.group(1)) <= last_completed
    )
    return found[-pool_size:] if pool_size > 0 else []


def pool_schedule(
    seed: int, generation: int, num_envs: int, pool_envs: int, pool_generations: Sequence[int]
) -> List[Dict[str, Any]]:
    """Pool environments of `generation` as `{"env", "frozen_team", "member_gen"}` entries (empty without a pool).

    `pool_envs` distinct environments are drawn from `np.random.default_rng([seed, generation, 0xB5])`; the k-th one
    freezes team B when `generation + k` is even and team A otherwise, and draws one member of `pool_generations`
    uniformly. The result depends only on the arguments.
    """
    if pool_envs < 0 or pool_envs > num_envs:
        raise ValueError(f"pool_envs must be between 0 and num_envs ({num_envs}), got {pool_envs}")
    if pool_envs == 0 or len(pool_generations) == 0:
        return []
    rng = np.random.default_rng([int(seed), int(generation), POOL_SCHEDULE_SALT])
    envs = rng.choice(num_envs, size=pool_envs, replace=False)
    members = rng.integers(len(pool_generations), size=pool_envs)
    return [
        {
            "env": int(env),
            "frozen_team": "B" if (int(generation) + k) % 2 == 0 else "A",
            "member_gen": int(pool_generations[int(member)]),
        }
        for k, (env, member) in enumerate(zip(envs, members))
    ]


def pool_generator(seed: int, generation: int) -> torch.Generator:
    """CPU generator, seeded from `(seed, generation, 0xB5)`, for the draws of the frozen pool seats of one generation."""
    state = np.random.SeedSequence([int(seed), int(generation), POOL_SCHEDULE_SALT]).generate_state(1, dtype=np.uint64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(state[0]))
    return generator


class FrozenSeatBatch:
    """Batched inference for the frozen pool seats of one generation.

    `groups` are `(env indices, seat indices, member network)` triples, one per (member, team, role). The actors and
    critics of all groups are stacked into one batched matrix product per layer (the actor input is padded with zero
    weights over the central-state columns); groups with fewer rows than the largest are zero padded. Every random
    draw comes from the generator passed to `apply`, in a fixed order.
    """

    def __init__(self, groups: List[Tuple[np.ndarray, List[int], ActorCritic]]) -> None:
        self.groups = groups
        count = len(groups)
        nets = [net for _, _, net in groups]
        self.hybrid = nets[0].hybrid
        self.rows = [len(envs) * len(seats) for envs, seats, _ in groups]
        self._index = [np.ix_(envs, seats) for envs, seats, _ in groups]
        max_rows = max(self.rows)
        width = OBS_DIM + CENTRAL_STATE_DIM
        out = HYBRID_ACTION_DIM if self.hybrid else ACTION_DIM
        h1, h2 = nets[0].actor_mean[0].out_features, nets[0].actor_mean[2].out_features
        w0, b0 = torch.zeros(2 * count, width, h1), torch.zeros(2 * count, 1, h1)
        w1, b1 = torch.zeros(2 * count, h1, h2), torch.zeros(2 * count, 1, h2)
        w2, b2 = torch.zeros(2 * count, h2, out), torch.zeros(2 * count, 1, out)
        with torch.no_grad():
            for g, net in enumerate(nets):
                for slot, layers in ((g, (net.actor_mean[0], net.actor_mean[2], net.actor_mean[4])),
                                     (count + g, (net.critic[0], net.critic[2], net.critic[4]))):
                    first, second, last = layers
                    w0[slot, : first.in_features] = first.weight.t()
                    b0[slot, 0] = first.bias
                    w1[slot], b1[slot, 0] = second.weight.t(), second.bias
                    w2[slot, :, : last.out_features] = last.weight.t()
                    b2[slot, 0, : last.out_features] = last.bias
                    affine = net.value_affine() if slot >= count else None
                    if affine is not None:  # critic head in raw return units: `head * ret_std + ret_mean`
                        mean, std = affine
                        w2[slot, :, 0] *= std
                        b2[slot, 0, 0] = last.bias[0] * std + mean
            self._weights = (w0, b0, w1, b1, w2, b2)
            self._std = torch.stack([torch.exp(net.actor_logstd) for net in nets])
            if self.hybrid:
                self._floor = torch.stack([net.fire_prob_floor for net in nets]).view(count, 1, 1)
                self._bound = torch.stack([net.fire_logit_bound for net in nets]).view(count, 1, 1)
        self._inputs = np.zeros((2 * count, max_rows, width), dtype=np.float32)

    @torch.no_grad()
    def apply(
        self,
        obs: np.ndarray,
        central: np.ndarray,
        generator: torch.Generator,
        actions: np.ndarray,
        logprobs: np.ndarray,
        values: np.ndarray,
        action_means: Optional[np.ndarray],
    ) -> None:
        """Overwrite the frozen seats' entries of `actions`, `logprobs`, `values` and `action_means` in place.

        `obs` and `central` are the current `(num_envs, 12, dim)` observations. Stored log-probabilities of frozen
        seats are zero: those transitions are never trained on. Replay means are written only when asked for.
        """
        count = len(self.groups)
        buf = self._inputs
        for g, ix in enumerate(self._index):
            rows = self.rows[g]
            buf[g, :rows, :OBS_DIM] = obs[ix].reshape(rows, OBS_DIM)
            buf[g, :rows, OBS_DIM:] = central[ix].reshape(rows, CENTRAL_STATE_DIM)
        buf[count:] = buf[:count]  # the critics read the same rows as the actors
        w0, b0, w1, b1, w2, b2 = self._weights
        h = torch.baddbmm(b0, torch.from_numpy(buf), w0).tanh_()
        h = torch.baddbmm(b1, h, w1).tanh_()
        y = torch.baddbmm(b2, h, w2)
        z, value = y[:count], y[count:, :, 0]
        if self.hybrid:
            mean3 = torch.tanh(z[..., :3])
            p = fire_probability(z[..., 3:], self._floor, self._bound)
            action = torch.cat(
                [torch.normal(mean3, self._std.expand_as(mean3), generator=generator), torch.bernoulli(p, generator=generator)],
                dim=-1,
            )
            mean = torch.cat([mean3[..., :2], 2.0 * p - 1.0], dim=-1) if action_means is not None else None
        else:
            mean = torch.tanh(z)
            action = torch.normal(mean, self._std.expand_as(mean), generator=generator)
        action_np, value_np = action.numpy(), value.numpy()
        mean_np = mean.numpy() if action_means is not None else None
        for g, ix in enumerate(self._index):
            rows, (envs, seats, _) = self.rows[g], self.groups[g]
            shape = (len(envs), len(seats))
            actions[ix] = action_np[g, :rows].reshape(*shape, -1)
            logprobs[ix] = 0.0
            values[ix] = value_np[g, :rows].reshape(shape)
            if action_means is not None:
                action_means[ix] = mean_np[g, :rows].reshape(*shape, -1)


def gae_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_values: np.ndarray,
    gamma: float,
    lam: Any,
) -> np.ndarray:
    """Generalised advantage estimates of a rollout, computed backwards; the result has the dtype of `rewards`.

    `rewards`, `values` and `dones` have shape `(T, ...)`; `dones[t]` is non-zero when the episode ends after step `t`,
    so neither the bootstrap nor later advantages reach across it. `next_values` (shape `(...)`) is the value after the
    last step. `lam` is a float, or an array that broadcasts over the trailing agent axis to give each agent its own lambda.
    """
    steps = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    lastgaelam = np.zeros(rewards.shape[1:], dtype=rewards.dtype)
    for t in reversed(range(steps)):
        next_nonterminal = 1.0 - dones[t]
        next_val = next_values if t == steps - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * next_nonterminal - values[t]
        lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam
        advantages[t] = lastgaelam
    return advantages


def compute_mc_returns(
    rewards: np.ndarray, dones: Optional[np.ndarray], gamma: float
) -> np.ndarray:
    """Compute backwards Monte-Carlo discounted returns inside each rollout episode segment.

    A segment ends at a step where `dones` is non-zero or at the end of the buffer.
    """
    rew = np.asarray(rewards, dtype=np.float64)
    if rew.shape[0] == 0:
        return np.empty_like(rew, dtype=np.float64)
    if dones is None:
        nonterminal = np.ones_like(rew, dtype=np.float64)
    else:
        done_arr = np.asarray(dones, dtype=np.float64)
        while done_arr.ndim < rew.ndim:
            done_arr = done_arr[..., None]
        nonterminal = 1.0 - done_arr
    returns_mc = np.empty_like(rew, dtype=np.float64)
    running = np.zeros(rew.shape[1:], dtype=np.float64)
    for t in range(rew.shape[0] - 1, -1, -1):
        running = rew[t] + gamma * running * nonterminal[t]
        returns_mc[t] = running
    return returns_mc


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Return the Pearson correlation between flattened `x` and `y`, or 0.0 when undefined."""
    ax = np.asarray(x, dtype=np.float64).reshape(-1)
    ay = np.asarray(y, dtype=np.float64).reshape(-1)
    if ax.size < 2 or ay.size != ax.size:
        return 0.0
    dx = ax - ax.mean()
    dy = ay - ay.mean()
    denom = float(np.sqrt(np.dot(dx, dx) * np.dot(dy, dy)))
    if denom <= 1e-12 or not np.isfinite(denom):
        return 0.0
    corr = float(np.dot(dx, dy) / denom)
    if not np.isfinite(corr):
        return 0.0
    return max(-1.0, min(1.0, corr))


TEAM_COLOR_SCHEMA = 2


def migrate_team_color_keys(history: Dict[str, Any]) -> Dict[str, Any]:
    """Return `history` with team-colour keyed generation records in the current schema.

    Team A (indices 0..5) is Red and team B (indices 6..11) is Blue. Records written before
    `team_color_schema` was 2 stored team A's mean reward under `blue_mean_reward` and team B's
    under `red_mean_reward`; those two values are swapped and the schema marker is set. Histories
    already marked as schema 2 are returned unchanged. The input is never mutated.
    """
    if history.get("team_color_schema") == TEAM_COLOR_SCHEMA:
        return history
    migrated = dict(history)
    generations = history.get("generations")
    if generations is not None:
        swapped: List[Dict[str, Any]] = []
        for record in generations:
            rec = dict(record)
            has_blue = "blue_mean_reward" in rec
            has_red = "red_mean_reward" in rec
            blue_val = rec.pop("blue_mean_reward", None)
            red_val = rec.pop("red_mean_reward", None)
            if has_blue:
                rec["red_mean_reward"] = blue_val
            if has_red:
                rec["blue_mean_reward"] = red_val
            swapped.append(rec)
        migrated["generations"] = swapped
    migrated["team_color_schema"] = TEAM_COLOR_SCHEMA
    return migrated


_ACTOR_INPUT_WEIGHT = "actor_mean.0.weight"
_CRITIC_INPUT_WEIGHT = "critic.0.weight"


def _input_column_maps(
    weights: Dict[str, torch.Tensor], obs_dim: int, central_state_dim: int
) -> Dict[str, torch.Tensor]:
    """Destination column of each first-layer input column of a checkpoint in an `obs_dim`-input `ActorCritic`.

    Observation column j stays column j; a critic's central-state block moves to start at column `obs_dim`.
    Only first layers whose input width differs from the network's are listed. The checkpoint's observation
    width is its actor's input width, else its critic's width less any central-state block. Raises
    `ValueError` when the checkpoint observes more columns than the network, or its critic is neither
    observation-only nor observation plus central state.
    """
    actor_w = weights.get(_ACTOR_INPUT_WEIGHT)
    critic_w = weights.get(_CRITIC_INPUT_WEIGHT)
    if actor_w is not None:
        old_obs = int(actor_w.shape[1])
    elif critic_w is not None:
        width = int(critic_w.shape[1])
        old_obs = width if width <= obs_dim else width - central_state_dim
    else:
        return {}
    if old_obs > obs_dim:
        raise ValueError(f"checkpoint observes {old_obs} columns but the network observes {obs_dim}")
    obs_cols = torch.arange(old_obs)
    maps: Dict[str, torch.Tensor] = {}
    if actor_w is not None and old_obs != obs_dim:
        maps[_ACTOR_INPUT_WEIGHT] = obs_cols
    if critic_w is not None:
        width = int(critic_w.shape[1])
        if width == old_obs:
            cols = obs_cols
        elif width == old_obs + central_state_dim:
            cols = torch.cat([obs_cols, torch.arange(obs_dim, obs_dim + central_state_dim)])
        else:
            raise ValueError(
                f"critic input width {width} is neither {old_obs} nor {old_obs} + {central_state_dim}"
            )
        if width != obs_dim + central_state_dim:
            maps[_CRITIC_INPUT_WEIGHT] = cols
    return maps


def _scatter_columns(src: torch.Tensor, cols: torch.Tensor, width: int) -> torch.Tensor:
    """Zero `(rows, width)` tensor holding the columns of `src` at positions `cols`."""
    out = src.new_zeros((src.shape[0], width))
    out[:, cols] = src
    return out


def adapt_policy_weights(
    weights: Dict[str, torch.Tensor],
    obs_dim: int = OBS_DIM,
    central_state_dim: int = CENTRAL_STATE_DIM,
) -> Dict[str, torch.Tensor]:
    """Return an `ActorCritic` state dict whose first-layer inputs are placed for an `obs_dim`-input network.

    Observation columns the checkpoint lacks (those after its last one) get zero weights, as does the central
    state of a critic trained on observations only, so the adapted actor and critic compute what the
    checkpoint computed from the columns it knows. All other tensors are returned as given.
    """
    maps = _input_column_maps(weights, obs_dim, central_state_dim)
    widths = {_ACTOR_INPUT_WEIGHT: obs_dim, _CRITIC_INPUT_WEIGHT: obs_dim + central_state_dim}
    adapted = dict(weights)
    for key, cols in maps.items():
        adapted[key] = _scatter_columns(weights[key], cols, widths[key])
    return adapted


def adapt_optimizer_state(
    opt_state: Dict[str, Any],
    policy: ActorCritic,
    weights: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, Any]:
    """Return optimizer `opt_state` for `policy`, widening first-layer moments as `adapt_policy_weights` does.

    `weights` is the policy checkpoint saved with `opt_state`; the optimizer's parameters are `policy.parameters()`
    in order. Moment tensors shaped like a widened first-layer weight get zeros in the new columns; every other
    entry is returned as given.
    """
    if weights is None:
        return opt_state
    maps = _input_column_maps(weights, policy.obs_dim, policy.central_state_dim)
    if not maps:
        return opt_state
    widths = {_ACTOR_INPUT_WEIGHT: policy.obs_dim, _CRITIC_INPUT_WEIGHT: policy.obs_dim + policy.central_state_dim}
    names = [name for name, _ in policy.named_parameters()]
    ids = [pid for group in opt_state["param_groups"] for pid in group["params"]]
    state = dict(opt_state["state"])
    for pid, name in zip(ids, names):
        if name not in maps or pid not in state:
            continue
        old_shape = weights[name].shape
        state[pid] = {
            k: _scatter_columns(v, maps[name], widths[name]) if torch.is_tensor(v) and v.shape == old_shape else v
            for k, v in state[pid].items()
        }
    return {**opt_state, "state": state}


CONFIG_OVERRIDE_SECTIONS = {"ppo": PPOHyperparams, "rewards": RewardConfig}


def coerce_override_value(raw: str, annotation: Any) -> Any:
    """Parse the text ``raw`` as the dataclass field type ``annotation`` (``float``, ``int``, ``str`` or ``bool``).

    Booleans are spelled ``true``/``false`` (case-insensitive); floats must be finite. Anything else raises ValueError.
    """
    name = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", str(annotation))
    if name == "bool":
        lowered = raw.strip().lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        raise ValueError(f"expected 'true' or 'false', got {raw!r}")
    if name == "int":
        return int(raw)
    if name == "float":
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"expected a finite number, got {raw!r}")
        return value
    if name == "str":
        return raw
    raise ValueError(f"field type {name} cannot be set from the command line")


def parse_config_overrides(
    specs: Sequence[str], base: EnvConfig = DEFAULT_CONFIG
) -> Tuple[EnvConfig, Dict[str, Any]]:
    """Apply ``SECTION.KEY=VALUE`` overrides (sections ``rewards`` and ``ppo``) to ``base``.

    Returns the new config and the applied overrides as a dict sorted by ``SECTION.KEY`` with values coerced
    to the field types. Any malformed spec, unknown section or key, duplicate, unparseable value or
    failed dataclass validation raises ValueError naming the offending item.
    """
    values: Dict[str, Dict[str, Any]] = {section: {} for section in CONFIG_OVERRIDE_SECTIONS}
    sources: Dict[str, List[str]] = {section: [] for section in CONFIG_OVERRIDE_SECTIONS}
    applied: Dict[str, Any] = {}
    for spec in specs:
        target, eq, raw = spec.partition("=")
        section, dot, key = target.partition(".")
        if not eq or not dot or not section or not key:
            raise ValueError(f"--config-override {spec!r}: expected SECTION.KEY=VALUE")
        cls = CONFIG_OVERRIDE_SECTIONS.get(section)
        if cls is None:
            raise ValueError(
                f"--config-override {spec!r}: unknown section {section!r} "
                f"(allowed: {', '.join(sorted(CONFIG_OVERRIDE_SECTIONS))})"
            )
        field_types = {f.name: f.type for f in fields(cls)}
        if key not in field_types:
            raise ValueError(f"--config-override {spec!r}: {section!r} has no key {key!r}")
        if target in applied:
            raise ValueError(f"--config-override {spec!r}: {target} is set more than once")
        try:
            value = coerce_override_value(raw, field_types[key])
        except ValueError as exc:
            raise ValueError(f"--config-override {spec!r}: cannot read {raw!r} as {field_types[key]}: {exc}") from exc
        applied[target] = value
        values[section][key] = value
        sources[section].append(spec)

    cfg = base
    for section, changes in values.items():
        if not changes:
            continue
        try:
            cfg = replace(cfg, **{section: replace(getattr(cfg, section), **changes)})
        except ValueError as exc:
            raise ValueError(f"--config-override {', '.join(sources[section])}: {exc}") from exc
    return cfg, dict(sorted(applied.items()))


def _run_git(args: Sequence[str]) -> Optional[str]:
    """Run ``git <args>`` in the package directory; return stdout, or None when git is unavailable or fails."""
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def current_git_commit() -> str:
    """Short HEAD hash of the package checkout, suffixed ``-dirty`` when the tree has changes; ``unknown`` without git."""
    rev = _run_git(["rev-parse", "--short", "HEAD"])
    if rev is None or not rev.strip():
        return "unknown"
    status = _run_git(["status", "--porcelain"])
    if status is None:
        return "unknown"
    return rev.strip() + ("-dirty" if status.strip() else "")


class ParameterSharingPPOTrainer:
    """MARL PPO trainer with independent Red (`TEAM_A`) vs. Blue (`TEAM_B`) role policies by default.

    - Red Team (`TEAM_A`, indices 0..5): `blue_keeper_policy` ([0]) + `blue_field_policy` ([1..5])
    - Blue Team (`TEAM_B`, indices 6..11): `red_keeper_policy` ([6]) + `red_field_policy` ([7..11])
    If `separate_teams=False`, both teams share `blue_field_policy` and `blue_keeper_policy`.
    With `shared_keepers=True` (separate teams only) both keeper seats play `blue_keeper_policy`, which is updated once per
    generation on the rows of both keeper seats, while the two field policies stay separate.
    """

    def __init__(
        self,
        config: EnvConfig = DEFAULT_CONFIG,
        num_envs: int = 1,
        rollout_steps: int = 2700,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
        seed: int = 42,
        separate_teams: bool = True,
        save_best_only: bool = False,
        write_root_summary: bool = False,
        critic_only_until_gen: int = 0,
        config_overrides: Optional[Dict[str, Any]] = None,
        accept_config_change: bool = False,
        pool_envs: int = 0,
        pool_size: int = DEFAULT_POOL_SIZE,
        replay_every: int = 1,
        replay_kind: str = "both",
        shared_keepers: bool = False,
    ) -> None:
        validate_pool_settings(
            int(pool_envs), int(pool_size), int(num_envs), separate_teams, int(seed), bool(shared_keepers)
        )
        self.replay_every, self.replay_kind = validate_replay_settings(replay_every, replay_kind)
        self._export_replays_active: bool = True
        self._uniform_capture_active: bool = False
        self._forced_replay_generation: Optional[int] = None
        self.cfg = config
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.device = torch.device(device)
        self.inference_device = torch.device("cpu") if self.device.type == "cuda" else self.device
        self.checkpoint_dir = checkpoint_dir
        self.seed = seed
        self.separate_teams = separate_teams
        self.shared_keepers = bool(shared_keepers)
        self.write_root_summary = write_root_summary
        self.critic_only_until_gen = int(critic_only_until_gen)
        self.config_overrides: Dict[str, Any] = dict(sorted((config_overrides or {}).items()))
        self.accept_config_change = bool(accept_config_change)
        self.pool_envs = int(pool_envs)
        self.pool_size = int(pool_size)
        # Frozen pool members of the generation being played, keyed by (member generation, team) -> (keeper, field).
        self._pool_members: Dict[Tuple[int, str], Tuple[ActorCritic, ActorCritic]] = {}
        self._pool_skipped: set = set()
        self.git_commit: str = current_git_commit()
        self.policy_head: str = self.cfg.ppo.policy_head
        self.policy_action_dim: int = HYBRID_ACTION_DIM if self.policy_head == "hybrid5" else ACTION_DIM
        self.save_best_only = save_best_only or self.cfg.replay.save_best_only
        if self.save_best_only != self.cfg.replay.save_best_only:
            self.cfg = replace(
                self.cfg,
                replay=replace(self.cfg.replay, save_best_only=self.save_best_only),
            )

        # Cumulative training & match tracking across resumed sessions
        self.current_generation: int = 0
        self.total_env_steps: int = 0
        self.match_history: List[Dict[str, Any]] = []
        self.generation_history: List[Dict[str, Any]] = []
        self._history_backup_checked: bool = False
        self.recent_match_scores: Deque[Tuple[int, int]] = deque(maxlen=COEVO_WINDOW_MATCHES)
        self.next_match_id: int = self._get_max_saved_match_id() + 1

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Instantiate vectorized environments (SubprocVecFootballEnv for num_envs > 1, DummyVecFootballEnv for 1)
        self.vec_env = make_vec_football_env(config=self.cfg, num_envs=num_envs)
        self.envs: List[FootballParallelEnv] = getattr(self.vec_env, "envs", [])

        # Role & Team index masks across the 12 agents:
        # Storage names: `blue_*` identifiers and checkpoint files hold team A (rendered red);
        # `red_*` hold team B (rendered blue). Kept for checkpoint compatibility.
        self.blue_keeper_indices = [0]                    # a_keeper
        self.blue_field_indices = [1, 2, 3, 4, 5]         # a_def_1, a_def_2, a_mid_1, a_mid_2, a_atk_1
        self.red_keeper_indices = [6]                     # b_keeper
        self.red_field_indices = [7, 8, 9, 10, 11]        # b_def_1, b_def_2, b_mid_1, b_mid_2, b_atk_1

        self.keeper_indices = [0, 6]
        self.field_indices = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]

        # Red Team (TEAM_A) Policies (on self.device, e.g. CUDA)
        self.blue_field_policy = self._new_policy(self.device)
        self.blue_keeper_policy = self._new_policy(self.device)

        self.blue_field_optimizer = optim.Adam(
            self.blue_field_policy.parameters(), lr=self.cfg.ppo.learning_rate, eps=1e-5
        )
        self.blue_keeper_optimizer = optim.Adam(
            self.blue_keeper_policy.parameters(), lr=self.cfg.ppo.learning_rate, eps=1e-5
        )

        # Blue Team (TEAM_B) Policies (independent weights & optimizers when separate_teams=True)
        if self.separate_teams:
            self.red_field_policy = self._new_policy(self.device)
            self.red_keeper_policy = self.blue_keeper_policy if self.shared_keepers else self._new_policy(self.device)
            self.red_field_optimizer = optim.Adam(
                self.red_field_policy.parameters(), lr=self.cfg.ppo.learning_rate, eps=1e-5
            )
            self.red_keeper_optimizer = (
                self.blue_keeper_optimizer
                if self.shared_keepers
                else optim.Adam(self.red_keeper_policy.parameters(), lr=self.cfg.ppo.learning_rate, eps=1e-5)
            )
        else:
            self.red_field_policy = self.blue_field_policy
            self.red_keeper_policy = self.blue_keeper_policy
            self.red_field_optimizer = self.blue_field_optimizer
            self.red_keeper_optimizer = self.blue_keeper_optimizer

        # CPU shadow policies for fast rollout inference (when device == 'cuda')
        if self.device.type == "cuda":
            self.cpu_blue_field_policy = self._new_policy(self.inference_device)
            self.cpu_blue_keeper_policy = self._new_policy(self.inference_device)
            if self.separate_teams:
                self.cpu_red_field_policy = self._new_policy(self.inference_device)
                self.cpu_red_keeper_policy = (
                    self.cpu_blue_keeper_policy if self.shared_keepers else self._new_policy(self.inference_device)
                )
            else:
                self.cpu_red_field_policy = self.cpu_blue_field_policy
                self.cpu_red_keeper_policy = self.cpu_blue_keeper_policy
            self._sync_cpu_policies()
        else:
            self.cpu_blue_field_policy = self.blue_field_policy
            self.cpu_blue_keeper_policy = self.blue_keeper_policy
            self.cpu_red_field_policy = self.red_field_policy
            self.cpu_red_keeper_policy = self.red_keeper_policy

        # Initialize environments and store current observations and centralized states
        self.num_agents: int = len(ALL_AGENTS)
        self.current_obs = np.zeros((num_envs, self.num_agents, OBS_DIM), dtype=np.float32)
        self.current_central_state = np.zeros(
            (num_envs, self.num_agents, CENTRAL_STATE_DIM), dtype=np.float32
        )
        self._reset_all_envs(initial=True)

    def _new_policy(self, device: torch.device) -> ActorCritic:
        """A freshly initialised role network with the configured policy head."""
        ppo = self.cfg.ppo
        return ActorCritic(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dims=ppo.hidden_dims,
            policy_head=ppo.policy_head,
            fire_prob_floor=ppo.fire_prob_floor,
            fire_logit_bound=ppo.fire_logit_bound,
            value_norm=ppo.value_norm,
        ).to(device)

    def _sync_cpu_policies(self) -> None:
        """Synchronize training weights from CUDA to the CPU rollout inference policies."""
        if self.device.type != "cuda":
            return
        self.cpu_blue_field_policy.load_state_dict(self.blue_field_policy.state_dict())
        self.cpu_blue_keeper_policy.load_state_dict(self.blue_keeper_policy.state_dict())
        if self.separate_teams:
            self.cpu_red_field_policy.load_state_dict(self.red_field_policy.state_dict())
            if not self.shared_keepers:
                self.cpu_red_keeper_policy.load_state_dict(self.red_keeper_policy.state_dict())

    def _get_max_saved_match_id(self) -> int:
        """Scan match_history and replay directory to determine the highest existing match ID."""
        max_id = len(self.match_history)
        for m in self.match_history:
            mid = m.get("match_id", 0)
            if isinstance(mid, int) and mid > max_id:
                max_id = mid

        rep_dir = getattr(self.cfg.replay, "save_dir", getattr(self.cfg.replay, "replay_dir", "replays"))
        if os.path.exists(rep_dir):
            for fname in os.listdir(rep_dir):
                if fname.startswith("match_") and fname.endswith(".json"):
                    parts = fname.split("_")
                    if len(parts) >= 3 and parts[1].isdigit() and parts[2].startswith("gen"):
                        fid = int(parts[1])
                        if fid > max_id:
                            max_id = fid
        return max_id

    def _allocate_match_id(self) -> int:
        mid = max(self.next_match_id, self._get_max_saved_match_id() + 1)
        self.next_match_id = mid + 1
        return mid

    def _reset_all_envs(self, initial: bool = False) -> None:
        """Reset all parallel environments with sequential match IDs."""
        target_gen = self.current_generation + 1
        seeds = [
            (self.seed + self.current_generation * 100 + env_i) if initial else None
            for env_i in range(self.num_envs)
        ]
        options_list = [
            {"generation": target_gen, "match_id": self._allocate_match_id()}
            for _ in range(self.num_envs)
        ]
        obs_batch, central_batch = self.vec_env.reset_array(
            seeds=seeds, options_list=options_list
        )
        self.current_obs[:] = obs_batch
        self.current_central_state[:] = central_batch
        # Pool assignment of the match each environment is playing (None: no frozen seat so far in that match).
        self._pool_match_tags: List[Optional[Dict[str, Any]]] = [None] * self.num_envs

    def _rollout_groups(self) -> List[Tuple[List[int], ActorCritic]]:
        """`(agent indices, CPU policy)` groups in the order their random draws are made."""
        if not self.separate_teams:
            return [
                (self.keeper_indices, self.cpu_blue_keeper_policy),
                (self.field_indices, self.cpu_blue_field_policy),
            ]
        return [
            (self.blue_keeper_indices, self.cpu_blue_keeper_policy),
            (self.blue_field_indices, self.cpu_blue_field_policy),
            (self.red_keeper_indices, self.cpu_red_keeper_policy),
            (self.red_field_indices, self.cpu_red_field_policy),
        ]

    @torch.no_grad()
    def _select_actions(
        self,
        obs_batch: np.ndarray,
        central_batch: Optional[np.ndarray] = None,
        deterministic: bool = False,
        with_means: bool = False,
    ) -> Tuple[np.ndarray, ...]:
        """Query Red and Blue role policies for `(num_envs, len(ALL_AGENTS), OBS_DIM)` observations.

        Returns `(actions, logprobs, values)`; with `with_means` the policy mean actions are appended.
        Each role group is one forward pass and one random draw, taken in agent order; under `hybrid5` the groups'
        forward passes are followed by one draw for all groups together.
        """
        n = self.num_envs
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.inference_device)
        cen_t = (
            torch.as_tensor(central_batch, dtype=torch.float32, device=self.inference_device)
            if central_batch is not None
            else None
        )
        outputs: List[Tuple[List[int], Tuple[torch.Tensor, ...]]] = []
        hybrid = self.policy_head == "hybrid5"
        pending: List[Tuple[List[int], ActorCritic, torch.Tensor, torch.Tensor]] = []
        for indices, policy in self._rollout_groups():
            rows = slice(indices[0], indices[-1] + 1) if indices[-1] - indices[0] + 1 == len(indices) else indices
            # Contiguous rows: the CPU matmul's summation order depends on the input row stride.
            g_obs = obs_t[:, rows, :].reshape(-1, OBS_DIM).contiguous()
            g_cen = cen_t[:, rows, :].reshape(-1, CENTRAL_STATE_DIM) if cen_t is not None else None
            if hybrid:
                z, value = policy.hybrid_logits_and_value(g_obs, central_state=g_cen)
                pending.append((indices, policy, z, value))
                continue
            outputs.append(
                (
                    indices,
                    policy.sample_action_and_value(
                        g_obs, central_state=g_cen, deterministic=deterministic, return_mean=with_means
                    ),
                )
            )
        if hybrid:
            outputs = self._sample_hybrid_groups(pending, deterministic, with_means)

        covers_agents_in_order = [i for indices, _ in outputs for i in indices] == list(range(self.num_agents))
        results: List[np.ndarray] = []
        for field in range(4 if with_means else 3):
            # Actions have the policy's width; the means are env-space (ACTION_DIM wide).
            per_agent_dim = self.policy_action_dim if field == 0 else ACTION_DIM if field == 3 else None
            if covers_agents_in_order:
                parts = [
                    out[field].view(n, len(indices), per_agent_dim)
                    if per_agent_dim
                    else out[field].view(n, len(indices))
                    for indices, out in outputs
                ]
                results.append(torch.cat(parts, dim=1).numpy())
            else:
                shape = (n, self.num_agents, per_agent_dim) if per_agent_dim else (n, self.num_agents)
                merged = torch.zeros(shape, dtype=torch.float32, device=self.inference_device)
                for indices, out in outputs:
                    merged[:, indices] = out[field].view(n, len(indices), *shape[2:])
                results.append(merged.numpy())
        return tuple(results)

    @torch.no_grad()
    def _sample_hybrid_groups(
        self,
        pending: List[Tuple[List[int], ActorCritic, torch.Tensor, torch.Tensor]],
        deterministic: bool,
        with_means: bool,
    ) -> List[Tuple[List[int], Tuple[torch.Tensor, ...]]]:
        """Sample hybrid5 actions for all role groups at once from their output logits.

        The fire probabilities, one Gaussian draw, one Bernoulli draw and the log-probabilities are computed on the
        rows of every group stacked in group order, each row with its own policy's `actor_logstd`; the results are
        split back per group as `(action, log_prob, value[, env-space mean])`.
        """
        first = pending[0][1]
        counts = [z.shape[0] for _, _, z, _ in pending]
        z = torch.cat([z for _, _, z, _ in pending])
        logstd = torch.cat([pol.actor_logstd.expand(c, 3) for (_, pol, _, _), c in zip(pending, counts)])
        mean3, p = first._hybrid_split(z)
        if deterministic:
            action = first._hybrid_mode_action(mean3, p[..., 0], p[..., 1])
        else:
            action = torch.cat([torch.normal(mean3, torch.exp(logstd)), torch.bernoulli(p)], dim=-1)
        log_prob = _hybrid_log_prob(mean3, logstd, p, action)
        env_mean = first._hybrid_env_mean(mean3, p[..., 0], p[..., 1]) if with_means else None
        outputs: List[Tuple[List[int], Tuple[torch.Tensor, ...]]] = []
        start = 0
        for (indices, _, _, value), count in zip(pending, counts):
            part = slice(start, start + count)
            fields: Tuple[torch.Tensor, ...] = (action[part], log_prob[part], value)
            if env_mean is not None:
                fields = fields + (env_mean[part],)
            outputs.append((indices, fields))
            start += count
        return outputs

    @torch.no_grad()
    def _compute_values(
        self, obs_batch: np.ndarray, central_batch: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute critic values for observations and centralized states."""
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.inference_device)
        cen_t = (
            torch.as_tensor(central_batch, dtype=torch.float32, device=self.inference_device)
            if central_batch is not None
            else None
        )
        values = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)

        if not self.separate_teams:
            k_obs = obs_t[:, self.keeper_indices, :].reshape(-1, OBS_DIM)
            k_cen = (
                cen_t[:, self.keeper_indices, :].reshape(-1, CENTRAL_STATE_DIM)
                if cen_t is not None
                else None
            )
            values[:, self.keeper_indices] = (
                self.cpu_blue_keeper_policy.get_value(k_obs, central_state=k_cen)
                .view(self.num_envs, len(self.keeper_indices))
                .cpu()
                .numpy()
            )
            f_obs = obs_t[:, self.field_indices, :].reshape(-1, OBS_DIM)
            f_cen = (
                cen_t[:, self.field_indices, :].reshape(-1, CENTRAL_STATE_DIM)
                if cen_t is not None
                else None
            )
            values[:, self.field_indices] = (
                self.cpu_blue_field_policy.get_value(f_obs, central_state=f_cen)
                .view(self.num_envs, len(self.field_indices))
                .cpu()
                .numpy()
            )
        else:
            groups = [
                (self.blue_keeper_indices, self.cpu_blue_keeper_policy),
                (self.blue_field_indices, self.cpu_blue_field_policy),
                (self.red_keeper_indices, self.cpu_red_keeper_policy),
                (self.red_field_indices, self.cpu_red_field_policy),
            ]
            for indices, policy in groups:
                cnt = len(indices)
                g_obs = obs_t[:, indices, :].reshape(-1, OBS_DIM)
                g_cen = (
                    cen_t[:, indices, :].reshape(-1, CENTRAL_STATE_DIM)
                    if cen_t is not None
                    else None
                )
                values[:, indices] = (
                    policy.get_value(g_obs, central_state=g_cen)
                    .view(self.num_envs, cnt)
                    .cpu()
                    .numpy()
                )
        return values

    def _rollout_thread_count(self) -> int:
        """Intra-op threads for rollout inference: leave a core to each environment worker.

        Idle OpenMP threads spin for a while after every operation; with more of them than spare cores they slow
        the environment workers down. Selection output does not depend on the thread count.
        """
        current = torch.get_num_threads()
        if self.num_envs <= 1:
            return current
        return max(1, min(current, (os.cpu_count() or 1) - self.num_envs - 1))

    def pool_schedule_for(self, generation: int) -> List[Dict[str, Any]]:
        """Pool schedule of `generation`: snapshots completed before it, `pool_size` newest, `pool_envs` environments."""
        if self.pool_envs <= 0:
            return []
        members = list_pool_generations(self.checkpoint_dir, generation - 1, self.pool_size)
        return pool_schedule(self.seed, generation, self.num_envs, self.pool_envs, members)

    def _load_pool_member(self, member_gen: int, team: str) -> Optional[Tuple[ActorCritic, ActorCritic]]:
        """`(keeper, field)` networks of `team` from snapshot `member_gen`, frozen and in eval mode; None when skipped.

        A snapshot whose policy head differs from the trainer's is skipped, with one printed line per generation.
        The networks are built without consuming the trainer's random streams.
        """
        key = (member_gen, team)
        if key in self._pool_members:
            return self._pool_members[key]
        if member_gen in self._pool_skipped:
            return None
        directory = os.path.join(self.checkpoint_dir, "snapshots", f"gen{member_gen:04d}")
        weights = [
            torch.load(
                os.path.join(directory, f"{POOL_TEAM_PREFIX[team]}_{role}_policy.pt"),
                map_location=self.inference_device,
                weights_only=True,
            )
            for role in ("keeper", "field")
        ]
        for w in weights:
            head = policy_head_of_weights(w)
            if head != self.policy_head:
                print(
                    f"  [Pool] Skipping snapshot gen{member_gen:04d}: its policies use the {head!r} head but the "
                    f"trainer's head is {self.policy_head!r}; environments that drew it play self-play."
                )
                self._pool_skipped.add(member_gen)
                return None
        nets: List[ActorCritic] = []
        with torch.random.fork_rng(devices=[]):
            for w in weights:
                net = self._new_policy(self.inference_device)
                if holds_return_statistics(w):
                    net.enable_value_norm()
                self._load_adapted_weights(net, w)
                net.eval()
                net.requires_grad_(False)
                nets.append(net)
        self._pool_members[key] = (nets[0], nets[1])
        return self._pool_members[key]

    def _prepare_pool(
        self, generation: int
    ) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray], Optional[FrozenSeatBatch]]:
        """`(played schedule, frozen-seat mask (num_envs, 12) or None, batched frozen-seat inference or None)`.

        The frozen groups are ordered by (member generation, team, role), so the frozen draws are made in a fixed order.
        """
        self._pool_members = {}
        self._pool_skipped = set()
        played: List[Dict[str, Any]] = []
        for entry in self.pool_schedule_for(generation):
            if self._load_pool_member(entry["member_gen"], entry["frozen_team"]) is not None:
                played.append(entry)
        if not played:
            return [], None, None
        frozen = np.zeros((self.num_envs, self.num_agents), dtype=bool)
        grouped: Dict[Tuple[int, str, str], Tuple[ActorCritic, List[int], List[int]]] = {}
        for entry in played:
            team = entry["frozen_team"]
            keeper_net, field_net = self._pool_members[(entry["member_gen"], team)]
            keeper_seats = self.blue_keeper_indices if team == "A" else self.red_keeper_indices
            field_seats = self.blue_field_indices if team == "A" else self.red_field_indices
            frozen[entry["env"], keeper_seats + field_seats] = True
            for role, net, seats in (("keeper", keeper_net, keeper_seats), ("field", field_net, field_seats)):
                grouped.setdefault((entry["member_gen"], team, role), (net, seats, []))[2].append(entry["env"])
        groups = [(np.array(envs), seats, net) for _, (net, seats, envs) in sorted(grouped.items())]
        return played, frozen, FrozenSeatBatch(groups)

    def _replay_flags(self, generation: int) -> Tuple[bool, bool]:
        """Return `(write_uniform, write_regular)` for `generation` under the configured replay cap."""
        is_last_gen = self._forced_replay_generation == generation
        write_gen = (generation % self.replay_every == 0) or is_last_gen
        if not write_gen:
            return False, False
        return (
            self.replay_kind in ("both", "uniform"),
            self.replay_kind in ("both", "best"),
        )

    def _sync_export_replays(self, enabled: bool) -> None:
        """Update worker replay export/deferral gating only when the flag changes."""
        if enabled != self._export_replays_active:
            self.vec_env.set_export_replays(enabled)
            self._export_replays_active = enabled

    def collect_rollouts(self, active_generation: int) -> Dict[str, Any]:
        """Collect `self.rollout_steps` across all parallel environments and compute per-agent GAE."""
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(self._rollout_thread_count())
        try:
            return self._collect_rollouts(active_generation)
        finally:
            torch.set_num_threads(previous_threads)

    def _collect_rollouts(self, active_generation: int) -> Dict[str, Any]:
        T = self.rollout_steps
        N = self.num_envs

        buf_obs = np.zeros((T, N, self.num_agents, OBS_DIM), dtype=np.float32)
        buf_central = np.zeros((T, N, self.num_agents, CENTRAL_STATE_DIM), dtype=np.float32)
        buf_act = np.zeros((T, N, self.num_agents, self.policy_action_dim), dtype=np.float32)
        buf_logp = np.zeros((T, N, self.num_agents), dtype=np.float32)
        buf_rew = np.zeros((T, N, self.num_agents), dtype=np.float32)
        buf_done = np.zeros((T, N, self.num_agents), dtype=np.float32)
        buf_val = np.zeros((T, N, self.num_agents), dtype=np.float32)

        completed_episodes = 0
        max_interest = 0.0
        exported_replays: List[str] = []
        gen_completed_matches: List[Tuple[int, Dict[str, Any]]] = []

        self.vec_env.set_generation(active_generation)
        recording = bool(self.cfg.replay.enable_recording)
        _, write_regular = self._replay_flags(active_generation)
        self._sync_export_replays(recording and write_regular)

        pool_played, frozen_seats, pool_batch = (
            self._prepare_pool(active_generation) if self.pool_envs > 0 else ([], None, None)
        )
        pool_generator_ = pool_generator(self.seed, active_generation) if pool_batch is not None else None
        pool_assign = {
            e["env"]: {"frozen_team": e["frozen_team"], "member_gen": e["member_gen"]} for e in pool_played
        }
        for env_i, tag in pool_assign.items():
            self._pool_match_tags[env_i] = dict(tag)

        for t in range(T):
            buf_obs[t] = self.current_obs
            buf_central[t] = self.current_central_state
            if recording:
                actions, logprobs, values, action_means = self._select_actions(
                    self.current_obs, self.current_central_state, with_means=True
                )
            else:
                actions, logprobs, values = self._select_actions(
                    self.current_obs, self.current_central_state
                )
                action_means = None
            if pool_batch is not None:
                pool_batch.apply(
                    self.current_obs, self.current_central_state, pool_generator_, actions, logprobs, values, action_means
                )
            buf_act[t] = actions
            buf_logp[t] = logprobs
            buf_val[t] = values

            env_actions = (
                actions
                if self.policy_head == "gaussian4"
                else to_env_actions(actions, self.cfg.mechanics.kick_trigger_threshold)
            )
            act_clipped = np.clip(env_actions, -1.0, 1.0)
            obs_batch, central_batch, rew_batch, done_batch, ep_summaries = (
                self.vec_env.step_array(act_clipped, values, action_means=action_means)
            )
            buf_rew[t] = rew_batch
            buf_done[t] = done_batch[:, None].astype(np.float32)

            if bool(np.any(done_batch)):
                done_env_indices = np.nonzero(done_batch)[0]
                for env_i in done_env_indices:
                    completed_episodes += 1
                    ep_summary = ep_summaries[env_i] or {}
                    ep_interest = float(ep_summary.get("interest_score", 0.0))
                    max_interest = max(max_interest, ep_interest)
                    rpath = ep_summary.get("last_saved_replay")
                    if rpath:
                        exported_replays.append(rpath)

                    # Record match summary entry
                    fstats = ep_summary.get("final_stats", {})
                    total_poss = max(1, sum(fstats.get("possession_frames", [1, 1])))
                    poss_pct = [
                        round(100.0 * fstats.get("possession_frames", [1, 1])[0] / total_poss, 1),
                        round(100.0 * fstats.get("possession_frames", [1, 1])[1] / total_poss, 1),
                    ]
                    total_match_frames = max(1, int(ep_summary.get("total_frames", 10800)))
                    poss_a = int(fstats.get("possession_frames", [0, 0])[0])
                    poss_b = int(fstats.get("possession_frames", [0, 0])[1])
                    stat_frames = int(fstats.get("stationary_ball_frames", 0))
                    loose_frames = max(0, total_match_frames - poss_a - poss_b)
                    true_poss_pct = [
                        round(100.0 * poss_a / total_match_frames, 1),
                        round(100.0 * poss_b / total_match_frames, 1),
                        round(100.0 * loose_frames / total_match_frames, 1),
                    ]
                    match_record = {
                        "match_id": int(ep_summary.get("match_id", self.next_match_id)),
                        "generation": int(active_generation),
                        "score": list(fstats.get("score", [0, 0])),
                        "goals": int(fstats.get("goals", 0)),
                        "interest_score": round(ep_interest, 2),
                        "possession_pct": poss_pct,
                        "true_possession_pct": true_poss_pct,
                        "stationary_ball_frames": stat_frames,
                        "shots": list(fstats.get("shots", [0, 0])),
                        "team_passes": list(fstats.get("team_passes", [0, 0])),
                        "completed_passes": int(fstats.get("completed_passes", 0)),
                        "team_tackles": list(fstats.get("team_tackles", [0, 0])),
                        "successful_tackles": int(fstats.get("successful_tackles", 0)),
                        "team_saves": list(fstats.get("team_saves", [0, 0])),
                        "keeper_saves": int(fstats.get("keeper_saves", 0)),
                        "field_5s_violations": int(fstats.get("field_5s_violations", 0)),
                        "keeper_5s_violations": int(fstats.get("keeper_5s_violations", 0)),
                        "gk_holds": int(fstats.get("gk_holds", 0)),
                        "kicks": int(fstats.get("kicks", 0)),
                        "kick_to_teammate": int(fstats.get("kick_to_teammate", 0)),
                        "kick_to_opponent": int(fstats.get("kick_to_opponent", 0)),
                        "kick_other": int(fstats.get("kick_other", 0)),
                        "pool": dict(self._pool_match_tags[env_i]) if self._pool_match_tags[env_i] else None,
                        "replay_path": (rpath or "") if write_regular else None,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    self.match_history.append(match_record)
                    gen_completed_matches.append((env_i, match_record))

                    # The next match of this environment starts under this generation's pool assignment, if any.
                    self._pool_match_tags[env_i] = dict(pool_assign[env_i]) if env_i in pool_assign else None

                    # Auto-reset environment for seamless rollout collection
                    next_gen_hint = active_generation + (1 if t == T - 1 else 0)
                    reset_obs, reset_central = self.vec_env.reset_array(
                        options_list=[{
                            "generation": next_gen_hint,
                            "match_id": self._allocate_match_id(),
                        }],
                        env_indices=[env_i],
                    )
                    obs_batch[env_i] = reset_obs[0]
                    central_batch[env_i] = reset_central[0]

            self.current_obs[:] = obs_batch
            self.current_central_state[:] = central_batch

        # If save_best_only is enabled, export only the single best match replay from the winning worker
        if write_regular and self.save_best_only and gen_completed_matches and self.cfg.replay.enable_recording:
            eligible = [
                (env_idx, m)
                for env_idx, m in gen_completed_matches
                if float(m.get("interest_score", 0.0)) >= self.cfg.replay.interest_threshold
            ]
            if eligible:
                best_env_idx, best_match = max(
                    eligible,
                    key=lambda item: (
                        float(item[1].get("interest_score", 0.0)),
                        int(item[1].get("goals", 0)),
                    ),
                )
                rpath = self.vec_env.export_replay(best_env_idx)
                if rpath:
                    best_match["replay_path"] = rpath
                    exported_replays = [rpath]

        # Compute Generalized Advantage Estimation (GAE) per agent trajectory
        next_values = self._compute_values(self.current_obs, self.current_central_state)

        gamma = self.cfg.ppo.gamma
        gae_lambda = self.cfg.ppo.gae_lambda

        advantages = gae_advantages(buf_rew, buf_val, buf_done, next_values, gamma, gae_lambda)
        returns = advantages + buf_val

        # Field seats take their policy-gradient advantages from their own lambda; the critic targets above and the
        # keeper seats' advantages keep `gae_lambda`.
        field_lambda = self.cfg.ppo.field_actor_gae_lambda
        if field_lambda != gae_lambda:
            field = self.field_indices
            advantages[:, :, field] = gae_advantages(
                buf_rew[:, :, field], buf_val[:, :, field], buf_done[:, :, field], next_values[:, field], gamma, field_lambda
            )

        return {
            "obs": buf_obs,
            "central_state": buf_central,
            "actions": buf_act,
            "logprobs": buf_logp,
            "advantages": advantages,
            "returns": returns,
            "values": buf_val,
            "rewards": buf_rew,
            "dones": buf_done,
            "completed_episodes": completed_episodes,
            "max_interest": max_interest,
            "exported_replays": exported_replays,
            "frozen_seats": frozen_seats,
            "pool_schedule": pool_played,
        }

    @staticmethod
    def _normalize_advantages_per_role(
        obs: torch.Tensor, advantages: torch.Tensor
    ) -> torch.Tensor:
        """Normalize advantages across the pooled team unit without intra-role zero-sum washing."""
        if advantages.numel() <= 1:
            return advantages
        mean = advantages.mean()
        std = advantages.std()
        return (advantages - mean) / (std + 1e-8)

    def _value_loss(self, newvalue: torch.Tensor, mb_values: torch.Tensor, mb_returns: torch.Tensor) -> torch.Tensor:
        """Critic loss of a minibatch: PPO's value clip around `mb_values` when `ppo.value_clip`, else the plain squared error."""
        if not self.cfg.ppo.value_clip:
            return 0.5 * ((newvalue - mb_returns) ** 2).mean()
        clip_coef = self.cfg.ppo.clip_coef
        v_loss_unclipped = (newvalue - mb_returns) ** 2
        v_clipped = mb_values + torch.clamp(
            newvalue - mb_values, -clip_coef, clip_coef
        )
        v_loss_clipped = (v_clipped - mb_returns) ** 2
        return 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

    def _ppo_update_role(
        self,
        policy: ActorCritic,
        optimizer: optim.Optimizer,
        b_obs: torch.Tensor,
        b_actions: torch.Tensor,
        b_logprobs: torch.Tensor,
        b_advantages: torch.Tensor,
        b_returns: torch.Tensor,
        b_values: torch.Tensor,
        b_central: Optional[torch.Tensor] = None,
        critic_only: bool = False,
    ) -> Dict[str, float]:
        """Run multi-epoch minibatch PPO update on pooled role transitions.

        A role without any learning transition (every seat frozen) is left untouched: no optimizer step is taken and
        its statistics are NaN.
        """
        batch_size = b_obs.shape[0]
        if batch_size == 0:
            nan = float("nan")
            empty = {"pg_loss": nan, "v_loss": nan, "entropy": nan}
            if self.policy_head == "hybrid5":
                empty.update({"fire_sat_kick": nan, "fire_sat_spec": nan, "p_kick": nan, "p_spec": nan})
            return empty
        num_minibatches = min(self.cfg.ppo.num_minibatches, batch_size)
        minibatch_size = max(1, batch_size // num_minibatches)
        clip_coef = self.cfg.ppo.clip_coef
        separate_fire_entropy = self.policy_head == "hybrid5" and self.cfg.ppo.fire_entropy == "separate"

        # Fire statistics of the policy that produced this batch, taken before any update.
        fire_metrics: Dict[str, float] = {}
        if self.policy_head == "hybrid5":
            sat_kick, sat_spec, p_kick_mean, p_spec_mean = policy.fire_stats(b_obs[::FIRE_STATS_STRIDE])
            fire_metrics = {
                "fire_sat_kick": sat_kick,
                "fire_sat_spec": sat_spec,
                "p_kick": p_kick_mean,
                "p_spec": p_spec_mean,
            }

        # Normalize once across the full rollout batch for the team unit
        if not critic_only:
            b_advantages = self._normalize_advantages_per_role(b_obs, b_advantages)

        # Critic targets and the rollout's value estimates move to the policy's standardised return units, with the
        # statistics already updated by this batch; the critic head is read in those units too.
        value_kwargs: Dict[str, bool] = {}
        if self.cfg.ppo.value_norm:
            if not policy.value_norm:
                raise ValueError("ppo.value_norm is on but the policy has no return statistics")
            stat = policy.return_stat(self.cfg.ppo.value_norm_momentum)
            stat.update(b_returns)
            policy.set_return_stat(stat)
            b_returns = (b_returns - policy.ret_mean) / policy.ret_std
            b_values = (b_values - policy.ret_mean) / policy.ret_std
            value_kwargs = {"normalized_value": True}

        pg_losses: List[float] = []
        v_losses: List[float] = []
        entropy_losses: List[float] = []

        device = b_obs.device
        for _ in range(self.cfg.ppo.update_epochs):
            if num_minibatches > 1:
                b_inds = torch.randperm(batch_size, device=device)
            else:
                b_inds = None

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                if b_inds is not None:
                    mb_inds = b_inds[start:end]
                    mb_obs = b_obs[mb_inds]
                    mb_actions = b_actions[mb_inds]
                    mb_logprobs = b_logprobs[mb_inds]
                    mb_advantages = b_advantages[mb_inds]
                    mb_returns = b_returns[mb_inds]
                    mb_values = b_values[mb_inds]
                    cen_mb = b_central[mb_inds] if b_central is not None else None
                else:
                    mb_obs = b_obs
                    mb_actions = b_actions
                    mb_logprobs = b_logprobs
                    mb_advantages = b_advantages
                    mb_returns = b_returns
                    mb_values = b_values
                    cen_mb = b_central

                if critic_only:
                    newvalue = policy.get_value(mb_obs, central_state=cen_mb, **value_kwargs)
                    v_loss = self._value_loss(newvalue, mb_values, mb_returns)
                    loss = v_loss * self.cfg.ppo.vf_coef

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), self.cfg.ppo.max_grad_norm)
                    optimizer.step()

                    pg_losses.append(0.0)
                    v_losses.append(float(v_loss.item()))
                    entropy_losses.append(0.0)
                    continue

                if separate_fire_entropy:
                    _, newlogprob, entropy, newvalue, (head_logstd, head_p) = policy.get_action_and_value(
                        mb_obs, mb_actions, central_state=cen_mb, return_heads=True, **value_kwargs
                    )
                else:
                    _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                        mb_obs, mb_actions, central_state=cen_mb, **value_kwargs
                    )
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - clip_coef, 1.0 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = self._value_loss(newvalue, mb_values, mb_returns)

                entropy_loss = entropy.mean()
                if separate_fire_entropy:
                    bonus = hybrid_entropy_bonus(
                        head_logstd, head_p, self.cfg.ppo.ent_coef, "separate", self.cfg.ppo.fire_ent_coef
                    ).mean()
                    loss = pg_loss - bonus + v_loss * self.cfg.ppo.vf_coef
                else:
                    loss = (
                        pg_loss
                        - self.cfg.ppo.ent_coef * entropy_loss
                        + v_loss * self.cfg.ppo.vf_coef
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), self.cfg.ppo.max_grad_norm)
                optimizer.step()

                pg_losses.append(float(pg_loss.item()))
                v_losses.append(float(v_loss.item()))
                entropy_losses.append(float(entropy_loss.item()))

        return {
            "pg_loss": float(np.mean(pg_losses)),
            "v_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(entropy_losses)),
            **fire_metrics,
        }

    def update_policies(
        self, rollout: Dict[str, Any], critic_only: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """Update Red and Blue role networks independently (or pooled if `separate_teams=False`)."""
        obs = rollout["obs"]
        central_state = rollout.get("central_state")
        actions = rollout["actions"]
        logprobs = rollout["logprobs"]
        advantages = rollout["advantages"]
        returns = rollout["returns"]
        values = rollout["values"]
        rewards = rollout.get("rewards")
        mc_returns = (
            compute_mc_returns(rewards, rollout.get("dones"), self.cfg.ppo.gamma)
            if rewards is not None
            else None
        )
        frozen_seats = rollout.get("frozen_seats")

        def _role_rows(arr: np.ndarray, role_indices: List[int]) -> np.ndarray:
            """Rollout array `(T, num_envs, 12, ...)` restricted to the role's seats, without the frozen ones."""
            rows = arr[:, :, role_indices]
            if frozen_seats is not None:
                rows = rows[:, ~np.asarray(frozen_seats, dtype=bool)[:, role_indices]]
            return rows

        def _attach_v_corr(role_metrics: Dict[str, float], role_indices: List[int]) -> None:
            role_val = _role_rows(values, role_indices)
            role_ret = _role_rows(returns, role_indices)
            role_metrics["v_corr_lambda"] = pearson_correlation(role_val, role_ret)
            role_metrics["v_corr_mc"] = (
                pearson_correlation(role_val, _role_rows(mc_returns, role_indices))
                if mc_returns is not None
                else 0.0
            )

        def _extract_role_tensors(
            role_indices: List[int],
        ) -> Tuple[torch.Tensor, ...]:
            r_obs = torch.as_tensor(
                _role_rows(obs, role_indices).reshape(-1, OBS_DIM),
                dtype=torch.float32,
                device=self.device,
            )
            r_cen = (
                torch.as_tensor(
                    _role_rows(central_state, role_indices).reshape(-1, CENTRAL_STATE_DIM),
                    dtype=torch.float32,
                    device=self.device,
                )
                if central_state is not None
                else None
            )
            r_act = torch.as_tensor(
                _role_rows(actions, role_indices).reshape(-1, self.policy_action_dim),
                dtype=torch.float32,
                device=self.device,
            )
            r_logp = torch.as_tensor(
                _role_rows(logprobs, role_indices).reshape(-1),
                dtype=torch.float32,
                device=self.device,
            )
            r_adv = torch.as_tensor(
                _role_rows(advantages, role_indices).reshape(-1),
                dtype=torch.float32,
                device=self.device,
            )
            r_ret = torch.as_tensor(
                _role_rows(returns, role_indices).reshape(-1),
                dtype=torch.float32,
                device=self.device,
            )
            r_val = torch.as_tensor(
                _role_rows(values, role_indices).reshape(-1),
                dtype=torch.float32,
                device=self.device,
            )
            return r_obs, r_act, r_logp, r_adv, r_ret, r_val, r_cen

        if not self.separate_teams:
            f_tensors = _extract_role_tensors(self.field_indices)
            field_metrics = self._ppo_update_role(
                self.blue_field_policy,
                self.blue_field_optimizer,
                f_tensors[0], f_tensors[1], f_tensors[2], f_tensors[3], f_tensors[4], f_tensors[5],
                b_central=f_tensors[6],
                critic_only=critic_only,
            )
            _attach_v_corr(field_metrics, self.field_indices)
            k_tensors = _extract_role_tensors(self.keeper_indices)
            keeper_metrics = self._ppo_update_role(
                self.blue_keeper_policy,
                self.blue_keeper_optimizer,
                k_tensors[0], k_tensors[1], k_tensors[2], k_tensors[3], k_tensors[4], k_tensors[5],
                b_central=k_tensors[6],
                critic_only=critic_only,
            )
            _attach_v_corr(keeper_metrics, self.keeper_indices)
            metrics = {
                "field": field_metrics,
                "keeper": keeper_metrics,
                "blue_field": field_metrics,
                "blue_keeper": keeper_metrics,
                "red_field": field_metrics,
                "red_keeper": keeper_metrics,
            }
            self._sync_cpu_policies()
            return metrics

        bf_tensors = _extract_role_tensors(self.blue_field_indices)
        blue_field_m = self._ppo_update_role(
            self.blue_field_policy,
            self.blue_field_optimizer,
            bf_tensors[0], bf_tensors[1], bf_tensors[2], bf_tensors[3], bf_tensors[4], bf_tensors[5],
            b_central=bf_tensors[6],
            critic_only=critic_only,
        )
        _attach_v_corr(blue_field_m, self.blue_field_indices)
        # Shared keepers take one update per generation on the rows of both keeper seats.
        keeper_seats = self.keeper_indices if self.shared_keepers else self.blue_keeper_indices
        bk_tensors = _extract_role_tensors(keeper_seats)
        blue_keeper_m = self._ppo_update_role(
            self.blue_keeper_policy,
            self.blue_keeper_optimizer,
            bk_tensors[0], bk_tensors[1], bk_tensors[2], bk_tensors[3], bk_tensors[4], bk_tensors[5],
            b_central=bk_tensors[6],
            critic_only=critic_only,
        )
        _attach_v_corr(blue_keeper_m, keeper_seats)
        rf_tensors = _extract_role_tensors(self.red_field_indices)
        red_field_m = self._ppo_update_role(
            self.red_field_policy,
            self.red_field_optimizer,
            rf_tensors[0], rf_tensors[1], rf_tensors[2], rf_tensors[3], rf_tensors[4], rf_tensors[5],
            b_central=rf_tensors[6],
            critic_only=critic_only,
        )
        _attach_v_corr(red_field_m, self.red_field_indices)
        if self.shared_keepers:
            red_keeper_m = blue_keeper_m
        else:
            rk_tensors = _extract_role_tensors(self.red_keeper_indices)
            red_keeper_m = self._ppo_update_role(
                self.red_keeper_policy,
                self.red_keeper_optimizer,
                rk_tensors[0], rk_tensors[1], rk_tensors[2], rk_tensors[3], rk_tensors[4], rk_tensors[5],
                b_central=rk_tensors[6],
                critic_only=critic_only,
            )
            _attach_v_corr(red_keeper_m, self.red_keeper_indices)
        metrics = {
            "field": blue_field_m,
            "keeper": blue_keeper_m,
            "blue_field": blue_field_m,
            "blue_keeper": blue_keeper_m,
            "red_field": red_field_m,
            "red_keeper": red_keeper_m,
        }
        self._sync_cpu_policies()
        return metrics

    def save_checkpoints(self) -> Tuple[str, str]:
        """Save Red and Blue policies, optimizers, generation counter, and match summaries to disk."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        field_path = os.path.join(self.checkpoint_dir, "field_policy.pt")
        keeper_path = os.path.join(self.checkpoint_dir, "keeper_policy.pt")
        blue_field_path = os.path.join(self.checkpoint_dir, "blue_field_policy.pt")
        blue_keeper_path = os.path.join(self.checkpoint_dir, "blue_keeper_policy.pt")
        red_field_path = os.path.join(self.checkpoint_dir, "red_field_policy.pt")
        red_keeper_path = os.path.join(self.checkpoint_dir, "red_keeper_policy.pt")
        state_path = os.path.join(self.checkpoint_dir, "training_state.pt")
        history_path = os.path.join(self.checkpoint_dir, "training_history.json")

        torch.save(self.blue_field_policy.state_dict(), field_path)
        torch.save(self.blue_keeper_policy.state_dict(), keeper_path)
        torch.save(self.blue_field_policy.state_dict(), blue_field_path)
        torch.save(self.blue_keeper_policy.state_dict(), blue_keeper_path)
        torch.save(self.red_field_policy.state_dict(), red_field_path)
        torch.save(self.red_keeper_policy.state_dict(), red_keeper_path)

        torch.save(
            {
                "current_generation": self.current_generation,
                "total_env_steps": self.total_env_steps,
                "seed": int(self.seed),
                "config_overrides": dict(self.config_overrides),
                "git_commit": self.git_commit,
                "next_match_id": self._get_max_saved_match_id() + 1,
                "separate_teams": self.separate_teams,
                "shared_keepers": self.shared_keepers,
                "policy_head": self.policy_head,
                "pool_envs": self.pool_envs,
                "pool_size": self.pool_size,
                "replay_every": self.replay_every,
                "replay_kind": self.replay_kind,
                "field_policy": self.blue_field_policy.state_dict(),
                "keeper_policy": self.blue_keeper_policy.state_dict(),
                "field_optimizer": self.blue_field_optimizer.state_dict(),
                "keeper_optimizer": self.blue_keeper_optimizer.state_dict(),
                "blue_field_policy": self.blue_field_policy.state_dict(),
                "blue_keeper_policy": self.blue_keeper_policy.state_dict(),
                "blue_field_optimizer": self.blue_field_optimizer.state_dict(),
                "blue_keeper_optimizer": self.blue_keeper_optimizer.state_dict(),
                "red_field_policy": self.red_field_policy.state_dict(),
                "red_keeper_policy": self.red_keeper_policy.state_dict(),
                "red_field_optimizer": self.red_field_optimizer.state_dict(),
                "red_keeper_optimizer": self.red_keeper_optimizer.state_dict(),
                "match_history": self.match_history,
                "generation_history": self.generation_history,
                "recent_match_scores": [list(score) for score in self.recent_match_scores],
                "team_color_schema": TEAM_COLOR_SCHEMA,
            },
            state_path,
        )

        history_payload = {
            "current_generation": self.current_generation,
            "total_matches": len(self.match_history),
            "total_env_steps": self.total_env_steps,
            "separate_teams": self.separate_teams,
            "shared_keepers": self.shared_keepers,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "team_color_schema": TEAM_COLOR_SCHEMA,
            "matches": self.match_history,
            "generations": self.generation_history,
        }
        self._backup_pre_migration_history(history_path)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history_payload, f, indent=2)

        if self.cfg.replay.enable_recording:
            self._write_matches_summary_markdown()

        self._write_generation_snapshot()

        return field_path, keeper_path

    def _write_generation_snapshot(self) -> Optional[str]:
        """Copy the policy files and training state to `snapshots/genNNNN/` on every 50th completed generation.

        A snapshot that already exists is left untouched, so resuming or saving again at the same
        generation never rewrites it. The directory appears atomically (copy to a temp name, then rename).
        """
        gen = self.current_generation
        if gen <= 0 or gen % SNAPSHOT_INTERVAL != 0:
            return None
        root = os.path.join(self.checkpoint_dir, "snapshots")
        final = os.path.join(root, f"gen{gen:04d}")
        if os.path.isdir(final):
            return final
        staging = final + ".tmp"
        if os.path.isdir(staging):
            shutil.rmtree(staging)
        os.makedirs(staging)
        for name in SNAPSHOT_POLICY_FILES + ("training_state.pt",):
            shutil.copyfile(os.path.join(self.checkpoint_dir, name), os.path.join(staging, name))
        os.replace(staging, final)
        return final

    def _backup_pre_migration_history(self, history_path: str) -> None:
        """Copy an on-disk history that predates the current team-colour schema, once per trainer."""
        if self._history_backup_checked:
            return
        self._history_backup_checked = True
        backup_path = os.path.join(self.checkpoint_dir, "training_history.pre_color_schema.json")
        if not os.path.exists(history_path) or os.path.exists(backup_path):
            return
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                on_disk_schema = json.load(f).get("team_color_schema")
        except ValueError:
            on_disk_schema = None
        if on_disk_schema != TEAM_COLOR_SCHEMA:
            shutil.copy2(history_path, backup_path)

    def _compute_series_summary(self) -> Dict[str, int]:
        """Compute cumulative Red (team A) vs. Blue (team B) wins, draws, and goals across `self.match_history`."""
        red_wins = 0
        blue_wins = 0
        draws = 0
        red_goals = 0
        blue_goals = 0
        for m in self.match_history:
            sr, sb = int(m["score"][0]), int(m["score"][1])
            red_goals += sr
            blue_goals += sb
            if sr > sb:
                red_wins += 1
            elif sb > sr:
                blue_wins += 1
            else:
                draws += 1
        return {
            "red_wins": red_wins,
            "blue_wins": blue_wins,
            "draws": draws,
            "red_goals": red_goals,
            "blue_goals": blue_goals,
        }

    def _write_root_summary_copy(self, summary_path: str, content: str) -> None:
        """Mirror the summary to `./matches_summary.md` when enabled; never writes through a symlink."""
        if not self.write_root_summary:
            return
        root_summary = "matches_summary.md"
        if os.path.islink(root_summary):
            return
        if os.path.abspath(summary_path) == os.path.abspath(root_summary):
            return
        try:
            with open(root_summary, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError:
            pass

    def _write_matches_summary_markdown(self) -> Optional[str]:
        """Write a human-friendly `replays/matches_summary.md` table for quick scanning."""
        os.makedirs(self.cfg.replay.save_dir, exist_ok=True)
        summary_path = os.path.join(self.cfg.replay.save_dir, "matches_summary.md")
        series = self._compute_series_summary()
        mode_str = "Independent Team Policies (Red vs. Blue Co-Evolution)" if self.separate_teams else "Shared Mirror Policies"

        lines: List[str] = [
            "# Match & Generation Summary",
            "",
            f"- **Training Mode:** `{mode_str}`",
            f"- **Generations Trained:** `{self.current_generation}`",
            f"- **Total Completed Matches:** `{len(self.match_history)}`",
            f"- **Head-to-Head Series (Red - Draws - Blue):** `{series['red_wins']} W - {series['draws']} D - {series['blue_wins']} W` (Goals: **Red `{series['red_goals']}` - `{series['blue_goals']}` Blue**)",
            f"- **Total RL Steps:** `{self.total_env_steps:,}`",
            f"- **Last Updated:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
            "",
            "### Quick Commands",
            "- **Watch any match by number (e.g. Match #15):** `.venv/bin/python replay.py 15`",
            "- **List matches or filter in terminal:** `.venv/bin/python replay.py --list` or `.venv/bin/python replay.py --range 1 100 --top 10`",
            "- **Resume training next 100 generations:** `.venv/bin/python train_ppo.py --total-updates 100`",
            "",
        ]

        if not self.match_history:
            lines.append("_No completed matches recorded yet._\n")
            content = "\n".join(lines)
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(content)
            self._write_root_summary_copy(summary_path, content)
            return summary_path

        # 1. Top 15 Most Exciting Matches Leaderboard
        top_matches = sorted(
            self.match_history, key=lambda m: float(m["interest_score"]), reverse=True
        )[:15]
        lines.extend(
            [
                "## Top 15 Most Exciting Matches (by Interest Score)",
                "",
                "| Rank | Match # | Gen | Score (Red - Blue) | Interest | Poss% (R/B) | Shots (R/B) | Passes (R/B) | Tackles (R/B) | Saves (R/B) | Watch Command |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for rank, m in enumerate(top_matches, start=1):
            mid = m["match_id"]
            gen = m["generation"]
            sc = f"**{m['score'][0]} - {m['score'][1]}**"
            inter = f"`{m['interest_score']:.1f}`"
            poss = f"{m['possession_pct'][0]:.0f}% / {m['possession_pct'][1]:.0f}%"
            shots = f"{m['shots'][0]}/{m['shots'][1]}"
            passes = f"{m['team_passes'][0]}/{m['team_passes'][1]}"
            tackles = f"{m['team_tackles'][0]}/{m['team_tackles'][1]}"
            saves = f"{m['team_saves'][0]}/{m['team_saves'][1]}"
            cmd = f"`.venv/bin/python replay.py {mid}`" if m.get("replay_path") else "_(no replay)_"
            lines.append(
                f"| #{rank} | **Match {mid}** | Gen {gen} | {sc} | {inter} | {poss} | {shots} | {passes} | {tackles} | {saves} | {cmd} |"
            )

        # 2. Complete Chronological Match Table (1 .. N)
        lines.extend(
            [
                "",
                "---",
                "",
                "## All Matches (Chronological: Match 1 to " + str(len(self.match_history)) + ")",
                "",
                "| Match # | Gen | Score (Red - Blue) | Interest | Poss% (R/B) | Shots (R/B) | Passes (R/B) | Tackles (R/B) | Saves (R/B) | Fouls (Field 5s/GK 5s) | Replay File |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for m in self.match_history:
            mid = m["match_id"]
            gen = m["generation"]
            sc = f"**{m['score'][0]} - {m['score'][1]}**"
            inter = f"`{m['interest_score']:.1f}`"
            poss = f"{m['possession_pct'][0]:.0f}%/{m['possession_pct'][1]:.0f}%"
            shots = f"{m['shots'][0]}/{m['shots'][1]}"
            passes = f"{m['team_passes'][0]}/{m['team_passes'][1]}"
            tackles = f"{m['team_tackles'][0]}/{m['team_tackles'][1]}"
            saves = f"{m['team_saves'][0]}/{m['team_saves'][1]}"
            f_viol = m.get("field_5s_violations", 0)
            k_viol = m.get("keeper_5s_violations", 0)
            fouls = f"{f_viol}/{k_viol}"
            rfile = os.path.basename(m["replay_path"]) if m["replay_path"] else "_(not saved)_"
            lines.append(
                f"| **Match {mid}** | Gen {gen} | {sc} | {inter} | {poss} | {shots} | {passes} | {tackles} | {saves} | {fouls} | `{rfile}` |"
            )

        lines.append("")
        content = "\n".join(lines)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(content)

        self._write_root_summary_copy(summary_path, content)

        return summary_path

    def _load_adapted_weights(self, policy: ActorCritic, weights: Dict[str, torch.Tensor]) -> None:
        """Load a state dict into `policy`, zero-padding narrower first-layer inputs (`adapt_policy_weights`).

        Raises ValueError when the checkpoint's policy head differs from the network's; heads are not convertible, and
        when it holds return statistics but the network has none. A checkpoint without statistics leaves a value-normalising
        network on its identity statistics.
        """
        saved_head = policy_head_of_weights(weights)
        if saved_head is not None and saved_head != policy.policy_head:
            raise ValueError(
                f"Checkpoint holds {saved_head!r} policies but the configured policy head is {policy.policy_head!r}; "
                "heads cannot be converted."
            )
        if holds_return_statistics(weights) and not policy.value_norm:
            raise ValueError(
                "Checkpoint holds critic return statistics but ppo.value_norm is off; its critic is in "
                "normalised units and cannot be read as raw returns."
            )
        policy.load_state_dict(
            adapt_policy_weights(weights, policy.obs_dim, policy.central_state_dim), strict=False
        )

    def load_checkpoints(self) -> bool:
        """Load Red and Blue policies, optimizers, and cumulative generation/match state."""
        state_path = os.path.join(self.checkpoint_dir, "training_state.pt")
        field_path = os.path.join(self.checkpoint_dir, "field_policy.pt")
        keeper_path = os.path.join(self.checkpoint_dir, "keeper_policy.pt")

        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device, weights_only=False)
            stored_head = state.get("policy_head", "gaussian4")
            if stored_head != self.policy_head:
                raise ValueError(
                    f"Checkpoint was trained with policy head {stored_head!r} but the configured policy head is "
                    f"{self.policy_head!r}; heads cannot be converted."
                )
            stored_overrides = dict(sorted((state.get("config_overrides") or {}).items()))
            if stored_overrides != self.config_overrides:
                if not self.accept_config_change:
                    raise ValueError(
                        f"Config overrides differ from the checkpoint: stored {stored_overrides!r}, "
                        f"requested {self.config_overrides!r}. Pass --accept-config-change to resume anyway."
                    )
                print(
                    f"[Resume] Config overrides changed: stored {stored_overrides!r}, "
                    f"requested {self.config_overrides!r}; continuing with the requested overrides."
                )
            stored_pool = (int(state.get("pool_envs", 0)), int(state.get("pool_size", DEFAULT_POOL_SIZE)))
            requested_pool = (self.pool_envs, self.pool_size)
            if (stored_pool[0] or requested_pool[0]) and stored_pool != requested_pool:
                described = (
                    f"stored pool_envs={stored_pool[0]}, pool_size={stored_pool[1]}, "
                    f"requested pool_envs={requested_pool[0]}, pool_size={requested_pool[1]}"
                )
                if not self.accept_config_change:
                    raise ValueError(
                        f"Pool settings differ from the checkpoint: {described}. "
                        "Pass --accept-config-change to resume anyway."
                    )
                print(f"[Resume] Pool settings changed: {described}; continuing with the requested settings.")
            stored_shared_keepers = bool(state.get("shared_keepers", False))
            if stored_shared_keepers != self.shared_keepers:
                described = f"stored shared_keepers={stored_shared_keepers}, requested shared_keepers={self.shared_keepers}"
                if not self.accept_config_change:
                    raise ValueError(
                        f"Keeper sharing differs from the checkpoint: {described}. "
                        "Pass --accept-config-change to resume anyway."
                    )
                print(f"[Resume] Keeper sharing changed: {described}; continuing with the requested setting.")
            stored_replay_every = int(state.get("replay_every", 1))
            stored_replay_kind = str(state.get("replay_kind", "both"))
            if (stored_replay_every, stored_replay_kind) != (self.replay_every, self.replay_kind):
                print(
                    f"[Resume] Replay cap changed: "
                    f"stored replay_every={stored_replay_every}, replay_kind={stored_replay_kind!r}, "
                    f"requested replay_every={self.replay_every}, replay_kind={self.replay_kind!r}; "
                    "continuing with the requested settings."
                )
            saved_seed = state.get("seed")
            if saved_seed is not None and int(saved_seed) != int(self.seed):
                print(f"[Resume] Checkpoint seed was {int(saved_seed)}; continuing with seed={int(self.seed)}.")
            self.current_generation = int(state.get("current_generation", 0))
            self.total_env_steps = int(state.get("total_env_steps", 0))
            self.match_history = list(state.get("match_history", []))
            self.generation_history = migrate_team_color_keys(
                {
                    "generations": list(state.get("generation_history", [])),
                    "team_color_schema": state.get("team_color_schema"),
                }
            )["generations"]
            self.recent_match_scores = deque(
                ((int(a), int(b)) for a, b in state.get("recent_match_scores", [])),
                maxlen=COEVO_WINDOW_MATCHES,
            )
            self.next_match_id = self._get_max_saved_match_id() + 1

            blue_f_weights = state.get("blue_field_policy", state.get("field_policy"))
            blue_k_weights = state.get("blue_keeper_policy", state.get("keeper_policy"))
            red_f_weights = state.get("red_field_policy", blue_f_weights)
            red_k_weights = state.get("red_keeper_policy", blue_k_weights)

            if blue_f_weights is not None:
                self._load_adapted_weights(self.blue_field_policy, blue_f_weights)
            if blue_k_weights is not None:
                self._load_adapted_weights(self.blue_keeper_policy, blue_k_weights)
            if self.separate_teams:
                if red_f_weights is not None:
                    self._load_adapted_weights(self.red_field_policy, red_f_weights)
                if red_k_weights is not None and not self.shared_keepers:
                    self._load_adapted_weights(self.red_keeper_policy, red_k_weights)

            if "blue_field_optimizer" in state:
                try:
                    self.blue_field_optimizer.load_state_dict(
                        adapt_optimizer_state(state["blue_field_optimizer"], self.blue_field_policy, blue_f_weights)
                    )
                except Exception:
                    pass
            elif "field_optimizer" in state:
                try:
                    self.blue_field_optimizer.load_state_dict(
                        adapt_optimizer_state(state["field_optimizer"], self.blue_field_policy, blue_f_weights)
                    )
                except Exception:
                    pass

            if "blue_keeper_optimizer" in state:
                try:
                    self.blue_keeper_optimizer.load_state_dict(
                        adapt_optimizer_state(state["blue_keeper_optimizer"], self.blue_keeper_policy, blue_k_weights)
                    )
                except Exception:
                    pass
            elif "keeper_optimizer" in state:
                try:
                    self.blue_keeper_optimizer.load_state_dict(
                        adapt_optimizer_state(state["keeper_optimizer"], self.blue_keeper_policy, blue_k_weights)
                    )
                except Exception:
                    pass

            if self.separate_teams:
                if "red_field_optimizer" in state:
                    try:
                        self.red_field_optimizer.load_state_dict(
                            adapt_optimizer_state(state["red_field_optimizer"], self.red_field_policy, red_f_weights)
                        )
                    except Exception:
                        pass
                if "red_keeper_optimizer" in state and not self.shared_keepers:
                    try:
                        self.red_keeper_optimizer.load_state_dict(
                            adapt_optimizer_state(state["red_keeper_optimizer"], self.red_keeper_policy, red_k_weights)
                        )
                    except Exception:
                        pass

            # Re-number active environments so the next match starts cleanly at len(match_history) + 1
            self._reset_all_envs(initial=True)
            self._sync_cpu_policies()
            return True

        loaded_any = False
        if os.path.exists(field_path):
            w = torch.load(field_path, map_location=self.device, weights_only=True)
            self._load_adapted_weights(self.blue_field_policy, w)
            if self.separate_teams:
                self._load_adapted_weights(self.red_field_policy, w)
            loaded_any = True
        if os.path.exists(keeper_path):
            w = torch.load(keeper_path, map_location=self.device, weights_only=True)
            self._load_adapted_weights(self.blue_keeper_policy, w)
            if self.separate_teams:
                self._load_adapted_weights(self.red_keeper_policy, w)
            loaded_any = True
        self._sync_cpu_policies()
        return loaded_any

    def close(self) -> None:
        """Close vector environments and worker processes."""
        if hasattr(self, "vec_env"):
            self.vec_env.close()

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def _soft_blend(dst: nn.Module, src: nn.Module, tau: float) -> None:
        """Move `dst` parameters a fraction `tau` of the way towards `src` (in place).

        Value-normalising critics keep their head in units set by their own `ret_mean` / `ret_std`. Before blending,
        `src`'s last critic layer is re-expressed in `dst`'s units, `W' = W * std_src / std_dst` and
        `b' = (b * std_src + mean_src - mean_dst) / std_dst`, so that the blended head reads
        `(1 - tau) * raw_dst + tau * raw_src` in raw return units; `dst` keeps its own statistics.
        """
        with torch.no_grad():
            src_params = list(src.parameters())
            dst_affine = dst.value_affine() if isinstance(dst, ActorCritic) else None
            src_affine = src.value_affine() if isinstance(src, ActorCritic) else None
            if (dst_affine is None) != (src_affine is None):
                raise ValueError("cannot blend a value-normalising critic with one that has no return statistics")
            if dst_affine is not None:
                (mean_d, std_d), (mean_s, std_s) = dst_affine, src_affine
                last = src.critic[-1]
                weight = (last.weight.double() * std_s.double() / std_d.double()).to(last.weight.dtype)
                bias = ((last.bias.double() * std_s.double() + mean_s.double() - mean_d.double()) / std_d.double()).to(last.bias.dtype)
                src_params = [weight if p is last.weight else bias if p is last.bias else p for p in src_params]
            for p_dst, p_src in zip(dst.parameters(), src_params):
                p_dst.data.copy_((1.0 - tau) * p_dst.data + tau * p_src.data)

    def _coevolution_step(self, completed_matches: List[Dict[str, Any]]) -> bool:
        """Record newly completed matches and soft-sync the trailing team when one side dominates.

        Scores enter the rolling window in completion order; matches played with frozen pool seats (record `pool` set)
        are skipped. The condition is evaluated only once
        the window holds `COEVO_WINDOW_MATCHES` matches: at least 8 total goals with one side
        scoring >= 3x the other (opponent floored at 1 goal). The trailing team's policies are then
        blended 35% towards the dominant team's and the window is cleared; with shared keepers only the field
        policy is blended and the keeper network is left as it is. Returns True on a sync.
        """
        if not self.separate_teams:
            return False
        for m in completed_matches:
            if m.get("pool") is not None:
                continue  # a match with frozen opponent seats says nothing about the balance of the two live teams
            self.recent_match_scores.append((int(m["score"][0]), int(m["score"][1])))
        if len(self.recent_match_scores) < COEVO_WINDOW_MATCHES:
            return False

        sum_a = sum(x[0] for x in self.recent_match_scores)
        sum_b = sum(x[1] for x in self.recent_match_scores)
        if sum_a + sum_b < 8:
            return False
        if sum_a >= 3 * max(1, sum_b):
            # Team A is dominating >= 75% -> soft-sync Team B towards Team A
            self._soft_blend(self.red_field_policy, self.blue_field_policy, 0.35)
            if not self.shared_keepers:
                self._soft_blend(self.red_keeper_policy, self.blue_keeper_policy, 0.35)
            print(
                f"  [Co-Evolution Sync] Red dominated ({sum_a}-{sum_b} over {COEVO_WINDOW_MATCHES} matches). "
                f"Soft-synced Blue policy (tau=0.35) to restore parity."
            )
        elif sum_b >= 3 * max(1, sum_a):
            # Team B is dominating >= 75% -> soft-sync Team A towards Team B
            self._soft_blend(self.blue_field_policy, self.red_field_policy, 0.35)
            if not self.shared_keepers:
                self._soft_blend(self.blue_keeper_policy, self.red_keeper_policy, 0.35)
            print(
                f"  [Co-Evolution Sync] Blue dominated ({sum_b}-{sum_a} over {COEVO_WINDOW_MATCHES} matches). "
                f"Soft-synced Red policy (tau=0.35) to restore parity."
            )
        else:
            return False
        self.recent_match_scores.clear()
        self._sync_cpu_policies()
        return True

    def train(self, total_updates: int = 100) -> Dict[str, Any]:
        """Run `total_updates` generations of MARL PPO training."""
        start_time = time.time()
        last_metrics: Dict[str, Any] = {}
        start_gen = self.current_generation + 1
        end_gen = self.current_generation + total_updates

        try:
            for gen in range(start_gen, end_gen + 1):
                matches_before = len(self.match_history)
                recording = bool(self.cfg.replay.enable_recording)
                self._forced_replay_generation = gen if gen == end_gen else None
                write_uniform, _ = self._replay_flags(gen)
                uniform_env = uniform_env_index(self.seed, gen, self.num_envs)
                if recording and write_uniform:
                    self.vec_env.set_uniform_capture(uniform_env)
                    self._uniform_capture_active = True
                elif self._uniform_capture_active:
                    self.vec_env.set_uniform_capture(None)
                    self._uniform_capture_active = False
                rollout = self.collect_rollouts(active_generation=gen)
                uniform_replay = None
                if recording and write_uniform:
                    uniform_replay = self.vec_env.export_uniform_replay(uniform_env)
                    self.vec_env.set_uniform_capture(None)
                    self._uniform_capture_active = False
                critic_only = gen <= self.critic_only_until_gen
                metrics = self.update_policies(rollout, critic_only=critic_only)
                elapsed = max(1e-6, time.time() - start_time)

                step_delta = self.rollout_steps * self.num_envs
                self.total_env_steps += step_delta
                session_steps = (gen - start_gen + 1) * step_delta
                sps = int(session_steps / elapsed)

                mean_field_rew = float(rollout["rewards"][:, :, self.field_indices].mean())
                mean_keeper_rew = float(rollout["rewards"][:, :, self.keeper_indices].mean())
                team_a_rew = float(
                    rollout["rewards"][:, :, self.blue_keeper_indices + self.blue_field_indices].mean()
                )
                team_b_rew = float(
                    rollout["rewards"][:, :, self.red_keeper_indices + self.red_field_indices].mean()
                )

                # Exploration LogStd Clamp (prevent action noise explosion / collapse)
                if not critic_only:
                    logstd_min, logstd_max = self.cfg.ppo.logstd_min, self.cfg.ppo.logstd_max
                    if self.separate_teams:
                        for pol in (
                            self.blue_field_policy,
                            self.blue_keeper_policy,
                            self.red_field_policy,
                            self.red_keeper_policy,
                        ):
                            torch.clamp_(pol.actor_logstd.data, logstd_min, logstd_max)
                    else:
                        for pol in (self.blue_field_policy, self.blue_keeper_policy):
                            torch.clamp_(pol.actor_logstd.data, logstd_min, logstd_max)
                self._sync_cpu_policies()

                # Balanced Co-Evolution & Underdog Soft-Sync (over the rolling window of completed matches)
                if not critic_only:
                    self._coevolution_step(self.match_history[matches_before:])

                gen_record = {
                    "generation": gen,
                    "total_env_steps": self.total_env_steps,
                    "sps": sps,
                    "seed": int(self.seed),
                    "config_overrides": dict(self.config_overrides),
                    "git_commit": self.git_commit,
                    "critic_only": bool(critic_only),
                    "red_mean_reward": round(team_a_rew, 5),
                    "blue_mean_reward": round(team_b_rew, 5),
                    "mean_field_reward": round(mean_field_rew, 5),
                    "mean_keeper_reward": round(mean_keeper_rew, 5),
                    "v_corr_lambda": {
                        role: metrics[role]["v_corr_lambda"]
                        for role in ("blue_field", "blue_keeper", "red_field", "red_keeper")
                    },
                    "v_corr_mc": {
                        role: metrics[role]["v_corr_mc"]
                        for role in ("blue_field", "blue_keeper", "red_field", "red_keeper")
                    },
                    "max_interest": round(float(rollout["max_interest"]), 2),
                    "completed_matches_in_gen": int(rollout["completed_episodes"]),
                    "num_envs": self.num_envs,
                    "rollout_steps": self.rollout_steps,
                    "total_matches_so_far": len(self.match_history),
                    "uniform_env": uniform_env,
                    "uniform_replay": uniform_replay,
                    "pool_envs": self.pool_envs,
                    "pool_size": self.pool_size,
                    "pool_schedule": list(rollout.get("pool_schedule", [])),
                    "replay_every": self.replay_every,
                    "replay_kind": self.replay_kind,
                }
                if self.cfg.ppo.value_norm:
                    role_policies = {
                        "blue_field": self.blue_field_policy,
                        "blue_keeper": self.blue_keeper_policy,
                        "red_field": self.red_field_policy,
                        "red_keeper": self.red_keeper_policy,
                    }
                    gen_record["ret_mean"] = {role: pol.return_stats()[0] for role, pol in role_policies.items()}
                    gen_record["ret_std"] = {role: pol.return_stats()[1] for role, pol in role_policies.items()}
                if self.policy_head == "hybrid5":
                    roles = ("blue_field", "blue_keeper", "red_field", "red_keeper")
                    gen_record["fire_sat"] = {
                        role: {"kick": metrics[role]["fire_sat_kick"], "spec": metrics[role]["fire_sat_spec"]}
                        for role in roles
                    }
                    gen_record["p_kick"] = {role: metrics[role]["p_kick"] for role in roles}
                    gen_record["p_spec"] = {role: metrics[role]["p_spec"] for role in roles}
                self.generation_history.append(gen_record)
                self.current_generation = gen

                print(
                    f"[Gen {gen:04d}/{end_gen:04d}] "
                    f"Matches: {len(self.match_history):4d} | "
                    f"Steps: {self.total_env_steps:8d} ({sps:5d} SPS) | "
                    f"Red Rew: {team_a_rew:+.4f} | Blue Rew: {team_b_rew:+.4f} | "
                    f"Interest: {rollout['max_interest']:.1f}"
                )
                for rpath in rollout["exported_replays"]:
                    print(f"  -> Saved replay: {rpath}")
                if uniform_replay:
                    print(f"  -> Saved uniform replay (env {uniform_env}): {uniform_replay}")

                last_metrics = {
                    "update": gen,
                    "generation": gen,
                    "total_env_steps": self.total_env_steps,
                    "sps": sps,
                    "field": metrics["field"],
                    "keeper": metrics["keeper"],
                    "blue_field": metrics["blue_field"],
                    "red_field": metrics["red_field"],
                    "mean_field_reward": mean_field_rew,
                    "mean_keeper_reward": mean_keeper_rew,
                    "red_mean_reward": team_a_rew,
                    "blue_mean_reward": team_b_rew,
                    "total_matches": len(self.match_history),
                }

                # Persist checkpoints and update `replays/matches_summary.md` on every generation
                self.save_checkpoints()

        except KeyboardInterrupt:
            print(
                f"\n[Training Interrupted] Caught KeyboardInterrupt / SIGINT. "
                f"Safely persisting active checkpoint at Generation {self.current_generation}..."
            )
            self.save_checkpoints()
            print(
                f"[Checkpoint Saved] State safely preserved at Generation {self.current_generation} "
                f"(Completed Matches: {len(self.match_history)}). Exiting gracefully."
            )
        finally:
            self._forced_replay_generation = None

        return last_metrics


# ==============================================================================
# Batch Archiving, Loading, Resetting & Status Helpers
# ==============================================================================
def reset_active_batch(checkpoint_dir: str, replay_dir: str) -> None:
    """Wipe active `checkpoints/` and `replays/` to start a brand-new batch from scratch."""
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    if os.path.exists(replay_dir):
        shutil.rmtree(replay_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(replay_dir, exist_ok=True)


def select_archive_replays(
    replay_files: List[str],
    replay_dir: Optional[str] = None,
    num_first: int = 5,
    num_mid: int = 5,
    num_last: int = 5,
    seed: Optional[int] = None,
) -> List[str]:
    """Select up to (num_first + num_mid + num_last) match replay files across training generations.

    Retains the first `num_first` generations, the last `num_last` generations, and `num_mid`
    generations randomly spread out in-between, keeping at most one match replay per generation.
    If the total number of match replay files is <= (num_first + num_mid + num_last), all files are kept.
    """
    total_target = num_first + num_mid + num_last
    if len(replay_files) <= total_target:
        return list(replay_files)

    items = []
    for f in replay_files:
        full_path = os.path.join(replay_dir, f) if replay_dir else f
        base = os.path.basename(f)
        m = re.search(r"match_(\d+)_gen(\d+)(?:_score(\d+(?:_\d+)?))?", base)
        if m:
            mid = int(m.group(1))
            gen = int(m.group(2))
            try:
                score = float(m.group(3).replace("_", ".")) if m.group(3) else 0.0
            except (ValueError, TypeError):
                score = 0.0
        else:
            m_gen = re.search(r"gen(\d+)", base)
            m_mid = re.search(r"match_(\d+)", base)
            m_score = re.search(r"score(\d+(?:_\d+)?)", base)
            gen = int(m_gen.group(1)) if m_gen else None
            mid = int(m_mid.group(1)) if m_mid else 0
            try:
                score = float(m_score.group(1).replace("_", ".")) if m_score else 0.0
            except (ValueError, TypeError):
                score = 0.0

        if gen is None and os.path.isfile(full_path):
            try:
                with open(full_path, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                    meta = data.get("metadata", {})
                    gen = meta.get("generation", data.get("generation"))
                    if mid == 0:
                        mid = meta.get("match_id", data.get("match_id", 0))
                    if score == 0.0:
                        score = float(meta.get("interest_score", data.get("interest_score", 0.0)))
            except Exception:
                pass

        if gen is None:
            gen = mid if mid > 0 else 0

        items.append({"file": f, "gen": gen, "mid": mid, "score": score, "is_uniform": "_uniform_" in base})

    gen_to_items = defaultdict(list)
    for item in items:
        gen_to_items[item["gen"]].append(item)

    for gen in gen_to_items:
        gen_to_items[gen].sort(key=lambda x: (x["is_uniform"], x["score"], x["mid"]), reverse=True)

    sorted_gens = sorted(gen_to_items.keys())
    N = len(sorted_gens)

    if N <= total_target:
        chosen_files = [gen_to_items[g][0]["file"] for g in sorted_gens]
        if len(chosen_files) < total_target:
            remaining_items = []
            for g in sorted_gens:
                remaining_items.extend(gen_to_items[g][1:])
            remaining_items.sort(key=lambda x: (x["score"], x["mid"]), reverse=True)
            for item in remaining_items[: total_target - len(chosen_files)]:
                chosen_files.append(item["file"])
        return chosen_files

    first_gens = sorted_gens[:num_first]
    last_gens = sorted_gens[-num_last:]
    in_between = sorted_gens[num_first:-num_last]
    M = len(in_between)

    rng = random.Random(seed) if seed is not None else random
    mid_gens = []
    if M <= num_mid:
        mid_gens = list(in_between)
    else:
        for i in range(num_mid):
            start = int(i * M / num_mid)
            end = int((i + 1) * M / num_mid)
            segment = in_between[start:end]
            mid_gens.append(rng.choice(segment))
    mid_gens.sort()

    chosen_gens = first_gens + mid_gens + last_gens
    return [gen_to_items[g][0]["file"] for g in chosen_gens]


def archive_batch(
    name: Optional[str] = None,
    checkpoint_dir: str = "checkpoints",
    replay_dir: str = "replays",
    archive_root: str = "archives",
    num_first: int = 5,
    num_mid: int = 5,
    num_last: int = 5,
) -> str:
    """Save active `checkpoints/` and `replays/` into `archives/<name>/`.

    When archiving, keeps the first 5 and last 5 generations as well as 5 randomly spread
    out in-between generations, retaining a total of 15 match replays only.
    If `name` is omitted or empty, automatically generates a timestamp-based name
    formatted as `%Y%m%d_%H%M%S`.
    """
    if not name or not name.strip():
        name = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        name = name.strip()

    state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if not os.path.exists(state_path) and not os.path.exists(
        os.path.join(checkpoint_dir, "field_policy.pt")
    ):
        raise FileNotFoundError(
            f"No active training checkpoints found in `{checkpoint_dir}/` to archive."
        )

    dest_dir = os.path.join(archive_root, name)
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    if os.path.exists(checkpoint_dir):
        shutil.copytree(checkpoint_dir, os.path.join(dest_dir, "checkpoints"))

    if os.path.exists(replay_dir):
        dest_rep = os.path.join(dest_dir, "replays")
        os.makedirs(dest_rep, exist_ok=True)

        all_entries = os.listdir(replay_dir)
        replay_files = []
        for entry in all_entries:
            src_path = os.path.join(replay_dir, entry)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, os.path.join(dest_rep, entry), dirs_exist_ok=True)
            elif entry.endswith(".json") and entry != "training_history.json":
                replay_files.append(entry)
            else:
                shutil.copy2(src_path, os.path.join(dest_rep, entry))

        kept_replays = select_archive_replays(
            replay_files,
            replay_dir=replay_dir,
            num_first=num_first,
            num_mid=num_mid,
            num_last=num_last,
        )
        for rf in kept_replays:
            src_file = os.path.join(replay_dir, rf)
            if os.path.exists(src_file):
                shutil.copy2(src_file, os.path.join(dest_rep, rf))

    return dest_dir


def load_archived_batch(
    name: str,
    checkpoint_dir: str = "checkpoints",
    replay_dir: str = "replays",
    archive_root: str = "archives",
) -> str:
    """Restore an archived batch from `archives/<name>/` into active `checkpoints/` and `replays/`."""
    src_dir = os.path.join(archive_root, name)
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Archive `{src_dir}` does not exist.")

    reset_active_batch(checkpoint_dir, replay_dir)
    src_ckpt = os.path.join(src_dir, "checkpoints")
    src_rep = os.path.join(src_dir, "replays")

    if os.path.exists(src_ckpt):
        shutil.copytree(src_ckpt, checkpoint_dir, dirs_exist_ok=True)
    if os.path.exists(src_rep):
        shutil.copytree(src_rep, replay_dir, dirs_exist_ok=True)

    return src_dir


def list_archived_batches(archive_root: str = "archives") -> List[Dict[str, Any]]:
    """Return metadata about all saved training batches in `archives/`."""
    if not os.path.exists(archive_root):
        return []
    entries: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(archive_root)):
        bdir = os.path.join(archive_root, name)
        if not os.path.isdir(bdir):
            continue
        hist_path = os.path.join(bdir, "checkpoints", "training_history.json")
        if os.path.exists(hist_path):
            with open(hist_path, "r", encoding="utf-8") as f:
                h = migrate_team_color_keys(json.load(f))
            matches = h.get("matches", [])
            best_score = max((m.get("interest_score", 0.0) for m in matches), default=0.0)
            entries.append(
                {
                    "name": name,
                    "generations": h.get("current_generation", 0),
                    "matches": h.get("total_matches", len(matches)),
                    "total_env_steps": h.get("total_env_steps", 0),
                    "best_interest": best_score,
                    "last_updated": h.get("last_updated", ""),
                }
            )
        else:
            entries.append(
                {
                    "name": name,
                    "generations": 0,
                    "matches": 0,
                    "total_env_steps": 0,
                    "best_interest": 0.0,
                    "last_updated": "unknown",
                }
            )
    return entries


def print_active_status(checkpoint_dir: str = "checkpoints", replay_dir: str = "replays") -> None:
    """Print the current active batch's generation count, total matches, and top highlights."""
    hist_path = os.path.join(checkpoint_dir, "training_history.json")
    if not os.path.exists(hist_path):
        print("No active training history found in `checkpoints/training_history.json`.")
        print("Run `.venv/bin/python train_ppo.py --total-updates 100` to start training!")
        return

    with open(hist_path, "r", encoding="utf-8") as f:
        h = migrate_team_color_keys(json.load(f))

    matches = h.get("matches", [])
    red_w = sum(1 for m in matches if m["score"][0] > m["score"][1])
    blue_w = sum(1 for m in matches if m["score"][1] > m["score"][0])
    draws = len(matches) - red_w - blue_w
    red_g = sum(int(m["score"][0]) for m in matches)
    blue_g = sum(int(m["score"][1]) for m in matches)

    print("=" * 80)
    print(f"ACTIVE TRAINING BATCH STATUS")
    print("=" * 80)
    print(f"  Generations Trained : {h.get('current_generation', 0)}")
    print(f"  Completed Matches   : {len(matches)}")
    print(
        f"  Series (Red - Blue) : {red_w} Wins - {draws} Draws - {blue_w} Wins "
        f"(Goals: Red {red_g} - {blue_g} Blue)"
    )
    print(f"  Total RL Steps      : {h.get('total_env_steps', 0):,}")
    print(f"  Last Updated        : {h.get('last_updated', 'N/A')}")
    print(f"  Summary File        : {os.path.join(replay_dir, 'matches_summary.md')}")
    print("-" * 80)

    if matches:
        top = sorted(matches, key=lambda m: float(m.get("interest_score", 0.0)), reverse=True)[:10]
        print("  TOP 10 MOST EXCITING MATCHES:")
        print(
            f"  {'Match #':<9} {'Gen':<7} {'Score (R-B)':<13} {'Interest':<10} "
            f"{'Shots':<9} {'Passes':<9} {'Tackles':<9} {'Saves':<7}"
        )
        for m in top:
            sc = f"{m['score'][0]} - {m['score'][1]}"
            sh = f"{m['shots'][0]}/{m['shots'][1]}"
            pa = f"{m['team_passes'][0]}/{m['team_passes'][1]}"
            ta = f"{m['team_tackles'][0]}/{m['team_tackles'][1]}"
            sa = f"{m['team_saves'][0]}/{m['team_saves'][1]}"
            print(
                f"  #{m['match_id']:<8d} {m['generation']:<7d} {sc:<13} "
                f"{m['interest_score']:<10.1f} {sh:<9} {pa:<9} {ta:<9} {sa:<7}"
            )
    print("=" * 80)


def resume_num_envs_warning(checkpoint_dir: str, num_envs: int) -> Optional[str]:
    """Return a warning when the last recorded generation ran with a different `num_envs`.

    Returns None when the history is absent/unreadable, empty, or its last generation record
    carries no `num_envs` (legacy records), or when the values agree.
    """
    history_path = os.path.join(checkpoint_dir, "training_history.json")
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            generations = json.load(f).get("generations") or []
    except (OSError, ValueError, AttributeError):
        return None
    if not generations or not isinstance(generations[-1], dict):
        return None
    recorded = generations[-1].get("num_envs")
    if recorded is None or int(recorded) == int(num_envs):
        return None
    return (
        f"WARNING: resuming with num_envs={num_envs} but generation "
        f"{generations[-1].get('generation', '?')} was trained with num_envs={int(recorded)}. "
        f"Samples per generation and the co-evolution window pacing will differ from the earlier run."
    )


def _positive_int(raw: str) -> int:
    """Argparse type parser accepting integers >= 1."""
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer value: {raw!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train 6v6 MARL Football with Independent Red vs. Blue Team PPO Policies."
    )
    parser.add_argument(
        "--total-updates",
        type=int,
        default=100,
        help="Number of PPO generations to train in this session (default: 100).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel environments (default: 8 so 1 generation = 8 full matches).",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=2700,
        help="RL steps per environment per generation (default: 2700 = 1 full 180s match).",
    )
    parser.add_argument(
        "--interest-threshold",
        type=float,
        default=0.0,
        help="Minimum episode interest_score to save a 60 FPS JSON replay (default: 0.0 = save all matches).",
    )
    parser.add_argument(
        "--save-best-only",
        "--best-match-per-gen",
        "--save-best-match-only",
        dest="save_best_only",
        action="store_true",
        help="Only save the single best match replay (highest interest score) per generation when running with num_envs > 1.",
    )
    parser.add_argument(
        "--replay-every",
        type=_positive_int,
        default=1,
        help="Write replays only when generation %% N == 0 or on the last generation of this invocation (default: 1 = every generation).",
    )
    parser.add_argument(
        "--replay-kind",
        type=str,
        choices=REPLAY_KINDS,
        default="both",
        help="Which replay kinds to write on writing generations: `both`, `uniform`, or `best` (default: both).",
    )
    parser.add_argument(
        "--shared-teams",
        action="store_true",
        help="Share a single field_policy and keeper_policy across both Red and Blue (default: False, Red and Blue train separate brains).",
    )
    parser.add_argument(
        "--shared-keepers",
        action="store_true",
        help="Play both teams' keepers with one network, updated once per generation on both keeper seats' data; the two "
        "field networks stay separate (needs separate teams and no opponent pool). Stored with the checkpoint and must "
        "match on resume.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (`cpu` or `cuda`).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--critic-only-until-gen",
        type=int,
        default=0,
        help="Update only critics (freeze actors and skip co-evolution sync) for generations <= N (default: 0).",
    )
    parser.add_argument(
        "--config-override",
        dest="config_overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override one `rewards` or `ppo` config field, e.g. `rewards.shaping_mode=legacy` (repeatable). "
        "Applied overrides are stored with the checkpoint and must match on resume.",
    )
    parser.add_argument(
        "--pool-envs",
        type=int,
        default=0,
        help="Environments per generation whose one team is frozen to a snapshot from the opponent pool "
        "(default: 0 = pure self-play). Stored with the checkpoint and must match on resume.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=DEFAULT_POOL_SIZE,
        help="Number of newest generation snapshots forming the opponent pool (default: 10).",
    )
    parser.add_argument(
        "--accept-config-change",
        action="store_true",
        help="Resume even when --config-override or pool values differ from those stored in the checkpoint.",
    )

    # Lifecycle & Batch Management Flags
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from active `checkpoints/training_state.pt` (automatic if checkpoints exist).",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Wipe active `checkpoints/` and `replays/` and start a brand-new batch from Generation 1.",
    )
    parser.add_argument(
        "--archive",
        type=str,
        nargs="?",
        const="",
        default=None,
        metavar="BATCH_NAME",
        help="Archive active `checkpoints/` and `replays/` to `archives/<BATCH_NAME>/` (defaults to a timestamp-based name if omitted) and exit.",
    )
    parser.add_argument(
        "--load-archive",
        type=str,
        default=None,
        metavar="BATCH_NAME",
        help="Restore `archives/<BATCH_NAME>/` into active `checkpoints/` and `replays/` and exit.",
    )
    parser.add_argument(
        "--list-archives",
        action="store_true",
        help="List all archived training batches in `archives/` and exit.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current active batch generation count, match count, and top matches and exit.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    ckpt_dir = "checkpoints"
    rep_dir = DEFAULT_CONFIG.replay.save_dir

    # 1. Handle standalone management commands
    if args.status:
        print_active_status(ckpt_dir, rep_dir)
        return

    if args.list_archives:
        batches = list_archived_batches()
        if not batches:
            print("No archived batches found in `archives/`.")
        else:
            print(
                f"{'Archive Name':<28} {'Generations':<13} {'Matches':<10} "
                f"{'Best Interest':<15} {'Last Updated'}"
            )
            print("-" * 88)
            for b in batches:
                print(
                    f"{b['name']:<28} {b['generations']:<13d} {b['matches']:<10d} "
                    f"{b['best_interest']:<15.1f} {b['last_updated']}"
                )
        return

    if args.archive is not None:
        archive_name = args.archive.strip() if args.archive.strip() else None
        dest = archive_batch(archive_name, ckpt_dir, rep_dir)
        print(f"Archived active training batch (checkpoints + replays) to: {dest}")
        return

    if args.load_archive:
        src = load_archived_batch(args.load_archive, ckpt_dir, rep_dir)
        print(f"Restored archived batch from `{src}` into active `checkpoints/` and `replays/`.")
        print_active_status(ckpt_dir, rep_dir)
        return

    # 2. Build the run config first so a bad override aborts before anything on disk changes
    cfg, applied_overrides = parse_config_overrides(
        args.config_overrides,
        replace(
            DEFAULT_CONFIG,
            replay=replace(
                DEFAULT_CONFIG.replay,
                interest_threshold=args.interest_threshold,
                save_best_only=args.save_best_only,
            ),
        ),
    )

    validate_pool_settings(
        args.pool_envs, args.pool_size, args.num_envs, not args.shared_teams, args.seed, args.shared_keepers
    )
    validate_replay_settings(args.replay_every, args.replay_kind)

    # 3. Handle `--from-scratch` vs automatic/explicit `--resume`
    if args.from_scratch:
        print("Starting a NEW training batch from scratch (clearing active `checkpoints/` and `replays/`)...")
        reset_active_batch(ckpt_dir, rep_dir)

    trainer = ParameterSharingPPOTrainer(
        config=cfg,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        device=args.device,
        checkpoint_dir=ckpt_dir,
        seed=args.seed,
        separate_teams=not args.shared_teams,
        save_best_only=args.save_best_only,
        write_root_summary=True,
        critic_only_until_gen=args.critic_only_until_gen,
        config_overrides=applied_overrides,
        accept_config_change=args.accept_config_change,
        pool_envs=args.pool_envs,
        pool_size=args.pool_size,
        replay_every=args.replay_every,
        replay_kind=args.replay_kind,
        shared_keepers=args.shared_keepers,
    )

    state_file = os.path.join(ckpt_dir, "training_state.pt")
    if not args.from_scratch and (args.resume or os.path.exists(state_file)):
        mismatch = resume_num_envs_warning(ckpt_dir, args.num_envs)
        if mismatch:
            print("!" * 88)
            print(mismatch)
            print("!" * 88)
        if trainer.load_checkpoints():
            print(
                f"Resuming active batch from Generation {trainer.current_generation} "
                f"(Completed Matches: {len(trainer.match_history)}) -> "
                f"Training +{args.total_updates} generations "
                f"(Gen {trainer.current_generation + 1}..{trainer.current_generation + args.total_updates})..."
            )
    else:
        mode_label = "Independent Red vs. Blue Policies" if not args.shared_teams else "Shared Mirror Policies"
        if args.shared_keepers:
            mode_label += ", one shared keeper network"
        print(
            f"Training new batch ({mode_label}) from Generation 1..{args.total_updates} "
            f"(num_envs={args.num_envs}, rollout_steps={args.rollout_steps}, "
            f"interest_threshold={args.interest_threshold})..."
        )

    trainer.train(total_updates=args.total_updates)
    print(f"\nTraining complete! View match summary at: {os.path.join(rep_dir, 'matches_summary.md')}")


if __name__ == "__main__":
    main()
