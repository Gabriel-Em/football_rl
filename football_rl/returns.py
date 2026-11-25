"""Advantage and return estimation (GAE, Monte Carlo returns) and the critic's correlation diagnostic."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


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
