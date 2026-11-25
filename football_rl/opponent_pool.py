"""Opponent snapshot pool: settings validation, snapshot discovery, the seeded schedule and frozen-seat inference."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from football_rl.config import ACTION_DIM, CENTRAL_STATE_DIM, OBS_DIM
from football_rl.policy import ActorCritic, HYBRID_ACTION_DIM, fire_probability


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
