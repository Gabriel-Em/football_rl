"""Saved-state migrations: team-colour history keys and first-layer widening of policy weights and optimizer state."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from football_rl.config import CENTRAL_STATE_DIM, OBS_DIM
from football_rl.policy import ActorCritic


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
