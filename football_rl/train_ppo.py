"""Multi-Agent PPO with Per-Team Role Parameter Sharing (Red Team vs. Blue Team) for 6v6 MARL Football."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import replace
from datetime import datetime
import json
import os
import random
import re
import shutil
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
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


class ActorCritic(nn.Module):
    """Continuous-action Gaussian Actor-Critic network for shared role policies.

    Supports CTDE (Centralized Training with Decentralized Execution):
    - Actor policy network observes decentralized observations (obs_dim = 76).
    - Critic value network observes augmented centralized state (critic_obs_dim = 130 = 76 + 54).
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dims: Tuple[int, int] = (256, 256),
        central_state_dim: int = CENTRAL_STATE_DIM,
    ) -> None:
        super().__init__()
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
        out_layer = layer_init(nn.Linear(h2, action_dim), std=0.01)
        if action_dim == 4:
            # Sensible initial action priors before Tanh:
            # [0] throttle = +0.35 (move forward), [1] steer = 0.0 (unbiased),
            # [2] kick = -0.35 (hold & dribble ball by default until intentionally kicking > 0.35),
            # [3] special/tackle = -0.8 (do not spam air-tackles by default until > 0.5)
            with torch.no_grad():
                out_layer.bias.copy_(torch.tensor([0.35, 0.0, -0.35, -0.8], dtype=torch.float32))

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, h1)),
            nn.Tanh(),
            layer_init(nn.Linear(h1, h2)),
            nn.Tanh(),
            out_layer,
            nn.Tanh(),  # Bound mean to [-1.0, 1.0]
        )
        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), -0.5))

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
        self, obs: torch.Tensor, central_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        critic_in = self._prepare_critic_input(obs, central_state)
        return self.critic(critic_in).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        central_state: Optional[torch.Tensor] = None,
        return_mean: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Return `(action, log_prob, entropy, value)`; with `return_mean` the policy mean action is appended."""
        actor_obs = obs[..., :self.obs_dim] if obs.shape[-1] > self.obs_dim else obs
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
        if return_mean:
            return action, log_prob, entropy, value, action_mean
        return action, log_prob, entropy, value


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


class ParameterSharingPPOTrainer:
    """MARL PPO trainer with independent Red (`TEAM_A`) vs. Blue (`TEAM_B`) role policies by default.

    - Red Team (`TEAM_A`, indices 0..5): `blue_keeper_policy` ([0]) + `blue_field_policy` ([1..5])
    - Blue Team (`TEAM_B`, indices 6..11): `red_keeper_policy` ([6]) + `red_field_policy` ([7..11])
    If `separate_teams=False`, both teams share `blue_field_policy` and `blue_keeper_policy`.
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
    ) -> None:
        self.cfg = config
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.device = torch.device(device)
        self.inference_device = torch.device("cpu") if self.device.type == "cuda" else self.device
        self.checkpoint_dir = checkpoint_dir
        self.seed = seed
        self.separate_teams = separate_teams
        self.write_root_summary = write_root_summary
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
        self.blue_field_policy = ActorCritic(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dims=self.cfg.ppo.hidden_dims,
        ).to(self.device)
        self.blue_keeper_policy = ActorCritic(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dims=self.cfg.ppo.hidden_dims,
        ).to(self.device)

        self.blue_field_optimizer = optim.Adam(
            self.blue_field_policy.parameters(), lr=self.cfg.ppo.learning_rate, eps=1e-5
        )
        self.blue_keeper_optimizer = optim.Adam(
            self.blue_keeper_policy.parameters(), lr=self.cfg.ppo.learning_rate, eps=1e-5
        )

        # Blue Team (TEAM_B) Policies (independent weights & optimizers when separate_teams=True)
        if self.separate_teams:
            self.red_field_policy = ActorCritic(
                obs_dim=OBS_DIM,
                action_dim=ACTION_DIM,
                hidden_dims=self.cfg.ppo.hidden_dims,
            ).to(self.device)
            self.red_keeper_policy = ActorCritic(
                obs_dim=OBS_DIM,
                action_dim=ACTION_DIM,
                hidden_dims=self.cfg.ppo.hidden_dims,
            ).to(self.device)
            self.red_field_optimizer = optim.Adam(
                self.red_field_policy.parameters(), lr=self.cfg.ppo.learning_rate, eps=1e-5
            )
            self.red_keeper_optimizer = optim.Adam(
                self.red_keeper_policy.parameters(), lr=self.cfg.ppo.learning_rate, eps=1e-5
            )
        else:
            self.red_field_policy = self.blue_field_policy
            self.red_keeper_policy = self.blue_keeper_policy
            self.red_field_optimizer = self.blue_field_optimizer
            self.red_keeper_optimizer = self.blue_keeper_optimizer

        # CPU shadow policies for fast rollout inference (when device == 'cuda')
        if self.device.type == "cuda":
            self.cpu_blue_field_policy = ActorCritic(
                obs_dim=OBS_DIM,
                action_dim=ACTION_DIM,
                hidden_dims=self.cfg.ppo.hidden_dims,
            ).to(self.inference_device)
            self.cpu_blue_keeper_policy = ActorCritic(
                obs_dim=OBS_DIM,
                action_dim=ACTION_DIM,
                hidden_dims=self.cfg.ppo.hidden_dims,
            ).to(self.inference_device)
            if self.separate_teams:
                self.cpu_red_field_policy = ActorCritic(
                    obs_dim=OBS_DIM,
                    action_dim=ACTION_DIM,
                    hidden_dims=self.cfg.ppo.hidden_dims,
                ).to(self.inference_device)
                self.cpu_red_keeper_policy = ActorCritic(
                    obs_dim=OBS_DIM,
                    action_dim=ACTION_DIM,
                    hidden_dims=self.cfg.ppo.hidden_dims,
                ).to(self.inference_device)
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

    def _sync_cpu_policies(self) -> None:
        """Synchronize training weights from CUDA to the CPU rollout inference policies."""
        if self.device.type != "cuda":
            return
        self.cpu_blue_field_policy.load_state_dict(self.blue_field_policy.state_dict())
        self.cpu_blue_keeper_policy.load_state_dict(self.blue_keeper_policy.state_dict())
        if self.separate_teams:
            self.cpu_red_field_policy.load_state_dict(self.red_field_policy.state_dict())
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
        """
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.inference_device)
        cen_t = (
            torch.as_tensor(central_batch, dtype=torch.float32, device=self.inference_device)
            if central_batch is not None
            else None
        )
        actions = torch.zeros((self.num_envs, self.num_agents, ACTION_DIM), dtype=torch.float32, device=self.inference_device)
        logprobs = torch.zeros((self.num_envs, self.num_agents), dtype=torch.float32, device=self.inference_device)
        values = torch.zeros((self.num_envs, self.num_agents), dtype=torch.float32, device=self.inference_device)
        means = (
            torch.zeros((self.num_envs, self.num_agents, ACTION_DIM), dtype=torch.float32, device=self.inference_device)
            if with_means
            else None
        )

        if not self.separate_teams:
            k_obs = obs_t[:, self.keeper_indices, :].reshape(-1, OBS_DIM)
            k_cen = (
                cen_t[:, self.keeper_indices, :].reshape(-1, CENTRAL_STATE_DIM)
                if cen_t is not None
                else None
            )
            k_act, k_logp, _, k_val, *k_mean = self.cpu_blue_keeper_policy.get_action_and_value(
                k_obs, deterministic=deterministic, central_state=k_cen, return_mean=with_means
            )
            if with_means:
                means[:, self.keeper_indices, :] = k_mean[0].view(self.num_envs, len(self.keeper_indices), ACTION_DIM)
            actions[:, self.keeper_indices, :] = k_act.view(self.num_envs, len(self.keeper_indices), ACTION_DIM)
            logprobs[:, self.keeper_indices] = k_logp.view(self.num_envs, len(self.keeper_indices))
            values[:, self.keeper_indices] = k_val.view(self.num_envs, len(self.keeper_indices))

            f_obs = obs_t[:, self.field_indices, :].reshape(-1, OBS_DIM)
            f_cen = (
                cen_t[:, self.field_indices, :].reshape(-1, CENTRAL_STATE_DIM)
                if cen_t is not None
                else None
            )
            f_act, f_logp, _, f_val, *f_mean = self.cpu_blue_field_policy.get_action_and_value(
                f_obs, deterministic=deterministic, central_state=f_cen, return_mean=with_means
            )
            if with_means:
                means[:, self.field_indices, :] = f_mean[0].view(self.num_envs, len(self.field_indices), ACTION_DIM)
            actions[:, self.field_indices, :] = f_act.view(self.num_envs, len(self.field_indices), ACTION_DIM)
            logprobs[:, self.field_indices] = f_logp.view(self.num_envs, len(self.field_indices))
            values[:, self.field_indices] = f_val.view(self.num_envs, len(self.field_indices))
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
                g_act, g_logp, _, g_val, *g_mean = policy.get_action_and_value(
                    g_obs, deterministic=deterministic, central_state=g_cen, return_mean=with_means
                )
                if with_means:
                    means[:, indices, :] = g_mean[0].view(self.num_envs, cnt, ACTION_DIM)
                actions[:, indices, :] = g_act.view(self.num_envs, cnt, ACTION_DIM)
                logprobs[:, indices] = g_logp.view(self.num_envs, cnt)
                values[:, indices] = g_val.view(self.num_envs, cnt)

        result = (
            actions.cpu().numpy(),
            logprobs.cpu().numpy(),
            values.cpu().numpy(),
        )
        if with_means:
            return result + (means.cpu().numpy(),)
        return result

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

    def collect_rollouts(self, active_generation: int) -> Dict[str, Any]:
        """Collect `self.rollout_steps` across all parallel environments and compute per-agent GAE."""
        T = self.rollout_steps
        N = self.num_envs

        buf_obs = np.zeros((T, N, self.num_agents, OBS_DIM), dtype=np.float32)
        buf_central = np.zeros((T, N, self.num_agents, CENTRAL_STATE_DIM), dtype=np.float32)
        buf_act = np.zeros((T, N, self.num_agents, ACTION_DIM), dtype=np.float32)
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
            buf_act[t] = actions
            buf_logp[t] = logprobs
            buf_val[t] = values

            act_clipped = np.clip(actions, -1.0, 1.0)
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
                        "replay_path": rpath or "",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    self.match_history.append(match_record)
                    gen_completed_matches.append((env_i, match_record))

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
        if self.save_best_only and gen_completed_matches and self.cfg.replay.enable_recording:
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
        advantages = np.zeros_like(buf_rew)
        lastgaelam = np.zeros((N, self.num_agents), dtype=np.float32)

        gamma = self.cfg.ppo.gamma
        gae_lambda = self.cfg.ppo.gae_lambda

        for t in reversed(range(T)):
            next_nonterminal = 1.0 - buf_done[t]
            next_val = next_values if t == T - 1 else buf_val[t + 1]
            delta = buf_rew[t] + gamma * next_val * next_nonterminal - buf_val[t]
            lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + buf_val

        return {
            "obs": buf_obs,
            "central_state": buf_central,
            "actions": buf_act,
            "logprobs": buf_logp,
            "advantages": advantages,
            "returns": returns,
            "values": buf_val,
            "rewards": buf_rew,
            "completed_episodes": completed_episodes,
            "max_interest": max_interest,
            "exported_replays": exported_replays,
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
    ) -> Dict[str, float]:
        """Run multi-epoch minibatch PPO update on pooled role transitions."""
        batch_size = b_obs.shape[0]
        num_minibatches = min(self.cfg.ppo.num_minibatches, batch_size)
        minibatch_size = max(1, batch_size // num_minibatches)
        clip_coef = self.cfg.ppo.clip_coef

        # Normalize once across the full rollout batch for the team unit
        b_advantages = self._normalize_advantages_per_role(b_obs, b_advantages)

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

                _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                    mb_obs, mb_actions, central_state=cen_mb
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
                v_loss_unclipped = (newvalue - mb_returns) ** 2
                v_clipped = mb_values + torch.clamp(
                    newvalue - mb_values, -clip_coef, clip_coef
                )
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
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
        }

    def update_policies(self, rollout: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Update Red and Blue role networks independently (or pooled if `separate_teams=False`)."""
        obs = rollout["obs"]
        central_state = rollout.get("central_state")
        actions = rollout["actions"]
        logprobs = rollout["logprobs"]
        advantages = rollout["advantages"]
        returns = rollout["returns"]
        values = rollout["values"]

        def _extract_role_tensors(
            role_indices: List[int],
        ) -> Tuple[torch.Tensor, ...]:
            r_obs = torch.as_tensor(
                obs[:, :, role_indices, :].reshape(-1, OBS_DIM),
                dtype=torch.float32,
                device=self.device,
            )
            r_cen = (
                torch.as_tensor(
                    central_state[:, :, role_indices, :].reshape(-1, CENTRAL_STATE_DIM),
                    dtype=torch.float32,
                    device=self.device,
                )
                if central_state is not None
                else None
            )
            r_act = torch.as_tensor(
                actions[:, :, role_indices, :].reshape(-1, ACTION_DIM),
                dtype=torch.float32,
                device=self.device,
            )
            r_logp = torch.as_tensor(
                logprobs[:, :, role_indices].reshape(-1),
                dtype=torch.float32,
                device=self.device,
            )
            r_adv = torch.as_tensor(
                advantages[:, :, role_indices].reshape(-1),
                dtype=torch.float32,
                device=self.device,
            )
            r_ret = torch.as_tensor(
                returns[:, :, role_indices].reshape(-1),
                dtype=torch.float32,
                device=self.device,
            )
            r_val = torch.as_tensor(
                values[:, :, role_indices].reshape(-1),
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
            )
            k_tensors = _extract_role_tensors(self.keeper_indices)
            keeper_metrics = self._ppo_update_role(
                self.blue_keeper_policy,
                self.blue_keeper_optimizer,
                k_tensors[0], k_tensors[1], k_tensors[2], k_tensors[3], k_tensors[4], k_tensors[5],
                b_central=k_tensors[6],
            )
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
        )
        bk_tensors = _extract_role_tensors(self.blue_keeper_indices)
        blue_keeper_m = self._ppo_update_role(
            self.blue_keeper_policy,
            self.blue_keeper_optimizer,
            bk_tensors[0], bk_tensors[1], bk_tensors[2], bk_tensors[3], bk_tensors[4], bk_tensors[5],
            b_central=bk_tensors[6],
        )
        rf_tensors = _extract_role_tensors(self.red_field_indices)
        red_field_m = self._ppo_update_role(
            self.red_field_policy,
            self.red_field_optimizer,
            rf_tensors[0], rf_tensors[1], rf_tensors[2], rf_tensors[3], rf_tensors[4], rf_tensors[5],
            b_central=rf_tensors[6],
        )
        rk_tensors = _extract_role_tensors(self.red_keeper_indices)
        red_keeper_m = self._ppo_update_role(
            self.red_keeper_policy,
            self.red_keeper_optimizer,
            rk_tensors[0], rk_tensors[1], rk_tensors[2], rk_tensors[3], rk_tensors[4], rk_tensors[5],
            b_central=rk_tensors[6],
        )
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
                "next_match_id": self._get_max_saved_match_id() + 1,
                "separate_teams": self.separate_teams,
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
        """Load state dict into ActorCritic, adapting 76-dim critic weights to 130-dim if needed."""
        target_dict = policy.state_dict()
        adapted = {}
        for k, v in weights.items():
            if k == "critic.0.weight" and v.shape != target_dict[k].shape:
                # If old critic was trained on obs_dim (76), pad extra central_state_dim (54) columns with 0
                new_w = target_dict[k].clone()
                old_in_dim = v.shape[1]
                new_w[:, :old_in_dim] = v
                new_w[:, old_in_dim:] = 0.0
                adapted[k] = new_w
            else:
                adapted[k] = v
        policy.load_state_dict(adapted, strict=False)

    def load_checkpoints(self) -> bool:
        """Load Red and Blue policies, optimizers, and cumulative generation/match state."""
        state_path = os.path.join(self.checkpoint_dir, "training_state.pt")
        field_path = os.path.join(self.checkpoint_dir, "field_policy.pt")
        keeper_path = os.path.join(self.checkpoint_dir, "keeper_policy.pt")

        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device, weights_only=False)
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
                if red_k_weights is not None:
                    self._load_adapted_weights(self.red_keeper_policy, red_k_weights)

            if "blue_field_optimizer" in state:
                try:
                    self.blue_field_optimizer.load_state_dict(state["blue_field_optimizer"])
                except Exception:
                    pass
            elif "field_optimizer" in state:
                try:
                    self.blue_field_optimizer.load_state_dict(state["field_optimizer"])
                except Exception:
                    pass

            if "blue_keeper_optimizer" in state:
                try:
                    self.blue_keeper_optimizer.load_state_dict(state["blue_keeper_optimizer"])
                except Exception:
                    pass
            elif "keeper_optimizer" in state:
                try:
                    self.blue_keeper_optimizer.load_state_dict(state["keeper_optimizer"])
                except Exception:
                    pass

            if self.separate_teams:
                if "red_field_optimizer" in state:
                    try:
                        self.red_field_optimizer.load_state_dict(state["red_field_optimizer"])
                    except Exception:
                        pass
                if "red_keeper_optimizer" in state:
                    try:
                        self.red_keeper_optimizer.load_state_dict(state["red_keeper_optimizer"])
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
        """Move `dst` parameters a fraction `tau` of the way towards `src` (in place)."""
        with torch.no_grad():
            for p_dst, p_src in zip(dst.parameters(), src.parameters()):
                p_dst.data.copy_((1.0 - tau) * p_dst.data + tau * p_src.data)

    def _coevolution_step(self, completed_matches: List[Dict[str, Any]]) -> bool:
        """Record newly completed matches and soft-sync the trailing team when one side dominates.

        Scores enter the rolling window in completion order. The condition is evaluated only once
        the window holds `COEVO_WINDOW_MATCHES` matches: at least 8 total goals with one side
        scoring >= 3x the other (opponent floored at 1 goal). The trailing team's policies are then
        blended 35% towards the dominant team's and the window is cleared. Returns True on a sync.
        """
        if not self.separate_teams:
            return False
        for m in completed_matches:
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
            self._soft_blend(self.red_keeper_policy, self.blue_keeper_policy, 0.35)
            print(
                f"  [Co-Evolution Sync] Red dominated ({sum_a}-{sum_b} over {COEVO_WINDOW_MATCHES} matches). "
                f"Soft-synced Blue policy (tau=0.35) to restore parity."
            )
        elif sum_b >= 3 * max(1, sum_a):
            # Team B is dominating >= 75% -> soft-sync Team A towards Team B
            self._soft_blend(self.blue_field_policy, self.red_field_policy, 0.35)
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
                uniform_env = uniform_env_index(self.seed, gen, self.num_envs)
                if recording:
                    self.vec_env.set_uniform_capture(uniform_env)
                rollout = self.collect_rollouts(active_generation=gen)
                uniform_replay = None
                if recording:
                    uniform_replay = self.vec_env.export_uniform_replay(uniform_env)
                    self.vec_env.set_uniform_capture(None)
                metrics = self.update_policies(rollout)
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
                if self.separate_teams:
                    for pol in (
                        self.blue_field_policy,
                        self.blue_keeper_policy,
                        self.red_field_policy,
                        self.red_keeper_policy,
                    ):
                        torch.clamp_(pol.actor_logstd.data, -1.25, -0.35)
                else:
                    for pol in (self.blue_field_policy, self.blue_keeper_policy):
                        torch.clamp_(pol.actor_logstd.data, -1.25, -0.35)
                self._sync_cpu_policies()

                # Balanced Co-Evolution & Underdog Soft-Sync (over the rolling window of completed matches)
                self._coevolution_step(self.match_history[matches_before:])

                gen_record = {
                    "generation": gen,
                    "total_env_steps": self.total_env_steps,
                    "sps": sps,
                    "red_mean_reward": round(team_a_rew, 5),
                    "blue_mean_reward": round(team_b_rew, 5),
                    "mean_field_reward": round(mean_field_rew, 5),
                    "mean_keeper_reward": round(mean_keeper_rew, 5),
                    "max_interest": round(float(rollout["max_interest"]), 2),
                    "completed_matches_in_gen": int(rollout["completed_episodes"]),
                    "num_envs": self.num_envs,
                    "rollout_steps": self.rollout_steps,
                    "total_matches_so_far": len(self.match_history),
                    "uniform_env": uniform_env,
                    "uniform_replay": uniform_replay,
                }
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

        items.append({"file": f, "gen": gen, "mid": mid, "score": score})

    gen_to_items = defaultdict(list)
    for item in items:
        gen_to_items[item["gen"]].append(item)

    for gen in gen_to_items:
        gen_to_items[gen].sort(key=lambda x: (x["score"], x["mid"]), reverse=True)

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
        "--shared-teams",
        action="store_true",
        help="Share a single field_policy and keeper_policy across both Red and Blue (default: False, Red and Blue train separate brains).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (`cpu` or `cuda`).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

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

    # 2. Handle `--from-scratch` vs automatic/explicit `--resume`
    if args.from_scratch:
        print("Starting a NEW training batch from scratch (clearing active `checkpoints/` and `replays/`)...")
        reset_active_batch(ckpt_dir, rep_dir)

    cfg = replace(
        DEFAULT_CONFIG,
        replay=replace(
            DEFAULT_CONFIG.replay,
            interest_threshold=args.interest_threshold,
            save_best_only=args.save_best_only,
        ),
    )
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
        print(
            f"Training new batch ({mode_label}) from Generation 1..{args.total_updates} "
            f"(num_envs={args.num_envs}, rollout_steps={args.rollout_steps}, "
            f"interest_threshold={args.interest_threshold})..."
        )

    trainer.train(total_updates=args.total_updates)
    print(f"\nTraining complete! View match summary at: {os.path.join(rep_dir, 'matches_summary.md')}")


if __name__ == "__main__":
    main()
