"""Multi-Agent PPO with Per-Team Role Parameter Sharing (Blue Team vs. Red Team) for 6v6 MARL Football."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return action, log_prob, entropy, value


class ParameterSharingPPOTrainer:
    """MARL PPO trainer with independent Blue (`TEAM_A`) vs. Red (`TEAM_B`) role policies by default.

    - Blue Team (`TEAM_A`, indices 0..5): `blue_keeper_policy` ([0]) + `blue_field_policy` ([1..5])
    - Red Team (`TEAM_B`, indices 6..11): `red_keeper_policy` ([6]) + `red_field_policy` ([7..11])
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
    ) -> None:
        self.cfg = config
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.device = torch.device(device)
        self.inference_device = torch.device("cpu") if self.device.type == "cuda" else self.device
        self.checkpoint_dir = checkpoint_dir
        self.seed = seed
        self.separate_teams = separate_teams
        self.save_best_only = save_best_only or self.cfg.replay.save_best_only

        # Cumulative training & match tracking across resumed sessions
        self.current_generation: int = 0
        self.total_env_steps: int = 0
        self.match_history: List[Dict[str, Any]] = []
        self.generation_history: List[Dict[str, Any]] = []
        self.recent_team_goals: List[Tuple[int, int]] = []
        self.next_match_id: int = self._get_max_saved_match_id() + 1

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Instantiate vectorized environments (SubprocVecFootballEnv for num_envs > 1, DummyVecFootballEnv for 1)
        self.vec_env = make_vec_football_env(config=self.cfg, num_envs=num_envs)
        self.envs: List[FootballParallelEnv] = getattr(self.vec_env, "envs", [])

        # Role & Team index masks across the 12 agents:
        self.blue_keeper_indices = [0]                    # a_keeper
        self.blue_field_indices = [1, 2, 3, 4, 5]         # a_def_1, a_def_2, a_mid_1, a_mid_2, a_atk_1
        self.red_keeper_indices = [6]                     # b_keeper
        self.red_field_indices = [7, 8, 9, 10, 11]        # b_def_1, b_def_2, b_mid_1, b_mid_2, b_atk_1

        self.keeper_indices = [0, 6]
        self.field_indices = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]

        # Blue Team (TEAM_A) Policies (on self.device, e.g. CUDA)
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

        # Red Team (TEAM_B) Policies (independent weights & optimizers when separate_teams=True)
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

        # Backwards-compatible aliases
        self.field_policy = self.blue_field_policy
        self.keeper_policy = self.blue_keeper_policy
        self.field_optimizer = self.blue_field_optimizer
        self.keeper_optimizer = self.blue_keeper_optimizer

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
        obs_list, info_list = self.vec_env.reset(seeds=seeds, options_list=options_list)
        for env_i in range(self.num_envs):
            for agent_i, agent_name in enumerate(ALL_AGENTS):
                self.current_obs[env_i, agent_i] = obs_list[env_i][agent_name]
                if "central_state" in info_list[env_i].get(agent_name, {}):
                    self.current_central_state[env_i, agent_i] = info_list[env_i][agent_name]["central_state"]

    @torch.no_grad()
    def _select_actions(
        self,
        obs_batch: np.ndarray,
        central_batch: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Query Blue and Red role policies for `(num_envs, len(ALL_AGENTS), OBS_DIM)` observations."""
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.inference_device)
        cen_t = (
            torch.as_tensor(central_batch, dtype=torch.float32, device=self.inference_device)
            if central_batch is not None
            else None
        )
        actions = torch.zeros((self.num_envs, self.num_agents, ACTION_DIM), dtype=torch.float32, device=self.inference_device)
        logprobs = torch.zeros((self.num_envs, self.num_agents), dtype=torch.float32, device=self.inference_device)
        values = torch.zeros((self.num_envs, self.num_agents), dtype=torch.float32, device=self.inference_device)

        if not self.separate_teams:
            k_obs = obs_t[:, self.keeper_indices, :].reshape(-1, OBS_DIM)
            k_cen = (
                cen_t[:, self.keeper_indices, :].reshape(-1, CENTRAL_STATE_DIM)
                if cen_t is not None
                else None
            )
            k_act, k_logp, _, k_val = self.cpu_blue_keeper_policy.get_action_and_value(
                k_obs, deterministic=deterministic, central_state=k_cen
            )
            actions[:, self.keeper_indices, :] = k_act.view(self.num_envs, len(self.keeper_indices), ACTION_DIM)
            logprobs[:, self.keeper_indices] = k_logp.view(self.num_envs, len(self.keeper_indices))
            values[:, self.keeper_indices] = k_val.view(self.num_envs, len(self.keeper_indices))

            f_obs = obs_t[:, self.field_indices, :].reshape(-1, OBS_DIM)
            f_cen = (
                cen_t[:, self.field_indices, :].reshape(-1, CENTRAL_STATE_DIM)
                if cen_t is not None
                else None
            )
            f_act, f_logp, _, f_val = self.cpu_blue_field_policy.get_action_and_value(
                f_obs, deterministic=deterministic, central_state=f_cen
            )
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
                g_act, g_logp, _, g_val = policy.get_action_and_value(
                    g_obs, deterministic=deterministic, central_state=g_cen
                )
                actions[:, indices, :] = g_act.view(self.num_envs, cnt, ACTION_DIM)
                logprobs[:, indices] = g_logp.view(self.num_envs, cnt)
                values[:, indices] = g_val.view(self.num_envs, cnt)

        return (
            actions.cpu().numpy(),
            logprobs.cpu().numpy(),
            values.cpu().numpy(),
        )

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
        gen_completed_matches: List[Dict[str, Any]] = []

        self.vec_env.set_generation(active_generation)

        for t in range(T):
            buf_obs[t] = self.current_obs
            buf_central[t] = self.current_central_state
            actions, logprobs, values = self._select_actions(
                self.current_obs, self.current_central_state
            )
            buf_act[t] = actions
            buf_logp[t] = logprobs
            buf_val[t] = values

            actions_by_env = [
                {
                    agent_name: np.clip(actions[env_i, agent_i], -1.0, 1.0)
                    for agent_i, agent_name in enumerate(ALL_AGENTS)
                }
                for env_i in range(N)
            ]
            obs_list, rew_list, term_list, trunc_list, info_list, done_list = self.vec_env.step(actions_by_env)

            for env_i in range(N):
                next_obs_dict = obs_list[env_i]
                rew_dict = rew_list[env_i]
                term_dict = term_list[env_i]
                trunc_dict = trunc_list[env_i]
                info_dict = info_list[env_i]
                is_env_done = done_list[env_i]

                for agent_i, agent_name in enumerate(ALL_AGENTS):
                    buf_rew[t, env_i, agent_i] = rew_dict[agent_name]
                    is_done = term_dict[agent_name] or trunc_dict[agent_name]
                    buf_done[t, env_i, agent_i] = 1.0 if is_done else 0.0

                # Check if episode ended
                if is_env_done:
                    completed_episodes += 1
                    k_info = info_dict["a_keeper"]
                    ep_interest = float(k_info["interest_score"])
                    max_interest = max(max_interest, ep_interest)
                    rpath = k_info["last_saved_replay"]
                    if rpath:
                        exported_replays.append(rpath)

                    # Record match summary entry
                    fstats = k_info.get("final_stats", {})
                    total_poss = max(1, sum(fstats.get("possession_frames", [1, 1])))
                    poss_pct = [
                        round(100.0 * fstats.get("possession_frames", [1, 1])[0] / total_poss, 1),
                        round(100.0 * fstats.get("possession_frames", [1, 1])[1] / total_poss, 1),
                    ]
                    match_record = {
                        "match_id": int(k_info.get("match_id", self.next_match_id)),
                        "generation": int(active_generation),
                        "score": list(fstats.get("score", [0, 0])),
                        "goals": int(fstats.get("goals", 0)),
                        "interest_score": round(ep_interest, 2),
                        "possession_pct": poss_pct,
                        "shots": list(fstats.get("shots", [0, 0])),
                        "team_passes": list(fstats.get("team_passes", [0, 0])),
                        "completed_passes": int(fstats.get("completed_passes", 0)),
                        "team_tackles": list(fstats.get("team_tackles", [0, 0])),
                        "successful_tackles": int(fstats.get("successful_tackles", 0)),
                        "team_saves": list(fstats.get("team_saves", [0, 0])),
                        "keeper_saves": int(fstats.get("keeper_saves", 0)),
                        "field_5s_violations": int(fstats.get("field_5s_violations", fstats.get("field_3s_violations", 0))),
                        "field_3s_violations": int(fstats.get("field_3s_violations", fstats.get("field_5s_violations", 0))),
                        "keeper_5s_violations": int(fstats.get("keeper_5s_violations", 0)),
                        "replay_path": rpath or "",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    self.match_history.append(match_record)
                    gen_completed_matches.append(match_record)

                    # Auto-reset environment for seamless rollout collection
                    next_gen_hint = active_generation + (1 if t == T - 1 else 0)
                    reset_obs_list, reset_info_list = self.vec_env.reset(
                        options_list=[{
                            "generation": next_gen_hint,
                            "match_id": self._allocate_match_id(),
                        }],
                        env_indices=[env_i],
                    )
                    next_obs_dict = reset_obs_list[0]
                    info_dict = reset_info_list[0]

                for agent_i, agent_name in enumerate(ALL_AGENTS):
                    self.current_obs[env_i, agent_i] = next_obs_dict[agent_name]
                    if "central_state" in info_dict.get(agent_name, {}):
                        self.current_central_state[env_i, agent_i] = info_dict[agent_name]["central_state"]

        # If save_best_only is enabled, retain only the single best match replay per generation
        if self.save_best_only and gen_completed_matches:
            matches_with_replays = [m for m in gen_completed_matches if m.get("replay_path")]
            if matches_with_replays:
                best_match = max(
                    matches_with_replays,
                    key=lambda m: (float(m.get("interest_score", 0.0)), int(m.get("goals", 0))),
                )
                for m in matches_with_replays:
                    if m is not best_match:
                        rpath_to_remove = m.get("replay_path")
                        if rpath_to_remove and os.path.exists(rpath_to_remove):
                            try:
                                os.remove(rpath_to_remove)
                            except OSError:
                                pass
                        m["replay_path"] = ""
                exported_replays = [best_match["replay_path"]] if best_match.get("replay_path") else []

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

        pg_losses: List[float] = []
        v_losses: List[float] = []
        entropy_losses: List[float] = []

        b_inds = np.arange(batch_size)
        for _ in range(self.cfg.ppo.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                cen_mb = b_central[mb_inds] if b_central is not None else None
                _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds], central_state=cen_mb
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if mb_advantages.numel() > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - clip_coef, 1.0 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds], -clip_coef, clip_coef
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - self.cfg.ppo.ent_coef * entropy_loss
                    + v_loss * self.cfg.ppo.vf_coef
                )

                optimizer.zero_grad()
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
        """Update Blue and Red role networks independently (or pooled if `separate_teams=False`)."""
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
        """Save Blue and Red policies, optimizers, generation counter, and match summaries to disk."""
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
                "recent_team_goals": self.recent_team_goals,
            },
            state_path,
        )

        history_payload = {
            "current_generation": self.current_generation,
            "total_matches": len(self.match_history),
            "total_env_steps": self.total_env_steps,
            "separate_teams": self.separate_teams,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "matches": self.match_history,
            "generations": self.generation_history,
        }
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history_payload, f, indent=2)

        if self.cfg.replay.enable_recording:
            self._write_matches_summary_markdown()

        return field_path, keeper_path

    def _compute_series_summary(self) -> Dict[str, int]:
        """Compute cumulative Blue vs. Red wins, draws, and goals across `self.match_history`."""
        blue_wins = 0
        red_wins = 0
        draws = 0
        blue_goals = 0
        red_goals = 0
        for m in self.match_history:
            sb, sr = int(m["score"][0]), int(m["score"][1])
            blue_goals += sb
            red_goals += sr
            if sb > sr:
                blue_wins += 1
            elif sr > sb:
                red_wins += 1
            else:
                draws += 1
        return {
            "blue_wins": blue_wins,
            "red_wins": red_wins,
            "draws": draws,
            "blue_goals": blue_goals,
            "red_goals": red_goals,
        }

    def _write_matches_summary_markdown(self) -> Optional[str]:
        """Write a human-friendly `replays/matches_summary.md` table for quick scanning."""
        os.makedirs(self.cfg.replay.save_dir, exist_ok=True)
        summary_path = os.path.join(self.cfg.replay.save_dir, "matches_summary.md")
        series = self._compute_series_summary()
        mode_str = "Independent Team Policies (Blue vs. Red Co-Evolution)" if self.separate_teams else "Shared Mirror Policies"

        lines: List[str] = [
            "# Match & Generation Summary",
            "",
            f"- **Training Mode:** `{mode_str}`",
            f"- **Generations Trained:** `{self.current_generation}`",
            f"- **Total Completed Matches:** `{len(self.match_history)}`",
            f"- **Head-to-Head Series (Blue - Draws - Red):** `{series['blue_wins']} W - {series['draws']} D - {series['red_wins']} W` (Goals: **Blue `{series['blue_goals']}` - `{series['red_goals']}` Red**)",
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
            root_summary = "matches_summary.md"
            if os.path.abspath(summary_path) != os.path.abspath(root_summary):
                try:
                    with open(root_summary, "w", encoding="utf-8") as f:
                        f.write(content)
                except Exception:
                    pass
            return summary_path

        # 1. Top 15 Most Exciting Matches Leaderboard
        top_matches = sorted(
            self.match_history, key=lambda m: float(m["interest_score"]), reverse=True
        )[:15]
        lines.extend(
            [
                "## Top 15 Most Exciting Matches (by Interest Score)",
                "",
                "| Rank | Match # | Gen | Score (Blue - Red) | Interest | Poss% (B/R) | Shots (B/R) | Passes (B/R) | Tackles (B/R) | Saves (B/R) | Watch Command |",
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
                "| Match # | Gen | Score (Blue - Red) | Interest | Poss% (B/R) | Shots (B/R) | Passes (B/R) | Tackles (B/R) | Saves (B/R) | Fouls (Field 5s/GK 5s) | Replay File |",
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
            f_viol = m.get("field_5s_violations", m.get("field_3s_violations", 0))
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

        root_summary = "matches_summary.md"
        if os.path.abspath(summary_path) != os.path.abspath(root_summary):
            try:
                with open(root_summary, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception:
                pass

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
        """Load Blue and Red policies, optimizers, and cumulative generation/match state."""
        state_path = os.path.join(self.checkpoint_dir, "training_state.pt")
        field_path = os.path.join(self.checkpoint_dir, "field_policy.pt")
        keeper_path = os.path.join(self.checkpoint_dir, "keeper_policy.pt")

        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device, weights_only=False)
            self.current_generation = int(state.get("current_generation", 0))
            self.total_env_steps = int(state.get("total_env_steps", 0))
            self.match_history = list(state.get("match_history", []))
            self.generation_history = list(state.get("generation_history", []))
            self.recent_team_goals = [tuple(x) for x in state.get("recent_team_goals", [])]
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

    def train(self, total_updates: int = 100) -> Dict[str, Any]:
        """Run `total_updates` generations of MARL PPO training."""
        start_time = time.time()
        last_metrics: Dict[str, Any] = {}
        start_gen = self.current_generation + 1
        end_gen = self.current_generation + total_updates

        try:
            for gen in range(start_gen, end_gen + 1):
                self.current_generation = gen
                rollout = self.collect_rollouts(active_generation=gen)
                metrics = self.update_policies(rollout)
                elapsed = max(1e-6, time.time() - start_time)

                step_delta = self.rollout_steps * self.num_envs
                self.total_env_steps += step_delta
                session_steps = (gen - start_gen + 1) * step_delta
                sps = int(session_steps / elapsed)

                mean_field_rew = float(rollout["rewards"][:, :, self.field_indices].mean())
                mean_keeper_rew = float(rollout["rewards"][:, :, self.keeper_indices].mean())
                blue_rew = float(
                    rollout["rewards"][:, :, self.blue_keeper_indices + self.blue_field_indices].mean()
                )
                red_rew = float(
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

                # Balanced Co-Evolution & Underdog Soft-Sync (over rolling 3-generation window)
                if self.separate_teams:
                    gen_matches = [m for m in self.match_history if m.get("generation") == gen]
                    gen_goals_a = sum(m["score"][0] for m in gen_matches)
                    gen_goals_b = sum(m["score"][1] for m in gen_matches)
                    self.recent_team_goals.append((gen_goals_a, gen_goals_b))
                    if len(self.recent_team_goals) > 3:
                        self.recent_team_goals.pop(0)

                    if len(self.recent_team_goals) >= 3:
                        sum_a = sum(x[0] for x in self.recent_team_goals)
                        sum_b = sum(x[1] for x in self.recent_team_goals)
                        if sum_a + sum_b >= 8:
                            if sum_a >= 3 * max(1, sum_b):
                                # Team A is dominating >= 75% -> soft-sync Team B towards Team A
                                with torch.no_grad():
                                    for p_dst, p_src in zip(
                                        self.red_field_policy.parameters(),
                                        self.blue_field_policy.parameters(),
                                    ):
                                        p_dst.data.copy_(0.65 * p_dst.data + 0.35 * p_src.data)
                                    for p_dst, p_src in zip(
                                        self.red_keeper_policy.parameters(),
                                        self.blue_keeper_policy.parameters(),
                                    ):
                                        p_dst.data.copy_(0.65 * p_dst.data + 0.35 * p_src.data)
                                print(
                                    f"  [Co-Evolution Sync] Team A dominated ({sum_a}-{sum_b} over 3 gens). "
                                    f"Soft-synced Red policy (tau=0.35) to restore parity."
                                )
                                self.recent_team_goals.clear()
                                self._sync_cpu_policies()
                            elif sum_b >= 3 * max(1, sum_a):
                                # Team B is dominating >= 75% -> soft-sync Team A towards Team B
                                with torch.no_grad():
                                    for p_dst, p_src in zip(
                                        self.blue_field_policy.parameters(),
                                        self.red_field_policy.parameters(),
                                    ):
                                        p_dst.data.copy_(0.65 * p_dst.data + 0.35 * p_src.data)
                                    for p_dst, p_src in zip(
                                        self.blue_keeper_policy.parameters(),
                                        self.red_keeper_policy.parameters(),
                                    ):
                                        p_dst.data.copy_(0.65 * p_dst.data + 0.35 * p_src.data)
                                print(
                                    f"  [Co-Evolution Sync] Team B dominated ({sum_b}-{sum_a} over 3 gens). "
                                    f"Soft-synced Blue policy (tau=0.35) to restore parity."
                                )
                                self.recent_team_goals.clear()
                                self._sync_cpu_policies()

                gen_record = {
                    "generation": gen,
                    "total_env_steps": self.total_env_steps,
                    "sps": sps,
                    "blue_mean_reward": round(blue_rew, 5),
                    "red_mean_reward": round(red_rew, 5),
                    "mean_field_reward": round(mean_field_rew, 5),
                    "mean_keeper_reward": round(mean_keeper_rew, 5),
                    "max_interest": round(float(rollout["max_interest"]), 2),
                    "completed_matches_in_gen": int(rollout["completed_episodes"]),
                    "total_matches_so_far": len(self.match_history),
                }
                self.generation_history.append(gen_record)

                print(
                    f"[Gen {gen:04d}/{end_gen:04d}] "
                    f"Matches: {len(self.match_history):4d} | "
                    f"Steps: {self.total_env_steps:8d} ({sps:5d} SPS) | "
                    f"Blue Rew: {blue_rew:+.4f} | Red Rew: {red_rew:+.4f} | "
                    f"Interest: {rollout['max_interest']:.1f}"
                )
                for rpath in rollout["exported_replays"]:
                    print(f"  -> Saved replay: {rpath}")

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
                    "blue_mean_reward": blue_rew,
                    "red_mean_reward": red_rew,
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


def archive_batch(
    name: str,
    checkpoint_dir: str = "checkpoints",
    replay_dir: str = "replays",
    archive_root: str = "archives",
) -> str:
    """Save the current active `checkpoints/` and `replays/` into `archives/<name>/`."""
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
        shutil.copytree(replay_dir, os.path.join(dest_dir, "replays"))

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
                h = json.load(f)
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
        h = json.load(f)

    matches = h.get("matches", [])
    blue_w = sum(1 for m in matches if m["score"][0] > m["score"][1])
    red_w = sum(1 for m in matches if m["score"][1] > m["score"][0])
    draws = len(matches) - blue_w - red_w
    blue_g = sum(int(m["score"][0]) for m in matches)
    red_g = sum(int(m["score"][1]) for m in matches)

    print("=" * 80)
    print(f"ACTIVE TRAINING BATCH STATUS")
    print("=" * 80)
    print(f"  Generations Trained : {h.get('current_generation', 0)}")
    print(f"  Completed Matches   : {len(matches)}")
    print(
        f"  Series (Blue - Red) : {blue_w} Wins - {draws} Draws - {red_w} Wins "
        f"(Goals: Blue {blue_g} - {red_g} Red)"
    )
    print(f"  Total RL Steps      : {h.get('total_env_steps', 0):,}")
    print(f"  Last Updated        : {h.get('last_updated', 'N/A')}")
    print(f"  Summary File        : {os.path.join(replay_dir, 'matches_summary.md')}")
    print("-" * 80)

    if matches:
        top = sorted(matches, key=lambda m: float(m.get("interest_score", 0.0)), reverse=True)[:10]
        print("  TOP 10 MOST EXCITING MATCHES:")
        print(
            f"  {'Match #':<9} {'Gen':<7} {'Score (B-R)':<13} {'Interest':<10} "
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 6v6 MARL Football with Independent Blue vs. Red Team PPO Policies."
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
        default=1,
        help="Number of parallel environments (default: 1 so 1 generation = 1 full match).",
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
        help="Share a single field_policy and keeper_policy across both Blue and Red (default: False, Blue and Red train separate brains).",
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
        default=None,
        metavar="BATCH_NAME",
        help="Archive the active `checkpoints/` and `replays/` to `archives/<BATCH_NAME>/` and exit.",
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
    args = parser.parse_args()

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

    if args.archive:
        dest = archive_batch(args.archive, ckpt_dir, rep_dir)
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
    )

    state_file = os.path.join(ckpt_dir, "training_state.pt")
    if not args.from_scratch and (args.resume or os.path.exists(state_file)):
        if trainer.load_checkpoints():
            print(
                f"Resuming active batch from Generation {trainer.current_generation} "
                f"(Completed Matches: {len(trainer.match_history)}) -> "
                f"Training +{args.total_updates} generations "
                f"(Gen {trainer.current_generation + 1}..{trainer.current_generation + args.total_updates})..."
            )
    else:
        mode_label = "Independent Blue vs. Red Policies" if not args.shared_teams else "Shared Mirror Policies"
        print(
            f"Training new batch ({mode_label}) from Generation 1..{args.total_updates} "
            f"(num_envs={args.num_envs}, rollout_steps={args.rollout_steps}, "
            f"interest_threshold={args.interest_threshold})..."
        )

    trainer.train(total_updates=args.total_updates)
    print(f"\nTraining complete! View match summary at: {os.path.join(rep_dir, 'matches_summary.md')}")


if __name__ == "__main__":
    main()
