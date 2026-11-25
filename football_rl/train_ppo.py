"""Multi-Agent PPO with Per-Team Role Parameter Sharing (Red Team vs. Blue Team) for 6v6 MARL Football."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import fields, replace
from datetime import datetime
import os
import subprocess
import time
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.optim as optim

from football_rl.archive import (
    archive_batch,
    list_archived_batches,
    load_archived_batch,
    print_active_status,
    reset_active_batch,
    resume_num_envs_warning,
)
from football_rl.coevolution import COEVO_WINDOW_MATCHES, CoevolutionMixin
from football_rl.config import ACTION_DIM, ALL_AGENTS, CENTRAL_STATE_DIM, DEFAULT_CONFIG, EnvConfig, OBS_DIM
from football_rl.config_overrides import parse_config_overrides
from football_rl.env import FootballParallelEnv
from football_rl.opponent_pool import (
    DEFAULT_POOL_SIZE,
    FrozenSeatBatch,
    POOL_TEAM_PREFIX,
    list_pool_generations,
    pool_generator,
    pool_schedule,
    validate_pool_settings,
)
from football_rl.policy import (
    ActorCritic,
    HYBRID_ACTION_DIM,
    _hybrid_log_prob,
    holds_return_statistics,
    policy_head_of_weights,
    to_env_actions,
)
from football_rl.ppo_update import PPOUpdateMixin
from football_rl.returns import gae_advantages
from football_rl.trainer_checkpoints import CheckpointMixin
from football_rl.vec_env import make_vec_football_env


def uniform_env_index(seed: int, generation: int, num_envs: int) -> int:
    """Environment whose match is exported as the unbiased replay of `generation`.

    Drawn from a generator seeded with `seed + generation` before the rollout, so it is independent of
    how the matches turn out and reproducible across resumes.
    """
    return int(np.random.default_rng(int(seed) + int(generation)).integers(int(num_envs)))


REPLAY_KINDS = ("both", "uniform", "best")


def validate_replay_settings(replay_every: Any, replay_kind: Any) -> Tuple[int, str]:
    """Validate `--replay-every` (integer >= 1) and `--replay-kind` (`both`, `uniform`, `best`)."""
    if isinstance(replay_every, bool) or not isinstance(replay_every, int) or replay_every < 1:
        raise ValueError(f"--replay-every / replay_every must be an integer >= 1, got {replay_every!r}")
    if not isinstance(replay_kind, str) or replay_kind not in REPLAY_KINDS:
        raise ValueError(f"--replay-kind / replay_kind must be one of {REPLAY_KINDS}, got {replay_kind!r}")
    return int(replay_every), replay_kind


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


class ParameterSharingPPOTrainer(PPOUpdateMixin, CheckpointMixin, CoevolutionMixin):
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
