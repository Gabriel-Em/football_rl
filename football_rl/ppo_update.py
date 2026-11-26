"""PPO update: per-role advantage normalisation, clipped policy and value losses, and per-generation critic metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from football_rl.config import CENTRAL_STATE_DIM, OBS_DIM
from football_rl.policy import (
    ActorCritic,
    FIRE_HEADS,
    FIRE_STATS_STRIDE,
    HOLDING_OBS_INDEX,
    KICK_TARGET_FREE,
    _validate_hybrid7_targets,
    hybrid_entropy_bonus,
    kick_targets,
    masked_target_log_probs_and_entropy,
)
from football_rl.returns import compute_mc_returns, pearson_correlation


class PPOUpdateMixin:
    """PPO update of `ParameterSharingPPOTrainer`."""

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
            if self.policy_head in FIRE_HEADS:
                empty.update({"fire_sat_kick": nan, "fire_sat_spec": nan, "p_kick": nan, "p_spec": nan})
            if self.policy_head == "hybrid7":
                empty.update({"target_rows": 0, "free_share": nan, "target_entropy": nan})
            return empty
        num_minibatches = min(self.cfg.ppo.num_minibatches, batch_size)
        minibatch_size = max(1, batch_size // num_minibatches)
        clip_coef = self.cfg.ppo.clip_coef
        separate_fire_entropy = self.policy_head in FIRE_HEADS and self.cfg.ppo.fire_entropy == "separate"

        # Fire statistics of the policy that produced this batch, taken before any update.
        fire_metrics: Dict[str, float] = {}
        if self.policy_head in FIRE_HEADS:
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

        target_metrics: Dict[str, Any] = {}
        b_ready: Optional[torch.Tensor] = None
        b_valid: Optional[torch.Tensor] = None
        b_multi: Optional[torch.Tensor] = None
        if self.policy_head == "hybrid7":
            b_ready, b_valid, _ = kick_targets(b_obs)
            _validate_hybrid7_targets(b_actions, b_ready, b_valid)
            b_multi = b_ready & (b_valid.sum(dim=-1) > 1)
            t_mask = (b_actions[..., 5] > 0.5) & b_multi
            t_rows = int(t_mask.sum().item())
            t_free = (
                float((torch.round(b_actions[t_mask, 4]).to(torch.int64) == KICK_TARGET_FREE).to(torch.float32).mean().item())
                if t_rows > 0
                else float("nan")
            )
            target_metrics = {"target_rows": t_rows, "free_share": t_free}

        pg_losses: List[float] = []
        v_losses: List[float] = []
        entropy_losses: List[float] = []
        device = b_obs.device
        target_ent_sum = torch.zeros((), dtype=torch.float64, device=device)
        target_ent_count = torch.zeros((), dtype=torch.int64, device=device)

        last_epoch = self.cfg.ppo.update_epochs - 1
        for epoch in range(self.cfg.ppo.update_epochs):
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
                    mb_ready = b_ready[mb_inds] if b_ready is not None else None
                    mb_valid = b_valid[mb_inds] if b_valid is not None else None
                    mb_multi = b_multi[mb_inds] if b_multi is not None else None
                else:
                    mb_obs = b_obs
                    mb_actions = b_actions
                    mb_logprobs = b_logprobs
                    mb_advantages = b_advantages
                    mb_returns = b_returns
                    mb_values = b_values
                    cen_mb = b_central
                    mb_ready = b_ready
                    mb_valid = b_valid
                    mb_multi = b_multi

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
                    extra = (
                        {"ready": mb_ready, "valid": mb_valid, "validate_targets": False}
                        if policy.kick_target
                        else {}
                    )
                    _, newlogprob, entropy, newvalue, heads = policy.get_action_and_value(
                        mb_obs, mb_actions, central_state=cen_mb, return_heads=True, **extra, **value_kwargs
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
                    if policy.kick_target:
                        head_logstd, head_p, head_target_logits = heads
                        bonus = hybrid_entropy_bonus(
                            head_logstd,
                            head_p,
                            self.cfg.ppo.ent_coef,
                            "separate",
                            self.cfg.ppo.fire_ent_coef,
                            target_logits=head_target_logits,
                            target_ent_coef=self.cfg.ppo.target_ent_coef,
                            target=mb_actions[..., 4],
                            ready=mb_ready,
                            valid=mb_valid,
                        ).mean()
                        if epoch == last_epoch:
                            with torch.no_grad():
                                _, h_pi = masked_target_log_probs_and_entropy(head_target_logits.detach(), mb_valid)
                                target_ent_sum = target_ent_sum + torch.where(
                                    mb_multi, h_pi.to(torch.float64), torch.zeros_like(h_pi, dtype=torch.float64)
                                ).sum()
                                target_ent_count = target_ent_count + mb_multi.sum()
                    else:
                        head_logstd, head_p = heads
                        # hybrid6 weights its aim entropy by each row's holding flag, read from the raw minibatch observations.
                        holding = mb_obs[..., HOLDING_OBS_INDEX] > 0.5 if policy.aim else None
                        bonus = hybrid_entropy_bonus(
                            head_logstd, head_p, self.cfg.ppo.ent_coef, "separate", self.cfg.ppo.fire_ent_coef, holding
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

        if self.policy_head == "hybrid7":
            cnt = int(target_ent_count.item())
            target_metrics["target_entropy"] = (
                float(target_ent_sum.item() / cnt) if cnt > 0 else float("nan")
            )

        return {
            "pg_loss": float(np.mean(pg_losses)),
            "v_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(entropy_losses)),
            **fire_metrics,
            **target_metrics,
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
