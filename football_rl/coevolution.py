"""Co-evolution soft sync: blends the dominated team's field networks toward the dominant team's."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from football_rl.policy import ActorCritic


# Co-evolution soft-sync is evaluated over the last this-many completed matches, independent of how
# many parallel environments (and therefore matches per generation) a run uses.
COEVO_WINDOW_MATCHES = 24


class CoevolutionMixin:
    """Co-evolution sync of `ParameterSharingPPOTrainer`."""

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
