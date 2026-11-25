"""Team rewards and potential-based shaping, with the per-step potential telemetry."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from football_rl.potentials import ROLE_GK, PotentialResult, PotentialState, compute_potentials, pbrs_shaping
from football_rl.recording import _LUNGE_ROLE_CODE, _STUN_ROLE_CODE
from football_rl.scalar_math import _fast_clip


class RewardShapingMixin:
    """Reward assembly of `FootballParallelEnv`: team rewards, ball and role potentials, shaping telemetry."""

    def _compute_ball_potential(
        self,
        team_id: int,
        ref_world_pos: Optional[np.ndarray] = None,
    ) -> float:
        """Compute strictly anti-symmetric territorial ball threat potential in [-0.5, +0.5] for `team_id`."""
        if ref_world_pos is None:
            ref_world_pos = (
                self.player_pos[self.ball_holder]
                if self.ball_holder is not None
                else self.ball_pos
            )
        c_ref = self._to_canonical_pos(ref_world_pos, team_id)
        cx, cy = float(c_ref[0]), float(c_ref[1])

        # Distance to opponent goal (1400, 420)
        d_att = math.hypot(1400.0 - cx, 420.0 - cy)
        lin_att = _fast_clip(1.0 - (d_att / self.max_pitch_diag), 0.0, 1.0)
        gauss_att = math.exp(-(d_att * d_att) / (2.0 * 380.0 * 380.0))
        phi_att = 0.65 * lin_att + 0.35 * gauss_att

        # Distance to own goal (0, 420)
        d_def = math.hypot(cx, 420.0 - cy)
        lin_def = _fast_clip(1.0 - (d_def / self.max_pitch_diag), 0.0, 1.0)
        gauss_def = math.exp(-(d_def * d_def) / (2.0 * 380.0 * 380.0))
        phi_def = 0.65 * lin_def + 0.35 * gauss_def

        return 0.5 * (phi_att - phi_def)

    # ==========================================================================
    # Reward, Event & Replay Serialization Helpers
    # ==========================================================================
    def _add_team_reward(
        self, step_rewards: np.ndarray, team_id: int, amount: float
    ) -> None:
        """Add a shared team reward to all 6 members of `team_id`."""
        if amount == 0.0:
            return
        mask = self.player_team == team_id
        step_rewards[mask] += amount

    def _compute_potential_shaping(
        self,
        step_rewards: np.ndarray,
        any_teleport: bool = False,
        r_sparse: Optional[np.ndarray] = None,
        r_ball_adv: Optional[np.ndarray] = None,
        *,
        pay: bool = True,
        terminal: bool = False,
    ) -> np.ndarray:
        """Evaluate the tactical potential field and apply potential-based shaping to ``step_rewards``.

        In ``"all"`` mode (with ``pay=True``), adds discounted PBRS ``gamma * Phi(s') - Phi(s)`` across
        every transition (teleports included) using ``cfg.ppo.gamma``, with ``Phi(terminal) = 0``.
        In ``"legacy"`` and ``"ball"`` modes, adds undiscounted ``Phi(s') - Phi(s)`` only when
        ``not any_teleport``. When ``pay=False`` (e.g. ``reset()``), initializes
        ``prev_tactical_potential`` without modifying ``step_rewards`` and returns zeros.
        """
        stunned = self.stun_timer > 0.0
        holder = self.ball_holder
        last_team = self.last_ball_team
        state = PotentialState(
            pos=self.player_pos,
            angle=self.player_angle,
            ball=self.ball_pos,
            holder=-1 if holder is None else holder,
            ball_in_flight=self.ball_kicked_in_flight,
            last_ball_team=-1 if last_team is None else last_team,
            team_side=self.team_side,
            disabled=stunned | self.is_ghost,
            stunned=stunned,
            post_kick_timer=self.post_kick_pbrs_timer,
            depth_bias=self._tactical_depth_bias,
            flank_bias=self._tactical_flank_bias,
            active_box_radius=self.active_box_radius,
        )
        result = compute_potentials(state, self.cfg.rewards, self.cfg.physics)
        phi_curr = result.phi
        self._phi_curr = phi_curr

        if not pay:
            pot_delta = np.zeros_like(phi_curr)
        elif self.cfg.rewards.shaping_mode == "all":
            pot_delta = pbrs_shaping(
                self.prev_tactical_potential,
                phi_curr,
                self.cfg.ppo.gamma,
                terminal,
            )
            step_rewards += pot_delta
        elif not any_teleport:
            pot_delta = phi_curr - self.prev_tactical_potential
            step_rewards += pot_delta
        else:
            pot_delta = np.zeros_like(phi_curr)

        if self.cfg.replay.enable_recording:
            self._record_potential_telemetry(
                state, result, any_teleport, r_sparse, r_ball_adv, pot_delta=pot_delta
            )

        self.prev_tactical_potential[:] = phi_curr
        return pot_delta

    def _record_potential_telemetry(
        self,
        state: PotentialState,
        result: PotentialResult,
        any_teleport: bool,
        r_sparse: Optional[np.ndarray],
        r_ball_adv: Optional[np.ndarray],
        pot_delta: Optional[np.ndarray] = None,
    ) -> None:
        """Store per-player role labels, role targets and the reward breakdown for replay frames."""
        n = self.num_players
        role_codes = np.asarray(result.role_code)
        label_codes = np.where(
            state.stunned, _STUN_ROLE_CODE, np.where(self.lunge_timer > 0.0, _LUNGE_ROLE_CODE, role_codes)
        )
        if pot_delta is None:
            pot_delta = result.phi - self.prev_tactical_potential if not any_teleport else np.zeros_like(result.phi)
        self._tele_breakdown = np.stack(
            [
                r_ball_adv if r_ball_adv is not None else np.zeros(n),
                pot_delta,
                result.phi_role,
                np.where(role_codes == ROLE_GK, 0.0, -result.crowd),
                r_sparse if r_sparse is not None else np.zeros(n),
            ]
        ).astype(np.float64)
        self._tele_role_codes = label_codes.astype(np.int64)
        self._tele_targets = np.array(result.target, dtype=np.float64)
