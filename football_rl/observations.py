"""Team-canonical observations for every agent and the centralized critic state."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from football_rl.config import ROLE_KEEPER, SIDE_LEFT, TEAM_A, TEAM_B
from football_rl.scalar_math import _fast_clip, wrap_to_pi


class ObservationMixin:
    """Observation builders of `FootballParallelEnv`, in team-canonical coordinates."""

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Convenience alias for _get_observations()."""
        return self._get_observations()

    # ==========================================================================
    # Team-Canonical Observation Builder (82 Floats, Vectorized Across 12 Agents)
    # ==========================================================================
    def _get_observations_array(self) -> np.ndarray:
        """Compute the (12, 82) normalized observation matrix for all 12 agents simultaneously."""
        phys = self.cfg.physics
        mech = self.cfg.mechanics
        vis = self.cfg.vision

        c_p_a, c_v_a, c_ang_a = self._to_team_canonical(
            self.player_pos, self.player_vel, self.player_angle, TEAM_A
        )
        c_b_a, c_bv_a, _ = self._to_team_canonical(
            self.ball_pos, self.ball_vel, 0.0, TEAM_A
        )
        c_p_b, c_v_b, c_ang_b = self._to_team_canonical(
            self.player_pos, self.player_vel, self.player_angle, TEAM_B
        )
        c_b_b, c_bv_b, _ = self._to_team_canonical(
            self.ball_pos, self.ball_vel, 0.0, TEAM_B
        )

        obs = np.zeros((self.num_players, vis.obs_dim), dtype=np.float32)

        # Own canonical state for each of the 12 agents (0..5 = Team A, 6..11 = Team B)
        sx, sy = self._obs_sx, self._obs_sy
        svx, svy = self._obs_svx, self._obs_svy
        s_ang = self._obs_s_ang

        sx[:6], sy[:6] = c_p_a[:6, 0], c_p_a[:6, 1]
        sx[6:], sy[6:] = c_p_b[6:, 0], c_p_b[6:, 1]
        svx[:6], svy[:6] = c_v_a[:6, 0], c_v_a[:6, 1]
        svx[6:], svy[6:] = c_v_b[6:, 0], c_v_b[6:, 1]
        s_ang[:6] = c_ang_a[:6]
        s_ang[6:] = c_ang_b[6:]

        # Part A: Self & Global State (indices 0..15)
        obs[:, 0] = (sx / phys.field_width) * 2.0 - 1.0
        obs[:, 1] = (sy / phys.field_height) * 2.0 - 1.0
        obs[:, 2] = np.sin(s_ang)
        obs[:, 3] = np.cos(s_ang)
        obs[:, 4] = svx / self._max_self_vel
        obs[:, 5] = svy / self._max_self_vel

        scores = self.stats.get("score", [0, 0])
        score_diff_a = (scores[0] - scores[1]) / 3.0
        obs[:6, 6] = score_diff_a
        obs[6:, 6] = -score_diff_a

        holder = self.ball_holder
        if holder is not None:
            obs[holder, 7] = 1.0
            holder_team = int(self.player_team[holder])
            max_hold = (
                mech.keeper_max_hold_time
                if self.player_role[holder] == ROLE_KEEPER
                else mech.field_max_hold_time
            )
            # Carrier: ball_hold_timer / 5.0 in [0, 1]
            obs[holder, 8] = min(1.0, max(0.0, self.ball_hold_timer / max_hold))

            # Teammates of carrier: -0.25
            tm_mask = (self.player_team == holder_team)
            tm_mask[holder] = False
            obs[tm_mask, 8] = -0.25

            # Opponents of carrier: continuous tackle readiness in [-1.0, -0.3]
            opp_mask = ~tm_mask
            opp_mask[holder] = False
            dx_h = self.player_pos[holder, 0] - self.player_pos[opp_mask, 0]
            dy_h = self.player_pos[holder, 1] - self.player_pos[opp_mask, 1]
            d_h = np.hypot(dx_h, dy_h)
            r_max = self._role_tackle_reach[opp_mask]
            dist_excess = np.clip((d_h - r_max) / (2.0 * r_max), 0.0, 1.0)

            can_tackle_carrier = (
                self._is_field_role(self.player_role[holder])
                and self.pickup_tackle_grace_timer <= 0.0
                and self.kickoff_protection_timer <= 0.0
            )
            if can_tackle_carrier:
                opp_ang = self.player_angle[opp_mask]
                u_fx = np.cos(opp_ang)
                u_fy = np.sin(opp_ang)
                facing_dot = np.where(
                    d_h < 1e-4,
                    1.0,
                    (u_fx * dx_h + u_fy * dy_h) / np.maximum(d_h, 1e-6),
                )
                tackleable_now = (d_h <= r_max) & (facing_dot >= 0.25)
                ang_deficit = np.clip((0.25 - facing_dot) / 1.25, 0.0, 1.0)
                not_ready_penalty = np.maximum(dist_excess, 0.3 + 0.7 * ang_deficit)
                obs[opp_mask, 8] = np.where(
                    tackleable_now,
                    -1.0,
                    -1.0 + 0.7 * np.where(facing_dot >= 0.25, dist_excess, not_ready_penalty),
                )
            else:
                obs[opp_mask, 8] = -0.65 + 0.35 * dist_excess
        else:
            obs[:, 8] = 0.0

        # Ability cooldown normalized to [0.0, 1.0]
        max_tackle_cd = max(mech.tackle_fail_cooldown, mech.field_violation_tackle_cooldown)
        obs[:, 9] = np.clip(self.tackle_cooldown / max_tackle_cd, 0.0, 1.0)
        keeper_cd_denom = max(1e-6, mech.keeper_lunge_cooldown)
        obs[0, 9] = min(1.0, max(0.0, self.lunge_cooldown[0] / keeper_cd_denom))
        obs[6, 9] = min(1.0, max(0.0, self.lunge_cooldown[6] / keeper_cd_denom))

        # Ghost or Stun timer normalized to [0.0, 1.0]
        obs[:, 10] = np.clip(
            np.maximum(self.ghost_timer, self.stun_timer) / mech.ghost_duration, 0.0, 1.0
        )

        # Relative angle and distance to Enemy Goal (1400.0, 420.0) in canonical space
        enemy_goal_dx = phys.field_width - sx
        enemy_goal_dy = (phys.field_height * 0.5) - sy
        goal_dist = np.hypot(enemy_goal_dx, enemy_goal_dy)
        rel_goal_angle = (
            np.arctan2(enemy_goal_dy, enemy_goal_dx) - s_ang + math.pi
        ) % (2.0 * math.pi) - math.pi
        obs[:, 11] = rel_goal_angle / math.pi
        obs[:, 12] = np.clip(goal_dist / self._max_field_dist, 0.0, 1.0)

        # Symmetry-Breaking Role Depth + Flank Channel Bias + Universal Active Box Radius in obs[:, 13:16]
        obs[:, 13] = self._tactical_depth_bias
        obs[:, 14] = self._tactical_flank_bias

        side_a = int(self.team_side[TEAM_A])
        side_b = int(self.team_side[TEAM_B])
        norm_box_a = _fast_clip(
            float(self.active_box_radius[side_a]) * self._inv_keeper_box_radius, 0.0, 1.0
        )
        norm_box_b = _fast_clip(
            float(self.active_box_radius[side_b]) * self._inv_keeper_box_radius, 0.0, 1.0
        )

        # Outfield players observe the minimum active box radius on the pitch (alerts when any box shrinks)
        min_active_box_norm = min(norm_box_a, norm_box_b)
        obs[:, 15] = min_active_box_norm

        # Goalkeepers observe own active box radius in [14] and own loose-ball timer in [15]
        obs[0, 14] = norm_box_a
        obs[0, 15] = _fast_clip(
            float(self._box_loose_ball_timer[side_a]) * self._inv_keeper_box_loose_max_time, 0.0, 1.0
        )
        obs[6, 14] = norm_box_b
        obs[6, 15] = _fast_clip(
            float(self._box_loose_ball_timer[side_b]) * self._inv_keeper_box_loose_max_time, 0.0, 1.0
        )

        # Part B: Vision Cone Entity Slots (12 Entities x 5 floats = 60 floats, indices 16..75)
        ent_pos = self._ent_pos
        ent_vel = self._ent_vel

        ent_pos[:6, 0] = c_b_a
        ent_vel[:6, 0] = c_bv_a
        ent_pos[:6, 1:] = c_p_a[self._entity_slot_matrix[:6]]
        ent_vel[:6, 1:] = c_v_a[self._entity_slot_matrix[:6]]

        ent_pos[6:, 0] = c_b_b
        ent_vel[6:, 0] = c_bv_b
        ent_pos[6:, 1:] = c_p_b[self._entity_slot_matrix[6:]]
        ent_vel[6:, 1:] = c_v_b[self._entity_slot_matrix[6:]]

        dx = ent_pos[:, :, 0] - sx[:, None]
        dy = ent_pos[:, :, 1] - sy[:, None]
        dist = np.hypot(dx, dy)
        ang_to_ent = np.arctan2(dy, dx)
        rel_ang = np.where(
            dist < 1e-6,
            0.0,
            (ang_to_ent - s_ang[:, None] + math.pi) % (2.0 * math.pi) - math.pi,
        )
        abs_rel_ang = np.abs(rel_ang)

        # Slot 0 (Ball): Stationary 360-degree FOV presence (no half-time clock corruption)
        in_ball_cone = (dist[:, 0] <= vis.max_range) & (abs_rel_ang[:, 0] <= vis.half_fov_rad)
        ball_vis = np.where(in_ball_cone, 1.0, vis.peripheral_radar_presence)

        # Slots 1..11 (Teammates & Opponents)
        dist_p = dist[:, 1:]
        abs_ang_p = abs_rel_ang[:, 1:]
        in_ent_cone = (dist_p <= vis.max_range) & (abs_ang_p <= self._rad_105)
        w_phi = np.where(
            abs_ang_p <= self._rad_90, 1.0, (self._rad_105 - abs_ang_p) * self._inv_fade_rad
        )
        w_d = 1.0 - 0.35 * (dist_p * self._inv_vis_max_range)
        fov_factor = np.where(in_ent_cone, w_phi * w_d, 0.0)

        base_presence = np.clip(0.25 + 0.65 * fov_factor, 0.25, 0.90)
        disabled_presence = -np.clip(0.25 + 0.65 * fov_factor, 0.25, 0.95)
        other_disabled = (np.maximum(self.ghost_timer, self.stun_timer) > 0.0)[
            self._entity_slot_matrix
        ]
        vis_p = np.where(other_disabled, disabled_presence, base_presence)
        if holder is not None:
            vis_p = np.where(self._entity_slot_matrix == holder, 1.0, vis_p)

        m0 = self._match_state_col
        ent_view = obs[:, vis.self_feature_dim : m0].reshape(
            self.num_players, vis.num_entity_slots, vis.entity_feature_dim
        )
        ent_view[:, 0, 0] = ball_vis
        ent_view[:, 1:, 0] = vis_p
        ent_view[:, :, 1] = np.clip(dist * self._inv_max_field_dist, 0.0, 1.0)
        ent_view[:, :, 2] = rel_ang * self._inv_pi
        ent_view[:, :, 3] = (ent_vel[:, :, 0] - svx[:, None]) * self._inv_max_rel_vel
        ent_view[:, :, 4] = (ent_vel[:, :, 1] - svy[:, None]) * self._inv_max_rel_vel

        # Part C: Match State (indices 76..81)
        # [76] possession owner: +1/-1 when held or kick in flight, +0.5/-0.5 when loose after touch, 0 if none
        in_flight = self.ball_kicked_in_flight
        if holder is not None:
            obs[:, m0] = self._possession_owner_rows[int(self.player_team[holder])]
        elif self.last_ball_team is not None:
            rows = self._possession_owner_rows if in_flight else self._loose_owner_rows
            obs[:, m0] = rows[int(self.last_ball_team)]
        # [77] ball in flight after a kick
        if in_flight:
            obs[:, m0 + 1] = 1.0
        # [78] fraction of the match remaining
        obs[:, m0 + 2] = max(0.0, 1.0 - self.rl_step_count * self._inv_max_rl_steps)
        # [79] second half
        if self.current_half == 2:
            obs[:, m0 + 3] = 1.0
        # [80] post-kick PBRS timer / 0.50 s (strictly mirrors PotentialState.post_kick_timer)
        pk_timer = self.post_kick_pbrs_timer
        if np.count_nonzero(pk_timer):
            for i in pk_timer.nonzero()[0]:
                obs[i, m0 + 4] = pk_timer[i] * self._inv_post_kick_window
        # [81] global turnover-reward cooldown remaining / turnover_reward_cooldown
        to_rem = self._turnover_cooldown - (self.match_time - self.last_turnover_reward_time)
        if to_rem > 0.0:
            obs[:, m0 + 5] = to_rem * self._inv_turnover_cooldown

        np.clip(obs, -1.0, 1.0, out=obs)
        return obs

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Compute the 82-float normalized observation dictionary for all 12 agents."""
        obs = self._get_observations_array()
        return {
            agent_name: obs[idx]
            for idx, agent_name in enumerate(self.possible_agents)
        }

    def _build_single_observation(
        self,
        idx: int,
        c_player_pos: Optional[np.ndarray] = None,
        c_player_vel: Optional[np.ndarray] = None,
        c_player_angle: Optional[np.ndarray | float] = None,
        c_ball_pos: Optional[np.ndarray] = None,
        c_ball_vel: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return the 82-float canonical observation for player `idx`."""
        return self._get_observations_array()[idx]

    def _to_canonical_pos(self, pos: np.ndarray, team_id: int) -> np.ndarray:
        """Transform world (x, y) coordinates into Team-Canonical (x, y)."""
        if self.team_side[team_id] == SIDE_LEFT:
            return pos.copy()
        canon_pos = np.empty_like(pos)
        canon_pos[..., 0] = self.cfg.physics.field_width - pos[..., 0]
        canon_pos[..., 1] = self.cfg.physics.field_height - pos[..., 1]
        return canon_pos

    def _to_team_canonical(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        angle: np.ndarray | float,
        team_id: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | float]:
        """Transform world coordinates into Team-Canonical Coordinates."""
        if self.team_side[team_id] == SIDE_LEFT:
            return pos.copy(), vel.copy(), angle

        canon_pos = np.empty_like(pos)
        canon_pos[..., 0] = self.cfg.physics.field_width - pos[..., 0]
        canon_pos[..., 1] = self.cfg.physics.field_height - pos[..., 1]
        canon_vel = -vel
        canon_angle = wrap_to_pi(angle + math.pi)
        return canon_pos, canon_vel, canon_angle

    def get_centralized_state(self) -> np.ndarray:
        """Compute the (12, 54) ego-consistent, team-canonical centralized state matrix for the CTDE Critic."""
        central_state = np.empty((self.num_players, 54), dtype=np.float32)

        c_p_a, c_v_a, _ = self._to_team_canonical(self.player_pos, self.player_vel, 0.0, TEAM_A)
        c_b_a, c_bv_a, _ = self._to_team_canonical(self.ball_pos, self.ball_vel, 0.0, TEAM_A)

        c_p_b, c_v_b, _ = self._to_team_canonical(self.player_pos, self.player_vel, 0.0, TEAM_B)
        c_b_b, c_bv_b, _ = self._to_team_canonical(self.ball_pos, self.ball_vel, 0.0, TEAM_B)

        inv_w2 = self._inv_field_half_w
        inv_h2 = self._inv_field_half_h
        inv_v = self._inv_critic_vel_norm

        feat_a = self._critic_feat_a
        feat_a[:, 0] = c_p_a[:, 0] * inv_w2 - 1.0
        feat_a[:, 1] = c_p_a[:, 1] * inv_h2 - 1.0
        np.clip(c_v_a[:, 0] * inv_v, -1.0, 1.0, out=feat_a[:, 2])
        np.clip(c_v_a[:, 1] * inv_v, -1.0, 1.0, out=feat_a[:, 3])

        feat_b = self._critic_feat_b
        feat_b[:, 0] = c_p_b[:, 0] * inv_w2 - 1.0
        feat_b[:, 1] = c_p_b[:, 1] * inv_h2 - 1.0
        np.clip(c_v_b[:, 0] * inv_v, -1.0, 1.0, out=feat_b[:, 2])
        np.clip(c_v_b[:, 1] * inv_v, -1.0, 1.0, out=feat_b[:, 3])

        side_a = int(self.team_side[TEAM_A])
        side_b = int(self.team_side[TEAM_B])
        box_a_norm = float(self.active_box_radius[side_a]) * self._inv_keeper_box_radius
        box_b_norm = float(self.active_box_radius[side_b]) * self._inv_keeper_box_radius

        inv_launch = self._inv_ball_max_launch_speed
        ctx_a = self._critic_ctx_a
        ctx_a[0] = c_b_a[0] * inv_w2 - 1.0
        ctx_a[1] = c_b_a[1] * inv_h2 - 1.0
        ctx_a[2] = c_bv_a[0] * inv_launch
        ctx_a[3] = c_bv_a[1] * inv_launch
        ctx_a[4] = box_a_norm
        ctx_a[5] = box_b_norm

        ctx_b = self._critic_ctx_b
        ctx_b[0] = c_b_b[0] * inv_w2 - 1.0
        ctx_b[1] = c_b_b[1] * inv_h2 - 1.0
        ctx_b[2] = c_bv_b[0] * inv_launch
        ctx_b[3] = c_bv_b[1] * inv_launch
        ctx_b[4] = box_b_norm
        ctx_b[5] = box_a_norm

        # Populate Ego-First Own Team (24 floats) + Canonical Opponent Team (24 floats) + Context (6 floats)
        central_state[:6, :24] = feat_a[self._critic_own_order_a].reshape(6, 24)
        central_state[:6, 24:48] = feat_a[self._critic_opp_order_a].reshape(6, 24)
        central_state[:6, 48:54] = ctx_a

        central_state[6:, :24] = feat_b[self._critic_own_order_b].reshape(6, 24)
        central_state[6:, 24:48] = feat_b[self._critic_opp_order_b].reshape(6, 24)
        central_state[6:, 48:54] = ctx_b

        np.clip(central_state, -1.0, 1.0, out=central_state)
        return central_state
