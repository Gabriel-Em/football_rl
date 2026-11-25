"""Ball contests: kicks and their resolution, tackles, keeper strips and free-ball pickups."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from football_rl.config import ROLE_ATK, ROLE_DEF, ROLE_KEEPER, ROLE_MID, SIDE_LEFT, SIDE_RIGHT
from football_rl.scalar_math import _fast_clip, wrap_to_pi


class BallPlayMixin:
    """Ball contests of `FootballParallelEnv`: kicks, tackles, keeper strips and free-ball pickups."""

    def _is_ghost(self, idx: int) -> bool:
        """Return True if player idx is non-collidable (ghost mode or immobile post-lunge/victim stun)."""
        return bool(
            (self.ghost_timer[idx] > 0.0)
            or ((self.stun_timer[idx] > 0.0) and (self.lunge_timer[idx] <= 0.0))
        )

    def _get_role_kick_params(self, idx: int) -> Tuple[float, float]:
        role = self.player_role[idx]
        phys = self.cfg.physics
        if role == ROLE_KEEPER:
            return (phys.keeper_max_kick_speed, 0.0)
        elif role == ROLE_DEF:
            return (phys.def_max_kick_speed, phys.def_shot_jitter_rad)
        elif role == ROLE_MID:
            return (phys.mid_max_kick_speed, phys.mid_shot_jitter_rad)
        else:
            return (phys.atk_max_kick_speed, phys.atk_shot_jitter_rad)

    def _get_role_tackle_params(self, idx: int) -> Tuple[float, float]:
        role = self.player_role[idx]
        mech = self.cfg.mechanics
        if role == ROLE_DEF:
            return (mech.tackle_radius_def, mech.tackle_prob_def)
        elif role == ROLE_MID:
            return (mech.tackle_radius_mid, mech.tackle_prob_mid)
        elif role == ROLE_ATK:
            return (mech.tackle_radius_atk, mech.tackle_prob_atk)
        else:
            return (mech.tackle_radius_mid, mech.tackle_prob_mid)

    @staticmethod
    def _point_to_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ax, ay = float(a[0]), float(a[1])
        abx = float(b[0]) - ax
        aby = float(b[1]) - ay
        px, py = float(p[0]), float(p[1])
        d2 = abx * abx + aby * aby
        if d2 < 1e-12:
            return math.hypot(px - ax, py - ay)
        t = _fast_clip(((px - ax) * abx + (py - ay) * aby) / d2, 0.0, 1.0)
        return math.hypot(px - (ax + t * abx), py - (ay + t * aby))

    def _is_ball_heading_into_goal(
        self,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        target_side: int,
        y_margin: float = 0.0,
    ) -> bool:
        phys = self.cfg.physics
        goal_x = 0.0 if target_side == SIDE_LEFT else phys.field_width
        vx, vy = float(ball_vel[0]), float(ball_vel[1])
        if target_side == SIDE_LEFT and vx > -80.0:
            return False
        if target_side == SIDE_RIGHT and vx < 80.0:
            return False
        t_hit = (goal_x - float(ball_pos[0])) / vx
        if t_hit <= 0.0 or t_hit > 3.0:
            return False
        y_intersect = float(ball_pos[1]) + vy * t_hit
        dist_to_goal_line = math.hypot(goal_x - float(ball_pos[0]), y_intersect - float(ball_pos[1]))
        max_reach = (vx * vx + vy * vy) / (2.0 * phys.ball_friction_decel) + 35.0
        return bool(
            dist_to_goal_line <= max_reach
            and (phys.goal_y_min - y_margin) <= y_intersect <= (phys.goal_y_max + y_margin)
        )

    # ==========================================================================
    # Mechanics Helpers: Kicks, Tackles, Violations, Pickups, Collisions
    # ==========================================================================
    def _resolve_pending_kick_as_other(self) -> None:
        """Count the unresolved kick, if any, as neither received by a teammate nor won by an opponent."""
        if self._pending_kick_idx >= 0:
            self.stats["kick_other"] += 1
            self._pending_kick_idx = -1

    def _resolve_pending_kick(self, receiver_idx: int) -> None:
        """Attribute the unresolved kick to its first receiver: teammate, opponent, or the kicker again."""
        kicker_idx = self._pending_kick_idx
        if kicker_idx < 0:
            return
        self._pending_kick_idx = -1
        if receiver_idx == kicker_idx:
            self.stats["kick_other"] += 1
        elif self.player_team[receiver_idx] == self.player_team[kicker_idx]:
            self.stats["kick_to_teammate"] += 1
        else:
            self.stats["kick_to_opponent"] += 1

    def _execute_kick(
        self,
        kicker_idx: int,
        kick_power: float,
        step_rewards: Optional[np.ndarray] = None,
    ) -> None:
        """Launch the ball from kicker, record release state, check shot-on-target, and apply pickup cooldown."""
        self._resolve_pending_kick_as_other()
        self.stats["kicks"] += 1
        self._pending_kick_idx = int(kicker_idx)
        self.kickoff_protection_timer = 0.0
        self.pickup_tackle_grace_timer = 0.0
        phys = self.cfg.physics
        mech = self.cfg.mechanics
        rew = self.cfg.rewards

        kicker_role = int(self.player_role[kicker_idx])
        kicker_team = int(self.player_team[kicker_idx])

        power_norm = _fast_clip(
            (kick_power - mech.kick_trigger_threshold)
            / max(1e-6, 1.0 - mech.kick_trigger_threshold),
            0.0,
            1.0,
        )
        max_kick_spd, jitter_rad = self._get_role_kick_params(kicker_idx)
        launch_speed = phys.ball_min_kick_speed + power_norm * (max_kick_spd - phys.ball_min_kick_speed)

        max_defl_rad = math.radians(mech.kick_steer_max_deflection_deg)
        steer_defl = float(self.raw_steer_input[kicker_idx]) * max_defl_rad
        base_angle = float(wrap_to_pi(self.player_angle[kicker_idx] + steer_defl))

        if power_norm > 0.65 and jitter_rad > 0.0:
            kick_angle = base_angle + float(self.np_random.uniform(-jitter_rad, jitter_rad))
        else:
            kick_angle = base_angle

        k_angle = float(kick_angle)
        k_dx = math.cos(k_angle)
        k_dy = math.sin(k_angle)
        if kicker_role == ROLE_KEEPER:
            attack_side = int(self.team_side[kicker_team]) if hasattr(self, "team_side") else kicker_team
            attack_dir = 1.0 if attack_side == SIDE_LEFT else -1.0
            # Never allow a goalkeeper to kick backward toward their own goal line
            if k_dx * attack_dir < 0.25:
                k_dx = 0.25 * attack_dir
                dy_sign = 1.0 if k_dy >= 0.0 else -1.0
                k_dy = dy_sign * math.sqrt(max(0.0, 1.0 - k_dx * k_dx))
                k_angle = math.atan2(k_dy, k_dx)
                self.player_angle[kicker_idx] = k_angle
                base_angle = k_angle

        facing_vec = np.array([math.cos(base_angle), math.sin(base_angle)], dtype=np.float64)
        kick_vec = np.array([k_dx, k_dy], dtype=np.float64)

        self.ball_pos = self.player_pos[kicker_idx] + facing_vec * mech.ball_carry_offset
        self.ball_vel = kick_vec * launch_speed

        opp_mask = self.player_team != kicker_team
        opp_dists = np.hypot(
            self.player_pos[opp_mask, 0] - self.ball_pos[0],
            self.player_pos[opp_mask, 1] - self.ball_pos[1],
        )
        self.ball_kick_orig_opp_dist = float(np.min(opp_dists)) if len(opp_dists) > 0 else 999.0

        self.ball_holder = None
        self.ball_hold_timer = 0.0
        self.last_ball_holder = kicker_idx
        self.last_ball_team = kicker_team
        self.ball_kick_origin = self.ball_pos.copy()
        self.ball_kick_start_pos = self.ball_pos.copy()
        self.ball_kicked_in_flight = True
        self.ball_kick_flight_timer = 0.0
        self.ball_wall_bounces = 0
        self.is_wall_pass_in_flight = False

        self.post_kick_pbrs_timer[kicker_idx] = 0.50

        self.pickup_cooldown[kicker_idx] = max(
            self.pickup_cooldown[kicker_idx], mech.kick_pickup_cooldown
        )
        self.kick_cooldown_active[kicker_idx] = True

        now_t = self.match_time
        self.ball_kick_time = now_t
        self.last_kick_time[kicker_idx] = now_t
        if kicker_role == ROLE_KEEPER:
            self.last_keeper_kick_time[kicker_team] = now_t

        # Require minimum launch speed (240 u/s) to qualify as a true Shot on Target
        min_shot_spd = getattr(mech, "min_shot_speed", 240.0)
        enemy_side = 1 - int(self.team_side[kicker_team])
        is_true_sot = (
            launch_speed >= min_shot_spd
            and self._is_ball_heading_into_goal(self.ball_pos, self.ball_vel, enemy_side, y_margin=0.0)
        )
        cx = 0.0
        cy = 420.0
        flank_angle_mult = 1.0
        if is_true_sot:
            c_kicker_pos = self._to_canonical_pos(self.player_pos[kicker_idx], kicker_team)
            cx = float(c_kicker_pos[0])
            cy = float(c_kicker_pos[1])
            if cx >= 1210.0 and abs(cy - 420.0) >= 135.0:
                flank_angle_mult = 0.35
            if cx < 680.0:
                is_true_sot = False
            else:
                # Disqualify if aimed at an intermediate teammate in the pass reception corridor
                ball_spd = float(np.linalg.norm(self.ball_vel))
                if ball_spd > 1e-6:
                    u_kick = self.ball_vel / ball_spd
                    goal_x = 0.0 if enemy_side == SIDE_LEFT else phys.field_width
                    dist_to_goal = abs(goal_x - float(self.ball_pos[0]))
                    tau_lead = 0.35
                    for tm_idx in range(self.num_players):
                        if tm_idx != kicker_idx and self.player_team[tm_idx] == kicker_team:
                            raw_lead = self.player_pos[tm_idx] + tau_lead * self.player_vel[tm_idx]
                            for test_pt in (self.player_pos[tm_idx], raw_lead):
                                to_tm = test_pt - self.ball_pos
                                proj_along = float(np.dot(to_tm, u_kick))
                                if 30.0 <= proj_along <= min(500.0, dist_to_goal - 70.0):
                                    perp_vec = to_tm - proj_along * u_kick
                                    dist_perp = float(np.linalg.norm(perp_vec))
                                    d_tm = float(np.linalg.norm(to_tm))
                                    cos_angle = proj_along / max(1e-6, d_tm)
                                    if dist_perp <= 45.0 or cos_angle >= 0.95:
                                        is_true_sot = False
                                        break
                            if not is_true_sot:
                                break

        self.ball_kick_was_shot_on_target = is_true_sot
        self.last_shot_sot_reward = 0.0
        self.last_shot_team_reward = 0.0

        if is_true_sot:
            self.stats["shots"][kicker_team] += 1
            if step_rewards is not None and self.shot_on_target_cd[kicker_idx] <= 0.0:
                d_shot = math.hypot(1400.0 - cx, 420.0 - cy)
                mu_dist = math.exp(-((max(0.0, d_shot - 220.0) ** 2) / (2.0 * (210.0 ** 2))))
                opp_gk_idx = 0 if enemy_side == SIDE_LEFT else 6
                to_gk = self.player_pos[opp_gk_idx] - self.ball_pos
                b_spd = max(1e-6, float(np.linalg.norm(self.ball_vel)))
                u_kick = self.ball_vel / b_spd
                proj_gk = float(np.dot(to_gk, u_kick))
                if proj_gk > 0.0:
                    d_gk_ray = math.hypot(to_gk[0] - proj_gk * u_kick[0], to_gk[1] - proj_gk * u_kick[1])
                else:
                    d_gk_ray = float(np.linalg.norm(to_gk))
                mu_gk_open = _fast_clip(d_gk_ray / 45.0, 0.15, 1.0)
                power_factor = 0.6 + 0.4 * power_norm
                mu_xg = _fast_clip(mu_dist * mu_gk_open * flank_angle_mult * power_factor, 0.05, 1.0)

                sot_rew = rew.shot_on_target_ind * mu_xg
                team_sot_rew = rew.shot_on_target_team * mu_xg
                step_rewards[kicker_idx] += sot_rew
                self._add_team_reward(step_rewards, kicker_team, team_sot_rew)
                self.last_shot_sot_reward = float(sot_rew)
                self.last_shot_team_reward = float(team_sot_rew)
                self.shot_on_target_cd[kicker_idx] = 1.5
                self._emit_event(
                    "SHOT ON TARGET!",
                    self.player_pos[kicker_idx],
                    team=kicker_team,
                    agent=self.idx_to_agent[kicker_idx],
                    reward=sot_rew + team_sot_rew,
                )

    def _attempt_tackle(self, tackler_idx: int, step_rewards: np.ndarray) -> None:
        """Execute kinematic lunge and dynamic distance-based tackle check when action[3] > 0.5."""
        if self.kickoff_protection_timer > 0.0:
            return

        mech = self.cfg.mechanics
        rew = self.cfg.rewards
        tackler_team = int(self.player_team[tackler_idx])
        holder = self.ball_holder

        # Ignore tackle when own team holds the ball
        if holder is not None and int(self.player_team[holder]) == tackler_team:
            return

        # Fix 4.2: If ball is loose (holder is None), tackling is a harmless no-op.
        # Tackles apply strictly against an active opponent ball carrier.
        if holder is None:
            return

        # Gate tackle execution with tackle_whiff_max_ball_dist (180u)
        max_ball_dist = getattr(mech, "tackle_whiff_max_ball_dist", getattr(rew, "tackle_whiff_max_ball_dist", 180.0))
        dist_to_ball = math.hypot(
            float(self.player_pos[tackler_idx, 0] - self.ball_pos[0]),
            float(self.player_pos[tackler_idx, 1] - self.ball_pos[1]),
        )
        if holder is not None:
            dist_to_carrier = math.hypot(
                float(self.player_pos[tackler_idx, 0] - self.player_pos[holder, 0]),
                float(self.player_pos[tackler_idx, 1] - self.player_pos[holder, 1]),
            )
            dist_to_ball = min(dist_to_ball, dist_to_carrier)
        if dist_to_ball > max_ball_dist:
            return

        # Always execute physical forward lunge along player_angle
        t_angle = float(self.player_angle[tackler_idx])
        u_facing = np.array([math.cos(t_angle), math.sin(t_angle)], dtype=np.float64)
        self.player_vel[tackler_idx] = u_facing * mech.field_lunge_speed
        self.lunge_timer[tackler_idx] = mech.field_lunge_duration
        self.tackle_cooldown[tackler_idx] = max(
            self.tackle_cooldown[tackler_idx], mech.field_lunge_duration + mech.tackle_fail_cooldown
        )
        self.is_ghost[tackler_idx] = False

        R_max_lunge, P_max = self._get_role_tackle_params(tackler_idx)

        valid_target = (
            holder is not None
            and int(self.player_team[holder]) != tackler_team
            and self._is_field_role(self.player_role[holder])
            and self.pickup_tackle_grace_timer <= 0.0
        )

        if valid_target:
            to_carrier = self.player_pos[holder] - self.player_pos[tackler_idx]
            dist = math.hypot(float(to_carrier[0]), float(to_carrier[1]))
            if dist < 1e-4:
                facing_dot = 1.0
            else:
                facing_dot = (
                    u_facing[0] * float(to_carrier[0]) + u_facing[1] * float(to_carrier[1])
                ) / dist

            contact_clearance = max(0.0, dist - (2.0 * self.cfg.physics.player_radius - 8.0))
            effective_range = R_max_lunge
            if contact_clearance <= effective_range and facing_dot >= 0.25:
                p_success = max(
                    0.0,
                    min(1.0, float(P_max * (1.0 - (contact_clearance / effective_range) ** 2))),
                )
                if self.np_random.random() < p_success:
                    old_carrier = holder
                    old_team = int(self.player_team[old_carrier])
                    self._reset_pass_chains(old_team)
                    self.ball_holder = tackler_idx
                    self.ball_hold_timer = 0.0
                    self.kick_grace_timer = mech.pickup_kick_grace
                    self.last_ball_holder = tackler_idx
                    self.last_ball_team = tackler_team
                    self.ball_kicked_in_flight = False
                    self.ball_kick_orig_opp_dist = -1.0

                    self._on_new_player_touched_ball(tackler_idx)

                    self.stun_timer[old_carrier] = mech.tackle_victim_stun_time
                    self.player_vel[old_carrier] = 0.0
                    self.is_ghost[old_carrier] = True
                    self.tackle_cooldown[old_carrier] = max(
                        self.tackle_cooldown[old_carrier], mech.tackled_opponent_cooldown
                    )
                    self.pickup_cooldown[old_carrier] = max(
                        self.pickup_cooldown[old_carrier], mech.tackle_victim_stun_time
                    )

                    self.ball_pos = (
                        self.player_pos[tackler_idx] + u_facing * mech.ball_carry_offset
                    )
                    self.ball_vel[:] = self.player_vel[tackler_idx]

                    now_t = self.match_time
                    to_cd = getattr(mech, "turnover_reward_cooldown", 1.0)
                    if step_rewards is not None and (now_t - self.last_turnover_reward_time) >= to_cd:
                        self.last_turnover_reward_time = now_t
                        win_ind = rew.successful_tackle_tackler_ind
                        win_team = rew.successful_tackle_team
                        # Strict zero-sum: losing carrier and team mirror winning tackle rewards
                        step_rewards[tackler_idx] += win_ind
                        step_rewards[old_carrier] -= win_ind
                        self._add_team_reward(step_rewards, tackler_team, win_team)
                        self._add_team_reward(step_rewards, old_team, -win_team)
                    self.spell_max_threat[tackler_team] = self._compute_ball_potential(tackler_team, self.ball_pos)
                    self.stats["successful_tackles"] += 1
                    self.stats["team_tackles"][tackler_team] += 1
                    self._emit_event(
                        "TACKLE!",
                        self.player_pos[tackler_idx],
                        team=tackler_team,
                        agent=self.idx_to_agent[tackler_idx],
                        reward=rew.successful_tackle_tackler_ind + rew.successful_tackle_team,
                    )
                    return

        # Uniform whiff cost and post-lunge recovery stun
        self.stun_timer[tackler_idx] = mech.tackle_miss_stun_time
        self.is_ghost[tackler_idx] = self._is_ghost(tackler_idx)
        if step_rewards is not None:
            step_rewards[tackler_idx] += rew.failed_tackle_ind
        self.stats["failed_tackles"] += 1
        self._emit_event(
            "TACKLE FAILED!",
            self.player_pos[tackler_idx],
            team=tackler_team,
            agent=self.idx_to_agent[tackler_idx],
            reward=rew.failed_tackle_ind,
        )

    def _check_keeper_perimeter_strip(
        self, carrier_idx: int, step_rewards: np.ndarray
    ) -> None:
        """Keeper strips ball if opponent field player holds it within pickup distance."""
        mech = self.cfg.mechanics
        rew = self.cfg.rewards

        carrier_team = int(self.player_team[carrier_idx])
        opp_team = 1 - carrier_team
        keeper_idx = self._team_keeper_idx(opp_team)

        if self.stun_timer[keeper_idx] > 0.0 or self.pickup_cooldown[keeper_idx] > 0.0:
            return

        dist_to_ball = float(np.linalg.norm(self.player_pos[keeper_idx] - self.ball_pos))
        if dist_to_ball <= mech.pickup_distance:
            d_ball = self.ball_pos - self.player_pos[keeper_idx]
            d_norm = max(1e-6, dist_to_ball)
            u_ball_x = float(d_ball[0]) / d_norm
            u_ball_y = float(d_ball[1]) / d_norm
            gk_ang = float(self.player_angle[keeper_idx])
            cos_phi = math.cos(gk_ang) * u_ball_x + math.sin(gk_ang) * u_ball_y
            if cos_phi < -0.15:
                return

            now_t = self.physics_frame_count * self.cfg.physics.dt
            recently_kicked_by_gk = (now_t - self.last_keeper_kick_time[opp_team]) < mech.keeper_strip_self_kick_cooldown

            self.ball_holder = keeper_idx
            self.ball_hold_timer = 0.0
            self.last_ball_holder = keeper_idx
            self.last_ball_team = opp_team
            self.ball_kicked_in_flight = False
            self.ball_kick_orig_opp_dist = -1.0
            self._reset_pass_chains(carrier_team)

            # Stun the stripped field player for 0.2s
            self.stun_timer[carrier_idx] = mech.keeper_strip_stun_duration
            self.player_vel[carrier_idx] = 0.0
            self.pickup_cooldown[carrier_idx] = max(
                self.pickup_cooldown[carrier_idx], mech.keeper_strip_stun_duration
            )
            self.tackle_cooldown[carrier_idx] = max(
                self.tackle_cooldown[carrier_idx], mech.tackled_opponent_cooldown
            )

            self._on_new_player_touched_ball(keeper_idx)

            facing_vec = np.array(
                [
                    math.cos(self.player_angle[keeper_idx]),
                    math.sin(self.player_angle[keeper_idx]),
                ],
                dtype=np.float64,
            )
            self.ball_pos = self.player_pos[keeper_idx] + facing_vec * mech.ball_carry_offset
            self.ball_vel[:] = self.player_vel[keeper_idx]

            if not recently_kicked_by_gk:
                step_rewards[keeper_idx] += rew.successful_tackle_tackler_ind
                step_rewards[carrier_idx] += rew.successful_tackle_carrier_ind
                self._add_team_reward(step_rewards, opp_team, rew.successful_tackle_team)
                self._add_team_reward(step_rewards, carrier_team, rew.intercepted_team)
                self._emit_event(
                    "GK STRIP!",
                    self.player_pos[keeper_idx],
                    team=opp_team,
                    agent=self.idx_to_agent[keeper_idx],
                    reward=rew.successful_tackle_tackler_ind + rew.successful_tackle_team,
                )

    def _check_free_ball_pickup(self, step_rewards: np.ndarray) -> None:
        """Check if any eligible player picks up the free ball and award Markovian pass/transition rewards."""
        if self.ball_holder is not None:
            return

        mech = self.cfg.mechanics
        rew = self.cfg.rewards
        phys = self.cfg.physics

        dx_b = self.ball_pos[0] - self.player_pos[:, 0]
        dy_b = self.ball_pos[1] - self.player_pos[:, 1]
        dists = np.hypot(dx_b, dy_b)
        if not np.any(dists <= mech.pickup_distance):
            return

        eligible = np.where(
            (dists <= mech.pickup_distance)
            & (self.pickup_cooldown <= 0.0)
            & (self.stun_timer <= 0.0)
            & (self.ghost_timer <= 0.0)
        )[0]

        if len(eligible) == 0:
            return

        new_holder = int(eligible[np.argmin(dists[eligible])])
        new_team = int(self.player_team[new_holder])
        new_role = int(self.player_role[new_holder])

        self._resolve_pending_kick(new_holder)
        if new_role == ROLE_KEEPER:
            self.stats["gk_holds"] += 1
        self._on_new_player_touched_ball(new_holder)

        is_teammate_pass = False
        d_dest_goal = 999.0
        max_flight = (
            mech.max_wall_pass_flight_time
            if self.is_wall_pass_in_flight
            else mech.max_pass_flight_time
        )
        if (
            self.ball_kicked_in_flight
            and self.ball_kick_flight_timer <= max_flight
            and self.last_ball_holder is not None
        ):
            kicker_idx = self.last_ball_holder
            kicker_team = int(self.last_ball_team) if self.last_ball_team is not None else new_team
            kick_travel_dist = math.hypot(
                float(self.ball_pos[0] - self.ball_kick_origin[0]),
                float(self.ball_pos[1] - self.ball_kick_origin[1]),
            )
            is_wall_pass = bool(self.is_wall_pass_in_flight and self.ball_wall_bounces == 1)

            if new_team == kicker_team and new_holder != kicker_idx:
                is_teammate_pass = True
                c_dest = self._to_canonical_pos(self.ball_pos, new_team)
                d_dest_goal = math.hypot(1400.0 - float(c_dest[0]), 420.0 - float(c_dest[1]))

                # Pair-oscillation detection (blocks A <-> B ping-pong farming)
                canonical_pair = (min(kicker_idx, new_holder), max(kicker_idx, new_holder))
                if self.last_pass_pair[new_team] == canonical_pair:
                    self.pair_pass_count[new_team] += 1
                else:
                    self.last_pass_pair[new_team] = canonical_pair
                    self.pair_pass_count[new_team] = 1

                # Allow shorter passes (55u) only on the first pass when escaping heavy pressure (d_opp_orig <= 70u)
                is_first_pair_pass = (self.pair_pass_count[new_team] == 1)
                eff_min_dist = (
                    mech.min_wall_pass_distance
                    if is_wall_pass
                    else (55.0 if (is_first_pair_pass and self.ball_kick_orig_opp_dist <= 70.0) else mech.min_pass_distance)
                )

                if kick_travel_dist >= eff_min_dist:
                    # 1. Territorial Threat Gain above spell high-water mark (prevents backward-forward pass pumping)
                    phi_orig = self._compute_ball_potential(new_team, self.ball_kick_origin)
                    phi_dest = self._compute_ball_potential(new_team, self.ball_pos)
                    high_water = max(
                        float(phi_orig),
                        float(self._compute_ball_potential(new_team, self.player_pos[kicker_idx])),
                        float(self.spell_max_threat[new_team]),
                    )
                    d_phi_threat = max(0.0, float(phi_dest) - high_water)
                    self.spell_max_threat[new_team] = max(high_water, float(phi_dest))

                    # 2. Pressure Relief / Space Gain (gated to zero on repeat A <-> B pair ping-pong)
                    opp_slice = slice(6, 12) if new_team == 0 else slice(0, 6)
                    if self.ball_kick_orig_opp_dist >= 0.0:
                        d_opp_orig = self.ball_kick_orig_opp_dist
                    else:
                        orig_opp_dists = np.hypot(
                            self.player_pos[opp_slice, 0] - self.ball_kick_origin[0],
                            self.player_pos[opp_slice, 1] - self.ball_kick_origin[1],
                        )
                        d_opp_orig = float(np.min(orig_opp_dists)) if len(orig_opp_dists) > 0 else 999.0
                    dest_opp_dists = np.hypot(
                        self.player_pos[opp_slice, 0] - float(self.ball_pos[0]),
                        self.player_pos[opp_slice, 1] - float(self.ball_pos[1]),
                    )
                    d_opp_dest = float(np.min(dest_opp_dists)) if len(dest_opp_dists) > 0 else 999.0

                    sigma_orig = _fast_clip((d_opp_orig - 45.0) / 135.0, 0.0, 1.0)
                    sigma_dest = _fast_clip((d_opp_dest - 45.0) / 135.0, 0.0, 1.0)
                    d_sigma_press = (sigma_dest - sigma_orig) if is_first_pair_pass else 0.0

                    # Pair oscillation decay: 1.0 on 1st pass between pair, 0.0 on immediate ping-pong return
                    osc_mult = 1.0 if is_first_pair_pass else 0.0

                    kick_start = self.ball_kick_start_pos if np.any(self.ball_kick_start_pos != 0.0) else self.ball_kick_origin
                    pass_dist = math.hypot(
                        float(self.ball_pos[0] - kick_start[0]),
                        float(self.ball_pos[1] - kick_start[1]),
                    )
                    current_chain = self.team_pass_chain[new_team]
                    chain_mult = 1.0 if current_chain < 3 else (0.5 if current_chain == 3 else 0.0)

                    dy_pass = abs(float(self.ball_pos[1] - kick_start[1]))
                    kicker_role = int(self.player_role[kicker_idx])
                    is_cb_lateral = (kicker_role == ROLE_DEF and new_role == ROLE_DEF)
                    r_circulation = (
                        0.15
                        if (
                            current_chain == 0
                            and pass_dist >= 110.0
                            and dy_pass >= 80.0
                            and sigma_dest >= 0.75
                            and not is_wall_pass
                            and not is_cb_lateral
                        )
                        else 0.0
                    )
                    r_pass = osc_mult * chain_mult * max(
                        rew.pass_threat_coeff * d_phi_threat + rew.pass_press_escape_coeff * d_sigma_press,
                        r_circulation,
                    )
                    qualifies_bonus = osc_mult > 0.0 and (
                        (d_phi_threat > 1e-4) or (d_sigma_press > 1e-4) or (r_circulation > 0.0)
                    )

                    # 3. In-Stride Through-Ball Bonus (R_stride)
                    r_stride = 0.0
                    ball_speed = math.hypot(float(self.ball_vel[0]), float(self.ball_vel[1]))
                    if ball_speed == 0.0 and self.ball_kicked_in_flight:
                        ball_speed = 100.0
                    attack_sign = 1.0 if self.team_side[new_team] == SIDE_LEFT else -1.0
                    dx_forward = (float(self.ball_pos[0]) - float(kick_start[0])) * attack_sign

                    if (
                        not is_wall_pass
                        and ball_speed >= 45.0
                        and pass_dist >= 100.0
                        and dx_forward >= 50.0
                        and d_phi_threat >= 0.04
                        and current_chain < 4
                    ):
                        e_goal_x = float(self.right_goal_center[0] if self.team_side[new_team] == SIDE_LEFT else self.left_goal_center[0])
                        e_goal_y = float(self.right_goal_center[1] if self.team_side[new_team] == SIDE_LEFT else self.left_goal_center[1])
                        to_gx = e_goal_x - float(self.player_pos[new_holder, 0])
                        to_gy = e_goal_y - float(self.player_pos[new_holder, 1])
                        to_g_norm = math.hypot(to_gx, to_gy)
                        if to_g_norm > 1e-5:
                            u_gx = to_gx / to_g_norm
                            u_gy = to_gy / to_g_norm
                            v_goal = (
                                float(self.player_vel[new_holder, 0]) * u_gx
                                + float(self.player_vel[new_holder, 1]) * u_gy
                            )
                            if v_goal >= 60.0:
                                v_max = max(phys.player_max_speed, 1e-5)
                                stride_scale = min(1.0, max(0.0, (v_goal - 60.0) / (v_max - 60.0)))
                                threat_scale = min(1.0, d_phi_threat / 0.12)
                                r_stride = rew.through_ball_stride_bonus * stride_scale * threat_scale * chain_mult
                                step_rewards[new_holder] += r_stride
                                step_rewards[kicker_idx] += r_stride
                                self.stats["through_balls"] += 1

                    touch_bonus = rew.completed_pass_receiver_ind if qualifies_bonus else 0.0
                    team_pass_bonus = getattr(rew, "completed_pass_team", 0.0) if qualifies_bonus else 0.0
                    passer_bonus = 0.60 * r_pass
                    receiver_bonus = 0.40 * r_pass + touch_bonus

                    if qualifies_bonus or r_stride > 0.0:
                        self.team_pass_chain[new_team] = min(self.team_pass_chain[new_team] + 1, 10)
                        self._add_team_reward(step_rewards, new_team, team_pass_bonus)

                        self.last_completed_pass_passer[new_team] = kicker_idx
                        self.last_completed_pass_receiver[new_team] = new_holder
                        self.last_completed_pass_time[new_team] = self.match_time

                        step_rewards[kicker_idx] += passer_bonus
                        step_rewards[new_holder] += receiver_bonus
                        self.stats["completed_passes"] += 1
                        self.stats["team_passes"][new_team] += 1
                        if is_wall_pass:
                            self.stats["wall_passes"] = self.stats.get("wall_passes", 0) + 1

                        if r_stride > 0.0:
                            lbl = "THROUGH BALL!"
                        elif is_wall_pass:
                            lbl = "WALL PASS!"
                        elif d_sigma_press > 0.25 and d_phi_threat <= 0.01:
                            lbl = "ESCAPE PRESS!"
                        else:
                            lbl = "PASS!"

                        self._emit_event(
                            lbl,
                            self.player_pos[new_holder],
                            team=new_team,
                            agent=self.idx_to_agent[new_holder],
                            reward=passer_bonus + receiver_bonus + 2.0 * r_stride,
                        )
            elif new_team != kicker_team:
                # Strictly Zero-Sum Interception or Keeper Save
                self._reset_pass_chains(kicker_team)
                self.spell_max_threat[new_team] = self._compute_ball_potential(new_team, self.ball_pos)
                my_side = int(self.team_side[new_team])
                keeper_facing_ball = False
                if new_role == ROLE_KEEPER:
                    inv_d = 1.0 / max(float(dists[new_holder]), 1e-6)
                    gk_ang = float(self.player_angle[new_holder])
                    keeper_facing_ball = (
                        math.cos(gk_ang) * (float(dx_b[new_holder]) * inv_d)
                        + math.sin(gk_ang) * (float(dy_b[new_holder]) * inv_d)
                    ) >= -0.15
                is_real_save = (
                    new_role == ROLE_KEEPER
                    and keeper_facing_ball
                    and (
                        self._is_ball_heading_into_goal(self.ball_pos, self.ball_vel, my_side)
                        or (
                            self.ball_kick_was_shot_on_target
                            and (
                                (my_side == SIDE_LEFT and self.ball_vel[0] < -40.0)
                                or (my_side == SIDE_RIGHT and self.ball_vel[0] > 40.0)
                            )
                        )
                    )
                )

                now_t = self.match_time
                to_cd = getattr(mech, "turnover_reward_cooldown", 1.0)
                if not is_real_save:
                    if (now_t - self.last_turnover_reward_time) >= to_cd:
                        self.last_turnover_reward_time = now_t
                        win_ind = rew.successful_tackle_tackler_ind
                        win_team = rew.successful_tackle_team
                        step_rewards[new_holder] += win_ind
                        step_rewards[kicker_idx] -= win_ind
                        self._add_team_reward(step_rewards, new_team, win_team)
                        self._add_team_reward(step_rewards, kicker_team, -win_team)
                    self.stats["interceptions"] += 1
                else:
                    self.post_save_box_grace_timer = mech.post_save_box_grace_time
                    gk_gx = 0.0 if my_side == SIDE_LEFT else phys.field_width
                    d_orig = math.hypot(float(self.ball_kick_origin[0]) - gk_gx, float(self.ball_kick_origin[1]) - 420.0)
                    v_arr = math.hypot(float(self.ball_vel[0]), float(self.ball_vel[1]))
                    v_fact = _fast_clip((v_arr - 120.0) / (420.0 - 120.0), 0.15, 1.0)
                    d_fact = _fast_clip((550.0 - d_orig) / (550.0 - 220.0), 0.20, 1.0)
                    mu_save = _fast_clip(v_fact * d_fact, 0.10, 1.0)

                    save_ind = rew.keeper_save_ind * mu_save
                    save_team = rew.keeper_save_team * mu_save
                    step_rewards[new_holder] += save_ind
                    self._add_team_reward(step_rewards, new_team, save_team)

                    # Strict Zero-Sum Save/Shot Rebate: claw back SOT reward + mirror save value
                    kicker_pen = max(save_ind + self.last_shot_sot_reward, abs(rew.intercepted_kicker_ind))
                    team_pen = save_team + self.last_shot_team_reward
                    step_rewards[kicker_idx] -= kicker_pen
                    self._add_team_reward(step_rewards, kicker_team, -team_pen)
                    self.last_shot_sot_reward = 0.0
                    self.last_shot_team_reward = 0.0

                    self.stats["keeper_saves"] += 1
                    self.stats["team_saves"][new_team] += 1
                    lbl = "LUNGE SAVE!" if self.lunge_timer[new_holder] > 0.0 else "KEEPER SAVE!"
                    self._emit_event(
                        lbl,
                        self.player_pos[new_holder],
                        team=new_team,
                        agent=self.idx_to_agent[new_holder],
                        reward=save_ind + save_team,
                    )
        else:
            # Loose ball recovery: strictly zero-sum across teams & gated by turnover cooldown
            if self.last_ball_team is not None and self.last_ball_team != new_team:
                old_team = int(self.last_ball_team)
                self._reset_pass_chains(old_team)
                self.spell_max_threat[new_team] = self._compute_ball_potential(new_team, self.ball_pos)
                now_t = self.match_time
                to_cd = getattr(mech, "turnover_reward_cooldown", 1.0)
                win_ind = rew.successful_tackle_tackler_ind
                win_team = rew.successful_tackle_team
                if (now_t - self.last_turnover_reward_time) >= to_cd:
                    self.last_turnover_reward_time = now_t
                    step_rewards[new_holder] += win_ind
                    if self.last_ball_holder is not None and int(self.player_team[self.last_ball_holder]) == old_team:
                        step_rewards[self.last_ball_holder] -= win_ind
                    self._add_team_reward(step_rewards, new_team, win_team)
                    self._add_team_reward(step_rewards, old_team, -win_team)
                    won_rew = win_ind + win_team
                else:
                    won_rew = 0.0

                opp_slice = slice(6, 12) if new_team == 0 else slice(0, 6)
                opp_dists = np.hypot(
                    self.player_pos[opp_slice, 0] - self.ball_pos[0],
                    self.player_pos[opp_slice, 1] - self.ball_pos[1],
                )
                if len(opp_dists) > 0 and float(np.min(opp_dists)) <= mech.contested_ball_distance:
                    self._emit_event(
                        "BALL WON!",
                        self.player_pos[new_holder],
                        team=new_team,
                        agent=self.idx_to_agent[new_holder],
                        reward=won_rew,
                    )

        # Assign possession to new_holder
        self.ball_holder = new_holder
        self.ball_hold_timer = 0.0
        for side_idx in (0, 1):
            defending_team = 0 if self.team_side[0] == side_idx else 1
            goal_cx = 0.0 if side_idx == 0 else phys.field_width
            d_box = math.hypot(float(self.ball_pos[0]) - goal_cx, float(self.ball_pos[1]) - 420.0)
            if new_team != defending_team or d_box > 205.0:
                self._box_loose_ball_timer[side_idx] = 0.0
        self.kick_grace_timer = mech.pickup_kick_grace
        if not is_teammate_pass:
            self.pickup_tackle_grace_timer = mech.pickup_tackle_grace
        else:
            if d_dest_goal > 225.0:
                self.pickup_tackle_grace_timer = max(
                    self.pickup_tackle_grace_timer,
                    mech.pass_reception_tackle_grace,
                )
        self.last_ball_holder = new_holder
        self.last_ball_team = new_team
        self.ball_kicked_in_flight = False
        self.ball_kick_orig_opp_dist = -1.0
        self.ball_wall_bounces = 0
        self.is_wall_pass_in_flight = False

        holder_ang = float(self.player_angle[new_holder])
        self.ball_pos[0] = float(self.player_pos[new_holder, 0]) + math.cos(holder_ang) * mech.ball_carry_offset
        self.ball_pos[1] = float(self.player_pos[new_holder, 1]) + math.sin(holder_ang) * mech.ball_carry_offset
        if new_role == ROLE_KEEPER:
            if self.team_side[new_team] == SIDE_LEFT:
                self.ball_pos[0] = max(float(self.ball_pos[0]), phys.ball_radius)
            else:
                self.ball_pos[0] = min(
                    float(self.ball_pos[0]), phys.field_width - phys.ball_radius
                )
        self.ball_vel[0] = float(self.player_vel[new_holder, 0])
        self.ball_vel[1] = float(self.player_vel[new_holder, 1])

    def _on_new_player_touched_ball(self, toucher_idx: int) -> None:
        """Reset violation and kick pickup cooldowns when toucher grabs ball, preventing instant point-blank scrum loops."""
        toucher_team = int(self.player_team[toucher_idx])
        now_t = self.match_time
        other_mask = np.ones(self.num_players, dtype=bool)
        other_mask[toucher_idx] = False

        # Clear violation cooldown only for opponents
        clear_violation = other_mask & self.violation_cooldown_active & (self.player_team != toucher_team)
        self.pickup_cooldown[clear_violation] = 0.0
        self.violation_cooldown_active[clear_violation] = False

        # Only clear kick cooldown immediately if touched by a teammate or after 0.35s minimum flight
        can_clear_kick = (self.player_team == toucher_team) | ((now_t - self.last_kick_time) >= 0.35)
        clear_kick = other_mask & self.kick_cooldown_active & can_clear_kick
        self.pickup_cooldown[clear_kick] = 0.0
        self.kick_cooldown_active[clear_kick] = False
