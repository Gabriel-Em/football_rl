"""Fixed-order 60 Hz physics substep, player collisions, boundary clamping and wall bounces."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from football_rl.config import ROLE_DEF, ROLE_KEEPER, ROLE_MID, SIDE_LEFT, SIDE_RIGHT
from football_rl.scalar_math import wrap_to_pi


class PhysicsMixin:
    """Physics of `FootballParallelEnv`: the 60 Hz substep pipeline and the contact and boundary passes it runs."""

    def _get_player_max_speed(self, idx: int) -> float:
        role = self.player_role[idx]
        phys = self.cfg.physics
        if role == ROLE_KEEPER:
            return phys.keeper_max_speed
        elif role == ROLE_DEF:
            return phys.def_max_speed
        elif role == ROLE_MID:
            return phys.mid_max_speed
        else:
            return phys.atk_max_speed

    # ==========================================================================
    # Physics Sub-Step Engine (60 Hz)
    # ==========================================================================
    def _physics_substep(
        self,
        act_array: np.ndarray,
        step_rewards: Optional[np.ndarray] = None,
    ) -> bool:
        """Execute a single 1/60s physics sub-step.

        Returns True if a set-piece teleport occurred during this sub-step so that
        dense ball-advance shaping ignores the teleport discontinuity.
        """
        if step_rewards is None:
            step_rewards = np.zeros(self.num_players, dtype=np.float64)

        dt = self.cfg.physics.dt
        mech = self.cfg.mechanics
        phys = self.cfg.physics

        # ----------------------------------------------------------------------
        # 1. Advance Timers
        # ----------------------------------------------------------------------
        np.maximum(0.0, self.stun_timer - dt, out=self.stun_timer)
        np.maximum(0.0, self.ghost_timer - dt, out=self.ghost_timer)
        self.is_ghost[:] = (self.ghost_timer > 0.0) | ((self.stun_timer > 0.0) & (self.lunge_timer <= 0.0))
        np.maximum(0.0, self.tackle_cooldown - dt, out=self.tackle_cooldown)
        np.maximum(0.0, self.lunge_cooldown - dt, out=self.lunge_cooldown)
        np.maximum(0.0, self.lunge_timer - dt, out=self.lunge_timer)
        np.maximum(0.0, self.pickup_cooldown - dt, out=self.pickup_cooldown)
        np.maximum(0.0, self.shot_on_target_cd - dt, out=self.shot_on_target_cd)
        np.maximum(0.0, self.post_kick_pbrs_timer - dt, out=self.post_kick_pbrs_timer)
        self.kick_grace_timer = max(0.0, self.kick_grace_timer - dt)
        self.kickoff_protection_timer = max(0.0, self.kickoff_protection_timer - dt)
        self.pickup_tackle_grace_timer = max(0.0, self.pickup_tackle_grace_timer - dt)
        self.post_save_box_grace_timer = max(0.0, self.post_save_box_grace_timer - dt)

        # Clear tracking flags when cooldown naturally expires
        expired_mask = self.pickup_cooldown <= 0.0
        self.kick_cooldown_active[expired_mask] = False
        self.violation_cooldown_active[expired_mask] = False

        # Track possession stats, hold timer & per-box anti-stall timers
        bx = float(self.ball_pos[0])
        by = float(self.ball_pos[1])
        bvx = float(self.ball_vel[0])
        bvy = float(self.ball_vel[1])
        ball_spd = math.hypot(bvx, bvy)

        goal_cy = phys.field_center[1]
        box_reach = phys.keeper_box_radius + phys.ball_radius
        d_left = math.hypot(bx - 0.0, by - goal_cy)
        d_right = math.hypot(bx - phys.field_width, by - goal_cy)
        in_left_box = d_left <= box_reach
        in_right_box = d_right <= box_reach

        for side_idx, (in_box, d_goal) in enumerate(((in_left_box, d_left), (in_right_box, d_right))):
            defending_team = 0 if self.team_side[0] == side_idx else 1
            keeper_idx = self._team_keeper_idx(defending_team)
            goal_cx = 0.0 if side_idx == 0 else phys.field_width

            # Check if ball is actively escaping the box radially
            if d_goal > 1e-6:
                u_rad_0 = (bx - goal_cx) / d_goal
                u_rad_1 = (by - goal_cy) / d_goal
                rad_ball_spd = bvx * u_rad_0 + bvy * u_rad_1
            else:
                rad_ball_spd = 0.0

            ball_moving_or_cleared = (
                not in_box
                or (self.ball_holder is not None and int(self.player_team[self.ball_holder]) != defending_team)
                or (self.ball_holder is None and (ball_spd > mech.box_ball_stopped_threshold or rad_ball_spd > 35.0))
            )

            if ball_moving_or_cleared:
                # Ball cleared the box, opponent claimed it, or ball is moving -> reset timer & expand
                self._box_loose_ball_timer[side_idx] = 0.0
                if self.active_box_radius[side_idx] < phys.keeper_box_radius:
                    self.active_box_radius[side_idx] = min(
                        phys.keeper_box_radius,
                        float(self.active_box_radius[side_idx]) + mech.box_expand_rate * dt,
                    )
            else:
                # Ball sits dead/unclaimed inside side_idx's box (or GK micro-tapping): accumulate loose timer
                if self.ball_holder is None:
                    if self.post_save_box_grace_timer <= 0.0:
                        self._box_loose_ball_timer[side_idx] += dt

                if self._box_loose_ball_timer[side_idx] > mech.box_shrink_grace_time:
                    self.active_box_radius[side_idx] = max(
                        mech.box_min_radius,
                        float(self.active_box_radius[side_idx]) - mech.box_shrink_rate * dt,
                    )
                elif self.active_box_radius[side_idx] < phys.keeper_box_radius:
                    self.active_box_radius[side_idx] = min(
                        phys.keeper_box_radius,
                        float(self.active_box_radius[side_idx]) + mech.box_expand_rate * dt,
                    )

        if self.ball_holder is not None:
            holder_team = int(self.player_team[self.ball_holder])
            self.stats["possession_frames"][holder_team] += 1
            self.ball_hold_timer += dt

            # Check 5-second Field Player Violation or 5-second Keeper Violation
            holder_role = int(self.player_role[self.ball_holder])
            if (
                self._is_field_role(holder_role)
                and self.ball_hold_timer >= mech.field_max_hold_time - 1e-9
            ):
                self._handle_field_5s_violation(self.ball_holder, step_rewards)
                return True
            elif (
                holder_role == ROLE_KEEPER
                and self.ball_hold_timer >= mech.keeper_max_hold_time - 1e-9
            ):
                self._handle_keeper_5s_violation(self.ball_holder, step_rewards)
                return True
        else:
            if math.hypot(float(self.ball_vel[0]), float(self.ball_vel[1])) < 1.0:
                self.stats["stationary_ball_frames"] += 1

        # ----------------------------------------------------------------------
        # 2. Process Player Actions (Steering, Throttle, Kick, Special)
        # ----------------------------------------------------------------------
        current_holder = self.ball_holder
        is_keeper = self._is_keeper_list
        max_speed_by_player = self._max_speed_list
        turn_rate = phys.player_max_turn_rate
        dz_pos = phys.throttle_deadzone
        dz_neg = phys.backpedal_threshold
        forward_denom = max(1e-6, 1.0 - dz_pos)
        backward_denom = max(1e-6, 1.0 + dz_neg)
        accel_alpha = min(1.0, dt * phys.player_accel_rate)

        for idx in range(self.num_players):
            if self.stun_timer[idx] > 0.0:
                if self.lunge_timer[idx] > 0.0:
                    pass  # Active kinematic lunge impulse
                else:
                    self.player_vel[idx] = 0.0
                continue

            throttle_raw, steer_raw, kick_power, special_trig = act_array[idx]

            self.raw_steer_input[idx] = steer_raw

            # Decouple body yaw spin from side-foot kick aiming when actively triggering a kick
            is_aiming_kick = (self.ball_holder == idx and kick_power > mech.kick_trigger_threshold)
            if not (not is_keeper[idx] and self.lunge_timer[idx] > 0.0) and not is_aiming_kick:
                turn_delta = steer_raw * turn_rate * dt
                self.player_angle[idx] = float(wrap_to_pi(self.player_angle[idx] + turn_delta))

            # Special Action (action[3] > 0.5): Lunge for Keepers, Tackle for Field Players
            if special_trig > mech.special_trigger_threshold:
                if self.kickoff_protection_timer > 0.0 and self.ball_holder is not None:
                    pass
                elif self.player_role[idx] == ROLE_KEEPER:
                    if self.ball_holder != idx and self.lunge_cooldown[idx] <= 0.0:
                        self.lunge_timer[idx] = mech.lunge_duration
                        self.lunge_cooldown[idx] = mech.keeper_lunge_cooldown
                        dive_angle = float(
                            wrap_to_pi(
                                self.player_angle[idx]
                                + steer_raw * math.radians(mech.kick_steer_max_deflection_deg)
                            )
                        )
                        u_dive = np.array(
                            [math.cos(dive_angle), math.sin(dive_angle)],
                            dtype=np.float64,
                        )
                        self.dive_dir[idx] = u_dive
                        self.dive_speed[idx] = mech.keeper_dive_speed
                        self.player_vel[idx] = u_dive * mech.keeper_dive_speed
                else:
                    # Field Player Tackle attempt
                    if (
                        self.ball_holder != idx
                        and self.tackle_cooldown[idx] <= 0.0
                        and not self.is_ghost[idx]
                    ):
                        self._attempt_tackle(idx, step_rewards)

            # Forward / Backpedal Velocity with Asymmetric Deadzone & Exponential Velocity Inertia:
            # Lunge overrides normal throttle speed for Keepers and Field Players
            if self.player_role[idx] == ROLE_KEEPER and self.lunge_timer[idx] > 0.0:
                self.player_vel[idx] = self.dive_dir[idx] * (
                    self.dive_speed[idx] if self.dive_speed[idx] > 1e-6 else mech.keeper_dive_speed
                )
            elif not is_keeper[idx] and self.lunge_timer[idx] > 0.0:
                # Field player lunge: instant override at field_lunge_speed,
                # standard exponential acceleration smoothing is bypassed for the duration
                pass
            else:
                max_spd = max_speed_by_player[idx]
                if dz_neg <= throttle_raw <= dz_pos:
                    target_speed = 0.0
                elif throttle_raw > dz_pos:
                    throttle_norm = (throttle_raw - dz_pos) / forward_denom
                    target_speed = throttle_norm * max_spd
                else:
                    back_norm = (throttle_raw - dz_neg) / backward_denom  # in [-1.0, 0.0)
                    back_mult = (
                        phys.keeper_backpedal_mult
                        if self.player_role[idx] == ROLE_KEEPER
                        else phys.field_backpedal_mult
                    )
                    target_speed = back_norm * back_mult * max_spd

                desired_vx = math.cos(self.player_angle[idx]) * target_speed
                desired_vy = math.sin(self.player_angle[idx]) * target_speed
                self.player_vel[idx, 0] += (desired_vx - self.player_vel[idx, 0]) * accel_alpha
                self.player_vel[idx, 1] += (desired_vy - self.player_vel[idx, 1]) * accel_alpha

            # Kick / Pass / Shoot: Active if action[2] > kick_trigger_threshold, grace period expired, and player holds ball
            if (
                kick_power > mech.kick_trigger_threshold
                and self.kick_grace_timer <= 0.0
                and self.ball_holder == idx
                and current_holder == idx
            ):
                self._execute_kick(idx, float(kick_power), step_rewards)

        # ----------------------------------------------------------------------
        # 3. Integrate Player Positions & Resolve Collisions / Boundaries
        # ----------------------------------------------------------------------
        self.player_pos += self.player_vel * dt

        # Player-to-player solid circle collisions (skip players in Ghost Mode)
        self._resolve_player_collisions()

        # Semi-Circle Barrier & Field Wall Clamping
        self._clamp_players_to_boundaries(apply_dynamic_push=True)

        # ----------------------------------------------------------------------
        # 4. Update Ball Kinematics & Check Goals / Wall Bounces
        # ----------------------------------------------------------------------
        if self.ball_kicked_in_flight and self.ball_holder is not None:
            self.ball_holder = None

        if self.ball_holder is not None:
            self.ball_kick_flight_timer = 0.0
            # Lock ball to front edge of carrier: player_pos + facing_vec * 24.0
            h_idx = self.ball_holder
            facing_vec = np.array(
                [math.cos(self.player_angle[h_idx]), math.sin(self.player_angle[h_idx])],
                dtype=np.float64,
            )
            self.ball_pos = self.player_pos[h_idx] + facing_vec * mech.ball_carry_offset
            self.ball_vel[:] = self.player_vel[h_idx]

            # Carried-ball boundary clamping
            r_b = phys.ball_radius
            W = phys.field_width
            H = phys.field_height
            h_team = int(self.player_team[h_idx])
            h_side = int(self.team_side[h_team])
            in_goal_mouth_y = (phys.goal_y_min <= self.ball_pos[1] <= phys.goal_y_max)

            # Left boundary: cannot cross unless scoring into enemy goal on left
            if not in_goal_mouth_y or h_side == SIDE_LEFT:
                self.ball_pos[0] = max(float(self.ball_pos[0]), r_b)
            # Right boundary: cannot cross unless scoring into enemy goal on right
            if not in_goal_mouth_y or h_side == SIDE_RIGHT:
                self.ball_pos[0] = min(float(self.ball_pos[0]), W - r_b)
            # Always clamp sideline Y
            self.ball_pos[1] = float(np.clip(self.ball_pos[1], r_b, H - r_b))

            # Push carrier backward if carried ball is pressed into boundary
            d_ball_carrier = float(np.linalg.norm(self.ball_pos - self.player_pos[h_idx]))
            if d_ball_carrier < mech.ball_carry_offset - 1e-4:
                self.player_pos[h_idx] = self.ball_pos - facing_vec * mech.ball_carry_offset

            self.pinned_ball_timer = 0.0

            # Check if a held ball protruding near the goal line crosses into a goal
            goal_scored_by = self._check_goal_scored()
            if goal_scored_by is not None:
                self._handle_goal_scored(goal_scored_by, step_rewards)
                return True

            # Keeper Perimeter Strip
            if self._is_field_role(self.player_role[h_idx]):
                self._check_keeper_perimeter_strip(h_idx, step_rewards)
        else:
            # Free ball kinematics with linear friction deceleration (-155 units/sec^2)
            ball_speed = float(np.linalg.norm(self.ball_vel))
            if ball_speed > 0.0:
                new_speed = max(0.0, ball_speed - phys.ball_friction_decel * dt)
                self.ball_vel *= new_speed / ball_speed

            self.ball_pos += self.ball_vel * dt

            # Check if ball crossed either goal line before clamping walls
            goal_scored_by = self._check_goal_scored()
            if goal_scored_by is not None:
                self._handle_goal_scored(goal_scored_by, step_rewards)
                return True

            # Bounce off walls (restitution = 0.85)
            self._bounce_ball_off_walls()

            # Pass Flight Timeout
            if self.ball_kicked_in_flight:
                self.ball_kick_flight_timer += dt
                flight_time = (
                    self.match_time - self.ball_kick_time
                    if self.ball_kick_time >= 0.0
                    else self.ball_kick_flight_timer
                )
                max_flight = (
                    mech.max_wall_pass_flight_time
                    if self.is_wall_pass_in_flight
                    else mech.max_pass_flight_time
                )
                ball_speed_sq = float(
                    self.ball_vel[0] * self.ball_vel[0] + self.ball_vel[1] * self.ball_vel[1]
                )
                if flight_time > max_flight or self.ball_kick_flight_timer > max_flight or ball_speed_sq < 625.0:  # 25.0 u/s minimum pass speed
                    self.ball_kicked_in_flight = False
                    self.is_wall_pass_in_flight = False
                    self.ball_kick_was_shot_on_target = False
                    self.ball_kick_orig_opp_dist = -1.0

            # Pinned Wall/Semicircle Ball Unsticking
            ball_speed_now = float(np.linalg.norm(self.ball_vel))
            bx, by = float(self.ball_pos[0]), float(self.ball_pos[1])
            near_wall = (
                bx <= phys.ball_radius + 22.0
                or bx >= phys.field_width - (phys.ball_radius + 22.0)
                or by <= phys.ball_radius + 22.0
                or by >= phys.field_height - (phys.ball_radius + 22.0)
            )
            d_left_box = math.hypot(bx - phys.left_goal_center[0], by - phys.left_goal_center[1])
            d_right_box = math.hypot(bx - phys.right_goal_center[0], by - phys.right_goal_center[1])
            in_any_box = (
                d_left_box <= phys.keeper_box_radius + phys.ball_radius
                or d_right_box <= phys.keeper_box_radius + phys.ball_radius
            )
            near_box_edge = (not in_any_box) and (
                abs(d_left_box - phys.keeper_box_radius) <= 24.0
                or abs(d_right_box - phys.keeper_box_radius) <= 24.0
            )
            if (
                ball_speed_now < mech.pinned_ball_speed_thresh
                and (near_wall or near_box_edge)
                and int(np.sum(np.linalg.norm(self.player_pos - self.ball_pos[None, :], axis=1) <= 68.0)) >= 2
            ):
                self.pinned_ball_timer += dt
                if self.pinned_ball_timer >= mech.pinned_ball_max_time:
                    center_vec = np.array(phys.field_center, dtype=np.float64) - self.ball_pos
                    c_dist = float(np.linalg.norm(center_vec))
                    if c_dist > 1e-6:
                        nudge_dir = center_vec / c_dist
                        self.ball_vel = nudge_dir * mech.pinned_ball_nudge_speed
                    self.pinned_ball_timer = 0.0
            else:
                self.pinned_ball_timer = 0.0

            # Check free ball pickup by eligible players (dist <= 28.0 and pickup_cooldown == 0)
            self._check_free_ball_pickup(step_rewards)

        return False

    # ==========================================================================
    # Collisions, Semi-Circle Barriers & Boundary Clamping
    # ==========================================================================
    def _resolve_player_collisions(self) -> None:
        """Solid circle-circle collision resolution between non-ghost players (R=18 -> min_dist=36)."""
        min_dist = 2.0 * self.cfg.physics.player_radius
        pos = self.player_pos
        ghost = self.is_ghost
        n = self.num_players
        for i in range(n):
            if ghost[i]:
                continue
            xi = float(pos[i, 0])
            yi = float(pos[i, 1])
            for j in range(i + 1, n):
                if ghost[j]:
                    continue
                dx = float(pos[j, 0]) - xi
                if abs(dx) >= min_dist:
                    continue
                dy = float(pos[j, 1]) - yi
                if abs(dy) >= min_dist:
                    continue
                dist = math.hypot(dx, dy)
                if dist < min_dist:
                    if dist < 1e-6:
                        angle = i * 0.7 + j * 1.3
                        nx = math.cos(angle)
                        ny = math.sin(angle)
                        overlap = min_dist
                    else:
                        inv_d = 1.0 / dist
                        nx = dx * inv_d
                        ny = dy * inv_d
                        overlap = min_dist - dist
                    half_ov = 0.5 * overlap
                    pos[i, 0] -= half_ov * nx
                    pos[i, 1] -= half_ov * ny
                    pos[j, 0] += half_ov * nx
                    pos[j, 1] += half_ov * ny
                    xi = float(pos[i, 0])
                    yi = float(pos[i, 1])

    def _outfield_boundary_candidates(self) -> np.ndarray:
        """Boolean (12,) mask of outfield players close enough to a wall or keeper semicircle to need clamping.

        A player outside every band cannot trigger any branch of the outfield clamp, so skipping it is exact.
        """
        phys = self.cfg.physics
        r_player = phys.player_radius
        x = self.player_pos[:, 0]
        y = self.player_pos[:, 1]
        band_left = max(float(self.active_box_radius[0]), r_player) + r_player + 2.0
        band_right = max(float(self.active_box_radius[1]), r_player) + r_player + 2.0
        near = (
            (x < band_left)
            | (x > phys.field_width - band_right)
            | (y < r_player)
            | (y > phys.field_height - r_player)
        )
        return near & ~self._is_keeper

    def _clamp_players_to_boundaries(self, apply_dynamic_push: bool = True) -> None:
        """Enforce field walls and semi-circle barriers with smooth tangential sliding and corner trap resolution."""
        phys = self.cfg.physics
        r_player = phys.player_radius
        box_r = phys.keeper_box_radius
        goal_centers = self._goal_centers

        # 1. Keepers clamped inside own box
        for k_idx in (0, 6):
            my_side = int(self.team_side[self.player_team[k_idx]])
            own_goal = goal_centers[my_side]
            delta = self.player_pos[k_idx] - own_goal
            dist = math.hypot(float(delta[0]), float(delta[1]))
            if dist > box_r:
                self.player_pos[k_idx] = own_goal + (delta / dist) * box_r

        # 2. Outfield players outside keeper semicircle envelopes and inside field walls
        # Smooth continuous boundary envelope: resolves the wall-semicircle intersection
        # and projects velocity tangentially around the box to eliminate zero-velocity fixed points.
        eff_box_left = float(self.active_box_radius[0])
        eff_box_right = float(self.active_box_radius[1])

        for p_idx in np.flatnonzero(self._outfield_boundary_candidates()).tolist():
            px = float(self.player_pos[p_idx, 0])
            py = float(self.player_pos[p_idx, 1])
            vx = float(self.player_vel[p_idx, 0])
            vy = float(self.player_vel[p_idx, 1])

            # Left goalmouth boundary envelope
            dy_l = py - 420.0
            if abs(dy_l) < eff_box_left and eff_box_left > 1e-3:
                dx_arc_l = math.sqrt(eff_box_left * eff_box_left - dy_l * dy_l)
            else:
                dx_arc_l = 0.0
            x_min = max(r_player, dx_arc_l + 1e-3 if dx_arc_l > 0.0 else r_player)

            if px < x_min:
                px = x_min

            dist_l = math.hypot(px, dy_l)
            if abs(dy_l) < eff_box_left and dist_l <= eff_box_left + 1e-3 and eff_box_left > 1e-3:
                dist_safe = max(1e-6, dist_l)
                nx_l = px / dist_safe
                ny_l = dy_l / dist_safe

                if apply_dynamic_push:
                    # Tangent vector along arc directed into the field (tx > 0) toward goalmouth
                    s_ny = 1.0 if ny_l >= 0.0 else -1.0
                    tx_l = abs(ny_l)
                    ty_l = -nx_l * s_ny

                    # Corner vertex near the wall (px <= r_player + 0.5)
                    if px <= r_player + 0.5:
                        pushes_away = (dy_l < 0.0 and vy < 0.0) or (dy_l > 0.0 and vy > 0.0)
                        if pushes_away:
                            vx = 0.0
                        else:
                            spd = math.hypot(vx, vy)
                            vt_l = max(spd * tx_l, vx * tx_l + vy * ty_l)
                            vx = vt_l * tx_l
                            vy = vt_l * ty_l
                    else:
                        vn_l = vx * nx_l + vy * ny_l
                        if vn_l < 0.0:
                            vx -= vn_l * nx_l
                            vy -= vn_l * ny_l
                        if vx < 0.0 and px <= r_player + 2.0:
                            vt_l = max(0.0, vx * tx_l + vy * ty_l)
                            vx = vt_l * tx_l
                            vy = vt_l * ty_l
            elif px <= r_player:
                px = r_player
                if apply_dynamic_push and vx < 0.0:
                    vx = 0.0

            # Right goalmouth boundary envelope
            dy_r = py - 420.0
            if abs(dy_r) < eff_box_right and eff_box_right > 1e-3:
                dx_arc_r = math.sqrt(eff_box_right * eff_box_right - dy_r * dy_r)
            else:
                dx_arc_r = 0.0
            x_max = min(phys.field_width - r_player, phys.field_width - (dx_arc_r + 1e-3 if dx_arc_r > 0.0 else 0.0))

            if px > x_max:
                px = x_max

            dist_r = math.hypot(phys.field_width - px, dy_r)
            if abs(dy_r) < eff_box_right and dist_r <= eff_box_right + 1e-3 and eff_box_right > 1e-3:
                dist_safe = max(1e-6, dist_r)
                nx_r = (px - phys.field_width) / dist_safe
                ny_r = dy_r / dist_safe

                if apply_dynamic_push:
                    # Tangent vector along arc directed into the field (tx < 0) toward goalmouth
                    s_ny = 1.0 if ny_r >= 0.0 else -1.0
                    tx_r = -abs(ny_r)
                    ty_r = nx_r * s_ny

                    # Corner vertex near the wall (px >= field_width - r_player - 0.5)
                    if px >= phys.field_width - r_player - 0.5:
                        pushes_away = (dy_r < 0.0 and vy < 0.0) or (dy_r > 0.0 and vy > 0.0)
                        if pushes_away:
                            vx = 0.0
                        else:
                            spd = math.hypot(vx, vy)
                            vt_r = max(spd * abs(tx_r), vx * tx_r + vy * ty_r)
                            vx = vt_r * tx_r
                            vy = vt_r * ty_r
                    else:
                        vn_r = vx * nx_r + vy * ny_r
                        if vn_r < 0.0:
                            vx -= vn_r * nx_r
                            vy -= vn_r * ny_r
                        if vx > 0.0 and px >= phys.field_width - r_player - 2.0:
                            vt_r = max(0.0, vx * tx_r + vy * ty_r)
                            vx = vt_r * tx_r
                            vy = vt_r * ty_r
            elif px >= phys.field_width - r_player:
                px = phys.field_width - r_player
                if apply_dynamic_push and vx > 0.0:
                    vx = 0.0

            # Top & bottom field walls with normal dissipation
            if py < r_player:
                py = r_player
                if apply_dynamic_push and vy < 0.0:
                    vy = 0.0
            elif py > phys.field_height - r_player:
                py = phys.field_height - r_player
                if apply_dynamic_push and vy > 0.0:
                    vy = 0.0

            self.player_pos[p_idx, 0] = px
            self.player_pos[p_idx, 1] = py
            self.player_vel[p_idx, 0] = vx
            self.player_vel[p_idx, 1] = vy

        # Kickoff Exclusion Circle: Smoothly or firmly keep opposing players outside center spot (r = 130.0)
        if self.kickoff_protection_timer > 0.0 and self.ball_holder is not None:
            start_team = int(self.player_team[self.ball_holder])
            field_center = np.array(phys.field_center, dtype=np.float64)
            excl_r = self.cfg.mechanics.kickoff_exclusion_radius
            for idx in range(self.num_players):
                if self.player_team[idx] != start_team:
                    delta = self.player_pos[idx] - field_center
                    dist = math.hypot(float(delta[0]), float(delta[1]))
                    if dist < excl_r:
                        push_dir = (
                            np.array([1.0, 0.0], dtype=np.float64)
                            if dist < 1e-6
                            else delta / dist
                        )
                        new_pos = field_center + push_dir * excl_r
                        self.player_pos[idx] = new_pos

        # Smooth GK Possession Repulsion Wave (gk_hold_opponent_buffer = 265.0):
        # Genuine physical pushout capped at gk_hold_push_speed without 1-frame teleporting or wave surfing!
        if (
            apply_dynamic_push
            and self.ball_holder is not None
            and self.player_role[self.ball_holder] == ROLE_KEEPER
        ):
            gk_team = int(self.player_team[self.ball_holder])
            gk_side = int(self.team_side[gk_team])
            gk_goal_center = goal_centers[gk_side]
            buf_r = self.cfg.mechanics.gk_hold_opponent_buffer
            max_step_push = self.cfg.mechanics.gk_hold_push_speed * phys.dt
            clamp_spd = self.cfg.mechanics.gk_hold_push_speed
            opp_slice = slice(6, 12) if gk_team == 0 else slice(0, 6)

            delta = self.player_pos[opp_slice] - gk_goal_center
            dist = np.hypot(delta[:, 0], delta[:, 1])
            in_buf = dist < buf_r

            if np.any(in_buf):
                dist_safe = np.maximum(dist, 1e-6)
                push_dir = delta / dist_safe[:, None]
                too_close = dist < 1e-6
                if np.any(too_close):
                    push_dir[too_close, 0] = 1.0 if gk_side == SIDE_LEFT else -1.0
                    push_dir[too_close, 1] = 0.0

                rad_v = np.sum(self.player_vel[opp_slice] * push_dir, axis=1)
                already_moved_out = np.maximum(0.0, rad_v) * phys.dt
                rem_step_push = np.maximum(0.0, max_step_push - already_moved_out)
                push_mag = np.minimum(buf_r - dist, rem_step_push)

                mask = in_buf[:, None]
                self.player_pos[opp_slice] += push_dir * (push_mag[:, None] * mask)

                vel_adj = np.where(
                    rad_v < 0.0,
                    rad_v,
                    np.where(rad_v > clamp_spd, rad_v - clamp_spd, 0.0),
                )
                self.player_vel[opp_slice] -= push_dir * (vel_adj[:, None] * mask)

        # Single vectorized outer boundary clamp for all players
        np.clip(
            self.player_pos[:, 0],
            r_player,
            phys.field_width - r_player,
            out=self.player_pos[:, 0],
        )
        np.clip(
            self.player_pos[:, 1],
            r_player,
            phys.field_height - r_player,
            out=self.player_pos[:, 1],
        )

        hit_left = self.player_pos[:, 0] <= r_player
        self.player_vel[hit_left & (self.player_vel[:, 0] < 0.0), 0] = 0.0

        hit_right = self.player_pos[:, 0] >= phys.field_width - r_player
        self.player_vel[hit_right & (self.player_vel[:, 0] > 0.0), 0] = 0.0

        hit_top = self.player_pos[:, 1] <= r_player
        self.player_vel[hit_top & (self.player_vel[:, 1] < 0.0), 1] = 0.0

        hit_bottom = self.player_pos[:, 1] >= phys.field_height - r_player
        self.player_vel[hit_bottom & (self.player_vel[:, 1] > 0.0), 1] = 0.0

    def _bounce_ball_off_walls(self) -> bool:
        """Bounce free ball elastically off non-goal walls."""
        phys = self.cfg.physics
        r_ball = phys.ball_radius
        rest = phys.ball_wall_restitution
        bounced = False
        bounce_side_wall = False
        bounce_back_wall = False

        bx, by = float(self.ball_pos[0]), float(self.ball_pos[1])
        in_goal_y = phys.goal_y_min <= by <= phys.goal_y_max

        # Left and Right walls (Back walls / Endlines)
        if not in_goal_y:
            if bx < r_ball:
                self.ball_pos[0] = r_ball + (r_ball - bx) * rest
                self.ball_vel[0] = abs(self.ball_vel[0]) * rest
                self.ball_vel[1] *= rest
                bounced = True
                bounce_back_wall = True
            elif bx > phys.field_width - r_ball:
                over = bx - (phys.field_width - r_ball)
                self.ball_pos[0] = (phys.field_width - r_ball) - over * rest
                self.ball_vel[0] = -abs(self.ball_vel[0]) * rest
                self.ball_vel[1] *= rest
                bounced = True
                bounce_back_wall = True

        # Top and Bottom walls (Longitudinal Side boards Y = 0 and Y = 840)
        by = float(self.ball_pos[1])
        if by < r_ball:
            self.ball_pos[1] = r_ball + (r_ball - by) * rest
            self.ball_vel[1] = abs(self.ball_vel[1]) * rest
            self.ball_vel[0] *= rest
            bounced = True
            bounce_side_wall = True
        elif by > phys.field_height - r_ball:
            over = by - (phys.field_height - r_ball)
            self.ball_pos[1] = (phys.field_height - r_ball) - over * rest
            self.ball_vel[1] = -abs(self.ball_vel[1]) * rest
            self.ball_vel[0] *= rest
            bounced = True
            bounce_side_wall = True

        if bounced:
            self.ball_kick_was_shot_on_target = False
            self.ball_wall_bounces += 1
            if bounce_back_wall and self.last_ball_team is not None:
                self._reset_pass_chains(self.last_ball_team)
            if (
                bounce_side_wall
                and not bounce_back_wall
                and self.ball_wall_bounces == 1
                and self.ball_kicked_in_flight
            ):
                # Valid intentional single side-wall bank pass: keep in-flight active with max_wall_pass_flight_time
                self.is_wall_pass_in_flight = True
            else:
                # Back wall bounce, corner rattle, or multi-wall bounce: cancel in-flight pass status
                self.ball_kicked_in_flight = False
                self.is_wall_pass_in_flight = False

        return bounced
