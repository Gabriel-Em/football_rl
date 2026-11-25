"""Match rules: 5-second violations, goals, halftime side switch, kickoff setup and the per-side box timer."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from football_rl.config import SIDE_LEFT, SIDE_RIGHT, TEAM_A
from football_rl.scalar_math import wrap_to_pi


TEAM_LEFT = SIDE_LEFT
TEAM_RIGHT = SIDE_RIGHT


class _DualBoxTimer:
    """Per-side box loose-ball timer supporting both side indexing [0]/[1] and scalar float operations."""

    def __init__(self, left: float = 0.0, right: float = 0.0) -> None:
        self._vals: List[float] = [float(left), float(right)]

    def __getitem__(self, idx: int) -> float:
        return self._vals[idx]

    def __setitem__(self, idx: int, val: float) -> None:
        self._vals[idx] = float(val)

    def fill(self, val: float) -> None:
        v = float(val)
        self._vals[0] = v
        self._vals[1] = v

    @property
    def max_val(self) -> float:
        return max(self._vals[0], self._vals[1])

    def __float__(self) -> float:
        return self.max_val

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (int, float, np.floating)):
            return self.max_val == float(other)
        if isinstance(other, (list, tuple, np.ndarray)):
            return len(other) == 2 and self._vals[0] == float(other[0]) and self._vals[1] == float(other[1])
        if isinstance(other, _DualBoxTimer):
            return self._vals == other._vals
        return False

    def __lt__(self, other: float) -> bool:
        return self.max_val < float(other)

    def __le__(self, other: float) -> bool:
        return self.max_val <= float(other)

    def __gt__(self, other: float) -> bool:
        return self.max_val > float(other)

    def __ge__(self, other: float) -> bool:
        return self.max_val >= float(other)

    def __add__(self, other: float) -> float:
        return self.max_val + float(other)

    def __radd__(self, other: float) -> float:
        return float(other) + self.max_val

    def __sub__(self, other: float) -> float:
        return self.max_val - float(other)

    def __rsub__(self, other: float) -> float:
        return float(other) - self.max_val

    def __mul__(self, other: float) -> float:
        return self.max_val * float(other)

    def __rmul__(self, other: float) -> float:
        return float(other) * self.max_val

    def __truediv__(self, other: float) -> float:
        return self.max_val / float(other)

    def __repr__(self) -> str:
        return f"{self.max_val}"


class MatchRulesMixin:
    """Match rules of `FootballParallelEnv`: violations, goals, halftime, kickoff and timer resets."""

    def _handle_field_5s_violation(
        self, violator_idx: int, step_rewards: np.ndarray
    ) -> None:
        """Enforce Field Player 5-Second Hold Violation: drop ball and enter Ghost Mode for 3.0s."""
        mech = self.cfg.mechanics
        rew = self.cfg.rewards
        team = int(self.player_team[violator_idx])

        self.ball_holder = None
        self.ball_hold_timer = 0.0
        self.ball_vel[:] = 0.0
        self.ball_kicked_in_flight = False
        self._reset_pass_chains(team)

        self.ghost_timer[violator_idx] = mech.ghost_duration
        self.pickup_cooldown[violator_idx] = mech.field_violation_pickup_cooldown
        self.tackle_cooldown[violator_idx] = max(
            self.tackle_cooldown[violator_idx], mech.field_violation_tackle_cooldown
        )
        self.violation_cooldown_active[violator_idx] = True
        self.kick_cooldown_active[violator_idx] = False

        step_rewards[violator_idx] += rew.field_5s_violation_ind
        self._add_team_reward(step_rewards, team, rew.field_5s_violation_team)
        self.stats["field_5s_violations"] += 1
        self._emit_event(
            "5s VIOLATION!",
            self.player_pos[violator_idx],
            team=team,
            agent=self.idx_to_agent[violator_idx],
            reward=rew.field_5s_violation_ind + rew.field_5s_violation_team,
        )

    def _handle_keeper_5s_violation(self, keeper_idx: int, step_rewards: np.ndarray) -> None:
        """Penalize 5s GK violation by awarding wide flank possession to opponents while resetting GK to goal mouth."""
        phys = self.cfg.physics
        mech = self.cfg.mechanics
        rew = self.cfg.rewards

        keeper_team = int(self.player_team[keeper_idx])
        self._reset_pass_chains(keeper_team)
        self.stats["keeper_5s_violations"] += 1
        if step_rewards is not None:
            step_rewards[keeper_idx] += rew.keeper_5s_violation_ind

        own_side = int(self.team_side[keeper_team])
        own_goal = self.left_goal_center if own_side == SIDE_LEFT else self.right_goal_center
        outward_sign = 1.0 if own_side == SIDE_LEFT else -1.0

        # 1. Return GK to goal-mouth home anchor so the foul is not an automatic open-goal tap-in
        self.player_pos[keeper_idx, 0] = float(own_goal[0]) + outward_sign * 40.0
        self.player_pos[keeper_idx, 1] = float(own_goal[1])
        self.player_vel[keeper_idx] = 0.0
        self.player_angle[keeper_idx] = 0.0 if own_side == SIDE_LEFT else math.pi

        # 2. Place ball at wide flank indirect outlet (250u out, y=140 or y=700 toward nearest opponent)
        opp_indices = self._team_right_indices if keeper_team == TEAM_LEFT else self._team_left_indices
        flank_y = 140.0 if float(self.ball_pos[1]) < phys.field_height * 0.5 else (phys.field_height - 140.0)
        opp_y = self.player_pos[opp_indices, 1]
        if opp_y.size > 0:
            if float(np.min(np.abs(opp_y - 140.0))) < float(np.min(np.abs(opp_y - (phys.field_height - 140.0)))):
                flank_y = 140.0
            else:
                flank_y = phys.field_height - 140.0

        outlet_x = float(own_goal[0]) + outward_sign * (phys.keeper_box_radius + 55.0)
        self.ball_pos[0] = outlet_x
        self.ball_pos[1] = flank_y
        self.ball_vel[0] = outward_sign * 120.0
        self.ball_vel[1] = 0.0

        # 3. Lock out the offending team from immediate recovery for 1.5s
        team_mask = self.player_team == keeper_team
        self.pickup_cooldown[team_mask] = np.maximum(self.pickup_cooldown[team_mask], 1.5)
        self.pickup_cooldown[keeper_idx] = max(
            float(self.pickup_cooldown[keeper_idx]), mech.keeper_foul_pickup_cooldown
        )

        self.ball_holder = None
        self.ball_hold_timer = 0.0
        self.ball_kicked_in_flight = False
        self.is_wall_pass_in_flight = False
        self._box_loose_ball_timer[own_side] = 0.0
        self.active_box_radius[own_side] = phys.keeper_box_radius

        self._emit_event(
            "5s KEEPER HOLD FOUL!",
            self.ball_pos,
            team=keeper_team,
            agent=self.idx_to_agent[keeper_idx],
            reward=rew.keeper_5s_violation_ind,
        )

    def _check_goal_scored(self) -> Optional[int]:
        """Check if the ball crossed X <= 0 or X >= 1400 within Y in [340.0, 500.0]."""
        phys = self.cfg.physics
        bx, by = float(self.ball_pos[0]), float(self.ball_pos[1])

        if phys.goal_y_min <= by <= phys.goal_y_max:
            if bx <= 0.0:
                scoring_team = int(np.where(self.team_side == SIDE_RIGHT)[0][0])
                return scoring_team
            if bx >= phys.field_width:
                scoring_team = int(np.where(self.team_side == SIDE_LEFT)[0][0])
                return scoring_team
        return None

    # ==========================================================================
    # Set-Pieces: Kickoff, Goal Scored, Half-Time Switch, 5s Keeper Foul
    # ==========================================================================
    def _handle_goal_scored(
        self, scoring_team: int, step_rewards: np.ndarray
    ) -> None:
        """Award goal rewards, update scoreboard, and reset to Kickoff for conceding team."""
        rew = self.cfg.rewards
        conceding_team = 1 - scoring_team

        is_own_goal = False
        scorer_ind = 0.0

        if self.last_ball_holder is not None:
            if int(self.player_team[self.last_ball_holder]) == scoring_team:
                scorer_ind = rew.goal_scorer_ind
                step_rewards[self.last_ball_holder] += scorer_ind

                # Goal Assist Award (within goal_assist_window_sec)
                if (
                    self.last_completed_pass_receiver[scoring_team] == self.last_ball_holder
                    and self.last_completed_pass_passer[scoring_team] is not None
                    and self.last_completed_pass_passer[scoring_team] != self.last_ball_holder
                    and (self.match_time - self.last_completed_pass_time[scoring_team]) <= rew.goal_assist_window_sec
                ):
                    assister = self.last_completed_pass_passer[scoring_team]
                    step_rewards[assister] += rew.goal_assist_ind
                    self.stats["assists"] = self.stats.get("assists", 0) + 1
                    if "team_assists" in self.stats:
                        self.stats["team_assists"][scoring_team] += 1
            else:
                scorer_ind = rew.own_goal_ind
                step_rewards[self.last_ball_holder] += scorer_ind
                self.stats["own_goals"] = self.stats.get("own_goals", 0) + 1
                is_own_goal = True

        scoring_team_reward = rew.goal_team_shared
        self._add_team_reward(step_rewards, scoring_team, scoring_team_reward)
        self._add_team_reward(step_rewards, conceding_team, rew.goal_concede_team_shared)

        self.stats["score"][scoring_team] += 1
        self.stats["goals"] += 1

        scorer_name = (
            self.idx_to_agent[self.last_ball_holder]
            if self.last_ball_holder is not None
            else f"team_{scoring_team}"
        )
        if is_own_goal:
            self._emit_event(
                "OWN GOAL!",
                self.ball_pos.copy(),
                team=conceding_team,
                agent=scorer_name,
                reward=rew.own_goal_ind + rew.goal_concede_team_shared,
            )
        else:
            self._emit_event(
                "GOAL!",
                self.ball_pos.copy(),
                team=scoring_team,
                agent=scorer_name,
                reward=scorer_ind + scoring_team_reward,
            )

        self._reset_pass_chains()
        self._setup_kickoff(starting_team=conceding_team)

    def _handle_halftime(self) -> None:
        """At t = 90.0s, physically switch team sides and alternate kickoff team."""
        self.current_half = 2
        self.team_side = 1 - self.team_side
        second_half_kickoff_team = 1 - self.first_half_kickoff_team
        self._setup_kickoff(starting_team=second_half_kickoff_team)
        self._emit_event(
            "HALFTIME!",
            np.array(self.cfg.physics.field_center, dtype=np.float64),
            team=second_half_kickoff_team,
            agent="referee",
        )

    def _clear_all_dynamic_timers(self) -> None:
        """Reset velocities and temporary status timers on a set-piece teleport."""
        self.player_vel[:] = 0.0
        self.stun_timer[:] = 0.0
        self.ghost_timer[:] = 0.0
        self.is_ghost[:] = False
        self.tackle_cooldown[:] = 0.0
        self.lunge_cooldown[:] = 0.0
        self.lunge_timer[:] = 0.0
        self.pickup_cooldown[:] = 0.0
        self.kick_cooldown_active[:] = False
        self.violation_cooldown_active[:] = False
        self.shot_on_target_cd[:] = 0.0
        self.ball_vel[:] = 0.0
        self.ball_hold_timer = 0.0
        self.kick_grace_timer = 0.0
        self._box_loose_ball_timer = _DualBoxTimer(0.0, 0.0)
        self.post_save_box_grace_timer = 0.0
        self.active_box_radius[:] = self.cfg.physics.keeper_box_radius
        self.ball_kicked_in_flight = False
        self.ball_kick_was_shot_on_target = False
        self.ball_kick_flight_timer = 0.0
        self.ball_kick_orig_opp_dist = -1.0
        self.ball_wall_bounces = 0
        self.is_wall_pass_in_flight = False
        self.pinned_ball_timer = 0.0
        self.kickoff_protection_timer = 0.0
        self.pickup_tackle_grace_timer = 0.0
        self.dive_dir.fill(0.0)
        self.dive_speed.fill(0.0)
        self.last_turnover_reward_time = -999.0
        self.last_kick_time.fill(-999.0)
        self.last_shot_sot_reward = 0.0
        self.last_shot_team_reward = 0.0
        self.spell_max_threat.fill(-1.0)
        self._reset_pass_chains()

    def _reset_pass_chains(self, team: Optional[int] = None) -> None:
        """Reset pass chain, pair oscillation, and assist tracking on turnover or dead ball."""
        if team is None:
            self.team_pass_chain = [0, 0]
            self.last_completed_pass_passer = [None, None]
            self.last_completed_pass_receiver = [None, None]
            self.last_completed_pass_time.fill(-999.0)
            self.last_pass_pair = [None, None]
            self.pair_pass_count = [0, 0]
            self.spell_max_threat.fill(-1.0)
        else:
            self.team_pass_chain[team] = 0
            self.last_completed_pass_passer[team] = None
            self.last_completed_pass_receiver[team] = None
            self.last_completed_pass_time[team] = -999.0
            self.last_pass_pair[team] = None
            self.pair_pass_count[team] = 0
            self.spell_max_threat[team] = -1.0

    def _CanonicalToWorld(
        self, pos: Tuple[float, float], angle: float, side: int
    ) -> Tuple[np.ndarray, float]:
        """Transform a canonical Left-defending coordinate to world space based on physical side."""
        if side == SIDE_LEFT:
            return np.array(pos, dtype=np.float64), float(wrap_to_pi(angle))
        wx = self.cfg.physics.field_width - pos[0]
        wy = self.cfg.physics.field_height - pos[1]
        w_angle = float(wrap_to_pi(angle + math.pi))
        return np.array([wx, wy], dtype=np.float64), w_angle

    def _setup_kickoff(self, starting_team: int) -> None:
        """Position both teams in Kickoff formation, respecting their current physical side."""
        self._resolve_pending_kick_as_other()
        self._clear_all_dynamic_timers()
        self.kickoff_protection_timer = self.cfg.mechanics.kickoff_protection_time
        form = self.cfg.formations
        start_side = int(self.team_side[starting_team])

        for idx in range(self.num_players):
            team = int(self.player_team[idx])
            role_key = self.player_role_key[idx]
            if team == starting_team:
                canon_pos = form.kickoff_starting_left[role_key]
                canon_angle = 2.4189 if role_key == "atk_1" else 0.0
            else:
                canon_pos = form.kickoff_receiving_right[role_key]
                canon_angle = math.pi

            world_pos, world_angle = self._CanonicalToWorld(canon_pos, canon_angle, start_side)
            self.player_pos[idx] = world_pos
            self.player_angle[idx] = world_angle

        # Starting team's atk_1 holds the ball at center (700, 420)
        carrier_name = "a_atk_1" if starting_team == TEAM_A else "b_atk_1"
        carrier_idx = self.agent_to_idx[carrier_name]
        self.ball_holder = carrier_idx
        self.last_ball_holder = carrier_idx
        self.last_ball_team = starting_team

        self._reset_pass_chains()

        facing_vec = np.array(
            [
                math.cos(self.player_angle[carrier_idx]),
                math.sin(self.player_angle[carrier_idx]),
            ],
            dtype=np.float64,
        )
        self.ball_pos = (
            self.player_pos[carrier_idx] + facing_vec * self.cfg.mechanics.ball_carry_offset
        )
