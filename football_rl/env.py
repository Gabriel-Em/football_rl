"""PettingZoo ParallelEnv implementation for 6v6 Continuous 2D MARL Football."""

from __future__ import annotations

from datetime import datetime
import functools
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pettingzoo import ParallelEnv

from football_rl.config import (
    ALL_AGENTS,
    DEFAULT_CONFIG,
    EnvConfig,
    ROLE_ATK,
    ROLE_DEF,
    ROLE_FIELD,
    ROLE_KEEPER,
    ROLE_MID,
    SIDE_LEFT,
    SIDE_RIGHT,
    TEAM_A,
    TEAM_A_AGENTS,
    TEAM_B,
    TEAM_B_AGENTS,
)


def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) in radians to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def parallel_env(config: Optional[EnvConfig] = None, render_mode: Optional[str] = None) -> "FootballParallelEnv":
    """Factory function for creating the PettingZoo ParallelEnv."""
    return FootballParallelEnv(config=config, render_mode=render_mode)


env = parallel_env


class FootballParallelEnv(ParallelEnv):
    """6v6 (1 GK + 2 DEF + 2 MID + 1 ATK per team) Continuous 2D Physics Sports Environment.

    Physics runs at 60 Hz (`dt = 1/60s`) with an RL decision rate of 15 Hz (`frame_skip = 4`).
    All observations are returned in 180-degree rotated Team-Canonical Coordinates so every
    agent perceives its own goal at `(0.0, 420.0)` and the opponent's goal at `(1400.0, 420.0)`
    across both halves, even after teams physically switch sides at Half-Time (`t = 90.0s`).
    """

    metadata = {
        "render_modes": ["human", "rgb_array", None],
        "name": "football_6v6_v0",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.cfg: EnvConfig = config if config is not None else DEFAULT_CONFIG
        self.render_mode: Optional[str] = render_mode

        self.possible_agents: List[str] = list(ALL_AGENTS)
        self.agents: List[str] = []
        self.num_players: int = len(self.possible_agents)

        # Agent index lookup tables
        self.agent_to_idx: Dict[str, int] = {name: i for i, name in enumerate(self.possible_agents)}
        self.idx_to_agent: Dict[int, str] = {i: name for i, name in enumerate(self.possible_agents)}

        # Static team & role arrays
        self.player_team: np.ndarray = np.array(
            [TEAM_A if name in TEAM_A_AGENTS else TEAM_B for name in self.possible_agents],
            dtype=np.int32,
        )
        roles = []
        for name in self.possible_agents:
            if "keeper" in name:
                roles.append(ROLE_KEEPER)
            elif "def" in name:
                roles.append(ROLE_DEF)
            elif "mid" in name:
                roles.append(ROLE_MID)
            else:
                roles.append(ROLE_ATK)
        self.player_role: np.ndarray = np.array(roles, dtype=np.int32)
        self.player_role_key: List[str] = [
            name.split("_", 1)[1] for name in self.possible_agents
        ]  # ["keeper", "def_1", "def_2", "mid_1", "mid_2", "atk_1"] * 2

        # Precompute fixed entity slot indices (11 other players per agent: 5 teammates + 5 field opps + 1 enemy keeper)
        # Slot 0 is always the Ball.
        # Slots 1..5: The 5 other teammates (sorted by ID)
        # Slots 6..10: The 5 opponent field players (sorted by ID)
        # Slot 11: The opponent keeper
        self._entity_player_slots: Dict[int, List[int]] = {}
        for idx in range(self.num_players):
            my_team = self.player_team[idx]
            opp_team = 1 - my_team
            teammates = [
                j for j in range(self.num_players)
                if j != idx and self.player_team[j] == my_team
            ]
            field_opponents = [
                j for j in range(self.num_players)
                if self.player_team[j] == opp_team and self.player_role[j] != ROLE_KEEPER
            ]
            enemy_keeper = [
                j for j in range(self.num_players)
                if self.player_team[j] == opp_team and self.player_role[j] == ROLE_KEEPER
            ]
            assert len(teammates) == 5, f"Expected 5 teammates, got {len(teammates)}"
            assert len(field_opponents) == 5, f"Expected 5 field opponents, got {len(field_opponents)}"
            assert len(enemy_keeper) == 1, f"Expected 1 enemy keeper, got {len(enemy_keeper)}"
            self._entity_player_slots[idx] = teammates + field_opponents + enemy_keeper

        # Precompute max diagonal distance for normalization
        self._max_field_dist: float = math.hypot(
            self.cfg.physics.field_width, self.cfg.physics.field_height
        )

        # Spaces
        self._obs_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.cfg.vision.obs_dim,),
            dtype=np.float32,
        )
        self._act_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.cfg.vision.action_dim,),
            dtype=np.float32,
        )
        self.observation_spaces: Dict[str, spaces.Space] = {
            a: self._obs_space for a in self.possible_agents
        }
        self.action_spaces: Dict[str, spaces.Space] = {
            a: self._act_space for a in self.possible_agents
        }

        # RNG
        self.np_random: np.random.Generator = np.random.default_rng()

        # Dynamic state arrays (initialized in reset)
        self.player_pos: np.ndarray = np.zeros((self.num_players, 2), dtype=np.float64)
        self.player_vel: np.ndarray = np.zeros((self.num_players, 2), dtype=np.float64)
        self.player_angle: np.ndarray = np.zeros(self.num_players, dtype=np.float64)

        self.stun_timer: np.ndarray = np.zeros(self.num_players, dtype=np.float64)
        self.ghost_timer: np.ndarray = np.zeros(self.num_players, dtype=np.float64)
        self.tackle_cooldown: np.ndarray = np.zeros(self.num_players, dtype=np.float64)
        self.lunge_cooldown: np.ndarray = np.zeros(self.num_players, dtype=np.float64)
        self.lunge_timer: np.ndarray = np.zeros(self.num_players, dtype=np.float64)
        self.pickup_cooldown: np.ndarray = np.zeros(self.num_players, dtype=np.float64)

        # Flags tracking whether a player's pickup_cooldown came from a kick or violation
        self.kick_cooldown_active: np.ndarray = np.zeros(self.num_players, dtype=bool)
        self.violation_cooldown_active: np.ndarray = np.zeros(self.num_players, dtype=bool)

        # Anti-farming and shot-on-target tracking
        self.last_keeper_kick_time: np.ndarray = np.full(2, -999.0, dtype=np.float64)
        self.shot_on_target_cd: np.ndarray = np.zeros(self.num_players, dtype=np.float64)
        self.popup_cooldown: np.ndarray = np.zeros(self.num_players, dtype=np.float64)
        self.last_step_rewards: np.ndarray = np.zeros(self.num_players, dtype=np.float32)
        self.cum_player_rewards: np.ndarray = np.zeros(self.num_players, dtype=np.float32)
        self.ball_kick_was_shot_on_target: bool = False

        # Ball state
        self.ball_pos: np.ndarray = np.zeros(2, dtype=np.float64)
        self.ball_vel: np.ndarray = np.zeros(2, dtype=np.float64)
        self.ball_holder: Optional[int] = None
        self.ball_hold_timer: float = 0.0
        self.kick_grace_timer: float = 0.0
        self.box_loose_ball_timer: float = 0.0
        self.last_ball_holder: Optional[int] = None
        self.last_ball_team: Optional[int] = None
        self.ball_kick_origin: np.ndarray = np.zeros(2, dtype=np.float64)
        self.ball_kicked_in_flight: bool = False
        self.ball_kick_flight_timer: float = 0.0
        self.pinned_ball_timer: float = 0.0
        self.kickoff_protection_timer: float = 0.0
        self.prev_longitudinal_vel: np.ndarray = np.zeros(self.num_players, dtype=np.float64)
        self.prev_ball_x: float = 700.0

        # Match progression & side assignment
        self.rl_step_count: int = 0
        self.physics_frame_count: int = 0
        self.current_half: int = 1
        self.first_half_kickoff_team: int = TEAM_A
        # team_side[team_id] -> SIDE_LEFT (0) or SIDE_RIGHT (1)
        self.team_side: np.ndarray = np.array([SIDE_LEFT, SIDE_RIGHT], dtype=np.int32)

        # Statistics & replay buffer
        self.stats: Dict[str, Any] = {}
        self.replay_frames: List[Dict[str, Any]] = []
        self._frame_events: List[Dict[str, Any]] = []
        self.last_saved_replay_path: Optional[str] = None
        self.current_generation: int = 0
        self.current_match_id: int = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        return self._obs_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        return self._act_space

    def _is_field_role(self, role: int) -> bool:
        return int(role) != ROLE_KEEPER

    def _team_keeper_idx(self, team: int) -> int:
        return 0 if team == TEAM_A else 6

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
            return (mech.def_tackle_range, mech.def_tackle_success_prob)
        elif role == ROLE_MID:
            return (mech.mid_tackle_range, mech.mid_tackle_success_prob)
        elif role == ROLE_ATK:
            return (mech.atk_tackle_range, mech.atk_tackle_success_prob)
        else:
            return (mech.tackle_range, mech.tackle_success_prob)

    @staticmethod
    def _point_to_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        d2 = float(np.dot(ab, ab))
        if d2 < 1e-12:
            return float(np.linalg.norm(p - a))
        t = float(np.clip(np.dot(p - a, ab) / d2, 0.0, 1.0))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    def _is_ball_heading_into_goal(
        self, ball_pos: np.ndarray, ball_vel: np.ndarray, target_side: int
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
        max_reach = (float(np.linalg.norm(ball_vel)) ** 2) / (2.0 * phys.ball_friction_decel) + 35.0
        return bool(
            dist_to_goal_line <= max_reach
            and (phys.goal_y_min - 18.0) <= y_intersect <= (phys.goal_y_max + 18.0)
        )

    # ==========================================================================
    # Core PettingZoo API: reset & step
    # ==========================================================================
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.agents = list(self.possible_agents)
        self.rl_step_count = 0
        self.physics_frame_count = 0
        self.current_half = 1
        self.last_saved_replay_path = None

        # In the 1st half, Team A defends Left (X=0), Team B defends Right (X=1400)
        self.team_side = np.array([SIDE_LEFT, SIDE_RIGHT], dtype=np.int32)

        if options and "kickoff_team" in options:
            self.first_half_kickoff_team = int(options["kickoff_team"])
        else:
            self.first_half_kickoff_team = TEAM_A

        if options and "generation" in options:
            self.current_generation = int(options["generation"])
        if options and "match_id" in options:
            self.current_match_id = int(options["match_id"])

        # Initialize match stats
        self.stats = {
            "score": [0, 0],
            "goals": 0,
            "own_goals": 0,
            "completed_passes": 0,
            "team_passes": [0, 0],
            "interceptions": 0,
            "successful_tackles": 0,
            "team_tackles": [0, 0],
            "failed_tackles": 0,
            "keeper_saves": 0,
            "team_saves": [0, 0],
            "shots": [0, 0],
            "field_3s_violations": 0,
            "keeper_5s_violations": 0,
            "possession_frames": [0, 0],
        }

        self.replay_frames = []
        self._frame_events = []
        self.last_keeper_kick_time.fill(-999.0)
        self.shot_on_target_cd.fill(0.0)
        self.popup_cooldown.fill(0.0)
        self.last_step_rewards.fill(0.0)
        self.cum_player_rewards.fill(0.0)
        self.ball_kick_was_shot_on_target = False
        self.ball_kick_flight_timer = 0.0
        self.pinned_ball_timer = 0.0
        self.kickoff_protection_timer = 0.0
        self.prev_longitudinal_vel[:] = 0.0

        # Set initial kickoff formation
        self._setup_kickoff(starting_team=self.first_half_kickoff_team)

        # Record initial frame (t = 0.0s) if recording enabled
        if self.cfg.replay.enable_recording:
            self._record_replay_frame()

        observations = self._get_observations()
        infos = {agent: self._build_agent_info(agent) for agent in self.agents}
        return observations, infos

    def step(
        self, actions: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        if not self.agents:
            return {}, {}, {}, {}, {}

        # Parse and validate actions into (12, 4) float64 array
        act_array = np.zeros((self.num_players, self.cfg.vision.action_dim), dtype=np.float64)
        for agent_name, act in actions.items():
            if agent_name in self.agent_to_idx:
                idx = self.agent_to_idx[agent_name]
                act_array[idx] = np.clip(np.asarray(act, dtype=np.float64), -1.0, 1.0)

        step_rewards = np.zeros(self.num_players, dtype=np.float64)
        net_goal_advance = np.zeros(2, dtype=np.float64)
        prev_player_pos = self.player_pos.copy()
        prev_player_ball_dists = np.linalg.norm(
            self.player_pos - self.ball_pos[None, :], axis=1
        )
        any_teleport = False

        # Continuous per-step penalty for holding the tackle button in empty space (0% chance of tackle)
        mech = self.cfg.mechanics
        rew = self.cfg.rewards
        holder = self.ball_holder
        for idx in range(self.num_players):
            if self._is_field_role(self.player_role[idx]) and act_array[idx, 3] > mech.special_trigger_threshold:
                valid_opp_near = (
                    holder is not None
                    and self.player_team[holder] != self.player_team[idx]
                    and self._is_field_role(self.player_role[holder])
                    and float(np.linalg.norm(self.player_pos[idx] - self.player_pos[holder]))
                    <= mech.tackle_near_range
                )
                if not valid_opp_near:
                    step_rewards[idx] += rew.reckless_tackle_spam_step_pen

        # Run 4 physics sub-steps (60 Hz physics / 15 Hz RL)
        for _ in range(self.cfg.physics.frame_skip):
            self._frame_events = []
            prev_ball = self.ball_pos.copy()
            teleported = self._physics_substep(act_array, step_rewards)
            if teleported:
                any_teleport = True
            else:
                for team_id in (TEAM_A, TEAM_B):
                    enemy_goal = np.array(
                        self.cfg.physics.right_goal_center
                        if self.team_side[team_id] == SIDE_LEFT
                        else self.cfg.physics.left_goal_center,
                        dtype=np.float64,
                    )
                    d_before = float(np.linalg.norm(prev_ball - enemy_goal))
                    d_after = float(np.linalg.norm(self.ball_pos - enemy_goal))
                    net_goal_advance[team_id] += d_before - d_after
            self.prev_ball_x = float(self.ball_pos[0])

            self.physics_frame_count += 1

            # Check for Half-Time at exact frame 5400 (t = 90.0s)
            if (
                self.physics_frame_count == self.cfg.physics.half_physics_frames
                and self.current_half == 1
            ):
                self._handle_halftime()
                any_teleport = True

            if self.cfg.replay.enable_recording:
                self._record_replay_frame()

        self.rl_step_count += 1

        # Dense per-RL-step rewards:
        # 1. Possession Tick + Individual Carrier Goal-Advance Bonus
        if self.ball_holder is not None:
            carrier_idx = self.ball_holder
            carrier_team = int(self.player_team[carrier_idx])
            step_rewards[carrier_idx] += self.cfg.rewards.possession_carrier_ind
            self._add_team_reward(step_rewards, carrier_team, self.cfg.rewards.possession_team)
            carrier_adv = float(net_goal_advance[carrier_team])
            if carrier_adv > 0.0:
                step_rewards[carrier_idx] += (
                    self.cfg.rewards.carrier_goal_advance_ind_scale * carrier_adv
                )

            # Prime-Scoring-Zone Hesitation Penalty
            if self._is_field_role(self.player_role[carrier_idx]) and self.kick_grace_timer <= 0.0:
                enemy_goal_c = (
                    np.array(self.cfg.physics.right_goal_center, dtype=np.float64)
                    if self.team_side[carrier_team] == SIDE_LEFT
                    else np.array(self.cfg.physics.left_goal_center, dtype=np.float64)
                )
                dist_to_enemy_goal = float(
                    np.linalg.norm(self.player_pos[carrier_idx] - enemy_goal_c)
                )
                if dist_to_enemy_goal <= 320.0:
                    enemy_side = 1 - int(self.team_side[carrier_team])
                    max_kick_spd, _ = self._get_role_kick_params(carrier_idx)
                    facing_vec = np.array(
                        [
                            math.cos(self.player_angle[carrier_idx]),
                            math.sin(self.player_angle[carrier_idx]),
                        ],
                        dtype=np.float64,
                    )
                    hyp_ball_pos = (
                        self.player_pos[carrier_idx]
                        + facing_vec * self.cfg.mechanics.ball_carry_offset
                    )
                    hyp_ball_vel = facing_vec * max_kick_spd
                    if self._is_ball_heading_into_goal(hyp_ball_pos, hyp_ball_vel, enemy_side):
                        step_rewards[carrier_idx] += self.cfg.rewards.prime_shot_hesitation_pen

        # 2. Dense Ball-Advance Shaping towards Enemy Goal Center (1400, 420) / (0, 420)
        for team_id in (TEAM_A, TEAM_B):
            adv = float(net_goal_advance[team_id])
            if adv < 0.0 and self.ball_kicked_in_flight and self.last_ball_team == team_id:
                continue
            if abs(adv) > 1e-9:
                self._add_team_reward(
                    step_rewards,
                    team_id,
                    self.cfg.rewards.ball_advance_team_scale * adv,
                )

        # 2b. Anti-Jitter Velocity-Reversal Shaping
        if not any_teleport:
            for idx in range(self.num_players):
                long_v = float(
                    self.player_vel[idx, 0] * math.cos(self.player_angle[idx])
                    + self.player_vel[idx, 1] * math.sin(self.player_angle[idx])
                )
                if long_v * self.prev_longitudinal_vel[idx] < -400.0:
                    step_rewards[idx] += self.cfg.rewards.jitter_reverse_pen
                self.prev_longitudinal_vel[idx] = long_v
        else:
            self.prev_longitudinal_vel[:] = 0.0

        # 3. Context-Dependent Tactical Positioning (Spread Out / Loose Ball Chase / Press & Cover / Keeper Arc)
        self._compute_tactical_positioning_rewards(
            step_rewards, prev_player_pos, prev_player_ball_dists, any_teleport
        )

        # Update reward tracking arrays for visual feedback & replay recording
        self.last_step_rewards = step_rewards.astype(np.float32)
        self.cum_player_rewards += self.last_step_rewards
        if self.cfg.replay.enable_recording and len(self.replay_frames) > 0:
            p_rews = [round(float(r), 4) for r in self.last_step_rewards]
            c_rews = [round(float(r), 2) for r in self.cum_player_rewards]
            start_back = max(0, len(self.replay_frames) - self.cfg.physics.frame_skip)
            for f_idx in range(start_back, len(self.replay_frames)):
                self.replay_frames[f_idx]["player_rewards"] = p_rews
                self.replay_frames[f_idx]["cum_rewards"] = c_rews

        # Episode termination / truncation at t = 180.0s (2,700 RL steps)
        is_truncated = self.rl_step_count >= self.cfg.physics.max_rl_steps
        is_terminated = False

        observations = self._get_observations()
        rewards = {
            agent: float(step_rewards[self.agent_to_idx[agent]]) for agent in self.agents
        }
        terminations = {agent: is_terminated for agent in self.agents}
        truncations = {agent: is_truncated for agent in self.agents}

        if is_truncated or is_terminated:
            self._emit_event(
                "FULLTIME!",
                np.array(self.cfg.physics.field_center, dtype=np.float64),
                team=-1,
                agent="referee",
            )
            if self.cfg.replay.enable_recording and len(self.replay_frames) > 0:
                self.replay_frames[-1]["events"].append(
                    {
                        "type": "FULLTIME!",
                        "x": float(self.cfg.physics.field_center[0]),
                        "y": float(self.cfg.physics.field_center[1]),
                        "team": -1,
                        "agent": "referee",
                        "reward": None,
                    }
                )
            if self.cfg.replay.enable_recording:
                self._maybe_export_replay()

        infos = {agent: self._build_agent_info(agent) for agent in self.agents}

        if is_truncated or is_terminated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

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
        self.stun_timer = np.maximum(0.0, self.stun_timer - dt)
        self.ghost_timer = np.maximum(0.0, self.ghost_timer - dt)
        self.tackle_cooldown = np.maximum(0.0, self.tackle_cooldown - dt)
        self.lunge_cooldown = np.maximum(0.0, self.lunge_cooldown - dt)
        self.lunge_timer = np.maximum(0.0, self.lunge_timer - dt)
        self.pickup_cooldown = np.maximum(0.0, self.pickup_cooldown - dt)
        self.kick_grace_timer = max(0.0, self.kick_grace_timer - dt)
        self.shot_on_target_cd = np.maximum(0.0, self.shot_on_target_cd - dt)
        self.popup_cooldown = np.maximum(0.0, self.popup_cooldown - dt)
        self.kickoff_protection_timer = max(0.0, self.kickoff_protection_timer - dt)

        # Clear tracking flags when cooldown naturally expires
        expired_mask = self.pickup_cooldown <= 0.0
        self.kick_cooldown_active[expired_mask] = False
        self.violation_cooldown_active[expired_mask] = False

        # Track possession stats & hold timer
        if self.ball_holder is not None:
            self.box_loose_ball_timer = 0.0
            holder_team = int(self.player_team[self.ball_holder])
            self.stats["possession_frames"][holder_team] += 1
            self.ball_hold_timer += dt

            # Check 3-second Field Player Violation or 5-second Keeper Violation
            holder_role = int(self.player_role[self.ball_holder])
            if (
                self._is_field_role(holder_role)
                and self.ball_hold_timer >= mech.field_max_hold_time - 1e-9
            ):
                self._handle_field_3s_violation(self.ball_holder, step_rewards)
            elif (
                holder_role == ROLE_KEEPER
                and self.ball_hold_timer >= mech.keeper_max_hold_time - 1e-9
            ):
                self._handle_keeper_5s_violation(self.ball_holder, step_rewards)
                return True
        else:
            # Check if loose ball is sitting unclaimed inside either Keeper's semicircle (dist <= 145 + r_ball)
            left_center = np.array(phys.left_goal_center, dtype=np.float64)
            right_center = np.array(phys.right_goal_center, dtype=np.float64)
            d_left = float(np.linalg.norm(self.ball_pos - left_center))
            d_right = float(np.linalg.norm(self.ball_pos - right_center))
            box_limit = phys.keeper_box_radius + phys.ball_radius

            if d_left <= box_limit or d_right <= box_limit:
                self.box_loose_ball_timer += dt
                if self.box_loose_ball_timer >= mech.keeper_box_loose_max_time - 1e-9:
                    box_side = SIDE_LEFT if d_left <= box_limit else SIDE_RIGHT
                    penalized_team = int(np.where(self.team_side == box_side)[0][0])
                    keeper_idx = self._team_keeper_idx(penalized_team)
                    self._handle_keeper_5s_violation(keeper_idx, step_rewards)
                    return True
            else:
                self.box_loose_ball_timer = 0.0

        # ----------------------------------------------------------------------
        # 2. Process Player Actions (Steering, Throttle, Kick, Special)
        # ----------------------------------------------------------------------
        current_holder = self.ball_holder

        for idx in range(self.num_players):
            if self.stun_timer[idx] > 0.0:
                self.player_vel[idx] = 0.0
                continue

            throttle_raw, steer_raw, kick_power, special_trig = act_array[idx]

            # Steering: [-1, 1] rotates facing_angle
            turn_delta = steer_raw * phys.player_max_turn_rate * dt
            self.player_angle[idx] = float(wrap_to_pi(self.player_angle[idx] + turn_delta))

            # Special Action (action[3] > 0.5): Lunge for Keepers, Tackle for Field Players
            if special_trig > mech.special_trigger_threshold:
                if self.player_role[idx] == ROLE_KEEPER:
                    if self.ball_holder != idx and self.lunge_cooldown[idx] <= 0.0:
                        self.lunge_timer[idx] = mech.lunge_duration
                        self.lunge_cooldown[idx] = mech.lunge_cooldown
                        d_to_ball = float(np.linalg.norm(self.player_pos[idx] - self.ball_pos))
                        if (
                            self.ball_kicked_in_flight
                            and self.ball_kick_was_shot_on_target
                            and self.last_ball_team != self.player_team[idx]
                            and d_to_ball <= 450.0
                        ):
                            if step_rewards is not None:
                                step_rewards[idx] += self.cfg.rewards.keeper_dive_save_attempt_ind
                            self._emit_event(
                                "GK DIVE!",
                                self.player_pos[idx],
                                team=int(self.player_team[idx]),
                                agent=self.idx_to_agent[idx],
                                reward=self.cfg.rewards.keeper_dive_save_attempt_ind,
                            )
                        elif d_to_ball > 450.0 and self.ball_holder != idx:
                            if step_rewards is not None:
                                step_rewards[idx] += self.cfg.rewards.keeper_reckless_lunge_pen
                else:
                    # Field Player Tackle attempt
                    if (
                        self.ball_holder != idx
                        and self.tackle_cooldown[idx] <= 0.0
                        and self.ghost_timer[idx] <= 0.0
                    ):
                        self._attempt_tackle(idx, step_rewards)

            # Forward / Backpedal Velocity with Asymmetric Deadzone & Exponential Velocity Inertia:
            # Lunge overrides normal throttle speed for Keepers
            if self.player_role[idx] == ROLE_KEEPER and self.lunge_timer[idx] > 0.0:
                target_speed = mech.lunge_speed
                accel_alpha = 1.0
            else:
                dz_pos = phys.throttle_deadzone
                dz_neg = phys.backpedal_threshold
                max_spd = self._get_player_max_speed(idx)
                if dz_neg <= throttle_raw <= dz_pos:
                    target_speed = 0.0
                elif throttle_raw > dz_pos:
                    throttle_norm = (throttle_raw - dz_pos) / max(1e-6, 1.0 - dz_pos)
                    target_speed = throttle_norm * max_spd
                else:
                    back_norm = (throttle_raw - dz_neg) / max(1e-6, 1.0 + dz_neg)  # in [-1.0, 0.0)
                    back_mult = (
                        phys.keeper_backpedal_mult
                        if self.player_role[idx] == ROLE_KEEPER
                        else phys.field_backpedal_mult
                    )
                    target_speed = back_norm * back_mult * max_spd
                accel_alpha = min(1.0, dt * phys.player_accel_rate)

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
        self._clamp_players_to_boundaries()

        # Re-run collision & boundary pass once more to guarantee hard constraints after pushing
        self._clamp_players_to_boundaries()

        # ----------------------------------------------------------------------
        # 4. Update Ball Kinematics & Check Goals / Wall Bounces
        # ----------------------------------------------------------------------
        if self.ball_holder is not None:
            self.pinned_ball_timer = 0.0
            self.ball_kick_flight_timer = 0.0
            # Lock ball to front edge of carrier: player_pos + facing_vec * 24.0
            h_idx = self.ball_holder
            facing_vec = np.array(
                [math.cos(self.player_angle[h_idx]), math.sin(self.player_angle[h_idx])],
                dtype=np.float64,
            )
            self.ball_pos = self.player_pos[h_idx] + facing_vec * mech.ball_carry_offset
            self.ball_vel[:] = self.player_vel[h_idx]

            # Prevent a Goalkeeper holding the ball in their hands from protruding the ball into their OWN goal
            if self.player_role[h_idx] == ROLE_KEEPER:
                k_team = int(self.player_team[h_idx])
                if self.team_side[k_team] == SIDE_LEFT:
                    self.ball_pos[0] = max(float(self.ball_pos[0]), phys.ball_radius)
                else:
                    self.ball_pos[0] = min(
                        float(self.ball_pos[0]), phys.field_width - phys.ball_radius
                    )

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

            # 3-Second Pass Flight Timeout
            if self.ball_kicked_in_flight:
                self.ball_kick_flight_timer += dt
                if self.ball_kick_flight_timer > mech.max_pass_flight_time:
                    self.ball_kicked_in_flight = False
                    self.ball_kick_was_shot_on_target = False

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
            near_box_edge = (
                abs(d_left_box - phys.keeper_box_radius) <= 24.0
                or abs(d_right_box - phys.keeper_box_radius) <= 24.0
            )
            players_crowding_ball = int(
                np.sum(np.linalg.norm(self.player_pos - self.ball_pos[None, :], axis=1) <= 68.0)
            )
            if (
                ball_speed_now < mech.pinned_ball_speed_thresh
                and (near_wall or near_box_edge)
                and players_crowding_ball >= 2
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
    # Mechanics Helpers: Kicks, Tackles, Violations, Pickups, Collisions
    # ==========================================================================
    def _execute_kick(
        self,
        kicker_idx: int,
        kick_power: float,
        step_rewards: Optional[np.ndarray] = None,
    ) -> None:
        """Launch the ball from kicker with role-specific speed & jitter, check shot-on-target, and apply cooldown."""
        self.kickoff_protection_timer = 0.0
        phys = self.cfg.physics
        mech = self.cfg.mechanics
        rew = self.cfg.rewards

        kicker_role = int(self.player_role[kicker_idx])
        kicker_team = int(self.player_team[kicker_idx])

        power_norm = float(
            np.clip(
                (kick_power - mech.kick_trigger_threshold)
                / max(1e-6, 1.0 - mech.kick_trigger_threshold),
                0.0,
                1.0,
            )
        )
        max_kick_spd, jitter_rad = self._get_role_kick_params(kicker_idx)
        launch_speed = phys.ball_min_kick_speed + power_norm * (max_kick_spd - phys.ball_min_kick_speed)

        base_angle = float(self.player_angle[kicker_idx])
        if power_norm > 0.65 and jitter_rad > 0.0:
            kick_angle = base_angle + float(self.np_random.uniform(-jitter_rad, jitter_rad))
        else:
            kick_angle = base_angle

        # GK Forward-Hemisphere Distribution Clamp
        if kicker_role == ROLE_KEEPER:
            cos_forward = (
                math.cos(base_angle)
                if self.team_side[kicker_team] == SIDE_LEFT
                else -math.cos(base_angle)
            )
            if cos_forward < 0.15:
                if step_rewards is not None:
                    step_rewards[kicker_idx] += rew.keeper_kick_backward_pen
                    self._emit_event(
                        "GK BACK-KICK!",
                        self.player_pos[kicker_idx],
                        team=kicker_team,
                        agent=self.idx_to_agent[kicker_idx],
                        reward=rew.keeper_kick_backward_pen,
                    )
                sin_sign = 1.0 if math.sin(base_angle) >= 0.0 else -1.0
                theta = math.acos(0.35)
                if self.team_side[kicker_team] == SIDE_LEFT:
                    base_angle = sin_sign * theta
                else:
                    base_angle = float(wrap_to_pi(math.pi - sin_sign * theta))
                kick_angle = base_angle

        facing_vec = np.array([math.cos(base_angle), math.sin(base_angle)], dtype=np.float64)
        kick_vec = np.array([math.cos(kick_angle), math.sin(kick_angle)], dtype=np.float64)

        self.ball_pos = self.player_pos[kicker_idx] + facing_vec * mech.ball_carry_offset
        self.ball_vel = kick_vec * launch_speed

        self.ball_holder = None
        self.ball_hold_timer = 0.0
        self.last_ball_holder = kicker_idx
        self.last_ball_team = kicker_team
        self.ball_kick_origin = self.ball_pos.copy()
        self.ball_kicked_in_flight = True
        self.ball_kick_flight_timer = 0.0

        # 1.0s pickup cooldown on kicker
        self.pickup_cooldown[kicker_idx] = max(
            self.pickup_cooldown[kicker_idx], mech.kick_pickup_cooldown
        )
        self.kick_cooldown_active[kicker_idx] = True

        if kicker_role == ROLE_KEEPER:
            self.last_keeper_kick_time[kicker_team] = self.physics_frame_count * phys.dt

        # Check if this kick is a Shot on Target against the enemy goal
        enemy_side = 1 - int(self.team_side[kicker_team])
        is_sot = self._is_ball_heading_into_goal(self.ball_pos, self.ball_vel, enemy_side)
        self.ball_kick_was_shot_on_target = is_sot

        if is_sot:
            self.stats["shots"][kicker_team] += 1
            if step_rewards is not None and self.shot_on_target_cd[kicker_idx] <= 0.0:
                c_kicker_pos, _, _ = self._to_team_canonical(
                    self.player_pos[kicker_idx:kicker_idx + 1],
                    self.player_vel[kicker_idx:kicker_idx + 1],
                    self.player_angle[kicker_idx],
                    kicker_team,
                )
                cx = float(c_kicker_pos[0, 0])
                if cx >= 680.0:
                    prime_zone_mult = 1.0 if cx >= 980.0 else 0.55
                    power_factor = 0.6 + 0.4 * power_norm
                    base_sot = (
                        rew.atk_shot_on_target_ind
                        if kicker_role == ROLE_ATK
                        else rew.other_shot_on_target_ind
                    )
                    sot_rew = base_sot * prime_zone_mult * power_factor
                    team_sot_rew = rew.shot_on_target_team * prime_zone_mult
                    step_rewards[kicker_idx] += sot_rew
                    self._add_team_reward(step_rewards, kicker_team, team_sot_rew)
                    self.shot_on_target_cd[kicker_idx] = 1.5
                    self._emit_event(
                        "SHOT ON TARGET!",
                        self.player_pos[kicker_idx],
                        team=kicker_team,
                        agent=self.idx_to_agent[kicker_idx],
                        reward=sot_rew + team_sot_rew,
                    )

        # Keeper distribution / clearance forward bonus
        if step_rewards is not None and kicker_role == ROLE_KEEPER:
            cos_fwd = (
                float(facing_vec[0])
                if self.team_side[kicker_team] == SIDE_LEFT
                else float(-facing_vec[0])
            )
            if cos_fwd > 0.25:
                aim_bonus = 1.0
                for j in range(self.num_players):
                    if j != kicker_idx and self.player_team[j] == kicker_team:
                        to_tm = self.player_pos[j] - self.player_pos[kicker_idx]
                        d_tm = float(np.linalg.norm(to_tm))
                        if d_tm > 40.0 and float(np.dot(facing_vec, to_tm / d_tm)) >= 0.85:
                            aim_bonus = 1.5
                            break
                step_rewards[kicker_idx] += (
                    rew.keeper_clear_forward_ind
                    * cos_fwd
                    * (0.5 + 0.5 * power_norm)
                    * aim_bonus
                )

        # Immediate Kick-Direction Evaluation (for all players when step_rewards is not None)
        if step_rewards is not None:
            opp_team = 1 - kicker_team
            opp_indices = [idx for idx in range(self.num_players) if self.player_team[idx] == opp_team]

            # Check if any opponent is standing directly in the kick path (within 110u and cos >= 0.92)
            is_blocked = False
            for opp_idx in opp_indices:
                to_opp = self.player_pos[opp_idx] - self.ball_pos
                d_opp = float(np.linalg.norm(to_opp))
                if 0.0 < d_opp <= 110.0:
                    if float(np.dot(kick_vec, to_opp / d_opp)) >= 0.92:
                        is_blocked = True
                        break

            if is_blocked:
                step_rewards[kicker_idx] += rew.blocked_kick_pen
            else:
                # Check if kick_vec is aimed at an open teammate
                aimed_at_teammate = False
                for tm_idx in range(self.num_players):
                    if tm_idx != kicker_idx and self.player_team[tm_idx] == kicker_team:
                        to_tm = self.player_pos[tm_idx] - self.ball_pos
                        d_tm = float(np.linalg.norm(to_tm))
                        if 75.0 <= d_tm <= 560.0:
                            unit_tm = to_tm / d_tm
                            if float(np.dot(kick_vec, unit_tm)) >= 0.93:
                                lane_clear = True
                                for opp_idx in opp_indices:
                                    to_opp = self.player_pos[opp_idx] - self.ball_pos
                                    proj = float(np.dot(to_opp, unit_tm))
                                    if 0.0 < proj < d_tm:
                                        perp_dist = math.hypot(
                                            to_opp[0] - proj * unit_tm[0],
                                            to_opp[1] - proj * unit_tm[1],
                                        )
                                        if perp_dist < 36.0:
                                            lane_clear = False
                                            break
                                if lane_clear:
                                    aimed_at_teammate = True
                                    break

                if aimed_at_teammate:
                    step_rewards[kicker_idx] += rew.pass_target_aim_ind
                elif not is_sot:
                    # Open-Space Clearance check
                    nearest_opp_dist = min(
                        (
                            float(np.linalg.norm(self.player_pos[opp_idx] - self.player_pos[kicker_idx]))
                            for opp_idx in opp_indices
                        ),
                        default=999.0,
                    )
                    if self.ball_hold_timer >= 2.2 or nearest_opp_dist <= 95.0:
                        landing_pt = self.ball_pos + kick_vec * 260.0
                        landing_pt[0] = np.clip(landing_pt[0], 40.0, 1360.0)
                        landing_pt[1] = np.clip(landing_pt[1], 40.0, 800.0)

                        my_side = int(self.team_side[kicker_team])
                        own_goal_c = np.array(
                            phys.left_goal_center if my_side == SIDE_LEFT else phys.right_goal_center,
                            dtype=np.float64,
                        )
                        in_own_box = float(np.linalg.norm(landing_pt - own_goal_c)) <= phys.keeper_box_radius

                        if not in_own_box:
                            nearest_opp_to_landing = min(
                                (
                                    float(np.linalg.norm(self.player_pos[opp_idx] - landing_pt))
                                    for opp_idx in opp_indices
                                ),
                                default=999.0,
                            )
                            if nearest_opp_to_landing >= 120.0:
                                step_rewards[kicker_idx] += rew.open_clearance_ind

    def _attempt_tackle(self, tackler_idx: int, step_rewards: np.ndarray) -> None:
        """Execute role-specific tackle check when action[3] > 0.5."""
        if self.kickoff_protection_timer > 0.0:
            return

        mech = self.cfg.mechanics
        rew = self.cfg.rewards

        tackle_range, tackle_prob = self._get_role_tackle_params(tackler_idx)

        # Valid target: an opponent field player currently holding the ball
        holder = self.ball_holder
        if (
            holder is not None
            and self.player_team[holder] != self.player_team[tackler_idx]
            and self._is_field_role(self.player_role[holder])
        ):
            dist = float(np.linalg.norm(self.player_pos[tackler_idx] - self.player_pos[holder]))
            if dist <= tackle_range:
                if self.np_random.random() < tackle_prob:
                    # Successful tackle!
                    old_carrier = holder
                    self.ball_holder = tackler_idx
                    self.ball_hold_timer = 0.0
                    self.kick_grace_timer = mech.pickup_kick_grace
                    self.last_ball_holder = tackler_idx
                    self.last_ball_team = int(self.player_team[tackler_idx])
                    self.ball_kicked_in_flight = False

                    self._on_new_player_touched_ball(tackler_idx)

                    self.stun_timer[old_carrier] = mech.tackle_stun_duration
                    self.player_vel[old_carrier] = 0.0
                    self.tackle_cooldown[old_carrier] = max(
                        self.tackle_cooldown[old_carrier], mech.tackled_opponent_cooldown
                    )
                    self.pickup_cooldown[old_carrier] = max(
                        self.pickup_cooldown[old_carrier], mech.tackle_stun_duration
                    )
                    self.tackle_cooldown[tackler_idx] = mech.tackle_stun_duration

                    facing_vec = np.array(
                        [
                            math.cos(self.player_angle[tackler_idx]),
                            math.sin(self.player_angle[tackler_idx]),
                        ],
                        dtype=np.float64,
                    )
                    self.ball_pos = (
                        self.player_pos[tackler_idx] + facing_vec * mech.ball_carry_offset
                    )
                    self.ball_vel[:] = self.player_vel[tackler_idx]

                    tackler_team = int(self.player_team[tackler_idx])
                    tackler_ind = rew.successful_tackle_tackler_ind

                    # Check Defender own half ball-winning bonus
                    if self.player_role[tackler_idx] == ROLE_DEF:
                        c_tackler_pos, _, _ = self._to_team_canonical(
                            self.player_pos[tackler_idx:tackler_idx + 1],
                            self.player_vel[tackler_idx:tackler_idx + 1],
                            0.0,
                            tackler_team,
                        )
                        if float(c_tackler_pos[0, 0]) < 700.0:
                            tackler_ind += rew.def_own_half_tackle_bonus

                    step_rewards[tackler_idx] += tackler_ind
                    step_rewards[old_carrier] += rew.successful_tackle_carrier_ind
                    self._add_team_reward(step_rewards, tackler_team, rew.successful_tackle_team)
                    self.stats["successful_tackles"] += 1
                    self.stats["team_tackles"][tackler_team] += 1
                    self._emit_event(
                        "TACKLE!",
                        self.player_pos[tackler_idx],
                        team=tackler_team,
                        agent=self.idx_to_agent[tackler_idx],
                        reward=tackler_ind + rew.successful_tackle_team,
                    )
                    return

                # RNG failure on valid in-range attempt
                self.tackle_cooldown[tackler_idx] = mech.tackle_fail_cooldown
                step_rewards[tackler_idx] += rew.failed_tackle_in_range_ind
                self.stats["failed_tackles"] += 1
                self._emit_event(
                    "TACKLE FAILED!",
                    self.player_pos[tackler_idx],
                    team=int(self.player_team[tackler_idx]),
                    agent=self.idx_to_agent[tackler_idx],
                    reward=rew.failed_tackle_in_range_ind,
                )
                return

            if dist <= mech.tackle_near_range:
                # Close miss outside reach
                self.tackle_cooldown[tackler_idx] = mech.tackle_fail_cooldown
                step_rewards[tackler_idx] += rew.failed_tackle_ind
                self.stats["failed_tackles"] += 1
                self._emit_event(
                    "TACKLE FAILED!",
                    self.player_pos[tackler_idx],
                    team=int(self.player_team[tackler_idx]),
                    agent=self.idx_to_agent[tackler_idx],
                    reward=rew.failed_tackle_ind,
                )
                return

        # Reckless Air Tackle
        self.tackle_cooldown[tackler_idx] = mech.tackle_fail_cooldown
        step_rewards[tackler_idx] += rew.reckless_tackle_ind
        self.stats["failed_tackles"] += 1
        self._emit_event(
            "AIR TACKLE!",
            self.player_pos[tackler_idx],
            team=int(self.player_team[tackler_idx]),
            agent=self.idx_to_agent[tackler_idx],
            reward=rew.reckless_tackle_ind,
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
            now_t = self.physics_frame_count * self.cfg.physics.dt
            recently_kicked_by_gk = (now_t - self.last_keeper_kick_time[opp_team]) < mech.keeper_strip_self_kick_cooldown

            self.ball_holder = keeper_idx
            self.ball_hold_timer = 0.0
            self.last_ball_holder = keeper_idx
            self.last_ball_team = opp_team
            self.ball_kicked_in_flight = False

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

            k_ang = float(self.player_angle[keeper_idx])
            if self.team_side[opp_team] == SIDE_LEFT and math.cos(k_ang) < 0.0:
                self.player_angle[keeper_idx] = float(
                    math.atan2(math.sin(k_ang), abs(math.cos(k_ang)))
                )
            elif self.team_side[opp_team] == SIDE_RIGHT and math.cos(k_ang) > 0.0:
                self.player_angle[keeper_idx] = float(
                    math.atan2(math.sin(k_ang), -abs(math.cos(k_ang)))
                )

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
                step_rewards[keeper_idx] += rew.keeper_strip_ind
                self._add_team_reward(step_rewards, opp_team, rew.keeper_strip_team)
                self._emit_event(
                    "GK STRIP!",
                    self.player_pos[keeper_idx],
                    team=opp_team,
                    agent=self.idx_to_agent[keeper_idx],
                    reward=rew.keeper_strip_ind + rew.keeper_strip_team,
                )

    def _check_free_ball_pickup(self, step_rewards: np.ndarray) -> None:
        """Check if any eligible player picks up the free ball."""
        if self.ball_holder is not None:
            return

        mech = self.cfg.mechanics
        rew = self.cfg.rewards

        dists = np.linalg.norm(self.player_pos - self.ball_pos[None, :], axis=1)
        eligible = np.where(
            (dists <= mech.pickup_distance)
            & (self.pickup_cooldown <= 0.0)
            & (self.stun_timer <= 0.0)
        )[0]

        if len(eligible) == 0:
            return

        new_holder = int(eligible[np.argmin(dists[eligible])])
        new_team = int(self.player_team[new_holder])
        new_role = int(self.player_role[new_holder])

        self._on_new_player_touched_ball(new_holder)

        if (
            self.ball_kicked_in_flight
            and self.ball_kick_flight_timer <= mech.max_pass_flight_time
            and self.last_ball_holder is not None
        ):
            kicker_idx = self.last_ball_holder
            kicker_team = int(self.last_ball_team) if self.last_ball_team is not None else new_team
            kick_travel_dist = float(np.linalg.norm(self.ball_pos - self.ball_kick_origin))

            if new_team == kicker_team and new_holder != kicker_idx:
                if kick_travel_dist >= mech.min_pass_distance:
                    # Completed pass
                    passer_bonus = rew.completed_pass_passer_ind
                    receiver_bonus = rew.completed_pass_receiver_ind
                    team_bonus = rew.completed_pass_team

                    kicker_role = int(self.player_role[kicker_idx])
                    receiver_role = int(self.player_role[new_holder])

                    if kicker_role == ROLE_DEF and receiver_role in (ROLE_MID, ROLE_ATK):
                        passer_bonus += rew.def_breakout_pass_bonus
                    if receiver_role == ROLE_MID and kicker_role in (ROLE_KEEPER, ROLE_DEF):
                        receiver_bonus += rew.mid_link_receive_bonus
                    if kicker_role == ROLE_MID and receiver_role == ROLE_ATK:
                        passer_bonus += rew.mid_to_atk_pass_bonus
                    if receiver_role == ROLE_ATK:
                        c_rx_pos, _, _ = self._to_team_canonical(
                            self.player_pos[new_holder:new_holder + 1],
                            self.player_vel[new_holder:new_holder + 1],
                            0.0,
                            new_team,
                        )
                        if float(c_rx_pos[0, 0]) >= 680.0:
                            passer_bonus += rew.pass_to_atk_bonus

                    c_orig_pos, _, _ = self._to_team_canonical(
                        self.ball_kick_origin.reshape(1, 2),
                        np.zeros((1, 2)),
                        0.0,
                        new_team,
                    )
                    c_ball_rx, _, _ = self._to_team_canonical(
                        self.ball_pos.reshape(1, 2),
                        np.zeros((1, 2)),
                        0.0,
                        new_team,
                    )
                    if float(c_ball_rx[0, 0]) - float(c_orig_pos[0, 0]) <= 40.0:
                        passer_bonus += rew.recycle_pass_bonus

                    step_rewards[kicker_idx] += passer_bonus
                    step_rewards[new_holder] += receiver_bonus
                    self._add_team_reward(step_rewards, new_team, team_bonus)
                    self.stats["completed_passes"] += 1
                    self.stats["team_passes"][new_team] += 1
                    self._emit_event(
                        "PASS!",
                        self.player_pos[new_holder],
                        team=new_team,
                        agent=self.idx_to_agent[new_holder],
                        reward=passer_bonus + receiver_bonus + team_bonus,
                    )
            elif new_team != kicker_team:
                # Interception or Keeper Save
                my_side = int(self.team_side[new_team])
                is_real_save = (
                    new_role == ROLE_KEEPER
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

                if not is_real_save:
                    step_rewards[kicker_idx] += rew.intercepted_kicker_ind
                    self._add_team_reward(step_rewards, kicker_team, rew.intercepted_team)
                    self.stats["interceptions"] += 1

                if new_role == ROLE_KEEPER:
                    if is_real_save:
                        step_rewards[new_holder] += rew.keeper_save_ind
                        self._add_team_reward(step_rewards, new_team, rew.keeper_save_team)
                        self.stats["keeper_saves"] += 1
                        self.stats["team_saves"][new_team] += 1
                        lbl = "LUNGE SAVE!" if self.lunge_timer[new_holder] > 0.0 else "KEEPER SAVE!"
                        self._emit_event(
                            lbl,
                            self.player_pos[new_holder],
                            team=new_team,
                            agent=self.idx_to_agent[new_holder],
                            reward=rew.keeper_save_ind + rew.keeper_save_team,
                        )
                    else:
                        step_rewards[new_holder] += rew.keeper_harmless_catch_ind
                        self._add_team_reward(step_rewards, new_team, rew.keeper_harmless_catch_team)
                else:
                    # Field player interceptor
                    intercept_rew = rew.loose_ball_pickup_ind
                    if new_role == ROLE_DEF:
                        c_rx_pos, _, _ = self._to_team_canonical(
                            self.player_pos[new_holder:new_holder + 1],
                            self.player_vel[new_holder:new_holder + 1],
                            0.0,
                            new_team,
                        )
                        if float(c_rx_pos[0, 0]) < 700.0:
                            intercept_rew += rew.def_own_half_intercept_bonus
                    step_rewards[new_holder] += intercept_rew
                    self._add_team_reward(step_rewards, new_team, rew.loose_ball_pickup_team)
        else:
            # Loose ball recovery
            step_rewards[new_holder] += rew.loose_ball_pickup_ind
            self._add_team_reward(step_rewards, new_team, rew.loose_ball_pickup_team)
            if new_role == ROLE_KEEPER:
                step_rewards[new_holder] += rew.keeper_box_recovery_ind

        # Assign possession to new_holder
        self.ball_holder = new_holder
        self.ball_hold_timer = 0.0
        self.box_loose_ball_timer = 0.0
        self.kick_grace_timer = mech.pickup_kick_grace
        self.last_ball_holder = new_holder
        self.last_ball_team = new_team
        self.ball_kicked_in_flight = False

        if new_role == ROLE_KEEPER:
            k_ang = float(self.player_angle[new_holder])
            if self.team_side[new_team] == SIDE_LEFT and math.cos(k_ang) < 0.0:
                self.player_angle[new_holder] = float(
                    math.atan2(math.sin(k_ang), abs(math.cos(k_ang)))
                )
            elif self.team_side[new_team] == SIDE_RIGHT and math.cos(k_ang) > 0.0:
                self.player_angle[new_holder] = float(
                    math.atan2(math.sin(k_ang), -abs(math.cos(k_ang)))
                )

        facing_vec = np.array(
            [
                math.cos(self.player_angle[new_holder]),
                math.sin(self.player_angle[new_holder]),
            ],
            dtype=np.float64,
        )
        self.ball_pos = self.player_pos[new_holder] + facing_vec * mech.ball_carry_offset
        if new_role == ROLE_KEEPER:
            phys = self.cfg.physics
            if self.team_side[new_team] == SIDE_LEFT:
                self.ball_pos[0] = max(float(self.ball_pos[0]), phys.ball_radius)
            else:
                self.ball_pos[0] = min(
                    float(self.ball_pos[0]), phys.field_width - phys.ball_radius
                )
        self.ball_vel[:] = self.player_vel[new_holder]

    def _on_new_player_touched_ball(self, toucher_idx: int) -> None:
        """Reset violation and kick pickup cooldowns for all other players when toucher grabs ball."""
        toucher_team = int(self.player_team[toucher_idx])
        kick_dist = float(np.linalg.norm(self.ball_pos - self.ball_kick_origin))
        for idx in range(self.num_players):
            if idx == toucher_idx:
                continue
            if self.violation_cooldown_active[idx]:
                self.pickup_cooldown[idx] = 0.0
                self.violation_cooldown_active[idx] = False
            if self.kick_cooldown_active[idx]:
                if int(self.player_team[idx]) != toucher_team or kick_dist >= 30.0:
                    self.pickup_cooldown[idx] = 0.0
                    self.kick_cooldown_active[idx] = False

    def _handle_field_3s_violation(
        self, violator_idx: int, step_rewards: np.ndarray
    ) -> None:
        """Enforce Field Player 3-Second Hold Violation: drop ball and enter Ghost Mode for 3.0s."""
        mech = self.cfg.mechanics
        rew = self.cfg.rewards
        team = int(self.player_team[violator_idx])

        self.ball_holder = None
        self.ball_hold_timer = 0.0
        self.ball_vel[:] = 0.0
        self.ball_kicked_in_flight = False

        self.ghost_timer[violator_idx] = mech.ghost_duration
        self.pickup_cooldown[violator_idx] = mech.field_violation_pickup_cooldown
        self.tackle_cooldown[violator_idx] = max(
            self.tackle_cooldown[violator_idx], mech.field_violation_tackle_cooldown
        )
        self.violation_cooldown_active[violator_idx] = True
        self.kick_cooldown_active[violator_idx] = False

        step_rewards[violator_idx] += rew.field_3s_violation_ind
        self._add_team_reward(step_rewards, team, rew.field_3s_violation_team)
        self.stats["field_3s_violations"] += 1
        self._emit_event(
            "3s VIOLATION!",
            self.player_pos[violator_idx],
            team=team,
            agent=self.idx_to_agent[violator_idx],
            reward=rew.field_3s_violation_ind + rew.field_3s_violation_team,
        )

    def _handle_keeper_5s_violation(
        self, keeper_idx: int, step_rewards: np.ndarray
    ) -> None:
        """Enforce Keeper 5-Second Hold Violation: reward penalty + Foul Teleport Reset."""
        rew = self.cfg.rewards
        penalized_team = int(self.player_team[keeper_idx])

        step_rewards[keeper_idx] += rew.keeper_5s_violation_ind
        self._add_team_reward(step_rewards, penalized_team, rew.keeper_5s_violation_team)
        self.stats["keeper_5s_violations"] += 1
        self._emit_event(
            "5s KEEPER FOUL!",
            self.player_pos[keeper_idx],
            team=penalized_team,
            agent=self.idx_to_agent[keeper_idx],
            reward=rew.keeper_5s_violation_ind + rew.keeper_5s_violation_team,
        )

        self._setup_keeper_foul_teleport(penalized_team=penalized_team)

    # ==========================================================================
    # Collisions, Semi-Circle Barriers & Boundary Clamping
    # ==========================================================================
    def _resolve_player_collisions(self) -> None:
        """Solid circle-circle collision resolution between non-ghost players (R=18 -> min_dist=36)."""
        min_dist = 2.0 * self.cfg.physics.player_radius
        for i in range(self.num_players):
            if self.ghost_timer[i] > 0.0:
                continue
            for j in range(i + 1, self.num_players):
                if self.ghost_timer[j] > 0.0:
                    continue
                delta = self.player_pos[j] - self.player_pos[i]
                dist = float(np.linalg.norm(delta))
                if dist < min_dist:
                    if dist < 1e-6:
                        angle = (i * 0.7 + j * 1.3)
                        normal = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
                        overlap = min_dist
                    else:
                        normal = delta / dist
                        overlap = min_dist - dist
                    self.player_pos[i] -= 0.5 * overlap * normal
                    self.player_pos[j] += 0.5 * overlap * normal

    def _clamp_players_to_boundaries(self) -> None:
        """Enforce field walls and semi-circle barriers (Keepers inside own box, Field outside both)."""
        phys = self.cfg.physics
        r_player = phys.player_radius
        box_r = phys.keeper_box_radius
        goal_centers = (
            np.array(phys.left_goal_center, dtype=np.float64),
            np.array(phys.right_goal_center, dtype=np.float64),
        )

        for idx in range(self.num_players):
            role = self.player_role[idx]
            team = self.player_team[idx]
            my_side = int(self.team_side[team])

            if role == ROLE_KEEPER:
                own_goal = goal_centers[my_side]
                delta = self.player_pos[idx] - own_goal
                dist = float(np.linalg.norm(delta))
                if dist > box_r:
                    self.player_pos[idx] = own_goal + (delta / dist) * box_r

                self.player_pos[idx, 0] = float(
                    np.clip(self.player_pos[idx, 0], r_player, phys.field_width - r_player)
                )
                self.player_pos[idx, 1] = float(
                    np.clip(self.player_pos[idx, 1], r_player, phys.field_height - r_player)
                )
            else:
                self.player_pos[idx, 0] = float(
                    np.clip(self.player_pos[idx, 0], r_player, phys.field_width - r_player)
                )
                self.player_pos[idx, 1] = float(
                    np.clip(self.player_pos[idx, 1], r_player, phys.field_height - r_player)
                )

                for g_center in goal_centers:
                    delta = self.player_pos[idx] - g_center
                    dist = float(np.linalg.norm(delta))
                    if dist <= box_r:
                        if dist < 1e-6:
                            push_dir = np.array(
                                [1.0 if g_center[0] == 0.0 else -1.0, 0.0], dtype=np.float64
                            )
                        else:
                            push_dir = delta / dist
                        self.player_pos[idx] = g_center + push_dir * (box_r + 1e-3)

                self.player_pos[idx, 0] = float(
                    np.clip(self.player_pos[idx, 0], r_player, phys.field_width - r_player)
                )
                self.player_pos[idx, 1] = float(
                    np.clip(self.player_pos[idx, 1], r_player, phys.field_height - r_player)
                )

        # Kickoff Exclusion Circle: Keep opposing players outside center spot (r = 130.0) during kickoff protection
        if self.kickoff_protection_timer > 0.0 and self.ball_holder is not None:
            start_team = int(self.player_team[self.ball_holder])
            field_center = np.array(phys.field_center, dtype=np.float64)
            excl_r = self.cfg.mechanics.kickoff_exclusion_radius
            for idx in range(self.num_players):
                if self.player_team[idx] != start_team:
                    delta = self.player_pos[idx] - field_center
                    dist = float(np.linalg.norm(delta))
                    if dist < excl_r:
                        if dist < 1e-6:
                            push_dir = np.array([1.0, 0.0], dtype=np.float64)
                        else:
                            push_dir = delta / dist
                        self.player_pos[idx] = field_center + push_dir * excl_r

        # GK Possession Buffer (gk_hold_opponent_buffer = 230.0):
        # Keep opposing field players away from GK's goal center while GK holds the ball
        if self.ball_holder is not None and self.player_role[self.ball_holder] == ROLE_KEEPER:
            gk_team = int(self.player_team[self.ball_holder])
            gk_side = int(self.team_side[gk_team])
            gk_goal_center = goal_centers[gk_side]
            buf_r = self.cfg.mechanics.gk_hold_opponent_buffer
            for idx in range(self.num_players):
                if self.player_team[idx] != gk_team:
                    delta = self.player_pos[idx] - gk_goal_center
                    dist = float(np.linalg.norm(delta))
                    if dist < buf_r:
                        if dist < 1e-6:
                            push_dir = np.array([1.0 if gk_side == SIDE_LEFT else -1.0, 0.0], dtype=np.float64)
                        else:
                            push_dir = delta / dist
                        self.player_pos[idx] = gk_goal_center + push_dir * buf_r

        # Final pass: re-clamp all players to pitch boundaries
        for idx in range(self.num_players):
            self.player_pos[idx, 0] = float(
                np.clip(self.player_pos[idx, 0], r_player, phys.field_width - r_player)
            )
            self.player_pos[idx, 1] = float(
                np.clip(self.player_pos[idx, 1], r_player, phys.field_height - r_player)
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

    def _bounce_ball_off_walls(self) -> bool:
        """Bounce free ball elastically off non-goal walls."""
        phys = self.cfg.physics
        r_ball = phys.ball_radius
        rest = phys.ball_wall_restitution
        bounced = False

        bx, by = float(self.ball_pos[0]), float(self.ball_pos[1])
        in_goal_y = phys.goal_y_min <= by <= phys.goal_y_max

        # Left and Right walls
        if not in_goal_y:
            if bx < r_ball:
                self.ball_pos[0] = r_ball + (r_ball - bx) * rest
                self.ball_vel[0] = abs(self.ball_vel[0]) * rest
                self.ball_vel[1] *= rest
                bounced = True
            elif bx > phys.field_width - r_ball:
                over = bx - (phys.field_width - r_ball)
                self.ball_pos[0] = (phys.field_width - r_ball) - over * rest
                self.ball_vel[0] = -abs(self.ball_vel[0]) * rest
                self.ball_vel[1] *= rest
                bounced = True

        # Top and Bottom walls
        by = float(self.ball_pos[1])
        if by < r_ball:
            self.ball_pos[1] = r_ball + (r_ball - by) * rest
            self.ball_vel[1] = abs(self.ball_vel[1]) * rest
            self.ball_vel[0] *= rest
            bounced = True
        elif by > phys.field_height - r_ball:
            over = by - (phys.field_height - r_ball)
            self.ball_pos[1] = (phys.field_height - r_ball) - over * rest
            self.ball_vel[1] = -abs(self.ball_vel[1]) * rest
            self.ball_vel[0] *= rest
            bounced = True

        return bounced

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
                if int(self.player_role[self.last_ball_holder]) == ROLE_ATK:
                    scorer_ind = rew.atk_goal_scorer_ind
                else:
                    scorer_ind = rew.goal_scorer_ind
                step_rewards[self.last_ball_holder] += scorer_ind
            else:
                scorer_ind = rew.own_goal_ind
                step_rewards[self.last_ball_holder] += scorer_ind
                self.stats["own_goals"] = self.stats.get("own_goals", 0) + 1
                is_own_goal = True

        self._add_team_reward(step_rewards, scoring_team, rew.goal_team_shared)
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
                reward=scorer_ind + rew.goal_team_shared,
            )

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
        self.tackle_cooldown[:] = 0.0
        self.lunge_cooldown[:] = 0.0
        self.lunge_timer[:] = 0.0
        self.pickup_cooldown[:] = 0.0
        self.kick_cooldown_active[:] = False
        self.violation_cooldown_active[:] = False
        self.shot_on_target_cd[:] = 0.0
        self.popup_cooldown[:] = 0.0
        self.ball_vel[:] = 0.0
        self.ball_hold_timer = 0.0
        self.kick_grace_timer = 0.0
        self.box_loose_ball_timer = 0.0
        self.ball_kicked_in_flight = False
        self.ball_kick_was_shot_on_target = False
        self.ball_kick_flight_timer = 0.0
        self.pinned_ball_timer = 0.0
        self.kickoff_protection_timer = 0.0
        self.prev_longitudinal_vel[:] = 0.0

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
        self._clear_all_dynamic_timers()
        self.kickoff_protection_timer = self.cfg.mechanics.kickoff_protection_time
        form = self.cfg.formations
        start_side = int(self.team_side[starting_team])

        for idx in range(self.num_players):
            team = int(self.player_team[idx])
            role_key = self.player_role_key[idx]
            if team == starting_team:
                canon_pos = form.kickoff_starting_left[role_key]
                canon_angle = -0.45 if role_key == "atk_1" else 0.0
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
        self.prev_ball_x = float(self.ball_pos[0])

    def _setup_keeper_foul_teleport(self, penalized_team: int) -> None:
        """Position both teams for the Keeper 5-Second Foul Teleport, respecting physical side."""
        self._clear_all_dynamic_timers()
        form = self.cfg.formations
        attacking_team = 1 - penalized_team
        penalized_side = int(self.team_side[penalized_team])

        for idx in range(self.num_players):
            team = int(self.player_team[idx])
            role_key = self.player_role_key[idx]
            if team == penalized_team:
                canon_pos = form.foul_penalized_left[role_key]
                canon_angle = 0.0
            else:
                canon_pos = form.foul_attacking_right[role_key]
                canon_angle = math.pi

            world_pos, world_angle = self._CanonicalToWorld(
                canon_pos, canon_angle, penalized_side
            )
            self.player_pos[idx] = world_pos
            self.player_angle[idx] = world_angle

        # Attacking team's atk_1 starts holding the ball
        carrier_name = "a_atk_1" if attacking_team == TEAM_A else "b_atk_1"
        carrier_idx = self.agent_to_idx[carrier_name]
        self.ball_holder = carrier_idx
        self.last_ball_holder = carrier_idx
        self.last_ball_team = attacking_team

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
        self.prev_ball_x = float(self.ball_pos[0])

    # ==========================================================================
    # Team-Canonical Observation Builder (76 Floats)
    # ==========================================================================
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Compute the 76-float normalized observation vector for all 12 agents."""
        obs_dict: Dict[str, np.ndarray] = {}
        for idx, agent_name in enumerate(self.possible_agents):
            obs_dict[agent_name] = self._build_single_observation(idx)
        return obs_dict

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

    def _build_single_observation(self, idx: int) -> np.ndarray:
        """Build the 76-float canonical observation for player `idx`."""
        phys = self.cfg.physics
        mech = self.cfg.mechanics
        vis = self.cfg.vision

        my_team = int(self.player_team[idx])
        my_role = int(self.player_role[idx])

        # Transform all players and ball into `my_team`'s canonical coordinate system
        c_player_pos, c_player_vel, c_player_angle = self._to_team_canonical(
            self.player_pos, self.player_vel, self.player_angle, my_team
        )
        c_ball_pos, c_ball_vel, _ = self._to_team_canonical(
            self.ball_pos, self.ball_vel, 0.0, my_team
        )

        obs = np.zeros(vis.obs_dim, dtype=np.float32)

        # ----------------------------------------------------------------------
        # Part A: Self & Global State (16 floats, indices 0..15)
        # ----------------------------------------------------------------------
        sx, sy = float(c_player_pos[idx, 0]), float(c_player_pos[idx, 1])
        s_angle = float(c_player_angle[idx])
        svx, svy = float(c_player_vel[idx, 0]), float(c_player_vel[idx, 1])

        # Normalize self_x, self_y to [-1.0, 1.0]
        obs[0] = np.clip((sx / phys.field_width) * 2.0 - 1.0, -1.0, 1.0)
        obs[1] = np.clip((sy / phys.field_height) * 2.0 - 1.0, -1.0, 1.0)

        # sin and cos of canonical facing angle
        obs[2] = math.sin(s_angle)
        obs[3] = math.cos(s_angle)

        # Normalized self velocity
        max_self_v = mech.lunge_speed if my_role == ROLE_KEEPER else self._get_player_max_speed(idx)
        obs[4] = np.clip(svx / max_self_v, -1.0, 1.0)
        obs[5] = np.clip(svy / max_self_v, -1.0, 1.0)

        # Score Differential
        my_score = self.stats.get("score", [0, 0])[my_team]
        opp_score = self.stats.get("score", [0, 0])[1 - my_team]
        obs[6] = float(np.clip((my_score - opp_score) / 3.0, -1.0, 1.0))
        has_ball = 1.0 if self.ball_holder == idx else 0.0
        obs[7] = has_ball

        # Possession / hold / tackle status
        max_hold = (
            mech.keeper_max_hold_time if my_role == ROLE_KEEPER else mech.field_max_hold_time
        )
        if has_ball > 0.5:
            obs[8] = np.clip(self.ball_hold_timer / max_hold, 0.0, 1.0)
        elif self.ball_holder is None:
            if my_role == ROLE_KEEPER and self.box_loose_ball_timer > 0.0:
                obs[8] = np.clip(
                    self.box_loose_ball_timer / mech.keeper_box_loose_max_time, 0.0, 1.0
                )
            else:
                obs[8] = 0.0
        elif int(self.player_team[self.ball_holder]) == my_team:
            obs[8] = 0.5
        else:
            holder_idx = self.ball_holder
            d_holder = float(np.linalg.norm(self.player_pos[idx] - self.player_pos[holder_idx]))
            my_tackle_reach = self._get_role_tackle_params(idx)[0]
            if (
                self._is_field_role(self.player_role[holder_idx])
                and d_holder <= my_tackle_reach
            ):
                obs[8] = -1.0
            else:
                obs[8] = -0.5

        # Ability cooldown normalized to [0.0, 1.0]
        if my_role == ROLE_KEEPER:
            obs[9] = np.clip(self.lunge_cooldown[idx] / mech.lunge_cooldown, 0.0, 1.0)
        else:
            max_tackle_cd = max(mech.tackle_fail_cooldown, mech.field_violation_tackle_cooldown)
            obs[9] = np.clip(self.tackle_cooldown[idx] / max_tackle_cd, 0.0, 1.0)

        # Ghost or Stun timer normalized to [0.0, 1.0]
        active_disable_timer = max(float(self.ghost_timer[idx]), float(self.stun_timer[idx]))
        obs[10] = np.clip(active_disable_timer / mech.ghost_duration, 0.0, 1.0)

        # Relative angle and distance to Enemy Goal (always at (1400.0, 420.0) in canonical space)
        enemy_goal_dx = phys.field_width - sx
        enemy_goal_dy = (phys.field_height * 0.5) - sy
        goal_dist = math.hypot(enemy_goal_dx, enemy_goal_dy)
        angle_to_goal = math.atan2(enemy_goal_dy, enemy_goal_dx)
        rel_goal_angle = float(wrap_to_pi(angle_to_goal - s_angle))

        obs[11] = np.clip(rel_goal_angle / math.pi, -1.0, 1.0)
        obs[12] = np.clip(goal_dist / self._max_field_dist, 0.0, 1.0)

        # Role One-Hot Encoding (indices 13..15: [is_def, is_mid, is_atk]; keeper is [0, 0, 0])
        obs[13] = 1.0 if my_role == ROLE_DEF else 0.0
        obs[14] = 1.0 if my_role == ROLE_MID else 0.0
        obs[15] = 1.0 if my_role == ROLE_ATK else 0.0

        # ----------------------------------------------------------------------
        # Part B: Vision Cone Entity Slots (12 Entities x 5 floats = 60 floats, indices 16..75)
        # Slot 0: Ball
        # Slots 1..5: 5 Teammates
        # Slots 6..10: 5 Field Opponents
        # Slot 11: Enemy Keeper
        # ----------------------------------------------------------------------
        slot_positions = [c_ball_pos]
        slot_velocities = [c_ball_vel]
        for other_idx in self._entity_player_slots[idx]:
            slot_positions.append(c_player_pos[other_idx])
            slot_velocities.append(c_player_vel[other_idx])

        max_rel_vel = phys.ball_max_launch_speed + phys.player_max_speed

        base_offset = vis.self_feature_dim
        for slot_i in range(vis.num_entity_slots):
            epos = slot_positions[slot_i]
            evel = slot_velocities[slot_i]

            dx = float(epos[0]) - sx
            dy = float(epos[1]) - sy
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                rel_ang = 0.0
            else:
                ang_to_ent = math.atan2(dy, dx)
                rel_ang = float(wrap_to_pi(ang_to_ent - s_angle))

            slot_start = base_offset + slot_i * vis.entity_feature_dim

            if slot_i == 0:
                # Slot 0 (Ball): 360-degree awareness
                in_cone = (dist <= vis.max_range) and (abs(rel_ang) <= vis.half_fov_rad)
                obs[slot_start + 0] = 1.0 if in_cone else 0.5
                obs[slot_start + 1] = np.clip(dist / self._max_field_dist, 0.0, 1.0)
                obs[slot_start + 2] = np.clip(rel_ang / math.pi, -1.0, 1.0)
                obs[slot_start + 3] = np.clip(
                    (float(evel[0]) - svx) / max_rel_vel, -1.0, 1.0
                )
                obs[slot_start + 4] = np.clip(
                    (float(evel[1]) - svy) / max_rel_vel, -1.0, 1.0
                )
            else:
                # Slots 1..11: Vision cone limited to vis.max_range and half_fov_rad
                if dist <= vis.max_range and abs(rel_ang) <= vis.half_fov_rad:
                    obs[slot_start + 0] = 1.0
                    obs[slot_start + 1] = np.clip(dist / vis.max_range, 0.0, 1.0)
                    obs[slot_start + 2] = np.clip(
                        rel_ang / vis.half_fov_rad, -1.0, 1.0
                    )
                    obs[slot_start + 3] = np.clip(
                        (float(evel[0]) - svx) / max_rel_vel, -1.0, 1.0
                    )
                    obs[slot_start + 4] = np.clip(
                        (float(evel[1]) - svy) / max_rel_vel, -1.0, 1.0
                    )

        return obs

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

    def _compute_tactical_positioning_rewards(
        self,
        step_rewards: np.ndarray,
        prev_player_pos: np.ndarray,
        prev_player_ball_dists: np.ndarray,
        any_teleport: bool,
    ) -> None:
        """Context-dependent off-ball shaping on 1400x840 pitch with role pockets & anti-swarming."""
        rew = self.cfg.rewards
        phys = self.cfg.physics
        curr_player_ball_dists = np.linalg.norm(
            self.player_pos - self.ball_pos[None, :], axis=1
        )
        holder_team = (
            int(self.player_team[self.ball_holder])
            if self.ball_holder is not None
            else -1
        )

        for team_id in (TEAM_A, TEAM_B):
            c_player_pos, _, c_player_angle = self._to_team_canonical(
                self.player_pos, self.player_vel, self.player_angle, team_id
            )
            c_prev_pos, _, _ = self._to_team_canonical(
                prev_player_pos, self.player_vel, self.player_angle, team_id
            )
            c_ball_pos, _, _ = self._to_team_canonical(
                self.ball_pos, self.ball_vel, 0.0, team_id
            )

            team_fields = [
                i
                for i in range(self.num_players)
                if self.player_team[i] == team_id and self._is_field_role(self.player_role[i])
            ]
            opp_fields = [
                i
                for i in range(self.num_players)
                if self.player_team[i] != team_id and self._is_field_role(self.player_role[i])
            ]
            keeper_idx = self._team_keeper_idx(team_id)

            score_diff = (
                self.stats.get("score", [0, 0])[team_id]
                - self.stats.get("score", [0, 0])[1 - team_id]
            )
            press_mult = 1.35 if score_diff < 0 else 1.0
            support_mult = 1.25 if score_diff < 0 else 1.0
            cover_mult = 1.25 if score_diff > 0 else 1.0

            # 1. Anti-Clumping across all 5 field players
            for idx_a in range(len(team_fields)):
                for idx_b in range(idx_a + 1, len(team_fields)):
                    p_i = team_fields[idx_a]
                    p_j = team_fields[idx_b]
                    d_ij = float(np.linalg.norm(self.player_pos[p_i] - self.player_pos[p_j]))
                    if d_ij < rew.teammate_min_spacing:
                        pen = rew.crowding_penalty_scale * (
                            1.0 - d_ij / rew.teammate_min_spacing
                        )
                        step_rewards[p_i] -= pen
                        step_rewards[p_j] -= pen

            # 2. State A: Our Team Has the Ball
            if holder_team == team_id and self.ball_holder is not None:
                carrier_idx = self.ball_holder
                carrier_c = c_player_pos[carrier_idx]
                cx_car = float(carrier_c[0])
                is_gk_carrier = (carrier_idx == keeper_idx)

                for idx in team_fields:
                    if idx == carrier_idx:
                        continue

                    d_curr_car = float(
                        np.linalg.norm(self.player_pos[idx] - self.player_pos[carrier_idx])
                    )
                    d_prev_car = float(
                        np.linalg.norm(prev_player_pos[idx] - prev_player_pos[carrier_idx])
                    )

                    # Anti-Carrier Crowding & Approach Penalty
                    if d_curr_car < rew.carrier_min_spacing:
                        step_rewards[idx] -= rew.crowding_penalty_scale * (
                            1.0 - d_curr_car / rew.carrier_min_spacing
                        )
                    if not any_teleport and d_curr_car < 190.0 and d_curr_car < d_prev_car:
                        step_rewards[idx] -= rew.carrier_approach_penalty_scale * (
                            d_prev_car - d_curr_car
                        )

                    role_key = self.player_role_key[idx]

                    if is_gk_carrier:
                        # Case 1: GK Holds the Ball -> "Show for the Keeper" wide outlet pockets
                        if role_key in ("def_1", "mid_1"):
                            x_cand_list = (200.0, 330.0) if role_key == "def_1" else (420.0, 560.0)
                            y_cand_list = (95.0, 195.0, 295.0)
                        elif role_key in ("def_2", "mid_2"):
                            x_cand_list = (200.0, 330.0) if role_key == "def_2" else (420.0, 560.0)
                            y_cand_list = (545.0, 645.0, 745.0)
                        else:
                            x_cand_list = (720.0, 920.0)
                            y_cand_list = (260.0, 420.0, 580.0)

                        best_cand = None
                        best_score = -1e9
                        for cx in x_cand_list:
                            for cy in y_cand_list:
                                cand_pt = np.array([cx, cy], dtype=np.float64)
                                pass_lane_clearance = min(
                                    self._point_to_segment_dist(c_player_pos[opp], carrier_c, cand_pt)
                                    for opp in opp_fields
                                )
                                nearest_opp_dist = min(
                                    float(np.linalg.norm(c_player_pos[opp] - cand_pt))
                                    for opp in opp_fields
                                )
                                continuity = -0.25 * float(np.linalg.norm(cand_pt - c_player_pos[idx]))
                                sc = pass_lane_clearance + 0.6 * nearest_opp_dist + continuity
                                if sc > best_score:
                                    best_score = sc
                                    best_cand = cand_pt
                        target_pocket = best_cand
                    else:
                        # Case 2: Field Player Holds the Ball
                        if role_key in ("def_1", "def_2"):
                            if cx_car < 700.0:
                                ideal_x = float(np.clip(cx_car + 90.0, 240.0, 580.0))
                            else:
                                ideal_x = float(np.clip(cx_car - 240.0, 520.0, 780.0))
                            ideal_y = 250.0 if role_key == "def_1" else 590.0
                            target_pocket = np.array([ideal_x, ideal_y], dtype=np.float64)
                        elif role_key in ("mid_1", "mid_2"):
                            dx_list = (120.0, 220.0, -90.0)
                            y_cand_list = (150.0, 290.0) if role_key == "mid_1" else (550.0, 690.0)
                            best_cand = None
                            best_score = -1e9
                            for dx in dx_list:
                                cand_x = float(np.clip(cx_car + dx, 360.0, 1120.0))
                                for cy in y_cand_list:
                                    cand_pt = np.array([cand_x, cy], dtype=np.float64)
                                    pass_lane_clearance = min(
                                        self._point_to_segment_dist(c_player_pos[opp], carrier_c, cand_pt)
                                        for opp in opp_fields
                                    )
                                    nearest_opp_dist = min(
                                        float(np.linalg.norm(c_player_pos[opp] - cand_pt))
                                        for opp in opp_fields
                                    )
                                    continuity = -0.30 * float(np.linalg.norm(cand_pt - c_player_pos[idx]))
                                    sc = pass_lane_clearance + 0.5 * nearest_opp_dist + continuity
                                    if sc > best_score:
                                        best_score = sc
                                        best_cand = cand_pt
                            target_pocket = best_cand
                        else:
                            # Attacker (atk_1)
                            x_atk = float(np.clip(max(720.0, cx_car + 180.0), 720.0, 1240.0))
                            best_cand = None
                            best_score = -1e9
                            for cy in (200.0, 310.0, 420.0, 530.0, 640.0):
                                cand_pt = np.array([x_atk, cy], dtype=np.float64)
                                pass_lane_clearance = min(
                                    self._point_to_segment_dist(c_player_pos[opp], carrier_c, cand_pt)
                                    for opp in opp_fields
                                )
                                nearest_opp_dist = min(
                                    float(np.linalg.norm(c_player_pos[opp] - cand_pt))
                                    for opp in opp_fields
                                )
                                continuity = -0.30 * float(np.linalg.norm(cand_pt - c_player_pos[idx]))
                                sc = pass_lane_clearance + 0.6 * nearest_opp_dist + continuity
                                if sc > best_score:
                                    best_score = sc
                                    best_cand = cand_pt
                            target_pocket = best_cand

                    # Potential-Based Movement Reward Toward target_pocket
                    if not any_teleport and target_pocket is not None:
                        d_prev_p = float(np.linalg.norm(c_prev_pos[idx] - target_pocket))
                        d_curr_p = float(np.linalg.norm(c_player_pos[idx] - target_pocket))
                        step_rewards[idx] += (
                            support_mult
                            * rew.support_pocket_approach_scale
                            * (d_prev_p - d_curr_p)
                        )

                    # "Wait for the Pass" Open-Pocket Bonus
                    lane_clear = min(
                        self._point_to_segment_dist(c_player_pos[opp], carrier_c, c_player_pos[idx])
                        for opp in opp_fields
                    )
                    if 120.0 <= d_curr_car <= 520.0 and lane_clear >= 45.0:
                        to_ball_c = c_ball_pos - c_player_pos[idx]
                        d_ball_c = float(np.linalg.norm(to_ball_c))
                        if d_ball_c > 1e-6:
                            cos_face_ball = float(
                                np.dot(
                                    [math.cos(c_player_angle[idx]), math.sin(c_player_angle[idx])],
                                    to_ball_c / d_ball_c,
                                )
                            )
                        else:
                            cos_face_ball = 1.0
                        face_bonus = 0.4 + 0.6 * max(0.0, cos_face_ball)
                        lane_factor = float(np.clip(lane_clear / 110.0, 0.3, 1.0))
                        step_rewards[idx] += (
                            support_mult
                            * rew.open_support_bonus_scale
                            * lane_factor
                            * face_bonus
                        )

            # 3. State B: Ball is Loose
            elif self.ball_holder is None:
                role_anchors = {
                    "def_1": np.array([np.clip(c_ball_pos[0] * 0.55, 220.0, 580.0), 250.0], dtype=np.float64),
                    "def_2": np.array([np.clip(c_ball_pos[0] * 0.55, 220.0, 580.0), 590.0], dtype=np.float64),
                    "mid_1": np.array([np.clip(c_ball_pos[0] * 0.85, 380.0, 980.0), 200.0], dtype=np.float64),
                    "mid_2": np.array([np.clip(c_ball_pos[0] * 0.85, 380.0, 980.0), 640.0], dtype=np.float64),
                    "atk_1": np.array([np.clip(c_ball_pos[0] + 220.0, 720.0, 1180.0), 420.0], dtype=np.float64),
                }

                # Case B1: Teammate pass in flight
                if self.ball_kicked_in_flight and self.last_ball_team == team_id:
                    candidates = [i for i in team_fields if i != self.last_ball_holder]
                    if len(candidates) > 0:
                        ray_end = self.ball_pos + self.ball_vel * 1.5
                        best_rx = min(
                            candidates,
                            key=lambda i: self._point_to_segment_dist(self.player_pos[i], self.ball_pos, ray_end),
                        )
                        for idx in team_fields:
                            if idx == best_rx:
                                if not any_teleport:
                                    dist_closed = float(prev_player_ball_dists[idx]) - float(
                                        curr_player_ball_dists[idx]
                                    )
                                    step_rewards[idx] += press_mult * rew.loose_ball_chase_scale * dist_closed
                                to_ball = self.ball_pos - self.player_pos[idx]
                                d_ball = float(np.linalg.norm(to_ball))
                                if d_ball > 1e-6:
                                    facing_vec = np.array(
                                        [math.cos(self.player_angle[idx]), math.sin(self.player_angle[idx])],
                                        dtype=np.float64,
                                    )
                                    cos_align = float(np.dot(facing_vec, to_ball / d_ball))
                                    if cos_align > 0.0:
                                        step_rewards[idx] += press_mult * rew.ball_heading_align_scale * cos_align
                            else:
                                anc = role_anchors[self.player_role_key[idx]]
                                if not any_teleport:
                                    d_prev_anc = float(np.linalg.norm(c_prev_pos[idx] - anc))
                                    d_curr_anc = float(np.linalg.norm(c_player_pos[idx] - anc))
                                    step_rewards[idx] += rew.role_zone_anchor_scale * (d_prev_anc - d_curr_anc)
                    else:
                        pass
                else:
                    # Case B2: Contested / Unclaimed Loose Ball
                    ranked = sorted(team_fields, key=lambda i: float(curr_player_ball_dists[i]))
                    for rank_i, idx in enumerate(ranked):
                        is_chaser = (rank_i == 0) or (rank_i == 1 and float(curr_player_ball_dists[idx]) <= 260.0)
                        if is_chaser:
                            w = (1.0 if rank_i == 0 else 0.45) * press_mult
                            if not any_teleport:
                                dist_closed = float(prev_player_ball_dists[idx]) - float(
                                    curr_player_ball_dists[idx]
                                )
                                step_rewards[idx] += w * rew.loose_ball_chase_scale * dist_closed
                            to_ball = self.ball_pos - self.player_pos[idx]
                            d_ball = float(np.linalg.norm(to_ball))
                            if d_ball > 1e-6:
                                facing_vec = np.array(
                                    [math.cos(self.player_angle[idx]), math.sin(self.player_angle[idx])],
                                    dtype=np.float64,
                                )
                                cos_align = float(np.dot(facing_vec, to_ball / d_ball))
                                if cos_align > 0.0:
                                    step_rewards[idx] += w * rew.ball_heading_align_scale * cos_align
                        else:
                            anc = role_anchors[self.player_role_key[idx]]
                            if not any_teleport:
                                d_prev_anc = float(np.linalg.norm(c_prev_pos[idx] - anc))
                                d_curr_anc = float(np.linalg.norm(c_player_pos[idx] - anc))
                                step_rewards[idx] += rew.role_zone_anchor_scale * (d_prev_anc - d_curr_anc)
                            d_ball = float(curr_player_ball_dists[idx])
                            if d_ball < rew.non_chaser_ball_repulsion_radius:
                                step_rewards[idx] -= rew.non_chaser_ball_repulsion_scale * (
                                    1.0 - d_ball / rew.non_chaser_ball_repulsion_radius
                                )

            # 4. State C: Opponent Has the Ball
            else:
                cover_anchors = {
                    "def_1": np.array([np.clip(c_ball_pos[0] * 0.55, 200.0, 560.0), 270.0], dtype=np.float64),
                    "def_2": np.array([np.clip(c_ball_pos[0] * 0.55, 200.0, 560.0), 570.0], dtype=np.float64),
                    "mid_1": np.array([np.clip(c_ball_pos[0] * 0.78, 360.0, 820.0), 210.0], dtype=np.float64),
                    "mid_2": np.array([np.clip(c_ball_pos[0] * 0.78, 360.0, 820.0), 630.0], dtype=np.float64),
                    "atk_1": np.array([np.clip(c_ball_pos[0] + 180.0, 680.0, 1080.0), 420.0], dtype=np.float64),
                }

                ranked = sorted(team_fields, key=lambda i: float(curr_player_ball_dists[i]))
                press_radius_2nd = 320.0 if score_diff < 0 else 260.0
                for rank_i, idx in enumerate(ranked):
                    d_curr = float(curr_player_ball_dists[idx])
                    if rank_i == 0 or (rank_i == 1 and d_curr <= press_radius_2nd):
                        w = (1.0 if rank_i == 0 else 0.6) * press_mult
                        if not any_teleport:
                            dist_closed = float(prev_player_ball_dists[idx]) - d_curr
                            step_rewards[idx] += w * rew.presser_approach_scale * dist_closed
                        to_ball = self.ball_pos - self.player_pos[idx]
                        if d_curr > 1e-6:
                            facing_vec = np.array(
                                [
                                    math.cos(self.player_angle[idx]),
                                    math.sin(self.player_angle[idx]),
                                ],
                                dtype=np.float64,
                            )
                            cos_align = float(np.dot(facing_vec, to_ball / d_curr))
                            if cos_align > 0.0:
                                step_rewards[idx] += w * rew.ball_heading_align_scale * cos_align
                    else:
                        anc = cover_anchors[self.player_role_key[idx]]
                        if not any_teleport:
                            d_prev_anc = float(np.linalg.norm(c_prev_pos[idx] - anc))
                            d_curr_anc = float(np.linalg.norm(c_player_pos[idx] - anc))
                            step_rewards[idx] += (
                                cover_mult
                                * rew.cover_approach_scale
                                * (d_prev_anc - d_curr_anc)
                            )
                        d_ball = float(curr_player_ball_dists[idx])
                        if d_ball < rew.non_chaser_ball_repulsion_radius:
                            step_rewards[idx] -= rew.non_chaser_ball_repulsion_scale * (
                                1.0 - d_ball / rew.non_chaser_ball_repulsion_radius
                            )

            # 5. Keeper Goal-Line Positioning
            own_goal_c = np.array([0.0, 420.0], dtype=np.float64)
            ray = c_ball_pos - own_goal_c
            ray_norm = max(1e-6, float(np.linalg.norm(ray)))

            k_angle_c = float(c_player_angle[keeper_idx])
            k_facing_c = np.array([math.cos(k_angle_c), math.sin(k_angle_c)], dtype=np.float64)
            to_ball_k = c_ball_pos - c_player_pos[keeper_idx]
            d_ball_k = float(np.linalg.norm(to_ball_k))
            cos_k_ball = (
                float(np.dot(k_facing_c, to_ball_k / d_ball_k)) if d_ball_k > 1e-6 else 1.0
            )

            if math.cos(k_angle_c) < -0.15 and self.ball_holder != keeper_idx:
                step_rewards[keeper_idx] -= rew.keeper_back_wall_face_pen

            if (
                self.ball_holder is None
                and ray_norm <= phys.keeper_box_radius + 18.0
            ):
                if not any_teleport:
                    dist_closed_k = float(prev_player_ball_dists[keeper_idx]) - float(
                        curr_player_ball_dists[keeper_idx]
                    )
                    step_rewards[keeper_idx] += rew.keeper_fetch_box_scale * dist_closed_k
                if cos_k_ball > 0.0:
                    step_rewards[keeper_idx] += rew.ball_heading_align_scale * cos_k_ball
                # Urgency penalty every step loose ball sits in GK box
                step_rewards[keeper_idx] -= 0.03 * (
                    self.box_loose_ball_timer / max(1e-6, self.cfg.mechanics.keeper_box_loose_max_time)
                )
            else:
                raw_ideal = own_goal_c + (ray / ray_norm) * 36.0
                ideal_k_pos = np.array(
                    [
                        np.clip(raw_ideal[0], 20.0, 48.0),
                        np.clip(raw_ideal[1], phys.goal_y_min + 8.0, phys.goal_y_max - 8.0),
                    ],
                    dtype=np.float64,
                )
                if not any_teleport:
                    d_prev_k_ideal = float(np.linalg.norm(c_prev_pos[keeper_idx] - ideal_k_pos))
                    d_curr_k_ideal = float(np.linalg.norm(c_player_pos[keeper_idx] - ideal_k_pos))
                    delta_ideal = d_prev_k_ideal - d_curr_k_ideal
                    # Gate positive reward on facing field (cos >= 0) or retreat (delta <= 0)
                    if delta_ideal <= 0.0 or math.cos(k_angle_c) >= 0.0:
                        step_rewards[keeper_idx] += rew.keeper_anchor_approach_scale * delta_ideal

                d_curr_k_ideal = float(np.linalg.norm(c_player_pos[keeper_idx] - ideal_k_pos))
                dist_from_goal = float(np.linalg.norm(c_player_pos[keeper_idx] - own_goal_c))
                if not (dist_from_goal > 95.0 and self.ball_holder != keeper_idx):
                    step_rewards[keeper_idx] += (
                        rew.keeper_angle_track_scale
                        * max(0.0, 1.0 - d_curr_k_ideal / 70.0)
                        * (0.25 + 0.75 * max(0.0, cos_k_ball))
                    )

                # Semicircle Edge-Camping Penalty (> keeper_edge_camp_radius units from own goal center)
                if dist_from_goal > rew.keeper_edge_camp_radius and self.ball_holder != keeper_idx:
                    excess = (dist_from_goal - rew.keeper_edge_camp_radius) / max(
                        1.0, phys.keeper_box_radius - rew.keeper_edge_camp_radius
                    )
                    step_rewards[keeper_idx] -= rew.keeper_edge_camp_pen * (
                        excess + 0.5 * excess * excess
                    )

    def _emit_event(
        self,
        label: str,
        pos: np.ndarray,
        team: Optional[int] = None,
        agent: Optional[str] = None,
        reward: Optional[float] = None,
    ) -> None:
        """Record a discrete game event on the current physics frame for UI popups."""
        self._frame_events.append(
            {
                "type": label,
                "x": round(float(pos[0]), 2),
                "y": round(float(pos[1]), 2),
                "team": int(team) if team is not None else -1,
                "agent": agent or "",
                "reward": round(float(reward), 2) if reward is not None else None,
            }
        )

    def _record_replay_frame(self) -> None:
        """Append the current 60 Hz state snapshot to `self.replay_frames`."""
        players_frame = []
        for idx in range(self.num_players):
            flags = 0
            if self.ball_holder == idx:
                flags |= 1
            if self.ghost_timer[idx] > 0.0:
                flags |= 2
            if self.stun_timer[idx] > 0.0:
                flags |= 4
            if self.lunge_timer[idx] > 0.0:
                flags |= 8
            players_frame.append(
                [
                    round(float(self.player_pos[idx, 0]), 2),
                    round(float(self.player_pos[idx, 1]), 2),
                    round(float(self.player_angle[idx]), 4),
                    flags,
                ]
            )

        ball_team_color = (
            int(self.player_team[self.ball_holder])
            if self.ball_holder is not None
            else (int(self.last_ball_team) if self.last_ball_team is not None else -1)
        )
        ball_frame = [
            round(float(self.ball_pos[0]), 2),
            round(float(self.ball_pos[1]), 2),
            ball_team_color,
        ]

        total_poss = max(1, sum(self.stats["possession_frames"]))
        poss_pct = [
            round(100.0 * self.stats["possession_frames"][0] / total_poss, 1),
            round(100.0 * self.stats["possession_frames"][1] / total_poss, 1),
        ]

        self.replay_frames.append(
            {
                "frame": self.physics_frame_count,
                "time": round(self.physics_frame_count * self.cfg.physics.dt, 3),
                "half": self.current_half,
                "team_side": [int(self.team_side[0]), int(self.team_side[1])],
                "players": players_frame,
                "ball": ball_frame,
                "player_rewards": [round(float(r), 4) for r in self.last_step_rewards],
                "cum_rewards": [round(float(r), 2) for r in self.cum_player_rewards],
                "stats": {
                    "score": list(self.stats["score"]),
                    "possession_pct": poss_pct,
                    "passes": list(self.stats["team_passes"]),
                    "tackles": list(self.stats["team_tackles"]),
                    "saves": list(self.stats["team_saves"]),
                    "shots": list(self.stats["shots"]),
                },
                "events": list(self._frame_events),
            }
        )

    def compute_interest_score(self) -> float:
        """Compute a competitive episode interest_score that rewards two-sided play and penalizes blowouts/own-goals."""
        rep = self.cfg.replay
        score_a, score_b = int(self.stats["score"][0]), int(self.stats["score"][1])
        total_goals = score_a + score_b
        own_goals = int(self.stats.get("own_goals", 0))
        valid_goals = max(0, total_goals - own_goals)
        goal_diff = abs(score_a - score_b)

        # Diminishing returns on excess goals in blowouts (first 5 valid goals count full, excess counts 35%)
        effective_goals = min(valid_goals, 5) + max(0, valid_goals - 5) * 0.35

        passes_a, passes_b = int(self.stats["team_passes"][0]), int(self.stats["team_passes"][1])
        tackles_a, tackles_b = int(self.stats["team_tackles"][0]), int(self.stats["team_tackles"][1])
        shots_a, shots_b = int(self.stats["shots"][0]), int(self.stats["shots"][1])

        # Cap raw tackle spam contribution at 18 total tackles
        effective_tackles = min(int(self.stats["successful_tackles"]), 18)

        base_score = (
            (effective_goals * rep.goal_weight)
            + (self.stats["completed_passes"] * rep.completed_pass_weight)
            + (effective_tackles * rep.successful_tackle_weight)
            + (self.stats["keeper_saves"] * rep.keeper_save_weight)
        )

        # Two-sided play bonus (rewards matches where BOTH teams pass, shoot, and tackle)
        two_sided_bonus = (
            min(passes_a, passes_b) * 2.5
            + min(shots_a, shots_b) * 2.0
            + min(tackles_a, tackles_b) * 1.0
        )

        # Closeness multiplier: boost tight contested games, discount lopsided blowouts
        if valid_goals > 0 and goal_diff <= 1:
            closeness_mult = 1.25
        elif goal_diff <= 2:
            closeness_mult = 1.0
        else:
            closeness_mult = 1.0 / (1.0 + 0.22 * (goal_diff - 2))

        penalties = (own_goals * 4.0) + (int(self.stats.get("keeper_5s_violations", 0)) * 1.5)
        final_score = max(0.0, (base_score + two_sided_bonus) * closeness_mult - penalties)
        return round(float(final_score), 2)

    def _maybe_export_replay(self) -> Optional[str]:
        """Export the full 60 Hz JSON trajectory if `interest_score >= threshold`."""
        interest_score = self.compute_interest_score()
        if interest_score < self.cfg.replay.interest_threshold:
            return None

        os.makedirs(self.cfg.replay.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        score_str = f"{interest_score:.1f}".replace(".", "_")
        if self.current_match_id > 0:
            filename = (
                f"match_{self.current_match_id:04d}_gen{self.current_generation:04d}"
                f"_score{score_str}_{timestamp}.json"
            )
        else:
            filename = f"match_{timestamp}_score{score_str}.json"
        filepath = os.path.join(self.cfg.replay.save_dir, filename)

        highlights = []
        for fr in self.replay_frames:
            for ev in fr.get("events", []):
                if ev.get("type") != "TACKLE FAILED!":
                    highlights.append(
                        {
                            "frame": fr["frame"],
                            "time": fr["time"],
                            "half": fr["half"],
                            "type": ev["type"],
                            "agent": ev.get("agent", ""),
                            "team": ev.get("team", -1),
                            "x": ev["x"],
                            "y": ev["y"],
                            "reward": ev.get("reward"),
                        }
                    )

        metadata_block = {
            "env_name": self.metadata["name"],
            "match_id": self.current_match_id,
            "generation": self.current_generation,
            "timestamp": timestamp,
            "interest_score": round(interest_score, 2),
            "physics_hz": self.cfg.physics.physics_hz,
            "rl_hz": self.cfg.physics.rl_hz,
            "field_width": self.cfg.physics.field_width,
            "field_height": self.cfg.physics.field_height,
            "goal_y_min": self.cfg.physics.goal_y_min,
            "goal_y_max": self.cfg.physics.goal_y_max,
            "keeper_box_radius": self.cfg.physics.keeper_box_radius,
            "player_radius": self.cfg.physics.player_radius,
            "ball_radius": self.cfg.physics.ball_radius,
            "vision_fov_deg": self.cfg.vision.fov_deg,
            "vision_max_range": self.cfg.vision.max_range,
            "agents": self.possible_agents,
            "final_stats": self.stats,
            "highlights": highlights,
        }

        meta_json = json.dumps(metadata_block, indent=2)
        meta_indented = "\n".join("  " + line for line in meta_json.splitlines())
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("{\n  \"metadata\": ")
            f.write(meta_indented.lstrip())
            f.write(",\n  \"frames\": [\n")
            last_idx = len(self.replay_frames) - 1
            for i, fr in enumerate(self.replay_frames):
                suffix = ",\n" if i < last_idx else "\n"
                f.write("    " + json.dumps(fr) + suffix)
            f.write("  ]\n}\n")

        self.last_saved_replay_path = filepath
        return filepath

    def _build_agent_info(self, agent: str) -> Dict[str, Any]:
        idx = self.agent_to_idx[agent]
        team = int(self.player_team[idx])
        role_int = int(self.player_role[idx])
        if role_int == ROLE_KEEPER:
            role_str = "keeper"
        elif role_int == ROLE_DEF:
            role_str = "def"
        elif role_int == ROLE_MID:
            role_str = "mid"
        else:
            role_str = "atk"
        return {
            "team": team,
            "role": role_str,
            "policy_id": "keeper_policy" if role_int == ROLE_KEEPER else "field_policy",
            "team_side": int(self.team_side[team]),
            "rl_step": self.rl_step_count,
            "physics_frame": self.physics_frame_count,
            "match_id": self.current_match_id,
            "generation": self.current_generation,
            "interest_score": self.compute_interest_score(),
            "final_stats": dict(self.stats),
            "last_saved_replay": self.last_saved_replay_path,
        }

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
