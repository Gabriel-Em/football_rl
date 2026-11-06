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


def _fast_clip(val: float, low: float, high: float) -> float:
    """Fast inline scalar clamp avoiding NumPy dispatch overhead."""
    if val < low:
        return low
    if val > high:
        return high
    return val


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
        self.max_pitch_diag: float = float(self._max_field_dist)

        # Tactical Pocket Smoothing & Observation Vectors
        self.active_tactical_target_canon: np.ndarray = np.zeros((self.num_players, 2), dtype=np.float32)
        self.raw_steer_input: np.ndarray = np.zeros(self.num_players, dtype=np.float32)
        self.prev_best_cand_canon: np.ndarray = np.zeros((self.num_players, 2), dtype=np.float32)
        self.smoothed_pocket_canon: np.ndarray = np.zeros((self.num_players, 2), dtype=np.float32)
        self.pocket_initialized: np.ndarray = np.zeros(self.num_players, dtype=bool)

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
        self.is_ghost: np.ndarray = np.zeros(self.num_players, dtype=bool)
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
        self.ball_wall_bounces: int = 0
        self.is_wall_pass_in_flight: bool = False
        self.pinned_ball_timer: float = 0.0
        self.kickoff_protection_timer: float = 0.0
        self.pickup_tackle_grace_timer: float = 0.0
        self.post_save_box_grace_timer: float = 0.0
        self.active_box_radius: np.ndarray = np.array(
            [self.cfg.physics.keeper_box_radius, self.cfg.physics.keeper_box_radius],
            dtype=np.float64,
        )
        self.kick_frame_player_pos: np.ndarray = np.zeros((self.num_players, 2), dtype=np.float64)
        self.prev_longitudinal_vel: np.ndarray = np.zeros(self.num_players, dtype=np.float64)
        self.prev_ball_x: float = 700.0
        self.step_external_push: np.ndarray = np.zeros((self.num_players, 2), dtype=np.float64)
        self.prev_ball_at_step_start: np.ndarray = np.zeros(2, dtype=np.float64)
        self.smoothed_steer: np.ndarray = np.zeros(self.num_players, dtype=np.float64)
        self.prev_steer: np.ndarray = np.zeros(self.num_players, dtype=np.float64)
        self.dive_dir: np.ndarray = np.zeros((self.num_players, 2), dtype=np.float64)
        self.prev_dynamic_role_key: Dict[int, str] = {i: self.player_role_key[i] for i in range(self.num_players)}

        # Pass chains & pairwise history tracking
        self.team_pass_chain: List[int] = [0, 0]
        self.last_pass_pair: Optional[Tuple[int, int]] = None
        self.last_passer_idx: Optional[int] = None
        self.pairwise_repeat_count: int = 0
        self.last_kick_pos: np.ndarray = np.zeros((self.num_players, 2), dtype=np.float64)
        self.last_kick_hold_time: np.ndarray = np.zeros(self.num_players, dtype=np.float64)
        self.primary_loose_ball_chaser: np.ndarray = np.full(2, -1, dtype=np.int32)
        self.dive_speed: np.ndarray = np.zeros(self.num_players, dtype=np.float64)

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

    @property
    def phys_cfg(self) -> PhysicsConfig:
        return self.cfg.physics

    @property
    def mech_cfg(self) -> MechanicsConfig:
        return self.cfg.mechanics

    @property
    def vision_cfg(self) -> VisionConfig:
        return self.cfg.vision

    @property
    def rew_cfg(self) -> RewardConfig:
        return self.cfg.rewards

    @property
    def pass_chain_count(self) -> List[int]:
        return self.team_pass_chain

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Convenience alias for _get_observations()."""
        return self._get_observations()

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
            return (mech.tackle_radius_def, mech.tackle_prob_def)
        elif role == ROLE_MID:
            return (mech.tackle_radius_mid, mech.tackle_prob_mid)
        elif role == ROLE_ATK:
            return (mech.tackle_radius_atk, mech.tackle_prob_atk)
        else:
            return (mech.tackle_radius_mid, mech.tackle_prob_mid)

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
        max_reach = (float(np.linalg.norm(ball_vel)) ** 2) / (2.0 * phys.ball_friction_decel) + 35.0
        return bool(
            dist_to_goal_line <= max_reach
            and (phys.goal_y_min - y_margin) <= y_intersect <= (phys.goal_y_max + y_margin)
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
            "field_5s_violations": 0,
            "field_3s_violations": 0,  # alias for backwards compatibility
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
        self.pickup_tackle_grace_timer = 0.0
        self.prev_longitudinal_vel[:] = 0.0
        self.step_external_push.fill(0.0)
        self.prev_ball_at_step_start = self.ball_pos.copy()
        self.smoothed_steer.fill(0.0)
        self.prev_steer.fill(0.0)
        self.dive_dir.fill(0.0)
        self.prev_dynamic_role_key = {i: self.player_role_key[i] for i in range(self.num_players)}
        self.last_kick_pos.fill(0.0)
        self.last_kick_hold_time.fill(0.0)
        self.primary_loose_ball_chaser.fill(-1)
        self.dive_speed.fill(0.0)
        self.raw_steer_input.fill(0.0)
        self.prev_best_cand_canon.fill(0.0)
        self.smoothed_pocket_canon.fill(0.0)
        self.pocket_initialized[:] = False
        self._reset_pass_chains()

        # Set initial kickoff formation
        self._setup_kickoff(starting_team=self.first_half_kickoff_team)

        # Initialize tactical target pockets to initial canonical positions
        c_p_a, _, _ = self._to_team_canonical(self.player_pos, self.player_vel, self.player_angle, TEAM_A)
        c_p_b, _, _ = self._to_team_canonical(self.player_pos, self.player_vel, self.player_angle, TEAM_B)
        self.active_tactical_target_canon[:6] = c_p_a[:6]
        self.active_tactical_target_canon[6:] = c_p_b[6:]

        # Record initial frame (t = 0.0s) if recording enabled
        if self.cfg.replay.enable_recording:
            self._record_replay_frame()

        observations = self._get_observations()
        central_states = self.get_centralized_state()
        infos = {
            agent: self._build_agent_info(agent, central_state=central_states[self.agent_to_idx[agent]])
            for agent in self.agents
        }
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

        # Store raw steering input for instantaneous kick deflection
        self.raw_steer_input = act_array[:, 1].astype(np.float32)

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
                    <= 1.5 * self._get_role_tackle_params(idx)[0]
                )
                if not valid_opp_near:
                    step_rewards[idx] += rew.reckless_tackle_spam_step_pen

        # Run 4 physics sub-steps (60 Hz physics / 15 Hz RL)
        self.prev_ball_at_step_start = self.ball_pos.copy()
        self.step_external_push[:] = 0.0
        for _ in range(self.cfg.physics.frame_skip):
            self._frame_events = []
            prev_ball = (
                self.player_pos[self.ball_holder].copy()
                if self.ball_holder is not None
                else self.ball_pos.copy()
            )
            prev_holder = self.ball_holder
            teleported = self._physics_substep(act_array, step_rewards)
            if teleported:
                any_teleport = True
                act_array = np.zeros_like(act_array)
            else:
                curr_ref_pos = (
                    self.player_pos[self.ball_holder]
                    if (self.ball_holder is not None and self.ball_holder == prev_holder)
                    else self.ball_pos
                )
                for team_id in (TEAM_A, TEAM_B):
                    enemy_goal = np.array(
                        self.cfg.physics.right_goal_center
                        if self.team_side[team_id] == SIDE_LEFT
                        else self.cfg.physics.left_goal_center,
                        dtype=np.float64,
                    )
                    d_before = float(np.linalg.norm(prev_ball - enemy_goal))
                    d_after = float(np.linalg.norm(curr_ref_pos - enemy_goal))
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
            is_field_carrier = self._is_field_role(self.player_role[carrier_idx])

            # Gate possession tick: <= 2.5s for field players, <= 1.2s for Goalkeepers (prevents stall farming)
            max_poss_tick_time = 2.5 if is_field_carrier else 1.2
            if self.ball_hold_timer <= max_poss_tick_time:
                step_rewards[carrier_idx] += self.cfg.rewards.possession_carrier_ind
                self._add_team_reward(step_rewards, carrier_team, self.cfg.rewards.possession_team)

            carrier_adv = float(net_goal_advance[carrier_team])
            if carrier_adv > 0.0:
                long_v_car = float(
                    self.player_vel[carrier_idx, 0] * math.cos(self.player_angle[carrier_idx])
                    + self.player_vel[carrier_idx, 1] * math.sin(self.player_angle[carrier_idx])
                )
                # Only reward forward dribbling (never backpedaling into the enemy semicircle)
                if long_v_car >= -5.0:
                    role_adv_mult = 1.0
                    if self.player_role[carrier_idx] == ROLE_DEF:
                        c_car_x = (
                            float(self.player_pos[carrier_idx, 0])
                            if self.team_side[carrier_team] == SIDE_LEFT
                            else float(self.cfg.physics.field_width - self.player_pos[carrier_idx, 0])
                        )
                        if c_car_x > 680.0:
                            role_adv_mult = 0.25
                    hold_decay = (
                        1.0
                        if self.ball_hold_timer <= self.cfg.rewards.carrier_dribble_decay_start
                        else max(
                            0.20,
                            1.0 - (self.ball_hold_timer - self.cfg.rewards.carrier_dribble_decay_start) / 2.0,
                        )
                    )
                    step_rewards[carrier_idx] += (
                        self.cfg.rewards.carrier_goal_advance_ind_scale * carrier_adv * role_adv_mult * hold_decay
                    )

            # Escalating Hold-Timer Urgency Penalty
            if is_field_carrier and self.ball_hold_timer > 3.2:
                urg_ratio = (self.ball_hold_timer - 3.2) / max(
                    1e-6, self.cfg.mechanics.field_max_hold_time - 3.2
                )
                step_rewards[carrier_idx] -= (
                    self.cfg.rewards.hold_urgency_pen_scale * (urg_ratio ** 2)
                )
            elif (not is_field_carrier) and self.ball_hold_timer > self.cfg.rewards.keeper_hold_urgency_start:
                gk_urg = (
                    (self.ball_hold_timer - self.cfg.rewards.keeper_hold_urgency_start)
                    / max(1e-6, self.cfg.mechanics.keeper_max_hold_time - self.cfg.rewards.keeper_hold_urgency_start)
                ) ** 2
                step_rewards[carrier_idx] -= self.cfg.rewards.keeper_hold_urgency_scale * gk_urg

            # Prime-Scoring-Zone Hesitation & Stall Penalty
            if is_field_carrier and self.kick_grace_timer <= 0.0:
                enemy_goal_c = (
                    np.array(self.cfg.physics.right_goal_center, dtype=np.float64)
                    if self.team_side[carrier_team] == SIDE_LEFT
                    else np.array(self.cfg.physics.left_goal_center, dtype=np.float64)
                )
                dist_to_enemy_goal = float(
                    np.linalg.norm(self.player_pos[carrier_idx] - enemy_goal_c)
                )
                if dist_to_enemy_goal <= 360.0:
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
                    is_aimed_at_goal = self._is_ball_heading_into_goal(
                        hyp_ball_pos, hyp_ball_vel, enemy_side
                    )
                    is_arc_stall = (
                        dist_to_enemy_goal <= 235.0
                        and self.ball_hold_timer >= 0.75
                        and carrier_adv <= 1.5
                    )
                    if is_aimed_at_goal or is_arc_stall or (
                        dist_to_enemy_goal <= 320.0
                        and self.ball_hold_timer >= 1.8
                        and carrier_adv <= 1.0
                    ):
                        step_rewards[carrier_idx] += self.cfg.rewards.prime_shot_hesitation_pen

        # 2. Dense Ball-Advance Shaping towards Enemy Goal Center (1400, 420) / (0, 420)
        skip_advance_shaping = False
        if self.ball_kicked_in_flight and self.last_ball_team is not None:
            if float(net_goal_advance[self.last_ball_team]) < 0.0:
                skip_advance_shaping = True

        if not skip_advance_shaping:
            for team_id in (TEAM_A, TEAM_B):
                adv = float(net_goal_advance[team_id])
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

        central_states = self.get_centralized_state()
        infos = {
            agent: self._build_agent_info(agent, central_state=central_states[self.agent_to_idx[agent]])
            for agent in self.agents
        }

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
        self.is_ghost[:] = (self.ghost_timer > 0.0) | (self.stun_timer > 0.0)
        self.tackle_cooldown = np.maximum(0.0, self.tackle_cooldown - dt)
        self.lunge_cooldown = np.maximum(0.0, self.lunge_cooldown - dt)
        self.lunge_timer = np.maximum(0.0, self.lunge_timer - dt)
        self.pickup_cooldown = np.maximum(0.0, self.pickup_cooldown - dt)
        self.kick_grace_timer = max(0.0, self.kick_grace_timer - dt)
        self.shot_on_target_cd = np.maximum(0.0, self.shot_on_target_cd - dt)
        self.popup_cooldown = np.maximum(0.0, self.popup_cooldown - dt)
        self.kickoff_protection_timer = max(0.0, self.kickoff_protection_timer - dt)
        self.pickup_tackle_grace_timer = max(0.0, self.pickup_tackle_grace_timer - dt)
        self.post_save_box_grace_timer = max(0.0, self.post_save_box_grace_timer - dt)

        # Clear tracking flags when cooldown naturally expires
        expired_mask = self.pickup_cooldown <= 0.0
        self.kick_cooldown_active[expired_mask] = False
        self.violation_cooldown_active[expired_mask] = False

        # Track possession stats & hold timer
        left_center = np.array(phys.left_goal_center, dtype=np.float64)
        right_center = np.array(phys.right_goal_center, dtype=np.float64)
        d_left = float(np.linalg.norm(self.ball_pos - left_center))
        d_right = float(np.linalg.norm(self.ball_pos - right_center))
        box_limit = phys.keeper_box_radius + phys.ball_radius

        if self.ball_holder is not None:
            self.box_loose_ball_timer = 0.0
            # Smoothly re-expand both semicircles back to nominal keeper_box_radius (195.0u)
            for s_idx in (0, 1):
                if self.active_box_radius[s_idx] < phys.keeper_box_radius:
                    self.active_box_radius[s_idx] = min(
                        phys.keeper_box_radius,
                        float(self.active_box_radius[s_idx]) + mech.box_expand_rate * dt,
                    )
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
            elif (
                holder_role == ROLE_KEEPER
                and self.ball_hold_timer >= mech.keeper_max_hold_time - 1e-9
            ):
                self._handle_keeper_5s_violation(self.ball_holder, step_rewards, is_hold_violation=True)
                return True
        else:
            # Ball is loose: check if sitting inside either Keeper's nominal 195u semicircle
            in_left_box = d_left <= box_limit
            in_right_box = d_right <= box_limit
            ball_spd = float(np.linalg.norm(self.ball_vel))
            ball_stopped_in_box = (in_left_box or in_right_box) and (ball_spd <= mech.box_ball_stopped_threshold)

            if ball_stopped_in_box:
                active_side = SIDE_LEFT if in_left_box else SIDE_RIGHT
                other_side = 1 - active_side
                if self.post_save_box_grace_timer <= 0.0:
                    self.box_loose_ball_timer += dt
                # After box_shrink_grace_time (3.0s of completely stopped ball), shrink active_box_radius[active_side]
                if self.box_loose_ball_timer > mech.box_shrink_grace_time:
                    self.active_box_radius[active_side] = max(
                        mech.box_min_radius,
                        float(self.active_box_radius[active_side]) - mech.box_shrink_rate * dt,
                    )
                # Re-expand the opposite side if needed
                if self.active_box_radius[other_side] < phys.keeper_box_radius:
                    self.active_box_radius[other_side] = min(
                        phys.keeper_box_radius,
                        float(self.active_box_radius[other_side]) + mech.box_expand_rate * dt,
                    )
            else:
                self.box_loose_ball_timer = 0.0
                for s_idx in (0, 1):
                    if self.active_box_radius[s_idx] < phys.keeper_box_radius:
                        self.active_box_radius[s_idx] = min(
                            phys.keeper_box_radius,
                            float(self.active_box_radius[s_idx]) + mech.box_expand_rate * dt,
                        )

        # ----------------------------------------------------------------------
        # 2. Process Player Actions (Steering, Throttle, Kick, Special)
        # ----------------------------------------------------------------------
        current_holder = self.ball_holder

        for idx in range(self.num_players):
            if self.stun_timer[idx] > 0.0:
                if self.lunge_timer[idx] > 0.0:
                    pass  # Active kinematic lunge impulse
                else:
                    self.player_vel[idx] = 0.0
                continue

            throttle_raw, steer_raw, kick_power, special_trig = act_array[idx]

            # Steering smoothing filter: 0.65 * raw_steer + 0.35 * prev_steer
            smoothed_steer = 0.65 * steer_raw + 0.35 * self.prev_steer[idx]
            self.prev_steer[idx] = smoothed_steer
            self.smoothed_steer[idx] = smoothed_steer

            # Steering: [-1, 1] rotates facing_angle (locked during field lunge)
            if not (self._is_field_role(self.player_role[idx]) and self.lunge_timer[idx] > 0.0):
                turn_delta = smoothed_steer * phys.player_max_turn_rate * dt
                self.player_angle[idx] = float(wrap_to_pi(self.player_angle[idx] + turn_delta))

            # Special Action (action[3] > 0.5): Lunge for Keepers, Tackle for Field Players
            if special_trig > mech.special_trigger_threshold:
                if self.kickoff_protection_timer > 0.0 and self.ball_holder is not None:
                    pass
                elif self.player_role[idx] == ROLE_KEEPER:
                    if self.ball_holder != idx and self.lunge_cooldown[idx] <= 0.0:
                        self.lunge_timer[idx] = mech.lunge_duration
                        self.lunge_cooldown[idx] = mech.lunge_cooldown
                        d_to_ball = float(np.linalg.norm(self.player_pos[idx] - self.ball_pos))
                        k_team = int(self.player_team[idx])
                        k_side = int(self.team_side[k_team])
                        own_goal_x = 0.0 if k_side == SIDE_LEFT else phys.field_width
                        own_goal_center = np.array([own_goal_x, 420.0], dtype=np.float64)

                        # Check if ball heading toward own goal line
                        ball_vx = float(self.ball_vel[0])
                        ball_vy = float(self.ball_vel[1])
                        heading_toward_goal = (k_side == SIDE_LEFT and ball_vx < -40.0) or (
                            k_side == SIDE_RIGHT and ball_vx > 40.0
                        )

                        # Ball must be in front of keeper along goal direction (not already flown past keeper towards goal)
                        is_ball_in_front = (
                            (k_side == SIDE_LEFT and self.ball_pos[0] >= self.player_pos[idx, 0] - 15.0)
                            or (k_side == SIDE_RIGHT and self.ball_pos[0] <= self.player_pos[idx, 0] + 15.0)
                        )
                        t_goal = (
                            (own_goal_x - float(self.ball_pos[0])) / (ball_vx if abs(ball_vx) > 1e-6 else 1e-6)
                            if heading_toward_goal
                            else 999.0
                        )

                        u_dive = None
                        is_shot_at_keeper = False
                        dive_target_cpa = None
                        v_dive = mech.keeper_dive_speed
                        if heading_toward_goal and d_to_ball <= 480.0 and is_ball_in_front and t_goal > dt:
                            ball_spd_sq = float(np.dot(self.ball_vel, self.ball_vel))
                            ball_spd = math.sqrt(ball_spd_sq)
                            t_cpa = float(
                                np.clip(
                                    np.dot(self.player_pos[idx] - self.ball_pos, self.ball_vel)
                                    / max(ball_spd_sq, 1e-6),
                                    0.0,
                                    t_goal,
                                )
                            )
                            a_fric = phys.ball_friction_decel  # 155.0
                            t_stop = ball_spd / a_fric if a_fric > 0.0 else 999.0
                            t_eff = min(t_cpa, t_stop)
                            u_ball = self.ball_vel / max(ball_spd, 1e-6)
                            dist_cpa_along_ray = ball_spd * t_eff - 0.5 * a_fric * (t_eff ** 2)
                            p_cpa = self.ball_pos + dist_cpa_along_ray * u_ball
                            if k_side == SIDE_LEFT:
                                p_cpa[0] = max(phys.player_radius, p_cpa[0])
                            else:
                                p_cpa[0] = min(phys.field_width - phys.player_radius, p_cpa[0])
                            p_cpa[1] = float(np.clip(p_cpa[1], phys.goal_y_min, phys.goal_y_max))
                            dive_target_cpa = p_cpa.copy()

                            d_cpa = float(np.linalg.norm(p_cpa - self.player_pos[idx]))
                            smother_radius = phys.keeper_smother_cpa_radius  # 30.0
                            if d_cpa <= phys.player_radius:
                                is_shot_at_keeper = True
                                req_speed = d_cpa / max(dt, min(0.25, max(t_cpa, 0.08)))
                                v_dive = min(140.0, req_speed)
                                if d_cpa > 2.0:
                                    u_dive = (p_cpa - self.player_pos[idx]) / d_cpa
                                else:
                                    u_dive = -u_ball
                            elif d_cpa <= smother_radius:
                                is_shot_at_keeper = False
                                req_speed = d_cpa / max(dt, min(0.25, max(t_cpa, 0.08)))
                                v_dive = min(140.0, req_speed)
                                u_dive = (p_cpa - self.player_pos[idx]) / d_cpa
                            else:
                                is_shot_at_keeper = False
                                u_dive = (p_cpa - self.player_pos[idx]) / d_cpa
                                time_window = max(0.06, min(0.25, t_cpa if t_cpa > 0.04 else 0.20))
                                v_dive = float(np.clip(d_cpa / time_window, 160.0, mech.keeper_dive_speed))
                        elif (
                            self.ball_holder is None
                            and float(np.linalg.norm(self.ball_pos - own_goal_center))
                            <= phys.keeper_box_radius
                        ):
                            dive_vec = self.ball_pos - self.player_pos[idx]
                            dive_dist = float(np.linalg.norm(dive_vec))
                            if dive_dist > 1e-6:
                                u_dive = dive_vec / dive_dist
                            v_dive = mech.keeper_dive_speed

                        if u_dive is None:
                            steer_input = float(smoothed_steer)
                            if abs(steer_input) > 0.1:
                                sign_y = 1.0 if steer_input > 0.0 else -1.0
                                u_dive = np.array([0.0, sign_y], dtype=np.float64)
                            else:
                                u_dive = np.array(
                                    [
                                        math.cos(self.player_angle[idx]),
                                        math.sin(self.player_angle[idx]),
                                    ],
                                    dtype=np.float64,
                                )
                            v_dive = mech.keeper_dive_speed

                        # Prevent diving into the back wall behind own goal line
                        if (
                            k_side == SIDE_LEFT
                            and self.player_pos[idx, 0] <= phys.player_radius + 2.0
                            and u_dive[0] < 0.0
                        ):
                            u_dive[0] = 0.0
                            norm_u = float(np.linalg.norm(u_dive))
                            if norm_u > 1e-6:
                                u_dive = u_dive / norm_u
                        elif (
                            k_side == SIDE_RIGHT
                            and self.player_pos[idx, 0]
                            >= phys.field_width - phys.player_radius - 2.0
                            and u_dive[0] > 0.0
                        ):
                            u_dive[0] = 0.0
                            norm_u = float(np.linalg.norm(u_dive))
                            if norm_u > 1e-6:
                                u_dive = u_dive / norm_u

                        self.dive_dir[idx] = u_dive
                        self.dive_speed[idx] = v_dive
                        self.player_vel[idx] = u_dive * self.dive_speed[idx]

                        # Dive Save Gating & Anti-Farming: verify ball is genuinely heading into net
                        ball_spd = float(np.linalg.norm(self.ball_vel))
                        c_ball_vel_x = self.ball_vel[0] if k_side == SIDE_LEFT else -self.ball_vel[0]
                        heading_goal_check = (
                            is_ball_in_front
                            and (not (heading_toward_goal and t_goal <= dt))
                            and (
                                self._is_ball_heading_into_goal(
                                    self.ball_pos, self.ball_vel, k_side, y_margin=15.0
                                )
                                or (self.ball_kick_was_shot_on_target and c_ball_vel_x < -100.0)
                            )
                        )

                        is_in_box = (
                            float(np.linalg.norm(self.ball_pos - own_goal_center))
                            <= phys.keeper_box_radius + 25.0
                        )
                        d_cpa_val = d_cpa if dive_target_cpa is not None else 999.0

                        valid_dive_trajectory = True
                        if heading_toward_goal and dive_target_cpa is not None and not is_shot_at_keeper:
                            to_target = dive_target_cpa - self.player_pos[idx]
                            d_tgt = float(np.linalg.norm(to_target))
                            if d_tgt > 1e-6:
                                valid_dive_trajectory = (float(np.dot(u_dive, to_target / d_tgt)) >= 0.35)
                            else:
                                valid_dive_trajectory = False

                        if (
                            self.ball_holder is None
                            and heading_goal_check
                            and not is_shot_at_keeper
                            and valid_dive_trajectory
                            and d_cpa_val > 18.0
                            and d_to_ball <= 480.0
                        ):
                            if step_rewards is not None:
                                step_rewards[idx] += self.cfg.rewards.keeper_dive_save_attempt_ind
                            self.post_save_box_grace_timer = mech.post_save_box_grace_time
                            self._emit_event(
                                "GK DIVE!",
                                self.player_pos[idx],
                                team=k_team,
                                agent=self.idx_to_agent[idx],
                                reward=self.cfg.rewards.keeper_dive_save_attempt_ind,
                            )
                        elif (
                            (is_in_box and ball_spd < 80.0)
                            or (dive_target_cpa is not None and is_shot_at_keeper)
                        ):
                            if step_rewards is not None:
                                step_rewards[idx] += self.cfg.rewards.keeper_unnecessary_dive_pen
                            self._emit_event(
                                "UNNECESSARY DIVE!",
                                self.player_pos[idx],
                                team=k_team,
                                agent=self.idx_to_agent[idx],
                                reward=self.cfg.rewards.keeper_unnecessary_dive_pen,
                            )
                        elif (
                            (d_to_ball > 480.0 or not is_ball_in_front or not valid_dive_trajectory)
                            and self.ball_holder != idx
                            and not is_shot_at_keeper
                        ):
                            if step_rewards is not None:
                                step_rewards[idx] += self.cfg.rewards.keeper_reckless_lunge_pen
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
                # Sub-Step Ray-Lock Braking: damp perpendicular velocity near shot line to avoid overshooting
                if self.ball_holder is None:
                    k_team = int(self.player_team[idx])
                    k_side = int(self.team_side[k_team])
                    v_x_ball_canon = float(self.ball_vel[0] if k_side == SIDE_LEFT else -self.ball_vel[0])
                    if v_x_ball_canon < -40.0:
                        b_spd = float(np.linalg.norm(self.ball_vel))
                        if b_spd > 1e-6:
                            u_b = self.ball_vel / b_spd
                            rel_pos = self.player_pos[idx] - self.ball_pos
                            s_proj = float(np.dot(rel_pos, u_b))
                            if s_proj > 0.0:
                                perp_vec = rel_pos - s_proj * u_b
                                d_perp = float(np.linalg.norm(perp_vec))
                                if d_perp <= 10.0:
                                    v_perp_vec = self.player_vel[idx] - np.dot(self.player_vel[idx], u_b) * u_b
                                    self.player_vel[idx] = self.player_vel[idx] - 0.85 * v_perp_vec
            elif self._is_field_role(self.player_role[idx]) and self.lunge_timer[idx] > 0.0:
                # Field player lunge: instant override at field_lunge_speed directed at target,
                # standard exponential acceleration smoothing is bypassed for the duration
                pass
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

        # Semi-Circle Barrier & Field Wall Clamping (apply smooth push on first pass)
        self._clamp_players_to_boundaries(apply_dynamic_push=True)

        # Re-run boundary pass once more without dynamic push to guarantee hard constraints
        self._clamp_players_to_boundaries(apply_dynamic_push=False)

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

            # Pass Flight Timeout
            if self.ball_kicked_in_flight:
                self.ball_kick_flight_timer += dt
                max_flight = (
                    mech.max_wall_pass_flight_time
                    if self.is_wall_pass_in_flight
                    else mech.max_pass_flight_time
                )
                if self.ball_kick_flight_timer > max_flight:
                    self.ball_kicked_in_flight = False
                    self.is_wall_pass_in_flight = False
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
            in_any_box = (
                d_left_box <= phys.keeper_box_radius + phys.ball_radius
                or d_right_box <= phys.keeper_box_radius + phys.ball_radius
            )
            near_box_edge = (not in_any_box) and (
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
    def _simulate_ball_trajectory(
        self,
        start_pos: np.ndarray,
        start_vel: np.ndarray,
        num_samples: int = 16,
        t_min: float = 0.35,
        t_max: float = 2.80,
    ) -> Tuple[np.ndarray, int, float]:
        """Simulate ball trajectory under linear friction (155.0 u/s^2) and up to 1 sideline wall bounce.
        Returns:
            sample_points: shape (num_samples, 2) in world coordinates
            n_side_bounces: 0 or 1
            first_wall_x_world: X-coordinate where ball hit sideline wall (or -1.0 if no bounce)
        """
        v0 = float(np.linalg.norm(start_vel))
        if v0 < 1e-4:
            pts = np.repeat(start_pos[None, :], num_samples, axis=0)
            return pts, 0, -1.0

        a_fric = self.cfg.physics.ball_friction_decel  # 155.0
        e_wall = self.cfg.physics.ball_wall_restitution  # 0.85
        t_stop = v0 / a_fric
        u0 = start_vel / v0

        # Check if trajectory hits top (y=10) or bottom (y=830) sideline before stopping
        d_stop = 0.5 * v0 * t_stop
        vy0 = float(start_vel[1])
        t_wall = 999.0
        wall_y = 0.0
        if abs(vy0) > 1e-4:
            target_y = 10.0 if vy0 < 0.0 else (self.cfg.physics.field_height - 10.0)
            dy_wall = abs(target_y - float(start_pos[1]))
            # Distance along ray to reach target_y
            sin_abs = abs(float(u0[1]))
            s_wall = dy_wall / max(sin_abs, 1e-6)
            if s_wall < d_stop:
                # Solve v0 * t - 0.5 * a_fric * t^2 = s_wall
                disc = max(0.0, v0 * v0 - 2.0 * a_fric * s_wall)
                t_wall = (v0 - math.sqrt(disc)) / a_fric
                wall_y = target_y

        times = np.linspace(t_min, t_max, num_samples, dtype=np.float32)
        pts = np.zeros((num_samples, 2), dtype=np.float32)
        n_bounces = 0
        wall_x_world = -1.0

        if t_wall < t_stop:
            n_bounces = 1
            s_at_wall = v0 * t_wall - 0.5 * a_fric * (t_wall ** 2)
            p_wall = start_pos + s_at_wall * u0
            p_wall[1] = wall_y
            wall_x_world = float(p_wall[0])
            v_before_wall = max(0.0, v0 - a_fric * t_wall)
            v_after_wall = e_wall * v_before_wall
            u_after = np.array([u0[0], -u0[1]], dtype=np.float32)
            t_stop_after = v_after_wall / a_fric
        else:
            p_wall = start_pos
            v_after_wall = 0.0
            u_after = u0
            t_stop_after = 0.0

        for idx, t in enumerate(times):
            t_val = float(t)
            if t_val <= t_wall:
                t_eff = min(t_val, t_stop)
                s = v0 * t_eff - 0.5 * a_fric * (t_eff ** 2)
                p = start_pos + s * u0
            else:
                dt_after = min(t_val - t_wall, t_stop_after)
                s_after = v_after_wall * dt_after - 0.5 * a_fric * (dt_after ** 2)
                p = p_wall + s_after * u_after
            p[0] = np.clip(p[0], 10.0, self.cfg.physics.field_width - 10.0)
            p[1] = np.clip(p[1], 10.0, self.cfg.physics.field_height - 10.0)
            pts[idx] = p

        return pts, n_bounces, wall_x_world

    def _execute_kick(
        self,
        kicker_idx: int,
        kick_power: float,
        step_rewards: Optional[np.ndarray] = None,
    ) -> None:
        """Launch the ball from kicker with role-specific speed & jitter, check shot-on-target, and apply cooldown."""
        self.kickoff_protection_timer = 0.0
        self.pickup_tackle_grace_timer = 0.0
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

        max_defl_rad = math.radians(mech.kick_steer_max_deflection_deg)
        steer_defl = float(self.raw_steer_input[kicker_idx]) * max_defl_rad
        base_angle = float(wrap_to_pi(self.player_angle[kicker_idx] + steer_defl))
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

        # Option A: Immediate Defensive-Third Own-Goal Kick Threat Shaping (resolves temporal credit assignment delay)
        if step_rewards is not None and kicker_role != ROLE_KEEPER:
            own_side = int(self.team_side[kicker_team])
            c_kp, _, _ = self._to_team_canonical(
                self.player_pos[kicker_idx : kicker_idx + 1], np.zeros((1, 2)), 0.0, kicker_team
            )
            if float(c_kp[0, 0]) <= 350.0:
                if self._is_ball_heading_into_goal(self.ball_pos, self.ball_vel, own_side, y_margin=25.0):
                    step_rewards[kicker_idx] += rew.dangerous_own_goal_kick_pen

        self.last_kick_pos[kicker_idx] = self.ball_pos.copy()
        self.last_kick_hold_time[kicker_idx] = self.ball_hold_timer
        self.kick_frame_player_pos = self.player_pos.copy()
        self.ball_holder = None
        self.ball_hold_timer = 0.0
        self.last_ball_holder = kicker_idx
        self.last_ball_team = kicker_team
        self.ball_kick_origin = self.ball_pos.copy()
        self.ball_kicked_in_flight = True
        self.ball_kick_flight_timer = 0.0
        self.ball_wall_bounces = 0
        self.is_wall_pass_in_flight = False

        # 1.0s pickup cooldown on kicker
        self.pickup_cooldown[kicker_idx] = max(
            self.pickup_cooldown[kicker_idx], mech.kick_pickup_cooldown
        )
        self.kick_cooldown_active[kicker_idx] = True

        if kicker_role == ROLE_KEEPER:
            self.last_keeper_kick_time[kicker_team] = self.physics_frame_count * phys.dt

        # Check if this kick is a True Shot on Target (inside goal mouth)
        # or a Near Shot (hitting the end wall beside the goal within 75u)
        enemy_side = 1 - int(self.team_side[kicker_team])
        is_true_sot = self._is_ball_heading_into_goal(self.ball_pos, self.ball_vel, enemy_side, y_margin=0.0)
        is_near_sot = (
            False
            if is_true_sot
            else self._is_ball_heading_into_goal(self.ball_pos, self.ball_vel, enemy_side, y_margin=75.0)
        )
        cx = 0.0
        cy = 420.0
        flank_angle_mult = 1.0
        if is_true_sot or is_near_sot:
            c_kicker_pos, _, _ = self._to_team_canonical(
                self.player_pos[kicker_idx:kicker_idx + 1],
                self.player_vel[kicker_idx:kicker_idx + 1],
                self.player_angle[kicker_idx],
                kicker_team,
            )
            cx = float(c_kicker_pos[0, 0])
            cy = float(c_kicker_pos[0, 1])
            if cx >= 1210.0 and abs(cy - 420.0) >= 135.0:
                flank_angle_mult = 0.35
            # A shot on target must originate from attacking territory (cx >= 680.0)
            if cx < 680.0:
                is_true_sot = False
                is_near_sot = False
            else:
                # Disqualify if this kick is aimed at an intermediate teammate in the pass reception corridor
                ball_spd = float(np.linalg.norm(self.ball_vel))
                if ball_spd > 1e-6:
                    u_kick = self.ball_vel / ball_spd
                    goal_x = 0.0 if enemy_side == SIDE_LEFT else phys.field_width
                    dist_to_goal = abs(goal_x - float(self.ball_pos[0]))
                    tau_lead = rew.pass_lead_tau
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
                                    if dist_perp <= 45.0 or cos_angle >= 0.95:  # within ~18 deg pass corridor
                                        is_true_sot = False
                                        is_near_sot = False
                                        break
                            if not is_true_sot and not is_near_sot:
                                break

        self.ball_kick_was_shot_on_target = is_true_sot

        if is_true_sot:
            self.stats["shots"][kicker_team] += 1
            if step_rewards is not None and self.shot_on_target_cd[kicker_idx] <= 0.0:
                prime_zone_mult = 1.0 if cx >= 980.0 else 0.55
                power_factor = 0.6 + 0.4 * power_norm
                base_sot = (
                    rew.atk_shot_on_target_ind
                    if kicker_role == ROLE_ATK
                    else rew.other_shot_on_target_ind
                )
                chain_len = self.team_pass_chain[kicker_team]
                chain_mult = 1.0 + (rew.pass_chain_multiplier_coeff * chain_len)

                sot_rew = base_sot * prime_zone_mult * flank_angle_mult * power_factor * chain_mult
                team_sot_rew = rew.shot_on_target_team * prime_zone_mult * flank_angle_mult * chain_mult
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
        elif is_near_sot:
            # Near Shot (hitting the end wall beside the goal): reward shooting ambition with 35% of shot reward
            # Does NOT increment shot-on-target stats or set ball_kick_was_shot_on_target
            if step_rewards is not None and self.shot_on_target_cd[kicker_idx] <= 0.0:
                prime_zone_mult = 1.0 if cx >= 980.0 else 0.55
                power_factor = 0.6 + 0.4 * power_norm
                base_sot = (
                    rew.atk_shot_on_target_ind
                    if kicker_role == ROLE_ATK
                    else rew.other_shot_on_target_ind
                )
                chain_len = self.team_pass_chain[kicker_team]
                chain_mult = 1.0 + (rew.pass_chain_multiplier_coeff * chain_len)

                near_sot_rew = rew.shot_near_target_mult * base_sot * prime_zone_mult * flank_angle_mult * power_factor * chain_mult
                near_team_sot_rew = rew.shot_near_target_mult * rew.shot_on_target_team * prime_zone_mult * flank_angle_mult * chain_mult
                step_rewards[kicker_idx] += near_sot_rew
                self._add_team_reward(step_rewards, kicker_team, near_team_sot_rew)
                self.shot_on_target_cd[kicker_idx] = 1.5
                self._emit_event(
                    "NEAR SHOT!",
                    self.player_pos[kicker_idx],
                    team=kicker_team,
                    agent=self.idx_to_agent[kicker_idx],
                    reward=near_sot_rew + near_team_sot_rew,
                )

        # Keeper distribution, wall bypass & Gaussian long pass evaluation
        if step_rewards is not None and kicker_role == ROLE_KEEPER:
            cos_kick_canon = (
                float(kick_vec[0])
                if self.team_side[kicker_team] == SIDE_LEFT
                else float(-kick_vec[0])
            )
            if cos_kick_canon > 0.20:
                opp_team = 1 - kicker_team
                opp_indices = [idx for idx in range(self.num_players) if self.player_team[idx] == opp_team]
                kick_dir = kick_vec / max(1e-6, float(np.linalg.norm(kick_vec)))

                # Dynamic time-of-flight interception check along the initial kick ray
                corridor_clear = True
                for opp_idx in opp_indices:
                    rel_opp = self.player_pos[opp_idx] - self.ball_pos
                    s_proj = float(np.dot(rel_opp, kick_dir))
                    if 0.0 < s_proj <= 380.0:
                        d_perp = float(abs(rel_opp[0] * kick_dir[1] - rel_opp[1] * kick_dir[0]))
                        t_ball_reach = s_proj / max(launch_speed, 260.0)
                        # Opponent can reach pickup radius (28u) + 45% of max speed * t_ball_reach
                        opp_reach_radius = 28.0 + 0.45 * 190.0 * t_ball_reach
                        if d_perp < opp_reach_radius:
                            corridor_clear = False
                            break

                if corridor_clear:
                    sim_pts, n_sim_bounces, wall_x_world = self._simulate_ball_trajectory(
                        self.ball_pos, self.ball_vel
                    )
                    sim_pts_canon, _, _ = self._to_team_canonical(
                        sim_pts, np.zeros_like(sim_pts), 0.0, kicker_team
                    )

                    # A. Deep Side-Wall Bypass Bonus (keeper_wall_bypass_deep_bonus = +1.40)
                    wall_bypass_awarded = False
                    if n_sim_bounces == 1 and power_norm >= 0.55:
                        wall_x_canon = (
                            wall_x_world
                            if self.team_side[kicker_team] == SIDE_LEFT
                            else (phys.field_width - wall_x_world)
                        )
                        if 150.0 <= wall_x_canon <= 1050.0:
                            end_pt_canon = sim_pts_canon[-1]
                            nearest_opp_dist_end = min(
                                (
                                    float(np.linalg.norm(self.player_pos[o] - sim_pts[-1]))
                                    for o in opp_indices
                                ),
                                default=999.0,
                            )
                            if end_pt_canon[0] >= 680.0 and nearest_opp_dist_end >= 115.0:
                                step_rewards[kicker_idx] += rew.keeper_wall_bypass_deep_bonus * power_norm
                                wall_bypass_awarded = True
                                self._emit_event(
                                    "DEEP WALL BYPASS!",
                                    self.player_pos[kicker_idx],
                                    team=kicker_team,
                                    agent=self.idx_to_agent[kicker_idx],
                                    reward=rew.keeper_wall_bypass_deep_bonus * power_norm,
                                )
                            elif 150.0 <= wall_x_canon <= 750.0 and power_norm >= 0.65:
                                step_rewards[kicker_idx] += rew.keeper_wall_clear_bonus * power_norm
                                wall_bypass_awarded = True
                                self._emit_event(
                                    "WALL CLEARANCE!",
                                    self.player_pos[kicker_idx],
                                    team=kicker_team,
                                    agent=self.idx_to_agent[kicker_idx],
                                    reward=rew.keeper_wall_clear_bonus * power_norm,
                                )

                    # B. Continuous Gaussian Trajectory Reward Toward Unpressed Teammates
                    best_target_score = 0.0
                    for j in range(self.num_players):
                        if j != kicker_idx and self.player_team[j] == kicker_team:
                            p_lead_j = self.player_pos[j] + 0.60 * self.player_vel[j]
                            p_lead_j_c, _, _ = self._to_team_canonical(
                                p_lead_j.reshape(1, 2), np.zeros((1, 2)), 0.0, kicker_team
                            )
                            x_lead_canon = float(p_lead_j_c[0, 0])
                            if x_lead_canon >= 380.0:
                                d_opp_j = min(
                                    (
                                        float(np.linalg.norm(self.player_pos[o] - p_lead_j))
                                        for o in opp_indices
                                    ),
                                    default=999.0,
                                )
                                u_j = float(np.clip((d_opp_j - 65.0) / (220.0 - 65.0), 0.0, 1.0))
                                d_miss_j = min(
                                    float(np.linalg.norm(sim_pts[m] - p_lead_j))
                                    for m in range(len(sim_pts))
                                )
                                if d_miss_j <= rew.keeper_long_pass_max_miss:
                                    a_j = math.exp(-(d_miss_j ** 2) / (2.0 * (rew.keeper_long_pass_sigma ** 2)))
                                    f_j = float(np.clip((x_lead_canon - 350.0) / 650.0, 0.25, 1.0))
                                    score_j = u_j * a_j * f_j
                                    if score_j > best_target_score:
                                        best_target_score = score_j

                    if best_target_score > 0.0:
                        long_pass_reward = rew.keeper_long_pass_target_max_bonus * best_target_score
                        step_rewards[kicker_idx] += long_pass_reward
                        self._emit_event(
                            "GK TARGET PASS!",
                            self.player_pos[kicker_idx],
                            team=kicker_team,
                            agent=self.idx_to_agent[kicker_idx],
                            reward=long_pass_reward,
                        )
                    elif not wall_bypass_awarded:
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
                            * cos_kick_canon
                            * (0.5 + 0.5 * power_norm)
                            * aim_bonus
                        )

        # Immediate Kick-Direction Evaluation (for all players when step_rewards is not None)
        if step_rewards is not None:
            opp_team = 1 - kicker_team
            opp_indices = [idx for idx in range(self.num_players) if self.player_team[idx] == opp_team]
            tm_indices = [idx for idx in range(self.num_players) if self.player_team[idx] == kicker_team and idx != kicker_idx]

            # Check if any opponent lies within 0.0 < proj <= 185.0 and perp_dist <= 30.0 along kick_vec (and closer than any intended teammate)
            is_blocked = False
            for opp_idx in opp_indices:
                to_opp = self.player_pos[opp_idx] - self.ball_pos
                proj = float(np.dot(to_opp, kick_vec))
                if 0.0 < proj <= 185.0:
                    perp_dist = math.hypot(
                        to_opp[0] - proj * kick_vec[0],
                        to_opp[1] - proj * kick_vec[1],
                    )
                    if perp_dist <= 30.0:
                        has_closer_tm = False
                        for tm_idx in tm_indices:
                            to_tm = self.player_pos[tm_idx] - self.ball_pos
                            tm_proj = float(np.dot(to_tm, kick_vec))
                            if 0.0 < tm_proj < proj:
                                tm_perp = math.hypot(
                                    to_tm[0] - tm_proj * kick_vec[0],
                                    to_tm[1] - tm_proj * kick_vec[1],
                                )
                                if tm_perp <= 30.0:
                                    has_closer_tm = True
                                    break
                        if not has_closer_tm:
                            is_blocked = True
                            break

            if is_blocked:
                step_rewards[kicker_idx] += rew.blocked_kick_pen
            else:
                # Check if kick_vec is aimed at an open teammate or their kinematic projection
                aimed_at_teammate = False
                matched_tm_idx = -1
                for tm_idx in range(self.num_players):
                    if tm_idx != kicker_idx and self.player_team[tm_idx] == kicker_team:
                        d_ij = float(np.linalg.norm(self.player_pos[tm_idx] - self.ball_pos))
                        t_tof = float(np.clip(d_ij / max(launch_speed, 260.0), 0.12, 1.15))
                        target_pt = np.clip(
                            self.player_pos[tm_idx] + t_tof * self.player_vel[tm_idx],
                            [40.0, 40.0],
                            [phys.field_width - 40.0, phys.field_height - 40.0],
                        )
                        to_target = target_pt - self.ball_pos
                        d_target = float(np.linalg.norm(to_target))
                        if 75.0 <= d_target <= 560.0:
                            unit_target = to_target / d_target
                            if float(np.dot(kick_vec, unit_target)) >= 0.90:
                                lane_clear = True
                                for opp_idx in opp_indices:
                                    to_opp = self.player_pos[opp_idx] - self.ball_pos
                                    proj = float(np.dot(to_opp, unit_target))
                                    if 0.0 < proj < d_target:
                                        perp_dist = math.hypot(
                                            to_opp[0] - proj * unit_target[0],
                                            to_opp[1] - proj * unit_target[1],
                                        )
                                        if perp_dist < 36.0:
                                            lane_clear = False
                                            break
                                if lane_clear:
                                    aimed_at_teammate = True
                                    matched_tm_idx = tm_idx
                                    break

                if aimed_at_teammate:
                    aim_bonus = rew.pass_target_aim_ind
                    if self.last_passer_idx is not None and matched_tm_idx == self.last_passer_idx:
                        aim_bonus *= (rew.pass_pairwise_decay_base ** (self.pairwise_repeat_count + 1))
                    enemy_side = 1 - int(self.team_side[kicker_team])
                    enemy_goal_c = np.array(
                        phys.left_goal_center if enemy_side == SIDE_LEFT else phys.right_goal_center,
                        dtype=np.float64,
                    )
                    d_kicker_enemy_goal = float(np.linalg.norm(self.player_pos[kicker_idx] - enemy_goal_c))
                    if d_kicker_enemy_goal <= 250.0:
                        aim_bonus *= 1.5
                    step_rewards[kicker_idx] += aim_bonus
                elif not (is_true_sot or is_near_sot):
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
        """Execute kinematic lunge and dynamic distance-based tackle check when action[3] > 0.5."""
        if self.kickoff_protection_timer > 0.0 or self.pickup_tackle_grace_timer > 0.0:
            return

        mech = self.cfg.mechanics
        rew = self.cfg.rewards

        # 0. A physical tackle may ONLY execute when an opposing player currently holds the ball.
        # If the ball is loose (ball_holder is None) or tackler's OWN team holds the ball,
        # return immediately (zero stun, zero popup).
        holder = self.ball_holder
        if holder is None or self.player_team[holder] == self.player_team[tackler_idx]:
            return

        dist_to_ball = min(
            float(np.linalg.norm(self.player_pos[tackler_idx] - self.ball_pos)),
            float(np.linalg.norm(self.player_pos[tackler_idx] - self.player_pos[holder])),
        )
        if dist_to_ball > mech.tackle_whiff_max_ball_dist:
            return

        R_max_lunge, P_max = self._get_role_tackle_params(tackler_idx)
        aiming_range = 1.5 * R_max_lunge

        # Valid target: an opponent field player currently holding the ball
        has_carrier = self._is_field_role(self.player_role[holder])
        dist = (
            float(np.linalg.norm(self.player_pos[tackler_idx] - self.player_pos[holder]))
            if has_carrier
            else 999.0
        )

        # 1. Opponent carrier not within aiming range -> Whiff
        if (not has_carrier) or dist > aiming_range:
            self.stun_timer[tackler_idx] = mech.tackle_miss_stun_time
            self.player_vel[tackler_idx] = 0.0
            self.is_ghost[tackler_idx] = True
            self.tackle_cooldown[tackler_idx] = max(
                self.tackle_cooldown[tackler_idx], mech.tackle_fail_cooldown
            )
            if step_rewards is not None:
                step_rewards[tackler_idx] += rew.reckless_tackle_ind
            self.stats["failed_tackles"] += 1
            self._emit_event(
                "AIR TACKLE!",
                self.player_pos[tackler_idx],
                team=int(self.player_team[tackler_idx]),
                agent=self.idx_to_agent[tackler_idx],
                reward=rew.reckless_tackle_ind,
            )
            return

        # 2. Carrier within aiming range -> Override velocity to exactly field_lunge_speed directed at carrier
        to_carrier = self.player_pos[holder] - self.player_pos[tackler_idx]
        dir_vec = to_carrier / max(1e-6, dist)
        self.player_vel[tackler_idx] = dir_vec * mech.field_lunge_speed
        self.player_angle[tackler_idx] = float(math.atan2(dir_vec[1], dir_vec[0]))
        self.lunge_timer[tackler_idx] = mech.field_lunge_duration
        self.tackle_cooldown[tackler_idx] = max(
            self.tackle_cooldown[tackler_idx], mech.field_lunge_duration + mech.tackle_fail_cooldown
        )

        # 3. Dynamic Tackle Evaluation: P_success = P_max * (1.0 - (d / R_max_lunge)**2)
        if dist > R_max_lunge:
            # Outside maximum reach -> Automatic failure
            self.stun_timer[tackler_idx] = mech.tackle_miss_stun_time
            self.is_ghost[tackler_idx] = True
            if step_rewards is not None:
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

        p_success = float(P_max * (1.0 - (dist / R_max_lunge) ** 2))
        p_success = max(0.0, min(1.0, p_success))

        if self.np_random.random() < p_success:
            # Successful tackle!
            old_carrier = holder
            self._reset_pass_chains(int(self.player_team[old_carrier]))
            self.ball_holder = tackler_idx
            self.ball_hold_timer = 0.0
            self.kick_grace_timer = mech.pickup_kick_grace
            self.last_ball_holder = tackler_idx
            self.last_ball_team = int(self.player_team[tackler_idx])
            self.ball_kicked_in_flight = False

            self._on_new_player_touched_ball(tackler_idx)

            # Freeze and ghost the victim
            self.stun_timer[old_carrier] = mech.tackle_victim_stun_time
            self.player_vel[old_carrier] = 0.0
            self.is_ghost[old_carrier] = True
            self.tackle_cooldown[old_carrier] = max(
                self.tackle_cooldown[old_carrier], mech.tackled_opponent_cooldown
            )
            self.pickup_cooldown[old_carrier] = max(
                self.pickup_cooldown[old_carrier], mech.tackle_victim_stun_time
            )

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
                    self.player_pos[tackler_idx : tackler_idx + 1],
                    self.player_vel[tackler_idx : tackler_idx + 1],
                    0.0,
                    tackler_team,
                )
                if float(c_tackler_pos[0, 0]) < 700.0:
                    tackler_ind += rew.def_own_half_tackle_bonus

            if step_rewards is not None:
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
        self.stun_timer[tackler_idx] = mech.tackle_miss_stun_time
        self.is_ghost[tackler_idx] = True
        if step_rewards is not None:
            step_rewards[tackler_idx] += rew.failed_tackle_in_range_ind
        self.stats["failed_tackles"] += 1
        self._emit_event(
            "TACKLE FAILED!",
            self.player_pos[tackler_idx],
            team=int(self.player_team[tackler_idx]),
            agent=self.idx_to_agent[tackler_idx],
            reward=rew.failed_tackle_in_range_ind,
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

        is_teammate_pass = False
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
            kicker_role = int(self.player_role[kicker_idx])
            kick_travel_dist = float(np.linalg.norm(self.ball_pos - self.ball_kick_origin))
            is_wall_pass = bool(self.is_wall_pass_in_flight and self.ball_wall_bounces == 1)

            if new_team == kicker_team and new_holder != kicker_idx:
                is_teammate_pass = True
                min_dist = mech.min_wall_pass_distance if is_wall_pass else mech.min_pass_distance
                if kick_travel_dist >= min_dist:
                    # Completed pass
                    self.last_passer_idx = kicker_idx
                    kicker_role = int(self.player_role[kicker_idx])
                    receiver_role = int(self.player_role[new_holder])

                    # 1. Compute canonical pass coordinates & goal-distance progression:
                    c_orig, _, _ = self._to_team_canonical(
                        self.ball_kick_origin.reshape(1, 2),
                        np.zeros((1, 2)),
                        0.0,
                        new_team,
                    )
                    c_dest, _, _ = self._to_team_canonical(
                        self.ball_pos.reshape(1, 2),
                        np.zeros((1, 2)),
                        0.0,
                        new_team,
                    )

                    canon_dx = float(c_dest[0, 0] - c_orig[0, 0])
                    enemy_goal_canon = np.array([1400.0, 420.0], dtype=np.float64)
                    d_orig_goal = float(np.linalg.norm(c_orig[0] - enemy_goal_canon))
                    d_dest_goal = float(np.linalg.norm(c_dest[0] - enemy_goal_canon))
                    goal_dist_gain = d_orig_goal - d_dest_goal

                    # Effective territorial gain: max of forward X gain and Euclidean goal-distance reduction
                    effective_dx = max(canon_dx, goal_dist_gain)
                    is_progressive = effective_dx >= rew.min_progressive_dx_threshold

                    # Check Final-Third "Better Angle Around the Keeper" (Cutback / Semicircle Switch):
                    # e.g. Striker below semicircle (y > 480) passing to Midfielder above semicircle (y < 360)
                    in_attacking_third = (
                        float(c_orig[0, 0]) >= 960.0
                        and float(c_dest[0, 0]) >= 960.0
                        and d_dest_goal <= 380.0
                    )
                    ang_orig = math.atan2(float(c_orig[0, 1]) - 420.0, 1400.0 - float(c_orig[0, 0]))
                    ang_dest = math.atan2(float(c_dest[0, 1]) - 420.0, 1400.0 - float(c_dest[0, 0]))
                    goal_angle_shift = abs(float(wrap_to_pi(ang_dest - ang_orig)))
                    is_box_angle_switch = in_attacking_third and (goal_angle_shift >= math.radians(25.0))

                    # Always ensure last_kick_pos for kicker_idx matches ball_kick_origin
                    self.last_kick_pos[kicker_idx] = self.ball_kick_origin.copy()

                    # 2. Pairwise repeat decay and Give-and-Go evaluation:
                    current_pair = (min(kicker_idx, new_holder), max(kicker_idx, new_holder))
                    is_return_pass = (self.last_pass_pair == current_pair)
                    if is_return_pass:
                        self.pairwise_repeat_count += 1
                    else:
                        self.last_pass_pair = current_pair
                        self.pairwise_repeat_count = 0

                    # Check Give-and-Go (One-Two):
                    # Player A previously passed to Player B, Player B returns to Player A (pairwise_repeat_count == 1)
                    # and Player A advanced downfield >= 35.0u since their initial kick.
                    is_give_and_go = False
                    has_prev_kick = float(np.linalg.norm(self.last_kick_pos[new_holder])) > 1e-6
                    if is_return_pass and self.pairwise_repeat_count == 1 and has_prev_kick:
                        c_prev_kick, _, _ = self._to_team_canonical(
                            self.last_kick_pos[new_holder:new_holder + 1],
                            np.zeros((1, 2)),
                            0.0,
                            new_team,
                        )
                        give_and_go_gain = float(c_dest[0, 0] - c_prev_kick[0, 0])
                        if give_and_go_gain >= 35.0:
                            is_give_and_go = True

                    if is_give_and_go:
                        pair_decay_multiplier = 1.0
                    else:
                        pair_decay_multiplier = rew.pass_pairwise_decay_base ** self.pairwise_repeat_count

                    # 3. Line-Breaking Defender Count:
                    opp_team = 1 - new_team
                    opp_indices = [
                        idx for idx in range(self.num_players) if self.player_team[idx] == opp_team
                    ]
                    c_opp_pos, _, _ = self._to_team_canonical(
                        self.player_pos[opp_indices],
                        np.zeros((len(opp_indices), 2)),
                        0.0,
                        new_team,
                    )
                    opp_xs = c_opp_pos[:, 0]
                    bypassed_defenders = int(
                        np.sum((opp_xs >= c_orig[0, 0]) & (opp_xs < c_dest[0, 0]))
                    )

                    # 4. Check Backwards / Recycling Tactical Value (Escape Press & Attacking Lay-Off):
                    is_escape_press = False
                    is_attacking_layoff = False

                    # Opponent proximity to kicker at pass origin vs receiver at destination
                    dists_to_passer = [
                        float(np.linalg.norm(self.player_pos[oi] - self.ball_kick_origin))
                        for oi in opp_indices
                    ]
                    nearest_opp_passer = min(dists_to_passer) if dists_to_passer else 999.0

                    dists_to_receiver = [
                        float(np.linalg.norm(self.player_pos[oi] - self.ball_pos))
                        for oi in opp_indices
                    ]
                    nearest_opp_receiver = min(dists_to_receiver) if dists_to_receiver else 999.0

                    # A) Escape Press: passer under close pressure (<= 140u) or held >= 1.8s, and receiver has more space
                    if (nearest_opp_passer <= 140.0 or self.last_kick_hold_time[kicker_idx] >= 1.8) and (
                        nearest_opp_receiver >= nearest_opp_passer + 35.0 and nearest_opp_receiver >= 110.0
                    ):
                        is_escape_press = True

                    # B) Attacking Lay-Off & Semicircle/Byline Cutback:
                    is_midfield_layoff = (
                        float(c_orig[0, 0]) >= 700.0
                        and 460.0 <= float(c_dest[0, 0]) < float(c_orig[0, 0])
                        and 180.0 <= float(c_dest[0, 1]) <= 660.0
                        and receiver_role in (ROLE_MID, ROLE_ATK)
                    )
                    is_box_cutback = (
                        (float(c_orig[0, 0]) >= 980.0 or d_orig_goal <= 260.0)
                        and float(c_dest[0, 0]) >= 720.0
                        and d_dest_goal <= 400.0
                        and (
                            float(c_dest[0, 0]) < float(c_orig[0, 0]) - 15.0
                            or abs(float(c_dest[0, 1]) - 420.0) < abs(float(c_orig[0, 1]) - 420.0) - 25.0
                        )
                    )
                    is_attacking_layoff = is_midfield_layoff or is_box_cutback

                    # 5. Compute Progressive, Box-Angle Switch, Give-and-Go, Escape Press & Lay-Off Rewards:
                    prog_bonus = 0.0
                    if is_progressive or is_box_angle_switch:
                        prog_bonus += (max(0.0, effective_dx) * rew.pass_progressive_dx_weight) + (
                            bypassed_defenders * rew.pass_line_break_bonus
                        )
                        if is_box_angle_switch:
                            prog_bonus += rew.box_angle_pass_bonus

                    if is_give_and_go:
                        prog_bonus += rew.give_and_go_bonus

                    if is_escape_press:
                        prog_bonus += rew.escape_press_pass_bonus

                    if is_attacking_layoff:
                        prog_bonus += rew.attacking_layoff_bonus

                    is_valid_wall_pass = False
                    if is_wall_pass:
                        # Wall pass: award wall_pass_bonus if it advances downfield (effective_dx >= 30u),
                        # switches angles, lays off, or escapes opponent pressure
                        if is_progressive or is_escape_press or is_box_angle_switch or is_attacking_layoff or effective_dx >= 30.0:
                            is_valid_wall_pass = True
                            prog_bonus += rew.wall_pass_bonus

                    # Check Markovian scanned-pass multiplier (receiver in kicker's FOV at kick or reception)
                    for test_rx_pos in (self.kick_frame_player_pos[new_holder], self.player_pos[new_holder]):
                        to_rx = test_rx_pos - self.ball_kick_origin
                        d_rx = float(np.linalg.norm(to_rx))
                        if 1e-4 < d_rx <= self.cfg.vision.max_range:
                            kick_disp = self.ball_pos - self.ball_kick_origin
                            if float(np.linalg.norm(kick_disp)) > 1e-4:
                                k_ang = math.atan2(float(kick_disp[1]), float(kick_disp[0]))
                            else:
                                k_ang = math.atan2(float(self.ball_vel[1]), float(self.ball_vel[0]))
                            rel_rx_ang = abs(float(wrap_to_pi(math.atan2(to_rx[1], to_rx[0]) - k_ang)))
                            if rel_rx_ang <= self.cfg.vision.half_fov_rad:
                                prog_bonus *= rew.scanned_pass_mult
                                break

                    prog_bonus *= pair_decay_multiplier

                    # Update Tiki-Taka chain:
                    if is_progressive or is_box_angle_switch or is_give_and_go or is_valid_wall_pass or is_attacking_layoff:
                        self.team_pass_chain[new_team] = min(
                            self.team_pass_chain[new_team] + 1, rew.pass_chain_max_count
                        )
                    elif is_escape_press:
                        # Backwards tactical passes preserve possession and sustain a minimum chain of 1
                        self.team_pass_chain[new_team] = max(1, self.team_pass_chain[new_team])

                    passer_bonus = prog_bonus
                    receiver_bonus = (rew.completed_pass_receiver_ind + rew.first_touch_reception_bonus) * pair_decay_multiplier
                    team_bonus = prog_bonus * 0.5

                    # Role tactical bonuses scaled by pair_decay_multiplier:
                    if kicker_role == ROLE_DEF and receiver_role in (ROLE_MID, ROLE_ATK):
                        passer_bonus += rew.def_breakout_pass_bonus * pair_decay_multiplier
                    if receiver_role == ROLE_MID and kicker_role in (ROLE_KEEPER, ROLE_DEF):
                        receiver_bonus += rew.mid_link_receive_bonus * pair_decay_multiplier
                    if kicker_role == ROLE_MID and receiver_role == ROLE_ATK:
                        passer_bonus += rew.mid_to_atk_pass_bonus * pair_decay_multiplier
                    if receiver_role == ROLE_ATK:
                        c_rx_pos, _, _ = self._to_team_canonical(
                            self.player_pos[new_holder:new_holder + 1],
                            self.player_vel[new_holder:new_holder + 1],
                            0.0,
                            new_team,
                        )
                        if float(c_rx_pos[0, 0]) >= 680.0:
                            passer_bonus += rew.pass_to_atk_bonus * pair_decay_multiplier

                    # Check Through-Ball / Lead-Pass Bonus:
                    # Completed pass to teammate running forward in stride (v_x,c >= 90 u/s)
                    # and displaced forward >= 35u since the pass was kicked
                    is_through_ball = False
                    c_rx_vel = float(
                        self.player_vel[new_holder, 0]
                        if self.team_side[new_team] == SIDE_LEFT
                        else -self.player_vel[new_holder, 0]
                    )
                    c_rx_pos_now, _, _ = self._to_team_canonical(
                        self.player_pos[new_holder : new_holder + 1],
                        np.zeros((1, 2)),
                        0.0,
                        new_team,
                    )
                    c_rx_pos_kick, _, _ = self._to_team_canonical(
                        self.kick_frame_player_pos[new_holder : new_holder + 1],
                        np.zeros((1, 2)),
                        0.0,
                        new_team,
                    )
                    rec_disp_canon = float(c_rx_pos_now[0, 0] - c_rx_pos_kick[0, 0])
                    if c_rx_vel >= 90.0 and rec_disp_canon >= 35.0:
                        is_through_ball = True
                        lead_bonus_half = 0.5 * rew.lead_pass_bonus * pair_decay_multiplier
                        passer_bonus += lead_bonus_half
                        receiver_bonus += lead_bonus_half

                    step_rewards[kicker_idx] += passer_bonus
                    step_rewards[new_holder] += receiver_bonus
                    self._add_team_reward(step_rewards, new_team, team_bonus)
                    self.stats["completed_passes"] += 1
                    self.stats["team_passes"][new_team] += 1

                    if is_through_ball:
                        lbl = "THROUGH BALL!"
                    elif is_valid_wall_pass:
                        lbl = "WALL PASS!"
                    elif is_give_and_go:
                        lbl = "ONE-TWO!"
                    elif is_attacking_layoff:
                        lbl = "LAY-OFF!"
                    elif is_escape_press:
                        lbl = "ESCAPE PRESS!"
                    else:
                        lbl = "PASS!"

                    self._emit_event(
                        lbl,
                        self.player_pos[new_holder],
                        team=new_team,
                        agent=self.idx_to_agent[new_holder],
                        reward=passer_bonus + receiver_bonus + team_bonus,
                    )
            elif new_team != kicker_team:
                # Interception or Keeper Save (Turnover)
                self._reset_pass_chains(kicker_team)
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
                    if kicker_role == ROLE_KEEPER:
                        c_rx_pos, _, _ = self._to_team_canonical(
                            self.player_pos[new_holder : new_holder + 1],
                            np.zeros((1, 2)),
                            0.0,
                            kicker_team,
                        )
                        c_rx_x = float(c_rx_pos[0, 0])
                        if c_rx_x <= 200.0:
                            gk_turnover_pen = rew.keeper_deep_turnover_pen
                        elif c_rx_x >= 700.0:
                            gk_turnover_pen = rew.intercepted_kicker_ind
                        else:
                            ratio = (c_rx_x - 200.0) / 500.0
                            gk_turnover_pen = (
                                rew.keeper_deep_turnover_pen
                                + (rew.intercepted_kicker_ind - rew.keeper_deep_turnover_pen) * ratio
                            )
                        step_rewards[kicker_idx] += gk_turnover_pen
                    else:
                        step_rewards[kicker_idx] += rew.intercepted_kicker_ind
                    self._add_team_reward(step_rewards, kicker_team, rew.intercepted_team)
                    self.stats["interceptions"] += 1

                if new_role == ROLE_KEEPER:
                    if is_real_save:
                        self.post_save_box_grace_timer = mech.post_save_box_grace_time
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
            if self.last_ball_team is not None and self.last_ball_team != new_team:
                self._reset_pass_chains(self.last_ball_team)
            step_rewards[new_holder] += rew.loose_ball_pickup_ind
            self._add_team_reward(step_rewards, new_team, rew.loose_ball_pickup_team)
            if new_role == ROLE_KEEPER:
                step_rewards[new_holder] += rew.keeper_box_recovery_ind
                # Clean Footwork Pickup Bonus: extra reward when loose ball is retrieved on foot without lunging
                if self.lunge_timer[new_holder] <= 0.0:
                    step_rewards[new_holder] += rew.keeper_clean_foot_recovery_bonus

            # Contested loose-ball recovery bonus: if an opponent is within contested_ball_distance
            opp_team = 1 - new_team
            opp_indices = [i for i in range(self.num_players) if self.player_team[i] == opp_team]
            if opp_indices:
                opp_dists = np.linalg.norm(
                    self.player_pos[opp_indices] - self.ball_pos[None, :],
                    axis=1,
                )
                if float(np.min(opp_dists)) <= mech.contested_ball_distance:
                    step_rewards[new_holder] += rew.contested_ball_recovery_bonus
                    # Contested ball micro-separation: apply speed damper (0.35) and 15u soft push away from ball
                    phys = self.cfg.physics
                    for oi in opp_indices:
                        to_oi = self.player_pos[oi] - self.ball_pos
                        d_oi = float(np.linalg.norm(to_oi))
                        if d_oi <= 45.0:
                            self.player_vel[oi] *= 0.35
                            if d_oi > 1e-4:
                                self.player_pos[oi] += (to_oi / d_oi) * 15.0
                                np.clip(self.player_pos[oi, 0], phys.player_radius, phys.field_width - phys.player_radius, out=self.player_pos[oi:oi+1, 0])
                                np.clip(self.player_pos[oi, 1], phys.player_radius, phys.field_height - phys.player_radius, out=self.player_pos[oi:oi+1, 1])
                    self._emit_event(
                        "BALL WON!",
                        self.player_pos[new_holder],
                        team=new_team,
                        agent=self.idx_to_agent[new_holder],
                        reward=rew.loose_ball_pickup_ind + rew.contested_ball_recovery_bonus,
                    )

        # Assign possession to new_holder
        self.ball_holder = new_holder
        self.ball_hold_timer = 0.0
        self.box_loose_ball_timer = 0.0
        self.kick_grace_timer = mech.pickup_kick_grace
        if not is_teammate_pass:
            self.pickup_tackle_grace_timer = mech.pickup_tackle_grace
        self.last_ball_holder = new_holder
        self.last_ball_team = new_team
        self.ball_kicked_in_flight = False
        self.ball_wall_bounces = 0
        self.is_wall_pass_in_flight = False

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
        self.stats["field_3s_violations"] += 1
        self._emit_event(
            "5s VIOLATION!",
            self.player_pos[violator_idx],
            team=team,
            agent=self.idx_to_agent[violator_idx],
            reward=rew.field_5s_violation_ind + rew.field_5s_violation_team,
        )

    # Backwards-compatible alias
    _handle_field_3s_violation = _handle_field_5s_violation

    def _handle_keeper_5s_violation(
        self, keeper_idx: int, step_rewards: np.ndarray, is_hold_violation: bool = True
    ) -> None:
        """Enforce Keeper 5-Second Hold Violation or Unclaimed Box Ball Timeout: reward penalty + Foul Teleport Reset."""
        rew = self.cfg.rewards
        penalized_team = int(self.player_team[keeper_idx])

        step_rewards[keeper_idx] += rew.keeper_5s_violation_ind
        self._add_team_reward(step_rewards, penalized_team, rew.keeper_5s_violation_team)
        self.stats["keeper_5s_violations"] += 1
        ev_name = "5s KEEPER HOLD FOUL!" if is_hold_violation else "5s UNCLAIMED BOX BALL!"
        self._emit_event(
            ev_name,
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
            if self.is_ghost[i]:
                continue
            for j in range(i + 1, self.num_players):
                if self.is_ghost[j]:
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

    def _clamp_players_to_boundaries(self, apply_dynamic_push: bool = True) -> None:
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
                dist = math.hypot(delta[0], delta[1])
                if dist > box_r:
                    self.player_pos[idx] = own_goal + (delta / dist) * box_r
            else:
                for side_i, g_center in enumerate(goal_centers):
                    eff_box_r = float(self.active_box_radius[side_i])
                    if eff_box_r <= 1e-3:
                        continue
                    delta = self.player_pos[idx] - g_center
                    dist = math.hypot(delta[0], delta[1])
                    if dist <= eff_box_r:
                        if dist < 1e-6:
                            push_dir = np.array(
                                [1.0 if g_center[0] == 0.0 else -1.0, 0.0], dtype=np.float64
                            )
                        else:
                            push_dir = delta / dist
                        new_pos = g_center + push_dir * (eff_box_r + 1e-3)
                        if apply_dynamic_push:
                            self.step_external_push[idx] += new_pos - self.player_pos[idx]
                            rad_v = float(np.dot(self.player_vel[idx], push_dir))
                            if rad_v < 0.0:
                                self.player_vel[idx] -= rad_v * push_dir
                        self.player_pos[idx] = new_pos

        # Vectorized outer boundary clamping for all players in a single operation
        np.clip(self.player_pos[:, 0], r_player, phys.field_width - r_player, out=self.player_pos[:, 0])
        np.clip(self.player_pos[:, 1], r_player, phys.field_height - r_player, out=self.player_pos[:, 1])

        # Kickoff Exclusion Circle: Smoothly or firmly keep opposing players outside center spot (r = 130.0)
        if self.kickoff_protection_timer > 0.0 and self.ball_holder is not None:
            start_team = int(self.player_team[self.ball_holder])
            field_center = np.array(phys.field_center, dtype=np.float64)
            excl_r = self.cfg.mechanics.kickoff_exclusion_radius
            for idx in range(self.num_players):
                if self.player_team[idx] != start_team:
                    delta = self.player_pos[idx] - field_center
                    dist = float(np.linalg.norm(delta))
                    if dist < excl_r:
                        push_dir = (
                            np.array([1.0, 0.0], dtype=np.float64)
                            if dist < 1e-6
                            else delta / dist
                        )
                        new_pos = field_center + push_dir * excl_r
                        self.step_external_push[idx] += new_pos - self.player_pos[idx]
                        self.player_pos[idx] = new_pos

        # Smooth GK Possession Repulsion Wave (gk_hold_opponent_buffer = 230.0):
        # Genuine physical pushout at gk_hold_push_speed (285.0 u/s) without 1-frame teleporting!
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
            for idx in range(self.num_players):
                if self.player_team[idx] != gk_team:
                    delta = self.player_pos[idx] - gk_goal_center
                    dist = float(np.linalg.norm(delta))
                    if dist < buf_r:
                        if dist < 1e-6:
                            push_dir = np.array(
                                [1.0 if gk_side == SIDE_LEFT else -1.0, 0.0], dtype=np.float64
                            )
                        else:
                            push_dir = delta / dist
                        push_mag = min(buf_r - dist, max_step_push)
                        disp = push_dir * push_mag
                        self.player_pos[idx] += disp
                        self.step_external_push[idx] += disp
                        # Strip any inward radial velocity so the player smoothly slides outward
                        rad_v = float(np.dot(self.player_vel[idx], push_dir))
                        if rad_v < 0.0:
                            self.player_vel[idx] -= rad_v * push_dir

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
        chain_len = self.team_pass_chain[scoring_team]
        chain_mult = 1.0 + (rew.pass_chain_multiplier_coeff * chain_len)

        if self.last_ball_holder is not None:
            if int(self.player_team[self.last_ball_holder]) == scoring_team:
                if int(self.player_role[self.last_ball_holder]) == ROLE_ATK:
                    scorer_ind = rew.atk_goal_scorer_ind * chain_mult
                else:
                    scorer_ind = rew.goal_scorer_ind * chain_mult
                step_rewards[self.last_ball_holder] += scorer_ind
            else:
                scorer_ind = rew.own_goal_ind
                step_rewards[self.last_ball_holder] += scorer_ind
                self.stats["own_goals"] = self.stats.get("own_goals", 0) + 1
                is_own_goal = True

        if is_own_goal:
            scoring_team_reward = rew.goal_team_shared
        else:
            scoring_team_reward = rew.goal_team_shared * chain_mult

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
        self.popup_cooldown[:] = 0.0
        self.ball_vel[:] = 0.0
        self.ball_hold_timer = 0.0
        self.kick_grace_timer = 0.0
        self.box_loose_ball_timer = 0.0
        self.post_save_box_grace_timer = 0.0
        self.active_box_radius[:] = self.cfg.physics.keeper_box_radius
        self.kick_frame_player_pos.fill(0.0)
        self.ball_kicked_in_flight = False
        self.ball_kick_was_shot_on_target = False
        self.ball_kick_flight_timer = 0.0
        self.ball_wall_bounces = 0
        self.is_wall_pass_in_flight = False
        self.pinned_ball_timer = 0.0
        self.kickoff_protection_timer = 0.0
        self.pickup_tackle_grace_timer = 0.0
        self.prev_longitudinal_vel[:] = 0.0
        self.step_external_push.fill(0.0)
        self.prev_ball_at_step_start = self.ball_pos.copy()
        self.smoothed_steer.fill(0.0)
        self.prev_steer.fill(0.0)
        self.dive_dir.fill(0.0)
        self.prev_dynamic_role_key = {i: self.player_role_key[i] for i in range(self.num_players)}
        self.last_kick_pos.fill(0.0)
        self.last_kick_hold_time.fill(0.0)
        self.primary_loose_ball_chaser.fill(-1)
        self.dive_speed.fill(0.0)
        self._reset_pass_chains()

    def _reset_pass_chains(self, team: Optional[int] = None) -> None:
        """Reset pass chain and pairwise history on turnover or dead ball."""
        if team is None:
            self.team_pass_chain = [0, 0]
        else:
            self.team_pass_chain[team] = 0
        self.last_pass_pair = None
        self.last_passer_idx = None
        self.pairwise_repeat_count = 0

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
        """Position both teams for the Foul Restart (Open 3-2 Attacking Overload & Zonal Block)."""
        self._clear_all_dynamic_timers()
        attacking_team = 1 - penalized_team
        attacking_side = int(self.team_side[attacking_team])

        att_c_pos = {
            "atk_1": (820.0, 420.0),
            "mid_1": (965.0, 235.0),
            "mid_2": (965.0, 605.0),
            "def_1": (540.0, 260.0),
            "def_2": (540.0, 580.0),
            "keeper": (40.0, 420.0),
        }
        att_c_ang = {
            "atk_1": 0.0,
            "mid_1": 0.32,
            "mid_2": -0.32,
            "def_1": 0.0,
            "def_2": 0.0,
            "keeper": 0.0,
        }
        def_c_pos = {
            "atk_1": (640.0, 420.0),
            "def_1": (1115.0, 315.0),
            "def_2": (1115.0, 525.0),
            "mid_1": (1030.0, 130.0),
            "mid_2": (1030.0, 710.0),
            "keeper": (1360.0, 420.0),
        }
        def_c_ang = {
            "atk_1": 0.0,
            "def_1": math.pi,
            "def_2": math.pi,
            "mid_1": math.pi,
            "mid_2": math.pi,
            "keeper": math.pi,
        }

        for idx in range(self.num_players):
            team = int(self.player_team[idx])
            role_key = self.player_role_key[idx]
            if team == attacking_team:
                canon_pos = att_c_pos[role_key]
                canon_angle = att_c_ang[role_key]
            else:
                canon_pos = def_c_pos[role_key]
                canon_angle = def_c_ang[role_key]

            world_pos, world_angle = self._CanonicalToWorld(
                canon_pos, canon_angle, attacking_side
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

        obs_dict: Dict[str, np.ndarray] = {}
        for idx, agent_name in enumerate(self.possible_agents):
            if idx < 6:
                obs_dict[agent_name] = self._build_single_observation(
                    idx, c_p_a, c_v_a, c_ang_a, c_b_a, c_bv_a
                )
            else:
                obs_dict[agent_name] = self._build_single_observation(
                    idx, c_p_b, c_v_b, c_ang_b, c_b_b, c_bv_b
                )
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

    def _build_single_observation(
        self,
        idx: int,
        c_player_pos: Optional[np.ndarray] = None,
        c_player_vel: Optional[np.ndarray] = None,
        c_player_angle: Optional[np.ndarray | float] = None,
        c_ball_pos: Optional[np.ndarray] = None,
        c_ball_vel: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build the 76-float canonical observation for player `idx`."""
        phys = self.cfg.physics
        mech = self.cfg.mechanics
        vis = self.cfg.vision

        my_team = int(self.player_team[idx])
        my_role = int(self.player_role[idx])

        # Transform all players and ball into `my_team`'s canonical coordinate system if not provided
        if c_player_pos is None or c_player_vel is None or c_player_angle is None:
            c_player_pos, c_player_vel, c_player_angle = self._to_team_canonical(
                self.player_pos, self.player_vel, self.player_angle, my_team
            )
        if c_ball_pos is None or c_ball_vel is None:
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
        obs[0] = _fast_clip((sx / phys.field_width) * 2.0 - 1.0, -1.0, 1.0)
        obs[1] = _fast_clip((sy / phys.field_height) * 2.0 - 1.0, -1.0, 1.0)

        # sin and cos of canonical facing angle
        obs[2] = math.sin(s_angle)
        obs[3] = math.cos(s_angle)

        # Normalized self velocity
        max_self_v = mech.lunge_speed if my_role == ROLE_KEEPER else self._get_player_max_speed(idx)
        obs[4] = _fast_clip(svx / max_self_v, -1.0, 1.0)
        obs[5] = _fast_clip(svy / max_self_v, -1.0, 1.0)

        # Score Differential
        my_score = self.stats.get("score", [0, 0])[my_team]
        opp_score = self.stats.get("score", [0, 0])[1 - my_team]
        obs[6] = _fast_clip((my_score - opp_score) / 3.0, -1.0, 1.0)
        has_ball = 1.0 if self.ball_holder == idx else 0.0
        obs[7] = has_ball

        # Possession / hold / tackle status
        max_hold = (
            mech.keeper_max_hold_time if my_role == ROLE_KEEPER else mech.field_max_hold_time
        )
        if has_ball > 0.5:
            obs[8] = _fast_clip(self.ball_hold_timer / max_hold, 0.0, 1.0)
        elif self.ball_holder is None:
            obs[8] = 0.0
        elif int(self.player_team[self.ball_holder]) == my_team:
            obs[8] = 0.5
        else:
            holder_idx = self.ball_holder
            d_holder = math.hypot(
                self.player_pos[idx, 0] - self.player_pos[holder_idx, 0],
                self.player_pos[idx, 1] - self.player_pos[holder_idx, 1],
            )
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
            obs[9] = _fast_clip(self.lunge_cooldown[idx] / mech.lunge_cooldown, 0.0, 1.0)
        else:
            max_tackle_cd = max(mech.tackle_fail_cooldown, mech.field_violation_tackle_cooldown)
            obs[9] = _fast_clip(self.tackle_cooldown[idx] / max_tackle_cd, 0.0, 1.0)

        # Ghost or Stun timer normalized to [0.0, 1.0]
        active_disable_timer = max(float(self.ghost_timer[idx]), float(self.stun_timer[idx]))
        obs[10] = _fast_clip(active_disable_timer / mech.ghost_duration, 0.0, 1.0)

        # Relative angle and distance to Enemy Goal (always at (1400.0, 420.0) in canonical space)
        enemy_goal_dx = phys.field_width - sx
        enemy_goal_dy = (phys.field_height * 0.5) - sy
        goal_dist = math.hypot(enemy_goal_dx, enemy_goal_dy)
        angle_to_goal = math.atan2(enemy_goal_dy, enemy_goal_dx)
        rel_goal_angle = float(wrap_to_pi(angle_to_goal - s_angle))

        obs[11] = _fast_clip(rel_goal_angle / math.pi, -1.0, 1.0)
        obs[12] = _fast_clip(goal_dist / self._max_field_dist, 0.0, 1.0)

        # Role Encoding, Tactical Pocket Vector & Pass Chain (indices 13..15)
        if my_role == ROLE_KEEPER:
            own_goal_center_y = phys.field_height * 0.5
            ball_dist_to_own_goal = math.hypot(
                float(c_ball_pos[0]), float(c_ball_pos[1]) - own_goal_center_y
            )
            is_ball_loose_in_own_box = (
                self.ball_holder is None
                and ball_dist_to_own_goal <= (phys.keeper_box_radius + phys.ball_radius)
            )
            if is_ball_loose_in_own_box:
                obs[13] = _fast_clip(
                    0.25 + 0.75 * (self.box_loose_ball_timer / mech.keeper_box_loose_max_time),
                    0.25,
                    1.0,
                )
            else:
                obs[13] = 0.0
            own_side = int(self.team_side[my_team])
            obs[14] = float(np.clip(self.active_box_radius[own_side] / phys.keeper_box_radius, 0.0, 1.0))
            obs[15] = float(np.clip(self.team_pass_chain[my_team] / 10.0, 0.0, 1.0))
        else:
            role_base = -0.75 if my_role == ROLE_DEF else (0.0 if my_role == ROLE_MID else 0.75)
            obs[13] = float(role_base + 0.20 * np.clip(self.team_pass_chain[my_team] / 10.0, 0.0, 1.0))
            obs[14] = float(np.clip((self.active_tactical_target_canon[idx, 0] - sx) / 450.0, -1.0, 1.0))
            obs[15] = float(np.clip((self.active_tactical_target_canon[idx, 1] - sy) / 450.0, -1.0, 1.0))

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
                obs[slot_start + 1] = _fast_clip(dist / self._max_field_dist, 0.0, 1.0)
                obs[slot_start + 2] = _fast_clip(rel_ang / math.pi, -1.0, 1.0)
                obs[slot_start + 3] = _fast_clip(
                    (float(evel[0]) - svx) / max_rel_vel, -1.0, 1.0
                )
                obs[slot_start + 4] = _fast_clip(
                    (float(evel[1]) - svy) / max_rel_vel, -1.0, 1.0
                )
            else:
                # Slots 1..11: Foveal vision + Peripheral radar (persistent 360 presence floor)
                fade_rad = math.radians(15.0)
                rad_90 = math.radians(90.0)
                rad_105 = vis.half_fov_rad
                if dist <= vis.max_range and abs(rel_ang) <= rad_105:
                    w_phi = 1.0 if abs(rel_ang) <= rad_90 else ((rad_105 - abs(rel_ang)) / fade_rad)
                    w_d = 1.0 - 0.35 * (dist / vis.max_range)
                    vis_presence = float(np.clip(vis.peripheral_radar_presence + (1.0 - vis.peripheral_radar_presence) * w_phi * w_d, vis.peripheral_radar_presence, 1.0))
                else:
                    vis_presence = vis.peripheral_radar_presence

                obs[slot_start + 0] = vis_presence
                obs[slot_start + 1] = float(np.clip(dist / self._max_field_dist, 0.0, 1.0))
                obs[slot_start + 2] = float(np.clip(rel_ang / math.pi, -1.0, 1.0))
                obs[slot_start + 3] = float(np.clip((float(evel[0]) - svx) / max_rel_vel, -1.0, 1.0))
                obs[slot_start + 4] = float(np.clip((float(evel[1]) - svy) / max_rel_vel, -1.0, 1.0))

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
        if any_teleport:
            self.pocket_initialized[:] = False
        rew = self.cfg.rewards
        phys = self.cfg.physics
        mech = self.cfg.mechanics
        curr_player_ball_dists = np.linalg.norm(
            self.player_pos - self.ball_pos[None, :], axis=1
        )
        holder_team = (
            int(self.player_team[self.ball_holder])
            if self.ball_holder is not None
            else -1
        )

        adj_prev_pos = prev_player_pos + self.step_external_push
        adj_prev_player_ball_dists = np.linalg.norm(
            adj_prev_pos - self.prev_ball_at_step_start[None, :], axis=1
        )

        for team_id in (TEAM_A, TEAM_B):
            c_player_pos, _, c_player_angle = self._to_team_canonical(
                self.player_pos, self.player_vel, self.player_angle, team_id
            )
            c_prev_pos, _, _ = self._to_team_canonical(
                adj_prev_pos, self.player_vel, self.player_angle, team_id
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

            # Dynamically assign Top ("_1") vs Bottom ("_2") tactical lane slots based on current canonical Y!
            # With a 25.0u hysteresis deadband using prev_dynamic_role_key to prevent 15 Hz role flapping.
            def_indices = [i for i in team_fields if self.player_role[i] == ROLE_DEF]
            mid_indices = [i for i in team_fields if self.player_role[i] == ROLE_MID]
            atk_indices = [i for i in team_fields if self.player_role[i] == ROLE_ATK]

            dynamic_role_key: Dict[int, str] = {}
            if len(def_indices) == 2:
                d0, d1 = def_indices[0], def_indices[1]
                prev_0 = self.prev_dynamic_role_key.get(d0)
                prev_1 = self.prev_dynamic_role_key.get(d1)
                y0 = float(c_player_pos[d0, 1])
                y1 = float(c_player_pos[d1, 1])
                if prev_0 == "def_1" and prev_1 == "def_2":
                    if y1 < y0 - 25.0:
                        dynamic_role_key[d1] = "def_1"
                        dynamic_role_key[d0] = "def_2"
                    else:
                        dynamic_role_key[d0] = "def_1"
                        dynamic_role_key[d1] = "def_2"
                elif prev_0 == "def_2" and prev_1 == "def_1":
                    if y0 < y1 - 25.0:
                        dynamic_role_key[d0] = "def_1"
                        dynamic_role_key[d1] = "def_2"
                    else:
                        dynamic_role_key[d1] = "def_1"
                        dynamic_role_key[d0] = "def_2"
                else:
                    if y0 <= y1:
                        dynamic_role_key[d0] = "def_1"
                        dynamic_role_key[d1] = "def_2"
                    else:
                        dynamic_role_key[d1] = "def_1"
                        dynamic_role_key[d0] = "def_2"
                self.prev_dynamic_role_key[d0] = dynamic_role_key[d0]
                self.prev_dynamic_role_key[d1] = dynamic_role_key[d1]
            else:
                for i in def_indices:
                    dynamic_role_key[i] = self.player_role_key[i]

            if len(mid_indices) == 2:
                m0, m1 = mid_indices[0], mid_indices[1]
                prev_0 = self.prev_dynamic_role_key.get(m0)
                prev_1 = self.prev_dynamic_role_key.get(m1)
                y0 = float(c_player_pos[m0, 1])
                y1 = float(c_player_pos[m1, 1])
                if prev_0 == "mid_1" and prev_1 == "mid_2":
                    if y1 < y0 - 25.0:
                        dynamic_role_key[m1] = "mid_1"
                        dynamic_role_key[m0] = "mid_2"
                    else:
                        dynamic_role_key[m0] = "mid_1"
                        dynamic_role_key[m1] = "mid_2"
                elif prev_0 == "mid_2" and prev_1 == "mid_1":
                    if y0 < y1 - 25.0:
                        dynamic_role_key[m0] = "mid_1"
                        dynamic_role_key[m1] = "mid_2"
                    else:
                        dynamic_role_key[m1] = "mid_1"
                        dynamic_role_key[m0] = "mid_2"
                else:
                    if y0 <= y1:
                        dynamic_role_key[m0] = "mid_1"
                        dynamic_role_key[m1] = "mid_2"
                    else:
                        dynamic_role_key[m1] = "mid_1"
                        dynamic_role_key[m0] = "mid_2"
                self.prev_dynamic_role_key[m0] = dynamic_role_key[m0]
                self.prev_dynamic_role_key[m1] = dynamic_role_key[m1]
            else:
                for i in mid_indices:
                    dynamic_role_key[i] = self.player_role_key[i]

            for i in atk_indices:
                dynamic_role_key[i] = "atk_1"

            c_bx = float(c_ball_pos[0])
            def _effective_chase_dist(p_idx: int) -> float:
                d_raw = float(curr_player_ball_dists[p_idx])
                r_p = self.player_role[p_idx]
                if r_p == ROLE_DEF and c_bx > 660.0:
                    # In the opponent half, MID and ATK take priority for high pressing / loose balls
                    # unless the defender is dramatically closer than any MID/ATK
                    d_raw += 160.0 + 0.35 * (c_bx - 660.0)
                elif r_p == ROLE_ATK and c_bx < 450.0:
                    # In deep own third, DEF and MID take priority for defensive recovery
                    d_raw += 130.0
                return d_raw

            # Determine tactical right-of-way exemptions for this team:
            # - Ball carrier is always exempt
            # - Primary ball chaser/presser (closest field player to ball in State B2/C) is exempt
            # - Intended pass receiver (best_rx) and passer (last_ball_holder) during pass-in-flight are exempt
            # - On loose balls, top-2 chasers and teammates contesting near loose ball (<=165u) are exempt from crowding
            if self.ball_holder is None:
                cand_0 = min(team_fields, key=_effective_chase_dist)
                curr_prim = self.primary_loose_ball_chaser[team_id]
                if curr_prim not in team_fields:
                    self.primary_loose_ball_chaser[team_id] = cand_0
                else:
                    d_curr = _effective_chase_dist(curr_prim)
                    d_cand = _effective_chase_dist(cand_0)
                    # Candidate must be at least 18 units closer to snatch primary status
                    if d_cand < d_curr - 18.0:
                        self.primary_loose_ball_chaser[team_id] = cand_0
                closest_field_to_ball = self.primary_loose_ball_chaser[team_id]
            else:
                self.primary_loose_ball_chaser[team_id] = -1
                closest_field_to_ball = min(team_fields, key=_effective_chase_dist)

            ranked_loose_fields = sorted(
                team_fields,
                key=lambda i: (0 if i == closest_field_to_ball else 1, _effective_chase_dist(i)),
            )
            top_2_loose_chasers = set(ranked_loose_fields[:2])

            pass_in_flight_for_team = (
                self.ball_holder is None
                and self.ball_kicked_in_flight
                and self.last_ball_team == team_id
            )
            best_rx_exempt = -1
            if pass_in_flight_for_team:
                rx_cands = [i for i in team_fields if i != self.last_ball_holder]
                if rx_cands:
                    ray_end = self.ball_pos + self.ball_vel * 1.5
                    best_rx_exempt = min(
                        rx_cands,
                        key=lambda i: self._point_to_segment_dist(
                            self.player_pos[i], self.ball_pos, ray_end
                        ),
                    )

            def _is_exempt_from_crowding(p_idx: int, other_idx: int) -> bool:
                if p_idx == self.ball_holder:
                    return True
                if self.ball_holder is None:
                    if p_idx in top_2_loose_chasers:
                        return True
                    if float(curr_player_ball_dists[p_idx]) <= 165.0 and float(curr_player_ball_dists[other_idx]) <= 165.0:
                        return True
                    if p_idx == best_rx_exempt and best_rx_exempt != -1:
                        return True
                    if pass_in_flight_for_team and p_idx == self.last_ball_holder and other_idx == best_rx_exempt:
                        return True
                if pass_in_flight_for_team:
                    if p_idx == best_rx_exempt:
                        return True
                    if p_idx == self.last_ball_holder and other_idx == best_rx_exempt:
                        return True
                if holder_team != team_id and p_idx == closest_field_to_ball:
                    return True
                return False

            enemy_goal_c_canon = np.array([1400.0, 420.0], dtype=np.float64)

            # 1. Context-Aware Asymmetric Anti-Clumping & Approach Repulsion across all 5 field players
            for idx_a in range(len(team_fields)):
                for idx_b in range(idx_a + 1, len(team_fields)):
                    p_i = team_fields[idx_a]
                    p_j = team_fields[idx_b]
                    d_ij = float(np.linalg.norm(self.player_pos[p_i] - self.player_pos[p_j]))
                    d_prev_ij = float(np.linalg.norm(adj_prev_pos[p_i] - adj_prev_pos[p_j]))

                    # Check if both players are in the attacking box zone at different angles around the keeper
                    di_goal = float(np.linalg.norm(c_player_pos[p_i] - enemy_goal_c_canon))
                    dj_goal = float(np.linalg.norm(c_player_pos[p_j] - enemy_goal_c_canon))
                    ang_i = math.atan2(
                        float(c_player_pos[p_i, 1]) - 420.0, 1400.0 - float(c_player_pos[p_i, 0])
                    )
                    ang_j = math.atan2(
                        float(c_player_pos[p_j, 1]) - 420.0, 1400.0 - float(c_player_pos[p_j, 0])
                    )
                    ang_diff = abs(float(wrap_to_pi(ang_i - ang_j)))

                    both_near_enemy_box = (di_goal <= 280.0) and (dj_goal <= 280.0)
                    is_defender_pair = (p_i in def_indices and p_j in def_indices)
                    in_defensive_half = (float(c_player_pos[p_i, 0]) <= 700.0 and float(c_player_pos[p_j, 0]) <= 700.0)
                    if is_defender_pair and in_defensive_half:
                        # Center back pair shielding own goal mouth: relax spacing to allow compact central zoning
                        min_space = rew.teammate_defender_spacing
                    elif both_near_enemy_box and ang_diff >= math.radians(40.0):
                        # Wide angular separation around keeper semicircle (e.g. ST below, MID above):
                        # Relax spacing threshold so cutback options are never punished!
                        min_space = rew.teammate_box_cutback_spacing
                    else:
                        min_space = rew.teammate_min_spacing

                    if d_ij < min_space:
                        overlap = 1.0 - (d_ij / max(1e-6, min_space))
                        pen = rew.crowding_penalty_scale * (overlap + 0.5 * overlap * overlap)
                        if not _is_exempt_from_crowding(p_i, p_j):
                            step_rewards[p_i] -= pen
                        if not _is_exempt_from_crowding(p_j, p_i):
                            step_rewards[p_j] -= pen

                    # Directional Teammate Approach Repulsion (prevents +0.010 pocket/ball chase from overpowering spacing)
                    if not any_teleport and d_ij < min(155.0, min_space) and d_ij < d_prev_ij:
                        # Check which player is actively moving toward the other
                        vec_ij = self.player_pos[p_j] - self.player_pos[p_i]
                        norm_ij = max(1e-6, float(np.linalg.norm(vec_ij)))
                        unit_ij = vec_ij / norm_ij
                        disp_i = self.player_pos[p_i] - adj_prev_pos[p_i]
                        disp_j = self.player_pos[p_j] - adj_prev_pos[p_j]
                        approach_i = max(0.0, float(np.dot(disp_i, unit_ij)))
                        approach_j = max(0.0, float(np.dot(disp_j, -unit_ij)))
                        if approach_i > 0.0 and not _is_exempt_from_crowding(p_i, p_j):
                            step_rewards[p_i] -= rew.teammate_approach_penalty_scale * approach_i
                        if approach_j > 0.0 and not _is_exempt_from_crowding(p_j, p_i):
                            step_rewards[p_j] -= rew.teammate_approach_penalty_scale * approach_j

                    # Same-Sector Enemy Box Crowding Penalty:
                    # If both players are within 240u of enemy goal AND clogging the SAME angular sector (<35 deg),
                    # penalize ONLY the non-carrier / further-from-ball player (never deterring shooter or wide support!)
                    if both_near_enemy_box and ang_diff < math.radians(35.0):
                        sector_overlap = 1.0 - (ang_diff / math.radians(35.0))
                        sector_pen = rew.box_sector_crowding_pen * sector_overlap
                        if p_i == self.ball_holder:
                            step_rewards[p_j] -= sector_pen
                        elif p_j == self.ball_holder:
                            step_rewards[p_i] -= sector_pen
                        elif float(curr_player_ball_dists[p_i]) <= float(curr_player_ball_dists[p_j]):
                            step_rewards[p_j] -= sector_pen
                        else:
                            step_rewards[p_i] -= sector_pen

            # 2. State A: Our Team Has the Ball
            if holder_team == team_id and self.ball_holder is not None:
                carrier_idx = self.ball_holder
                carrier_c = c_player_pos[carrier_idx]
                cx_car = float(carrier_c[0])
                is_gk_carrier = (carrier_idx == keeper_idx)
                self.active_tactical_target_canon[carrier_idx] = carrier_c.copy()

                for idx in team_fields:
                    if idx == carrier_idx:
                        continue

                    d_curr_car = float(
                        np.linalg.norm(self.player_pos[idx] - self.player_pos[carrier_idx])
                    )
                    d_prev_car = float(
                        np.linalg.norm(adj_prev_pos[idx] - adj_prev_pos[carrier_idx])
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

                    role_key = dynamic_role_key[idx]

                    if is_gk_carrier:
                        # Case 1: GK Holds the Ball -> Show for Keeper with both central channel and wide outlet options
                        if role_key in ("def_1", "mid_1"):
                            x_cand_list = (200.0, 330.0) if role_key == "def_1" else (420.0, 560.0)
                            y_cand_list = (180.0, 265.0, 345.0) if role_key == "def_1" else (95.0, 195.0, 295.0)
                        elif role_key in ("def_2", "mid_2"):
                            x_cand_list = (200.0, 330.0) if role_key == "def_2" else (420.0, 560.0)
                            y_cand_list = (495.0, 575.0, 660.0) if role_key == "def_2" else (545.0, 645.0, 745.0)
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
                                if self.pocket_initialized[idx] and float(np.linalg.norm(cand_pt - self.prev_best_cand_canon[idx])) < 15.0:
                                    sc += rew.pocket_sticky_hysteresis_bonus
                                if sc > best_score:
                                    best_score = sc
                                    best_cand = cand_pt
                        if best_cand is not None:
                            self.prev_best_cand_canon[idx] = best_cand
                            if not self.pocket_initialized[idx]:
                                self.smoothed_pocket_canon[idx] = best_cand.copy()
                                self.pocket_initialized[idx] = True
                            else:
                                alpha = rew.pocket_ema_alpha
                                self.smoothed_pocket_canon[idx] = (1.0 - alpha) * self.smoothed_pocket_canon[idx] + alpha * best_cand
                        target_pocket = self.smoothed_pocket_canon[idx]
                    else:
                        # Case 2: Field Player Holds the Ball
                        if role_key in ("def_1", "def_2"):
                            ideal_x = float(np.clip(min(0.50 * cx_car + 110.0, cx_car - 120.0), 200.0, 660.0))
                            y_shift = float(np.clip(0.15 * (float(carrier_c[1]) - 420.0), -35.0, 35.0))
                            if cx_car < 750.0:
                                ideal_y = (420.0 + y_shift - 175.0) if role_key == "def_1" else (420.0 + y_shift + 175.0)
                            else:
                                ideal_y = (420.0 + y_shift - 95.0) if role_key == "def_1" else (420.0 + y_shift + 95.0)
                            raw_pocket = np.array([ideal_x, ideal_y], dtype=np.float64)
                            self.prev_best_cand_canon[idx] = raw_pocket
                            if not self.pocket_initialized[idx]:
                                self.smoothed_pocket_canon[idx] = raw_pocket.copy()
                                self.pocket_initialized[idx] = True
                            else:
                                alpha = rew.pocket_ema_alpha
                                self.smoothed_pocket_canon[idx] = (1.0 - alpha) * self.smoothed_pocket_canon[idx] + alpha * raw_pocket
                            target_pocket = self.smoothed_pocket_canon[idx]
                        elif role_key in ("mid_1", "mid_2"):
                            dx_list = (120.0, 220.0, -90.0, -180.0)
                            y_cand_list = (110.0, 200.0, 290.0) if role_key == "mid_1" else (550.0, 640.0, 730.0)
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
                                    if self.pocket_initialized[idx] and float(np.linalg.norm(cand_pt - self.prev_best_cand_canon[idx])) < 15.0:
                                        sc += rew.pocket_sticky_hysteresis_bonus
                                    if cx_car >= 1020.0:
                                        if float(carrier_c[1]) > 460.0 and cy <= 310.0:
                                            sc += 40.0
                                        elif float(carrier_c[1]) < 380.0 and cy >= 530.0:
                                            sc += 40.0
                                    if sc > best_score:
                                        best_score = sc
                                        best_cand = cand_pt
                            if best_cand is not None:
                                self.prev_best_cand_canon[idx] = best_cand
                                if not self.pocket_initialized[idx]:
                                    self.smoothed_pocket_canon[idx] = best_cand.copy()
                                    self.pocket_initialized[idx] = True
                                else:
                                    alpha = rew.pocket_ema_alpha
                                    self.smoothed_pocket_canon[idx] = (1.0 - alpha) * self.smoothed_pocket_canon[idx] + alpha * best_cand
                            target_pocket = self.smoothed_pocket_canon[idx]
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
                                if self.pocket_initialized[idx] and float(np.linalg.norm(cand_pt - self.prev_best_cand_canon[idx])) < 15.0:
                                    sc += rew.pocket_sticky_hysteresis_bonus
                                if cx_car >= 1020.0:
                                    if float(carrier_c[1]) > 460.0 and cy in (200.0, 310.0):
                                        sc += 40.0
                                    elif float(carrier_c[1]) < 380.0 and cy in (530.0, 640.0):
                                        sc += 40.0
                                if sc > best_score:
                                    best_score = sc
                                    best_cand = cand_pt
                            if best_cand is not None:
                                self.prev_best_cand_canon[idx] = best_cand
                                if not self.pocket_initialized[idx]:
                                    self.smoothed_pocket_canon[idx] = best_cand.copy()
                                    self.pocket_initialized[idx] = True
                                else:
                                    alpha = rew.pocket_ema_alpha
                                    self.smoothed_pocket_canon[idx] = (1.0 - alpha) * self.smoothed_pocket_canon[idx] + alpha * best_cand
                            target_pocket = self.smoothed_pocket_canon[idx]

                    if target_pocket is not None:
                        self.active_tactical_target_canon[idx] = target_pocket.copy()

                    # Potential-Based Movement Reward Toward target_pocket
                    if not any_teleport and target_pocket is not None:
                        d_prev_p = max(0.0, float(np.linalg.norm(c_prev_pos[idx] - target_pocket)) - rew.pocket_deadzone_radius)
                        d_curr_p = max(0.0, float(np.linalg.norm(c_player_pos[idx] - target_pocket)) - rew.pocket_deadzone_radius)
                        step_rewards[idx] += (
                            support_mult
                            * rew.support_pocket_approach_scale
                            * (d_prev_p - d_curr_p)
                        )

                    # "Wait for the Pass" Open-Pocket Bonus / Half-Turn Readiness & Continuous Lateral Unmasking
                    lane_clear = min(
                        self._point_to_segment_dist(c_player_pos[opp], carrier_c, c_player_pos[idx])
                        for opp in opp_fields
                    )
                    if 120.0 <= d_curr_car <= 520.0 and lane_clear >= 45.0:
                        to_ball_c = c_ball_pos - c_player_pos[idx]
                        d_ball_c = float(np.linalg.norm(to_ball_c))
                        cos_face_ball = (
                            float(np.dot([math.cos(c_player_angle[idx]), math.sin(c_player_angle[idx])], to_ball_c / d_ball_c))
                            if d_ball_c > 1e-6 else 1.0
                        )
                        cos_fwd_c = math.cos(c_player_angle[idx])
                        # Half-Turn / Forward Readiness: ball within 210 deg FOV and angled toward attacking half
                        if cos_face_ball >= -0.20 and cos_fwd_c >= -0.10:
                            face_bonus = 0.5 + 0.5 * max(0.0, cos_fwd_c)
                            lane_factor = float(np.clip(lane_clear / 110.0, 0.3, 1.0))
                            step_rewards[idx] += (
                                support_mult
                                * rew.open_support_bonus_scale
                                * lane_factor
                                * face_bonus
                            )
                    elif not any_teleport and lane_clear < 42.0:
                        # Continuous Lateral Unmasking Reward:
                        # Teammate is shadowed on the passing lane. Find the closest blocker,
                        # compute unit normal n_perp away from the blocker, and reward lateral displacement unmasking the pass line.
                        best_blocker = min(
                            opp_fields,
                            key=lambda opp: self._point_to_segment_dist(c_player_pos[opp], carrier_c, c_player_pos[idx]),
                        )
                        pass_vec = c_player_pos[idx] - carrier_c
                        pass_len = float(np.linalg.norm(pass_vec))
                        if pass_len > 1e-4:
                            pass_dir = pass_vec / pass_len
                            perp_cand = np.array([-pass_dir[1], pass_dir[0]], dtype=np.float64)
                            to_blocker = c_player_pos[best_blocker] - carrier_c
                            perp_proj = float(np.dot(to_blocker, perp_cand))
                            n_perp = -perp_cand if perp_proj > 0.0 else perp_cand
                            lat_unmask_disp = float(np.dot(c_player_pos[idx] - c_prev_pos[idx], n_perp))
                            if lat_unmask_disp > 0.0:
                                step_rewards[idx] += support_mult * rew.unmask_lateral_scale * lat_unmask_disp

            # 3. State B: Ball is Loose
            elif self.ball_holder is None:
                # Keep non-chasing center backs compact (150u separation) guarding the central channel
                y_shift_loose = float(np.clip(0.15 * (float(c_ball_pos[1]) - 420.0), -35.0, 35.0))
                def1_loose_y = 420.0 + y_shift_loose - 75.0
                def2_loose_y = 420.0 + y_shift_loose + 75.0
                role_anchors = {
                    "def_1": np.array([np.clip(0.48 * c_bx + 90.0, 200.0, 640.0), def1_loose_y], dtype=np.float64),
                    "def_2": np.array([np.clip(0.48 * c_bx + 90.0, 200.0, 640.0), def2_loose_y], dtype=np.float64),
                    "mid_1": np.array([np.clip(0.62 * c_bx + 220.0, 360.0, 960.0), 200.0], dtype=np.float64),
                    "mid_2": np.array([np.clip(0.62 * c_bx + 220.0, 360.0, 960.0), 640.0], dtype=np.float64),
                    "atk_1": np.array([np.clip(0.62 * c_bx + 420.0, 600.0, 1180.0), 420.0], dtype=np.float64),
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
                        # Compute optimal lead intercept point p_intercept along [ball_pos, ball_pos + 0.8 * ball_vel]
                        v_ball_norm = float(np.linalg.norm(self.ball_vel))
                        if v_ball_norm > 1e-3:
                            u_b = self.ball_vel / v_ball_norm
                            d_rx = self.player_pos[best_rx] - self.ball_pos
                            t_proj = float(np.dot(d_rx, u_b))
                            t_target = float(np.clip(t_proj + 60.0, 0.0, 0.8 * v_ball_norm))
                            p_intercept = self.ball_pos + u_b * t_target
                        else:
                            p_intercept = self.ball_pos.copy()

                        for idx in team_fields:
                            if idx == best_rx:
                                c_p_int, _, _ = self._to_team_canonical(p_intercept, np.zeros(2), 0.0, team_id)
                                self.active_tactical_target_canon[idx] = c_p_int.copy()
                                if not any_teleport:
                                    player_disp = self.player_pos[idx] - adj_prev_pos[idx]
                                    to_int_vec = p_intercept - adj_prev_pos[idx]
                                    dist_to_int_prev = max(1e-6, float(np.linalg.norm(to_int_vec)))
                                    u_to_int = to_int_vec / dist_to_int_prev
                                    active_chase_closed = float(np.dot(player_disp, u_to_int))
                                    if float(curr_player_ball_dists[idx]) > 55.0:
                                        step_rewards[idx] += press_mult * rew.loose_ball_chase_scale * max(0.0, active_chase_closed)
                                to_int = p_intercept - self.player_pos[idx]
                                d_int = float(np.linalg.norm(to_int))
                                if d_int > 1e-6:
                                    facing_vec = np.array(
                                        [math.cos(self.player_angle[idx]), math.sin(self.player_angle[idx])],
                                        dtype=np.float64,
                                    )
                                    cos_align = float(np.dot(facing_vec, to_int / d_int))
                                    if cos_align >= 0.9781:  # within +/- 12 degrees
                                        align_factor = 1.0
                                    elif cos_align > 0.0:
                                        align_factor = cos_align
                                    else:
                                        align_factor = 0.0
                                    if align_factor > 0.0:
                                        step_rewards[idx] += press_mult * rew.ball_heading_align_scale * align_factor
                            else:
                                anc = role_anchors[dynamic_role_key[idx]]
                                self.active_tactical_target_canon[idx] = anc.copy()
                                if not any_teleport:
                                    d_prev_anc = max(0.0, float(np.linalg.norm(c_prev_pos[idx] - anc)) - rew.pocket_deadzone_radius)
                                    d_curr_anc = max(0.0, float(np.linalg.norm(c_player_pos[idx] - anc)) - rew.pocket_deadzone_radius)
                                    step_rewards[idx] += rew.role_zone_anchor_scale * (d_prev_anc - d_curr_anc)
                    else:
                        pass
                else:
                    # Case B2: Contested / Unclaimed Loose Ball
                    ranked = ranked_loose_fields
                    for rank_i, idx in enumerate(ranked):
                        is_chaser = (rank_i == 0) or (rank_i == 1 and float(curr_player_ball_dists[idx]) <= 280.0)
                        if is_chaser:
                            self.active_tactical_target_canon[idx] = c_ball_pos.copy()
                            w = (1.0 if rank_i == 0 else 0.45) * press_mult
                            if not any_teleport:
                                player_disp = self.player_pos[idx] - adj_prev_pos[idx]
                                to_ball_vec = self.ball_pos - adj_prev_pos[idx]
                                dist_to_ball_prev = max(1e-6, float(np.linalg.norm(to_ball_vec)))
                                u_to_ball = to_ball_vec / dist_to_ball_prev
                                active_chase_closed = float(np.dot(player_disp, u_to_ball))
                                # Reward active chase continuously all the way until pickup!
                                d_to_ball_curr = float(curr_player_ball_dists[idx])
                                if d_to_ball_curr > mech.pickup_distance:
                                    step_rewards[idx] += w * rew.loose_ball_chase_scale * max(0.0, active_chase_closed)
                                    # Continuous proximity gravitational pull within 95u to eliminate hesitation / deadlock near loose ball
                                    if d_to_ball_curr <= 95.0:
                                        prox_factor = max(0.0, 1.0 - (d_to_ball_curr - mech.pickup_distance) / (95.0 - mech.pickup_distance))
                                        step_rewards[idx] += w * rew.loose_ball_proximity_scale * prox_factor
                            to_ball = self.ball_pos - self.player_pos[idx]
                            d_ball = float(np.linalg.norm(to_ball))
                            if d_ball > 1e-6:
                                facing_vec = np.array(
                                    [math.cos(self.player_angle[idx]), math.sin(self.player_angle[idx])],
                                    dtype=np.float64,
                                )
                                cos_align = float(np.dot(facing_vec, to_ball / d_ball))
                                if cos_align >= 0.9781:  # within +/- 12 degrees
                                    align_factor = 1.0
                                elif cos_align > 0.0:
                                    align_factor = cos_align
                                else:
                                    align_factor = 0.0
                                if align_factor > 0.0:
                                    step_rewards[idx] += w * rew.ball_heading_align_scale * align_factor
                        else:
                            anc = role_anchors[dynamic_role_key[idx]]
                            self.active_tactical_target_canon[idx] = anc.copy()
                            if not any_teleport:
                                d_prev_anc = max(0.0, float(np.linalg.norm(c_prev_pos[idx] - anc)) - rew.pocket_deadzone_radius)
                                d_curr_anc = max(0.0, float(np.linalg.norm(c_player_pos[idx] - anc)) - rew.pocket_deadzone_radius)
                                step_rewards[idx] += rew.role_zone_anchor_scale * (d_prev_anc - d_curr_anc)
                            # non_chaser_ball_repulsion is disabled on loose balls

            # 4. State C: Opponent Has the Ball
            else:
                # Keep non-pressing center backs compact (150u separation) directly shielding the goal mouth
                y_shift_opp = float(np.clip(0.18 * (float(c_ball_pos[1]) - 420.0), -40.0, 40.0))
                def1_opp_y = 420.0 + y_shift_opp - 75.0
                def2_opp_y = 420.0 + y_shift_opp + 75.0
                cover_anchors = {
                    "def_1": np.array([np.clip(0.46 * c_bx + 80.0, 185.0, 600.0), def1_opp_y], dtype=np.float64),
                    "def_2": np.array([np.clip(0.46 * c_bx + 80.0, 185.0, 600.0), def2_opp_y], dtype=np.float64),
                    "mid_1": np.array([np.clip(0.58 * c_bx + 190.0, 310.0, 860.0), 210.0], dtype=np.float64),
                    "mid_2": np.array([np.clip(0.58 * c_bx + 190.0, 310.0, 860.0), 630.0], dtype=np.float64),
                    "atk_1": np.array([np.clip(0.58 * c_bx + 380.0, 500.0, 1080.0), 420.0], dtype=np.float64),
                }

                ranked = sorted(team_fields, key=_effective_chase_dist)
                press_radius_2nd = 320.0 if score_diff < 0 else 260.0
                for rank_i, idx in enumerate(ranked):
                    d_curr = float(curr_player_ball_dists[idx])
                    if rank_i == 0 or (rank_i == 1 and d_curr <= press_radius_2nd):
                        self.active_tactical_target_canon[idx] = c_ball_pos.copy()
                        w = (1.0 if rank_i == 0 else 0.6) * press_mult
                        if not any_teleport:
                            dist_closed = float(adj_prev_player_ball_dists[idx]) - d_curr
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
                        anc = cover_anchors[dynamic_role_key[idx]]
                        self.active_tactical_target_canon[idx] = anc.copy()
                        if not any_teleport:
                            d_prev_anc = max(0.0, float(np.linalg.norm(c_prev_pos[idx] - anc)) - rew.pocket_deadzone_radius)
                            d_curr_anc = max(0.0, float(np.linalg.norm(c_player_pos[idx] - anc)) - rew.pocket_deadzone_radius)
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

            # Active GK-Buffer Obstruction Penalty (Only After Enemy GK Holds the Ball):
            opp_keeper_idx = self._team_keeper_idx(1 - team_id)
            if self.ball_holder == opp_keeper_idx and self.ball_hold_timer >= 0.15:
                for idx in team_fields:
                    d_to_enemy_goal = float(np.linalg.norm(c_player_pos[idx] - enemy_goal_c_canon))
                    if d_to_enemy_goal < self.cfg.mechanics.gk_hold_opponent_buffer + 5.0:
                        # Check if player is trying to drive toward enemy goal center instead of vacating the buffer
                        to_goal_c = enemy_goal_c_canon - c_player_pos[idx]
                        u_goal = to_goal_c / max(1e-6, d_to_enemy_goal)
                        facing_c = np.array(
                            [math.cos(c_player_angle[idx]), math.sin(c_player_angle[idx])],
                            dtype=np.float64,
                        )
                        if float(np.dot(facing_c, u_goal)) > 0.35:
                            step_rewards[idx] -= rew.gk_buffer_obstruction_pen

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
                and ray_norm <= phys.keeper_box_radius + 25.0
            ):
                self.active_tactical_target_canon[keeper_idx] = c_ball_pos.copy()
                if not any_teleport:
                    player_disp_k = self.player_pos[keeper_idx] - adj_prev_pos[keeper_idx]
                    to_ball_vec_k = self.ball_pos - adj_prev_pos[keeper_idx]
                    dist_to_ball_prev_k = max(1e-6, float(np.linalg.norm(to_ball_vec_k)))
                    u_to_ball_k = to_ball_vec_k / dist_to_ball_prev_k
                    active_chase_closed_k = float(np.dot(player_disp_k, u_to_ball_k))
                    if active_chase_closed_k > 0.0:
                        step_rewards[keeper_idx] += rew.keeper_sweeper_chase_scale * active_chase_closed_k
                    dist_closed_k = float(adj_prev_player_ball_dists[keeper_idx]) - float(
                        curr_player_ball_dists[keeper_idx]
                    )
                    step_rewards[keeper_idx] += rew.keeper_fetch_box_scale * dist_closed_k
                if cos_k_ball >= 0.9781:
                    step_rewards[keeper_idx] += rew.ball_heading_align_scale
                elif cos_k_ball > 0.0:
                    step_rewards[keeper_idx] += rew.ball_heading_align_scale * cos_k_ball
                # Urgency penalty every step loose ball sits in GK box (flat Frame-1 penalty + ramp)
                step_rewards[keeper_idx] -= (
                    0.015
                    + 0.025
                    * (
                        self.box_loose_ball_timer
                        / max(1e-6, self.cfg.mechanics.keeper_box_loose_max_time)
                    )
                )
            else:
                # Dynamic bisector depth scaling (38.0u near midfield/deep down to 85.0u as opponent approaches within 420.0u)
                t_depth = float(np.clip((1400.0 - ray_norm) / (1400.0 - 420.0), 0.0, 1.0))
                target_depth = 38.0 + (85.0 - 38.0) * t_depth
                raw_ideal = own_goal_c + (ray / ray_norm) * target_depth
                ideal_k_pos = np.array(
                    [
                        np.clip(raw_ideal[0], 20.0, 95.0),
                        np.clip(raw_ideal[1], phys.goal_y_min + 8.0, phys.goal_y_max - 8.0),
                    ],
                    dtype=np.float64,
                )
                self.active_tactical_target_canon[keeper_idx] = ideal_k_pos.copy()
                if not any_teleport:
                    d_prev_k_ideal = float(np.linalg.norm(c_prev_pos[keeper_idx] - ideal_k_pos))
                    d_curr_k_ideal = float(np.linalg.norm(c_player_pos[keeper_idx] - ideal_k_pos))
                    delta_ideal = d_prev_k_ideal - d_curr_k_ideal
                    # Gate positive reward on facing field (cos >= 0) or retreat (delta <= 0)
                    if delta_ideal <= 0.0 or math.cos(k_angle_c) >= 0.0:
                        step_rewards[keeper_idx] += rew.keeper_anchor_approach_scale * delta_ideal

                d_curr_k_ideal = float(np.linalg.norm(c_player_pos[keeper_idx] - ideal_k_pos))
                dist_from_goal = float(np.linalg.norm(c_player_pos[keeper_idx] - own_goal_c))
                if self.ball_holder != keeper_idx and dist_from_goal <= rew.keeper_edge_camp_radius:
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

            # 6. Margin-Gated Soft Line-Ordering Shaping
            if self.ball_holder is None:
                primary_chaser_idx = self.primary_loose_ball_chaser[team_id]
            elif holder_team != team_id:
                primary_chaser_idx = ranked[0] if len(ranked) > 0 else -1
            else:
                primary_chaser_idx = -1

            if mid_indices:
                max_mid_x = max(float(c_player_pos[m, 0]) for m in mid_indices)
                for d_idx in def_indices:
                    if d_idx != self.ball_holder and d_idx != primary_chaser_idx:
                        d_x = float(c_player_pos[d_idx, 0])
                        if d_x > max_mid_x + 45.0:
                            step_rewards[d_idx] -= rew.line_order_pen_scale
                            if not any_teleport:
                                retreat_dist = float(c_prev_pos[d_idx, 0]) - d_x
                                if retreat_dist > 0.0:
                                    step_rewards[d_idx] += rew.role_zone_anchor_scale * retreat_dist

            if atk_indices:
                atk_x = float(c_player_pos[atk_indices[0], 0])
                for m_idx in mid_indices:
                    if m_idx != self.ball_holder and m_idx != primary_chaser_idx:
                        m_x = float(c_player_pos[m_idx, 0])
                        if m_x > atk_x + 60.0:
                            step_rewards[m_idx] -= 0.5 * rew.line_order_pen_scale

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
            if self.is_ghost[idx]:
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
                "box_radii": [round(float(self.active_box_radius[0]), 1), round(float(self.active_box_radius[1]), 1)],
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

    def get_centralized_state(self) -> np.ndarray:
        """Compute the (12, 54) canonical centralized state matrix for the CTDE Critic."""
        central_state = np.zeros((self.num_players, 54), dtype=np.float32)

        c_p_a, c_v_a, _ = self._to_team_canonical(self.player_pos, self.player_vel, 0.0, TEAM_A)
        c_b_a, c_bv_a, _ = self._to_team_canonical(self.ball_pos, self.ball_vel, 0.0, TEAM_A)

        c_p_b, c_v_b, _ = self._to_team_canonical(self.player_pos, self.player_vel, 0.0, TEAM_B)
        c_b_b, c_bv_b, _ = self._to_team_canonical(self.ball_pos, self.ball_vel, 0.0, TEAM_B)

        # Team A (agents 0..5) canonical state:
        tm_a = np.empty((6, 4), dtype=np.float32)
        tm_a[:, 0] = c_p_a[:6, 0] / 700.0 - 1.0
        tm_a[:, 1] = c_p_a[:6, 1] / 420.0 - 1.0
        tm_a[:, 2] = c_v_a[:6, 0] / 200.0
        tm_a[:, 3] = c_v_a[:6, 1] / 200.0

        opp_a = np.empty((6, 4), dtype=np.float32)
        opp_a[:, 0] = c_p_a[6:, 0] / 700.0 - 1.0
        opp_a[:, 1] = c_p_a[6:, 1] / 420.0 - 1.0
        opp_a[:, 2] = c_v_a[6:, 0] / 200.0
        opp_a[:, 3] = c_v_a[6:, 1] / 200.0

        ctx_a = np.array([
            c_b_a[0] / 700.0 - 1.0,
            c_b_a[1] / 420.0 - 1.0,
            c_bv_a[0] / 640.0,
            c_bv_a[1] / 640.0,
            self.team_pass_chain[TEAM_A] / 10.0,
            self.active_box_radius[self.team_side[TEAM_A]] / 195.0,
        ], dtype=np.float32)

        row_a = np.clip(np.concatenate([tm_a.reshape(-1), opp_a.reshape(-1), ctx_a]), -1.0, 1.0)
        central_state[:6] = row_a

        # Team B (agents 6..11) canonical state:
        tm_b = np.empty((6, 4), dtype=np.float32)
        tm_b[:, 0] = c_p_b[6:, 0] / 700.0 - 1.0
        tm_b[:, 1] = c_p_b[6:, 1] / 420.0 - 1.0
        tm_b[:, 2] = c_v_b[6:, 0] / 200.0
        tm_b[:, 3] = c_v_b[6:, 1] / 200.0

        opp_b = np.empty((6, 4), dtype=np.float32)
        opp_b[:, 0] = c_p_b[:6, 0] / 700.0 - 1.0
        opp_b[:, 1] = c_p_b[:6, 1] / 420.0 - 1.0
        opp_b[:, 2] = c_v_b[:6, 0] / 200.0
        opp_b[:, 3] = c_v_b[:6, 1] / 200.0

        ctx_b = np.array([
            c_b_b[0] / 700.0 - 1.0,
            c_b_b[1] / 420.0 - 1.0,
            c_bv_b[0] / 640.0,
            c_bv_b[1] / 640.0,
            self.team_pass_chain[TEAM_B] / 10.0,
            self.active_box_radius[self.team_side[TEAM_B]] / 195.0,
        ], dtype=np.float32)

        row_b = np.clip(np.concatenate([tm_b.reshape(-1), opp_b.reshape(-1), ctx_b]), -1.0, 1.0)
        central_state[6:] = row_b

        return central_state

    def _build_agent_info(self, agent: str, central_state: Optional[np.ndarray] = None) -> Dict[str, Any]:
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
        c_state = central_state if central_state is not None else self.get_centralized_state()[idx]
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
            "central_state": c_state,
        }

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
