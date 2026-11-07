"""PettingZoo ParallelEnv implementation for 6v6 Continuous 2D MARL Football."""

from __future__ import annotations

import copy
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

        self._entity_slot_matrix: np.ndarray = np.array(
            [self._entity_player_slots[i] for i in range(self.num_players)],
            dtype=np.int32,
        )
        self._row_idx_12x1: np.ndarray = np.arange(self.num_players, dtype=np.int32)[:, None]
        self._max_self_vel: np.ndarray = np.array(
            [
                self.cfg.mechanics.lunge_speed
                if self.player_role[i] == ROLE_KEEPER
                else self._get_player_max_speed(i)
                for i in range(self.num_players)
            ],
            dtype=np.float64,
        )
        self._role_tackle_reach: np.ndarray = np.array(
            [self._get_role_tackle_params(i)[0] for i in range(self.num_players)],
            dtype=np.float64,
        )
        self._field_role_onehot: np.ndarray = np.zeros((self.num_players, 3), dtype=np.float32)
        for i in range(self.num_players):
            if self.player_role[i] == ROLE_DEF:
                self._field_role_onehot[i, 0] = 1.0
            elif self.player_role[i] == ROLE_MID:
                self._field_role_onehot[i, 1] = 1.0
            elif self.player_role[i] == ROLE_ATK:
                self._field_role_onehot[i, 2] = 1.0

        # Precompute max diagonal distance for normalization
        self._max_field_dist: float = math.hypot(
            self.cfg.physics.field_width, self.cfg.physics.field_height
        )
        self.max_pitch_diag: float = float(self._max_field_dist)

        # Potential State & Steering Vectors
        self.raw_steer_input: np.ndarray = np.zeros(self.num_players, dtype=np.float32)
        self.prev_ball_potential: np.ndarray = np.zeros(2, dtype=np.float64)
        self.prev_tactical_potential: np.ndarray = np.zeros(self.num_players, dtype=np.float64)

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
        self._box_loose_ball_timer: _DualBoxTimer = _DualBoxTimer(0.0, 0.0)
        self.last_ball_holder: Optional[int] = None
        self.last_ball_team: Optional[int] = None
        self.ball_kick_origin: np.ndarray = np.zeros(2, dtype=np.float64)
        self.ball_kick_orig_opp_dist: float = -1.0
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
        self.smoothed_steer: np.ndarray = np.zeros(self.num_players, dtype=np.float64)
        self.prev_steer: np.ndarray = np.zeros(self.num_players, dtype=np.float64)
        self.dive_dir: np.ndarray = np.zeros((self.num_players, 2), dtype=np.float64)
        self.dive_speed: np.ndarray = np.zeros(self.num_players, dtype=np.float64)

        # Pass chain & goal assist tracking
        self.team_pass_chain: List[int] = [0, 0]
        self.last_completed_pass_passer: list[Optional[int]] = [None, None]
        self.last_completed_pass_receiver: list[Optional[int]] = [None, None]
        self.last_completed_pass_time: np.ndarray = np.full(2, -999.0, dtype=np.float32)

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
        self._deferred_replay: Optional[Dict[str, Any]] = None
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

    @property
    def prev_player_potential(self) -> np.ndarray:
        return self.prev_tactical_potential

    @property
    def box_loose_ball_timer(self) -> _DualBoxTimer:
        return self._box_loose_ball_timer

    @box_loose_ball_timer.setter
    def box_loose_ball_timer(self, val: Any) -> None:
        if isinstance(val, _DualBoxTimer):
            self._box_loose_ball_timer = val
        elif isinstance(val, (int, float, np.floating)):
            self._box_loose_ball_timer.fill(float(val))
        elif isinstance(val, (list, tuple, np.ndarray)):
            self._box_loose_ball_timer[0] = float(val[0])
            self._box_loose_ball_timer[1] = float(val[1])

    def _is_ghost(self, idx: int) -> bool:
        """Return True if player idx is non-collidable (ghost mode or immobile post-lunge/victim stun)."""
        return bool(
            (self.ghost_timer[idx] > 0.0)
            or ((self.stun_timer[idx] > 0.0) and (self.lunge_timer[idx] <= 0.0))
        )

    def _compute_ball_potential(
        self,
        team_id: int,
        ref_world_pos: Optional[np.ndarray] = None,
    ) -> float:
        """Compute bounded territorial ball threat potential in [0.0, 1.0] for `team_id`."""
        if ref_world_pos is None:
            ref_world_pos = (
                self.player_pos[self.ball_holder]
                if self.ball_holder is not None
                else self.ball_pos
            )
        c_ref = self._to_canonical_pos(ref_world_pos, team_id)
        d_goal = math.hypot(1400.0 - float(c_ref[0]), 420.0 - float(c_ref[1]))
        linear_term = _fast_clip(1.0 - (d_goal / self.max_pitch_diag), 0.0, 1.0)
        gauss_term = math.exp(-(d_goal * d_goal) / (2.0 * 380.0 * 380.0))
        return _fast_clip(0.65 * linear_term + 0.35 * gauss_term, 0.0, 1.0)

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
            "assists": 0,
            "team_assists": [0, 0],
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
        self.smoothed_steer.fill(0.0)
        self.prev_steer.fill(0.0)
        self.dive_dir.fill(0.0)
        self.dive_speed.fill(0.0)
        self.raw_steer_input.fill(0.0)
        self.prev_tactical_potential.fill(0.0)
        self._box_loose_ball_timer.fill(0.0)
        self._reset_pass_chains()

        # Set initial kickoff formation and initialize Markovian potentials
        self._setup_kickoff(starting_team=self.first_half_kickoff_team)
        self.prev_ball_potential[:] = [
            self._compute_ball_potential(TEAM_A),
            self._compute_ball_potential(TEAM_B),
        ]
        self._compute_potential_shaping(
            np.zeros(self.num_players, dtype=np.float64), any_teleport=True
        )

        # Record initial frame (t = 0.0s) if recording enabled
        if self.cfg.replay.enable_recording:
            self._record_replay_frame()

        observations = self._get_observations()
        central_states = self.get_centralized_state()
        infos = {
            agent: self._build_agent_info(
                agent,
                central_state=central_states[self.agent_to_idx[agent]],
                interest_score=0.0,
            )
            for agent in self.agents
        }
        return observations, infos

    def _step_core(self, act_array: np.ndarray) -> Tuple[np.ndarray, bool, bool]:
        """Execute physics sub-steps, PBRS, and end-of-episode bookkeeping for a (12, 4) action array."""
        # Store raw steering input for instantaneous kick deflection
        self.raw_steer_input = act_array[:, 1].astype(np.float32)

        step_rewards = np.zeros(self.num_players, dtype=np.float64)
        phi_ball_start = np.array(
            [
                self._compute_ball_potential(TEAM_A),
                self._compute_ball_potential(TEAM_B),
            ],
            dtype=np.float64,
        )
        any_teleport = False

        # Run 4 physics sub-steps (60 Hz physics / 15 Hz RL)
        for _ in range(self.cfg.physics.frame_skip):
            self._frame_events = []
            teleported = self._physics_substep(act_array, step_rewards)
            if teleported:
                any_teleport = True
                act_array = np.zeros_like(act_array)

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

        # Tier 2: Strict Policy-Invariant PBRS (Ball Threat & Off-Ball Positioning)
        phi_ball_end = np.array(
            [
                self._compute_ball_potential(TEAM_A),
                self._compute_ball_potential(TEAM_B),
            ],
            dtype=np.float64,
        )
        self.prev_ball_potential[:] = phi_ball_end
        if not any_teleport:
            gamma = self.cfg.ppo.gamma
            rew = self.cfg.rewards
            for team_id in (TEAM_A, TEAM_B):
                delta_phi = gamma * float(phi_ball_end[team_id]) - float(phi_ball_start[team_id])
                pbrs_adv = rew.team_advance_pbrs_weight * delta_phi
                if abs(pbrs_adv) > 1e-12:
                    self._add_team_reward(step_rewards, team_id, pbrs_adv)
                if self.ball_holder is not None and int(self.player_team[self.ball_holder]) == team_id:
                    step_rewards[self.ball_holder] += rew.carrier_advance_pbrs_weight * delta_phi

        self._compute_potential_shaping(step_rewards, any_teleport)

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

        return step_rewards, is_terminated, is_truncated

    def step_array(
        self, actions_array: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, Optional[Dict[str, Any]]]:
        """Execute one RL step using a (12, 4) action array and return contiguous NumPy arrays."""
        act_array = np.clip(np.asarray(actions_array, dtype=np.float64), -1.0, 1.0)
        step_rewards, is_terminated, is_truncated = self._step_core(act_array)

        obs_arr = self._get_observations_array()
        central_arr = self.get_centralized_state()
        rew_arr = self.last_step_rewards.copy()
        is_done = bool(is_terminated or is_truncated)

        if is_done:
            ep_interest = self.compute_interest_score()
            ep_summary: Optional[Dict[str, Any]] = {
                "match_id": self.current_match_id,
                "generation": self.current_generation,
                "interest_score": ep_interest,
                "final_stats": dict(self.stats),
                "last_saved_replay": self.last_saved_replay_path,
            }
            self.agents = []
        else:
            ep_summary = None

        return obs_arr, central_arr, rew_arr, is_done, ep_summary

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

        step_rewards, is_terminated, is_truncated = self._step_core(act_array)

        observations = self._get_observations()
        rewards = {
            agent: float(step_rewards[self.agent_to_idx[agent]]) for agent in self.agents
        }
        terminations = {agent: is_terminated for agent in self.agents}
        truncations = {agent: is_truncated for agent in self.agents}

        ep_interest = self.compute_interest_score() if (is_truncated or is_terminated) else 0.0
        central_states = self.get_centralized_state()
        infos = {
            agent: self._build_agent_info(
                agent,
                central_state=central_states[self.agent_to_idx[agent]],
                interest_score=ep_interest,
            )
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
        self.is_ghost[:] = (self.ghost_timer > 0.0) | ((self.stun_timer > 0.0) & (self.lunge_timer <= 0.0))
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

        # Track possession stats, hold timer & per-box anti-stall timers
        left_center = np.array(phys.left_goal_center, dtype=np.float64)
        right_center = np.array(phys.right_goal_center, dtype=np.float64)
        d_left = float(np.linalg.norm(self.ball_pos - left_center))
        d_right = float(np.linalg.norm(self.ball_pos - right_center))

        in_left_box = d_left <= 205.0
        in_right_box = d_right <= 205.0
        ball_spd = float(np.linalg.norm(self.ball_vel))

        for side_idx, (in_box, d_goal) in enumerate([(in_left_box, d_left), (in_right_box, d_right)]):
            defending_team = 0 if self.team_side[0] == side_idx else 1

            if not in_box or (self.ball_holder is not None and int(self.player_team[self.ball_holder]) != defending_team):
                # Ball cleared the box or opponent claimed it -> reset timer & expand
                self._box_loose_ball_timer[side_idx] = 0.0
                if self.active_box_radius[side_idx] < phys.keeper_box_radius:
                    self.active_box_radius[side_idx] = min(
                        phys.keeper_box_radius,
                        float(self.active_box_radius[side_idx]) + mech.box_expand_rate * dt,
                    )
            else:
                # Ball is still inside side_idx's box (either loose or held by defending team)
                if self.ball_holder is None and ball_spd <= mech.box_ball_stopped_threshold:
                    if self.post_save_box_grace_timer <= 0.0:
                        self._box_loose_ball_timer[side_idx] += dt
                # Soft kicks or micro-touches inside the box do NOT reset the timer
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
            elif (
                holder_role == ROLE_KEEPER
                and self.ball_hold_timer >= mech.keeper_max_hold_time - 1e-9
            ):
                self._handle_keeper_5s_violation(self.ball_holder, step_rewards, is_hold_violation=True)

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

            self.prev_steer[idx] = steer_raw
            self.smoothed_steer[idx] = steer_raw

            # Steering: [-1, 1] rotates facing_angle (locked during field lunge)
            if not (self._is_field_role(self.player_role[idx]) and self.lunge_timer[idx] > 0.0):
                turn_delta = steer_raw * phys.player_max_turn_rate * dt
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
                                + steer_raw * math.radians(phys.kick_steer_max_deflection_deg)
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
            elif self._is_field_role(self.player_role[idx]) and self.lunge_timer[idx] > 0.0:
                # Field player lunge: instant override at field_lunge_speed,
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

        # Semi-Circle Barrier & Field Wall Clamping
        self._clamp_players_to_boundaries(apply_dynamic_push=True)

        # ----------------------------------------------------------------------
        # 4. Update Ball Kinematics & Check Goals / Wall Bounces
        # ----------------------------------------------------------------------
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

            # Carrier corner-pin unsticking: increment pinned_ball_timer if ball speed < 18, near wall, opponent <= 55u
            ball_spd_carrier = float(np.linalg.norm(self.ball_vel))
            bx, by = float(self.ball_pos[0]), float(self.ball_pos[1])
            near_outer_wall = (
                bx <= r_b + 22.0
                or bx >= W - (r_b + 22.0)
                or by <= r_b + 22.0
                or by >= H - (r_b + 22.0)
            )
            opp_team = 1 - h_team
            opp_indices = [j for j in range(self.num_players) if self.player_team[j] == opp_team]
            opp_dists = np.linalg.norm(self.player_pos[opp_indices] - self.ball_pos[None, :], axis=1)
            opp_crowding = np.any(opp_dists <= 55.0)

            if ball_spd_carrier < 18.0 and near_outer_wall and opp_crowding:
                self.pinned_ball_timer += dt
                if self.pinned_ball_timer >= 2.0:
                    self.ball_holder = None
                    self.ball_hold_timer = 0.0
                    center_vec = np.array(phys.field_center, dtype=np.float64) - self.ball_pos
                    c_dist = float(np.linalg.norm(center_vec))
                    if c_dist > 1e-6:
                        self.ball_vel = (center_vec / c_dist) * mech.pinned_ball_nudge_speed
                    self.pinned_ball_timer = 0.0
            else:
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
    def _execute_kick(
        self,
        kicker_idx: int,
        kick_power: float,
        step_rewards: Optional[np.ndarray] = None,
    ) -> None:
        """Launch the ball from kicker, record release state, check shot-on-target, and apply pickup cooldown."""
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

        # Kick Deflection Realism & Taper
        abs_deflect_deg = abs(float(self.raw_steer_input[kicker_idx])) * phys.kick_steer_max_deflection_deg
        if abs_deflect_deg > 35.0 and kicker_role != ROLE_KEEPER:
            deflect_excess = _fast_clip(
                (abs_deflect_deg - 35.0)
                / max(1.0, phys.kick_steer_max_deflection_deg - 35.0),
                0.0,
                1.0,
            )
            bonus_speed = power_norm * (max_kick_spd - phys.ball_min_kick_speed) * (1.0 - 0.25 * deflect_excess)
            launch_speed = phys.ball_min_kick_speed + bonus_speed
        else:
            deflect_excess = 0.0

        if power_norm > 0.65 and jitter_rad > 0.0:
            kick_angle = base_angle + float(self.np_random.uniform(-jitter_rad, jitter_rad))
        else:
            kick_angle = base_angle

        if deflect_excess > 0.0 and power_norm > 0.50:
            kick_angle = float(
                wrap_to_pi(
                    kick_angle
                    + self.np_random.uniform(-1.0, 1.0) * math.radians(4.0 * deflect_excess)
                )
            )

        facing_vec = np.array([math.cos(base_angle), math.sin(base_angle)], dtype=np.float64)
        kick_vec = np.array([math.cos(kick_angle), math.sin(kick_angle)], dtype=np.float64)

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
        self.ball_kicked_in_flight = True
        self.ball_kick_flight_timer = 0.0
        self.ball_wall_bounces = 0
        self.is_wall_pass_in_flight = False

        self.pickup_cooldown[kicker_idx] = max(
            self.pickup_cooldown[kicker_idx], mech.kick_pickup_cooldown
        )
        self.kick_cooldown_active[kicker_idx] = True

        if kicker_role == ROLE_KEEPER:
            self.last_keeper_kick_time[kicker_team] = self.physics_frame_count * phys.dt

        # Check if this kick is a True Shot on Target (inside goal mouth)
        enemy_side = 1 - int(self.team_side[kicker_team])
        is_true_sot = self._is_ball_heading_into_goal(self.ball_pos, self.ball_vel, enemy_side, y_margin=0.0)
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

                base_sot = (
                    rew.atk_shot_on_target_ind
                    if kicker_role == ROLE_ATK
                    else rew.other_shot_on_target_ind
                )
                sot_rew = base_sot * mu_xg
                team_sot_rew = rew.shot_on_target_team * mu_xg
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

    def _attempt_tackle(self, tackler_idx: int, step_rewards: np.ndarray) -> None:
        """Execute kinematic lunge and dynamic distance-based tackle check when action[3] > 0.5."""
        if self.kickoff_protection_timer > 0.0:
            return

        mech = self.cfg.mechanics
        rew = self.cfg.rewards

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
        holder = self.ball_holder
        tackler_team = int(self.player_team[tackler_idx])

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

            if dist <= R_max_lunge and facing_dot >= 0.25:
                p_success = max(0.0, min(1.0, float(P_max * (1.0 - (dist / R_max_lunge) ** 2))))
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

                    if step_rewards is not None:
                        step_rewards[tackler_idx] += rew.successful_tackle_tackler_ind
                        step_rewards[old_carrier] += rew.successful_tackle_carrier_ind
                        self._add_team_reward(
                            step_rewards, tackler_team, rew.successful_tackle_team
                        )
                        self._add_team_reward(
                            step_rewards, old_team, rew.intercepted_team
                        )
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

        dists = np.hypot(
            self.player_pos[:, 0] - self.ball_pos[0],
            self.player_pos[:, 1] - self.ball_pos[1],
        )
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
                min_dist = mech.min_wall_pass_distance if is_wall_pass else mech.min_pass_distance
                if kick_travel_dist >= min_dist:
                    # 1. Territorial Threat Gain (Delta Phi_threat)
                    phi_orig = self._compute_ball_potential(new_team, self.ball_kick_origin)
                    phi_dest = self._compute_ball_potential(new_team, self.ball_pos)
                    d_phi_threat = phi_dest - phi_orig

                    # 2. Pressure Relief / Space Gain (Delta sigma_press)
                    opp_mask = self.player_team != new_team
                    if self.ball_kick_orig_opp_dist >= 0.0:
                        d_opp_orig = self.ball_kick_orig_opp_dist
                    else:
                        orig_opp_dists = np.hypot(
                            self.player_pos[opp_mask, 0] - self.ball_kick_origin[0],
                            self.player_pos[opp_mask, 1] - self.ball_kick_origin[1],
                        )
                        d_opp_orig = float(np.min(orig_opp_dists)) if len(orig_opp_dists) > 0 else 999.0
                    dest_opp_dists = np.hypot(
                        self.player_pos[opp_mask, 0] - self.ball_pos[0],
                        self.player_pos[opp_mask, 1] - self.ball_pos[1],
                    )
                    d_opp_dest = float(np.min(dest_opp_dists)) if len(dest_opp_dists) > 0 else 999.0

                    sigma_orig = _fast_clip((d_opp_orig - 45.0) / 135.0, 0.0, 1.0)
                    sigma_dest = _fast_clip((d_opp_dest - 45.0) / 135.0, 0.0, 1.0)
                    d_sigma_press = sigma_dest - sigma_orig

                    # 3. In-Stride Through-Ball Bonus (R_stride)
                    r_stride = 0.0
                    if d_phi_threat > 0.02:
                        c_rx_pos, c_rx_vel, _ = self._to_team_canonical(
                            self.player_pos[new_holder : new_holder + 1],
                            self.player_vel[new_holder : new_holder + 1],
                            0.0,
                            new_team,
                        )
                        gx_dx = 1400.0 - float(c_rx_pos[0, 0])
                        gx_dy = 420.0 - float(c_rx_pos[0, 1])
                        d_rx_goal = max(1e-6, math.hypot(gx_dx, gx_dy))
                        v_goal = (
                            float(c_rx_vel[0, 0]) * (gx_dx / d_rx_goal)
                            + float(c_rx_vel[0, 1]) * (gx_dy / d_rx_goal)
                        )
                        if v_goal >= 60.0:
                            r_stride = (
                                rew.lead_pass_bonus
                                * _fast_clip((v_goal - 60.0) / 130.0, 0.0, 1.0)
                                * _fast_clip(d_phi_threat / 0.08, 0.0, 1.0)
                            )

                    # Unified State-Based Pass Payout (60% passer, 40% receiver + conditional touch bonus)
                    r_pass = (
                        rew.pass_threat_weight * max(0.0, d_phi_threat)
                        + rew.pass_space_weight * max(0.0, d_sigma_press)
                        + r_stride
                    )
                    touch_bonus = (
                        rew.completed_pass_receiver_ind
                        if (d_phi_threat > 0.01 or d_sigma_press > 0.15)
                        else 0.0
                    )
                    passer_bonus = 0.60 * r_pass
                    receiver_bonus = 0.40 * r_pass + touch_bonus

                    if d_phi_threat > 0.01 or d_sigma_press > 0.15:
                        self.team_pass_chain[new_team] = min(self.team_pass_chain[new_team] + 1, 10)

                    self.last_completed_pass_passer[new_team] = kicker_idx
                    self.last_completed_pass_receiver[new_team] = new_holder
                    self.last_completed_pass_time[new_team] = self.match_time

                    step_rewards[kicker_idx] += passer_bonus
                    step_rewards[new_holder] += receiver_bonus
                    self.stats["completed_passes"] += 1
                    self.stats["team_passes"][new_team] += 1

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
                        reward=passer_bonus + receiver_bonus,
                    )
            elif new_team != kicker_team:
                # Interception or Keeper Save (Possession Transition)
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
                    step_rewards[kicker_idx] += rew.intercepted_kicker_ind
                    self._add_team_reward(step_rewards, kicker_team, rew.intercepted_team)
                    step_rewards[new_holder] += rew.successful_tackle_tackler_ind
                    self._add_team_reward(step_rewards, new_team, rew.successful_tackle_team)
                    self.stats["interceptions"] += 1
                else:
                    self.post_save_box_grace_timer = mech.post_save_box_grace_time
                    gk_goal_c = np.array(
                        [0.0 if my_side == SIDE_LEFT else phys.field_width, 420.0],
                        dtype=np.float64,
                    )
                    d_orig = float(np.linalg.norm(self.ball_kick_origin - gk_goal_c))
                    v_arr = float(np.linalg.norm(self.ball_vel))
                    v_fact = _fast_clip((v_arr - 120.0) / (420.0 - 120.0), 0.15, 1.0)
                    d_fact = _fast_clip((550.0 - d_orig) / (550.0 - 220.0), 0.20, 1.0)
                    mu_save = _fast_clip(v_fact * d_fact, 0.10, 1.0)

                    step_rewards[new_holder] += rew.keeper_save_ind * mu_save
                    self._add_team_reward(step_rewards, new_team, rew.keeper_save_team * mu_save)

                    if d_orig > 460.0 or v_arr < 220.0:
                        step_rewards[kicker_idx] += rew.intercepted_kicker_ind * (1.0 - mu_save)

                    self.stats["keeper_saves"] += 1
                    self.stats["team_saves"][new_team] += 1
                    lbl = "LUNGE SAVE!" if self.lunge_timer[new_holder] > 0.0 else "KEEPER SAVE!"
                    self._emit_event(
                        lbl,
                        self.player_pos[new_holder],
                        team=new_team,
                        agent=self.idx_to_agent[new_holder],
                        reward=(rew.keeper_save_ind + rew.keeper_save_team) * mu_save,
                    )
        else:
            # Loose ball recovery
            if self.last_ball_team is not None and self.last_ball_team != new_team:
                self._reset_pass_chains(self.last_ball_team)
                step_rewards[new_holder] += rew.successful_tackle_tackler_ind
                self._add_team_reward(step_rewards, new_team, rew.successful_tackle_team)
                opp_mask = self.player_team != new_team
                opp_dists = np.hypot(
                    self.player_pos[opp_mask, 0] - self.ball_pos[0],
                    self.player_pos[opp_mask, 1] - self.ball_pos[1],
                )
                if len(opp_dists) > 0 and float(np.min(opp_dists)) <= mech.contested_ball_distance:
                    self._emit_event(
                        "BALL WON!",
                        self.player_pos[new_holder],
                        team=new_team,
                        agent=self.idx_to_agent[new_holder],
                        reward=rew.successful_tackle_tackler_ind + rew.successful_tackle_team,
                    )

        # Assign possession to new_holder
        self.ball_holder = new_holder
        self.ball_hold_timer = 0.0
        for side_idx in (0, 1):
            defending_team = 0 if self.team_side[0] == side_idx else 1
            goal_c = np.array(phys.left_goal_center if side_idx == 0 else phys.right_goal_center, dtype=np.float64)
            d_box = float(np.linalg.norm(self.ball_pos - goal_c))
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
        for idx in range(self.num_players):
            if idx == toucher_idx:
                continue
            if self.violation_cooldown_active[idx]:
                if int(self.player_team[idx]) != toucher_team:
                    self.pickup_cooldown[idx] = 0.0
                    self.violation_cooldown_active[idx] = False
            if self.kick_cooldown_active[idx]:
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
        """Enforce Keeper 5-Second Hold Violation or Unclaimed Box Ball Timeout with a ball drop and radial nudge out of the box."""
        rew = self.cfg.rewards
        phys = self.cfg.physics
        penalized_team = int(self.player_team[keeper_idx])
        own_side = int(self.team_side[penalized_team])

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

        # Standard ball drop + 2.0s keeper pickup cooldown + radial nudge out of the box toward midfield
        self.ball_holder = None
        self.ball_hold_timer = 0.0
        self.ball_kicked_in_flight = False
        self._box_loose_ball_timer[own_side] = 0.0
        self._reset_pass_chains(penalized_team)

        self.pickup_cooldown[keeper_idx] = max(self.pickup_cooldown[keeper_idx], 2.0)
        self.violation_cooldown_active[keeper_idx] = True
        self.kick_cooldown_active[keeper_idx] = False

        own_goal_c = np.array(
            phys.left_goal_center if own_side == SIDE_LEFT else phys.right_goal_center,
            dtype=np.float64,
        )
        midfield_c = np.array(phys.field_center, dtype=np.float64)
        delta = self.ball_pos - own_goal_c
        dist = math.hypot(float(delta[0]), float(delta[1]))
        to_mid = midfield_c - own_goal_c
        to_mid_norm = math.hypot(float(to_mid[0]), float(to_mid[1]))
        u_mid = to_mid / max(1e-6, to_mid_norm)
        if dist < 1e-4:
            nudge_dir = u_mid
        else:
            u_rad = delta / dist
            blend = 0.65 * u_rad + 0.35 * u_mid
            blend_norm = math.hypot(float(blend[0]), float(blend[1]))
            nudge_dir = blend / max(1e-6, blend_norm)

        target_r = phys.keeper_box_radius + phys.ball_radius + 25.0
        self.ball_pos = own_goal_c + nudge_dir * max(dist + 35.0, target_r)
        self.ball_pos[0] = float(
            np.clip(self.ball_pos[0], phys.ball_radius, phys.field_width - phys.ball_radius)
        )
        self.ball_pos[1] = float(
            np.clip(self.ball_pos[1], phys.ball_radius, phys.field_height - phys.ball_radius)
        )
        self.ball_vel = nudge_dir * 160.0

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
                dist = math.hypot(float(delta[0]), float(delta[1]))
                if dist > box_r:
                    self.player_pos[idx] = own_goal + (delta / dist) * box_r
            else:
                for side_i, g_center in enumerate(goal_centers):
                    eff_box_r = float(self.active_box_radius[side_i])
                    if eff_box_r <= 1e-3:
                        continue
                    delta = self.player_pos[idx] - g_center
                    dist = math.hypot(float(delta[0]), float(delta[1]))
                    if dist <= eff_box_r:
                        if dist < 1e-6:
                            push_dir = np.array(
                                [1.0 if g_center[0] == 0.0 else -1.0, 0.0], dtype=np.float64
                            )
                        else:
                            push_dir = delta / dist
                        new_pos = g_center + push_dir * (eff_box_r + 1e-3)
                        if apply_dynamic_push:
                            rad_v = float(np.dot(self.player_vel[idx], push_dir))
                            if rad_v < 0.0:
                                self.player_vel[idx] -= rad_v * push_dir
                        self.player_pos[idx] = new_pos

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
                    dist = math.hypot(float(delta[0]), float(delta[1]))
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
                        # Strip any inward radial velocity so the player smoothly slides outward
                        rad_v = float(np.dot(self.player_vel[idx], push_dir))
                        if rad_v < 0.0:
                            self.player_vel[idx] -= rad_v * push_dir

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

        if self.last_ball_holder is not None:
            if int(self.player_team[self.last_ball_holder]) == scoring_team:
                if int(self.player_role[self.last_ball_holder]) == ROLE_ATK:
                    scorer_ind = rew.atk_goal_scorer_ind
                else:
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
        self.popup_cooldown[:] = 0.0
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
        self.smoothed_steer.fill(0.0)
        self.prev_steer.fill(0.0)
        self.dive_dir.fill(0.0)
        self.dive_speed.fill(0.0)
        self._reset_pass_chains()

    def _reset_pass_chains(self, team: Optional[int] = None) -> None:
        """Reset pass chain and assist tracking on turnover or dead ball."""
        if team is None:
            self.team_pass_chain = [0, 0]
            self.last_completed_pass_passer = [None, None]
            self.last_completed_pass_receiver = [None, None]
            self.last_completed_pass_time.fill(-999.0)
        else:
            self.team_pass_chain[team] = 0
            self.last_completed_pass_passer[team] = None
            self.last_completed_pass_receiver[team] = None
            self.last_completed_pass_time[team] = -999.0

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

    # ==========================================================================
    # Team-Canonical Observation Builder (76 Floats, Vectorized Across 12 Agents)
    # ==========================================================================
    def _get_observations_array(self) -> np.ndarray:
        """Compute the (12, 76) normalized observation matrix for all 12 agents simultaneously."""
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
        sx = np.empty(self.num_players, dtype=np.float64)
        sy = np.empty(self.num_players, dtype=np.float64)
        svx = np.empty(self.num_players, dtype=np.float64)
        svy = np.empty(self.num_players, dtype=np.float64)
        s_ang = np.empty(self.num_players, dtype=np.float64)

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

        # Clean Role One-Hot (field players) & GK Box State (goalkeepers) in obs[:, 13:16]
        obs[:, 13:16] = self._field_role_onehot
        side_a = int(self.team_side[TEAM_A])
        side_b = int(self.team_side[TEAM_B])
        obs[0, 13] = 0.0
        obs[0, 14] = np.clip(self.active_box_radius[side_a] / phys.keeper_box_radius, 0.0, 1.0)
        obs[0, 15] = np.clip(
            self._box_loose_ball_timer[side_a] / mech.keeper_box_loose_max_time, 0.0, 1.0
        )
        obs[6, 13] = 0.0
        obs[6, 14] = np.clip(self.active_box_radius[side_b] / phys.keeper_box_radius, 0.0, 1.0)
        obs[6, 15] = np.clip(
            self._box_loose_ball_timer[side_b] / mech.keeper_box_loose_max_time, 0.0, 1.0
        )

        # Part B: Vision Cone Entity Slots (12 Entities x 5 floats = 60 floats, indices 16..75)
        ent_pos = np.empty((self.num_players, vis.num_entity_slots, 2), dtype=np.float64)
        ent_vel = np.empty((self.num_players, vis.num_entity_slots, 2), dtype=np.float64)

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

        # Slot 0 (Ball): 360-degree awareness with half-time clock multiplexing
        in_ball_cone = (dist[:, 0] <= vis.max_range) & (abs_rel_ang[:, 0] <= vis.half_fov_rad)
        half_time_len = float(phys.half_duration)
        t_in_half = self.match_time % half_time_len
        t_rem_ratio = min(1.0, max(0.0, (half_time_len - t_in_half) / half_time_len))
        ball_vis = np.where(
            in_ball_cone,
            0.20 + 0.80 * t_rem_ratio,
            -0.20 - 0.80 * t_rem_ratio,
        )

        # Slots 1..11 (Teammates & Opponents)
        dist_p = dist[:, 1:]
        abs_ang_p = abs_rel_ang[:, 1:]
        fade_rad = math.radians(15.0)
        rad_90 = math.radians(90.0)
        rad_105 = vis.half_fov_rad
        in_ent_cone = (dist_p <= vis.max_range) & (abs_ang_p <= rad_105)
        w_phi = np.where(abs_ang_p <= rad_90, 1.0, (rad_105 - abs_ang_p) / fade_rad)
        w_d = 1.0 - 0.35 * (dist_p / vis.max_range)
        fov_factor = np.where(in_ent_cone, w_phi * w_d, 0.0)

        base_presence = np.clip(0.25 + 0.65 * fov_factor, 0.25, 0.90)
        disabled_presence = -np.clip(0.25 + 0.65 * fov_factor, 0.25, 0.95)
        other_disabled = (np.maximum(self.ghost_timer, self.stun_timer) > 0.0)[
            self._entity_slot_matrix
        ]
        vis_p = np.where(other_disabled, disabled_presence, base_presence)
        if holder is not None:
            vis_p = np.where(self._entity_slot_matrix == holder, 1.0, vis_p)

        max_rel_vel = phys.ball_max_launch_speed + phys.player_max_speed
        ent_view = obs[:, vis.self_feature_dim :].reshape(
            self.num_players, vis.num_entity_slots, vis.entity_feature_dim
        )
        ent_view[:, 0, 0] = ball_vis
        ent_view[:, 1:, 0] = vis_p
        ent_view[:, :, 1] = np.clip(dist / self._max_field_dist, 0.0, 1.0)
        ent_view[:, :, 2] = rel_ang / math.pi
        ent_view[:, :, 3] = (ent_vel[:, :, 0] - svx[:, None]) / max_rel_vel
        ent_view[:, :, 4] = (ent_vel[:, :, 1] - svy[:, None]) / max_rel_vel

        np.clip(obs, -1.0, 1.0, out=obs)
        return obs

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Compute the 76-float normalized observation dictionary for all 12 agents."""
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
        """Return the 76-float canonical observation for player `idx`."""
        return self._get_observations_array()[idx]

    @property
    def match_time(self) -> float:
        """Current elapsed match time in seconds."""
        return float(self.physics_frame_count * self.cfg.physics.dt)

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
    ) -> None:
        """Evaluate Tier 2 Markovian off-ball and goalkeeper potentials and apply strict PBRS."""
        rew = self.cfg.rewards
        phys = self.cfg.physics
        gamma = self.cfg.ppo.gamma
        inv_max_dist = 1.0 / self._max_field_dist

        pos = self.player_pos
        bx, by = float(self.ball_pos[0]), float(self.ball_pos[1])
        ball_dists = np.hypot(pos[:, 0] - bx, pos[:, 1] - by)
        dist_mat = np.hypot(
            pos[:, 0:1] - pos[None, :, 0],
            pos[:, 1:2] - pos[None, :, 1],
        )

        phi_curr = np.zeros(self.num_players, dtype=np.float64)
        holder = self.ball_holder
        holder_team = int(self.player_team[holder]) if holder is not None else -1

        for team_id in (TEAM_A, TEAM_B):
            own_side = int(self.team_side[team_id])
            own_gx = 0.0 if own_side == SIDE_LEFT else phys.field_width
            own_gy = phys.field_height * 0.5
            f_start = 1 if team_id == TEAM_A else 7
            o_start = 7 if team_id == TEAM_A else 1
            team_fields = range(f_start, f_start + 5)
            opp_fields = range(o_start, o_start + 5)
            keeper_idx = 0 if team_id == TEAM_A else 6

            in_possession = (holder_team == team_id) or (
                holder is None and self.ball_kicked_in_flight and self.last_ball_team == team_id
            )
            pass_chaser = -1
            if holder is None and self.ball_kicked_in_flight and self.last_ball_team == team_id:
                cands = [i for i in team_fields if i != self.last_ball_holder]
                if cands:
                    pass_chaser = min(cands, key=lambda i: float(ball_dists[i]))

            if in_possession:
                for i in team_fields:
                    if i == holder:
                        phi_curr[i] = 0.0
                        continue
                    if i == pass_chaser:
                        phi_curr[i] = -rew.def_press_potential_weight * (
                            float(ball_dists[i]) * inv_max_dist
                        )
                        continue
                    d_ib = float(ball_dists[i])
                    psi_ring = math.exp(-((d_ib - 240.0) ** 2) / (2.0 * 140.0 * 140.0))
                    min_seg = min(
                        self._point_to_segment_dist(pos[o], self.ball_pos, pos[i])
                        for o in opp_fields
                    )
                    psi_lane = _fast_clip(min_seg / 70.0, 0.1, 1.0)
                    sep_pen = 0.0
                    for j in team_fields:
                        if j != i:
                            d_ij = float(dist_mat[i, j])
                            if d_ij < 125.0:
                                ov = 1.0 - d_ij / 125.0
                                sep_pen += ov * ov
                    phi_curr[i] = (
                        rew.support_potential_weight * psi_lane * psi_ring
                        - rew.teammate_sep_potential_weight * sep_pen
                    )
            else:
                presser = min(team_fields, key=lambda i: float(ball_dists[i]))
                for i in team_fields:
                    if i == presser:
                        phi_curr[i] = -rew.def_press_potential_weight * (
                            float(ball_dists[i]) * inv_max_dist
                        )
                    else:
                        px, py = float(pos[i, 0]), float(pos[i, 1])
                        best_o = min(opp_fields, key=lambda o: float(dist_mat[i, o]))
                        ox, oy = float(pos[best_o, 0]), float(pos[best_o, 1])
                        cov_x = ox + 0.35 * (own_gx - ox)
                        cov_y = oy + 0.35 * (own_gy - oy)
                        d_cov = math.hypot(px - cov_x, py - cov_y)
                        sep_pen = 0.0
                        for j in team_fields:
                            if j != i:
                                d_ij = float(dist_mat[i, j])
                                if d_ij < 125.0:
                                    ov = 1.0 - d_ij / 125.0
                                    sep_pen += ov * ov
                        phi_curr[i] = (
                            -rew.def_cover_potential_weight * (d_cov * inv_max_dist)
                            - rew.teammate_sep_potential_weight * sep_pen
                        )

            # Goalkeeper continuous bisector potential
            ray_x = bx - own_gx
            ray_y = by - own_gy
            ray_norm = max(1e-6, math.hypot(ray_x, ray_y))
            t_depth = _fast_clip((1400.0 - ray_norm) / (1400.0 - 420.0), 0.0, 1.0)
            target_depth = 38.0 + (85.0 - 38.0) * t_depth
            bis_x = own_gx + (ray_x / ray_norm) * target_depth
            bis_y = _fast_clip(
                own_gy + (ray_y / ray_norm) * target_depth,
                phys.goal_y_min + 8.0,
                phys.goal_y_max - 8.0,
            )
            if own_side == SIDE_LEFT:
                bis_x = _fast_clip(bis_x, 20.0, 95.0)
            else:
                bis_x = _fast_clip(bis_x, phys.field_width - 95.0, phys.field_width - 20.0)
            gk_x, gk_y = float(pos[keeper_idx, 0]), float(pos[keeper_idx, 1])
            d_bisect = math.hypot(gk_x - bis_x, gk_y - bis_y)
            loose_in_box = (holder is None) and (ray_norm <= phys.keeper_box_radius + 25.0)
            d_loose = float(ball_dists[keeper_idx]) if loose_in_box else 0.0
            phi_curr[keeper_idx] = -rew.keeper_bisect_potential_weight * (
                (d_bisect + d_loose) / 195.0
            )

        if not any_teleport:
            step_rewards += gamma * phi_curr - self.prev_tactical_potential
        self.prev_tactical_potential[:] = phi_curr

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
                    round(float(self.player_vel[idx, 0]), 2),
                    round(float(self.player_vel[idx, 1]), 2),
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
            round(float(self.ball_vel[0]), 2),
            round(float(self.ball_vel[1]), 2),
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

    def _write_replay_to_disk(
        self,
        match_id: int,
        generation: int,
        interest_score: float,
        stats: Dict[str, Any],
        frames: List[Dict[str, Any]],
    ) -> str:
        """Serialize and write a 60 Hz JSON replay trajectory to disk."""
        os.makedirs(self.cfg.replay.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        score_str = f"{interest_score:.1f}".replace(".", "_")
        if match_id > 0:
            filename = (
                f"match_{match_id:04d}_gen{generation:04d}"
                f"_score{score_str}_{timestamp}.json"
            )
        else:
            filename = f"unindexed_{timestamp}_score{score_str}.json"
        filepath = os.path.join(self.cfg.replay.save_dir, filename)

        highlights = []
        for fr in frames:
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
            "match_id": match_id,
            "generation": generation,
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
            "final_stats": stats,
            "highlights": highlights,
        }

        meta_json = json.dumps(metadata_block, indent=2)
        meta_indented = "\n".join("  " + line for line in meta_json.splitlines())
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("{\n  \"metadata\": ")
            f.write(meta_indented.lstrip())
            f.write(",\n  \"frames\": [\n")
            last_idx = len(frames) - 1
            for i, fr in enumerate(frames):
                suffix = ",\n" if i < last_idx else "\n"
                f.write("    " + json.dumps(fr) + suffix)
            f.write("  ]\n}\n")

        self.last_saved_replay_path = filepath
        return filepath

    def _maybe_export_replay(self) -> Optional[str]:
        """Export or defer the full 60 Hz JSON trajectory if `interest_score >= threshold`."""
        interest_score = self.compute_interest_score()
        if interest_score < self.cfg.replay.interest_threshold:
            return None

        if self.cfg.replay.save_best_only:
            cur_goals = int(self.stats.get("goals", 0))
            if self._deferred_replay is None or (
                interest_score,
                cur_goals,
            ) >= (
                float(self._deferred_replay["interest_score"]),
                int(self._deferred_replay["stats"].get("goals", 0)),
            ):
                self._deferred_replay = {
                    "match_id": self.current_match_id,
                    "generation": self.current_generation,
                    "interest_score": interest_score,
                    "stats": copy.deepcopy(self.stats),
                    "frames": list(self.replay_frames),
                }
            self.last_saved_replay_path = None
            return None

        return self._write_replay_to_disk(
            match_id=self.current_match_id,
            generation=self.current_generation,
            interest_score=interest_score,
            stats=self.stats,
            frames=self.replay_frames,
        )

    def export_deferred_replay(self) -> Optional[str]:
        """Write the deferred replay snapshot to disk when selected as the winning worker."""
        if self._deferred_replay is not None:
            payload = self._deferred_replay
            self._deferred_replay = None
            return self._write_replay_to_disk(
                match_id=int(payload["match_id"]),
                generation=int(payload["generation"]),
                interest_score=float(payload["interest_score"]),
                stats=payload["stats"],
                frames=payload["frames"],
            )
        if self.replay_frames:
            interest_score = self.compute_interest_score()
            if interest_score >= self.cfg.replay.interest_threshold:
                return self._write_replay_to_disk(
                    match_id=self.current_match_id,
                    generation=self.current_generation,
                    interest_score=interest_score,
                    stats=self.stats,
                    frames=self.replay_frames,
                )
        return None

    def get_centralized_state(self) -> np.ndarray:
        """Compute the (12, 54) canonical centralized state matrix for the CTDE Critic."""
        central_state = np.zeros((self.num_players, 54), dtype=np.float32)

        c_p_a, c_v_a, _ = self._to_team_canonical(self.player_pos, self.player_vel, 0.0, TEAM_A)
        c_b_a, c_bv_a, _ = self._to_team_canonical(self.ball_pos, self.ball_vel, 0.0, TEAM_A)

        c_p_b, c_v_b, _ = self._to_team_canonical(self.player_pos, self.player_vel, 0.0, TEAM_B)
        c_b_b, c_bv_b, _ = self._to_team_canonical(self.ball_pos, self.ball_vel, 0.0, TEAM_B)

        # Team A (agents 0..5) canonical state:
        v_norm = float(self.cfg.vision.critic_vel_norm)
        tm_a = np.empty((6, 4), dtype=np.float32)
        tm_a[:, 0] = c_p_a[:6, 0] / 700.0 - 1.0
        tm_a[:, 1] = c_p_a[:6, 1] / 420.0 - 1.0
        tm_a[:, 2] = np.clip(c_v_a[:6, 0] / v_norm, -1.0, 1.0)
        tm_a[:, 3] = np.clip(c_v_a[:6, 1] / v_norm, -1.0, 1.0)

        opp_a = np.empty((6, 4), dtype=np.float32)
        opp_a[:, 0] = c_p_a[6:, 0] / 700.0 - 1.0
        opp_a[:, 1] = c_p_a[6:, 1] / 420.0 - 1.0
        opp_a[:, 2] = np.clip(c_v_a[6:, 0] / v_norm, -1.0, 1.0)
        opp_a[:, 3] = np.clip(c_v_a[6:, 1] / v_norm, -1.0, 1.0)

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
        tm_b[:, 2] = np.clip(c_v_b[6:, 0] / v_norm, -1.0, 1.0)
        tm_b[:, 3] = np.clip(c_v_b[6:, 1] / v_norm, -1.0, 1.0)

        opp_b = np.empty((6, 4), dtype=np.float32)
        opp_b[:, 0] = c_p_b[:6, 0] / 700.0 - 1.0
        opp_b[:, 1] = c_p_b[:6, 1] / 420.0 - 1.0
        opp_b[:, 2] = np.clip(c_v_b[:6, 0] / v_norm, -1.0, 1.0)
        opp_b[:, 3] = np.clip(c_v_b[:6, 1] / v_norm, -1.0, 1.0)

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

    def _build_agent_info(
        self,
        agent: str,
        central_state: Optional[np.ndarray] = None,
        interest_score: Optional[float] = None,
    ) -> Dict[str, Any]:
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
        if interest_score is None:
            is_done = (len(self.agents) == 0) or (self.rl_step_count >= self.cfg.physics.max_rl_steps)
            ep_interest = self.compute_interest_score() if is_done else 0.0
        else:
            ep_interest = float(interest_score)
        return {
            "team": team,
            "role": role_str,
            "policy_id": "keeper_policy" if role_int == ROLE_KEEPER else "field_policy",
            "team_side": int(self.team_side[team]),
            "rl_step": self.rl_step_count,
            "physics_frame": self.physics_frame_count,
            "match_id": self.current_match_id,
            "generation": self.current_generation,
            "interest_score": ep_interest,
            "final_stats": dict(self.stats),
            "last_saved_replay": self.last_saved_replay_path,
            "central_state": c_state,
        }

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
