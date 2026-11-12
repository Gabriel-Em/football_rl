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
from football_rl.potentials import (
    ROLE_GK,
    ROLE_LABELS,
    PotentialResult,
    PotentialState,
    compute_potentials,
)
from football_rl.recording import ReplayBuffer

TEAM_LEFT = SIDE_LEFT
TEAM_RIGHT = SIDE_RIGHT



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
        self._team_left_indices: slice = slice(0, 6)
        self._team_right_indices: slice = slice(6, 12)
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
        self._is_keeper: np.ndarray = self.player_role == ROLE_KEEPER
        self._is_keeper_list: List[bool] = self._is_keeper.tolist()
        self._max_speed_arr: np.ndarray = np.array(
            [self._get_player_max_speed(i) for i in range(len(roles))], dtype=np.float64
        )
        self._max_speed_list: List[float] = self._max_speed_arr.tolist()
        self._goal_centers: Tuple[np.ndarray, np.ndarray] = (
            np.array(self.cfg.physics.left_goal_center, dtype=np.float64),
            np.array(self.cfg.physics.right_goal_center, dtype=np.float64),
        )
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
        self._max_self_vel: np.ndarray = np.array(
            [
                self.cfg.mechanics.keeper_obs_speed_norm
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
        # Tactical Depth Bias (-1.0=DEF/GK, 0.0=MID, +1.0=ATK) & Lateral Flank Bias (-1.0=Left/Top, 0.0=Center, +1.0=Right/Bottom)
        self._tactical_depth_bias: np.ndarray = np.zeros(self.num_players, dtype=np.float32)
        self._tactical_flank_bias: np.ndarray = np.zeros(self.num_players, dtype=np.float32)
        for i, r_key in enumerate(self.player_role_key):
            if r_key == "keeper":
                self._tactical_depth_bias[i] = -1.0
                self._tactical_flank_bias[i] = 0.0
            elif r_key == "def_1":
                self._tactical_depth_bias[i] = -1.0
                self._tactical_flank_bias[i] = -1.0
            elif r_key == "def_2":
                self._tactical_depth_bias[i] = -1.0
                self._tactical_flank_bias[i] = +1.0
            elif r_key == "mid_1":
                self._tactical_depth_bias[i] = 0.0
                self._tactical_flank_bias[i] = -1.0
            elif r_key == "mid_2":
                self._tactical_depth_bias[i] = 0.0
                self._tactical_flank_bias[i] = +1.0
            else:  # atk_1
                self._tactical_depth_bias[i] = +1.0
                self._tactical_flank_bias[i] = 0.0

        # Precompute ego-first team-canonical index permutation for Centralized Critic (12, 6)
        self._critic_own_order: np.ndarray = np.zeros((self.num_players, 6), dtype=np.int32)
        self._critic_opp_order: np.ndarray = np.zeros((self.num_players, 6), dtype=np.int32)
        for i in range(self.num_players):
            t_start = 0 if i < 6 else 6
            o_start = 6 if i < 6 else 0
            own_others = [j for j in range(t_start, t_start + 6) if j != i]
            self._critic_own_order[i] = [i] + own_others
            self._critic_opp_order[i] = list(range(o_start, o_start + 6))

        self._critic_own_order_a: np.ndarray = self._critic_own_order[:6]
        self._critic_opp_order_a: np.ndarray = self._critic_opp_order[:6]
        self._critic_own_order_b: np.ndarray = self._critic_own_order[6:]
        self._critic_opp_order_b: np.ndarray = self._critic_opp_order[6:]

        # Precompute max diagonal distance for normalization
        self._max_field_dist: float = math.hypot(
            self.cfg.physics.field_width, self.cfg.physics.field_height
        )
        self.max_pitch_diag: float = float(self._max_field_dist)

        # Potential State, Anti-Loop Trackers & Steering Vectors
        self.raw_steer_input: np.ndarray = np.zeros(self.num_players, dtype=np.float32)
        self.prev_tactical_potential: np.ndarray = np.zeros(self.num_players, dtype=np.float64)
        self.last_turnover_reward_time: float = -999.0
        self.last_kick_time: np.ndarray = np.full(self.num_players, -999.0, dtype=np.float64)
        self.last_shot_sot_reward: float = 0.0
        self.last_shot_team_reward: float = 0.0
        self.last_pass_pair: List[Optional[Tuple[int, int]]] = [None, None]
        self.pair_pass_count: List[int] = [0, 0]
        self.spell_max_threat: np.ndarray = np.full(2, -1.0, dtype=np.float64)

        # Preallocated buffers and constants for high-performance observation and critic computation
        self._phi_curr: np.ndarray = np.zeros(self.num_players, dtype=np.float64)

        self._critic_feat_a: np.ndarray = np.empty((self.num_players, 4), dtype=np.float32)
        self._critic_feat_b: np.ndarray = np.empty((self.num_players, 4), dtype=np.float32)
        self._critic_ctx_a: np.ndarray = np.empty(6, dtype=np.float32)
        self._critic_ctx_b: np.ndarray = np.empty(6, dtype=np.float32)
        self._inv_critic_vel_norm: float = 1.0 / float(self.cfg.vision.critic_vel_norm)
        self._inv_ball_max_launch_speed: float = 1.0 / float(self.cfg.physics.ball_max_launch_speed)
        self._inv_field_half_w: float = 1.0 / (float(self.cfg.physics.field_width) * 0.5)
        self._inv_field_half_h: float = 1.0 / (float(self.cfg.physics.field_height) * 0.5)

        self._obs_sx: np.ndarray = np.empty(self.num_players, dtype=np.float64)
        self._obs_sy: np.ndarray = np.empty(self.num_players, dtype=np.float64)
        self._obs_svx: np.ndarray = np.empty(self.num_players, dtype=np.float64)
        self._obs_svy: np.ndarray = np.empty(self.num_players, dtype=np.float64)
        self._obs_s_ang: np.ndarray = np.empty(self.num_players, dtype=np.float64)

        self._ent_pos: np.ndarray = np.empty(
            (self.num_players, self.cfg.vision.num_entity_slots, 2), dtype=np.float64
        )
        self._ent_vel: np.ndarray = np.empty(
            (self.num_players, self.cfg.vision.num_entity_slots, 2), dtype=np.float64
        )

        self._inv_keeper_box_radius: float = 1.0 / float(self.cfg.physics.keeper_box_radius)
        self._inv_keeper_box_loose_max_time: float = 1.0 / float(self.cfg.mechanics.box_timer_obs_norm)
        self._inv_max_field_dist: float = 1.0 / self._max_field_dist
        self._inv_pi: float = 1.0 / math.pi
        self._fade_rad: float = math.radians(15.0)
        self._inv_fade_rad: float = 1.0 / self._fade_rad
        self._rad_90: float = math.radians(90.0)
        self._rad_105: float = float(self.cfg.vision.half_fov_rad)
        self._inv_vis_max_range: float = 1.0 / float(self.cfg.vision.max_range)
        self._inv_max_rel_vel: float = 1.0 / (
            float(self.cfg.physics.ball_max_launch_speed) + float(self.cfg.physics.player_max_speed)
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
        self.last_step_rewards: np.ndarray = np.zeros(self.num_players, dtype=np.float32)
        self.cum_player_rewards: np.ndarray = np.zeros(self.num_players, dtype=np.float32)
        self.ball_kick_was_shot_on_target: bool = False
        self.post_kick_pbrs_timer: np.ndarray = np.zeros(self.num_players, dtype=np.float64)

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
        self.ball_kick_start_pos: np.ndarray = np.zeros(2, dtype=np.float64)
        self.ball_kick_time: float = -999.0
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
        self._replay_buffer: Optional[ReplayBuffer] = None
        self._replay_frames_cache: Optional[List[Dict[str, Any]]] = None
        self._replay_frames_cache_version = -1
        self._replay_step_index = 0
        self._frame_events: List[Dict[str, Any]] = []
        self.last_saved_replay_path: Optional[str] = None
        self._deferred_replay: Optional[Dict[str, Any]] = None
        # When set, every completed match is kept (whatever its interest score) until exported.
        self.uniform_capture: bool = False
        self._uniform_replay: Optional[Dict[str, Any]] = None
        # Kicker index of the last kick whose receiver has not been determined yet (-1: none).
        self._pending_kick_idx: int = -1
        self.current_generation: int = 0
        self.current_match_id: int = 0

        # Telemetry & Critic recording buffers
        self.last_v_preds: Optional[np.ndarray] = None
        self.last_action_means: Optional[List[List[float]]] = None
        self.last_rew_breakdown: Optional[List[Dict[str, Any]]] = None
        self.last_role_states: List[str] = [
            "GK", "COVR", "COVR", "COVR", "COVR", "COVR",
            "GK", "COVR", "COVR", "COVR", "COVR", "COVR",
        ]
        self.last_role_targets: List[List[float]] = [[0.0, 0.0] for _ in range(self.num_players)]
        self.last_actions: List[List[float]] = [[0.0, 0.0, 0.0, 0.0] for _ in range(self.num_players)]

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
    def prev_potential(self) -> np.ndarray:
        return self.prev_tactical_potential

    @prev_potential.setter
    def prev_potential(self, val: np.ndarray) -> None:
        if isinstance(val, np.ndarray) and val.shape == self.prev_tactical_potential.shape:
            self.prev_tactical_potential[:] = val
        else:
            self.prev_tactical_potential = np.array(val, dtype=np.float64)

    @property
    def left_goal_center(self) -> Tuple[float, float]:
        return self.cfg.physics.left_goal_center

    @property
    def right_goal_center(self) -> Tuple[float, float]:
        return self.cfg.physics.right_goal_center

    @property
    def enable_replay_recording(self) -> bool:
        return bool(self.cfg.replay.enable_recording)

    @property
    def record_replay(self) -> bool:
        return bool(self.cfg.replay.enable_recording)

    @property
    def replay_frames(self) -> List[Dict[str, Any]]:
        """JSON-ready 60 Hz frames recorded since the last reset (empty when recording is disabled)."""
        buf = self._replay_buffer
        if buf is None:
            return []
        if self._replay_frames_cache is None or self._replay_frames_cache_version != buf.version:
            self._replay_frames_cache = buf.materialize()
            self._replay_frames_cache_version = buf.version
        return self._replay_frames_cache

    def _ensure_replay_buffer(self) -> ReplayBuffer:
        """Return the replay buffer, allocating it for a full match on first use."""
        if self._replay_buffer is None:
            self._replay_buffer = ReplayBuffer(
                max_frames=self.cfg.physics.max_physics_frames + 1, num_players=self.num_players
            )
        return self._replay_buffer

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
            "through_balls": 0,
            "wall_passes": 0,
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
            "keeper_5s_violations": 0,
            "possession_frames": [0, 0],
            "stationary_ball_frames": 0,
            "gk_holds": 0,
            "kicks": 0,
            "kick_to_teammate": 0,
            "kick_to_opponent": 0,
            "kick_other": 0,
        }
        self._pending_kick_idx = -1

        if self.cfg.replay.enable_recording:
            self._ensure_replay_buffer().reset()
        self._replay_step_index = 0
        self._frame_events = []
        self.last_v_preds = None
        self.last_action_means = None
        self.last_rew_breakdown = None
        self.last_role_states = [
            "GK", "COVR", "COVR", "COVR", "COVR", "COVR",
            "GK", "COVR", "COVR", "COVR", "COVR", "COVR",
        ]
        self.last_role_targets = [[0.0, 0.0] for _ in range(self.num_players)]
        self.last_actions = [[0.0, 0.0, 0.0, 0.0] for _ in range(self.num_players)]
        self.last_keeper_kick_time.fill(-999.0)
        self.shot_on_target_cd.fill(0.0)
        self.last_step_rewards.fill(0.0)
        self.cum_player_rewards.fill(0.0)
        self.ball_kick_was_shot_on_target = False
        self.ball_kick_flight_timer = 0.0
        self.ball_kick_time = -999.0
        self.ball_kick_start_pos = np.zeros(2, dtype=np.float64)
        self.post_kick_pbrs_timer = np.zeros(self.num_players, dtype=np.float64)
        self.pinned_ball_timer = 0.0
        self.kickoff_protection_timer = 0.0
        self.pickup_tackle_grace_timer = 0.0
        self.dive_dir.fill(0.0)
        self.dive_speed.fill(0.0)
        self.raw_steer_input.fill(0.0)
        self.prev_tactical_potential.fill(0.0)
        self._box_loose_ball_timer.fill(0.0)
        self.last_turnover_reward_time = -999.0
        self.last_kick_time.fill(-999.0)
        self.last_shot_sot_reward = 0.0
        self.last_shot_team_reward = 0.0
        self._reset_pass_chains()

        # Set initial kickoff formation and initialize Markovian potentials
        self._setup_kickoff(starting_team=self.first_half_kickoff_team)
        self._compute_potential_shaping(
            np.zeros(self.num_players, dtype=np.float64), any_teleport=True
        )

        # Record initial frame (t = 0.0s) if recording enabled
        if self.cfg.replay.enable_recording:
            self._ensure_replay_buffer().set_step_payload(0, self._replay_step_payload())
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

    def _step_core(
        self,
        act_array: np.ndarray,
        v_preds: Optional[np.ndarray] = None,
        action_means: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool, bool]:
        """Execute physics sub-steps, PBRS, and end-of-episode bookkeeping for a (12, 4) action array.

        `action_means` is the policy's mean action per player; it is only recorded in replay frames.
        """
        if action_means is not None and np.shape(action_means) != (self.num_players, 4):
            raise ValueError(
                f"action_means must have shape ({self.num_players}, 4), got {np.shape(action_means)}"
            )
        # Store raw steering input for instantaneous kick deflection
        self.raw_steer_input = act_array[:, 1].astype(np.float32)
        if self.cfg.replay.enable_recording:
            self._replay_step_index = self.rl_step_count + 1
            self.last_actions = [[round(float(a), 3) for a in act_array[p]] for p in range(self.num_players)]
            self.last_action_means = (
                None if action_means is None else np.round(np.asarray(action_means, dtype=np.float64), 3).tolist()
            )
        if v_preds is not None:
            self.last_v_preds = np.asarray(v_preds, dtype=np.float32)

        step_rewards = np.zeros(self.num_players, dtype=np.float64)
        phi_ball_start_a = self._compute_ball_potential(TEAM_A)
        phi_ball_start_b = self._compute_ball_potential(TEAM_B)
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

        r_sparse = step_rewards.copy()

        # Tier 2: Strict Policy-Invariant PBRS (Ball Threat & Off-Ball Positioning)
        phi_ball_end_a = self._compute_ball_potential(TEAM_A)
        phi_ball_end_b = self._compute_ball_potential(TEAM_B)
        r_ball_adv = np.zeros(self.num_players, dtype=np.float64)
        if not any_teleport:
            rew = self.cfg.rewards
            w_adv = rew.team_advance_pbrs_weight
            adv_a = w_adv * (phi_ball_end_a - phi_ball_start_a)
            adv_b = w_adv * (phi_ball_end_b - phi_ball_start_b)

            if abs(adv_a) > 1e-12:
                step_rewards[:6] += adv_a
                r_ball_adv[:6] = adv_a
            if abs(adv_b) > 1e-12:
                step_rewards[6:] += adv_b
                r_ball_adv[6:] = adv_b

        self._compute_potential_shaping(
            step_rewards, any_teleport, r_sparse=r_sparse, r_ball_adv=r_ball_adv
        )

        # Update reward tracking arrays for visual feedback & replay recording
        self.last_step_rewards = step_rewards.astype(np.float32)
        self.cum_player_rewards += self.last_step_rewards
        if self.cfg.replay.enable_recording and self._replay_buffer is not None:
            self._replay_buffer.set_step_payload(self._replay_step_index, self._replay_step_payload())

        # Episode termination / truncation at t = 180.0s (2,700 RL steps)
        is_truncated = self.rl_step_count >= self.cfg.physics.max_rl_steps
        is_terminated = False

        if is_truncated or is_terminated:
            self._resolve_pending_kick_as_other()
            self._emit_event(
                "FULLTIME!",
                np.array(self.cfg.physics.field_center, dtype=np.float64),
                team=-1,
                agent="referee",
            )
            if self.cfg.replay.enable_recording and self._replay_buffer is not None and len(self._replay_buffer) > 0:
                self._replay_buffer.append_event_to_last_frame(
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
        self,
        actions_array: np.ndarray,
        v_preds: Optional[np.ndarray] = None,
        action_means: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, Optional[Dict[str, Any]]]:
        """Execute one RL step using a (12, 4) action array and return contiguous NumPy arrays."""
        act_array = np.clip(np.asarray(actions_array, dtype=np.float64), -1.0, 1.0)
        step_rewards, is_terminated, is_truncated = self._step_core(
            act_array, v_preds=v_preds, action_means=action_means
        )

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

        ent_view = obs[:, vis.self_feature_dim :].reshape(
            self.num_players, vis.num_entity_slots, vis.entity_feature_dim
        )
        ent_view[:, 0, 0] = ball_vis
        ent_view[:, 1:, 0] = vis_p
        ent_view[:, :, 1] = np.clip(dist * self._inv_max_field_dist, 0.0, 1.0)
        ent_view[:, :, 2] = rel_ang * self._inv_pi
        ent_view[:, :, 3] = (ent_vel[:, :, 0] - svx[:, None]) * self._inv_max_rel_vel
        ent_view[:, :, 4] = (ent_vel[:, :, 1] - svy[:, None]) * self._inv_max_rel_vel

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
        r_sparse: Optional[np.ndarray] = None,
        r_ball_adv: Optional[np.ndarray] = None,
    ) -> None:
        """Evaluate the potential field and apply potential-based shaping: `step_rewards += Phi(s') - Phi(s)`."""
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

        if self.cfg.replay.enable_recording:
            self._record_potential_telemetry(state, result, any_teleport, r_sparse, r_ball_adv)

        if not any_teleport:
            step_rewards += phi_curr - self.prev_tactical_potential
        self.prev_tactical_potential[:] = phi_curr

    def _record_potential_telemetry(
        self,
        state: PotentialState,
        result: PotentialResult,
        any_teleport: bool,
        r_sparse: Optional[np.ndarray],
        r_ball_adv: Optional[np.ndarray],
    ) -> None:
        """Store per-player role labels, role targets and the reward breakdown for replay frames."""
        n = self.num_players
        stunned = state.stunned.tolist()
        lunging = (self.lunge_timer > 0.0).tolist()
        role_codes = result.role_code.tolist()
        role_states = [ROLE_LABELS[code] for code in role_codes]
        for i in range(n):
            if stunned[i]:
                role_states[i] = "STUN"
            elif lunging[i]:
                role_states[i] = "LNGE"

        pot_delta = (
            result.phi - self.prev_tactical_potential if not any_teleport else np.zeros_like(result.phi)
        ).tolist()
        r_adv = (r_ball_adv if r_ball_adv is not None else np.zeros(n)).tolist()
        r_sp = (r_sparse if r_sparse is not None else np.zeros(n)).tolist()
        phi_role = result.phi_role.tolist()
        crowd = result.crowd.tolist()
        self.last_rew_breakdown = [
            {
                "r_ball_adv": round(r_adv[i], 4),
                "r_pot_i": round(pot_delta[i], 4),
                "phi_role": round(phi_role[i], 4),
                "phi_crowd": round(0.0 if role_codes[i] == ROLE_GK else -crowd[i], 4),
                "r_sparse": round(r_sp[i], 4),
            }
            for i in range(n)
        ]
        self.last_role_states = role_states
        self.last_role_targets = [[round(x, 1), round(y, 1)] for x, y in result.target.tolist()]

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

    def _replay_step_payload(self) -> Dict[str, Any]:
        """Per-RL-step telemetry shared by every replay frame recorded during that step."""
        v_preds_data = [round(float(v), 3) for v in self.last_v_preds] if self.last_v_preds is not None else None
        return {
            "action_mean": self.last_action_means,
            "player_rewards": [round(float(r), 4) for r in self.last_step_rewards],
            "cum_rewards": [round(float(r), 2) for r in self.cum_player_rewards],
            "v_pred": v_preds_data,
            "rew_breakdown": self.last_rew_breakdown,
            "role_state": (list(self.last_role_states) if self.last_role_states is not None else None),
            "role_targets": self.last_role_targets,
            "actions": self.last_actions,
        }

    def _record_replay_frame(self) -> None:
        """Copy the current 60 Hz state snapshot into the replay buffer."""
        if not self.cfg.replay.enable_recording:
            return
        flags = self.is_ghost * 2.0 + (self.stun_timer > 0.0) * 4.0 + (self.lunge_timer > 0.0) * 8.0
        if self.ball_holder is not None:
            flags[self.ball_holder] += 1.0

        ball_team_color = (
            int(self.player_team[self.ball_holder])
            if self.ball_holder is not None
            else (int(self.last_ball_team) if self.last_ball_team is not None else -1)
        )

        stats = self.stats
        self._ensure_replay_buffer().record_frame(
            frame=self.physics_frame_count,
            time=self.physics_frame_count * self.cfg.physics.dt,
            half=self.current_half,
            team_side=self.team_side,
            box_radii=self.active_box_radius,
            player_pos=self.player_pos,
            player_angle=self.player_angle,
            player_flags=flags,
            player_vel=self.player_vel,
            ball_pos=self.ball_pos,
            ball_team=ball_team_color,
            ball_vel=self.ball_vel,
            step_index=self._replay_step_index,
            score=stats["score"],
            possession_frames=stats["possession_frames"],
            stationary=stats.get("stationary_ball_frames", 0),
            passes=stats["team_passes"],
            tackles=stats["team_tackles"],
            saves=stats["team_saves"],
            shots=stats["shots"],
            events=self._frame_events,
            inflight=self.ball_kicked_in_flight,
            post_kick=self.post_kick_pbrs_timer,
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
        tag: str = "",
    ) -> str:
        """Serialize and write a 60 Hz JSON replay trajectory to disk; `tag` is appended to the file name."""
        os.makedirs(self.cfg.replay.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        score_str = f"{interest_score:.1f}".replace(".", "_")
        tag_str = f"_{tag}" if tag else ""
        if match_id > 0:
            filename = (
                f"match_{match_id:04d}_gen{generation:04d}"
                f"_score{score_str}{tag_str}_{timestamp}.json"
            )
        else:
            filename = f"unindexed_{timestamp}_score{score_str}{tag_str}.json"
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
        if self.uniform_capture:
            self._uniform_replay = {
                "match_id": self.current_match_id,
                "generation": self.current_generation,
                "interest_score": interest_score,
                "stats": copy.deepcopy(self.stats),
                "frames": self._ensure_replay_buffer().snapshot(),
            }
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
                    "frames": self._ensure_replay_buffer().snapshot(),
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

    def export_uniform_replay(self) -> Optional[str]:
        """Write the most recent match captured while `uniform_capture` was set, once; `None` if there is none."""
        payload = self._uniform_replay
        if payload is None:
            return None
        self._uniform_replay = None
        previous = self.last_saved_replay_path
        path = self._write_replay_to_disk(
            match_id=int(payload["match_id"]),
            generation=int(payload["generation"]),
            interest_score=float(payload["interest_score"]),
            stats=payload["stats"],
            frames=payload["frames"].materialize(),
            tag="uniform",
        )
        self.last_saved_replay_path = previous
        return path

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
                frames=payload["frames"].materialize(),
            )
        if self._replay_buffer is not None and len(self._replay_buffer) > 0:
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


FootballEnv = FootballParallelEnv
