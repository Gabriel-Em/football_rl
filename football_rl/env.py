"""PettingZoo ParallelEnv implementation for 6v6 Continuous 2D MARL Football."""

from __future__ import annotations

import functools
import math
from typing import Any, Dict, List, Optional, Tuple

from gymnasium import spaces
import numpy as np
from pettingzoo import ParallelEnv

from football_rl.ball_play import BallPlayMixin
from football_rl.baselines import scripted_actions
from football_rl.config import (
    ALL_AGENTS,
    DEFAULT_CONFIG,
    EnvConfig,
    ROLE_ATK,
    ROLE_DEF,
    ROLE_KEEPER,
    ROLE_MID,
    SIDE_LEFT,
    SIDE_RIGHT,
    TEAM_A,
    TEAM_A_AGENTS,
    TEAM_B,
)
from football_rl.match_rules import _DualBoxTimer, MatchRulesMixin
from football_rl.observations import ObservationMixin
from football_rl.physics import PhysicsMixin
from football_rl.potentials import _POST_KICK_WINDOW, pbrs_shaping
from football_rl.recording import ReplayBuffer, format_breakdown, round_rows, _REPLAY_ROLE_LABELS, ReplayRecordingMixin
from football_rl.reward_shaping import RewardShapingMixin

_KICKOFF_ROLE_CODES = (0, 5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 5)  # GK + five COVR per team


def parallel_env(config: Optional[EnvConfig] = None, render_mode: Optional[str] = None) -> "FootballParallelEnv":
    """Factory function for creating the PettingZoo ParallelEnv."""
    return FootballParallelEnv(config=config, render_mode=render_mode)


env = parallel_env


class FootballParallelEnv(
    PhysicsMixin,
    BallPlayMixin,
    MatchRulesMixin,
    ObservationMixin,
    RewardShapingMixin,
    ReplayRecordingMixin,
    ParallelEnv,
):
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
        vis_cfg = self.cfg.vision
        self._match_state_col: int = (
            vis_cfg.self_feature_dim + vis_cfg.num_entity_slots * vis_cfg.entity_feature_dim
        )
        self._inv_max_rl_steps: float = 1.0 / float(self.cfg.physics.max_rl_steps)
        self._inv_post_kick_window: float = 1.0 / float(_POST_KICK_WINDOW)
        self._turnover_cooldown: float = float(
            getattr(self.cfg.mechanics, "turnover_reward_cooldown", 1.0)
        )
        self._inv_turnover_cooldown: float = 1.0 / max(1e-6, self._turnover_cooldown)
        # possession_owner column per owning team: +1/-1 for holder or in-flight kick, +0.5/-0.5 for loose ball
        self._possession_owner_rows: np.ndarray = np.array(
            [[1.0] * 6 + [-1.0] * 6, [-1.0] * 6 + [1.0] * 6], dtype=np.float32
        )
        self._loose_owner_rows: np.ndarray = 0.5 * self._possession_owner_rows

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
        self.last_shaping_rewards: np.ndarray = np.zeros(self.num_players, dtype=np.float64)
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
        # When cleared, non-uniform replay export and deferral are skipped without disabling recording.
        self.export_replays: bool = True
        # Team index (0 = A, 1 = B) whose six action rows `step_array` replaces with `scripted_actions`; None: none.
        self.scripted_team: Optional[int] = None
        # When set, every completed match is kept (whatever its interest score) until exported.
        self.uniform_capture: bool = False
        self._uniform_replay: Optional[Dict[str, Any]] = None
        # Kicker index of the last kick whose receiver has not been determined yet (-1: none).
        self._pending_kick_idx: int = -1
        self.current_generation: int = 0
        self.current_match_id: int = 0

        # Telemetry & Critic recording buffers
        self.last_v_preds: Optional[np.ndarray] = None
        self._reset_step_telemetry()

    def _reset_step_telemetry(self) -> None:
        """Raw per-step replay telemetry at its initial (kickoff) values."""
        n = self.num_players
        self._tele_actions = np.zeros((n, 4), dtype=np.float64)
        self._tele_action_means: Optional[np.ndarray] = None
        self._tele_breakdown: Optional[np.ndarray] = None
        self._tele_role_codes = np.array(_KICKOFF_ROLE_CODES, dtype=np.int64)
        self._tele_targets = np.zeros((n, 2), dtype=np.float64)

    @property
    def last_actions(self) -> List[List[float]]:
        """Clipped actions of the last recorded step, per player, rounded to 3 decimals."""
        return round_rows(self._tele_actions, 3)

    @property
    def last_action_means(self) -> Optional[List[List[float]]]:
        """Policy mean actions given with the last recorded step (3 decimals), or None."""
        return None if self._tele_action_means is None else np.round(self._tele_action_means, 3).tolist()

    @property
    def last_rew_breakdown(self) -> Optional[List[Dict[str, Any]]]:
        """Per-player reward components of the last recorded step (4 decimals), or None before any."""
        return None if self._tele_breakdown is None else format_breakdown(self._tele_breakdown)

    @property
    def last_role_states(self) -> List[str]:
        return [_REPLAY_ROLE_LABELS[c] for c in self._tele_role_codes.tolist()]

    @property
    def last_role_targets(self) -> List[List[float]]:
        return round_rows(self._tele_targets, 1)

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
        self._reset_step_telemetry()
        self.last_keeper_kick_time.fill(-999.0)
        self.shot_on_target_cd.fill(0.0)
        self.last_step_rewards.fill(0.0)
        self.last_shaping_rewards.fill(0.0)
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
            np.zeros(self.num_players, dtype=np.float64), pay=False
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
            self._tele_actions = np.array(act_array, dtype=np.float64)
            self._tele_action_means = None if action_means is None else np.array(action_means, dtype=np.float64)
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

        # Episode termination / truncation at t = 180.0s (max_rl_steps)
        is_truncated = self.rl_step_count >= self.cfg.physics.max_rl_steps
        is_terminated = False
        is_terminal = is_truncated or is_terminated

        r_sparse = step_rewards.copy()

        # Tier 2: Policy-Invariant PBRS (Ball Threat & Off-Ball Positioning)
        phi_ball_end_a = self._compute_ball_potential(TEAM_A)
        phi_ball_end_b = self._compute_ball_potential(TEAM_B)
        rew = self.cfg.rewards
        shaping_mode = rew.shaping_mode
        if shaping_mode == "legacy":
            r_ball_adv = np.zeros(self.num_players, dtype=np.float64)
            if not any_teleport:
                w_adv = rew.team_advance_pbrs_weight
                adv_a = w_adv * (phi_ball_end_a - phi_ball_start_a)
                adv_b = w_adv * (phi_ball_end_b - phi_ball_start_b)

                if abs(adv_a) > 1e-12:
                    step_rewards[:6] += adv_a
                    r_ball_adv[:6] = adv_a
                if abs(adv_b) > 1e-12:
                    step_rewards[6:] += adv_b
                    r_ball_adv[6:] = adv_b
        else:
            gamma = self.cfg.ppo.gamma
            w_adv = rew.team_advance_pbrs_weight
            adv_a = w_adv * pbrs_shaping(phi_ball_start_a, phi_ball_end_a, gamma, is_terminal)
            adv_b = w_adv * pbrs_shaping(phi_ball_start_b, phi_ball_end_b, gamma, is_terminal)
            step_rewards[:6] += adv_a
            step_rewards[6:] += adv_b
            r_ball_adv = np.empty(self.num_players, dtype=np.float64)
            r_ball_adv[:6] = adv_a
            r_ball_adv[6:] = adv_b

        r_role = self._compute_potential_shaping(
            step_rewards,
            any_teleport,
            r_sparse=r_sparse,
            r_ball_adv=r_ball_adv,
            pay=True,
            terminal=is_terminal,
        )
        np.add(r_ball_adv, r_role, out=self.last_shaping_rewards)

        # Update reward tracking arrays for visual feedback & replay recording
        self.last_step_rewards = step_rewards.astype(np.float32)
        self.cum_player_rewards += self.last_step_rewards
        if self.cfg.replay.enable_recording and self._replay_buffer is not None:
            self._replay_buffer.set_step_payload(self._replay_step_index, self._replay_step_payload())

        if is_terminal:
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
        """Execute one RL step using a (12, 4) action array and return contiguous NumPy arrays.

        The rows of `scripted_team`, when set, are replaced by `scripted_actions` of the state the step starts from;
        the caller's array is left as it is.
        """
        if self.scripted_team is not None:
            actions_array = np.array(actions_array, dtype=np.float64)
            first = 6 * self.scripted_team
            actions_array[first:first + 6] = scripted_actions(self, self.scripted_team)
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

    @property
    def match_time(self) -> float:
        """Current elapsed match time in seconds."""
        return float(self.physics_frame_count * self.cfg.physics.dt)

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
