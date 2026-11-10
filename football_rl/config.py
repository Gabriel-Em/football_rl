"""Configuration, physical constants, reward weights, formations, and hyperparameters for 6v6 MARL Football."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Tuple


# ==============================================================================
# Agent & Team Identifiers
# ==============================================================================
TEAM_A: int = 0
TEAM_B: int = 1

SIDE_LEFT: int = 0   # Defends Left goal (0.0, 420.0), attacks Right goal (1400.0, 420.0)
SIDE_RIGHT: int = 1  # Defends Right goal (1400.0, 420.0), attacks Left goal (0.0, 420.0)
TEAM_LEFT: int = SIDE_LEFT
TEAM_RIGHT: int = SIDE_RIGHT

TEAM_A_AGENTS: Tuple[str, ...] = (
    "a_keeper",
    "a_def_1",
    "a_def_2",
    "a_mid_1",
    "a_mid_2",
    "a_atk_1",
)
TEAM_B_AGENTS: Tuple[str, ...] = (
    "b_keeper",
    "b_def_1",
    "b_def_2",
    "b_mid_1",
    "b_mid_2",
    "b_atk_1",
)
ALL_AGENTS: Tuple[str, ...] = TEAM_A_AGENTS + TEAM_B_AGENTS

ROLE_KEEPER: int = 0
ROLE_FIELD: int = 1  # Kept as alias/base for general field-player checks (role != ROLE_KEEPER)
ROLE_DEF: int = 1
ROLE_MID: int = 2
ROLE_ATK: int = 3
ROLE_DEFENDER: int = ROLE_DEF
ROLE_MIDFIELDER: int = ROLE_MID
ROLE_ATTACKER: int = ROLE_ATK


# ==============================================================================
# Dataclass Configurations
# ==============================================================================
@dataclass(frozen=True)
class PhysicsConfig:
    """Exact geometry, entity kinematics, and sub-stepping constants."""

    # Field Dimensions
    field_width: float = 1400.0
    field_height: float = 840.0
    field_center: Tuple[float, float] = (700.0, 420.0)

    # Goals (centered on X = 0.0 and X = 1400.0 walls)
    goal_y_min: float = 320.0
    goal_y_max: float = 520.0
    goal_width: float = 200.0
    left_goal_center: Tuple[float, float] = (0.0, 420.0)
    right_goal_center: Tuple[float, float] = (1400.0, 420.0)

    # Keeper Semi-Circles (The Box)
    keeper_box_radius: float = 195.0

    # Player Kinematics (Base defaults + Role-Specific Profiles)
    player_radius: float = 18.0
    player_max_speed: float = 185.0
    def_max_speed: float = 188.0
    mid_max_speed: float = 192.0
    atk_max_speed: float = 190.0
    keeper_max_speed: float = 180.0
    player_max_turn_rate: float = 1.5 * math.pi  # 270 deg/sec in rad/sec

    # Ball Kinematics
    ball_radius: float = 10.0
    ball_min_kick_speed: float = 160.0
    ball_max_launch_speed: float = 640.0  # Maximum launch speed
    def_max_kick_speed: float = 560.0
    mid_max_kick_speed: float = 600.0
    atk_max_kick_speed: float = 640.0
    keeper_max_kick_speed: float = 640.0
    def_shot_jitter_rad: float = 0.0
    mid_shot_jitter_rad: float = 0.0
    atk_shot_jitter_rad: float = 0.0
    ball_friction_decel: float = 155.0               # Linear friction deceleration across 1400x840 pitch
    ball_wall_restitution: float = 0.85

    # Tick Rate & Frame Skipping
    physics_hz: int = 60
    dt: float = 1.0 / 60.0
    rl_hz: int = 15
    frame_skip: int = 4
    max_rl_steps: int = 2700        # 180.0s * 15 Hz
    max_physics_frames: int = 10800  # 180.0s * 60 Hz
    half_physics_frames: int = 5400  # 90.0s * 60 Hz

    # Throttle Deadzone & Backpedal Multipliers
    throttle_deadzone: float = 0.10         # |throttle_raw| <= 0.10 -> speed = 0.0 (stand still)
    backpedal_threshold: float = -0.10      # throttle_raw in [-0.10, +0.10] -> speed = 0.0; backpedal if < -0.10
    field_backpedal_mult: float = 0.65      # throttle_raw = -1.0 -> -0.65 * role_max_speed
    keeper_backpedal_mult: float = 1.0      # throttle_raw = -1.0 -> -1.0 * keeper_max_speed (full drop-back speed)
    player_accel_rate: float = 10.5         # Exponential velocity smoothing rate (1/s) for fluid accel/decel


@dataclass(frozen=True)
class MechanicsConfig:
    """Player state machines, possession timers, tackle and lunge parameters."""

    # Ball Pickup & Carrying
    pickup_distance: float = 28.0     # player_radius (18.0) + ball_radius (10.0)
    ball_carry_offset: float = 24.0   # player_pos + facing_vec * 24.0
    kick_pickup_cooldown: float = 0.45  # Anti-self-dribble cooldown; cleared on other touch
    pickup_kick_grace: float = 0.08   # Minimum hold duration (s) after picking up ball before kick can trigger
    pickup_tackle_grace: float = 0.15 # Shielding duration (s) after recovering a free loose ball during which carrier cannot be tackled
    pass_reception_tackle_grace: float = 0.10 # Shielding duration (s) after receiving a teammate pass outside goalmouth
    kick_steer_max_deflection_deg: float = 35.0  # Side-foot pass/clearance angle modulation (+/- 35 deg)
    contested_ball_distance: float = 65.0  # Max opponent distance (u) at recovery for a loose ball pickup to be classified as contested
    kick_trigger_threshold: float = 0.35  # action[2] must exceed 0.35 to intentionally kick
    min_pass_distance: float = 85.0   # Minimum kick travel distance to count as a Completed Pass (blocks micro-pass scrums)
    min_shot_speed: float = 240.0     # Minimum ball launch speed (u/s) required to qualify as a Shot on Target
    turnover_reward_cooldown: float = 1.0  # Minimum interval (s) between rewarded turnover events

    # Special Action Trigger (action[3])
    special_trigger_threshold: float = 0.5

    # Kinematic Field Lunges & Role-Specific Tackle Parameters
    field_lunge_speed: float = 320.0       # Units per second impulse during a field tackle
    field_lunge_duration: float = 0.25     # Seconds the lunge impulse lasts
    tackle_miss_stun_time: float = 0.8     # Seconds tackler is frozen/stunned if they miss or fail
    tackle_victim_stun_time: float = 1.0   # Seconds victim is frozen after losing the ball

    # Expanded Max Tackle Radii (Base Reach + Lunge)
    tackle_radius_def: float = 85.0
    tackle_radius_mid: float = 75.0
    tackle_radius_atk: float = 65.0

    # Base Probabilities (at point-blank range)
    tackle_prob_def: float = 0.90
    tackle_prob_mid: float = 0.82
    tackle_prob_atk: float = 0.74

    tackled_opponent_cooldown: float = 1.0
    tackle_fail_cooldown: float = 0.8  # Synced to tackle_miss_stun_time

    # Keeper Lunge & Perimeter Strip
    keeper_obs_speed_norm: float = 430.0  # observation normalisation for keeper velocity
    keeper_dive_speed: float = 340.0
    lunge_duration: float = 0.25
    lunge_cooldown: float = 1.8
    keeper_lunge_cooldown: float = 1.8
    keeper_strip_stun_duration: float = 0.20
    keeper_strip_self_kick_cooldown: float = 2.0  # 0.0 reward and no save stat if keeper kicked in last 2.0s

    # Hold Rules & Violation Penalties
    field_max_hold_time: float = 5.0
    ghost_duration: float = 3.0
    field_violation_pickup_cooldown: float = 3.0
    field_violation_tackle_cooldown: float = 3.0
    keeper_max_hold_time: float = 5.0
    box_timer_obs_norm: float = 5.0         # observation normalisation for the loose-box-ball timer
    keeper_foul_pickup_cooldown: float = 2.0  # Pickup lockout (s) applied to a keeper after a 5s hold foul
    post_save_box_grace_time: float = 2.0   # Grace period (s) after GK save before box loose timer accumulates

    # Dynamic Semicircle Shrinking Parameters (Anti-Stall on Unclaimed Loose Box Balls)
    box_shrink_grace_time: float = 2.0      # seconds an unclaimed loose ball must be completely stopped inside the box before semicircle starts shrinking
    box_ball_stopped_threshold: float = 15.0 # max ball speed (u/s) for ball to be considered completely stopped
    box_shrink_rate: float = 55.0           # units/sec radial shrink speed allowing outfielders closer and closer
    box_expand_rate: float = 220.0          # units/sec smooth radial re-expansion speed once ball is claimed or exits box
    box_min_radius: float = 0.0             # minimum radius the semicircle can shrink to (0.0 = completely open)

    # Pass Flight Timeout & Pinned Ball Unsticking
    max_pass_flight_time: float = 3.0       # Max time (s) after a kick for a direct pickup to count as a pass/interception
    max_wall_pass_flight_time: float = 3.5  # Max time (s) after a kick for a side-wall bank pass to be received
    min_wall_pass_distance: float = 85.0    # Min displacement (u) between kick origin and receiver for wall pass
    pinned_ball_max_time: float = 2.0       # Time (s) a slow loose ball near a wall/box can stay jammed before nudge
    pinned_ball_speed_thresh: float = 18.0  # Max ball speed (u/s) considered pinned/stuck
    pinned_ball_nudge_speed: float = 95.0   # Inward impulse speed (u/s) applied to unstick a pinned ball

    # Kickoff Protection & GK Possession Buffer
    kickoff_protection_time: float = 1.2    # Time (s) after kickoff during which carrier cannot be tackled
    kickoff_exclusion_radius: float = 130.0 # Radial buffer keeping opponents outside center spot at kickoff
    gk_hold_opponent_buffer: float = 265.0  # Min distance opposing field players must keep from goal center while GK holds ball
    gk_hold_push_speed: float = 185.0       # Smooth radial pushback speed (u/s) capped at player max speed (no wave surfing)
    tackle_whiff_max_ball_dist: float = 180.0  # Max distance to ball within which an out-of-range tackle triggers a physical whiff stun


@dataclass(frozen=True)
class VisionConfig:
    """Observation space & vision cone parameters."""

    fov_deg: float = 210.0
    half_fov_rad: float = math.radians(105.0)  # +/- 105 deg (210 deg total FOV)
    max_range: float = 650.0
    peripheral_radar_presence: float = 0.25  # Non-zero presence flag for entities outside the 650u / 210-deg cone

    self_feature_dim: int = 16  # indices 0..12 + indices 13..15 one-hot [is_def, is_mid, is_atk]
    num_entity_slots: int = 12  # Slot 0: Ball, Slots 1..5: Teammates, Slots 6..10: Field Opps, Slot 11: Enemy Keeper
    entity_feature_dim: int = 5
    obs_dim: int = 76          # 16 + (12 * 5) = 76 floats
    action_dim: int = 4
    critic_vel_norm: float = 340.0  # Velocity normalization denominator for centralized critic state


@dataclass(frozen=True)
class RewardConfig:
    """Markovian 3-tier zero-sum adversarial and team-cooperative reward weights."""

    # Tier 1: Objective Discrete Game Events (Strictly Zero-Sum Adversarial Transitions)
    goal_team_shared: float = 10.0
    goal_concede_team_shared: float = -10.0
    goal_scorer_ind: float = 3.0
    goal_assist_ind: float = 2.5
    goal_assist_window_sec: float = 4.0
    own_goal_ind: float = -3.0

    # Shot on Target & Save (Zero-Sum Paired Accounting)
    shot_on_target_ind: float = 0.45
    shot_on_target_team: float = 0.15
    keeper_save_ind: float = 1.20
    keeper_save_team: float = 0.30

    # Passing & Progression (Team-Cooperative + Anti-Oscillation)
    pass_threat_coeff: float = 6.0
    pass_press_escape_coeff: float = 0.60
    through_ball_stride_bonus: float = 0.80
    completed_pass_receiver_ind: float = 0.15
    completed_pass_team: float = 0.20

    # Strictly Zero-Sum Turnover Events (Win Ind + 6*Win Team + Loss Ind + 6*Loss Team == 0.0)
    successful_tackle_tackler_ind: float = 0.40
    successful_tackle_team: float = 0.25
    successful_tackle_carrier_ind: float = -0.40
    intercepted_kicker_ind: float = -0.40
    intercepted_team: float = -0.25

    # Tier 2: Policy-Invariant PBRS Weights (each Phi is bounded above by its weight; the crowd and line-order penalties can make Phi negative)
    team_advance_pbrs_weight: float = 6.0
    support_potential_weight: float = 0.85
    def_press_potential_weight: float = 0.65
    def_cover_potential_weight: float = 0.75
    keeper_bisect_potential_weight: float = 0.60
    teammate_crowd_weight: float = 0.35
    teammate_crowd_radius: float = 110.0
    line_order_weight: float = 0.6  # Phi penalty scale for a covering defender standing ahead of its own striker
    line_order_sigma: float = 120.0  # distance (u) over which the line-order penalty saturates (tanh)
    loose_ball_secondary_press_blend: float = 0.75
    press_trail_weight: float = 1.0  # presser cost = ball distance + this * how far the player stands upfield of the ball

    # Tier 3: Physical Regularization
    field_5s_violation_ind: float = -1.5
    field_5s_violation_team: float = -0.3
    keeper_5s_violation_ind: float = -1.5
    failed_tackle_ind: float = -0.20

    def __post_init__(self) -> None:
        if not (math.isfinite(self.line_order_weight) and self.line_order_weight >= 0.0):
            raise ValueError(f"line_order_weight must be finite and >= 0, got {self.line_order_weight}")
        if not (math.isfinite(self.line_order_sigma) and self.line_order_sigma > 0.0):
            raise ValueError(f"line_order_sigma must be finite and > 0, got {self.line_order_sigma}")


@dataclass(frozen=True)
class FormationConfig:
    """Canonical set-piece formations assuming the reference team defends Left (X = 0.0).

    Roles order: ("keeper", "def_1", "def_2", "mid_1", "mid_2", "atk_1")
    When a team defends the Right goal (X = 1400.0), coordinates are rotated 180 degrees
    around the pitch center (700.0, 420.0): (1400.0 - x, 840.0 - y), facing_angle + pi.
    """

    # 1. Kickoff / Post-Goal Reset (Starting team on Left vs Defending team on Right)
    kickoff_starting_left: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "keeper": (45.0, 420.0),
            "def_1": (320.0, 345.0),
            "def_2": (320.0, 495.0),
            "mid_1": (520.0, 260.0),
            "mid_2": (520.0, 580.0),
            "atk_1": (700.0, 420.0),  # Ball Carrier at center spot
        }
    )
    kickoff_receiving_right: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "keeper": (1355.0, 420.0),
            "def_1": (1120.0, 495.0),
            "def_2": (1120.0, 345.0),
            "mid_1": (960.0, 220.0),
            "mid_2": (960.0, 620.0),
            "atk_1": (910.0, 420.0),  # Outside center circle (210u from center spot)
        }
    )


@dataclass(frozen=True)
class ReplayConfig:
    """Episode evaluation & JSON trajectory export configuration."""

    save_dir: str = "replays"
    enable_recording: bool = True
    interest_threshold: float = 15.0
    save_best_only: bool = False

    # Interest Score Formula Weights
    goal_weight: float = 5.0
    completed_pass_weight: float = 2.0
    successful_tackle_weight: float = 1.5
    keeper_save_weight: float = 3.0

    # Broadcast & Visualizer Overlays
    show_telemetry: bool = True
    show_pass_lanes: bool = False
    show_wall_tracers: bool = True
    show_radar: bool = False
    auto_focus_carrier: bool = False


@dataclass(frozen=True)
class PPOHyperparams:
    """Reference PPO with Parameter Sharing hyperparameters (`field_policy` & `keeper_policy`)."""

    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_steps: int = 2700  # 1 full match per parallel env
    num_minibatches: int = 8
    update_epochs: int = 4
    hidden_dims: Tuple[int, int] = (256, 256)


@dataclass(frozen=True)
class EnvConfig:
    """Master configuration container for `FootballParallelEnv`."""

    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    mechanics: MechanicsConfig = field(default_factory=MechanicsConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    formations: FormationConfig = field(default_factory=FormationConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    ppo: PPOHyperparams = field(default_factory=PPOHyperparams)


DEFAULT_CONFIG = EnvConfig()

# Module-level convenience constants matching DEFAULT_CONFIG
FIELD_WIDTH: float = DEFAULT_CONFIG.physics.field_width
FIELD_HEIGHT: float = DEFAULT_CONFIG.physics.field_height
GOAL_Y_MIN: float = DEFAULT_CONFIG.physics.goal_y_min
GOAL_Y_MAX: float = DEFAULT_CONFIG.physics.goal_y_max
KEEPER_BOX_RADIUS: float = DEFAULT_CONFIG.physics.keeper_box_radius
PLAYER_RADIUS: float = DEFAULT_CONFIG.physics.player_radius
BALL_RADIUS: float = DEFAULT_CONFIG.physics.ball_radius
PHYSICS_DT: float = DEFAULT_CONFIG.physics.dt
FRAME_SKIP: int = DEFAULT_CONFIG.physics.frame_skip
MAX_RL_STEPS: int = DEFAULT_CONFIG.physics.max_rl_steps
OBS_DIM: int = DEFAULT_CONFIG.vision.obs_dim
ACTION_DIM: int = DEFAULT_CONFIG.vision.action_dim
CENTRAL_STATE_DIM: int = 54
CRITIC_STATE_DIM: int = OBS_DIM + CENTRAL_STATE_DIM  # 130 dims (76 + 54)


if __name__ == "__main__":
    print("Loaded 6v6 MARL Football Configuration:")
    print(f"  Field Dimensions : {FIELD_WIDTH} x {FIELD_HEIGHT}")
    print(f"  Physics / RL Rate: {DEFAULT_CONFIG.physics.physics_hz} Hz / {DEFAULT_CONFIG.physics.rl_hz} Hz (frame_skip={FRAME_SKIP})")
    print(f"  Episode Steps    : {MAX_RL_STEPS} RL steps ({DEFAULT_CONFIG.physics.max_physics_frames} physics frames)")
    print(f"  Observation Space: shape=({OBS_DIM},), Action Space: shape=({ACTION_DIM},)")
