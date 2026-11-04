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
    def_max_speed: float = 182.0
    mid_max_speed: float = 195.0
    atk_max_speed: float = 190.0
    keeper_max_speed: float = 180.0
    player_max_turn_rate_deg: float = 270.0
    player_max_turn_rate: float = 1.5 * math.pi  # 270 deg/sec in rad/sec

    # Ball Kinematics (Higher Shot/Pass Strength Limits & Lower Friction)
    ball_radius: float = 10.0
    ball_min_kick_speed: float = 260.0
    ball_max_launch_speed: float = 640.0  # Attacker max launch speed
    ball_kick_speed_span: float = 380.0   # 260.0 + power_norm * 380.0 -> 640.0
    def_max_kick_speed: float = 510.0
    mid_max_kick_speed: float = 560.0
    atk_max_kick_speed: float = 640.0
    keeper_max_kick_speed: float = 560.0
    def_shot_jitter_rad: float = math.radians(7.5)   # applied only when power_norm > 0.65
    mid_shot_jitter_rad: float = math.radians(3.8)   # applied only when power_norm > 0.65
    atk_shot_jitter_rad: float = 0.0                 # pinpoint laser accuracy at all power levels
    ball_friction_decel: float = 155.0               # passes/shots carry across 1400x840 pitch
    ball_wall_restitution: float = 0.85

    # Tick Rate & Frame Skipping
    match_duration: float = 180.0
    half_duration: float = 90.0
    physics_hz: int = 60
    dt: float = 1.0 / 60.0
    rl_hz: int = 15
    frame_skip: int = 4
    max_rl_steps: int = 2700        # 180.0s * 15 Hz
    half_rl_steps: int = 1350       # 90.0s * 15 Hz
    max_physics_frames: int = 10800  # 180.0s * 60 Hz
    half_physics_frames: int = 5400  # 90.0s * 60 Hz

    # Throttle Deadzone & Backpedal Multipliers
    throttle_deadzone: float = 0.15         # |throttle_raw| <= 0.15 -> speed = 0.0 (stand still)
    backpedal_threshold: float = -0.35      # throttle_raw in [-0.35, +0.15] -> speed = 0.0; backpedal if < -0.35
    field_backpedal_mult: float = 0.50      # throttle_raw = -1.0 -> -0.50 * role_max_speed
    keeper_backpedal_mult: float = 0.75     # throttle_raw = -1.0 -> -0.75 * keeper_max_speed
    player_accel_rate: float = 10.5         # Exponential velocity smoothing rate (1/s) for fluid accel/decel


@dataclass(frozen=True)
class MechanicsConfig:
    """Player state machines, possession timers, tackle and lunge parameters."""

    # Ball Pickup & Carrying
    pickup_distance: float = 28.0     # player_radius (18.0) + ball_radius (10.0)
    ball_carry_offset: float = 24.0   # player_pos + facing_vec * 24.0
    kick_pickup_cooldown: float = 1.0  # Anti-self-dribble cooldown; cleared on other touch
    pickup_kick_grace: float = 0.25   # Minimum hold duration (s) after picking up ball before kick can trigger
    kick_trigger_threshold: float = 0.35  # action[2] must exceed 0.35 to intentionally kick
    min_pass_distance: float = 75.0   # Minimum kick travel distance to count as a Completed Pass

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

    # Backward-compatible aliases and derived limits
    tackle_range: float = 75.0         # base / Midfielder
    def_tackle_range: float = 85.0
    mid_tackle_range: float = 75.0
    atk_tackle_range: float = 65.0
    tackle_near_range: float = 127.5   # Max aiming range (1.5 * tackle_radius_def = 127.5)
    tackle_success_prob: float = 0.82  # base / Midfielder
    def_tackle_success_prob: float = 0.90
    mid_tackle_success_prob: float = 0.82
    atk_tackle_success_prob: float = 0.74
    tackle_stun_duration: float = 1.0  # Synced to tackle_victim_stun_time
    tackled_opponent_cooldown: float = 1.0
    tackle_fail_cooldown: float = 0.8  # Synced to tackle_miss_stun_time

    # Keeper Lunge & Perimeter Strip
    lunge_speed: float = 430.0
    keeper_dive_speed: float = 340.0
    lunge_duration: float = 0.25
    lunge_cooldown: float = 1.8
    keeper_strip_stun_duration: float = 0.20
    keeper_strip_self_kick_cooldown: float = 2.0  # 0.0 reward and no save stat if keeper kicked in last 2.0s

    # Hold Rules & Violation Penalties
    field_max_hold_time: float = 5.0
    ghost_duration: float = 3.0
    field_violation_pickup_cooldown: float = 3.0
    field_violation_tackle_cooldown: float = 3.0
    keeper_max_hold_time: float = 5.0
    keeper_box_loose_max_time: float = 3.0  # Max time a loose ball can sit in keeper semicircle before foul teleport

    # Pass Flight Timeout & Pinned Ball Unsticking
    max_pass_flight_time: float = 3.0       # Max time (s) after a kick for a pickup to count as a pass/interception
    pinned_ball_max_time: float = 2.0       # Time (s) a slow loose ball near a wall/box can stay jammed before nudge
    pinned_ball_speed_thresh: float = 18.0  # Max ball speed (u/s) considered pinned/stuck
    pinned_ball_nudge_speed: float = 95.0   # Inward impulse speed (u/s) applied to unstick a pinned ball

    # Kickoff Protection & GK Possession Buffer
    kickoff_protection_time: float = 1.2    # Time (s) after kickoff during which carrier cannot be tackled
    kickoff_exclusion_radius: float = 130.0 # Radial buffer keeping opponents outside center spot at kickoff
    gk_hold_opponent_buffer: float = 265.0  # Min distance opposing field players must keep from goal center while GK holds ball (keeper_box_radius + 70.0)
    gk_hold_push_speed: float = 285.0       # Smooth radial pushback speed (u/s) when GK holds ball (no instant teleport)
    tackle_whiff_max_ball_dist: float = 180.0  # Max distance to ball within which an out-of-range tackle triggers a physical whiff stun


@dataclass(frozen=True)
class VisionConfig:
    """Observation space & vision cone parameters."""

    fov_deg: float = 210.0
    fov_rad: float = math.radians(210.0)
    half_fov_rad: float = math.radians(105.0)  # +/- 105 deg (210 deg total FOV)
    max_range: float = 650.0

    self_feature_dim: int = 16  # indices 0..12 + indices 13..15 one-hot [is_def, is_mid, is_atk]
    num_entity_slots: int = 12  # Slot 0: Ball, Slots 1..5: Teammates, Slots 6..10: Field Opps, Slot 11: Enemy Keeper
    entity_feature_dim: int = 5
    obs_dim: int = 76          # 16 + (12 * 5) = 76 floats
    action_dim: int = 4


@dataclass(frozen=True)
class RewardConfig:
    """Individual and team-shared reward weights."""

    # Goal & Shot-on-Target Rewards
    goal_scorer_ind: float = 5.0
    atk_goal_scorer_ind: float = 7.5
    goal_team_shared: float = 10.0
    goal_concede_team_shared: float = -10.0
    own_goal_ind: float = -8.0  # Additional individual penalty on top of -10.0 team penalty (-18.0 total)
    atk_shot_on_target_ind: float = 2.0        # Increased from 0.7: strong incentive for ATK to shoot on target
    other_shot_on_target_ind: float = 1.2      # Increased from 0.2: strong incentive for MID/DEF to shoot on target
    shot_on_target_team: float = 0.3           # Shared team bonus when firing a valid shot on target
    prime_shot_hesitation_pen: float = -0.025  # Per-RL-step penalty when aimed at goal within 320u and not shooting

    # Keeper Save, Off-Target Catch, & Strip
    keeper_save_ind: float = 5.0
    keeper_save_team: float = 1.0
    keeper_harmless_catch_ind: float = 0.5
    keeper_harmless_catch_team: float = 0.1
    keeper_strip_ind: float = 0.6
    keeper_strip_team: float = 0.15
    keeper_clear_forward_ind: float = 0.4   # Bonus when keeper kicks ball forward out of own box
    keeper_kick_backward_pen: float = -1.5  # Penalty when keeper kicks ball toward own goal wall

    # Pass & Role Link-Up Bonuses (Anti-Farming & Progressive)
    completed_pass_passer_ind: float = 0.0      # Zeroed out (was 1.0)
    completed_pass_receiver_ind: float = 0.2    # Modest receive touch bonus (was 0.8)
    completed_pass_team: float = 0.0            # Zeroed out (was 0.25)
    def_breakout_pass_bonus: float = 0.5
    mid_link_receive_bonus: float = 0.5
    mid_to_atk_pass_bonus: float = 0.8
    pass_to_atk_bonus: float = 0.6
    pass_target_aim_ind: float = 0.45       # Immediate bonus in _execute_kick when aiming toward open teammate
    recycle_pass_bonus: float = 0.0         # Completely removed (was 0.35)
    open_clearance_ind: float = 0.25        # Immediate bonus in _execute_kick when kicking to open pitch corridor
    blocked_kick_pen: float = -0.45         # Immediate penalty in _execute_kick when kicking into near opponent

    # Progressive Pass & Anti-Exploit Parameters
    pass_progressive_dx_weight: float = 0.006   # Bonus per unit of canonical X or goal-dist gained (e.g., +200u dx = +1.2)
    pass_line_break_bonus: float = 0.40        # Bonus per bypassed defender X-coordinate
    box_angle_pass_bonus: float = 0.85         # Bonus for final-third pass shifting goal angle >= 25 deg around semicircle
    pass_pairwise_decay_base: float = 0.25      # Multiplier decay: (0.25)^k for repeated pair exchanges
    min_progressive_dx_threshold: float = 30.0  # Minimum canonical dx or goal-dist reduction to qualify as progressive

    # Tiki-Taka Chain Parameters
    pass_chain_multiplier_coeff: float = 0.30  # Multiplier scale: (1.0 + 0.30 * C_chain)
    pass_chain_max_count: int = 10             # Cap at 10 passes (max multiplier 4.0x)

    # Kinematic Lead-Pass Targeting
    pass_lead_tau: float = 0.70                # Lookahead horizon in seconds: p_lead = p + tau * v

    # Defensive Half Ball-Winning Multiplier for DEF
    def_own_half_tackle_bonus: float = 0.75
    def_own_half_intercept_bonus: float = 0.5

    # Shot / Pass Intercepted by Opponent
    intercepted_kicker_ind: float = -0.8
    intercepted_team: float = -0.2

    # Tackle Outcomes
    successful_tackle_tackler_ind: float = 1.5
    successful_tackle_carrier_ind: float = -1.0
    successful_tackle_team: float = 0.3
    failed_tackle_in_range_ind: float = -0.10  # RNG fail when tackling an opponent carrier within range
    failed_tackle_ind: float = -0.30           # Close miss when opponent carrier is within range..near_range
    reckless_tackle_ind: float = -1.0          # Tackling with 0% chance (no opponent carrier within near_range)
    reckless_tackle_spam_step_pen: float = -0.03  # Per-RL-step penalty for holding tackle trigger in empty space
    failed_tackle_team: float = 0.0

    # Loose Ball Recovery
    loose_ball_pickup_ind: float = 0.5
    loose_ball_pickup_team: float = 0.15
    first_touch_reception_bonus: float = 0.35  # Discrete bonus when cleanly receiving a pass in flight within 55u
    keeper_box_recovery_ind: float = 1.2    # Bonus when GK picks up loose ball in own semicircle
    keeper_sweeper_chase_scale: float = 0.028  # Velocity reward scale for sweeper-keeper actively charging loose ball in box
    keeper_dive_save_attempt_ind: float = 0.60  # Bonus when GK triggers lunge towards shot-on-target
    keeper_reckless_lunge_pen: float = -0.15    # Penalty when GK triggers lunge away from play

    # Hold Violations (5s Field Hold & 5s Keeper Box Hold)
    field_5s_violation_ind: float = -2.0
    field_5s_violation_team: float = -0.5
    field_3s_violation_ind: float = -2.0   # Backwards-compatible alias
    field_3s_violation_team: float = -0.5  # Backwards-compatible alias
    keeper_5s_violation_ind: float = -4.0
    keeper_5s_violation_team: float = -1.5

    # Dense Per-Step Shaping (at 15 Hz RL step rate)
    possession_carrier_ind: float = 0.006
    possession_team: float = 0.001
    carrier_goal_advance_ind_scale: float = 0.008  # Individual bonus to carrier for dribbling toward enemy goal
    ball_advance_team_scale: float = 0.004         # Team reward per unit closer to enemy goal center
    hold_urgency_pen_scale: float = 0.035          # Escalating per-step penalty when field player holds ball > 3.2s

    # Context-Dependent Tactical & Role Positioning
    teammate_min_spacing: float = 165.0            # Increased from 135.0 to spread off-ball players across 1400x840 pitch
    teammate_box_cutback_spacing: float = 95.0     # Relaxed spacing when two attackers are in different angular sectors (>=40 deg) near box
    carrier_min_spacing: float = 135.0             # Increased from 120.0
    crowding_penalty_scale: float = 0.018          # Increased from 0.005 (applied asymmetrically with right-of-way exemptions)
    teammate_approach_penalty_scale: float = 0.012 # Penalty per unit closed when approaching a nearby off-ball teammate (<155u)
    carrier_approach_penalty_scale: float = 0.010  # Increased from 0.008
    box_sector_crowding_pen: float = 0.020         # Penalty for non-carrier clogging the same angular sector (<35 deg) within 240u of enemy goal
    gk_buffer_obstruction_pen: float = 0.015       # Penalty when actively driving toward enemy GK while enemy GK holds the ball
    support_pocket_approach_scale: float = 0.010
    open_support_bonus_scale: float = 0.010
    role_zone_anchor_scale: float = 0.005
    loose_ball_chase_scale: float = 0.010
    non_chaser_ball_repulsion_radius: float = 185.0
    non_chaser_ball_repulsion_scale: float = 0.012
    presser_approach_scale: float = 0.010
    ball_heading_align_scale: float = 0.003
    cover_approach_scale: float = 0.005
    keeper_angle_track_scale: float = 0.010
    keeper_anchor_approach_scale: float = 0.018
    keeper_edge_camp_pen: float = 0.035
    keeper_edge_camp_radius: float = 95.0  # Scaled to enlarged 195u semicircle to allow angle-narrowing bisector positioning
    keeper_fetch_box_scale: float = 0.045
    keeper_back_wall_face_pen: float = 0.035
    jitter_reverse_pen: float = -0.008


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
            "def_1": (320.0, 250.0),
            "def_2": (320.0, 590.0),
            "mid_1": (650.0, 260.0),
            "mid_2": (650.0, 580.0),
            "atk_1": (700.0, 420.0),  # Ball Carrier at center spot
        }
    )
    kickoff_receiving_right: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "keeper": (1355.0, 420.0),
            "def_1": (1120.0, 240.0),
            "def_2": (1120.0, 600.0),
            "mid_1": (960.0, 220.0),
            "mid_2": (960.0, 620.0),
            "atk_1": (910.0, 420.0),  # Outside center circle (210u from center spot)
        }
    )

    # 2. Foul Restart Formation (Open 3-2 Attacking Overload & Zonal Block)
    # Canonical coordinates where the Attacking Team attacks toward +X (1400.0, 420.0)
    foul_attacking_canonical: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "keeper": (40.0, 420.0),
            "def_1": (540.0, 260.0),
            "def_2": (540.0, 580.0),
            "mid_1": (965.0, 235.0),
            "mid_2": (965.0, 605.0),
            "atk_1": (820.0, 420.0),
        }
    )
    foul_defending_canonical: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "keeper": (1360.0, 420.0),
            "def_1": (1115.0, 315.0),
            "def_2": (1115.0, 525.0),
            "mid_1": (1030.0, 130.0),
            "mid_2": (1030.0, 710.0),
            "atk_1": (640.0, 420.0),
        }
    )
    # Backward-compatible mappings:
    foul_penalized_left: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "keeper": (40.0, 420.0),
            "def_1": (285.0, 525.0),
            "def_2": (285.0, 315.0),
            "mid_1": (370.0, 710.0),
            "mid_2": (370.0, 130.0),
            "atk_1": (760.0, 420.0),
        }
    )
    foul_attacking_right: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "keeper": (1360.0, 420.0),
            "def_1": (860.0, 580.0),
            "def_2": (860.0, 260.0),
            "mid_1": (435.0, 605.0),
            "mid_2": (435.0, 235.0),
            "atk_1": (580.0, 420.0),
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


if __name__ == "__main__":
    print("Loaded 6v6 MARL Football Configuration:")
    print(f"  Field Dimensions : {FIELD_WIDTH} x {FIELD_HEIGHT}")
    print(f"  Physics / RL Rate: {DEFAULT_CONFIG.physics.physics_hz} Hz / {DEFAULT_CONFIG.physics.rl_hz} Hz (frame_skip={FRAME_SKIP})")
    print(f"  Episode Steps    : {MAX_RL_STEPS} RL steps ({DEFAULT_CONFIG.physics.max_physics_frames} physics frames)")
    print(f"  Observation Space: shape=({OBS_DIM},), Action Space: shape=({ACTION_DIM},)")
