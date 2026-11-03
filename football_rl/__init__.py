"""6v6 Multi-Agent Reinforcement Learning Continuous 2D Football package."""

from football_rl.config import (
    DEFAULT_CONFIG,
    EnvConfig,
    FormationConfig,
    MechanicsConfig,
    PhysicsConfig,
    ReplayConfig,
    RewardConfig,
    VisionConfig,
)
from football_rl.env import FootballParallelEnv, env, parallel_env

__all__ = [
    "DEFAULT_CONFIG",
    "EnvConfig",
    "FormationConfig",
    "FootballParallelEnv",
    "MechanicsConfig",
    "PhysicsConfig",
    "ReplayConfig",
    "RewardConfig",
    "VisionConfig",
    "env",
    "parallel_env",
]
