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
from football_rl.vec_env import (
    DummyVecFootballEnv,
    SubprocVecFootballEnv,
    make_vec_football_env,
)

__all__ = [
    "DEFAULT_CONFIG",
    "DummyVecFootballEnv",
    "EnvConfig",
    "FormationConfig",
    "FootballParallelEnv",
    "MechanicsConfig",
    "PhysicsConfig",
    "ReplayConfig",
    "RewardConfig",
    "SubprocVecFootballEnv",
    "VisionConfig",
    "env",
    "make_vec_football_env",
    "parallel_env",
]
