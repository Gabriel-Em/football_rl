"""High-performance Vectorized Environment wrapper for PettingZoo MARL Football.

Provides:
- `SubprocVecFootballEnv`: Multi-process parallel environment stepping across independent CPU cores.
- `DummyVecFootballEnv`: Synchronous in-process wrapper for `num_envs == 1` with zero IPC overhead.
- `make_vec_football_env`: Unified factory for seamless instantiation.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from football_rl.config import ALL_AGENTS, DEFAULT_CONFIG, EnvConfig
from football_rl.env import FootballParallelEnv, parallel_env


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn: Callable[[], FootballParallelEnv],
) -> None:
    """Worker loop running in a dedicated child process."""
    parent_remote.close()
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                actions = data
                next_obs, rewards, terminations, truncations, infos = env.step(actions)
                env_done = (len(env.agents) == 0)
                remote.send((next_obs, rewards, terminations, truncations, infos, env_done))
            elif cmd == "step_array":
                if isinstance(data, tuple) and len(data) == 2:
                    act_data, val_data = data
                    obs_arr, central_arr, rew_arr, env_done, ep_summary = env.step_array(act_data, v_preds=val_data)
                else:
                    obs_arr, central_arr, rew_arr, env_done, ep_summary = env.step_array(data)
                remote.send((obs_arr, central_arr, rew_arr, env_done, ep_summary))
            elif cmd == "reset":
                options, seed = data
                obs, info = env.reset(seed=seed, options=options)
                remote.send((obs, info))
            elif cmd == "reset_array":
                options, seed = data
                env.reset(seed=seed, options=options)
                remote.send((env._get_observations_array(), env.get_centralized_state()))
            elif cmd == "export_replay":
                remote.send(env.export_deferred_replay())
            elif cmd == "set_generation":
                env._deferred_replay = None
                env.current_generation = int(data)
                remote.send(True)
            elif cmd == "get_stats":
                remote.send(env.stats)
            elif cmd == "close":
                env.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            remote.send(("ERROR", (str(e), tb)))
        except Exception:
            pass
        finally:
            env.close()
            remote.close()


class SubprocVecFootballEnv:
    """Multi-process vector environment using OS multiprocessing pipes."""

    def __init__(
        self,
        env_fns: List[Callable[[], FootballParallelEnv]],
        context: Optional[str] = "spawn",
    ) -> None:
        self.num_envs = len(env_fns)
        self.waiting = False
        self.closed = False

        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.processes = []

        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, env_fn)
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

    def reset(
        self,
        seeds: Optional[List[Optional[int]]] = None,
        options_list: Optional[List[Optional[Dict[str, Any]]]] = None,
        env_indices: Optional[List[int]] = None,
    ) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, Any]]]:
        """Reset environments (all or specified by `env_indices`)."""
        if env_indices is None:
            env_indices = list(range(self.num_envs))
        if seeds is None:
            seeds = [None] * len(env_indices)
        if options_list is None:
            options_list = [None] * len(env_indices)

        for idx, seed, options in zip(env_indices, seeds, options_list):
            self.remotes[idx].send(("reset", (options, seed)))

        obs_list, info_list = [], []
        for idx in env_indices:
            res = self.remotes[idx].recv()
            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], str) and res[0] == "ERROR":
                err_msg, tb = res[1]
                raise RuntimeError(f"Error in SubprocVec worker {idx}:\n{tb}")
            obs, info = res
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def set_generation(self, generation: int) -> None:
        """Update generation counter across all environments."""
        for remote in self.remotes:
            remote.send(("set_generation", generation))
        for remote in self.remotes:
            remote.recv()

    def reset_array(
        self,
        seeds: Optional[List[Optional[int]]] = None,
        options_list: Optional[List[Optional[Dict[str, Any]]]] = None,
        env_indices: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reset environments and return contiguous (K, 12, 76) obs and (K, 12, 54) central state arrays."""
        if env_indices is None:
            env_indices = list(range(self.num_envs))
        if seeds is None:
            seeds = [None] * len(env_indices)
        if options_list is None:
            options_list = [None] * len(env_indices)

        for idx, seed, options in zip(env_indices, seeds, options_list):
            self.remotes[idx].send(("reset_array", (options, seed)))

        k = len(env_indices)
        obs_batch = np.empty((k, 12, 76), dtype=np.float32)
        central_batch = np.empty((k, 12, 54), dtype=np.float32)
        for out_i, idx in enumerate(env_indices):
            res = self.remotes[idx].recv()
            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], str) and res[0] == "ERROR":
                err_msg, tb = res[1]
                raise RuntimeError(f"Error in SubprocVec worker {idx}:\n{tb}")
            obs_arr, central_arr = res
            obs_batch[out_i] = obs_arr
            central_batch[out_i] = central_arr
        return obs_batch, central_batch

    def step_array_async(
        self, actions_array: np.ndarray, v_preds: Optional[np.ndarray] = None
    ) -> None:
        """Send (num_envs, 12, 4) action array to all workers without dictionary construction."""
        act_np = np.asarray(actions_array, dtype=np.float32)
        val_np = np.asarray(v_preds, dtype=np.float32) if v_preds is not None else None
        for idx, (remote, act) in enumerate(zip(self.remotes, act_np)):
            v = val_np[idx] if val_np is not None else None
            remote.send(("step_array", (act, v)))
        self.waiting = True


    def step_array_wait(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[Optional[Dict[str, Any]]],
    ]:
        """Wait for all workers and return contiguous batch arrays for obs, central_state, rewards, and dones."""
        obs_batch = np.empty((self.num_envs, 12, 76), dtype=np.float32)
        central_batch = np.empty((self.num_envs, 12, 54), dtype=np.float32)
        rew_batch = np.empty((self.num_envs, 12), dtype=np.float32)
        done_batch = np.empty(self.num_envs, dtype=bool)
        ep_summaries: List[Optional[Dict[str, Any]]] = [None] * self.num_envs

        for idx, remote in enumerate(self.remotes):
            res = remote.recv()
            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], str) and res[0] == "ERROR":
                err_msg, tb = res[1]
                raise RuntimeError(f"Error in SubprocVec worker {idx}:\n{tb}")
            obs_arr, central_arr, rew_arr, env_done, ep_summary = res
            obs_batch[idx] = obs_arr
            central_batch[idx] = central_arr
            rew_batch[idx] = rew_arr
            done_batch[idx] = env_done
            ep_summaries[idx] = ep_summary

        self.waiting = False
        return obs_batch, central_batch, rew_batch, done_batch, ep_summaries

    def step_array(
        self, actions_array: np.ndarray, v_preds: Optional[np.ndarray] = None
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[Optional[Dict[str, Any]]],
    ]:
        """Execute a synchronous batched step using contiguous (num_envs, 12, 4) arrays."""
        self.step_array_async(actions_array, v_preds=v_preds)
        return self.step_array_wait()

    def export_replay(self, env_idx: int) -> Optional[str]:
        """Instruct worker `env_idx` to write its deferred replay trajectory to disk."""
        self.remotes[env_idx].send(("export_replay", None))
        res = self.remotes[env_idx].recv()
        if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], str) and res[0] == "ERROR":
            err_msg, tb = res[1]
            raise RuntimeError(f"Error in SubprocVec worker {env_idx}:\n{tb}")
        return res

    def step_async(self, actions_by_env: List[Dict[str, np.ndarray]]) -> None:
        """Send action dictionaries to all workers without blocking."""
        for remote, act in zip(self.remotes, actions_by_env):
            remote.send(("step", act))
        self.waiting = True

    def step_wait(
        self,
    ) -> Tuple[
        List[Dict[str, np.ndarray]],
        List[Dict[str, float]],
        List[Dict[str, bool]],
        List[Dict[str, bool]],
        List[Dict[str, Any]],
        List[bool],
    ]:
        """Wait for all workers to finish their physics step."""
        results = []
        for idx, remote in enumerate(self.remotes):
            res = remote.recv()
            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], str) and res[0] == "ERROR":
                err_msg, tb = res[1]
                raise RuntimeError(f"Error in SubprocVec worker {idx}:\n{tb}")
            results.append(res)
        self.waiting = False
        obs, rews, terms, truncs, infos, dones = zip(*results)
        return list(obs), list(rews), list(terms), list(truncs), list(infos), list(dones)

    def step(
        self, actions_by_env: List[Dict[str, np.ndarray]]
    ) -> Tuple[
        List[Dict[str, np.ndarray]],
        List[Dict[str, float]],
        List[Dict[str, bool]],
        List[Dict[str, bool]],
        List[Dict[str, Any]],
        List[bool],
    ]:
        self.step_async(actions_by_env)
        return self.step_wait()

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for process in self.processes:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
        self.closed = True

    def __del__(self) -> None:
        self.close()


class DummyVecFootballEnv:
    """Synchronous single-thread vector wrapper for num_envs == 1."""

    def __init__(self, env_fns: List[Callable[[], FootballParallelEnv]]) -> None:
        self.num_envs = len(env_fns)
        self.envs = [fn() for fn in env_fns]
        self.closed = False

    def reset(
        self,
        seeds: Optional[List[Optional[int]]] = None,
        options_list: Optional[List[Optional[Dict[str, Any]]]] = None,
        env_indices: Optional[List[int]] = None,
    ) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, Any]]]:
        if env_indices is None:
            env_indices = list(range(self.num_envs))
        if seeds is None:
            seeds = [None] * len(env_indices)
        if options_list is None:
            options_list = [None] * len(env_indices)

        obs_list, info_list = [], []
        for idx, seed, options in zip(env_indices, seeds, options_list):
            obs, info = self.envs[idx].reset(seed=seed, options=options)
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def set_generation(self, generation: int) -> None:
        for env in self.envs:
            env._deferred_replay = None
            env.current_generation = generation

    def reset_array(
        self,
        seeds: Optional[List[Optional[int]]] = None,
        options_list: Optional[List[Optional[Dict[str, Any]]]] = None,
        env_indices: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if env_indices is None:
            env_indices = list(range(self.num_envs))
        if seeds is None:
            seeds = [None] * len(env_indices)
        if options_list is None:
            options_list = [None] * len(env_indices)

        k = len(env_indices)
        obs_batch = np.empty((k, 12, 76), dtype=np.float32)
        central_batch = np.empty((k, 12, 54), dtype=np.float32)
        for out_i, (idx, seed, options) in enumerate(zip(env_indices, seeds, options_list)):
            self.envs[idx].reset(seed=seed, options=options)
            obs_batch[out_i] = self.envs[idx]._get_observations_array()
            central_batch[out_i] = self.envs[idx].get_centralized_state()
        return obs_batch, central_batch

    def step_array_async(
        self, actions_array: np.ndarray, v_preds: Optional[np.ndarray] = None
    ) -> None:
        self._pending_actions_array = np.asarray(actions_array, dtype=np.float32)
        self._pending_v_preds = np.asarray(v_preds, dtype=np.float32) if v_preds is not None else None

    def step_array_wait(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[Optional[Dict[str, Any]]],
    ]:
        obs_batch = np.empty((self.num_envs, 12, 76), dtype=np.float32)
        central_batch = np.empty((self.num_envs, 12, 54), dtype=np.float32)
        rew_batch = np.empty((self.num_envs, 12), dtype=np.float32)
        done_batch = np.empty(self.num_envs, dtype=bool)
        ep_summaries: List[Optional[Dict[str, Any]]] = [None] * self.num_envs

        val_np = getattr(self, "_pending_v_preds", None)
        for idx, (env, act) in enumerate(zip(self.envs, self._pending_actions_array)):
            v = val_np[idx] if val_np is not None else None
            obs_arr, central_arr, rew_arr, env_done, ep_summary = env.step_array(act, v_preds=v)
            obs_batch[idx] = obs_arr
            central_batch[idx] = central_arr
            rew_batch[idx] = rew_arr
            done_batch[idx] = env_done
            ep_summaries[idx] = ep_summary

        return obs_batch, central_batch, rew_batch, done_batch, ep_summaries

    def step_array(
        self, actions_array: np.ndarray, v_preds: Optional[np.ndarray] = None
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[Optional[Dict[str, Any]]],
    ]:
        self.step_array_async(actions_array, v_preds=v_preds)
        return self.step_array_wait()

    def export_replay(self, env_idx: int) -> Optional[str]:
        return self.envs[env_idx].export_deferred_replay()

    def step_async(self, actions_by_env: List[Dict[str, np.ndarray]]) -> None:
        self._pending_actions = actions_by_env

    def step_wait(
        self,
    ) -> Tuple[
        List[Dict[str, np.ndarray]],
        List[Dict[str, float]],
        List[Dict[str, bool]],
        List[Dict[str, bool]],
        List[Dict[str, Any]],
        List[bool],
    ]:
        obs_list, rew_list, term_list, trunc_list, info_list, done_list = [], [], [], [], [], []
        for env, act in zip(self.envs, self._pending_actions):
            next_obs, rew, term, trunc, info = env.step(act)
            obs_list.append(next_obs)
            rew_list.append(rew)
            term_list.append(term)
            trunc_list.append(trunc)
            info_list.append(info)
            done_list.append(len(env.agents) == 0)
        return obs_list, rew_list, term_list, trunc_list, info_list, done_list

    def step(
        self, actions_by_env: List[Dict[str, np.ndarray]]
    ) -> Tuple[
        List[Dict[str, np.ndarray]],
        List[Dict[str, float]],
        List[Dict[str, bool]],
        List[Dict[str, bool]],
        List[Dict[str, Any]],
        List[bool],
    ]:
        self.step_async(actions_by_env)
        return self.step_wait()

    def close(self) -> None:
        if self.closed:
            return
        for env in self.envs:
            env.close()
        self.closed = True

    def __del__(self) -> None:
        self.close()


class _EnvWorkerThunk:
    """Picklable top-level factory thunk for multiprocessing spawn workers."""

    def __init__(self, config: EnvConfig) -> None:
        self.config = config

    def __call__(self) -> FootballParallelEnv:
        return parallel_env(config=self.config)


def make_vec_football_env(
    config: EnvConfig = DEFAULT_CONFIG,
    num_envs: int = 1,
) -> SubprocVecFootballEnv | DummyVecFootballEnv:
    """Create a vector environment (SubprocVecFootballEnv for num_envs > 1, DummyVecFootballEnv for 1)."""
    env_fns = [_EnvWorkerThunk(config) for _ in range(num_envs)]
    if num_envs == 1:
        return DummyVecFootballEnv(env_fns)
    return SubprocVecFootballEnv(env_fns, context="spawn")
