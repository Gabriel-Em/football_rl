"""High-performance Vectorized Environment wrapper for PettingZoo MARL Football.

Provides:
- `SubprocVecFootballEnv`: Multi-process parallel environment stepping across independent CPU cores.
- `DummyVecFootballEnv`: Synchronous in-process wrapper for `num_envs == 1` with zero IPC overhead.
- `make_vec_football_env`: Unified factory for seamless instantiation.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import connection as mp_connection
from multiprocessing import shared_memory
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from football_rl.config import ACTION_DIM, ALL_AGENTS, CENTRAL_STATE_DIM, DEFAULT_CONFIG, OBS_DIM, EnvConfig
from football_rl.env import FootballParallelEnv, parallel_env

NUM_PLAYERS = len(ALL_AGENTS)
DEFAULT_STEP_TIMEOUT = 60.0

_SHM_ALIGN = 64
_SHM_HAS_VALUES = 1
_SHM_HAS_MEANS = 2
_WORKER_IDLE_POLL_S = 1.0
_PARENT_POLL_S = 0.05


def _shm_layout(num_envs: int) -> Tuple[Dict[str, Tuple[str, Tuple[int, ...], int]], int]:
    """Field table `name -> (dtype, shape, byte offset)` and total byte size of the step transport.

    Outputs (`obs`, `central`, `rew`, `done`) carry a leading slot axis of 2 so that the results of one step stay
    intact while the next step is produced; inputs (`act`, `vpred`, `mean`) have a single slot because workers copy
    them out before replying. `ctl` holds each worker's `(slot, flags, step number, last answered step number)`;
    `status` is set to 1 by a worker that failed.
    """
    fields = (
        ("obs", "float32", (2, num_envs, NUM_PLAYERS, OBS_DIM)),
        ("central", "float32", (2, num_envs, NUM_PLAYERS, CENTRAL_STATE_DIM)),
        ("rew", "float32", (2, num_envs, NUM_PLAYERS)),
        ("done", "bool", (2, num_envs)),
        ("act", "float32", (num_envs, NUM_PLAYERS, ACTION_DIM)),
        ("vpred", "float32", (num_envs, NUM_PLAYERS)),
        ("mean", "float32", (num_envs, NUM_PLAYERS, ACTION_DIM)),
        ("ctl", "int32", (num_envs, 4)),
        ("status", "uint8", (num_envs,)),
    )
    layout: Dict[str, Tuple[str, Tuple[int, ...], int]] = {}
    offset = 0
    for name, dtype, shape in fields:
        layout[name] = (dtype, shape, offset)
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        offset += -(-nbytes // _SHM_ALIGN) * _SHM_ALIGN
    return layout, offset


def _shm_views(
    buf: Any, layout: Dict[str, Tuple[str, Tuple[int, ...], int]]
) -> Dict[str, np.ndarray]:
    return {
        name: np.ndarray(shape, dtype=np.dtype(dtype), buffer=buf, offset=offset)
        for name, (dtype, shape, offset) in layout.items()
    }


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn: Callable[[], FootballParallelEnv],
    shm_name: Optional[str] = None,
    shm_layout: Optional[Dict[str, Tuple[str, Tuple[int, ...], int]]] = None,
    worker_index: int = 0,
    cmd_sem: Any = None,
    done_sem: Any = None,
) -> None:
    """Worker loop running in a dedicated child process.

    A step is signalled by releasing `cmd_sem` with no pipe message; its inputs and results live in the
    shared-memory transport and the worker releases `done_sem` when the results are written (an episode summary,
    if the match ended, is sent through the pipe first). Every other command is a pipe message followed by a
    `cmd_sem` release. A worker without a parent exits when the command pipe reports end-of-file.
    """
    parent_remote.close()
    shm = shared_memory.SharedMemory(name=shm_name) if shm_name is not None else None
    views = _shm_views(shm.buf, shm_layout) if shm is not None else {}
    env = env_fn()
    owed_permits = 0
    try:
        while True:
            acquired = cmd_sem.acquire(timeout=_WORKER_IDLE_POLL_S)
            has_message = remote.poll(0)
            if not acquired and not has_message:
                continue
            if has_message:
                if not acquired:
                    # The parent sends a command and then releases the semaphore; the message was noticed in
                    # between, so that release arrives later and must not be mistaken for a step signal.
                    owed_permits += 1
                cmd, data = remote.recv()
            elif owed_permits:
                owed_permits -= 1
                continue
            else:
                cmd, data = "step_array", (int(views["ctl"][worker_index, 0]), int(views["ctl"][worker_index, 1]))
            if cmd == "step":
                actions = data
                next_obs, rewards, terminations, truncations, infos = env.step(actions)
                env_done = (len(env.agents) == 0)
                remote.send((next_obs, rewards, terminations, truncations, infos, env_done))
            elif cmd == "step_array":
                slot, flags = data
                v_preds = views["vpred"][worker_index].copy() if flags & _SHM_HAS_VALUES else None
                means = views["mean"][worker_index].copy() if flags & _SHM_HAS_MEANS else None
                obs_arr, central_arr, rew_arr, env_done, ep_summary = env.step_array(
                    views["act"][worker_index].copy(), v_preds=v_preds, action_means=means
                )
                views["obs"][slot, worker_index] = obs_arr
                views["central"][slot, worker_index] = central_arr
                views["rew"][slot, worker_index] = rew_arr
                views["done"][slot, worker_index] = env_done
                if env_done:
                    remote.send(ep_summary)
                views["ctl"][worker_index, 3] = views["ctl"][worker_index, 2]
                done_sem.release()
                continue
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
            elif cmd == "export_uniform_replay":
                remote.send(env.export_uniform_replay())
            elif cmd == "set_uniform_capture":
                env.uniform_capture = bool(data)
                remote.send(True)
            elif cmd == "set_generation":
                env._deferred_replay = None
                env._uniform_replay = None
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
            views["status"][worker_index] = 1
            remote.send(("ERROR", (str(e), tb)))
        except Exception:
            pass
        finally:
            done_sem.release()
            env.close()
            remote.close()
    finally:
        views = {}
        if shm is not None:
            try:
                shm.close()
            except BufferError:
                pass


class SubprocVecFootballEnv:
    """Multi-process vector environment: worker processes, a shared-memory step transport and pipe commands.

    `step_array` exchanges actions, values, policy means, observations, centralized states, rewards and done flags
    through one shared-memory block. Results are written to one of two slots that alternate on every step: the
    arrays returned by a step are views of that slot and stay valid until the step after next is issued. Copy them
    to keep them longer. A worker that dies, raises or stops answering makes the pending call raise
    `RuntimeError` naming the worker (within `step_timeout` seconds for a stalled one).
    """

    def __init__(
        self,
        env_fns: List[Callable[[], FootballParallelEnv]],
        context: Optional[str] = "spawn",
        step_timeout: Optional[float] = DEFAULT_STEP_TIMEOUT,
    ) -> None:
        self.num_envs = len(env_fns)
        self.waiting = False
        self.closed = False
        self.step_timeout = step_timeout
        self.processes: List[Any] = []
        self.remotes: Tuple[Any, ...] = ()
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._views: Dict[str, np.ndarray] = {}
        self._broken = False
        self._write_slot = 0
        self._step_number = 0
        self._pending_slot = 0
        self._send_failed: List[int] = []
        self._cmd_sems: List[Any] = []
        self._done_sem: Any = None

        layout, nbytes = _shm_layout(self.num_envs)
        self._shm = shared_memory.SharedMemory(create=True, size=nbytes)
        self._views = _shm_views(self._shm.buf, layout)

        try:
            ctx = mp.get_context(context)
            self.remotes, work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
            self._cmd_sems = [ctx.Semaphore(0) for _ in range(self.num_envs)]
            self._done_sem = ctx.Semaphore(0)
            for idx, (work_remote, remote, env_fn) in enumerate(zip(work_remotes, self.remotes, env_fns)):
                args = (
                    work_remote, remote, env_fn, self._shm.name, layout, idx, self._cmd_sems[idx], self._done_sem,
                )
                process = ctx.Process(target=_worker, args=args, daemon=True)
                process.start()
                self.processes.append(process)
                work_remote.close()
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------------
    # Pipe receive with worker liveness and timeout
    # ------------------------------------------------------------------
    def _failure(self, idx: int, reason: str) -> RuntimeError:
        self._broken = True
        self.waiting = False
        return RuntimeError(f"SubprocVec worker {idx} {reason}")

    def _recv(self, idx: int) -> Any:
        """Receive worker `idx`'s next message; raise `RuntimeError` if it reports an error, dies or times out."""
        remote = self.remotes[idx]
        sentinel = self.processes[idx].sentinel
        deadline = None if self.step_timeout is None else time.monotonic() + self.step_timeout
        while True:
            if not remote.poll(0):
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                ready = mp_connection.wait([remote, sentinel], timeout=remaining)
                if not ready:
                    raise self._failure(idx, f"timed out after {self.step_timeout:g} s without answering")
                if remote not in ready and not remote.poll(0):
                    self.processes[idx].join(timeout=1.0)
                    raise self._failure(idx, f"died (exitcode={self.processes[idx].exitcode})")
            try:
                res = remote.recv()
            except (EOFError, OSError):
                self.processes[idx].join(timeout=1.0)
                raise self._failure(idx, f"died (exitcode={self.processes[idx].exitcode})") from None
            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], str) and res[0] == "ERROR":
                err_msg, tb = res[1]
                self._broken = True
                self.waiting = False
                raise RuntimeError(f"Error in SubprocVec worker {idx}:\n{tb}")
            return res

    def _send(self, idx: int, message: Any) -> None:
        """Send a pipe command to worker `idx` and wake it; a broken pipe is reported on the matching receive."""
        try:
            self.remotes[idx].send(message)
        except (BrokenPipeError, ConnectionResetError, OSError):
            self._send_failed.append(idx)
        self._cmd_sems[idx].release()

    def _recv_failed_sends(self) -> None:
        failed, self._send_failed = self._send_failed, []
        for idx in failed:
            self._recv(idx)
            raise self._failure(idx, "closed its command pipe")

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
            self._send(idx, ("reset", (options, seed)))
        self._recv_failed_sends()

        obs_list, info_list = [], []
        for idx in env_indices:
            obs, info = self._recv(idx)
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def set_generation(self, generation: int) -> None:
        """Update generation counter across all environments."""
        for idx in range(self.num_envs):
            self._send(idx, ("set_generation", generation))
        self._recv_failed_sends()
        for idx in range(self.num_envs):
            self._recv(idx)

    def reset_array(
        self,
        seeds: Optional[List[Optional[int]]] = None,
        options_list: Optional[List[Optional[Dict[str, Any]]]] = None,
        env_indices: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reset environments and return contiguous (K, 12, OBS_DIM) obs and (K, 12, CENTRAL_STATE_DIM) central state arrays."""
        if env_indices is None:
            env_indices = list(range(self.num_envs))
        if seeds is None:
            seeds = [None] * len(env_indices)
        if options_list is None:
            options_list = [None] * len(env_indices)

        for idx, seed, options in zip(env_indices, seeds, options_list):
            self._send(idx, ("reset_array", (options, seed)))
        self._recv_failed_sends()

        k = len(env_indices)
        obs_batch = np.empty((k, NUM_PLAYERS, OBS_DIM), dtype=np.float32)
        central_batch = np.empty((k, NUM_PLAYERS, CENTRAL_STATE_DIM), dtype=np.float32)
        for out_i, idx in enumerate(env_indices):
            obs_arr, central_arr = self._recv(idx)
            obs_batch[out_i] = obs_arr
            central_batch[out_i] = central_arr
        return obs_batch, central_batch

    def step_array_async(
        self,
        actions_array: np.ndarray,
        v_preds: Optional[np.ndarray] = None,
        action_means: Optional[np.ndarray] = None,
    ) -> None:
        """Publish the (num_envs, 12, 4) actions (and optional values / means) and start all workers."""
        views = self._views
        if np.shape(actions_array) != views["act"].shape:
            raise ValueError(f"actions must have shape {views['act'].shape}, got {np.shape(actions_array)}")
        flags = 0
        if v_preds is not None:
            if np.shape(v_preds) != views["vpred"].shape:
                raise ValueError(f"v_preds must have shape {views['vpred'].shape}, got {np.shape(v_preds)}")
            views["vpred"][:] = v_preds
            flags |= _SHM_HAS_VALUES
        if action_means is not None:
            if np.shape(action_means) != views["mean"].shape:
                raise ValueError(f"action_means must have shape {views['mean'].shape}, got {np.shape(action_means)}")
            views["mean"][:] = action_means
            flags |= _SHM_HAS_MEANS
        views["act"][:] = actions_array
        slot = self._write_slot
        self._pending_slot = slot
        views["ctl"][:, 0] = slot
        views["ctl"][:, 1] = flags
        self._step_number += 1
        views["ctl"][:, 2] = self._step_number
        for sem in self._cmd_sems:
            sem.release()
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
        """Wait for all workers; return slot views of obs, central_state, rewards and dones plus episode summaries."""
        deadline = None if self.step_timeout is None else time.monotonic() + self.step_timeout
        done_sem = self._done_sem
        acked = 0
        while acked < self.num_envs:
            if done_sem.acquire(timeout=_PARENT_POLL_S):
                acked += 1
                continue
            self._check_workers_alive()
            if deadline is not None and time.monotonic() > deadline:
                raise self._failure(self._first_unanswered(), f"timed out after {self.step_timeout:g} s without answering")
        slot = self._pending_slot
        views = self._views
        if views["status"].any():
            self._recv(int(np.flatnonzero(views["status"])[0]))
            raise self._failure(int(np.flatnonzero(views["status"])[0]), "failed")
        done = views["done"][slot]
        ep_summaries: List[Optional[Dict[str, Any]]] = [None] * self.num_envs
        for idx in np.flatnonzero(done):
            ep_summaries[idx] = self._recv(int(idx))
        self._write_slot = slot ^ 1
        self.waiting = False
        return views["obs"][slot], views["central"][slot], views["rew"][slot], done, ep_summaries

    def _check_workers_alive(self) -> None:
        """Raise if a worker exited (reporting its error message if it left one) or flagged a failure."""
        for idx, process in enumerate(self.processes):
            if not process.is_alive():
                if self.remotes[idx].poll(0):
                    self._recv(idx)
                raise self._failure(idx, f"died (exitcode={process.exitcode})")

    def _first_unanswered(self) -> int:
        """Index of the first worker that has not answered the pending step."""
        ctl = self._views["ctl"]
        pending = np.flatnonzero(ctl[:, 3] != ctl[:, 2])
        return int(pending[0]) if len(pending) else 0

    def step_array(
        self,
        actions_array: np.ndarray,
        v_preds: Optional[np.ndarray] = None,
        action_means: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[Optional[Dict[str, Any]]],
    ]:
        """Execute a synchronous batched step using contiguous (num_envs, 12, 4) arrays."""
        self.step_array_async(actions_array, v_preds=v_preds, action_means=action_means)
        return self.step_array_wait()

    def export_replay(self, env_idx: int) -> Optional[str]:
        """Instruct worker `env_idx` to write its deferred replay trajectory to disk."""
        self._send(env_idx, ("export_replay", None))
        self._recv_failed_sends()
        return self._recv(env_idx)

    def set_uniform_capture(self, env_idx: Optional[int]) -> None:
        """Keep every completed match of worker `env_idx` (and of no other worker) for `export_uniform_replay`."""
        for idx in range(self.num_envs):
            self._send(idx, ("set_uniform_capture", idx == env_idx))
        self._recv_failed_sends()
        for idx in range(self.num_envs):
            self._recv(idx)

    def export_uniform_replay(self, env_idx: int) -> Optional[str]:
        """Instruct worker `env_idx` to write its captured uniform replay to disk."""
        self._send(env_idx, ("export_uniform_replay", None))
        self._recv_failed_sends()
        return self._recv(env_idx)

    def step_async(self, actions_by_env: List[Dict[str, np.ndarray]]) -> None:
        """Send action dictionaries to all workers without blocking."""
        for idx, act in enumerate(actions_by_env):
            self._send(idx, ("step", act))
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
        self._recv_failed_sends()
        results = [self._recv(idx) for idx in range(self.num_envs)]
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
        """Stop the workers and release the shared memory; safe after worker failures and when called twice."""
        if self.closed:
            return
        self.closed = True
        if self.waiting and not self._broken:
            try:
                self.step_array_wait()
            except Exception:
                pass
        for remote, sem in zip(self.remotes, self._cmd_sems):
            try:
                remote.send(("close", None))
            except Exception:
                pass
            sem.release()
        for process in self.processes:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
        for remote in self.remotes:
            try:
                remote.close()
            except Exception:
                pass
        self._views = {}
        self._cmd_sems = []
        self._done_sem = None
        if self._shm is not None:
            try:
                self._shm.close()
            except BufferError:
                pass
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
            self._shm = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


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
            env._uniform_replay = None
            env.current_generation = generation

    def set_uniform_capture(self, env_idx: Optional[int]) -> None:
        for idx, env in enumerate(self.envs):
            env.uniform_capture = idx == env_idx

    def export_uniform_replay(self, env_idx: int) -> Optional[str]:
        return self.envs[env_idx].export_uniform_replay()

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
        obs_batch = np.empty((k, 12, OBS_DIM), dtype=np.float32)
        central_batch = np.empty((k, 12, CENTRAL_STATE_DIM), dtype=np.float32)
        for out_i, (idx, seed, options) in enumerate(zip(env_indices, seeds, options_list)):
            self.envs[idx].reset(seed=seed, options=options)
            obs_batch[out_i] = self.envs[idx]._get_observations_array()
            central_batch[out_i] = self.envs[idx].get_centralized_state()
        return obs_batch, central_batch

    def step_array_async(
        self,
        actions_array: np.ndarray,
        v_preds: Optional[np.ndarray] = None,
        action_means: Optional[np.ndarray] = None,
    ) -> None:
        self._pending_actions_array = np.asarray(actions_array, dtype=np.float32)
        self._pending_v_preds = np.asarray(v_preds, dtype=np.float32) if v_preds is not None else None
        self._pending_action_means = (
            np.asarray(action_means, dtype=np.float32) if action_means is not None else None
        )

    def step_array_wait(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[Optional[Dict[str, Any]]],
    ]:
        obs_batch = np.empty((self.num_envs, 12, OBS_DIM), dtype=np.float32)
        central_batch = np.empty((self.num_envs, 12, CENTRAL_STATE_DIM), dtype=np.float32)
        rew_batch = np.empty((self.num_envs, 12), dtype=np.float32)
        done_batch = np.empty(self.num_envs, dtype=bool)
        ep_summaries: List[Optional[Dict[str, Any]]] = [None] * self.num_envs

        val_np = getattr(self, "_pending_v_preds", None)
        mean_np = getattr(self, "_pending_action_means", None)
        for idx, (env, act) in enumerate(zip(self.envs, self._pending_actions_array)):
            v = val_np[idx] if val_np is not None else None
            m = mean_np[idx] if mean_np is not None else None
            obs_arr, central_arr, rew_arr, env_done, ep_summary = env.step_array(
                act, v_preds=v, action_means=m
            )
            obs_batch[idx] = obs_arr
            central_batch[idx] = central_arr
            rew_batch[idx] = rew_arr
            done_batch[idx] = env_done
            ep_summaries[idx] = ep_summary

        return obs_batch, central_batch, rew_batch, done_batch, ep_summaries

    def step_array(
        self,
        actions_array: np.ndarray,
        v_preds: Optional[np.ndarray] = None,
        action_means: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[Optional[Dict[str, Any]]],
    ]:
        self.step_array_async(actions_array, v_preds=v_preds, action_means=action_means)
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
