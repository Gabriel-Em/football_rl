"""Preallocated 60 Hz replay recorder.

`ReplayBuffer` stores each physics frame as slice assignments into fixed NumPy arrays and
builds the JSON-ready frame dictionaries only when `materialize()` is called. Telemetry that is
produced once per RL step (rewards, critic values, role labels, actions, ...) is stored once as a
step payload and shared by the frames of that step.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

_PLAYER_COLS = 6  # x, y, angle, flags, vx, vy
_BALL_COLS = 5  # x, y, team colour, vx, vy


class ReplayBuffer:
    """Fixed-capacity columnar store of replay frames; grows only if more frames are recorded than planned."""

    def __init__(self, max_frames: int, num_players: int) -> None:
        self.num_players = int(num_players)
        self._capacity = max(1, int(max_frames))
        self._count = 0
        self.version = 0  # incremented on every mutation; lets callers cache materialised frames
        self._allocate(self._capacity)
        self._steps: Dict[int, Dict[str, Any]] = {}
        self._events: Dict[int, List[Dict[str, Any]]] = {}

    def _allocate(self, capacity: int) -> None:
        self.players = np.zeros((capacity, self.num_players, _PLAYER_COLS), dtype=np.float64)
        self.ball = np.zeros((capacity, _BALL_COLS), dtype=np.float64)
        self.frame_no = np.zeros(capacity, dtype=np.int64)
        self.half = np.zeros(capacity, dtype=np.int64)
        self.team_side = np.zeros((capacity, 2), dtype=np.int64)
        self.step_index = np.zeros(capacity, dtype=np.int64)
        self.time = np.zeros(capacity, dtype=np.float64)
        self.box_radii = np.zeros((capacity, 2), dtype=np.float64)
        self.score = np.zeros((capacity, 2), dtype=np.int64)
        self.possession_frames = np.zeros((capacity, 2), dtype=np.int64)
        self.stationary = np.zeros(capacity, dtype=np.int64)
        self.passes = np.zeros((capacity, 2), dtype=np.int64)
        self.tackles = np.zeros((capacity, 2), dtype=np.int64)
        self.saves = np.zeros((capacity, 2), dtype=np.int64)
        self.shots = np.zeros((capacity, 2), dtype=np.int64)
        self.inflight = np.full(capacity, -1, dtype=np.int8)  # -1: not recorded, else 0/1
        self.post_kick = np.zeros((capacity, self.num_players), dtype=np.float64)

    _COLUMNS = (
        "players",
        "ball",
        "frame_no",
        "half",
        "team_side",
        "step_index",
        "time",
        "box_radii",
        "score",
        "possession_frames",
        "stationary",
        "passes",
        "tackles",
        "saves",
        "shots",
        "inflight",
        "post_kick",
    )

    def __len__(self) -> int:
        return self._count

    def reset(self) -> None:
        """Discard all recorded frames, events and step payloads; keeps the allocated arrays."""
        self._count = 0
        self.version += 1
        self._steps.clear()
        self._events.clear()

    def _grow(self) -> None:
        new_capacity = self._capacity * 2
        for name in self._COLUMNS:
            old = getattr(self, name)
            grown = np.zeros((new_capacity,) + old.shape[1:], dtype=old.dtype)
            if name == "inflight":
                grown[:] = -1
            grown[: self._capacity] = old
            setattr(self, name, grown)
        self._capacity = new_capacity

    def set_step_payload(self, step_index: int, payload: Dict[str, Any]) -> None:
        """Attach per-RL-step telemetry that every frame recorded with this `step_index` refers to."""
        self._steps[int(step_index)] = payload
        self.version += 1

    def record_frame(
        self,
        *,
        frame: int,
        time: float,
        half: int,
        team_side: np.ndarray,
        box_radii: np.ndarray,
        player_pos: np.ndarray,
        player_angle: np.ndarray,
        player_flags: np.ndarray,
        player_vel: np.ndarray,
        ball_pos: np.ndarray,
        ball_team: int,
        ball_vel: np.ndarray,
        step_index: int,
        score: Sequence[int],
        possession_frames: Sequence[int],
        stationary: int,
        passes: Sequence[int],
        tackles: Sequence[int],
        saves: Sequence[int],
        shots: Sequence[int],
        events: Sequence[Dict[str, Any]] | None = None,
        inflight: bool | None = None,
        post_kick: np.ndarray | None = None,
    ) -> None:
        """Copy one physics frame into the next free slot.

        `inflight` (a kicked ball is travelling) and `post_kick` (per-player post-kick shaping timers) are
        stored together; frames recorded without them materialise without the `inflight` / `pk` keys.
        """
        i = self._count
        if i == self._capacity:
            self._grow()
        row = self.players[i]
        row[:, 0:2] = player_pos
        row[:, 2] = player_angle
        row[:, 3] = player_flags
        row[:, 4:6] = player_vel
        ball = self.ball[i]
        ball[0:2] = ball_pos
        ball[2] = ball_team
        ball[3:5] = ball_vel
        self.frame_no[i] = frame
        self.time[i] = time
        self.half[i] = half
        self.team_side[i] = team_side
        self.box_radii[i] = box_radii
        self.step_index[i] = step_index
        self.score[i] = score
        self.possession_frames[i] = possession_frames
        self.stationary[i] = stationary
        self.passes[i] = passes
        self.tackles[i] = tackles
        self.saves[i] = saves
        self.shots[i] = shots
        if inflight is None:
            self.inflight[i] = -1
        else:
            self.inflight[i] = 1 if inflight else 0
            self.post_kick[i] = post_kick
        if events:
            self._events[i] = list(events)
        self._count = i + 1
        self.version += 1

    def append_event_to_last_frame(self, event: Dict[str, Any]) -> None:
        """Add an event to the most recently recorded frame."""
        if self._count == 0:
            raise IndexError("no recorded frame to attach the event to")
        self._events.setdefault(self._count - 1, []).append(event)
        self.version += 1

    def snapshot(self) -> "ReplayBuffer":
        """Return an independent, exactly-sized copy of the recorded frames, events and payloads."""
        snap = ReplayBuffer(max_frames=self._count, num_players=self.num_players)
        n = self._count
        for name in self._COLUMNS:
            getattr(snap, name)[:n] = getattr(self, name)[:n]
        snap._count = n
        snap._steps = dict(self._steps)
        snap._events = {i: list(evs) for i, evs in self._events.items() if i < n}
        return snap

    def materialize(self) -> List[Dict[str, Any]]:
        """Build the JSON-ready frame dictionaries for all recorded frames."""
        n = self._count
        players = self.players[:n].tolist()
        ball = self.ball[:n].tolist()
        frame_no = self.frame_no[:n].tolist()
        time = self.time[:n].tolist()
        half = self.half[:n].tolist()
        team_side = self.team_side[:n].tolist()
        box_radii = self.box_radii[:n].tolist()
        step_index = self.step_index[:n].tolist()
        score = self.score[:n].tolist()
        possession = self.possession_frames[:n].tolist()
        stationary = self.stationary[:n].tolist()
        passes = self.passes[:n].tolist()
        tackles = self.tackles[:n].tolist()
        saves = self.saves[:n].tolist()
        shots = self.shots[:n].tolist()
        inflight = self.inflight[:n].tolist()
        post_kick = self.post_kick[:n].round(3).tolist()

        frames: List[Dict[str, Any]] = []
        for i in range(n):
            poss = possession[i]
            total_poss = max(1, sum(poss))
            total_frames = max(1, frame_no[i])
            payload = self._steps[step_index[i]]
            b = ball[i]
            frames.append(
                {
                    "frame": frame_no[i],
                    "time": round(time[i], 3),
                    "half": half[i],
                    "team_side": team_side[i],
                    "box_radii": [round(box_radii[i][0], 1), round(box_radii[i][1], 1)],
                    "players": [
                        [round(r[0], 2), round(r[1], 2), round(r[2], 4), int(r[3]), round(r[4], 2), round(r[5], 2)]
                        for r in players[i]
                    ],
                    "ball": [round(b[0], 2), round(b[1], 2), int(b[2]), round(b[3], 2), round(b[4], 2)],
                    "player_rewards": payload["player_rewards"],
                    "cum_rewards": payload["cum_rewards"],
                    "v_pred": payload["v_pred"],
                    "rew_breakdown": payload["rew_breakdown"],
                    "role_state": payload["role_state"],
                    "role_targets": payload["role_targets"],
                    "actions": payload["actions"],
                    "stats": {
                        "score": score[i],
                        "possession_pct": [
                            round(100.0 * poss[0] / total_poss, 1),
                            round(100.0 * poss[1] / total_poss, 1),
                        ],
                        "true_possession_pct": [
                            round(100.0 * poss[0] / total_frames, 1),
                            round(100.0 * poss[1] / total_frames, 1),
                            round(100.0 * max(0, total_frames - sum(poss)) / total_frames, 1),
                        ],
                        "stationary_ball_frames": stationary[i],
                        "passes": passes[i],
                        "tackles": tackles[i],
                        "saves": saves[i],
                        "shots": shots[i],
                    },
                    "events": list(self._events.get(i, ())),
                }
            )
            if inflight[i] >= 0:
                frames[-1]["inflight"] = bool(inflight[i])
                frames[-1]["pk"] = post_kick[i]
            if payload.get("action_mean") is not None:
                frames[-1]["action_mean"] = payload["action_mean"]
        return frames
