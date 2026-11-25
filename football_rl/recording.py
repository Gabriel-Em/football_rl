"""Preallocated 60 Hz replay recorder.

`ReplayBuffer` stores each physics frame as slice assignments into fixed NumPy arrays and
builds the JSON-ready frame dictionaries only when `materialize()` is called. Telemetry that is
produced once per RL step (rewards, critic values, role labels, actions, ...) is stored once as a
step payload and shared by the frames of that step.

`ReplayRecordingMixin` is the environment side: it logs match events, captures each physics frame into the
buffer, scores a match's interest and writes replay files.
"""

from __future__ import annotations

import copy
from datetime import datetime
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from football_rl.potentials import ROLE_LABELS

BREAKDOWN_KEYS = ("r_ball_adv", "r_pot_i", "phi_role", "phi_crowd", "r_sparse")


def round_rows(values: np.ndarray, ndigits: int) -> List[List[float]]:
    """Rows of `values` as Python floats rounded half-even on their exact decimal value (builtin `round`)."""
    return [[round(x, ndigits) for x in row] for row in values.tolist()]


def round_vector(values: np.ndarray, ndigits: int) -> List[float]:
    return [round(x, ndigits) for x in values.tolist()]


def format_breakdown(columns: np.ndarray) -> List[Dict[str, float]]:
    """Per-player reward breakdown dictionaries from a `(len(BREAKDOWN_KEYS), players)` array."""
    rounded = round_rows(columns, 4)
    return [dict(zip(BREAKDOWN_KEYS, (col[i] for col in rounded))) for i in range(columns.shape[1])]


class LazyStepPayload:
    """Raw per-RL-step telemetry whose decimal rounding is deferred until the replay is materialised.

    The arrays are owned by the payload (they are never modified after construction). `resolve()` builds the
    JSON-ready dictionary once and returns the same object on every later call, so all frames of the step
    share it.
    """

    __slots__ = (
        "actions", "action_means", "v_preds", "player_rewards", "cum_rewards",
        "breakdown", "role_codes", "role_labels", "role_targets", "_resolved",
    )

    def __init__(
        self,
        *,
        actions: np.ndarray,
        action_means: Optional[np.ndarray],
        v_preds: Optional[np.ndarray],
        player_rewards: np.ndarray,
        cum_rewards: np.ndarray,
        breakdown: Optional[np.ndarray],
        role_codes: np.ndarray,
        role_labels: Tuple[str, ...],
        role_targets: np.ndarray,
    ) -> None:
        self.actions = actions
        self.action_means = action_means
        self.v_preds = v_preds
        self.player_rewards = player_rewards
        self.cum_rewards = cum_rewards
        self.breakdown = breakdown
        self.role_codes = role_codes
        self.role_labels = role_labels
        self.role_targets = role_targets
        self._resolved: Optional[Dict[str, Any]] = None

    def resolve(self) -> Dict[str, Any]:
        if self._resolved is None:
            self._resolved = {
                "action_mean": None if self.action_means is None else np.round(self.action_means, 3).tolist(),
                "player_rewards": round_vector(self.player_rewards, 4),
                "cum_rewards": round_vector(self.cum_rewards, 2),
                "v_pred": None if self.v_preds is None else round_vector(self.v_preds, 3),
                "rew_breakdown": None if self.breakdown is None else format_breakdown(self.breakdown),
                "role_state": [self.role_labels[c] for c in self.role_codes.tolist()],
                "role_targets": round_rows(self.role_targets, 1),
                "actions": round_rows(self.actions, 3),
            }
        return self._resolved


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
        self._steps: Dict[int, Any] = {}
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

    def set_step_payload(self, step_index: int, payload: "Dict[str, Any] | LazyStepPayload") -> None:
        """Attach per-RL-step telemetry (a ready dictionary or a `LazyStepPayload`) shared by the frames of `step_index`."""
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
            if isinstance(payload, LazyStepPayload):
                payload = payload.resolve()
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


# Replay role labels: potential-field roles plus the two status overrides shown while a player is stunned or lunging.
_STUN_ROLE_CODE = len(ROLE_LABELS)
_LUNGE_ROLE_CODE = len(ROLE_LABELS) + 1
_REPLAY_ROLE_LABELS = ROLE_LABELS + ("STUN", "LNGE")


class ReplayRecordingMixin:
    """Replay recording of `FootballParallelEnv`: events, per-frame capture, interest score and file export."""

    def _ensure_replay_buffer(self) -> ReplayBuffer:
        """Return the replay buffer, allocating it for a full match on first use."""
        if self._replay_buffer is None:
            self._replay_buffer = ReplayBuffer(
                max_frames=self.cfg.physics.max_physics_frames + 1, num_players=self.num_players
            )
        return self._replay_buffer

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

    def _replay_step_payload(self) -> LazyStepPayload:
        """Per-RL-step telemetry shared by every replay frame recorded during that step (rounded on export)."""
        return LazyStepPayload(
            actions=self._tele_actions,
            action_means=self._tele_action_means,
            v_preds=None if self.last_v_preds is None else np.array(self.last_v_preds, dtype=np.float32),
            player_rewards=self.last_step_rewards.copy(),
            cum_rewards=self.cum_player_rewards.copy(),
            breakdown=self._tele_breakdown,
            role_codes=self._tele_role_codes,
            role_labels=_REPLAY_ROLE_LABELS,
            role_targets=self._tele_targets,
        )

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
        if not self.uniform_capture and not self.export_replays:
            self.last_saved_replay_path = None
            return None
        interest_score = self.compute_interest_score()
        if self.uniform_capture:
            self._uniform_replay = {
                "match_id": self.current_match_id,
                "generation": self.current_generation,
                "interest_score": interest_score,
                "stats": copy.deepcopy(self.stats),
                "frames": self._ensure_replay_buffer().snapshot(),
            }
        if not self.export_replays:
            self.last_saved_replay_path = None
            return None
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
