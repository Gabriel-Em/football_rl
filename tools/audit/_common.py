"""Shared helpers for the audit tools: replay loading, selection and array views."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

SLOT_NAMES: tuple[str, ...] = ("GK", "D1", "D2", "M1", "M2", "ST")
TEAM_LABELS: tuple[str, ...] = ("A(red)", "B(blue)")
PLAYERS_PER_TEAM = 6
NUM_PLAYERS = 12
FLAG_HOLDER = 1
FLAG_GHOST = 2
FLAG_STUN = 4
FLAG_LUNGE = 8

_REPLAY_NAME_RE = re.compile(r"^match_(\d+)_gen(\d+)_.*\.json$")


def ensure_repo_on_path() -> None:
    """Make `football_rl` and `tools` importable when a script is run directly."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_replay_name(path: Path | str) -> Optional[tuple[int, int]]:
    """Return (match_id, generation) parsed from a replay file name, or None."""
    m = _REPLAY_NAME_RE.match(Path(path).name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def select_replays(
    replays_dir: Path | str,
    match_id: Optional[int] = None,
    generation: Optional[int] = None,
    last: Optional[int] = None,
) -> List[Path]:
    """List replay files sorted by match id, optionally filtered; `last` keeps the newest N."""
    d = Path(replays_dir)
    if not d.is_dir():
        return []
    found = []
    for p in d.iterdir():
        parsed = parse_replay_name(p)
        if parsed is None:
            continue
        mid, gen = parsed
        if match_id is not None and mid != match_id:
            continue
        if generation is not None and gen != generation:
            continue
        found.append((mid, p.name, p))
    found.sort()
    paths = [p for _, _, p in found]
    if last is not None and last > 0:
        paths = paths[-last:]
    return paths


@dataclass
class ReplayArrays:
    """Dense array view of a replay's raw (non-cinematic) frames."""

    meta: Dict[str, Any]
    frames: List[Dict[str, Any]]
    pos: np.ndarray          # (F, 12, 2)
    angle: np.ndarray        # (F, 12)
    flags: np.ndarray        # (F, 12) int
    vel: np.ndarray          # (F, 12, 2) zeros when absent
    ball: np.ndarray         # (F, 2)
    team_side: np.ndarray    # (F, 2) int, 0 = defends left goal
    holder: np.ndarray       # (F,) int, -1 = loose

    @property
    def n_frames(self) -> int:
        return int(self.pos.shape[0])

    def field_width(self) -> float:
        return float(self.meta.get("field_width", 1400.0))

    def field_height(self) -> float:
        return float(self.meta.get("field_height", 840.0))

    def goal_center_y(self) -> float:
        return 0.5 * (float(self.meta.get("goal_y_min", 320.0)) + float(self.meta.get("goal_y_max", 520.0)))

    def own_goal_x(self, team: int) -> np.ndarray:
        """(F,) x coordinate of the goal line `team` defends in each frame."""
        return np.where(self.team_side[:, team] == 0, 0.0, self.field_width())

    def canonical_x(self, team: int) -> np.ndarray:
        """(F, 12) distance of every player from `team`'s own goal line along the attacking axis."""
        gx = self.own_goal_x(team)[:, None]
        sign = np.where(self.team_side[:, team] == 0, 1.0, -1.0)[:, None]
        return (self.pos[:, :, 0] - gx) * sign


def replay_arrays(replay: Dict[str, Any]) -> ReplayArrays:
    frames = list(replay.get("frames") or [])
    meta = dict(replay.get("metadata") or {})
    n = len(frames)
    raw = np.zeros((n, NUM_PLAYERS, 6), dtype=np.float64)
    ball = np.zeros((n, 2), dtype=np.float64)
    sides = np.zeros((n, 2), dtype=np.int64)
    for f_i, fr in enumerate(frames):
        players = fr.get("players") or []
        for p_i, p in enumerate(players[:NUM_PLAYERS]):
            k = min(len(p), 6)
            raw[f_i, p_i, :k] = p[:k]
        b = fr.get("ball") or [0.0, 0.0]
        ball[f_i, 0] = b[0]
        ball[f_i, 1] = b[1]
        ts = fr.get("team_side") or [0, 1]
        sides[f_i, 0] = ts[0]
        sides[f_i, 1] = ts[1]
    flags = raw[:, :, 3].astype(np.int64)
    is_holder = (flags & FLAG_HOLDER) != 0
    holder = np.where(is_holder.any(axis=1), is_holder.argmax(axis=1), -1) if n else np.zeros(0, dtype=np.int64)
    return ReplayArrays(
        meta=meta,
        frames=frames,
        pos=raw[:, :, 0:2],
        angle=raw[:, :, 2],
        flags=flags,
        vel=raw[:, :, 4:6],
        ball=ball,
        team_side=sides,
        holder=holder.astype(np.int64),
    )


def safe_mean(values: Sequence[float] | np.ndarray) -> Optional[float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    return float(arr.mean())


def fmt(v: Any, nd: int = 3) -> str:
    """Compact formatter tolerant of None and nested lists."""
    if v is None:
        return "-"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(fmt(x, nd) for x in v) + "]"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_json(path: Path | str, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
