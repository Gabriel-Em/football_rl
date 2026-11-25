"""Standalone Pygame 60 FPS Replay Visualizer (`replay.py`) for 6v6 MARL Football."""

from __future__ import annotations

import argparse
from dataclasses import replace
import glob
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import pygame

from football_rl.config import DEFAULT_CONFIG, ReplayConfig
from football_rl.env import parallel_env
from football_rl.viewer.controls import ControlsMixin
from football_rl.viewer.hud import FloatingPopup, HudMixin
from football_rl.viewer.overlays import OverlaysMixin
from football_rl.viewer.pitch import PitchMixin
from football_rl.viewer.telemetry import TelemetryMixin
from football_rl.viewer.theme import COLOR_BG, FLAG_HAS_BALL
from football_rl.viewer.timeline import TimelineMixin


class ReplayViewer(TimelineMixin, ControlsMixin, PitchMixin, HudMixin, OverlaysMixin, TelemetryMixin):
    """Pygame visualizer for 60 Hz JSON match trajectories."""

    def __init__(
        self,
        replay_data: Dict[str, Any],
        headless: bool = False,
        show_vision_cones: bool = False,
        show_reward_hud: bool = False,
        show_telemetry: Optional[bool] = None,
        show_pass_lanes: Optional[bool] = None,
        show_wall_tracers: Optional[bool] = None,
        show_radar: Optional[bool] = None,
        auto_focus_carrier: Optional[bool] = None,
        show_critic: bool = False,
        show_gradient: bool = False,
    ) -> None:
        self.data = replay_data
        self.meta = replay_data["metadata"]

        self.field_w = float(self.meta.get("field_width", 1400.0))
        self.field_h = float(self.meta.get("field_height", 840.0))
        self.goal_y_min = float(self.meta.get("goal_y_min", 320.0))
        self.goal_y_max = float(self.meta.get("goal_y_max", 520.0))
        self.box_radius = float(self.meta.get("keeper_box_radius", 195.0))
        self.player_radius = float(self.meta.get("player_radius", 18.0))
        self.ball_radius = float(self.meta.get("ball_radius", 10.0))
        self.fov_deg = float(self.meta.get("vision_fov_deg", 90.0))
        self.vision_range = float(self.meta.get("vision_max_range", 650.0))
        self.physics_hz = int(self.meta.get("physics_hz", 60))

        raw_frames: List[Dict[str, Any]] = replay_data["frames"]
        if not self.meta.get("disable_cinematics", False):
            self.frames = self._build_cinematic_timeline(raw_frames)
        else:
            self.frames = raw_frames
        self.num_frames = len(self.frames)
        if self.num_frames == 0:
            raise ValueError("Replay file contains 0 frames.")
        self.frame_times: List[float] = self._build_frame_times(self.frames)
        self.match_duration: float = self._resolve_match_duration(raw_frames)

        # Scaling: keep pitch crisp on standard screens (~1200 max width)
        self.scale: float = min(1.0, 1200.0 / self.field_w)
        self.disp_field_w = int(round(self.field_w * self.scale))
        self.disp_field_h = int(round(self.field_h * self.scale))

        self.margin_x = 45
        self.hud_height = 95
        self.scrubber_height = 55
        self.side_panel_w = 340
        self.show_telemetry = (
            show_telemetry if show_telemetry is not None else DEFAULT_CONFIG.replay.show_telemetry
        )
        self.show_pass_lanes = (
            show_pass_lanes if show_pass_lanes is not None else DEFAULT_CONFIG.replay.show_pass_lanes
        )
        self.show_wall_tracers = (
            show_wall_tracers if show_wall_tracers is not None else DEFAULT_CONFIG.replay.show_wall_tracers
        )
        self.show_radar = (
            show_radar if show_radar is not None else DEFAULT_CONFIG.replay.show_radar
        )
        self.auto_focus_carrier = (
            auto_focus_carrier
            if auto_focus_carrier is not None
            else DEFAULT_CONFIG.replay.auto_focus_carrier
        )
        self.show_pocket_halos = True
        self.inspect_player_idx: Optional[int] = None
        self.hovered_player_idx: Optional[int] = None
        self.ball_trail: List[Tuple[float, float, float, float, int]] = []
        self.telemetry_row_rects: Dict[int, pygame.Rect] = {}
        self.telemetry_chip_rects: Dict[str, pygame.Rect] = {}
        self._gradient_cache: Dict[int, Tuple[List[List[float]], bool]] = {}
        self._potential_physics = replace(
            DEFAULT_CONFIG.physics,
            field_width=self.field_w,
            field_height=self.field_h,
            goal_y_min=self.goal_y_min,
            goal_y_max=self.goal_y_max,
            keeper_box_radius=self.box_radius,
        )

        self.win_w = self.disp_field_w + 2 * self.margin_x + (self.side_panel_w if self.show_telemetry else 0)
        self.win_h = self.disp_field_h + self.hud_height + self.scrubber_height

        # Playback state
        self.current_frame_idx: int = 0
        self.paused: bool = False
        self.playback_speed: float = 1.0
        self.subframe_accumulator: float = 0.0
        self._dragging_scrubber: bool = False
        self.show_vision_cones: bool = show_vision_cones
        self.selected_cone_players: Set[int] = set()
        self.show_reward_hud: bool = show_reward_hud
        self.show_critic: bool = show_critic
        self.show_gradient: bool = show_gradient
        self.running: bool = True
        self.headless: bool = headless

        # Floating text popups
        self.popups: List[FloatingPopup] = []
        self._last_processed_frame: int = -1

        # Pre-index timeline markers:
        # 1. Goal frames: (idx, team, is_own_goal)
        # 2. Save frames: (idx, team, is_lunge_save)
        # 3. High-reward (+5.0 or up) other event frames: (idx, team, label, reward)
        # 4. Foul / violation frames: (idx, team, label)
        self.goal_frames: List[Tuple[int, int, bool]] = []
        self.save_frames: List[Tuple[int, int, bool]] = []
        self.high_reward_frames: List[Tuple[int, int, str, float]] = []
        self.foul_frames: List[Tuple[int, int, str]] = []
        self.halftime_frame_idx: Optional[int] = None
        ht_cinematic_idx: Optional[int] = None
        ht_event_idx: Optional[int] = None
        second_half_idx: Optional[int] = None

        for idx, fr in enumerate(self.frames):
            if ht_cinematic_idx is None and fr.get("cinematic", {}).get("event_type") == "HALFTIME!":
                ht_cinematic_idx = idx
            if second_half_idx is None and fr.get("half") == 2:
                second_half_idx = idx
            for ev in fr.get("events", []):
                ev_t = str(ev.get("type", ""))
                if ev_t == "HALFTIME!" and ht_event_idx is None:
                    ht_event_idx = idx
                team = int(ev.get("team", -1))
                rew = ev.get("reward")
                rew_val = float(rew) if rew is not None else 0.0

                if ev_t in ("GOAL!", "OWN GOAL!"):
                    self.goal_frames.append((idx, team, ev_t == "OWN GOAL!"))
                elif ev_t in ("KEEPER SAVE!", "LUNGE SAVE!") or "SAVE" in ev_t.upper():
                    self.save_frames.append((idx, team, ev_t == "LUNGE SAVE!"))
                elif "FOUL" in ev_t.upper() or "VIOLATION" in ev_t.upper():
                    self.foul_frames.append((idx, team, ev_t))
                elif rew_val >= 5.0:
                    self.high_reward_frames.append((idx, team, ev_t, rew_val))

        for candidate in (ht_cinematic_idx, ht_event_idx, second_half_idx):
            if candidate is not None:
                self.halftime_frame_idx = candidate
                break

        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        pygame.init()
        pygame.font.init()
        if headless:
            self.screen = pygame.Surface((self.win_w, self.win_h))
        else:
            match_tag = (
                f"Match #{self.meta['match_id']} (Gen {self.meta.get('generation', 0)}) — "
                if self.meta.get("match_id", 0)
                else ""
            )
            pygame.display.set_caption(
                f"6v6 MARL Football Replay — {match_tag}Interest Score: {self.meta.get('interest_score', 0.0)}"
            )
            self.screen = pygame.display.set_mode((self.win_w, self.win_h))
            try:
                pygame.mouse.set_visible(False)
            except Exception:
                pass

        self.clock = pygame.time.Clock()
        self.pitch_surface = pygame.Surface(
            (self.disp_field_w, self.disp_field_h), pygame.SRCALPHA
        )
        self.cone_surface = pygame.Surface(
            (self.disp_field_w, self.disp_field_h), pygame.SRCALPHA
        )
        self.radar_surface = pygame.Surface(
            (self.disp_field_w, self.disp_field_h), pygame.SRCALPHA
        )
        self.tracer_surface = pygame.Surface(
            (self.disp_field_w, self.disp_field_h), pygame.SRCALPHA
        )

        # Fonts
        self.font_title = pygame.font.SysFont("dejavusans", 22, bold=True)
        self.font_score = pygame.font.SysFont("dejavusans", 30, bold=True)
        self.font_hud = pygame.font.SysFont("dejavusans", 13, bold=True)
        self.font_small = pygame.font.SysFont("dejavusans", 11, bold=False)
        self.font_jersey = pygame.font.SysFont("dejavusans", 12, bold=True)
        self.font_popup = pygame.font.SysFont("dejavusans", 16, bold=True)

        self._ingest_frame_events(0)
        self._init_possession_ribbon()

    def world_to_screen(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert pitch coordinates to window pixel coordinates."""
        return (
            int(round(self.margin_x + wx * self.scale)),
            int(round(self.hud_height + wy * self.scale)),
        )

    def _get_player_label(self, idx: int) -> str:
        """Format player role name, e.g. 'Red GK', 'Red D1', 'Blue ST'."""
        team_size = max(1, len(self.frames[0]["players"]) // 2) if self.frames else 6
        is_team_a = idx < team_size
        team_str = "Red" if is_team_a else "Blue"
        local_idx = idx if is_team_a else (idx - team_size)
        if team_size == 6:
            roles = ["GK", "D1", "D2", "M1", "M2", "ST"]
            role_str = roles[local_idx] if local_idx < len(roles) else f"P{local_idx}"
        else:
            role_str = "GK" if local_idx == 0 else f"P{local_idx}"
        return f"{team_str} {role_str}"

    def seek_to_frame(self, target_idx: int) -> None:
        """Jump to a specific frame index and clear stale popups."""
        self.current_frame_idx = max(0, min(self.num_frames - 1, target_idx))
        self.subframe_accumulator = 0.0
        self.popups.clear()
        self.ball_trail.clear()
        self._last_processed_frame = -1
        self._ingest_frame_events(self.current_frame_idx)

    def step_playback(self) -> None:
        """Advance playback by `self.playback_speed` frames when unpaused."""
        if self.paused:
            return

        self.subframe_accumulator += self.playback_speed
        frames_to_advance = int(self.subframe_accumulator)
        self.subframe_accumulator -= frames_to_advance

        for _ in range(frames_to_advance):
            if self.current_frame_idx < self.num_frames - 1:
                self.current_frame_idx += 1
                self._ingest_frame_events(self.current_frame_idx)
            else:
                self.paused = True
                break

    SPEED_LEVELS: List[float] = [0.25, 0.5, 0.75, 1.0, 2.0, 4.0]

    def change_playback_speed(self, delta: int) -> float:
        """Step playback speed: delta < 0 slows down (Down Arrow), delta > 0 speeds up (Up Arrow)."""
        curr_idx = min(
            range(len(self.SPEED_LEVELS)),
            key=lambda i: abs(self.SPEED_LEVELS[i] - self.playback_speed),
        )
        new_idx = max(0, min(len(self.SPEED_LEVELS) - 1, curr_idx + delta))
        self.playback_speed = self.SPEED_LEVELS[new_idx]
        return self.playback_speed

    def toggle_playback_speed(self, reverse: bool = False) -> float:
        """Step playback speed: reverse=True slows down (-1), reverse=False speeds up (+1)."""
        return self.change_playback_speed(-1 if reverse else 1)

    @property
    def focused_player_idx(self) -> Optional[int]:
        """Return the index of the currently focused player (selected or hovered), or None if nobody is focused."""
        if self.inspect_player_idx is not None:
            return self.inspect_player_idx
        return self.hovered_player_idx

    # ==========================================================================
    # Rendering Pipeline
    # ==========================================================================
    def render_frame(self) -> pygame.Surface:
        """Render the complete current frame."""
        frame = self.frames[self.current_frame_idx]
        if self.auto_focus_carrier:
            self._update_auto_carrier_focus(frame)
        self._update_ball_history(frame)
        self.screen.fill(COLOR_BG)

        # 1. Pitch Markings & Semi-Circles
        self._draw_pitch(frame)

        # 1b. Wall-Bank Rebound Tracers & Motion Ribbons (Toggleable via 'W')
        if self.show_wall_tracers:
            self._draw_wall_bank_tracers(frame)

        # 2. Vision Cones (Toggleable via 'V')
        if self.show_vision_cones:
            self._draw_vision_cones(frame)

        # 2b. Peripheral Radar 32-Ray 360 Fan (Toggleable via 'I' or Click)
        if self.show_radar or (self.inspect_player_idx is not None and self.show_radar):
            self._draw_peripheral_radar(frame)
        if self.show_radar:
            self._draw_role_tethers(frame)

        # 2c. Passing Lanes & Unpressed Lead Targets (Toggleable via 'P')
        if self.show_pass_lanes:
            self._draw_pass_lanes(frame)

        # 3. Goal Posts & Nets
        self._draw_goals(frame)

        # 4. Players (with role insignias, reward glows, lunge trails, ghost rings, stun icons, pocket halos)
        self._draw_players(frame)

        # 4b. Potential Gradient Compass (Toggleable via 'G')
        if self.show_gradient:
            self._draw_gradient_compass(frame)

        # 5. Ball
        self._draw_ball(frame)

        # 6. Rising & Fading Floating Event Popups
        self._draw_floating_popups()

        # 6b. Cinematic Broadcast Banner Overlay
        self._draw_cinematic_banner(frame)

        # 7. Top Scoreboard HUD & Bottom Timeline Scrubber
        self._draw_top_hud(frame)
        self._draw_scrubber_bar(frame)

        # 7b. Live Telemetry Side Panel (Toggleable via 'T')
        if self.show_telemetry:
            self._draw_telemetry_panel(frame)

        # 8. Software cursor (ensures clear visibility in WSLg / X11 environments)
        if not self.headless:
            self._draw_cursor()

        if not self.headless:
            pygame.display.flip()

        return self.screen

    def _toggle_telemetry(self) -> None:
        """Toggle side telemetry table HUD and adjust display dimensions."""
        self.show_telemetry = not self.show_telemetry
        self.win_w = self.disp_field_w + 2 * self.margin_x + (self.side_panel_w if self.show_telemetry else 0)
        if not self.headless:
            self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        else:
            self.screen = pygame.Surface((self.win_w, self.win_h))

    def _update_auto_carrier_focus(self, frame: Dict[str, Any]) -> None:
        """Automatically follow and focus the active ball carrier."""
        players = frame.get("players", [])
        carrier_idx = next(
            (i for i, p in enumerate(players) if int(p[3]) & FLAG_HAS_BALL),
            None,
        )
        if carrier_idx is not None:
            self.inspect_player_idx = carrier_idx
            self.selected_cone_players = {carrier_idx}
        else:
            self.inspect_player_idx = None
            self.selected_cone_players.clear()

    def _update_ball_history(self, frame: Dict[str, Any]) -> None:
        """Track ball positions and velocities over recent frames for trajectory tracing."""
        ball = frame.get("ball", [0.0, 0.0, -1])
        bx, by = float(ball[0]), float(ball[1])
        team_col = int(ball[2]) if len(ball) > 2 else -1
        bvx = float(ball[3]) if len(ball) > 3 else 0.0
        bvy = float(ball[4]) if len(ball) > 4 else 0.0
        self.ball_trail.append((bx, by, bvx, bvy, team_col))
        if len(self.ball_trail) > 45:
            self.ball_trail.pop(0)

    def run(self, max_frames: Optional[int] = None) -> None:
        """Run the main 60 FPS visualizer loop."""
        steps = 0
        while self.running:
            self.handle_events()
            self.step_playback()
            self.render_frame()
            self.clock.tick(self.physics_hz)
            steps += 1
            if max_frames is not None and steps >= max_frames:
                break
        pygame.quit()


# Backwards-compatible alias
ReplayVisualizer = ReplayViewer


def load_replay_json(filepath: str) -> Dict[str, Any]:
    """Load a JSON match trajectory from disk."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_replay(replay_dir: str = "replays") -> Optional[str]:
    """Return the most recently created `match_*.json` in `replay_dir` or subdirectories, if any."""
    files = glob.glob(os.path.join(replay_dir, "**", "match_*.json"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(replay_dir, "match_*.json"))
    files = sorted(files, key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def find_replay_by_match_id(
    match_id: int, replay_dir: str = "replays", checkpoint_dir: str = "checkpoints"
) -> Optional[str]:
    """Locate the replay JSON file for a specific `match_id`."""
    pattern = os.path.join(replay_dir, f"match_{match_id:04d}_*.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        pattern_sub = os.path.join(replay_dir, "**", f"match_{match_id:04d}_*.json")
        matches = sorted(glob.glob(pattern_sub, recursive=True))
    if matches:
        return matches[-1]

    hist_path = os.path.join(checkpoint_dir, "training_history.json")
    if os.path.exists(hist_path):
        with open(hist_path, "r", encoding="utf-8") as f:
            hist = json.load(f)
        for m in hist.get("matches", []):
            if int(m.get("match_id", -1)) == match_id and m.get("replay_path"):
                if os.path.exists(m["replay_path"]):
                    return m["replay_path"]

    return None


def list_matches_table(
    checkpoint_dir: str = "checkpoints",
    match_range: Optional[Tuple[int, int]] = None,
    top_k: Optional[int] = None,
) -> None:
    """Print a formatted table of recorded matches from `checkpoints/training_history.json`."""
    hist_path = os.path.join(checkpoint_dir, "training_history.json")
    if not os.path.exists(hist_path):
        print("No match history found at `checkpoints/training_history.json`.")
        return

    with open(hist_path, "r", encoding="utf-8") as f:
        hist = json.load(f)

    matches = list(hist.get("matches", []))
    if match_range is not None:
        lo, hi = match_range
        matches = [m for m in matches if lo <= int(m.get("match_id", 0)) <= hi]

    if top_k is not None:
        matches = sorted(
            matches, key=lambda m: float(m.get("interest_score", 0.0)), reverse=True
        )[:top_k]

    print(
        f"{'Match #':<9} {'Gen':<7} {'Score (B-R)':<13} {'Interest':<10} "
        f"{'Poss% (B/R)':<13} {'Shots':<9} {'Passes':<9} {'Tackles':<9} {'Saves':<7} {'Watch Command'}"
    )
    print("-" * 118)
    for m in matches:
        mid = int(m["match_id"])
        gen = int(m["generation"])
        sc = f"{m['score'][0]} - {m['score'][1]}"
        poss = f"{m['possession_pct'][0]:.0f}%/{m['possession_pct'][1]:.0f}%"
        sh = f"{m['shots'][0]}/{m['shots'][1]}"
        pa = f"{m['team_passes'][0]}/{m['team_passes'][1]}"
        ta = f"{m['team_tackles'][0]}/{m['team_tackles'][1]}"
        sa = f"{m['team_saves'][0]}/{m['team_saves'][1]}"
        cmd = f".venv/bin/python replay.py {mid}" if m.get("replay_path") else "(not saved)"
        print(
            f"#{mid:<8d} {gen:<7d} {sc:<13} {m['interest_score']:<10.1f} "
            f"{poss:<13} {sh:<9} {pa:<9} {ta:<9} {sa:<7} {cmd}"
        )


def generate_demo_replay(replay_dir: str = "replays", seed: int = 42) -> str:
    """Run a fast heuristic match and export a 60 FPS JSON replay for immediate viewing."""
    cfg = replace(
        DEFAULT_CONFIG,
        replay=ReplayConfig(
            save_dir=replay_dir,
            enable_recording=True,
            interest_threshold=0.0,
        ),
    )
    env = parallel_env(config=cfg)
    obs, _ = env.reset(seed=seed)

    rng = env.np_random
    while env.agents:
        actions = {}
        for agent in env.agents:
            a_obs = obs[agent]
            has_ball = a_obs[7] > 0.5
            hold_norm = a_obs[8]
            rel_goal_angle = a_obs[11]
            # Entity Slot 0 (Ball) is at indices 16..20 (16: is_visible, 18: rel_angle)
            ball_vis = a_obs[16] > 0.5
            ball_rel_ang = a_obs[18]

            throttle = 0.85
            if has_ball:
                steer = float(max(-1.0, min(1.0, rel_goal_angle * 2.5)))
                kick = 0.9 if (hold_norm > 0.55 or abs(rel_goal_angle) < 0.15) else -1.0
                special = -1.0
            else:
                steer = float(ball_rel_ang * 2.0) if ball_vis else 0.65
                kick = -1.0
                special = 1.0 if rng.random() < 0.08 else -1.0

            actions[agent] = [throttle, steer, kick, special]

        obs, _, _, _, _ = env.step(actions)

    assert env.last_saved_replay_path is not None
    return env.last_saved_replay_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pygame 60 FPS Replay Visualizer for 6v6 MARL Football."
    )
    parser.add_argument(
        "replay_path",
        nargs="?",
        default=None,
        help="Match number (e.g. `15`) or path to a `replays/match_*.json` file. Defaults to latest in `replays/`.",
    )
    parser.add_argument(
        "--match",
        "-m",
        type=int,
        default=None,
        help="Watch a specific match by its sequential Match ID (e.g. `--match 15` or `--match 157`).",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="Print a table of recorded matches and their stats to the terminal and exit.",
    )
    parser.add_argument(
        "--range",
        nargs=2,
        type=int,
        metavar=("START_MATCH", "END_MATCH"),
        default=None,
        help="Filter `--list` to a specific range of match numbers (e.g. `--range 1 100`).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Sort `--list` by highest interest score and show the top K matches (e.g. `--top 10`).",
    )
    parser.add_argument(
        "--generate-demo",
        action="store_true",
        help="Generate a fresh 60 FPS demo match replay before launching the viewer.",
    )
    parser.add_argument(
        "--headless-frames",
        type=int,
        default=None,
        help="Run headlessly for N frames and exit (for automated testing).",
    )
    parser.add_argument(
        "--show-cones",
        action="store_true",
        default=False,
        help="Enable vision cones overlay on startup (toggled in-viewer with 'V').",
    )
    parser.add_argument(
        "--show-rewards",
        action="store_true",
        default=False,
        help="Enable reward/punishment stats HUD overlay on startup (toggled in-viewer with 'E').",
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        default=False,
        help="Disable the live match telemetry sidebar table.",
    )
    parser.add_argument(
        "--show-pass-lanes",
        action="store_true",
        default=False,
        help="Enable passing lane vectors overlay on startup (toggled in-viewer with 'P').",
    )
    parser.add_argument(
        "--no-pass-lanes",
        action="store_true",
        default=False,
        help="Disable passing lane vectors overlay.",
    )
    parser.add_argument(
        "--no-walls",
        action="store_true",
        default=False,
        help="Disable boundary cushion rebound tracers.",
    )
    parser.add_argument(
        "--show-radar",
        action="store_true",
        default=False,
        help="Enable 360-degree peripheral radar overlay on startup (toggled in-viewer with 'I').",
    )
    parser.add_argument(
        "--auto-ball",
        action="store_true",
        default=False,
        help="Automatically follow and focus active ball carrier (toggled in-viewer with 'F').",
    )
    parser.add_argument(
        "--critic",
        action="store_true",
        default=False,
        help="Enable Critic value prediction overlay on startup (toggled in-viewer with 'C').",
    )
    parser.add_argument(
        "--grad",
        action="store_true",
        default=False,
        help="Enable potential gradient compass overlay on startup (toggled in-viewer with 'G').",
    )
    args = parser.parse_args()

    if args.list or args.range is not None or args.top is not None:
        m_range = (args.range[0], args.range[1]) if args.range else None
        list_matches_table(match_range=m_range, top_k=args.top)
        return

    filepath = args.replay_path
    target_match_id = args.match
    if target_match_id is None and filepath is not None and filepath.isdigit():
        target_match_id = int(filepath)
        filepath = None

    if target_match_id is not None:
        filepath = find_replay_by_match_id(target_match_id, DEFAULT_CONFIG.replay.save_dir)
        if filepath is None:
            raise FileNotFoundError(
                f"Could not find replay for Match #{target_match_id} in `{DEFAULT_CONFIG.replay.save_dir}/`."
            )

    if args.generate_demo or filepath is None:
        if not args.generate_demo:
            filepath = find_latest_replay(DEFAULT_CONFIG.replay.save_dir)
        if filepath is None:
            print("Generating a sample 60 FPS match replay in `replays/`...")
            filepath = generate_demo_replay(DEFAULT_CONFIG.replay.save_dir)
            print(f"Saved demo replay to: {filepath}")

    print(f"Loading replay: {filepath}")
    replay_data = load_replay_json(filepath)
    show_pass = True if args.show_pass_lanes else (False if args.no_pass_lanes else None)
    viewer = ReplayViewer(
        replay_data,
        headless=args.headless_frames is not None,
        show_vision_cones=args.show_cones,
        show_reward_hud=args.show_rewards,
        show_telemetry=not args.no_telemetry,
        show_pass_lanes=show_pass,
        show_wall_tracers=not args.no_walls,
        show_radar=args.show_radar,
        auto_focus_carrier=True if args.auto_ball else None,
        show_critic=args.critic,
        show_gradient=args.grad,
    )
    viewer.run(max_frames=args.headless_frames)


if __name__ == "__main__":
    main()
