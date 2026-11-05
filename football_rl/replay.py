"""Standalone Pygame 60 FPS Replay Visualizer (`replay.py`) for 6v6 MARL Football."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import glob
import json
import math
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import pygame

from football_rl.config import DEFAULT_CONFIG, ReplayConfig
from football_rl.env import parallel_env


# ==============================================================================
# Color Palette & Layout Constants
# ==============================================================================
COLOR_BG = (18, 22, 28)
COLOR_HUD_BG = (24, 30, 38)
COLOR_HUD_BORDER = (52, 64, 80)
COLOR_TURF_DARK = (34, 112, 58)
COLOR_TURF_LIGHT = (40, 124, 65)
COLOR_LINE = (235, 240, 245)
COLOR_GOAL_NET = (190, 200, 210)

# Team A (Red) & Team B (Blue)
COLOR_TEAM_A = (255, 60, 75)
COLOR_TEAM_A_DARK = (165, 25, 35)
COLOR_TEAM_A_CONE = (255, 65, 85, 70)

COLOR_TEAM_B = (55, 155, 255)
COLOR_TEAM_B_DARK = (25, 75, 165)
COLOR_TEAM_B_CONE = (55, 155, 255, 60)

# Role Accents
COLOR_KEEPER_GOLD = (255, 215, 70)
COLOR_ROLE_DEF = (85, 225, 255)   # Cyan Shield
COLOR_ROLE_MID = (110, 245, 155)  # Emerald Playmaker
COLOR_ROLE_ATK = (255, 185, 55)   # Orange-Gold Striker

COLOR_BALL_NEUTRAL = (248, 248, 248)
COLOR_GHOST_RING = (190, 230, 255)
COLOR_STUN_ALERT = (255, 50, 50)
COLOR_TEXT_MAIN = (240, 244, 248)
COLOR_TEXT_MUTED = (160, 172, 188)

# State Flag Bitmasks
FLAG_HAS_BALL = 1
FLAG_IS_GHOST = 2
FLAG_IS_STUNNED = 4
FLAG_IS_LUNGING = 8

EVENT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "GOAL!": (255, 225, 65),
    "OWN GOAL!": (255, 95, 65),
    "LUNGE SAVE!": (80, 245, 200),
    "KEEPER SAVE!": (80, 245, 200),
    "GK STRIP!": (110, 220, 250),
    "SHOT ON TARGET!": (255, 200, 90),
    "NEAR SHOT!": (255, 165, 80),
    "PASS!": (125, 245, 130),
    "THROUGH BALL!": (110, 255, 140),
    "ONE-TWO!": (100, 255, 180),
    "WALL PASS!": (95, 230, 255),
    "LAY-OFF!": (120, 240, 200),
    "ESCAPE PRESS!": (140, 230, 255),
    "BALL WON!": (100, 240, 180),
    "TACKLE!": (255, 175, 65),
    "TACKLE FAILED!": (255, 130, 130),
    "AIR TACKLE!": (255, 70, 70),
    "GK BACK-KICK!": (255, 80, 120),
    "5s VIOLATION!": (245, 110, 255),
    "3s VIOLATION!": (245, 110, 255),
    "5s KEEPER FOUL!": (255, 75, 150),
    "5s KEEPER HOLD FOUL!": (255, 75, 150),
    "5s UNCLAIMED BOX BALL!": (255, 110, 180),
    "HALFTIME!": (200, 220, 255),
    "FULLTIME!": (255, 215, 0),
    "GK DIVE!": (100, 230, 255),
}


@dataclass
class FloatingPopup:
    """Rising and fading floating text spawned at world event coordinates."""

    text: str
    world_x: float
    world_y: float
    color: Tuple[int, int, int]
    spawn_frame: int
    duration_frames: int = 75  # 1.25s at 60 FPS


class ReplayViewer:
    """Pygame visualizer for 60 Hz JSON match trajectories."""

    def __init__(
        self,
        replay_data: Dict[str, Any],
        headless: bool = False,
        show_vision_cones: bool = False,
        show_reward_hud: bool = False,
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

        # Scaling: keep pitch crisp on standard screens (~1200 max width)
        self.scale: float = min(1.0, 1200.0 / self.field_w)
        self.disp_field_w = int(round(self.field_w * self.scale))
        self.disp_field_h = int(round(self.field_h * self.scale))

        self.margin_x = 45
        self.hud_height = 95
        self.scrubber_height = 55
        self.win_w = self.disp_field_w + 2 * self.margin_x
        self.win_h = self.disp_field_h + self.hud_height + self.scrubber_height

        # Playback state
        self.current_frame_idx: int = 0
        self.paused: bool = False
        self.playback_speed: float = 1.0
        self.subframe_accumulator: float = 0.0
        self.show_vision_cones: bool = show_vision_cones
        self.selected_cone_players: Set[int] = set()
        self.show_reward_hud: bool = show_reward_hud
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

        for idx, fr in enumerate(self.frames):
            for ev in fr.get("events", []):
                ev_t = str(ev.get("type", ""))
                team = int(ev.get("team", -1))
                rew = ev.get("reward")
                rew_val = float(rew) if rew is not None else 0.0

                if ev_t in ("GOAL!", "OWN GOAL!"):
                    self.goal_frames.append((idx, team, ev_t == "OWN GOAL!"))
                elif ev_t in ("KEEPER SAVE!", "LUNGE SAVE!") or "SAVE" in ev_t.upper():
                    self.save_frames.append((idx, team, ev_t == "LUNGE SAVE!"))
                elif "FOUL" in ev_t.upper() or "VIOLATION" in ev_t.upper() or "UNCLAIMED BOX BALL" in ev_t.upper():
                    self.foul_frames.append((idx, team, ev_t))
                elif rew_val >= 5.0:
                    self.high_reward_frames.append((idx, team, ev_t, rew_val))

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

        self.clock = pygame.time.Clock()
        self.pitch_surface = pygame.Surface(
            (self.disp_field_w, self.disp_field_h), pygame.SRCALPHA
        )
        self.cone_surface = pygame.Surface(
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

    def world_to_screen(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert pitch coordinates to window pixel coordinates."""
        return (
            int(round(self.margin_x + wx * self.scale)),
            int(round(self.hud_height + wy * self.scale)),
        )

    # ==========================================================================
    # Event & Input Handling
    # ==========================================================================
    def handle_events(self) -> None:
        """Process keyboard and mouse controls."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_v:
                    self.show_vision_cones = not self.show_vision_cones
                elif event.key == pygame.K_c:
                    # Toggle between All cones, Selected cone, or OFF
                    if self.selected_cone_players:
                        self.selected_cone_players.clear()
                        self.show_vision_cones = False
                    else:
                        self.selected_cone_players.add(0)
                        self.show_vision_cones = True
                elif event.key == pygame.K_TAB:
                    # Cycle through individual players for cone isolation (0..11 -> OFF -> 0)
                    mods = pygame.key.get_mods()
                    is_shift = bool(mods & pygame.KMOD_SHIFT)
                    num_p = len(self.frames[0]["players"]) if self.frames else 12
                    if not self.selected_cone_players or not self.show_vision_cones:
                        next_idx = (num_p - 1) if is_shift else 0
                        self.selected_cone_players = {next_idx}
                        self.show_vision_cones = True
                    elif len(self.selected_cone_players) == 1:
                        curr_p = next(iter(self.selected_cone_players))
                        if is_shift:
                            if curr_p == 0:
                                self.selected_cone_players.clear()
                                self.show_vision_cones = False
                            else:
                                self.selected_cone_players = {curr_p - 1}
                                self.show_vision_cones = True
                        else:
                            if curr_p == num_p - 1:
                                self.selected_cone_players.clear()
                                self.show_vision_cones = False
                            else:
                                self.selected_cone_players = {curr_p + 1}
                                self.show_vision_cones = True
                    else:
                        self.selected_cone_players.clear()
                        self.show_vision_cones = False
                elif event.key == pygame.K_e:
                    self.show_reward_hud = not self.show_reward_hud
                elif event.key == pygame.K_r:
                    self.seek_to_frame(0)
                elif event.key in (pygame.K_0, pygame.K_s):
                    # Toggle slow-mo (0.25x) vs normal speed (1.0x)
                    if math.isclose(self.playback_speed, 0.25, abs_tol=1e-3):
                        self.playback_speed = 1.0
                    else:
                        self.playback_speed = 0.25
                elif event.key == pygame.K_1:
                    self.playback_speed = 1.0
                elif event.key == pygame.K_2:
                    self.playback_speed = 2.0
                elif event.key == pygame.K_4:
                    self.playback_speed = 4.0
                elif event.key in (pygame.K_LEFTBRACKET, pygame.K_COMMA):
                    speeds = [0.25, 0.5, 1.0, 2.0, 4.0]
                    curr_idx = min(range(len(speeds)), key=lambda i: abs(speeds[i] - self.playback_speed))
                    self.playback_speed = speeds[max(0, curr_idx - 1)]
                elif event.key in (pygame.K_RIGHTBRACKET, pygame.K_PERIOD):
                    speeds = [0.25, 0.5, 1.0, 2.0, 4.0]
                    curr_idx = min(range(len(speeds)), key=lambda i: abs(speeds[i] - self.playback_speed))
                    self.playback_speed = speeds[min(len(speeds) - 1, curr_idx + 1)]
                elif event.key == pygame.K_RIGHT:
                    mods = pygame.key.get_mods()
                    delta = 1 if (mods & pygame.KMOD_SHIFT or self.paused) else self.physics_hz * 5
                    self.seek_to_frame(self.current_frame_idx + delta)
                elif event.key == pygame.K_LEFT:
                    mods = pygame.key.get_mods()
                    delta = 1 if (mods & pygame.KMOD_SHIFT or self.paused) else self.physics_hz * 5
                    self.seek_to_frame(self.current_frame_idx - delta)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                bar_x = self.margin_x
                bar_w = self.disp_field_w
                bar_y = self.hud_height + self.disp_field_h + 12
                # 1. Left Click on Scrubber Bar -> Seek
                if event.button == 1 and bar_x <= mx <= bar_x + bar_w and bar_y - 10 <= my <= bar_y + 26:
                    ratio = (mx - bar_x) / float(bar_w)
                    self.seek_to_frame(int(ratio * (self.num_frames - 1)))
                # 2. Left Click on Pitch -> Toggle clicked player's vision cone
                elif event.button == 1 and self._is_pos_on_field(mx, my):
                    p_idx = self._find_player_at_screen_pos(mx, my)
                    if p_idx is not None:
                        if p_idx in self.selected_cone_players:
                            self.selected_cone_players.remove(p_idx)
                            # When deselecting the active player, turn cones OFF (do not revert to ALL)
                            if not self.selected_cone_players:
                                self.show_vision_cones = False
                        else:
                            self.selected_cone_players.add(p_idx)
                            self.show_vision_cones = True
                # 3. Right Click anywhere on Pitch -> Clear player cone selection and turn cones OFF
                elif event.button == 3:
                    if self.selected_cone_players:
                        self.selected_cone_players.clear()
                        self.show_vision_cones = False
                    else:
                        self.show_vision_cones = not self.show_vision_cones

    def _is_pos_on_field(self, mx: int, my: int) -> bool:
        """Check if mouse coordinate is within the pitch turf rendering area."""
        return (
            self.margin_x <= mx <= self.margin_x + self.disp_field_w
            and self.hud_height <= my <= self.hud_height + self.disp_field_h
        )

    def _find_player_at_screen_pos(self, mx: int, my: int) -> Optional[int]:
        """Find index of player closest to screen coordinate within click radius (default 28px)."""
        if not self.frames:
            return None
        frame = self.frames[self.current_frame_idx]
        players = frame.get("players", [])
        best_idx = None
        best_dist = 28.0
        for idx, pdata in enumerate(players):
            wx, wy = float(pdata[0]), float(pdata[1])
            sx, sy = self.world_to_screen(wx, wy)
            d = math.hypot(mx - sx, my - sy)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return best_idx

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
        self._last_processed_frame = -1
        self._ingest_frame_events(self.current_frame_idx)

    def _ingest_frame_events(self, frame_idx: int) -> None:
        """Spawn floating text popups for discrete events at `frame_idx` with signed reward values."""
        if frame_idx == self._last_processed_frame:
            return
        self._last_processed_frame = frame_idx
        frame = self.frames[frame_idx]
        for ev in frame.get("events", []):
            ev_type = str(ev.get("type", "EVENT"))
            color = EVENT_COLORS.get(ev_type, (255, 255, 255))
            rew_val = ev.get("reward")
            if rew_val is not None:
                text = f"{ev_type} ({float(rew_val):+.1f})"
                if float(rew_val) > 0 and ev_type not in EVENT_COLORS:
                    color = (120, 255, 140)
                elif float(rew_val) < 0 and ev_type not in EVENT_COLORS:
                    color = (255, 90, 90)
            else:
                text = ev_type

            self.popups.append(
                FloatingPopup(
                    text=text,
                    world_x=float(ev.get("x", self.field_w * 0.5)),
                    world_y=float(ev.get("y", self.field_h * 0.5)),
                    color=color,
                    spawn_frame=frame_idx,
                )
            )

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

    # ==========================================================================
    # Rendering Pipeline
    # ==========================================================================
    def render_frame(self) -> pygame.Surface:
        """Render the complete current frame."""
        frame = self.frames[self.current_frame_idx]
        self.screen.fill(COLOR_BG)

        # 1. Pitch Markings & Semi-Circles
        self._draw_pitch(frame)

        # 2. Vision Cones (Toggleable via 'V')
        if self.show_vision_cones:
            self._draw_vision_cones(frame)

        # 3. Goal Posts & Nets
        self._draw_goals(frame)

        # 4. Players (with role insignias, reward glows, lunge trails, ghost rings, stun icons)
        self._draw_players(frame)

        # 5. Ball
        self._draw_ball(frame)

        # 6. Rising & Fading Floating Event Popups
        self._draw_floating_popups()

        # 6b. Cinematic Broadcast Banner Overlay
        self._draw_cinematic_banner(frame)

        # 7. Top Scoreboard HUD & Bottom Timeline Scrubber
        self._draw_top_hud(frame)
        self._draw_scrubber_bar(frame)

        if not self.headless:
            pygame.display.flip()

        return self.screen

    def _draw_pitch(self, frame: Dict[str, Any]) -> None:
        """Draw turf stripes, boundary lines, center circle, and keeper semi-circles."""
        w = self.disp_field_w
        h = self.disp_field_h
        self.pitch_surface.fill(COLOR_TURF_DARK)

        # Alternating vertical grass stripes (10 stripes)
        stripe_w = w // 10
        for i in range(0, 10, 2):
            pygame.draw.rect(
                self.pitch_surface,
                COLOR_TURF_LIGHT,
                pygame.Rect(i * stripe_w, 0, stripe_w, h),
            )

        # Tint Keeper Semi-Circles based on which team defends Left vs Right
        team_side = frame.get("team_side", [0, 1])
        left_team_is_a = team_side[0] == 0
        red_box_color = (255, 65, 85, 75)
        blue_box_color = (55, 160, 255, 55)
        left_box_color = red_box_color if left_team_is_a else blue_box_color
        right_box_color = blue_box_color if left_team_is_a else red_box_color

        nom_box_r = int(round(self.box_radius * self.scale))
        active_radii = frame.get("box_radii", [self.box_radius, self.box_radius])
        left_box_r = max(1, int(round(float(active_radii[0]) * self.scale)))
        right_box_r = max(1, int(round(float(active_radii[1]) * self.scale)))

        box_overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        if left_box_r > 2:
            pygame.draw.circle(box_overlay, left_box_color, (0, h // 2), left_box_r)
        if right_box_r > 2:
            pygame.draw.circle(box_overlay, right_box_color, (w, h // 2), right_box_r)
        self.pitch_surface.blit(box_overlay, (0, 0))

        # If shrunken, draw a faint dashed/thin guide line at nominal 195u line
        if left_box_r < nom_box_r - 2:
            pygame.draw.circle(self.pitch_surface, (160, 175, 190, 80), (0, h // 2), nom_box_r, width=1)
        if right_box_r < nom_box_r - 2:
            pygame.draw.circle(self.pitch_surface, (160, 175, 190, 80), (w, h // 2), nom_box_r, width=1)

        # Keeper Semi-Circle solid boundary arcs with team-tinted accent inner ring
        left_arc_col = (255, 80, 100) if left_team_is_a else (70, 175, 255)
        right_arc_col = (70, 175, 255) if left_team_is_a else (255, 80, 100)
        if left_box_r > 2:
            pygame.draw.circle(self.pitch_surface, COLOR_LINE, (0, h // 2), left_box_r, width=2)
            pygame.draw.circle(self.pitch_surface, left_arc_col, (0, h // 2), max(1, left_box_r - 2), width=2)
        if right_box_r > 2:
            pygame.draw.circle(self.pitch_surface, COLOR_LINE, (w, h // 2), right_box_r, width=2)
            pygame.draw.circle(self.pitch_surface, right_arc_col, (w, h // 2), max(1, right_box_r - 2), width=2)

        # Center line & center circle
        pygame.draw.line(self.pitch_surface, COLOR_LINE, (w // 2, 0), (w // 2, h), width=2)
        pygame.draw.circle(
            self.pitch_surface, COLOR_LINE, (w // 2, h // 2), int(round(90 * self.scale)), width=2
        )
        pygame.draw.circle(
            self.pitch_surface, COLOR_LINE, (w // 2, h // 2), max(4, int(round(6 * self.scale)))
        )

        # Outer pitch boundary
        pygame.draw.rect(self.pitch_surface, COLOR_LINE, pygame.Rect(0, 0, w, h), width=3)

        self.screen.blit(self.pitch_surface, (self.margin_x, self.hud_height))

    def _draw_goals(self, frame: Dict[str, Any]) -> None:
        """Draw left and right goal nets and goal posts."""
        gy1 = self.hud_height + int(round(self.goal_y_min * self.scale))
        gy2 = self.hud_height + int(round(self.goal_y_max * self.scale))
        gh = gy2 - gy1
        depth = max(18, int(round(28 * self.scale)))

        team_side = frame.get("team_side", [0, 1])
        left_color = COLOR_TEAM_A if team_side[0] == 0 else COLOR_TEAM_B
        right_color = COLOR_TEAM_B if team_side[0] == 0 else COLOR_TEAM_A

        # Left Goal (X <= 0)
        left_rect = pygame.Rect(self.margin_x - depth, gy1, depth, gh)
        pygame.draw.rect(self.screen, (35, 42, 52), left_rect)
        pygame.draw.rect(self.screen, left_color, left_rect, width=3)
        pygame.draw.circle(self.screen, COLOR_LINE, (self.margin_x, gy1), 5)
        pygame.draw.circle(self.screen, COLOR_LINE, (self.margin_x, gy2), 5)

        # Right Goal (X >= 1400)
        right_x = self.margin_x + self.disp_field_w
        right_rect = pygame.Rect(right_x, gy1, depth, gh)
        pygame.draw.rect(self.screen, (35, 42, 52), right_rect)
        pygame.draw.rect(self.screen, right_color, right_rect, width=3)
        pygame.draw.circle(self.screen, COLOR_LINE, (right_x, gy1), 5)
        pygame.draw.circle(self.screen, COLOR_LINE, (right_x, gy2), 5)

    def _draw_vision_cones(self, frame: Dict[str, Any]) -> None:
        """Render semi-transparent vision cones, filtering by selected players if active."""
        self.cone_surface.fill((0, 0, 0, 0))
        half_fov = math.radians(self.fov_deg * 0.5)
        num_arc_pts = 16
        num_players = len(frame["players"])
        team_size = max(1, num_players // 2)

        for idx, pdata in enumerate(frame["players"]):
            if self.selected_cone_players and idx not in self.selected_cone_players:
                continue
            px, py, angle, flags = float(pdata[0]), float(pdata[1]), float(pdata[2]), int(pdata[3])
            if flags & FLAG_IS_STUNNED:
                continue

            is_team_a = idx < team_size
            is_selected = bool(self.selected_cone_players and idx in self.selected_cone_players)

            if is_team_a:
                # Vibrant coral-crimson for Red team cones
                cone_color = (255, 75, 95, 120) if is_selected else (255, 65, 85, 70)
                edge_color = (255, 130, 150, 180)
            else:
                # Crisp electric blue for Blue team cones
                cone_color = (55, 180, 255, 110) if is_selected else (55, 160, 255, 60)
                edge_color = (130, 215, 255, 170)

            points = [(int(round(px * self.scale)), int(round(py * self.scale)))]
            arc_points = []
            for step in range(num_arc_pts + 1):
                a = (angle - half_fov) + (2.0 * half_fov * step / num_arc_pts)
                cx = (px + math.cos(a) * self.vision_range) * self.scale
                cy = (py + math.sin(a) * self.vision_range) * self.scale
                pt = (int(round(cx)), int(round(cy)))
                points.append(pt)
                arc_points.append(pt)

            pygame.draw.polygon(self.cone_surface, cone_color, points)
            if len(arc_points) > 1:
                pygame.draw.lines(self.cone_surface, edge_color, False, [points[0], arc_points[0]], width=1)
                pygame.draw.lines(self.cone_surface, edge_color, False, arc_points, width=2)
                pygame.draw.lines(self.cone_surface, edge_color, False, [arc_points[-1], points[0]], width=1)

        self.screen.blit(self.cone_surface, (self.margin_x, self.hud_height))

    def _draw_players(self, frame: Dict[str, Any]) -> None:
        """Render players with distinct role insignia, reward feedback, lunge trails, ghost rings, and stun icons."""
        r = max(10, int(round(self.player_radius * self.scale)))
        num_players = len(frame["players"])
        team_size = max(1, num_players // 2)

        p_rews = frame.get("player_rewards")
        c_rews = frame.get("cum_rewards")

        for idx, pdata in enumerate(frame["players"]):
            wx, wy, angle, flags = (
                float(pdata[0]),
                float(pdata[1]),
                float(pdata[2]),
                int(pdata[3]),
            )
            sx, sy = self.world_to_screen(wx, wy)
            is_team_a = idx < team_size
            local_idx = idx % team_size

            # Role distinction
            if team_size == 6:
                if local_idx == 0:
                    role_type = "GK"
                    label_str = "GK"
                    role_accent = COLOR_KEEPER_GOLD
                elif local_idx in (1, 2):
                    role_type = "DEF"
                    label_str = f"D{local_idx}"
                    role_accent = COLOR_ROLE_DEF
                elif local_idx in (3, 4):
                    role_type = "MID"
                    label_str = f"M{local_idx - 2}"
                    role_accent = COLOR_ROLE_MID
                else:
                    role_type = "ATK"
                    label_str = "ST"
                    role_accent = COLOR_ROLE_ATK
            else:
                if local_idx == 0:
                    role_type = "GK"
                    label_str = "GK"
                    role_accent = COLOR_KEEPER_GOLD
                else:
                    role_type = "FIELD"
                    label_str = str(local_idx)
                    role_accent = COLOR_LINE

            base_color = COLOR_TEAM_A if is_team_a else COLOR_TEAM_B
            dark_color = COLOR_TEAM_A_DARK if is_team_a else COLOR_TEAM_B_DARK

            # 1. Lunge Trail
            if flags & FLAG_IS_LUNGING:
                self._draw_lunge_trail(idx, base_color)

            # 2. Reward Feedback Glow Ring & Live Cumulative Pill
            step_rew = float(p_rews[idx]) if (p_rews and idx < len(p_rews)) else 0.0
            cum_rew = float(c_rews[idx]) if (c_rews and idx < len(c_rews)) else 0.0
            if self.show_reward_hud:
                if step_rew >= 0.015:
                    pygame.draw.circle(self.screen, (75, 255, 120), (sx, sy), r + 6, width=3)
                elif step_rew <= -0.008:
                    pygame.draw.circle(self.screen, (255, 65, 85), (sx, sy), r + 6, width=3)

                cum_col = (110, 245, 130) if cum_rew >= 0 else (255, 120, 120)
                rew_lbl = self.font_small.render(f"{cum_rew:+.1f}", True, cum_col)
                self.screen.blit(rew_lbl, (sx - rew_lbl.get_width() // 2, sy + r + 8))

            # 3. Possession Halo
            if flags & FLAG_HAS_BALL:
                pygame.draw.circle(self.screen, COLOR_KEEPER_GOLD, (sx, sy), r + 5, width=2)

            # 3b. Cone Selection Halo (Cyan ring when this player's vision cone is explicitly isolated)
            if self.show_vision_cones and idx in self.selected_cone_players:
                pygame.draw.circle(self.screen, (80, 245, 200), (sx, sy), r + 7, width=2)

            # 4. Player Body & Role Insignia
            if flags & FLAG_IS_GHOST:
                ghost_surf = pygame.Surface((r * 2 + 10, r * 2 + 10), pygame.SRCALPHA)
                pygame.draw.circle(
                    ghost_surf,
                    (*base_color, 105),
                    (r + 5, r + 5),
                    r,
                )
                self.screen.blit(ghost_surf, (sx - r - 5, sy - r - 5))
                self._draw_dashed_circle((sx, sy), r + 4, COLOR_GHOST_RING, num_dashes=10)
            else:
                pygame.draw.circle(self.screen, dark_color, (sx, sy), r)
                pygame.draw.circle(self.screen, base_color, (sx, sy), r - 2)

                # Border per role
                if role_type == "GK":
                    pygame.draw.circle(self.screen, COLOR_KEEPER_GOLD, (sx, sy), r + 4, width=2)
                    pygame.draw.circle(self.screen, COLOR_KEEPER_GOLD, (sx, sy), r, width=3)
                elif role_type == "DEF":
                    pygame.draw.circle(self.screen, COLOR_ROLE_DEF, (sx, sy), r, width=3)
                    # Rear shield arc (±110° opposite to angle at radius r + 3)
                    rear_ang = angle + math.pi
                    shield_pts = []
                    for st in range(9):
                        ang = (rear_ang - math.radians(110)) + (2.0 * math.radians(110) * st / 8)
                        shield_pts.append((
                            int(round(sx + math.cos(ang) * (r + 3))),
                            int(round(sy + math.sin(ang) * (r + 3)))
                        ))
                    if len(shield_pts) > 1:
                        pygame.draw.lines(self.screen, COLOR_ROLE_DEF, False, shield_pts, width=3)
                elif role_type == "MID":
                    pygame.draw.circle(self.screen, COLOR_ROLE_MID, (sx, sy), r, width=2)
                    # 4 compass tick marks
                    for k in range(4):
                        tick_ang = angle + k * (math.pi / 2.0)
                        t1 = (int(round(sx + math.cos(tick_ang) * r)), int(round(sy + math.sin(tick_ang) * r)))
                        t2 = (int(round(sx + math.cos(tick_ang) * (r + 4))), int(round(sy + math.sin(tick_ang) * (r + 4))))
                        pygame.draw.line(self.screen, COLOR_ROLE_MID, t1, t2, width=2)
                elif role_type == "ATK":
                    pygame.draw.circle(self.screen, COLOR_ROLE_ATK, (sx, sy), r, width=3)
                    # Forward-pointing striker chevron / spearhead triangle extending r + 7 along angle
                    tip = (int(round(sx + math.cos(angle) * (r + 7))), int(round(sy + math.sin(angle) * (r + 7))))
                    left_w = (int(round(sx + math.cos(angle + math.radians(35)) * (r + 1))), int(round(sy + math.sin(angle + math.radians(35)) * (r + 1))))
                    right_w = (int(round(sx + math.cos(angle - math.radians(35)) * (r + 1))), int(round(sy + math.sin(angle - math.radians(35)) * (r + 1))))
                    pygame.draw.polygon(self.screen, COLOR_ROLE_ATK, [tip, left_w, right_w])
                else:
                    pygame.draw.circle(self.screen, COLOR_LINE, (sx, sy), r, width=2)

            # 5. Facing Direction Line
            fx = int(round(sx + math.cos(angle) * (r + 4)))
            fy = int(round(sy + math.sin(angle) * (r + 4)))
            pygame.draw.line(self.screen, COLOR_LINE, (sx, sy), (fx, fy), width=3)

            # 6. Jersey Label
            lbl = self.font_jersey.render(label_str, True, COLOR_TEXT_MAIN)
            self.screen.blit(lbl, (sx - lbl.get_width() // 2, sy - lbl.get_height() // 2))

            # 7. Flashing Red Stun Indicator if Stunned
            if flags & FLAG_IS_STUNNED:
                flash_on = (self.current_frame_idx // 3) % 2 == 0
                stun_col = COLOR_STUN_ALERT if flash_on else (255, 220, 60)
                pygame.draw.circle(self.screen, stun_col, (sx, sy), r + 6, width=3)
                stun_lbl = self.font_hud.render("STUN!", True, stun_col)
                self.screen.blit(stun_lbl, (sx - stun_lbl.get_width() // 2, sy - r - 22))

    def _draw_lunge_trail(
        self, player_idx: int, team_color: Tuple[int, int, int]
    ) -> None:
        """Draw rapid fading motion trail behind a lunging keeper."""
        trail_len = 7
        r = max(8, int(round(self.player_radius * self.scale)))
        for back in range(trail_len, 0, -1):
            prev_idx = max(0, self.current_frame_idx - back)
            pdata = self.frames[prev_idx]["players"][player_idx]
            tx, ty = self.world_to_screen(float(pdata[0]), float(pdata[1]))
            alpha = int(150 * (trail_len - back + 1) / (trail_len + 1))
            trail_surf = pygame.Surface((r * 2 + 6, r * 2 + 6), pygame.SRCALPHA)
            pygame.draw.circle(
                trail_surf,
                (*COLOR_KEEPER_GOLD[:3], alpha),
                (r + 3, r + 3),
                max(4, r - back),
            )
            pygame.draw.circle(
                trail_surf,
                (*team_color, alpha),
                (r + 3, r + 3),
                max(2, r - back - 2),
            )
            self.screen.blit(trail_surf, (tx - r - 3, ty - r - 3))

    def _draw_dashed_circle(
        self,
        center: Tuple[int, int],
        radius: int,
        color: Tuple[int, int, int],
        num_dashes: int = 10,
    ) -> None:
        """Draw a dashed ring around a Ghost Mode player."""
        cx, cy = center
        for i in range(num_dashes):
            start_ang = (2.0 * math.pi * i) / num_dashes
            end_ang = start_ang + (math.pi / num_dashes)
            x1 = int(round(cx + math.cos(start_ang) * radius))
            y1 = int(round(cy + math.sin(start_ang) * radius))
            x2 = int(round(cx + math.cos(end_ang) * radius))
            y2 = int(round(cy + math.sin(end_ang) * radius))
            pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), width=2)

    def _draw_ball(self, frame: Dict[str, Any]) -> None:
        """Render the ball colored dynamically by holding/last-kicking team."""
        bx, by, last_team = (
            float(frame["ball"][0]),
            float(frame["ball"][1]),
            int(frame["ball"][2]),
        )
        sx, sy = self.world_to_screen(bx, by)
        br = max(6, int(round(self.ball_radius * self.scale)))

        if last_team == 0:
            fill_col = COLOR_TEAM_A
        elif last_team == 1:
            fill_col = COLOR_TEAM_B
        else:
            fill_col = COLOR_BALL_NEUTRAL

        pygame.draw.circle(self.screen, (20, 20, 20), (sx, sy), br + 2)
        pygame.draw.circle(self.screen, fill_col, (sx, sy), br)
        pygame.draw.circle(self.screen, (255, 255, 255), (sx - 2, sy - 2), max(2, br // 3))

    def _draw_floating_popups(self) -> None:
        """Render rising and fading floating event labels."""
        active_popups: List[FloatingPopup] = []
        for p in self.popups:
            age = self.current_frame_idx - p.spawn_frame
            if 0 <= age < p.duration_frames:
                progress = age / float(p.duration_frames)
                rise_px = int(progress * 38.0)
                alpha = int(255 * (1.0 - progress**1.5))

                sx, sy = self.world_to_screen(p.world_x, p.world_y)
                sy -= 24 + rise_px

                text_surf = self.font_popup.render(p.text, True, p.color)
                shadow_surf = self.font_popup.render(p.text, True, (10, 12, 16))
                text_surf.set_alpha(alpha)
                shadow_surf.set_alpha(alpha)

                tx = sx - text_surf.get_width() // 2
                ty = max(self.hud_height + 4, sy - text_surf.get_height() // 2)
                self.screen.blit(shadow_surf, (tx + 2, ty + 2))
                self.screen.blit(text_surf, (tx, ty))
                active_popups.append(p)

        self.popups = active_popups

    # ==========================================================================
    # Cinematic Broadcast Sequences & Overlays
    # ==========================================================================
    def _build_cinematic_timeline(
        self, raw_frames: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Expand major set-piece events into 3-stage broadcast cinematic sequences."""
        if len(raw_frames) == 0:
            return raw_frames

        expanded_frames: List[Dict[str, Any]] = []

        for i in range(len(raw_frames)):
            curr_fr = raw_frames[i]
            if i == 0:
                expanded_frames.append(curr_fr)
                continue

            events = curr_fr.get("events", [])
            trans_events = [
                ev
                for ev in events
                if ev.get("type") in ("GOAL!", "OWN GOAL!", "5s KEEPER FOUL!", "5s KEEPER HOLD FOUL!", "5s UNCLAIMED BOX BALL!", "HALFTIME!", "FULLTIME!")
            ]

            # Trigger Full-Time cinematic at match conclusion on final frame if not already present
            if not trans_events and i == len(raw_frames) - 1 and float(curr_fr.get("time", 0.0)) >= 179.0:
                trans_events = [{"type": "FULLTIME!", "team": -1, "agent": "referee"}]

            if not trans_events:
                expanded_frames.append(curr_fr)
                continue

            prev_fr = raw_frames[i - 1]
            # Prioritize GOAL! / OWN GOAL! over HALFTIME! if both occur on frame 5400
            trans_ev = next(
                (ev for ev in trans_events if ev.get("type") in ("GOAL!", "OWN GOAL!")),
                trans_events[0],
            )
            ev_type = str(trans_ev.get("type", ""))
            ev_team = int(trans_ev.get("team", -1))
            ev_agent = str(trans_ev.get("agent", ""))

            if ev_type in ("GOAL!", "OWN GOAL!"):
                freeze_frames = 90
                walk_frames = 120
                hold_frames = 60
            elif ev_type == "FULLTIME!":
                freeze_frames = 150
                walk_frames = 0
                hold_frames = 0
            else:
                freeze_frames = 60
                walk_frames = 120
                hold_frames = 60

            # Determine Stage 1 end ball position (resting inside net for goals)
            bx0 = float(prev_fr["ball"][0])
            by0 = float(prev_fr["ball"][1])
            b_col = prev_fr["ball"][2] if len(prev_fr["ball"]) > 2 else -1

            if ev_type in ("GOAL!", "OWN GOAL!"):
                ev_x = float(trans_ev.get("x", bx0))
                ev_y = float(trans_ev.get("y", by0))
                bx_net = -14.0 if (ev_x < 700.0 or bx0 < 700.0) else (self.field_w + 14.0)
                by_net = ev_y
            else:
                bx_net = bx0
                by_net = by0

            stage1_end_ball = [bx_net, by_net, b_col]

            # Stage 1: Freeze & Celebration Hold
            for k in range(freeze_frames):
                if ev_type in ("GOAL!", "OWN GOAL!"):
                    t_ball = min(1.0, k / 20.0)
                    bx = bx0 + (bx_net - bx0) * t_ball
                    by = by0 + (by_net - by0) * t_ball
                else:
                    bx = bx0
                    by = by0

                p_list = [[p[0], p[1], p[2], 0] for p in prev_fr.get("players", [])]
                synth_fr = {
                    "time": prev_fr.get("time", 0.0),
                    "half": curr_fr.get("half", prev_fr.get("half", 1)),
                    "stats": curr_fr.get("stats", prev_fr.get("stats", {})),
                    "team_side": prev_fr.get("team_side", [0, 1]),
                    "events": curr_fr.get("events", []) if k == 0 else [],
                    "player_rewards": [0.0] * len(p_list),
                    "cum_rewards": curr_fr.get("cum_rewards", prev_fr.get("cum_rewards", [])),
                    "players": p_list,
                    "ball": [round(float(bx), 2), round(float(by), 2), b_col],
                    "cinematic": {
                        "phase": "FREEZE",
                        "event_type": ev_type,
                        "team": ev_team,
                        "agent": ev_agent,
                        "progress": k / float(freeze_frames),
                    },
                }
                expanded_frames.append(synth_fr)

            # Stage 2: Walk to Set-Piece Positions
            b_curr = curr_fr.get("ball", [700.0, 420.0, -1])
            bx_end = float(b_curr[0])
            by_end = float(b_curr[1])
            b_col_end = b_curr[2] if len(b_curr) > 2 else -1

            num_p = min(len(prev_fr.get("players", [])), len(curr_fr.get("players", [])))

            for k in range(walk_frames):
                u = (k + 1) / float(walk_frames)
                s = u * u * (3.0 - 2.0 * u)

                p_list = []
                for p_idx in range(num_p):
                    p0 = prev_fr["players"][p_idx]
                    p1 = curr_fr["players"][p_idx]
                    x0, y0, a0 = float(p0[0]), float(p0[1]), float(p0[2])
                    x1, y1, a1 = float(p1[0]), float(p1[1]), float(p1[2])

                    wx = x0 + (x1 - x0) * s
                    wy = y0 + (y1 - y0) * s
                    dist_to_walk = math.hypot(x1 - x0, y1 - y0)

                    if dist_to_walk > 8.0 and u < 0.82:
                        walk_angle = math.atan2(y1 - y0, x1 - x0)
                    else:
                        diff = math.atan2(math.sin(a1 - a0), math.cos(a1 - a0))
                        blend_u = (u - 0.82) / 0.18 if dist_to_walk > 8.0 else u
                        walk_angle = a0 + diff * blend_u

                    p_list.append([round(float(wx), 2), round(float(wy), 2), round(float(walk_angle), 4), 0])

                bx = stage1_end_ball[0] + (bx_end - stage1_end_ball[0]) * s
                by = stage1_end_ball[1] + (by_end - stage1_end_ball[1]) * s

                synth_fr = {
                    "time": prev_fr.get("time", 0.0),
                    "half": curr_fr.get("half", 1),
                    "stats": curr_fr.get("stats", {}),
                    "team_side": curr_fr.get("team_side", prev_fr.get("team_side", [0, 1])),
                    "events": [],
                    "player_rewards": [0.0] * len(p_list),
                    "cum_rewards": curr_fr.get("cum_rewards", []),
                    "players": p_list,
                    "ball": [round(float(bx), 2), round(float(by), 2), b_col_end],
                    "cinematic": {
                        "phase": "WALK",
                        "event_type": ev_type,
                        "team": ev_team,
                        "agent": ev_agent,
                        "progress": u,
                    },
                }
                expanded_frames.append(synth_fr)

            # Stage 3: Arbiter Whistle Hold
            for k in range(hold_frames):
                p_list = [[p[0], p[1], p[2], 0] for p in curr_fr.get("players", [])]
                synth_fr = {
                    "time": prev_fr.get("time", 0.0),
                    "half": curr_fr.get("half", 1),
                    "stats": curr_fr.get("stats", {}),
                    "team_side": curr_fr.get("team_side", [0, 1]),
                    "events": [],
                    "player_rewards": [0.0] * len(p_list),
                    "cum_rewards": curr_fr.get("cum_rewards", []),
                    "players": p_list,
                    "ball": list(curr_fr.get("ball", [])),
                    "cinematic": {
                        "phase": "WHISTLE_HOLD",
                        "event_type": ev_type,
                        "team": ev_team,
                        "agent": ev_agent,
                        "progress": k / float(hold_frames),
                    },
                }
                expanded_frames.append(synth_fr)

            # Finally, append curr_fr with events cleared
            curr_fr_copy = dict(curr_fr)
            curr_fr_copy["events"] = []
            expanded_frames.append(curr_fr_copy)

        return expanded_frames

    def _draw_cinematic_banner(self, frame: Dict[str, Any]) -> None:
        """Render broadcast-style banner overlays for cinematic transitions."""
        cin = frame.get("cinematic")
        if not cin:
            return

        phase = cin.get("phase")
        ev_type = cin.get("event_type")
        team = cin.get("team", -1)
        agent = cin.get("agent", "")
        progress = float(cin.get("progress", 0.0))

        stats = frame.get("stats", {})
        score = stats.get("score", [0, 0])

        cx = self.win_w // 2

        if phase == "FREEZE":
            banner_w = min(self.win_w - 40, 840)
            banner_h = 76
            by = self.hud_height + (self.disp_field_h - banner_h) // 2
            bx = cx - banner_w // 2

            banner_surf = pygame.Surface((banner_w, banner_h), pygame.SRCALPHA)
            banner_surf.fill((14, 18, 24, 215))

            accent_color = COLOR_KEEPER_GOLD
            if team == 0:
                accent_color = COLOR_TEAM_A
            elif team == 1:
                accent_color = COLOR_TEAM_B

            title_str = ""
            sub_str = ""
            goal_net_x = None

            if ev_type == "GOAL!":
                team_name = "TEAM A" if team == 0 else "TEAM B"
                title_str = f"GOAL!  —  {team_name} SCORES!"
                sub_str = f"Scored by {agent}   |   Score: {score[0]} - {score[1]}"
                team_side = frame.get("team_side", [0, 1])
                enemy_side = 1 - team_side[team] if team in (0, 1) else 1
                goal_net_x = 0.0 if enemy_side == 0 else self.field_w
            elif ev_type == "OWN GOAL!":
                title_str = "OWN GOAL!"
                sub_str = f"Touched by {agent}   |   Score: {score[0]} - {score[1]}"
                accent_color = EVENT_COLORS.get("OWN GOAL!", (255, 95, 65))
                team_side = frame.get("team_side", [0, 1])
                own_side = team_side[team] if team in (0, 1) else 0
                goal_net_x = 0.0 if own_side == 0 else self.field_w
            elif ev_type in ("5s KEEPER FOUL!", "5s KEEPER HOLD FOUL!"):
                title_str = "5s KEEPER HOLD VIOLATION  —  TURNOVER FREE KICK!"
                sub_str = "Goalkeeper held ball > 5.0s  |  Awarded to attacking team"
                accent_color = EVENT_COLORS.get("5s KEEPER HOLD FOUL!", (255, 75, 150))
            elif ev_type == "5s UNCLAIMED BOX BALL!":
                title_str = "5s UNCLAIMED BOX BALL  —  ATTACKING OVERLOAD FREE KICK!"
                sub_str = "Loose ball unclaimed in keeper arc > 5.0s  |  Turnover restart"
                accent_color = EVENT_COLORS.get("5s UNCLAIMED BOX BALL!", (255, 110, 180))
            elif ev_type == "HALFTIME!":
                title_str = "HALF-TIME  —  SWITCHING SIDES"
                sub_str = f"Second Half Kickoff   |   Score: {score[0]} - {score[1]}"
                accent_color = EVENT_COLORS.get("HALFTIME!", (200, 220, 255))
            elif ev_type == "FULLTIME!":
                title_str = "FULL-TIME  —  FINAL WHISTLE!"
                score_a, score_b = int(score[0]), int(score[1])
                shots = stats.get("shots", [0, 0])
                passes = stats.get("passes", [0, 0])
                saves = stats.get("saves", [0, 0])
                if score_a > score_b:
                    sub_str = f"TEAM A WINS!  ({score_a} - {score_b})   |   Shots: {shots[0]}-{shots[1]}   Passes: {passes[0]}-{passes[1]}   Saves: {saves[0]}-{saves[1]}"
                    accent_color = COLOR_TEAM_A
                elif score_b > score_a:
                    sub_str = f"TEAM B WINS!  ({score_b} - {score_a})   |   Shots: {shots[1]}-{shots[0]}   Passes: {passes[1]}-{passes[0]}   Saves: {saves[1]}-{saves[0]}"
                    accent_color = COLOR_TEAM_B
                else:
                    sub_str = f"HONORS EVEN  —  DRAW!  ({score_a} - {score_b})   |   Shots: {shots[0]}-{shots[1]}   Passes: {passes[0]}-{passes[1]}"
                    accent_color = COLOR_KEEPER_GOLD

            # Banner border
            pygame.draw.rect(
                banner_surf,
                (*accent_color[:3], 240),
                pygame.Rect(0, 0, banner_w, banner_h),
                width=2,
                border_radius=8,
            )

            # Pulsing golden/team-colored highlight glow around the scored goal net
            if goal_net_x is not None:
                pulse = 0.5 + 0.5 * math.sin(progress * math.pi * 6)
                alpha_glow = int(120 + 80 * pulse)
                gx, gy = self.world_to_screen(goal_net_x, (self.goal_y_min + self.goal_y_max) / 2.0)
                gh = int(round((self.goal_y_max - self.goal_y_min) * self.scale))
                gw = int(round(40 * self.scale))
                net_rect = pygame.Rect(gx - gw if goal_net_x == 0.0 else gx, gy - gh // 2, gw, gh)
                glow_surf = pygame.Surface((gw + 16, gh + 16), pygame.SRCALPHA)
                pygame.draw.rect(
                    glow_surf,
                    (*accent_color[:3], alpha_glow),
                    pygame.Rect(0, 0, gw + 16, gh + 16),
                    width=4,
                    border_radius=4,
                )
                self.screen.blit(glow_surf, (net_rect.x - 8, net_rect.y - 8))

            t_surf = self.font_title.render(title_str, True, accent_color)
            s_surf = self.font_hud.render(sub_str, True, COLOR_TEXT_MAIN)
            banner_surf.blit(t_surf, ((banner_w - t_surf.get_width()) // 2, 14))
            banner_surf.blit(s_surf, ((banner_w - s_surf.get_width()) // 2, 44))
            self.screen.blit(banner_surf, (bx, by))

        elif phase in ("WALK", "WHISTLE_HOLD"):
            pill_w = 460
            pill_h = 36
            py = self.hud_height + 12
            px = cx - pill_w // 2

            pill_surf = pygame.Surface((pill_w, pill_h), pygame.SRCALPHA)
            pill_surf.fill((14, 18, 24, 210))

            if phase == "WALK":
                text_str = "PLAYERS TAKING SET-PIECE POSITIONS..."
                text_col = COLOR_TEXT_MAIN
                border_col = (100, 120, 150)
            else:
                if progress < 0.65:
                    text_str = "WAITING FOR REFEREE WHISTLE..."
                    text_col = COLOR_KEEPER_GOLD
                    border_col = COLOR_KEEPER_GOLD
                else:
                    text_str = "WHISTLE!  PLAY ON!"
                    text_col = (70, 255, 120)
                    border_col = (70, 255, 120)

            pygame.draw.rect(
                pill_surf,
                (*border_col[:3], 230),
                pygame.Rect(0, 0, pill_w, pill_h),
                width=2,
                border_radius=18,
            )
            txt = self.font_hud.render(text_str, True, text_col)
            pill_surf.blit(txt, ((pill_w - txt.get_width()) // 2, (pill_h - txt.get_height()) // 2))
            self.screen.blit(pill_surf, (px, py))

    # ==========================================================================
    # HUD Scoreboard & Scrubber Bar
    # ==========================================================================
    def _draw_top_hud(self, frame: Dict[str, Any]) -> None:
        """Draw top scoreboard with timer, score, possession %, passes, tackles, saves, shots."""
        pygame.draw.rect(
            self.screen, COLOR_HUD_BG, pygame.Rect(0, 0, self.win_w, self.hud_height)
        )
        pygame.draw.line(
            self.screen,
            COLOR_HUD_BORDER,
            (0, self.hud_height - 2),
            (self.win_w, self.hud_height - 2),
            width=2,
        )

        stats = frame.get("stats", {})
        score = stats.get("score", [0, 0])
        poss = stats.get("possession_pct", [50.0, 50.0])
        passes = stats.get("passes", [0, 0])
        tackles = stats.get("tackles", [0, 0])
        saves = stats.get("saves", [0, 0])
        shots = stats.get("shots", [0, 0])
        team_side = frame.get("team_side", [0, 1])

        t_sec = float(frame.get("time", 0.0))
        mins = int(t_sec // 60)
        secs = t_sec % 60.0
        half_num = int(frame.get("half", 1))
        half_str = "1ST HALF" if half_num == 1 else "2ND HALF (SIDES SWAPPED)"

        # Center Score & Match Timer
        cx = self.win_w // 2
        score_text = f"{score[0]}   -   {score[1]}"
        score_surf = self.font_score.render(score_text, True, COLOR_TEXT_MAIN)
        self.screen.blit(score_surf, (cx - score_surf.get_width() // 2, 8))

        timer_str = f"{mins:02d}:{secs:04.1f}  |  {half_str}"
        timer_surf = self.font_hud.render(timer_str, True, COLOR_KEEPER_GOLD)
        self.screen.blit(timer_surf, (cx - timer_surf.get_width() // 2, 46))

        # Team A Left Panel
        side_a_tag = "LEFT GOAL" if team_side[0] == 0 else "RIGHT GOAL"
        team_a_hdr = self.font_title.render(f"TEAM A ({side_a_tag})", True, COLOR_TEAM_A)
        self.screen.blit(team_a_hdr, (self.margin_x, 8))
        stats_a_str = (
            f"Poss: {poss[0]:4.1f}%  |  Pass: {passes[0]}  |  "
            f"Tkl: {tackles[0]}  |  Save: {saves[0]}  |  Shot: {shots[0]}"
        )
        self.screen.blit(
            self.font_small.render(stats_a_str, True, COLOR_TEXT_MAIN),
            (self.margin_x, 42),
        )

        # Team B Right Panel
        side_b_tag = "LEFT GOAL" if team_side[1] == 0 else "RIGHT GOAL"
        team_b_hdr = self.font_title.render(f"TEAM B ({side_b_tag})", True, COLOR_TEAM_B)
        self.screen.blit(
            team_b_hdr,
            (self.win_w - self.margin_x - team_b_hdr.get_width(), 8),
        )
        stats_b_str = (
            f"Poss: {poss[1]:4.1f}%  |  Pass: {passes[1]}  |  "
            f"Tkl: {tackles[1]}  |  Save: {saves[1]}  |  Shot: {shots[1]}"
        )
        sb_surf = self.font_small.render(stats_b_str, True, COLOR_TEXT_MAIN)
        self.screen.blit(
            sb_surf,
            (self.win_w - self.margin_x - sb_surf.get_width(), 42),
        )

        # Controls & Status Strip
        role_legend = "Roles: GK | D1-D2 (Def) | M1-M2 (Mid) | ST (Atk)"
        if self.playback_speed < 1.0:
            speed_str = f"{self.playback_speed:g}x (SLOWMO)"
        else:
            speed_str = f"{int(self.playback_speed)}x"

        if not self.show_vision_cones:
            cone_str = "OFF"
        elif not self.selected_cone_players:
            cone_str = "ALL"
        elif len(self.selected_cone_players) == 1:
            sel_idx = next(iter(self.selected_cone_players))
            cone_str = self._get_player_label(sel_idx)
        else:
            cone_str = f"{len(self.selected_cone_players)} Selected"

        status_str = (
            f"[SPACE] {'PAUSED' if self.paused else 'PLAYING'}   "
            f"[0/1/2/4/S] Speed: {speed_str}   "
            f"[V/TAB/Click] Cones: {cone_str}   "
            f"[E] Rewards: {'ON' if self.show_reward_hud else 'OFF'}   "
            f"[LEFT/RIGHT] Scrub   [R] Restart"
        )
        ctrl_surf = self.font_small.render(status_str, True, COLOR_TEXT_MUTED)
        self.screen.blit(ctrl_surf, (cx - ctrl_surf.get_width() // 2, 70))

    def _draw_scrubber_bar(self, frame: Dict[str, Any]) -> None:
        """Draw bottom timeline scrubber bar with halftime tick, goal markers, and save markers."""
        bar_x = self.margin_x
        bar_w = self.disp_field_w
        bar_y = self.hud_height + self.disp_field_h + 16
        bar_h = 10

        pygame.draw.rect(
            self.screen,
            (40, 48, 60),
            pygame.Rect(bar_x, bar_y, bar_w, bar_h),
            border_radius=5,
        )

        progress = self.current_frame_idx / float(max(1, self.num_frames - 1))
        fill_w = int(bar_w * progress)
        pygame.draw.rect(
            self.screen,
            COLOR_KEEPER_GOLD,
            pygame.Rect(bar_x, bar_y, fill_w, bar_h),
            border_radius=5,
        )

        # Halftime marker at 50%
        ht_x = bar_x + bar_w // 2
        pygame.draw.line(
            self.screen, COLOR_LINE, (ht_x, bar_y - 4), (ht_x, bar_y + bar_h + 4), width=2
        )

        # 1. High-reward (+5 or up) other event markers along timeline (triangle shaped: ▲)
        for h_frame, h_team, h_label, h_rew in self.high_reward_frames:
            hx = bar_x + int((h_frame / float(max(1, self.num_frames - 1))) * bar_w)
            hy = bar_y + bar_h // 2
            h_col = COLOR_TEAM_A if h_team == 0 else (COLOR_TEAM_B if h_team == 1 else (75, 255, 120))
            tri_pts = [
                (hx, hy - 6),
                (hx + 5, hy + 5),
                (hx - 5, hy + 5),
            ]
            pygame.draw.polygon(self.screen, h_col, tri_pts)
            ring_col = (255, 215, 60)  # Bright gold outline for +5 highlights
            pygame.draw.polygon(self.screen, ring_col, tri_pts, width=1)

        # 2. Foul / violation markers along timeline (card shaped: ▮)
        for f_frame, f_team, f_type in self.foul_frames:
            fx = bar_x + int((f_frame / float(max(1, self.num_frames - 1))) * bar_w)
            fy = bar_y + bar_h // 2
            f_col = COLOR_TEAM_A if f_team == 0 else (COLOR_TEAM_B if f_team == 1 else (255, 80, 80))
            card_rect = pygame.Rect(fx - 3, fy - 6, 7, 12)
            pygame.draw.rect(self.screen, f_col, card_rect, border_radius=1)
            pygame.draw.rect(self.screen, (255, 170, 40), card_rect, width=1, border_radius=1)

        # 3. Save markers along timeline (diamond shaped: ◆)
        for s_frame, s_team, is_lunge in self.save_frames:
            sx = bar_x + int((s_frame / float(max(1, self.num_frames - 1))) * bar_w)
            sy = bar_y + bar_h // 2
            s_col = COLOR_TEAM_A if s_team == 0 else COLOR_TEAM_B
            diamond_pts = [
                (sx, sy - 6),
                (sx + 5, sy),
                (sx, sy + 6),
                (sx - 5, sy),
            ]
            pygame.draw.polygon(self.screen, s_col, diamond_pts)
            ring_col = (80, 245, 200) if is_lunge else COLOR_KEEPER_GOLD
            pygame.draw.polygon(self.screen, ring_col, diamond_pts, width=1)

        # 4. Goal markers along timeline (circular: ●)
        for g_frame, g_team, is_og in self.goal_frames:
            gx = bar_x + int((g_frame / float(max(1, self.num_frames - 1))) * bar_w)
            g_col = COLOR_TEAM_A if g_team == 0 else COLOR_TEAM_B
            pygame.draw.circle(self.screen, g_col, (gx, bar_y + bar_h // 2), 6)
            ring_col = (255, 120, 50) if is_og else COLOR_LINE
            ring_w = 2 if is_og else 1
            pygame.draw.circle(self.screen, ring_col, (gx, bar_y + bar_h // 2), 6, width=ring_w)

        # Scrubber handle
        knob_x = bar_x + fill_w
        pygame.draw.circle(self.screen, COLOR_TEXT_MAIN, (knob_x, bar_y + bar_h // 2), 8)

        # Frame counter label (left)
        frame_lbl = self.font_small.render(
            f"Frame {self.current_frame_idx} / {self.num_frames - 1}",
            True,
            COLOR_TEXT_MUTED,
        )
        self.screen.blit(frame_lbl, (bar_x, bar_y + 14))

        # Bottom Bar Legend (right)
        legend_x = bar_x + bar_w - 335
        legend_y = bar_y + 14

        # Circle icon for Goal
        pygame.draw.circle(self.screen, COLOR_LINE, (legend_x + 4, legend_y + 6), 4)
        pygame.draw.circle(self.screen, COLOR_KEEPER_GOLD, (legend_x + 4, legend_y + 6), 4, width=1)
        goal_leg_surf = self.font_small.render("Goal", True, COLOR_TEXT_MUTED)
        self.screen.blit(goal_leg_surf, (legend_x + 11, legend_y))

        # Diamond icon for Save
        d_leg_x = legend_x + 65
        d_pts = [
            (d_leg_x + 4, legend_y + 2),
            (d_leg_x + 8, legend_y + 6),
            (d_leg_x + 4, legend_y + 10),
            (d_leg_x, legend_y + 6),
        ]
        pygame.draw.polygon(self.screen, (80, 245, 200), d_pts)
        save_leg_surf = self.font_small.render("Save", True, COLOR_TEXT_MUTED)
        self.screen.blit(save_leg_surf, (d_leg_x + 11, legend_y))

        # Triangle icon for +5 Event
        t_leg_x = legend_x + 130
        t_pts = [
            (t_leg_x + 4, legend_y + 2),
            (t_leg_x + 8, legend_y + 10),
            (t_leg_x, legend_y + 10),
        ]
        pygame.draw.polygon(self.screen, (255, 215, 60), t_pts)
        plus5_leg_surf = self.font_small.render("+5 Event", True, COLOR_TEXT_MUTED)
        self.screen.blit(plus5_leg_surf, (t_leg_x + 11, legend_y))

        # Card icon for Foul
        f_leg_x = legend_x + 220
        card_leg_rect = pygame.Rect(f_leg_x + 2, legend_y + 1, 6, 10)
        pygame.draw.rect(self.screen, (255, 80, 80), card_leg_rect, border_radius=1)
        pygame.draw.rect(self.screen, (255, 170, 40), card_leg_rect, width=1, border_radius=1)
        foul_leg_surf = self.font_small.render("Foul", True, COLOR_TEXT_MUTED)
        self.screen.blit(foul_leg_surf, (f_leg_x + 12, legend_y))

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
    viewer = ReplayViewer(
        replay_data,
        headless=args.headless_frames is not None,
        show_vision_cones=args.show_cones,
        show_reward_hud=args.show_rewards,
    )
    viewer.run(max_frames=args.headless_frames)


if __name__ == "__main__":
    main()
