"""Replay viewer heads-up display: event popups, the cinematic banner and the top score bar."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Tuple

import pygame

from football_rl.viewer.theme import (
    COLOR_HUD_BG,
    COLOR_HUD_BORDER,
    COLOR_KEEPER_GOLD,
    COLOR_TEAM_A,
    COLOR_TEAM_B,
    COLOR_TEXT_MAIN,
    COLOR_TEXT_MUTED,
    EVENT_COLORS,
)


@dataclass
class FloatingPopup:
    """Rising and fading floating text spawned at world event coordinates."""

    text: str
    world_x: float
    world_y: float
    color: Tuple[int, int, int]
    spawn_frame: int
    duration_frames: int = 75  # 1.25s at 60 FPS


class HudMixin:
    """Event popups, banner and top HUD of `ReplayViewer`."""

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

        cx = self.margin_x + self.disp_field_w // 2

        if phase == "FREEZE":
            banner_w = min(self.disp_field_w - 40, 840)
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
                sub_str = "LIVE FLANK TURNOVER — Keeper reset to goal line, ball awarded wide"
                accent_color = EVENT_COLORS.get("5s KEEPER HOLD FOUL!", (255, 75, 150))
            elif ev_type == "5s UNCLAIMED BOX BALL!":
                title_str = "5s UNCLAIMED BOX BALL  —  ATTACKING OVERLOAD FREE KICK!"
                sub_str = "LIVE FLANK TURNOVER — Keeper reset to goal line, ball awarded wide"
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

        # Center Score & Match Timer over pitch
        cx = self.margin_x + self.disp_field_w // 2
        score_text = f"{score[0]}   -   {score[1]}"
        score_surf = self.font_score.render(score_text, True, COLOR_TEXT_MAIN)
        self.screen.blit(score_surf, (cx - score_surf.get_width() // 2, 4))

        timer_str = f"{mins:02d}:{secs:04.1f}  |  {half_str}"
        timer_surf = self.font_hud.render(timer_str, True, COLOR_KEEPER_GOLD)
        self.screen.blit(timer_surf, (cx - timer_surf.get_width() // 2, 36))

        # True Possession Center Sub-Header
        true_poss = stats.get("true_possession_pct")
        if true_poss and len(true_poss) >= 3:
            red_p, blue_p, loose_p = float(true_poss[0]), float(true_poss[1]), float(true_poss[2])
        else:
            red_p = float(poss[0]) if len(poss) > 0 else 50.0
            blue_p = float(poss[1]) if len(poss) > 1 else 50.0
            loose_p = max(0.0, 100.0 - red_p - blue_p)
        stat_frames = int(stats.get("stationary_ball_frames", 0))
        stall_sec = stat_frames / float(max(1, self.physics_hz))
        poss_sub_str = f"Poss: Red {red_p:.0f}% | Blue {blue_p:.0f}% | Loose {loose_p:.0f}% (Stall: {stall_sec:.1f}s)"
        poss_sub_surf = self.font_small.render(poss_sub_str, True, (215, 225, 240))
        self.screen.blit(poss_sub_surf, (cx - poss_sub_surf.get_width() // 2, 54))

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

        # Team B Right Panel (aligned to pitch right edge)
        right_edge = self.margin_x + self.disp_field_w
        side_b_tag = "LEFT GOAL" if team_side[1] == 0 else "RIGHT GOAL"
        team_b_hdr = self.font_title.render(f"TEAM B ({side_b_tag})", True, COLOR_TEAM_B)
        self.screen.blit(
            team_b_hdr,
            (right_edge - team_b_hdr.get_width(), 8),
        )
        stats_b_str = (
            f"Poss: {poss[1]:4.1f}%  |  Pass: {passes[1]}  |  "
            f"Tkl: {tackles[1]}  |  Save: {saves[1]}  |  Shot: {shots[1]}"
        )
        sb_surf = self.font_small.render(stats_b_str, True, COLOR_TEXT_MAIN)
        self.screen.blit(
            sb_surf,
            (right_edge - sb_surf.get_width(), 42),
        )

        # Controls & Status Strip
        role_legend = "Roles: GK | D1-D2 (Def) | M1-M2 (Mid) | ST (Atk)"
        if self.playback_speed < 1.0:
            speed_str = f"{self.playback_speed:g}x (SLOWMO)"
        elif math.isclose(self.playback_speed, int(self.playback_speed), abs_tol=1e-3):
            speed_str = f"{int(self.playback_speed)}x"
        else:
            speed_str = f"{self.playback_speed:g}x"

        if not self.show_vision_cones:
            cone_str = "OFF"
        elif not self.selected_cone_players:
            cone_str = "ALL"
        elif len(self.selected_cone_players) == 1:
            sel_idx = next(iter(self.selected_cone_players))
            cone_str = self._get_player_label(sel_idx)
        else:
            cone_str = f"{len(self.selected_cone_players)} Sel"

        left_part = f"[SPACE] {'PAUSED' if self.paused else 'PLAYING'}   "
        speed_part = f"[UP/DOWN] Speed: {speed_str}"
        right_part = (
            f"   [F] FollowBall:{'ON' if self.auto_focus_carrier else 'OFF'}   "
            f"[V] Cones:{cone_str}   "
            f"[P] Pass:{'ON' if self.show_pass_lanes else 'OFF'}   "
            f"[W] Walls:{'ON' if self.show_wall_tracers else 'OFF'}   "
            f"[I] Radar:{'ON' if self.show_radar else 'OFF'}   "
            f"[C] Critic:{'ON' if self.show_critic else 'OFF'}   "
            f"[G] Grad:{'ON' if self.show_gradient else 'OFF'}   "
            f"[T] Telemetry:{'ON' if self.show_telemetry else 'OFF'}   "
            f"[E] Rew:{'ON' if self.show_reward_hud else 'OFF'}"
        )
        status_str = left_part + speed_part + right_part
        ctrl_surf = self.font_small.render(status_str, True, COLOR_TEXT_MUTED)
        ctrl_x = cx - ctrl_surf.get_width() // 2
        ctrl_y = 74
        self.screen.blit(ctrl_surf, (ctrl_x, ctrl_y))
