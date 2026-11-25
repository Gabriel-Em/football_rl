"""Replay viewer pitch layer: turf, lines, goals, vision cones, players, lunge trails and the ball."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import pygame

from football_rl.viewer.theme import (
    COLOR_BALL_NEUTRAL,
    COLOR_GHOST_RING,
    COLOR_KEEPER_GOLD,
    COLOR_LINE,
    COLOR_ROLE_ATK,
    COLOR_ROLE_DEF,
    COLOR_ROLE_MID,
    COLOR_STUN_ALERT,
    COLOR_TEAM_A,
    COLOR_TEAM_A_DARK,
    COLOR_TEAM_B,
    COLOR_TEAM_B_DARK,
    COLOR_TEXT_MAIN,
    COLOR_TURF_DARK,
    COLOR_TURF_LIGHT,
    FLAG_HAS_BALL,
    FLAG_IS_GHOST,
    FLAG_IS_LUNGING,
    FLAG_IS_STUNNED,
    _get_critic_color,
)


class PitchMixin:
    """Pitch, player and ball drawing of `ReplayViewer`."""

    def _draw_cursor(self) -> None:
        """Draw a high-contrast software arrow cursor to ensure mouse visibility in WSLg/X11."""
        if not pygame.mouse.get_focused():
            return
        mx, my = pygame.mouse.get_pos()
        if not (0 <= mx < self.win_w and 0 <= my < self.win_h):
            return

        # Check if hovering over scrubber bar or an interactive player
        is_hovering_player = (
            self._is_pos_on_field(mx, my) and self._find_player_at_screen_pos(mx, my) is not None
        )
        bar_x = self.margin_x
        bar_w = self.disp_field_w
        bar_y = self.hud_height + self.disp_field_h + 12
        is_hovering_scrubber = bar_x <= mx <= bar_x + bar_w and bar_y - 10 <= my <= bar_y + 26

        if is_hovering_player or is_hovering_scrubber:
            accent_col = (80, 220, 255) if is_hovering_player else (255, 200, 60)
            pygame.draw.circle(self.screen, (15, 23, 42), (mx, my), 6)
            pygame.draw.circle(self.screen, accent_col, (mx, my), 5, width=2)
            pygame.draw.circle(self.screen, (255, 255, 255), (mx, my), 2)
        else:
            pts = [
                (mx, my),
                (mx, my + 16),
                (mx + 4, my + 12),
                (mx + 8, my + 20),
                (mx + 11, my + 18),
                (mx + 7, my + 11),
                (mx + 13, my + 11),
            ]
            shadow_pts = [(x + 1, y + 1) for x, y in pts]
            pygame.draw.polygon(self.screen, (15, 23, 42), shadow_pts)
            pygame.draw.polygon(self.screen, (245, 245, 250), pts)
            pygame.draw.polygon(self.screen, (15, 23, 42), pts, width=1)

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

            # 1b. Underfoot Pocket Marking Pressure Halo (Toggleable via 'H')
            if getattr(self, "show_pocket_halos", True):
                phys = self._get_player_physics(self.current_frame_idx, idx)
                pkt = phys["pocket"]
                halo_surf = pygame.Surface((r * 2 + 18, r * 2 + 18), pygame.SRCALPHA)
                if pkt >= 0.65:
                    halo_col = (255, 60, 75, 110)
                elif pkt >= 0.35:
                    halo_col = (245, 195, 45, 90)
                else:
                    halo_col = (35, 230, 160, 75)
                pygame.draw.circle(halo_surf, halo_col, (r + 9, r + 9), r + 4)
                pygame.draw.circle(
                    halo_surf,
                    (*halo_col[:3], min(255, halo_col[3] + 40)),
                    (r + 9, r + 9),
                    r + 4,
                    width=2,
                )
                self.screen.blit(halo_surf, (sx - r - 9, sy - r - 9))

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

            # 2b. Critic Value Overlay & Indicator Arc ([C] Critic:ON)
            if self.show_critic:
                v_preds = self.frames[self.current_frame_idx].get("v_pred")
                if v_preds and idx < len(v_preds) and v_preds[idx] is not None:
                    v_val = float(v_preds[idx])
                    v_col = _get_critic_color(v_val)
                    pygame.draw.circle(self.screen, v_col, (sx, sy), r + 5, width=2)
                    v_lbl = self.font_small.render(f"V:{v_val:+.1f}", True, v_col)
                    v_y = sy + r + (18 if self.show_reward_hud else 8)
                    self.screen.blit(v_lbl, (sx - v_lbl.get_width() // 2, v_y))

            # 3. Possession Halo
            if flags & FLAG_HAS_BALL:
                pygame.draw.circle(self.screen, COLOR_KEEPER_GOLD, (sx, sy), r + 5, width=2)

            # 3b. Cone Selection Halo (Cyan ring when this player's vision cone is explicitly isolated)
            if self.show_vision_cones and idx in self.selected_cone_players:
                pygame.draw.circle(self.screen, (80, 245, 200), (sx, sy), r + 7, width=2)

            # 3c. Telemetry Hover Focus Ring
            if getattr(self, "hovered_player_idx", None) == idx:
                pygame.draw.circle(self.screen, (130, 220, 255), (sx, sy), r + 8, width=1)

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

    def _get_player_role_short(self, idx: int, team_size: int = 6) -> str:
        """Return compact role tag (GK, D1, D2, M1, M2, ST)."""
        local_idx = idx % team_size
        if team_size == 6:
            roles = ["GK", "D1", "D2", "M1", "M2", "ST"]
            return roles[local_idx] if local_idx < len(roles) else f"P{local_idx}"
        return "GK" if local_idx == 0 else f"P{local_idx}"

    def _get_player_role_accent(self, idx: int, team_size: int = 6) -> Tuple[int, int, int]:
        """Return theme color for player role."""
        local_idx = idx % team_size
        if team_size == 6:
            if local_idx == 0:
                return COLOR_KEEPER_GOLD
            elif local_idx in (1, 2):
                return COLOR_ROLE_DEF
            elif local_idx in (3, 4):
                return COLOR_ROLE_MID
            else:
                return COLOR_ROLE_ATK
        return COLOR_KEEPER_GOLD if local_idx == 0 else COLOR_LINE
