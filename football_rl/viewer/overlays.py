"""Replay viewer tactical overlays: pass lanes, wall-bank tracers, peripheral radar, role tethers and potential gradients."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame

from football_rl.config import DEFAULT_CONFIG
from football_rl.potentials import TACTICAL_DEPTH_BIAS, TACTICAL_FLANK_BIAS, PotentialState, potential_gradient
from football_rl.viewer.theme import COLOR_TEAM_A, COLOR_TEAM_B, FLAG_HAS_BALL


def _draw_dashed_line(
    surf: pygame.Surface,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    color: Tuple[int, ...],
    dash_len: float = 6.0,
    space_len: float = 4.0,
    width: int = 1,
) -> None:
    """Draw a straight dashed line between two pixel coordinates."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 1e-4:
        return
    ux = dx / dist
    uy = dy / dist
    step = dash_len + space_len
    num_steps = int(dist // step)
    for i in range(num_steps + 1):
        s_dist = i * step
        e_dist = min(dist, s_dist + dash_len)
        if s_dist >= dist:
            break
        sx = int(round(x1 + ux * s_dist))
        sy = int(round(y1 + uy * s_dist))
        ex = int(round(x1 + ux * e_dist))
        ey = int(round(y1 + uy * e_dist))
        pygame.draw.line(surf, color, (sx, sy), (ex, ey), width)


class OverlaysMixin:
    """Tactical overlays of `ReplayViewer`."""

    def _draw_pass_lanes(self, frame: Dict[str, Any]) -> None:
        """Render dynamic passing lane vectors and unpressed lead targets from ball carrier."""
        players = frame.get("players", [])
        num_players = len(players)
        if num_players == 0:
            return
        team_size = max(1, num_players // 2)

        # Locate ball carrier
        carrier_idx = next((i for i, p in enumerate(players) if int(p[3]) & FLAG_HAS_BALL), None)
        if carrier_idx is None:
            return

        carrier_phys = self._get_player_physics(self.current_frame_idx, carrier_idx)
        c_x, c_y = carrier_phys["pos"]
        s_cx, s_cy = self.world_to_screen(c_x, c_y)
        is_carrier_team_a = carrier_idx < team_size

        opponents = [
            (float(p[0]), float(p[1]))
            for o_idx, p in enumerate(players)
            if (o_idx < team_size) != is_carrier_team_a
        ]

        for j in range(num_players):
            if j == carrier_idx:
                continue
            if (j < team_size) != is_carrier_team_a:
                continue

            j_phys = self._get_player_physics(self.current_frame_idx, j)
            jx, jy = j_phys["pos"]
            jvx, jvy = j_phys["vel"]

            # 0.60s lead position
            lead_x = max(25.0, min(self.field_w - 25.0, jx + 0.60 * jvx))
            lead_y = max(25.0, min(self.field_h - 25.0, jy + 0.60 * jvy))

            d_opp = min([math.hypot(lead_x - ox, lead_y - oy) for ox, oy in opponents], default=999.0)
            u_j = max(0.0, min(1.0, (d_opp - 65.0) / (220.0 - 65.0)))

            s_jx, s_jy = self.world_to_screen(jx, jy)
            s_lx, s_ly = self.world_to_screen(lead_x, lead_y)

            # Color coding: Emerald (open), Amber (contested), Crimson (tight)
            if u_j >= 0.65:
                lane_color = (45, 235, 130)
            elif u_j >= 0.35:
                lane_color = (255, 210, 60)
            else:
                lane_color = (255, 80, 90)

            # Dotted runner tether from current position to lead position
            lead_dist = math.hypot(s_lx - s_jx, s_ly - s_jy)
            if lead_dist > 5.0:
                pygame.draw.line(self.screen, (160, 190, 220), (s_jx, s_jy), (s_lx, s_ly), width=1)
                pygame.draw.circle(self.screen, (160, 190, 220), (s_jx, s_jy), 3)

            # Passing lane line from carrier to lead target
            pygame.draw.line(self.screen, lane_color, (s_cx, s_cy), (s_lx, s_ly), width=2)

            # Hollow lead target reticle
            pygame.draw.circle(self.screen, lane_color, (s_lx, s_ly), 11, width=2)
            pygame.draw.circle(self.screen, (255, 255, 255), (s_lx, s_ly), 3)

            # Target label: e.g. "M1 92%"
            role_lbl = self._get_player_role_short(j, team_size)
            pct_str = f"{role_lbl} {int(u_j * 100)}%"
            txt_surf = self.font_small.render(pct_str, True, lane_color)
            shadow_surf = self.font_small.render(pct_str, True, (15, 20, 25))
            tx = s_lx - txt_surf.get_width() // 2
            ty = s_ly - 22
            self.screen.blit(shadow_surf, (tx + 1, ty + 1))
            self.screen.blit(txt_surf, (tx, ty))

    def _draw_wall_bank_tracers(self, frame: Dict[str, Any]) -> None:
        """Render trajectory ribbon and boundary cushion reflection sparks when ball rebounds."""
        if not self.ball_trail:
            return

        self.tracer_surface.fill((0, 0, 0, 0))
        trail_len = len(self.ball_trail)

        # 1. Motion Trail Ribbon
        for i in range(1, trail_len):
            bx0, by0, _, _, _ = self.ball_trail[i - 1]
            bx1, by1, _, _, col1 = self.ball_trail[i]
            sx0 = int(round(bx0 * self.scale))
            sy0 = int(round(by0 * self.scale))
            sx1 = int(round(bx1 * self.scale))
            sy1 = int(round(by1 * self.scale))

            alpha = int(180 * (i / trail_len))
            w = max(1, int(3 * (i / trail_len)))
            line_col = (
                (255, 90, 105, alpha)
                if col1 == 0
                else ((80, 180, 255, alpha) if col1 == 1 else (240, 240, 240, alpha))
            )
            pygame.draw.line(self.tracer_surface, line_col, (sx0, sy0), (sx1, sy1), width=w)

        # 2. Wall Bounce Detection & Rebound Sparks
        for i in range(1, trail_len):
            bx, by, _, _, _ = self.ball_trail[i]
            is_wall_contact = (
                by <= 14.0
                or by >= self.field_h - 14.0
                or (
                    (bx <= 14.0 or bx >= self.field_w - 14.0)
                    and (by < self.goal_y_min or by > self.goal_y_max)
                )
            )
            if is_wall_contact:
                sx = int(round(bx * self.scale))
                sy = int(round(by * self.scale))
                age = trail_len - i
                spark_alpha = max(0, int(220 * (1.0 - age / 30.0)))
                if spark_alpha > 0:
                    pygame.draw.circle(
                        self.tracer_surface,
                        (100, 235, 255, spark_alpha),
                        (sx, sy),
                        max(4, 12 - age // 3),
                        width=2,
                    )
                    pygame.draw.circle(
                        self.tracer_surface,
                        (255, 255, 255, spark_alpha),
                        (sx, sy),
                        2,
                    )

        self.screen.blit(self.tracer_surface, (self.margin_x, self.hud_height))

    def _draw_peripheral_radar(self, frame: Dict[str, Any]) -> None:
        """Render 360-degree 8-sector peripheral radar fan for the selected/inspected player."""
        target_idx = (
            self.inspect_player_idx
            if self.inspect_player_idx is not None
            else (next(iter(self.selected_cone_players)) if self.selected_cone_players else None)
        )
        if target_idx is None or target_idx >= len(frame.get("players", [])):
            return

        players = frame["players"]
        pdata = players[target_idx]
        px, py, angle = float(pdata[0]), float(pdata[1]), float(pdata[2])
        team_size = max(1, len(players) // 2)
        is_team_a = target_idx < team_size

        self.radar_surface.fill((0, 0, 0, 0))
        cx = int(round(px * self.scale))
        cy = int(round(py * self.scale))

        # Range circles (150, 300, 450, 600 scaled)
        for rng in (150.0, 300.0, 450.0, 600.0):
            r_px = int(round(rng * self.scale))
            pygame.draw.circle(self.radar_surface, (70, 95, 125, 45), (cx, cy), r_px, width=1)

        # 8 Directional Rays (45 deg sectors)
        num_rays = 8
        ray_len = self.vision_range * self.scale
        for i in range(num_rays):
            ray_ang = angle + i * (2.0 * math.pi / num_rays)
            rx = cx + math.cos(ray_ang) * ray_len
            ry = cy + math.sin(ray_ang) * ray_len
            pygame.draw.line(
                self.radar_surface,
                (80, 115, 155, 60),
                (cx, cy),
                (int(round(rx)), int(round(ry))),
                width=1,
            )

            # Check entities in this sector
            for other_idx, other_p in enumerate(players):
                if other_idx == target_idx:
                    continue
                ox, oy = float(other_p[0]), float(other_p[1])
                dist = math.hypot(ox - px, oy - py)
                if dist > self.vision_range:
                    continue
                rel_ang = math.atan2(oy - py, ox - px)
                diff_ang = (rel_ang - ray_ang + math.pi) % (2.0 * math.pi) - math.pi
                if abs(diff_ang) <= (math.pi / num_rays):
                    blip_r = int(round(dist * self.scale))
                    bx_pt = int(round(cx + math.cos(ray_ang) * blip_r))
                    by_pt = int(round(cy + math.sin(ray_ang) * blip_r))
                    other_team_a = other_idx < team_size
                    b_col = (85, 220, 255) if (other_team_a == is_team_a) else (255, 90, 90)
                    pygame.draw.circle(self.radar_surface, b_col, (bx_pt, by_pt), 4)
                    pygame.draw.circle(self.radar_surface, (255, 255, 255), (bx_pt, by_pt), 2)

            # Check ball
            ball = frame.get("ball", [0.0, 0.0])
            bx, by = float(ball[0]), float(ball[1])
            b_dist = math.hypot(bx - px, by - py)
            if b_dist <= self.vision_range:
                rel_ang = math.atan2(by - py, bx - px)
                diff_ang = (rel_ang - ray_ang + math.pi) % (2.0 * math.pi) - math.pi
                if abs(diff_ang) <= (math.pi / num_rays):
                    blip_r = int(round(b_dist * self.scale))
                    bx_pt = int(round(cx + math.cos(ray_ang) * blip_r))
                    by_pt = int(round(cy + math.sin(ray_ang) * blip_r))
                    pygame.draw.circle(self.radar_surface, (255, 225, 60), (bx_pt, by_pt), 5)
                    pygame.draw.circle(self.radar_surface, (255, 255, 255), (bx_pt, by_pt), 2)

        self.screen.blit(self.radar_surface, (self.margin_x, self.hud_height))

    def _draw_role_tethers(self, frame: Dict[str, Any]) -> None:
        """Render tactical tethers for COVR (to anchor), 1stP (gold to ball), and 2ndP (silver to ball)."""
        role_states = frame.get("role_state")
        role_targets = frame.get("role_targets")
        players = frame.get("players", [])
        num_players = len(players)
        if num_players == 0:
            return

        team_size = max(1, num_players // 2)
        ball = frame.get("ball", [700.0, 420.0])
        bx, by = float(ball[0]), float(ball[1])
        ball_scr = self.world_to_screen(bx, by)

        for idx, p in enumerate(players):
            px, py = float(p[0]), float(p[1])
            p_scr = self.world_to_screen(px, py)
            is_team_a = idx < team_size
            team_col = COLOR_TEAM_A if is_team_a else COLOR_TEAM_B

            r_state = role_states[idx] if (role_states and idx < len(role_states)) else "COVR"

            if r_state == "COVR" and role_targets and idx < len(role_targets):
                tgt = role_targets[idx]
                if tgt and (tgt[0] != 0.0 or tgt[1] != 0.0):
                    tx, ty = float(tgt[0]), float(tgt[1])
                    tgt_scr = self.world_to_screen(tx, ty)
                    # Anchor dot
                    pygame.draw.circle(self.screen, team_col, tgt_scr, 4)
                    pygame.draw.circle(self.screen, (240, 245, 255), tgt_scr, 4, width=1)
                    # Subtle dashed tether
                    tether_col = (
                        team_col[0] // 2 + 50,
                        team_col[1] // 2 + 50,
                        team_col[2] // 2 + 50,
                    )
                    _draw_dashed_line(self.screen, p_scr, tgt_scr, tether_col, dash_len=5.0, space_len=4.0, width=1)
            elif r_state == "1stP":
                # Gold dashed tether toward ball
                _draw_dashed_line(self.screen, p_scr, ball_scr, (255, 215, 60), dash_len=6.0, space_len=3.0, width=2)
            elif r_state == "2ndP":
                # Silver dashed tether toward ball
                _draw_dashed_line(self.screen, p_scr, ball_scr, (195, 205, 220), dash_len=5.0, space_len=4.0, width=1)
            elif r_state == "CHSE":
                # Cyan dashed tether toward ball
                _draw_dashed_line(self.screen, p_scr, ball_scr, (80, 225, 255), dash_len=5.0, space_len=4.0, width=1)

    def gradient_arrows(self, frame_idx: int) -> Tuple[Optional[List[List[float]]], bool]:
        """Per-player ``[direction x, direction y, magnitude]`` arrows for one frame, and whether they are legacy.

        Frames that record the ball-flight state and post-kick timers rebuild the potential field and use the
        central-difference gradient of each player's potential with respect to its own position, with roles and
        marks frozen. Older frames fall back to the stored arrows (`legacy` is True). Results are cached per frame.
        """
        cached = self._gradient_cache.get(frame_idx)
        if cached is not None:
            return cached
        frame = self.frames[frame_idx]
        if "inflight" in frame and "pk" in frame:
            result = (self._true_gradient_arrows(frame), False)
        else:
            stored = frame.get("grad_phi")
            if stored is None:
                rb = frame.get("rew_breakdown")
                if rb:
                    stored = [b.get("grad_phi_xy", [0.0, 0.0, 0.0]) for b in rb]
            result = (stored, True)
        self._gradient_cache[frame_idx] = result
        return result

    def _true_gradient_arrows(self, frame: Dict[str, Any]) -> List[List[float]]:
        """Rebuild the potential state of a recorded frame and convert its gradients to arrow triples."""
        players = np.asarray([p[:4] for p in frame["players"]], dtype=np.float64)
        flags = players[:, 3].astype(np.int64)
        held = np.flatnonzero(flags & 1)
        disabled = (flags & 6) > 0
        ball_team = int(frame["ball"][2])
        state = PotentialState(
            pos=players[:, 0:2].copy(),
            angle=players[:, 2].copy(),
            ball=np.asarray(frame["ball"][0:2], dtype=np.float64),
            holder=int(held[0]) if held.size else -1,
            ball_in_flight=bool(frame["inflight"]),
            last_ball_team=ball_team,
            team_side=np.asarray(frame["team_side"], dtype=np.int32),
            disabled=disabled,
            stunned=(flags & 4) > 0,
            post_kick_timer=np.asarray(frame["pk"], dtype=np.float64),
            depth_bias=TACTICAL_DEPTH_BIAS,
            flank_bias=TACTICAL_FLANK_BIAS,
            active_box_radius=np.asarray(frame.get("box_radii", [self.box_radius] * 2), dtype=np.float64),
        )
        grad = potential_gradient(state, DEFAULT_CONFIG.rewards, self._potential_physics)
        mags = np.hypot(grad[:, 0], grad[:, 1])
        arrows: List[List[float]] = []
        for (gx, gy), mag in zip(grad.tolist(), mags.tolist()):
            if mag > 1e-6:
                arrows.append([round(gx / mag, 3), round(gy / mag, 3), round(mag, 4)])
            else:
                arrows.append([0.0, 0.0, 0.0])
        return arrows

    def _draw_gradient_compass(self, frame: Dict[str, Any]) -> None:
        """Render gold potential gradient arrows and warning red crosses for flat zones."""
        grad_phi, _ = self.gradient_arrows(self.current_frame_idx)
        if not grad_phi:
            return

        players = frame.get("players", [])
        gold_color = (255, 215, 0)
        warn_red = (255, 65, 80)

        for idx, p in enumerate(players):
            if idx >= len(grad_phi):
                break
            g = grad_phi[idx]
            if not g or len(g) < 3:
                continue
            ux, uy, mag = float(g[0]), float(g[1]), float(g[2])
            px, py = float(p[0]), float(p[1])
            sx, sy = self.world_to_screen(px, py)

            if mag < 1e-4:
                # Flat-gradient zone warning: tiny red ×
                pygame.draw.line(self.screen, warn_red, (sx - 3, sy - 3), (sx + 3, sy + 3), width=2)
                pygame.draw.line(self.screen, warn_red, (sx - 3, sy + 3), (sx + 3, sy - 3), width=2)
            else:
                # Gold arrow indicating potential gradient direction and strength
                arrow_len = min(36.0, 14.0 + min(1.0, mag * 25.0) * 20.0)
                tx = int(round(sx + ux * arrow_len))
                ty = int(round(sy + uy * arrow_len))
                pygame.draw.line(self.screen, gold_color, (sx, sy), (tx, ty), width=2)
                ang = math.atan2(uy, ux)
                b1 = (
                    int(round(tx - math.cos(ang + 0.55) * 6)),
                    int(round(ty - math.sin(ang + 0.55) * 6)),
                )
                b2 = (
                    int(round(tx - math.cos(ang - 0.55) * 6)),
                    int(round(ty - math.sin(ang - 0.55) * 6)),
                )
                pygame.draw.line(self.screen, gold_color, (tx, ty), b1, width=2)
                pygame.draw.line(self.screen, gold_color, (tx, ty), b2, width=2)
