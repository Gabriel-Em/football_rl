"""Replay viewer telemetry panel: per-player rows, the focused player's card and derived physics readouts."""

from __future__ import annotations

import math
from typing import Any, Dict

import pygame

from football_rl.viewer.theme import (
    COLOR_GHOST_RING,
    COLOR_KEEPER_GOLD,
    COLOR_STUN_ALERT,
    COLOR_TEAM_A,
    COLOR_TEAM_B,
    FLAG_HAS_BALL,
    FLAG_IS_GHOST,
    FLAG_IS_LUNGING,
    FLAG_IS_STUNNED,
)


class TelemetryMixin:
    """Telemetry panel and focus card of `ReplayViewer`."""

    def _get_player_physics(self, frame_idx: int, p_idx: int) -> Dict[str, Any]:
        """Extract or compute velocity, speed, pocket score, and threat distance for player."""
        frame = self.frames[frame_idx]
        players = frame.get("players", [])
        if p_idx >= len(players):
            return {
                "pos": (0.0, 0.0),
                "vel": (0.0, 0.0),
                "speed": 0.0,
                "pocket": 0.0,
                "d_opp": 999.0,
                "d_ball": 999.0,
                "flags": 0,
                "angle": 0.0,
            }
        pdata = players[p_idx]
        px, py, angle, flags = float(pdata[0]), float(pdata[1]), float(pdata[2]), int(pdata[3])

        # Velocity
        if len(pdata) >= 6:
            vx, vy = float(pdata[4]), float(pdata[5])
        elif frame_idx > 0 and p_idx < len(self.frames[frame_idx - 1].get("players", [])):
            prev_p = self.frames[frame_idx - 1]["players"][p_idx]
            dt = 1.0 / max(1, self.physics_hz)
            vx = (px - float(prev_p[0])) / dt
            vy = (py - float(prev_p[1])) / dt
        else:
            vx, vy = 0.0, 0.0
        speed = math.hypot(vx, vy)

        # Pocket marking score & threat distance
        team_size = max(1, len(players) // 2)
        is_team_a = p_idx < team_size
        opponents = [
            (float(p[0]), float(p[1]))
            for o_idx, p in enumerate(players)
            if (o_idx < team_size) != is_team_a
        ]

        if len(pdata) >= 7:
            pocket = float(pdata[6])
        else:
            density = sum(
                math.exp(-((px - ox) ** 2 + (py - oy) ** 2) / (2.0 * 180.0**2))
                for ox, oy in opponents
            )
            pocket = min(1.0, density)

        d_opp = min([math.hypot(px - ox, py - oy) for ox, oy in opponents], default=999.0)

        ball = frame.get("ball", [0.0, 0.0])
        bx, by = float(ball[0]), float(ball[1])
        d_ball = math.hypot(px - bx, py - by)

        return {
            "pos": (px, py),
            "vel": (vx, vy),
            "speed": speed,
            "pocket": pocket,
            "d_opp": d_opp,
            "d_ball": d_ball,
            "flags": flags,
            "angle": angle,
        }

    def _draw_telemetry_panel(self, frame: Dict[str, Any]) -> None:
        """Render modern broadcast telemetry table to the right of the pitch."""
        self.telemetry_row_rects.clear()
        self.telemetry_chip_rects.clear()
        panel_x = self.disp_field_w + 2 * self.margin_x + 6
        panel_y = self.hud_height
        panel_w = self.side_panel_w - 12
        panel_h = self.disp_field_h

        # 1. Panel Card Background & Border
        panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
        pygame.draw.rect(self.screen, (22, 28, 38), panel_rect, border_radius=8)
        pygame.draw.rect(self.screen, (46, 58, 76), panel_rect, width=2, border_radius=8)

        # 2. Header
        title_surf = self.font_hud.render("MATCH TELEMETRY", True, (240, 245, 250))
        self.screen.blit(title_surf, (panel_x + 12, panel_y + 10))

        # Pulsing 60Hz Live Pill
        live_rect = pygame.Rect(panel_x + panel_w - 68, panel_y + 10, 56, 18)
        pygame.draw.rect(self.screen, (16, 50, 30), live_rect, border_radius=4)
        pygame.draw.rect(self.screen, (45, 200, 110), live_rect, width=1, border_radius=4)
        live_txt = self.font_small.render("● 60 Hz", True, (60, 240, 130))
        self.screen.blit(live_txt, (panel_x + panel_w - 63, panel_y + 12))

        # 3. Momentum & Match Control Gauge
        stats = frame.get("stats", {})
        poss = stats.get("possession_pct", [50.0, 50.0])
        poss_a = (
            float(poss[0]) if isinstance(poss, (list, tuple)) and len(poss) > 0 else 50.0
        )
        score = stats.get("score", [0, 0])
        score_diff = int(score[0]) - int(score[1]) if len(score) >= 2 else 0

        # Ball territory tilt (-1.0 to +1.0)
        bx = float(frame.get("ball", [700.0, 420.0])[0])
        territory_tilt = (bx - self.field_w * 0.5) / (self.field_w * 0.5)
        momentum_score = (
            0.4 * territory_tilt
            + 0.4 * ((poss_a - 50.0) / 50.0)
            + 0.2 * max(-1.0, min(1.0, score_diff * 0.5))
        )
        momentum_score = max(-1.0, min(1.0, momentum_score))

        gauge_y = panel_y + 36
        lbl_red = self.font_small.render("RED MOMENTUM", True, COLOR_TEAM_A)
        lbl_blue = self.font_small.render("BLUE MOMENTUM", True, COLOR_TEAM_B)
        self.screen.blit(lbl_red, (panel_x + 12, gauge_y))
        self.screen.blit(lbl_blue, (panel_x + panel_w - lbl_blue.get_width() - 12, gauge_y))

        gauge_bar_rect = pygame.Rect(panel_x + 12, gauge_y + 16, panel_w - 24, 10)
        pygame.draw.rect(self.screen, (32, 40, 52), gauge_bar_rect, border_radius=5)
        half_w = (panel_w - 24) // 2
        center_x = panel_x + 12 + half_w
        if momentum_score < 0:
            fill_w = int(abs(momentum_score) * half_w)
            fill_rect = pygame.Rect(center_x - fill_w, gauge_y + 16, fill_w, 10)
            pygame.draw.rect(self.screen, COLOR_TEAM_A, fill_rect, border_radius=4)
        else:
            fill_w = int(abs(momentum_score) * half_w)
            fill_rect = pygame.Rect(center_x, gauge_y + 16, fill_w, 10)
            pygame.draw.rect(self.screen, COLOR_TEAM_B, fill_rect, border_radius=4)
        pygame.draw.line(
            self.screen,
            (255, 255, 255),
            (center_x, gauge_y + 14),
            (center_x, gauge_y + 28),
            width=2,
        )

        # Critic V(s) Team Meter
        critic_y = gauge_y + 30
        v_preds = frame.get("v_pred")
        num_players_fr = len(frame.get("players", []))
        t_size = max(1, num_players_fr // 2) if num_players_fr > 0 else 6
        if v_preds and len(v_preds) >= num_players_fr and any(v is not None and v != 0.0 for v in v_preds):
            va_vals = [float(v) for v in v_preds[:t_size] if v is not None]
            vb_vals = [float(v) for v in v_preds[t_size:num_players_fr] if v is not None]
            mean_v_a = float(sum(va_vals) / len(va_vals)) if va_vals else 0.0
            mean_v_b = float(sum(vb_vals) / len(vb_vals)) if vb_vals else 0.0
        else:
            mean_v_a = 0.0
            mean_v_b = 0.0

        lbl_va = self.font_small.render(f"RED V: {mean_v_a:+.2f}", True, COLOR_TEAM_A)
        lbl_vb = self.font_small.render(f"BLUE V: {mean_v_b:+.2f}", True, COLOR_TEAM_B)
        lbl_vc = self.font_small.render("CRITIC V(s)", True, (160, 175, 195))
        self.screen.blit(lbl_va, (panel_x + 12, critic_y))
        self.screen.blit(lbl_vc, (center_x - lbl_vc.get_width() // 2, critic_y))
        self.screen.blit(lbl_vb, (panel_x + panel_w - lbl_vb.get_width() - 12, critic_y))

        c_bar_rect = pygame.Rect(panel_x + 12, critic_y + 16, panel_w - 24, 8)
        pygame.draw.rect(self.screen, (32, 40, 52), c_bar_rect, border_radius=4)
        v_diff = mean_v_a - mean_v_b
        v_tilt = max(-1.0, min(1.0, v_diff / 2.0))
        if v_tilt > 0:
            fill_w = int(abs(v_tilt) * half_w)
            fill_rect = pygame.Rect(center_x - fill_w, critic_y + 16, fill_w, 8)
            pygame.draw.rect(self.screen, COLOR_TEAM_A, fill_rect, border_radius=3)
        elif v_tilt < 0:
            fill_w = int(abs(v_tilt) * half_w)
            fill_rect = pygame.Rect(center_x, critic_y + 16, fill_w, 8)
            pygame.draw.rect(self.screen, COLOR_TEAM_B, fill_rect, border_radius=3)
        pygame.draw.line(
            self.screen,
            (255, 255, 255),
            (center_x, critic_y + 14),
            (center_x, critic_y + 26),
            width=2,
        )

        # 4. Toggle Buttons / Chips
        chips_y = critic_y + 30
        chips = [
            ("F:BALL", self.auto_focus_carrier),
            ("P:PASS", self.show_pass_lanes),
            ("W:WALL", self.show_wall_tracers),
            ("V:CONE", self.show_vision_cones),
            ("I:RADAR", self.show_radar),
            ("H:HALO", getattr(self, "show_pocket_halos", True)),
            ("C:CRIT", self.show_critic),
            ("G:GRAD", self.show_gradient),
        ]
        cx = panel_x + 12
        cy = chips_y
        for tag, is_on in chips:
            chip_col = (45, 210, 120) if is_on else (120, 135, 155)
            bg_col = (18, 48, 32) if is_on else (28, 36, 48)
            chip_txt = self.font_small.render(tag, True, chip_col)
            c_w = chip_txt.get_width() + 8
            if cx + c_w > panel_x + panel_w - 8:
                cx = panel_x + 12
                cy += 20
            chip_rect = pygame.Rect(cx, cy, c_w, 16)
            self.telemetry_chip_rects[tag] = chip_rect
            pygame.draw.rect(self.screen, bg_col, chip_rect, border_radius=3)
            pygame.draw.rect(self.screen, chip_col, chip_rect, width=1, border_radius=3)
            self.screen.blit(chip_txt, (cx + 4, cy + 2))
            cx += c_w + 5

        if self.show_gradient and self.gradient_arrows(self.current_frame_idx)[1]:
            legacy_txt = self.font_small.render("legacy arrows", True, (255, 170, 70))
            if cx + legacy_txt.get_width() > panel_x + panel_w - 8:
                cx = panel_x + 12
                cy += 20
            self.screen.blit(legacy_txt, (cx, cy + 2))

        # 5. Player Tables Header
        tbl_y = cy + 24
        hdr_cols = [
            ("ROLE", 6),
            ("SPD", 48),
            ("POCKET", 92),
            ("THREAT", 160),
            ("STATUS", 224),
        ]
        for col_name, col_x in hdr_cols:
            hdr_surf = self.font_small.render(col_name, True, (160, 175, 195))
            self.screen.blit(hdr_surf, (panel_x + 8 + col_x, tbl_y))

        # Check mouse hover
        mx, my = pygame.mouse.get_pos() if pygame.mouse.get_focused() else (-1, -1)
        self.hovered_player_idx = None

        players = frame.get("players", [])
        num_players = len(players)
        team_size = max(1, num_players // 2)

        # 6. Team A (Red) Header Banner & Rows
        team_a_rect = pygame.Rect(panel_x + 8, tbl_y + 18, panel_w - 16, 20)
        pygame.draw.rect(self.screen, (55, 20, 26), team_a_rect, border_radius=4)
        pygame.draw.rect(self.screen, COLOR_TEAM_A, team_a_rect, width=1, border_radius=4)
        ta_txt = self.font_small.render("● TEAM A  (RED)", True, (255, 120, 130))
        self.screen.blit(ta_txt, (panel_x + 16, tbl_y + 21))

        row_y = tbl_y + 42
        for i in range(min(team_size, num_players)):
            row_y = self._draw_telemetry_row(
                panel_x + 8, row_y, panel_w - 16, i, is_team_a=True, mx=mx, my=my
            )

        # 7. Team B (Blue) Header Banner & Rows
        row_y += 6
        team_b_rect = pygame.Rect(panel_x + 8, row_y, panel_w - 16, 20)
        pygame.draw.rect(self.screen, (18, 38, 65), team_b_rect, border_radius=4)
        pygame.draw.rect(self.screen, COLOR_TEAM_B, team_b_rect, width=1, border_radius=4)
        tb_txt = self.font_small.render("● TEAM B  (BLUE)", True, (120, 195, 255))
        self.screen.blit(tb_txt, (panel_x + 16, row_y + 3))

        row_y += 24
        for i in range(team_size, num_players):
            row_y = self._draw_telemetry_row(
                panel_x + 8, row_y, panel_w - 16, i, is_team_a=False, mx=mx, my=my
            )

        # 8. Selected / Focused Player Card (at bottom)
        focus_idx = self.focused_player_idx
        card_h = 136
        card_x = panel_x + 10
        card_y = panel_y + panel_h - card_h - 8
        card_w = panel_w - 20
        if focus_idx is not None and focus_idx < num_players:
            self._draw_focus_player_card(card_x, card_y, card_w, focus_idx)
        else:
            self._draw_empty_focus_card(card_x, card_y, card_w)

    def _draw_telemetry_row(
        self, rx: int, ry: int, rw: int, idx: int, is_team_a: bool, mx: int, my: int
    ) -> int:
        """Render a single player telemetry row with role badges, speed, pocket rating, threat, and status."""
        row_h = 22
        row_rect = pygame.Rect(rx, ry, rw, row_h)
        self.telemetry_row_rects[idx] = row_rect

        is_hovered = rx <= mx <= rx + rw and ry <= my <= ry + row_h
        if is_hovered:
            self.hovered_player_idx = idx

        is_selected = (
            self.selected_cone_players and idx in self.selected_cone_players
        ) or (self.inspect_player_idx == idx)

        if is_selected:
            pygame.draw.rect(self.screen, (35, 60, 85), row_rect, border_radius=4)
            pygame.draw.rect(self.screen, (80, 220, 255), row_rect, width=1, border_radius=4)
        elif is_hovered:
            pygame.draw.rect(self.screen, (30, 42, 58), row_rect, border_radius=4)

        phys = self._get_player_physics(self.current_frame_idx, idx)
        team_size = max(1, len(self.frames[0]["players"]) // 2) if self.frames else 6
        role_str = self._get_player_role_short(idx, team_size)
        role_accent = self._get_player_role_accent(idx, team_size)

        # Role Badge
        badge_rect = pygame.Rect(rx + 6, ry + 2, 28, 18)
        badge_bg = (*role_accent[:3], 40)
        b_surf = pygame.Surface((28, 18), pygame.SRCALPHA)
        pygame.draw.rect(b_surf, badge_bg, (0, 0, 28, 18), border_radius=3)
        self.screen.blit(b_surf, (rx + 6, ry + 2))
        pygame.draw.rect(self.screen, role_accent, badge_rect, width=1, border_radius=3)
        role_txt = self.font_small.render(role_str, True, role_accent)
        self.screen.blit(role_txt, (rx + 6 + (28 - role_txt.get_width()) // 2, ry + 4))

        # Speed (u/s)
        spd_val = int(round(phys["speed"]))
        spd_txt = self.font_small.render(f"{spd_val:>3}", True, (220, 230, 242))
        self.screen.blit(spd_txt, (rx + 48, ry + 4))

        # Pocket Marking Score (% with color pill)
        pkt_pct = int(round(phys["pocket"] * 100))
        if pkt_pct >= 65:
            pkt_col = (255, 80, 95)
        elif pkt_pct >= 35:
            pkt_col = (245, 205, 50)
        else:
            pkt_col = (45, 230, 140)
        pkt_txt = self.font_small.render(f"{pkt_pct:>2}%", True, pkt_col)
        self.screen.blit(pkt_txt, (rx + 92, ry + 4))

        # Threat distance (closest opponent)
        d_opp_val = int(round(phys["d_opp"]))
        threat_str = f"{d_opp_val:>3}" if d_opp_val < 900 else "—"
        threat_txt = self.font_small.render(threat_str, True, (185, 200, 220))
        self.screen.blit(threat_txt, (rx + 160, ry + 4))

        # Status Icon (Ball holder, lunge, stun, ghost, tactical roles)
        role_states = self.frames[self.current_frame_idx].get("role_state", [])
        tactical_role = role_states[idx] if (role_states and idx < len(role_states)) else None

        status_str = "—"
        status_col = (130, 145, 165)
        flags = phys["flags"]
        if flags & FLAG_IS_STUNNED or tactical_role == "STUN":
            status_str = "💫STUN"
            status_col = COLOR_STUN_ALERT
        elif flags & FLAG_IS_LUNGING or tactical_role == "LNGE":
            status_str = "⚡DIVE" if "GK" in role_str else "⚡LNGE"
            status_col = (80, 240, 220)
        elif flags & FLAG_IS_GHOST:
            status_str = "🛡️GHST"
            status_col = COLOR_GHOST_RING
        elif flags & FLAG_HAS_BALL or tactical_role == "BALL":
            status_str = "●BALL"
            status_col = COLOR_KEEPER_GOLD
        elif tactical_role == "1stP":
            status_str = "⚡1stP"
            status_col = (255, 205, 50)
        elif tactical_role == "2ndP":
            status_str = "🔹2ndP"
            status_col = (195, 210, 230)
        elif tactical_role == "CHSE":
            status_str = "🎯CHSE"
            status_col = (80, 225, 255)
        elif tactical_role == "COVR":
            status_str = "⚓COVR"
            status_col = (130, 160, 200)
        elif tactical_role == "GK":
            status_str = "🧤GK"
            status_col = (245, 185, 75)

        sts_txt = self.font_small.render(status_str, True, status_col)
        self.screen.blit(sts_txt, (rx + 224, ry + 4))

        return ry + row_h

    def _draw_empty_focus_card(self, cx: int, cy: int, cw: int) -> None:
        """Render a placeholder card when no player is currently selected or focused."""
        card_h = 136
        card_rect = pygame.Rect(cx, cy, cw, card_h)
        pygame.draw.rect(self.screen, (16, 22, 30), card_rect, border_radius=6)
        pygame.draw.rect(self.screen, (40, 52, 70), card_rect, width=1, border_radius=6)
        hint_txt = self.font_small.render("NO PLAYER FOCUSED", True, (130, 145, 165))
        hint_sub = self.font_small.render("Click or hover a player to inspect", True, (85, 100, 120))
        self.screen.blit(hint_txt, (cx + (cw - hint_txt.get_width()) // 2, cy + 48))
        self.screen.blit(hint_sub, (cx + (cw - hint_sub.get_width()) // 2, cy + 70))

    def _draw_focus_player_card(self, cx: int, cy: int, cw: int, idx: int) -> None:
        """Render an expanded mini-dossier card for the active/selected player."""
        card_h = 136
        card_rect = pygame.Rect(cx, cy, cw, card_h)
        pygame.draw.rect(self.screen, (16, 22, 30), card_rect, border_radius=6)
        pygame.draw.rect(self.screen, (60, 80, 110), card_rect, width=1, border_radius=6)

        label = self._get_player_label(idx)
        phys = self._get_player_physics(self.current_frame_idx, idx)
        team_size = max(1, len(self.frames[0]["players"]) // 2) if self.frames else 6
        role_accent = self._get_player_role_accent(idx, team_size)
        curr_fr = self.frames[self.current_frame_idx]

        role_states = curr_fr.get("role_state", [])
        state_tag = f"  [{role_states[idx]}]" if (role_states and idx < len(role_states) and role_states[idx]) else ""
        hdr_txt = self.font_small.render(f"FOCUSED: {label.upper()}{state_tag}", True, role_accent)
        self.screen.blit(hdr_txt, (cx + 8, cy + 6))

        px, py = phys["pos"]
        deg = int(math.degrees(phys["angle"]) % 360)
        line1 = f"Pos: ({px:.0f}, {py:.0f})  Dir: {deg}°  Ball: {phys['d_ball']:.0f}u  Spd: {phys['speed']:.0f}"
        t1 = self.font_small.render(line1, True, (190, 205, 225))
        self.screen.blit(t1, (cx + 8, cy + 24))

        # Critic V(s) & TD-Surprise ΔV
        v_preds = curr_fr.get("v_pred", [])
        v_curr = float(v_preds[idx]) if (v_preds and idx < len(v_preds) and v_preds[idx] is not None) else 0.0
        if self.current_frame_idx > 0:
            prev_fr = self.frames[self.current_frame_idx - 1]
            prev_v = prev_fr.get("v_pred", [])
            v_prev = float(prev_v[idx]) if (prev_v and idx < len(prev_v) and prev_v[idx] is not None) else v_curr
            delta_v = v_curr - v_prev
        else:
            delta_v = 0.0

        p_rews = curr_fr.get("player_rewards", [])
        c_rews = curr_fr.get("cum_rewards", [])
        step_r = float(p_rews[idx]) if idx < len(p_rews) else 0.0
        cum_r = float(c_rews[idx]) if idx < len(c_rews) else 0.0
        r_col = (100, 245, 130) if cum_r >= 0 else (255, 110, 110)

        line2 = f"V(s): {v_curr:+.2f} | ΔV: {delta_v:+.2f} | StepR: {step_r:+.2f} (Cum: {cum_r:+.1f})"
        t2 = self.font_small.render(line2, True, r_col)
        self.screen.blit(t2, (cx + 8, cy + 44))

        # Reward breakdown
        rew_bd = curr_fr.get("rew_breakdown", [])
        bd_item = rew_bd[idx] if (rew_bd and idx < len(rew_bd) and isinstance(rew_bd[idx], dict)) else {}
        r_adv = float(bd_item.get("r_ball_adv", 0.0))
        r_pos = float(bd_item.get("r_pot_i", 0.0))
        r_spars = float(bd_item.get("r_sparse", 0.0))
        phi_r = float(bd_item.get("phi_role", 0.0))
        phi_c = float(bd_item.get("phi_crowd", 0.0))
        phi_tot = phi_r - phi_c

        line3 = f"Adv: {r_adv:+.3f} | Pos: {r_pos:+.3f} | Spars: {r_spars:+.2f}"
        t3 = self.font_small.render(line3, True, (215, 230, 245))
        self.screen.blit(t3, (cx + 8, cy + 64))

        line4 = f"Φ_i: {phi_tot:.2f} (Role: {phi_r:.2f}, Crowd: {-phi_c:.2f})"
        t4 = self.font_small.render(line4, True, (180, 205, 235))
        self.screen.blit(t4, (cx + 8, cy + 82))

        # Action readout
        actions = curr_fr.get("actions", [])
        act_item = actions[idx] if (actions and idx < len(actions) and len(actions[idx]) >= 4) else [0.0, 0.0, 0.0, 0.0]
        throt = float(act_item[0])
        steer = float(act_item[1])
        kick = float(act_item[2])
        tkl = float(act_item[3])

        act_x = cx + 8
        act_y = cy + 102
        lbl_act = self.font_small.render("Act: ", True, (160, 175, 195))
        self.screen.blit(lbl_act, (act_x, act_y))
        act_x += lbl_act.get_width()

        lbl_dir = self.font_small.render(f"Dir({throt:+.2f}, {steer:+.2f}) | ", True, (220, 230, 245))
        self.screen.blit(lbl_dir, (act_x, act_y))
        act_x += lbl_dir.get_width()

        kick_col = (255, 220, 70) if kick > 0.35 else (160, 175, 195)
        lbl_kick = self.font_small.render(f"Kick: {kick:.2f}", True, kick_col)
        self.screen.blit(lbl_kick, (act_x, act_y))
        act_x += lbl_kick.get_width()

        lbl_sep = self.font_small.render(" | ", True, (160, 175, 195))
        self.screen.blit(lbl_sep, (act_x, act_y))
        act_x += lbl_sep.get_width()

        tkl_col = (80, 240, 220) if tkl > 0.50 else (160, 175, 195)
        lbl_tkl = self.font_small.render(f"Tkl: {tkl:.2f}", True, tkl_col)
        self.screen.blit(lbl_tkl, (act_x, act_y))
