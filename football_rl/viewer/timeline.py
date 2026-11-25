"""Replay viewer timeline: frame times, the scrubber bar, the possession ribbon and the cinematic event timeline."""

from __future__ import annotations

from bisect import bisect_left
import math
from typing import Any, Dict, List, Optional

import pygame

from football_rl.viewer.theme import (
    COLOR_KEEPER_GOLD,
    COLOR_LINE,
    COLOR_TEAM_A,
    COLOR_TEAM_B,
    COLOR_TEXT_MAIN,
    COLOR_TEXT_MUTED,
)


class TimelineMixin:
    """Timeline and scrubber of `ReplayViewer`."""

    def _extract_frame_time(self, frame: Dict[str, Any], fallback_raw_idx: int) -> float:
        """Return match time in seconds from `time`, falling back to `frame / physics_hz`."""
        raw_t = frame.get("time")
        if raw_t is not None:
            return float(raw_t)
        raw_f = frame.get("frame")
        if raw_f is not None:
            return float(raw_f) / float(max(1, self.physics_hz))
        return float(fallback_raw_idx) / float(max(1, self.physics_hz))

    def _build_frame_times(self, frames: List[Dict[str, Any]]) -> List[float]:
        """Build a sorted array of match times aligned with `self.frames`."""
        times: List[float] = []
        raw_idx = 0
        prev_t = 0.0
        for fr in frames:
            t = self._extract_frame_time(fr, raw_idx)
            if times and t < prev_t:
                t = prev_t
            times.append(t)
            prev_t = t
            if "cinematic" not in fr:
                raw_idx += 1
        return times

    def _resolve_match_duration(self, raw_frames: List[Dict[str, Any]]) -> float:
        """Return match length T from metadata `max_rl_steps / rl_hz` when present, else last recorded frame time."""
        max_rl_steps = self.meta.get("max_rl_steps")
        rl_hz = self.meta.get("rl_hz")
        if max_rl_steps is not None and rl_hz is not None:
            steps_f = float(max_rl_steps)
            hz_f = float(rl_hz)
            if steps_f > 0.0 and hz_f > 0.0:
                return steps_f / hz_f
        return max(0.0, self._extract_frame_time(raw_frames[-1], len(raw_frames) - 1))

    def _time_to_bar_ratio(self, t: float) -> float:
        """Map match time `t` to a clamped [0, 1] fraction of the scrubber bar."""
        if self.match_duration <= 0.0:
            return 0.0
        return max(0.0, min(1.0, t / self.match_duration))

    def _frame_idx_to_bar_offset(self, frame_idx: int, bar_w: int) -> int:
        """Map a viewer frame index to its horizontal pixel offset within the scrubber bar."""
        idx = max(0, min(self.num_frames - 1, frame_idx))
        return int(round(self._time_to_bar_ratio(self.frame_times[idx]) * bar_w))

    def _frame_idx_to_bar_x(self, frame_idx: int, bar_x: int, bar_w: int) -> int:
        """Map a viewer frame index to its screen x-coordinate on the scrubber bar."""
        return bar_x + self._frame_idx_to_bar_offset(frame_idx, bar_w)

    def _bar_ratio_to_frame_idx(self, ratio: float) -> int:
        """Map a scrubber bar fraction in [0, 1] to the first viewer frame at or after that time."""
        clamped = max(0.0, min(1.0, ratio))
        target_time = clamped * self.match_duration
        lookup_time = max(0.0, target_time - 1e-12) if target_time > 0.0 else 0.0
        idx = bisect_left(self.frame_times, lookup_time)
        return min(idx, self.num_frames - 1)

    def _snap_scrubber_x_to_marker_frame(self, mx: int, bar_x: int, bar_w: int) -> Optional[int]:
        """Return target frame index if `mx` lands on a goal marker (+-6 px) or half-time line (+-1 px)."""
        best_goal_frame: Optional[int] = None
        best_goal_dist: Optional[int] = None
        for g_frame, _, _ in self.goal_frames:
            gx = self._frame_idx_to_bar_x(g_frame, bar_x, bar_w)
            dist = abs(mx - gx)
            if dist <= 6 and (best_goal_dist is None or dist < best_goal_dist):
                best_goal_dist = dist
                best_goal_frame = g_frame
                if dist == 0:
                    break
        if best_goal_frame is not None:
            return best_goal_frame

        if self.halftime_frame_idx is not None:
            ht_x = self._frame_idx_to_bar_x(self.halftime_frame_idx, bar_x, bar_w)
            if abs(mx - ht_x) <= 1:
                return self.halftime_frame_idx

        return None

    def _seek_to_scrubber_x(self, mx: int) -> None:
        """Seek to a snapped goal/halftime marker at `mx`, or the first frame at/after that match time."""
        bar_x = self.margin_x
        bar_w = max(1, self.disp_field_w)
        snapped_idx = self._snap_scrubber_x_to_marker_frame(mx, bar_x, bar_w)
        if snapped_idx is not None:
            self.seek_to_frame(snapped_idx)
            return
        ratio = (mx - bar_x) / float(bar_w)
        self.seek_to_frame(self._bar_ratio_to_frame_idx(ratio))

    def _init_possession_ribbon(self) -> None:
        """Precompute 4px color-coded possession & dead-ball stall ribbon along timeline."""
        bar_w = max(1, self.disp_field_w)
        self.possession_ribbon_surface = pygame.Surface((bar_w, 4))
        self.possession_ribbon_surface.fill((45, 55, 70))

        if self.num_frames == 0:
            return

        num_f = self.num_frames
        states = [2] * num_f
        stat_streak = 0
        last_play_state = 2
        team_size = max(1, len(self.frames[0].get("players", [])) // 2)

        for f_idx in range(num_f):
            frame = self.frames[f_idx]
            if "cinematic" in frame:
                stat_streak = 0
                states[f_idx] = last_play_state
                continue
            players = frame.get("players", [])
            holder_team = -1
            for p_idx, p in enumerate(players):
                if len(p) > 3 and (p[3] & 1):
                    holder_team = 0 if p_idx < team_size else 1
                    break

            if holder_team != -1:
                stat_streak = 0
                states[f_idx] = 0 if holder_team == 0 else 1
            else:
                ball = frame.get("ball", [0, 0, -1, 0, 0])
                vx = float(ball[3]) if len(ball) > 3 else 0.0
                vy = float(ball[4]) if len(ball) > 4 else 0.0
                if math.hypot(vx, vy) < 1.0:
                    stat_streak += 1
                else:
                    stat_streak = 0
                if stat_streak >= 15:
                    start_stall = max(0, f_idx - stat_streak + 1)
                    for k in range(start_stall, f_idx + 1):
                        states[k] = 3
                else:
                    states[f_idx] = 2
            last_play_state = states[f_idx]

        color_map = {
            0: COLOR_TEAM_A,
            1: COLOR_TEAM_B,
            2: (45, 55, 70),
            3: (255, 195, 20),
        }

        for px in range(bar_w):
            f_idx = self._bar_ratio_to_frame_idx(px / float(bar_w))
            st = states[f_idx]
            col = color_map.get(st, (45, 55, 70))
            pygame.draw.line(self.possession_ribbon_surface, col, (px, 0), (px, 3))

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
                if ev.get("type") in ("GOAL!", "OWN GOAL!", "HALFTIME!", "FULLTIME!")
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
            curr_time = curr_fr.get("time")
            curr_raw_frame = curr_fr.get("frame", i)

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
                    "frame": curr_raw_frame,
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
                if curr_time is not None:
                    synth_fr["time"] = curr_time
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
                    "frame": curr_raw_frame,
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
                if curr_time is not None:
                    synth_fr["time"] = curr_time
                expanded_frames.append(synth_fr)

            # Stage 3: Arbiter Whistle Hold
            for k in range(hold_frames):
                p_list = [[p[0], p[1], p[2], 0] for p in curr_fr.get("players", [])]
                synth_fr = {
                    "frame": curr_raw_frame,
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
                if curr_time is not None:
                    synth_fr["time"] = curr_time
                expanded_frames.append(synth_fr)

            # Finally, append curr_fr with events cleared
            curr_fr_copy = dict(curr_fr)
            curr_fr_copy["events"] = []
            expanded_frames.append(curr_fr_copy)

        return expanded_frames

    def _draw_scrubber_bar(self, frame: Dict[str, Any]) -> None:
        """Draw bottom timeline scrubber bar with halftime tick, goal markers, and save markers."""
        bar_x = self.margin_x
        bar_w = self.disp_field_w
        bar_y = self.hud_height + self.disp_field_h + 16
        bar_h = 10

        # 4px color-coded possession & dead-ball stall ribbon along timeline track
        if hasattr(self, "possession_ribbon_surface") and self.possession_ribbon_surface is not None:
            self.screen.blit(self.possession_ribbon_surface, (bar_x, bar_y - 6))

        pygame.draw.rect(
            self.screen,
            (40, 48, 60),
            pygame.Rect(bar_x, bar_y, bar_w, bar_h),
            border_radius=5,
        )

        fill_w = self._frame_idx_to_bar_offset(self.current_frame_idx, bar_w)
        pygame.draw.rect(
            self.screen,
            COLOR_KEEPER_GOLD,
            pygame.Rect(bar_x, bar_y, fill_w, bar_h),
            border_radius=5,
        )

        # Halftime marker at the time of the first second-half frame
        if self.halftime_frame_idx is not None:
            ht_x = self._frame_idx_to_bar_x(self.halftime_frame_idx, bar_x, bar_w)
            pygame.draw.line(
                self.screen, COLOR_LINE, (ht_x, bar_y - 4), (ht_x, bar_y + bar_h + 4), width=2
            )

        # 1. High-reward (+5 or up) other event markers along timeline (triangle shaped: ▲)
        for h_frame, h_team, h_label, h_rew in self.high_reward_frames:
            hx = self._frame_idx_to_bar_x(h_frame, bar_x, bar_w)
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
            fx = self._frame_idx_to_bar_x(f_frame, bar_x, bar_w)
            fy = bar_y + bar_h // 2
            f_col = COLOR_TEAM_A if f_team == 0 else (COLOR_TEAM_B if f_team == 1 else (255, 80, 80))
            card_rect = pygame.Rect(fx - 3, fy - 6, 7, 12)
            pygame.draw.rect(self.screen, f_col, card_rect, border_radius=1)
            pygame.draw.rect(self.screen, (255, 170, 40), card_rect, width=1, border_radius=1)

        # 3. Save markers along timeline (diamond shaped: ◆)
        for s_frame, s_team, is_lunge in self.save_frames:
            sx = self._frame_idx_to_bar_x(s_frame, bar_x, bar_w)
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
            gx = self._frame_idx_to_bar_x(g_frame, bar_x, bar_w)
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
        legend_x = bar_x + bar_w - 390
        legend_y = bar_y + 14

        # Stall icon for Dead-Ball Stall
        pygame.draw.rect(self.screen, (255, 195, 20), pygame.Rect(legend_x + 2, legend_y + 4, 8, 4), border_radius=1)
        stall_leg_surf = self.font_small.render("Stall", True, COLOR_TEXT_MUTED)
        self.screen.blit(stall_leg_surf, (legend_x + 13, legend_y))

        # Circle icon for Goal
        g_leg_x = legend_x + 55
        pygame.draw.circle(self.screen, COLOR_LINE, (g_leg_x + 4, legend_y + 6), 4)
        pygame.draw.circle(self.screen, COLOR_KEEPER_GOLD, (g_leg_x + 4, legend_y + 6), 4, width=1)
        goal_leg_surf = self.font_small.render("Goal", True, COLOR_TEXT_MUTED)
        self.screen.blit(goal_leg_surf, (g_leg_x + 11, legend_y))

        # Diamond icon for Save
        d_leg_x = g_leg_x + 65
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
        t_leg_x = d_leg_x + 65
        t_pts = [
            (t_leg_x + 4, legend_y + 2),
            (t_leg_x + 8, legend_y + 10),
            (t_leg_x, legend_y + 10),
        ]
        pygame.draw.polygon(self.screen, (255, 215, 60), t_pts)
        plus5_leg_surf = self.font_small.render("+5 Event", True, COLOR_TEXT_MUTED)
        self.screen.blit(plus5_leg_surf, (t_leg_x + 11, legend_y))

        # Card icon for Foul
        f_leg_x = t_leg_x + 85
        card_leg_rect = pygame.Rect(f_leg_x + 2, legend_y + 1, 6, 10)
        pygame.draw.rect(self.screen, (255, 80, 80), card_leg_rect, border_radius=1)
        pygame.draw.rect(self.screen, (255, 170, 40), card_leg_rect, width=1, border_radius=1)
        foul_leg_surf = self.font_small.render("Foul", True, COLOR_TEXT_MUTED)
        self.screen.blit(foul_leg_surf, (f_leg_x + 12, legend_y))
