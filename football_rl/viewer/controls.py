"""Replay viewer input: keyboard and mouse events and hit-testing of players, telemetry rows and chips."""

from __future__ import annotations

import math
from typing import Optional

import pygame


class ControlsMixin:
    """Keyboard and mouse handling of `ReplayViewer`."""

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
                elif event.key in (pygame.K_f, pygame.K_b):
                    self.auto_focus_carrier = not self.auto_focus_carrier
                    if self.auto_focus_carrier:
                        self._update_auto_carrier_focus(self.frames[self.current_frame_idx])
                    else:
                        self.inspect_player_idx = None
                        self.selected_cone_players.clear()
                        self.show_vision_cones = False
                elif event.key == pygame.K_c:
                    self.show_critic = not self.show_critic
                elif event.key == pygame.K_g:
                    self.show_gradient = not self.show_gradient
                elif event.key == pygame.K_TAB:
                    if not self.auto_focus_carrier:
                        # Cycle through individual players for cone isolation (0..11 -> OFF -> 0)
                        mods = pygame.key.get_mods()
                        is_shift = bool(mods & pygame.KMOD_SHIFT)
                        num_p = len(self.frames[0]["players"]) if self.frames else 12
                        if not self.selected_cone_players or not self.show_vision_cones:
                            next_idx = (num_p - 1) if is_shift else 0
                            self.selected_cone_players = {next_idx}
                            self.inspect_player_idx = next_idx
                            self.show_vision_cones = True
                        elif len(self.selected_cone_players) == 1:
                            curr_p = next(iter(self.selected_cone_players))
                            if is_shift:
                                if curr_p == 0:
                                    self.selected_cone_players.clear()
                                    self.show_vision_cones = False
                                    self.inspect_player_idx = None
                                else:
                                    self.selected_cone_players = {curr_p - 1}
                                    self.inspect_player_idx = curr_p - 1
                                    self.show_vision_cones = True
                            else:
                                if curr_p == num_p - 1:
                                    self.selected_cone_players.clear()
                                    self.show_vision_cones = False
                                    self.inspect_player_idx = None
                                else:
                                    self.selected_cone_players = {curr_p + 1}
                                    self.inspect_player_idx = curr_p + 1
                                    self.show_vision_cones = True
                        else:
                            self.selected_cone_players.clear()
                            self.show_vision_cones = False
                            self.inspect_player_idx = None
                elif event.key == pygame.K_e:
                    self.show_reward_hud = not self.show_reward_hud
                elif event.key == pygame.K_t:
                    self._toggle_telemetry()
                elif event.key == pygame.K_p:
                    self.show_pass_lanes = not self.show_pass_lanes
                elif event.key == pygame.K_w:
                    self.show_wall_tracers = not self.show_wall_tracers
                elif event.key == pygame.K_i:
                    self.show_radar = not self.show_radar
                elif event.key == pygame.K_h:
                    self.show_pocket_halos = not self.show_pocket_halos
                elif event.key == pygame.K_r:
                    self.seek_to_frame(0)
                elif event.key == pygame.K_UP:
                    self.change_playback_speed(1)
                elif event.key == pygame.K_DOWN:
                    self.change_playback_speed(-1)
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
                # 1. Left Click on Telemetry Quick-Toggle Chips
                if (clicked_chip := (self._find_chip_at_pos(mx, my) if (self.show_telemetry and event.button == 1) else None)) is not None:
                    if clicked_chip == "F:BALL":
                        self.auto_focus_carrier = not self.auto_focus_carrier
                        if self.auto_focus_carrier:
                            self._update_auto_carrier_focus(self.frames[self.current_frame_idx])
                        else:
                            self.inspect_player_idx = None
                            self.selected_cone_players.clear()
                            self.show_vision_cones = False
                    elif clicked_chip == "P:PASS":
                        self.show_pass_lanes = not self.show_pass_lanes
                    elif clicked_chip == "W:WALL":
                        self.show_wall_tracers = not self.show_wall_tracers
                    elif clicked_chip == "V:CONE":
                        self.show_vision_cones = not self.show_vision_cones
                    elif clicked_chip == "I:RADAR":
                        self.show_radar = not self.show_radar
                    elif clicked_chip == "H:HALO":
                        self.show_pocket_halos = not getattr(self, "show_pocket_halos", True)
                    elif clicked_chip == "C:CRIT":
                        self.show_critic = not self.show_critic
                    elif clicked_chip == "G:GRAD":
                        self.show_gradient = not self.show_gradient
                # 1. Left Click on Scrubber Bar -> Seek
                elif event.button == 1 and bar_x <= mx <= bar_x + bar_w and bar_y - 10 <= my <= bar_y + 26:
                    self._dragging_scrubber = True
                    self._seek_to_scrubber_x(mx)
                # 2. Click in Side Telemetry Panel -> Toggle focus / inspect player
                elif self.show_telemetry and self._find_player_at_telemetry_pos(mx, my) is not None:
                    if not self.auto_focus_carrier:
                        p_idx = self._find_player_at_telemetry_pos(mx, my)
                        if event.button == 1 and p_idx is not None:
                            if self.inspect_player_idx == p_idx:
                                self.selected_cone_players.clear()
                                self.inspect_player_idx = None
                                self.show_vision_cones = False
                            else:
                                self.selected_cone_players = {p_idx}
                                self.inspect_player_idx = p_idx
                                self.show_vision_cones = True
                        elif event.button == 3:
                            self.selected_cone_players.clear()
                            self.inspect_player_idx = None
                            self.show_vision_cones = False
                # 3. Left Click on Pitch -> Toggle clicked player's vision cone & inspect
                elif event.button == 1 and self._is_pos_on_field(mx, my):
                    if not self.auto_focus_carrier:
                        p_idx = self._find_player_at_screen_pos(mx, my)
                        if p_idx is not None:
                            if self.inspect_player_idx == p_idx:
                                self.selected_cone_players.clear()
                                self.inspect_player_idx = None
                                self.show_vision_cones = False
                            else:
                                self.selected_cone_players = {p_idx}
                                self.inspect_player_idx = p_idx
                                self.show_vision_cones = True
                # 4. Right Click anywhere -> Clear player selection and turn cones OFF
                elif event.button == 3:
                    if not self.auto_focus_carrier:
                        self.inspect_player_idx = None
                        if self.selected_cone_players:
                            self.selected_cone_players.clear()
                            self.show_vision_cones = False
                        else:
                            self.show_vision_cones = not self.show_vision_cones
            elif event.type == pygame.MOUSEMOTION and self._dragging_scrubber:
                buttons = getattr(event, "buttons", None)
                if buttons is not None and not buttons[0]:
                    self._dragging_scrubber = False
                else:
                    self._seek_to_scrubber_x(event.pos[0])
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._dragging_scrubber = False

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

    def _find_player_at_telemetry_pos(self, mx: int, my: int) -> Optional[int]:
        """Detect if mouse click is on a player row in the telemetry table."""
        if not self.show_telemetry:
            return None
        for idx, rect in self.telemetry_row_rects.items():
            if rect.collidepoint(mx, my):
                return idx
        return None

    def _find_chip_at_pos(self, mx: int, my: int) -> Optional[str]:
        """Detect if mouse click is on a telemetry toggle chip."""
        if not self.show_telemetry:
            return None
        for tag, rect in self.telemetry_chip_rects.items():
            if rect.collidepoint(mx, my):
                return tag
        return None
