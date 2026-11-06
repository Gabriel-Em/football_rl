"""Unit tests for ReplayVisualizer (`replay.py`).

Verifies:
1. Save markers and goal markers indexed on timeline.
2. Slow-mo playback speed (0.25x subframe accumulation) and speed toggles.
3. Player cone isolation (click toggle, TAB cycling, single/multi player filtering).
4. Headless frame rendering with saves, goals, and isolated cones.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List
import pytest
import pygame

from football_rl.replay import ReplayViewer, ReplayVisualizer


def _create_mock_replay_data() -> Dict[str, Any]:
    """Create a minimal mock replay JSON structure with 20 frames, goals, and saves."""
    frames: List[Dict[str, Any]] = []
    # 12 players: 6 Team A (0..5), 6 Team B (6..11)
    base_players = [
        [45.0, 420.0, 0.0, 0],    # 0: a_keeper
        [320.0, 345.0, 0.0, 0],   # 1: a_def_1
        [320.0, 495.0, 0.0, 0],   # 2: a_def_2
        [520.0, 260.0, 0.0, 0],   # 3: a_mid_1
        [520.0, 580.0, 0.0, 0],   # 4: a_mid_2
        [700.0, 420.0, 0.0, 0],   # 5: a_atk_1
        [1355.0, 420.0, math.pi, 0], # 6: b_keeper
        [1120.0, 495.0, math.pi, 0], # 7: b_def_1
        [1120.0, 345.0, math.pi, 0], # 8: b_def_2
        [960.0, 220.0, math.pi, 0],  # 9: b_mid_1
        [960.0, 620.0, math.pi, 0],  # 10: b_mid_2
        [910.0, 420.0, math.pi, 0],  # 11: b_atk_1
    ]

    for f in range(20):
        events = []
        if f == 3:
            # Minor tackle event (+1.5): should NOT appear on timeline
            events.append({
                "type": "TACKLE!",
                "x": 600.0,
                "y": 420.0,
                "team": 0,
                "agent": "a_mid_1",
                "reward": 1.5,
            })
        elif f == 5:
            # Save by Team A keeper (always shows up)
            events.append({
                "type": "KEEPER SAVE!",
                "x": 45.0,
                "y": 420.0,
                "team": 0,
                "agent": "a_keeper",
                "reward": 3.0,  # Even if reward < 5.0, saves always show up!
            })
        elif f == 8:
            # Lunge save by Team B keeper (always shows up)
            events.append({
                "type": "LUNGE SAVE!",
                "x": 1355.0,
                "y": 420.0,
                "team": 1,
                "agent": "b_keeper",
                "reward": 6.0,
            })
        elif f == 12:
            # Goal scored by Team A (always shows up)
            events.append({
                "type": "GOAL!",
                "x": 1400.0,
                "y": 420.0,
                "team": 0,
                "agent": "a_atk_1",
                "reward": 10.0,
            })
        elif f == 15:
            # High-reward (+5.0 or up) tactical pass breakthrough
            events.append({
                "type": "ESCAPE PRESS!",
                "x": 800.0,
                "y": 300.0,
                "team": 1,
                "agent": "b_mid_1",
                "reward": 5.4,
            })
        elif f == 17:
            # Keeper 5s foul (always shows up on timeline as a Foul)
            events.append({
                "type": "5s KEEPER FOUL!",
                "x": 45.0,
                "y": 420.0,
                "team": 0,
                "agent": "a_keeper",
                "reward": -5.5,
            })

        frames.append({
            "frame": f,
            "time": f * (1.0 / 60.0),
            "ball": [700.0, 420.0, 0.0, 0.0, 5],
            "players": [list(p) for p in base_players],
            "events": events,
            "stats": {
                "score": [1, 0] if f >= 12 else [0, 0],
                "possession_pct": [55.0, 45.0],
                "passes": [4, 3],
                "tackles": [2, 1],
                "saves": [1, 1],
                "shots": [2, 1],
            },
            "team_side": [0, 1],
            "half": 1,
            "phase": "PLAY",
        })

    return {
        "metadata": {
            "match_id": 999,
            "generation": 42,
            "interest_score": 25.5,
            "field_dimensions": [1400.0, 840.0],
            "goal_width": 200.0,
            "physics_fps": 60,
            "disable_cinematics": True,
        },
        "frames": frames,
    }


def test_saves_and_goals_indexing_on_timeline() -> None:
    """Verify saves and goals are indexed and drawn on the bottom scrubber bar."""
    data = _create_mock_replay_data()
    viz = ReplayVisualizer(data, headless=True)

    # 1. Goal frames verification
    assert len(viz.goal_frames) == 1
    g_frame, g_team, is_og = viz.goal_frames[0]
    assert g_frame == 12
    assert g_team == 0
    assert not is_og

    # 2. Save frames verification (both KEEPER SAVE! and LUNGE SAVE! always appear even if reward < 5.0)
    assert len(viz.save_frames) == 2
    s0_frame, s0_team, s0_lunge = viz.save_frames[0]
    assert s0_frame == 5
    assert s0_team == 0
    assert not s0_lunge

    s1_frame, s1_team, s1_lunge = viz.save_frames[1]
    assert s1_frame == 8
    assert s1_team == 1
    assert s1_lunge

    # 3. High-reward (+5.0 or up) other event frames verification
    assert len(viz.high_reward_frames) == 1
    h_frame, h_team, h_label, h_rew = viz.high_reward_frames[0]
    assert h_frame == 15
    assert h_team == 1
    assert h_label == "ESCAPE PRESS!"
    assert math.isclose(h_rew, 5.4, abs_tol=1e-3)

    # 4. Foul frames verification (5s KEEPER FOUL! / violations always appear on timeline)
    assert len(viz.foul_frames) == 1
    f_frame, f_team, f_type = viz.foul_frames[0]
    assert f_frame == 17
    assert f_team == 0
    assert f_type == "5s KEEPER FOUL!"

    # 5. Minor event at frame 3 (TACKLE!, +1.5) must NOT appear on timeline
    all_marked_frames = (
        {g[0] for g in viz.goal_frames}
        | {s[0] for s in viz.save_frames}
        | {h[0] for h in viz.high_reward_frames}
        | {f[0] for f in viz.foul_frames}
    )
    assert 3 not in all_marked_frames
    assert 5 in all_marked_frames
    assert 8 in all_marked_frames
    assert 12 in all_marked_frames
    assert 15 in all_marked_frames
    assert 17 in all_marked_frames

    # 6. Headless render executes without errors across all frames
    surf = viz.render_frame()
    assert isinstance(surf, pygame.Surface)
    assert surf.get_width() == viz.win_w
    assert surf.get_height() == viz.win_h


def test_cinematic_timeline_event_indexing() -> None:
    """Verify that when cinematics are enabled, all markers map and render cleanly on extended timeline."""
    data = _create_mock_replay_data()
    data["metadata"]["disable_cinematics"] = False
    viz = ReplayVisualizer(data, headless=True)

    assert viz.num_frames > 20  # Expanded by celebration frames
    assert len(viz.goal_frames) == 1
    assert len(viz.save_frames) == 2
    assert len(viz.high_reward_frames) == 1

    # Render frame at goal and frame at high reward event
    viz.seek_to_frame(viz.goal_frames[0][0])
    surf_goal = viz.render_frame()
    assert isinstance(surf_goal, pygame.Surface)

    viz.seek_to_frame(viz.high_reward_frames[0][0])
    surf_hrew = viz.render_frame()
    assert isinstance(surf_hrew, pygame.Surface)


def test_slowmo_playback_speed_and_accumulator() -> None:
    """Verify 0.25x slow-mo advances exactly 1 frame every 4 ticks and key toggles work."""
    data = _create_mock_replay_data()
    viz = ReplayVisualizer(data, headless=True)

    # Set slow-mo speed
    viz.playback_speed = 0.25
    viz.seek_to_frame(0)
    assert viz.current_frame_idx == 0

    # Tick 1: accumulator = 0.25 -> 0 frames advanced
    viz.step_playback()
    assert viz.current_frame_idx == 0

    # Tick 2: accumulator = 0.50 -> 0 frames advanced
    viz.step_playback()
    assert viz.current_frame_idx == 0

    # Tick 3: accumulator = 0.75 -> 0 frames advanced
    viz.step_playback()
    assert viz.current_frame_idx == 0

    # Tick 4: accumulator = 1.00 -> 1 frame advanced!
    viz.step_playback()
    assert viz.current_frame_idx == 1

    # Next 4 ticks advance to frame 2
    for _ in range(4):
        viz.step_playback()
    assert viz.current_frame_idx == 2


def test_vision_cone_player_isolation_and_tab_cycling() -> None:
    """Verify player cone toggling, TAB cycling, and single-player filtering."""
    data = _create_mock_replay_data()
    viz = ReplayVisualizer(data, headless=True, show_vision_cones=True)

    assert viz.show_vision_cones is True
    assert len(viz.selected_cone_players) == 0  # Empty means all cones

    # Verify label formatting
    assert viz._get_player_label(0) == "Red GK"
    assert viz._get_player_label(1) == "Red D1"
    assert viz._get_player_label(5) == "Red ST"
    assert viz._get_player_label(6) == "Blue GK"
    assert viz._get_player_label(11) == "Blue ST"

    # Simulate clicking on Player 0 (Red GK: world (45, 420))
    sx, sy = viz.world_to_screen(45.0, 420.0)
    found_idx = viz._find_player_at_screen_pos(sx, sy)
    assert found_idx == 0

    # Click toggles player 0 in selected_cone_players
    viz.selected_cone_players.add(0)
    assert viz.selected_cone_players == {0}

    # Verify rendering with only player 0 cone active works cleanly
    surf = viz.render_frame()
    assert isinstance(surf, pygame.Surface)

    # Post simulated TAB key event to cycle to Player 1
    tab_event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB, mod=0)
    pygame.event.post(tab_event)
    viz.handle_events()
    assert viz.selected_cone_players == {1}

    # Post TAB again -> cycles to Player 2
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB, mod=0))
    viz.handle_events()
    assert viz.selected_cone_players == {2}

    # Post '0' key -> toggles slowmo (0.25x)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_0, mod=0))
    viz.handle_events()
    assert math.isclose(viz.playback_speed, 0.25, abs_tol=1e-3)

    # Post 's' key -> toggles back to normal (1.0x)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s, mod=0))
    viz.handle_events()
    assert math.isclose(viz.playback_speed, 1.0, abs_tol=1e-3)

    # Post Right Click event to clear cone selection
    right_click = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=(sx, sy))
    pygame.event.post(right_click)
    viz.handle_events()
    assert len(viz.selected_cone_players) == 0


def test_real_saved_replay_render() -> None:
    """Verify loading and rendering an actual on-disk match JSON with saves and goals."""
    import glob
    import os
    from football_rl.replay import load_replay_json

    files = sorted(
        glob.glob("replays/**/match_*.json", recursive=True) + glob.glob("replays/match_*.json"),
        key=os.path.getmtime,
        reverse=True,
    )
    if not files:
        pytest.skip("No replay files in replays/")

    data = None
    # Skip any file that is currently being actively written to by a background training process
    for fpath in files:
        try:
            data = load_replay_json(fpath)
            break
        except Exception:
            continue

    if data is None:
        pytest.skip("No valid replay file could be loaded")

    viz = ReplayViewer(data, headless=True)
    assert viz.num_frames > 0
    # Render frame 0
    surf = viz.render_frame()
    assert isinstance(surf, pygame.Surface)
    # If saves exist, seek to first save and render
    if viz.save_frames:
        viz.seek_to_frame(viz.save_frames[0][0])
        surf_save = viz.render_frame()
        assert isinstance(surf_save, pygame.Surface)


def test_software_cursor_rendering(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that _draw_cursor renders cleanly in standard, player hover, and scrubber hover states."""
    data = _create_mock_replay_data()
    viz = ReplayViewer(data, headless=True)

    # 1. In headless, _draw_cursor is safely callable and returns without crashing
    viz._draw_cursor()

    # 2. Simulate mouse within window and call drawing directly
    viz.headless = False
    viz.screen = pygame.Surface((viz.win_w, viz.win_h))
    monkeypatch.setattr(pygame.mouse, "get_focused", lambda: True)

    # Standard arrow cursor test (in margin)
    monkeypatch.setattr(pygame.mouse, "get_pos", lambda: (10, 10))
    viz._draw_cursor()
    assert isinstance(viz.screen, pygame.Surface)

    # Hovering over scrubber bar
    bar_x = viz.margin_x
    bar_y = viz.hud_height + viz.disp_field_h + 12
    monkeypatch.setattr(pygame.mouse, "get_pos", lambda: (bar_x + 50, bar_y + 5))
    viz._draw_cursor()
    assert isinstance(viz.screen, pygame.Surface)

    # Hovering over player on field
    viz.player_screen_positions = [(viz.margin_x + 100, viz.hud_height + 100)]
    monkeypatch.setattr(pygame.mouse, "get_pos", lambda: (viz.margin_x + 100, viz.hud_height + 100))
    viz._draw_cursor()
    assert isinstance(viz.screen, pygame.Surface)


def test_telemetry_panel_rendering_and_toggles() -> None:
    """Verify right-side telemetry panel rendering, dimensions, and toggle mechanisms."""
    data = _create_mock_replay_data()
    # 1. With telemetry enabled by default, window width accommodates the side panel
    viz = ReplayViewer(data, headless=True, show_telemetry=True)
    expected_w = viz.disp_field_w + 2 * viz.margin_x + viz.side_panel_w
    assert viz.win_w == expected_w
    surf = viz.render_frame()
    assert surf.get_width() == expected_w

    # 2. Toggle telemetry off with 'T' key event
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_t, mod=0))
    viz.handle_events()
    assert viz.show_telemetry is False
    assert viz.win_w == viz.disp_field_w + 2 * viz.margin_x
    surf_off = viz.render_frame()
    assert surf_off.get_width() == viz.win_w

    # 3. Toggle back on
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_t, mod=0))
    viz.handle_events()
    assert viz.show_telemetry is True
    assert viz.win_w == expected_w

    # 4. Click inside telemetry row to inspect/focus player
    rect_0 = viz.telemetry_row_rects[0]
    click_event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rect_0.center)
    pygame.event.post(click_event)
    viz.handle_events()
    assert viz.inspect_player_idx == 0
    assert 0 in viz.selected_cone_players

    # 5. Repeat click on the same row deselects the player
    pygame.event.post(click_event)
    viz.handle_events()
    assert viz.inspect_player_idx is None
    assert len(viz.selected_cone_players) == 0


def test_pass_lanes_unpressed_lead_targets() -> None:
    """Verify passing lane calculations, lead positions, and unpressed factors."""
    data = _create_mock_replay_data()
    # Default ReplayViewer must start with pass lanes OFF
    viz_default = ReplayViewer(data, headless=True)
    assert viz_default.show_pass_lanes is False

    # Give ball to player 5 (Red ST)
    data["frames"][0]["players"][5][3] |= 1  # FLAG_HAS_BALL
    viz = ReplayViewer(data, headless=True, show_pass_lanes=True)

    # Render frame with active passing lanes
    surf = viz.render_frame()
    assert isinstance(surf, pygame.Surface)

    # Test toggling pass lanes via 'P'
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p, mod=0))
    viz.handle_events()
    assert viz.show_pass_lanes is False

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p, mod=0))
    viz.handle_events()
    assert viz.show_pass_lanes is True

    # Test toggling via clicking "P:PASS" chip in telemetry panel
    viz.render_frame()
    assert "P:PASS" in viz.telemetry_chip_rects
    p_rect = viz.telemetry_chip_rects["P:PASS"]
    pygame.event.post(
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(p_rect.centerx, p_rect.centery), button=1)
    )
    viz.handle_events()
    assert viz.show_pass_lanes is False


def test_wall_bank_tracers_and_radar_inspector() -> None:
    """Verify wall rebound tracer sparks and 32-ray peripheral radar inspector."""
    data = _create_mock_replay_data()
    # Position ball at top wall cushion
    data["frames"][0]["ball"] = [500.0, 10.0, 0, 150.0, -100.0]
    viz = ReplayViewer(data, headless=True, show_wall_tracers=True, show_radar=True)

    # Select player 3 for radar inspection
    viz.inspect_player_idx = 3
    surf = viz.render_frame()
    assert isinstance(surf, pygame.Surface)

    # Verify toggles for wall tracers ('W') and radar ('I')
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_w, mod=0))
    viz.handle_events()
    assert viz.show_wall_tracers is False

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_i, mod=0))
    viz.handle_events()
    assert viz.show_radar is False

    # Verify toggle for underfoot pocket marking halos ('H')
    assert viz.show_pocket_halos is True
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_h, mod=0))
    viz.handle_events()
    assert viz.show_pocket_halos is False


def test_auto_carrier_focus_and_pitch_single_select() -> None:
    """Verify follow-ball-carrier toggle and pitch body single-select behavior."""
    data = _create_mock_replay_data()
    # Player 5 (Red ST) has ball in frame 0
    data["frames"][0]["players"][5][3] |= 1  # FLAG_HAS_BALL
    viz = ReplayViewer(data, headless=True, auto_focus_carrier=False)
    viz.render_frame()

    # 1. Pitch body clicks in manual mode: single-select & deselect on repeat click
    p1 = data["frames"][0]["players"][1]
    p1_sx, p1_sy = viz.world_to_screen(float(p1[0]), float(p1[1]))
    p2 = data["frames"][0]["players"][2]
    p2_sx, p2_sy = viz.world_to_screen(float(p2[0]), float(p2[1]))

    # Click D1 body -> focus D1
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(p1_sx, p1_sy), button=1))
    viz.handle_events()
    assert viz.inspect_player_idx == 1
    assert viz.selected_cone_players == {1}

    # Click D2 body -> focus D2, deselect D1
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(p2_sx, p2_sy), button=1))
    viz.handle_events()
    assert viz.inspect_player_idx == 2
    assert viz.selected_cone_players == {2}

    # Click D2 body again -> deselect D2 (nobody focused)
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(p2_sx, p2_sy), button=1))
    viz.handle_events()
    assert viz.inspect_player_idx is None
    assert len(viz.selected_cone_players) == 0

    # 2. Toggle follow-ball-carrier via 'F' key
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f, mod=0))
    viz.handle_events()
    assert viz.auto_focus_carrier is True

    # Render frame -> carrier (player 5) automatically focused
    viz.render_frame()
    assert viz.inspect_player_idx == 5
    assert viz.selected_cone_players == {5}

    # With follow-ball ON, clicking pitch body or telemetry row MUST NOT change focus
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(p1_sx, p1_sy), button=1))
    viz.handle_events()
    assert viz.inspect_player_idx == 5

    t1_rect = viz.telemetry_row_rects[1]
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(t1_rect.centerx, t1_rect.centery), button=1))
    viz.handle_events()
    assert viz.inspect_player_idx == 5

    # 3. Toggle follow-ball-carrier OFF via 'F:BALL' chip click -> deselects player
    assert "F:BALL" in viz.telemetry_chip_rects
    f_rect = viz.telemetry_chip_rects["F:BALL"]
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(f_rect.centerx, f_rect.centery), button=1))
    viz.handle_events()
    assert viz.auto_focus_carrier is False
    assert viz.inspect_player_idx is None
    assert len(viz.selected_cone_players) == 0
    assert viz.show_vision_cones is False

    # 4. Toggle follow-ball ON via 'F' key, then toggle OFF via 'F' key -> deselects player
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f, mod=0))
    viz.handle_events()
    assert viz.auto_focus_carrier is True
    viz.render_frame()
    assert viz.inspect_player_idx == 5

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f, mod=0))
    viz.handle_events()
    assert viz.auto_focus_carrier is False
    assert viz.inspect_player_idx is None
    assert len(viz.selected_cone_players) == 0
    assert viz.show_vision_cones is False

    # Now manual clicks work again: click row 1 focuses player 1
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(t1_rect.centerx, t1_rect.centery), button=1))
    viz.handle_events()
    assert viz.inspect_player_idx == 1





