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


def _halftime_replay(goal_frames: List[int], halftime_frame: int, n_frames: int = 30,
                     cinematics: bool = True, with_halftime_event: bool = True) -> Dict[str, Any]:
    """Mock replay with goals at given raw frames and a half switch at ``halftime_frame``."""
    data = _create_mock_replay_data()
    base = data["frames"][0]
    frames: List[Dict[str, Any]] = []
    for f in range(n_frames):
        fr = dict(base)
        fr["frame"] = f
        fr["time"] = f / 60.0
        fr["events"] = []
        fr["half"] = 2 if f >= halftime_frame else 1
        if f in goal_frames:
            fr["events"].append({"type": "GOAL!", "x": 1400.0, "y": 420.0, "team": 0,
                                 "agent": "a_atk_1", "reward": 10.0})
        if with_halftime_event and f == halftime_frame:
            fr["events"].append({"type": "HALFTIME!", "x": 700.0, "y": 420.0, "team": 1,
                                 "agent": "referee", "reward": 0.0})
        frames.append(fr)
    data["frames"] = frames
    data["metadata"]["disable_cinematics"] = not cinematics
    return data


def test_halftime_tick_follows_halftime_cinematic_not_midpoint() -> None:
    """1 goal in half 1 and 3 in half 2: the tick sits at the HALFTIME cinematic start."""
    data = _halftime_replay(goal_frames=[5, 18, 22, 26], halftime_frame=15)
    viz = ReplayVisualizer(data, headless=True)

    # Each goal inserts 90 freeze + 120 walk + 60 hold = 270 frames; one goal precedes half-time.
    assert viz.halftime_frame_idx == 15 + 270
    assert viz.frames[viz.halftime_frame_idx]["cinematic"]["event_type"] == "HALFTIME!"
    assert viz.num_frames == 30 + 4 * 270 + 240  # plus the half-time cinematic (60 + 120 + 60)
    assert viz.halftime_frame_idx != viz.num_frames // 2


def test_halftime_tick_pixel_is_drawn_at_marker_position() -> None:
    """In a full match with uneven goals and cinematics, the half-time line sits at the midpoint."""
    data = _halftime_replay(goal_frames=[5, 18, 22, 26], halftime_frame=15, n_frames=31)
    viz = ReplayVisualizer(data, headless=True)
    viz.seek_to_frame(viz.num_frames - 1)
    surf = viz.render_frame()
    bar_x, bar_w = viz.margin_x, viz.disp_field_w
    bar_y = viz.hud_height + viz.disp_field_h + 16
    mid_x = bar_x + bar_w // 2
    frame_prop_x = bar_x + int((viz.halftime_frame_idx / float(viz.num_frames - 1)) * bar_w)
    probe_y = bar_y - 3
    assert viz.halftime_frame_idx != viz.num_frames // 2
    assert any(
        tuple(surf.get_at((mid_x + dx, probe_y)))[:3] == (235, 240, 245) for dx in (-1, 0, 1)
    )
    assert frame_prop_x != mid_x
    assert tuple(surf.get_at((frame_prop_x, probe_y)))[:3] != (235, 240, 245)


def test_markers_and_fill_follow_match_time_with_and_without_cinematics() -> None:
    """Goal, save, foul, high-reward markers and progress fill map by match time identically with/without cinematics."""
    bar_x_expected, bar_w_expected = 45, 1200
    for cinematics in (True, False):
        data = _halftime_replay(
            goal_frames=[15, 40, 50], halftime_frame=30, n_frames=61, cinematics=cinematics
        )
        data["frames"][12]["events"].append(
            {"type": "ESCAPE PRESS!", "x": 700.0, "y": 420.0, "team": 1, "agent": "b_mid_1", "reward": 6.0}
        )
        data["frames"][36]["events"].append(
            {"type": "5s KEEPER FOUL!", "x": 1355.0, "y": 420.0, "team": 1, "agent": "b_keeper", "reward": -5.5}
        )
        data["frames"][45]["events"].append(
            {"type": "KEEPER SAVE!", "x": 1355.0, "y": 420.0, "team": 1, "agent": "b_keeper", "reward": 2.0}
        )
        viz = ReplayVisualizer(data, headless=True)
        assert (viz.margin_x, viz.disp_field_w) == (bar_x_expected, bar_w_expected)
        bar_y = viz.hud_height + viz.disp_field_h + 16
        mid_y = bar_y + 5

        viz.seek_to_frame(0)
        surf = viz.render_frame()
        # Quarter-match goal (frame 15 of 60 -> 0.25 * 1200 = 300) drawn in Team A red
        assert any(
            tuple(surf.get_at((bar_x_expected + 300 + dx, mid_y)))[:3] == (255, 60, 75)
            for dx in (-1, 0, 1)
        )
        # High-reward at 20% (frame 12 -> 240), foul at 60% (frame 36 -> 720), save at 75% (frame 45 -> 900) in Team B blue
        for offset in (240, 720, 900):
            assert any(
                tuple(surf.get_at((bar_x_expected + offset + dx, mid_y)))[:3] == (55, 155, 255)
                for dx in (-1, 0, 1)
            )

        # Progress fill at halftime (t = 0.50 * T -> x = 45 + 600 = 645)
        assert viz.halftime_frame_idx is not None
        viz.seek_to_frame(viz.halftime_frame_idx)
        surf_ht = viz.render_frame()
        assert tuple(surf_ht.get_at((bar_x_expected + 585, mid_y)))[:3] == (255, 215, 70)
        assert tuple(surf_ht.get_at((bar_x_expected + 615, mid_y)))[:3] == (40, 48, 60)


def test_fulltime_cinematic_adds_zero_bar_width() -> None:
    """Full-time cinematic frames belong to the final recorded timestamp and add no bar width."""
    data = _halftime_replay(goal_frames=[], halftime_frame=30, n_frames=61, cinematics=True)
    data["frames"][-1]["events"].append(
        {"type": "FULLTIME!", "x": 700.0, "y": 420.0, "team": -1, "agent": "referee", "reward": None}
    )
    viz = ReplayVisualizer(data, headless=True)
    # 61 raw + 240 HALFTIME + 150 FULLTIME = 451 frames; FULLTIME starts at index 300.
    assert viz.num_frames == 451
    assert viz.frames[300]["cinematic"]["event_type"] == "FULLTIME!"

    viz.seek_to_frame(300)
    surf = viz.render_frame()
    bar_x, bar_w = viz.margin_x, viz.disp_field_w
    bar_y = viz.hud_height + viz.disp_field_h + 16
    # Progress fill reaches the right end even on the first FULLTIME cinematic frame
    assert tuple(surf.get_at((bar_x + bar_w - 15, bar_y + 5)))[:3] == (255, 215, 70)
    assert tuple(surf.get_at((bar_x + bar_w, bar_y + 5)))[:3] == (240, 244, 248)
    # Halftime tick remains centered at bar_x + bar_w // 2
    mid_x = bar_x + bar_w // 2
    assert any(
        tuple(surf.get_at((mid_x + dx, bar_y - 3)))[:3] == (235, 240, 245) for dx in (-1, 0, 1)
    )


def test_scrubber_click_and_drag_seek_by_match_time() -> None:
    """Click/drag seeks to the first frame with time >= clicked time, landing on cinematic starts."""
    data = _halftime_replay(goal_frames=[15], halftime_frame=30, n_frames=61, cinematics=True)
    viz = ReplayVisualizer(data, headless=True)
    bar_x, bar_w = viz.margin_x, viz.disp_field_w
    click_y = viz.hud_height + viz.disp_field_h + 16

    # 1. Click at 75% (t = 0.75s -> raw frame 45 -> viewer index 45 + 270 + 240 = 555)
    pygame.event.post(
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(bar_x + 900, click_y))
    )
    viz.handle_events()
    assert viz.current_frame_idx == 555
    assert viz.frames[viz.current_frame_idx]["time"] == pytest.approx(0.75)

    # 2. Click on the goal's moment at 25% -> lands on the FIRST frame of the GOAL cinematic (index 15)
    pygame.event.post(
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(bar_x + 300, click_y))
    )
    viz.handle_events()
    assert viz.current_frame_idx == 15
    assert viz.frames[15]["cinematic"]["event_type"] == "GOAL!"
    assert viz.frames[15]["cinematic"]["phase"] == "FREEZE"
    assert viz.frames[15]["cinematic"]["progress"] == 0.0

    # 3. Drag to 50% -> lands on the FIRST frame of the HALFTIME cinematic (index 300)
    pygame.event.post(
        pygame.event.Event(pygame.MOUSEMOTION, pos=(bar_x + 600, click_y), buttons=(1, 0, 0))
    )
    viz.handle_events()
    assert viz.current_frame_idx == 300
    assert viz.frames[300]["cinematic"]["event_type"] == "HALFTIME!"

    # 4. Drag past right edge clamps to last frame; release stops drag
    pygame.event.post(
        pygame.event.Event(pygame.MOUSEMOTION, pos=(bar_x + bar_w + 40, click_y), buttons=(1, 0, 0))
    )
    pygame.event.post(
        pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(bar_x + bar_w + 40, click_y))
    )
    pygame.event.post(
        pygame.event.Event(pygame.MOUSEMOTION, pos=(bar_x + 100, click_y), buttons=(0, 0, 0))
    )
    viz.handle_events()
    assert viz.current_frame_idx == viz.num_frames - 1

    # 5. Float-precision boundary: n_frames=25 (T=24/60=0.4s), 0.75*0.4 exceeds 18/60 by 1 ULP;
    # clicking 75% must still land on the first frame of raw frame 18's goal cinematic (index 18).
    data_ulp = _halftime_replay(
        goal_frames=[18], halftime_frame=10_000, n_frames=25, cinematics=True, with_halftime_event=False
    )
    viz_ulp = ReplayVisualizer(data_ulp, headless=True)
    pygame.event.post(
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(bar_x + 900, click_y))
    )
    viz_ulp.handle_events()
    assert viz_ulp.current_frame_idx == 18
    assert viz_ulp.frames[18]["cinematic"]["event_type"] == "GOAL!"


def test_legacy_replay_without_time_and_metadata_duration() -> None:
    """Replays without `time` fall back to `frame / physics_hz`; metadata `max_rl_steps / rl_hz` sets T."""
    for cinematics in (True, False):
        data = _halftime_replay(
            goal_frames=[15, 40, 50], halftime_frame=30, n_frames=61, cinematics=cinematics
        )
        for fr in data["frames"]:
            fr.pop("time", None)
        data["metadata"]["physics_hz"] = 30
        viz = ReplayVisualizer(data, headless=True)
        bar_x, bar_w = viz.margin_x, viz.disp_field_w
        bar_y = viz.hud_height + viz.disp_field_h + 16
        click_y = bar_y

        viz.seek_to_frame(0)
        surf = viz.render_frame()
        # Halftime at frame 30 / 60 -> midpoint (645); goal at frame 15 / 60 -> quarter (345)
        assert any(
            tuple(surf.get_at((bar_x + 600 + dx, bar_y - 3)))[:3] == (235, 240, 245)
            for dx in (-1, 0, 1)
        )
        assert any(
            tuple(surf.get_at((bar_x + 300 + dx, bar_y + 5)))[:3] == (255, 60, 75)
            for dx in (-1, 0, 1)
        )

        # Clicking 75% seeks to raw frame 45 (index 45 without cinematics; 45 + 2*270 + 240 = 825 with)
        pygame.event.post(
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(bar_x + 900, click_y))
        )
        viz.handle_events()
        expected_idx = 825 if cinematics else 45
        assert viz.current_frame_idx == expected_idx

    # Metadata max_rl_steps / rl_hz sets T=1.0s even when recorded frames stop at t=0.5s
    data_meta = _halftime_replay(
        goal_frames=[], halftime_frame=10_000, n_frames=31, cinematics=False, with_halftime_event=False
    )
    data_meta["metadata"]["max_rl_steps"] = 15
    data_meta["metadata"]["rl_hz"] = 15
    viz_meta = ReplayVisualizer(data_meta, headless=True)
    viz_meta.seek_to_frame(30)  # t = 0.5s out of T = 1.0s -> fills half the bar
    surf_meta = viz_meta.render_frame()
    bar_x = viz_meta.margin_x
    bar_y = viz_meta.hud_height + viz_meta.disp_field_h + 16
    assert tuple(surf_meta.get_at((bar_x + 580, bar_y + 5)))[:3] == (255, 215, 70)
    assert tuple(surf_meta.get_at((bar_x + 620, bar_y + 5)))[:3] == (40, 48, 60)

    # Without `time` + metadata T=2.0s + physics_hz=30: frame 30 -> 30/30 = 1.0s (50% of bar, not 25%)
    data_hz = _halftime_replay(
        goal_frames=[], halftime_frame=10_000, n_frames=31, cinematics=False, with_halftime_event=False
    )
    for fr in data_hz["frames"]:
        fr.pop("time", None)
    data_hz["metadata"]["physics_hz"] = 30
    data_hz["metadata"]["max_rl_steps"] = 30
    data_hz["metadata"]["rl_hz"] = 15
    viz_hz = ReplayVisualizer(data_hz, headless=True)
    viz_hz.seek_to_frame(30)
    surf_hz = viz_hz.render_frame()
    assert tuple(surf_hz.get_at((bar_x + 580, bar_y + 5)))[:3] == (255, 215, 70)
    assert tuple(surf_hz.get_at((bar_x + 620, bar_y + 5)))[:3] == (40, 48, 60)


def test_single_frame_replay_scrubber_handles_zero_duration() -> None:
    """A single-frame replay (T=0) renders and handles scrubber clicks without ZeroDivisionError."""
    data = _halftime_replay(
        goal_frames=[], halftime_frame=10_000, n_frames=1, cinematics=False, with_halftime_event=False
    )
    viz = ReplayVisualizer(data, headless=True)
    surf = viz.render_frame()
    assert isinstance(surf, pygame.Surface)
    click_y = viz.hud_height + viz.disp_field_h + 16
    pygame.event.post(
        pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(viz.margin_x + 600, click_y))
    )
    viz.handle_events()
    assert viz.current_frame_idx == 0


def test_halftime_index_falls_back_to_half_two_then_none() -> None:
    """Without cinematics or a HALFTIME event the first half-2 frame is used; no half-2 gives None."""
    data = _halftime_replay([], halftime_frame=12, cinematics=False, with_halftime_event=False)
    viz = ReplayVisualizer(data, headless=True)
    assert viz.halftime_frame_idx == 12

    data = _halftime_replay([], halftime_frame=10_000, cinematics=False, with_halftime_event=False)
    viz = ReplayVisualizer(data, headless=True)
    assert viz.halftime_frame_idx is None
    assert isinstance(viz.render_frame(), pygame.Surface)


def test_halftime_index_uses_event_when_goal_cinematic_shadows_it() -> None:
    """A goal on the half-time frame owns the cinematic; the tick uses the HALFTIME event frame."""
    data = _halftime_replay(goal_frames=[15], halftime_frame=15)
    viz = ReplayVisualizer(data, headless=True)
    idx = viz.halftime_frame_idx
    assert idx == 15
    assert any(ev.get("type") == "HALFTIME!" for ev in viz.frames[idx]["events"])


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

    # Post Down Arrow key -> slows speed down to 0.75x
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN, mod=0))
    viz.handle_events()
    assert math.isclose(viz.playback_speed, 0.75, abs_tol=1e-3)

    # Post Up Arrow key -> speeds back up to 1.0x
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP, mod=0))
    viz.handle_events()
    assert math.isclose(viz.playback_speed, 1.0, abs_tol=1e-3)

    # Post Right Click event to clear cone selection
    right_click = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=(sx, sy))
    pygame.event.post(right_click)
    viz.handle_events()
    assert len(viz.selected_cone_players) == 0


def test_real_saved_replay_render(tmp_path) -> None:
    """Load and render a replay that a training generation wrote to disk (critic values and policy means recorded)."""
    from dataclasses import replace

    from football_rl.config import DEFAULT_CONFIG
    from football_rl.replay import load_replay_json
    from football_rl.train_ppo import ParameterSharingPPOTrainer

    rep_dir = tmp_path / "replays"
    cfg = replace(
        DEFAULT_CONFIG,
        physics=replace(DEFAULT_CONFIG.physics, max_rl_steps=40),
        replay=replace(DEFAULT_CONFIG.replay, save_dir=str(rep_dir), enable_recording=True, interest_threshold=0.0),
    )
    trainer = ParameterSharingPPOTrainer(
        config=cfg, num_envs=1, rollout_steps=40, device="cpu", checkpoint_dir=str(tmp_path / "ckpt"), seed=7
    )
    try:
        trainer.train(total_updates=1)
    finally:
        trainer.close()

    files = sorted(rep_dir.glob("match_*.json"))
    assert files, "the training generation wrote no replay"
    data = load_replay_json(str(files[0]))
    # 40 decisions at 4 physics frames each, plus the reset frame
    assert len(data["frames"]) == 161
    assert data["frames"][-1]["v_pred"] is not None and len(data["frames"][-1]["v_pred"]) == 12
    assert data["frames"][-1]["action_mean"] is not None and len(data["frames"][-1]["action_mean"]) == 12

    # A keeper save on frame 80 must be indexed and renderable (training matches this short rarely contain one).
    data["frames"][80]["events"].append(
        {"type": "KEEPER SAVE!", "x": 45.0, "y": 420.0, "team": 0, "agent": "a_keeper", "reward": 0.2}
    )
    viz = ReplayViewer(data, headless=True)
    assert viz.num_frames >= 161  # the viewer may append cinematic frames
    surf = viz.render_frame()
    assert isinstance(surf, pygame.Surface)
    assert len(viz.save_frames) == 1
    save_idx = viz.save_frames[0][0]
    assert any(ev.get("type") == "KEEPER SAVE!" for ev in viz.frames[save_idx].get("events", []))
    viz.seek_to_frame(save_idx)
    assert viz.current_frame_idx == save_idx
    surf_save = viz.render_frame()
    assert isinstance(surf_save, pygame.Surface)
    viz.seek_to_frame(viz.num_frames - 1)
    assert isinstance(viz.render_frame(), pygame.Surface)


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


def test_speed_toggle_up_and_down_arrows() -> None:
    """Verify Down Arrow slows down and Up Arrow speeds up through playback speeds starting at normal (1.0x)."""
    data = _create_mock_replay_data()
    viz = ReplayVisualizer(data, headless=True)
    # Starts with normal (1.0x)
    assert math.isclose(viz.playback_speed, 1.0, abs_tol=1e-3)

    # 1. Down Arrow slows down: 1.0 -> 0.75 -> 0.5 -> 0.25 (clamped)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN, mod=0))
    viz.handle_events()
    assert math.isclose(viz.playback_speed, 0.75, abs_tol=1e-3)

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN, mod=0))
    viz.handle_events()
    assert math.isclose(viz.playback_speed, 0.50, abs_tol=1e-3)

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN, mod=0))
    viz.handle_events()
    assert math.isclose(viz.playback_speed, 0.25, abs_tol=1e-3)

    # Further Down Arrow stays clamped at min 0.25
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN, mod=0))
    viz.handle_events()
    assert math.isclose(viz.playback_speed, 0.25, abs_tol=1e-3)

    # 2. Up Arrow speeds up: 0.25 -> 0.50 -> 0.75 -> 1.0 -> 2.0 -> 4.0 (clamped)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP, mod=0))
    viz.handle_events()
    assert math.isclose(viz.playback_speed, 0.50, abs_tol=1e-3)

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP, mod=0))
    viz.handle_events()
    assert math.isclose(viz.playback_speed, 0.75, abs_tol=1e-3)

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP, mod=0))
    viz.handle_events()
    assert math.isclose(viz.playback_speed, 1.0, abs_tol=1e-3)

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP, mod=0))
    viz.handle_events()
    assert math.isclose(viz.playback_speed, 2.0, abs_tol=1e-3)

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP, mod=0))
    viz.handle_events()
    assert math.isclose(viz.playback_speed, 4.0, abs_tol=1e-3)

    # Further Up Arrow stays clamped at max 4.0
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP, mod=0))
    viz.handle_events()
    assert math.isclose(viz.playback_speed, 4.0, abs_tol=1e-3)

    # Down Arrow slows down from 4.0 -> 2.0 -> 1.0
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN, mod=0))
    viz.handle_events()
    assert math.isclose(viz.playback_speed, 2.0, abs_tol=1e-3)

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN, mod=0))
    viz.handle_events()
    assert math.isclose(viz.playback_speed, 1.0, abs_tol=1e-3)

    # 3. Ensure other keys (including A and D which were deprecated) do NOT trigger speed changes
    for ignored_key in (
        pygame.K_a,
        pygame.K_d,
        pygame.K_s,
        pygame.K_0,
        pygame.K_1,
        pygame.K_2,
        pygame.K_4,
        pygame.K_5,
        pygame.K_7,
        pygame.K_LEFTBRACKET,
        pygame.K_RIGHTBRACKET,
        pygame.K_COMMA,
        pygame.K_PERIOD,
    ):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=ignored_key, mod=0))
        viz.handle_events()
        assert math.isclose(viz.playback_speed, 1.0, abs_tol=1e-3)

    # 4. Ensure mouse clicks do NOT trigger speed changes
    viz.render_frame()
    for button in (1, 2, 3):
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(100, 70), button=button))
        viz.handle_events()
        assert math.isclose(viz.playback_speed, 1.0, abs_tol=1e-3)

    # 5. Method API toggle_playback_speed
    assert math.isclose(viz.toggle_playback_speed(reverse=False), 2.0, abs_tol=1e-3)
    assert math.isclose(viz.toggle_playback_speed(reverse=True), 1.0, abs_tol=1e-3)


def test_default_no_player_focused() -> None:
    """Verify that by default when no player is selected, nobody is focused (not Red GK)."""
    data = _create_mock_replay_data()
    viz = ReplayViewer(data, headless=True, show_telemetry=True)

    # Initially before and after rendering, no player is inspected or hovered
    assert viz.inspect_player_idx is None
    assert viz.hovered_player_idx is None
    assert viz.focused_player_idx is None

    # Render initial frame
    surf = viz.render_frame()
    assert isinstance(surf, pygame.Surface)
    assert viz.focused_player_idx is None
    assert viz.inspect_player_idx is None

    # Explicitly focus player 1 -> focused_player_idx becomes 1
    viz.inspect_player_idx = 1
    assert viz.focused_player_idx == 1

    # Deselect player -> focused_player_idx returns to None (nobody focused)
    viz.inspect_player_idx = None
    assert viz.focused_player_idx is None

    # Hovering player 2 temporarily sets focused_player_idx to 2
    viz.hovered_player_idx = 2
    assert viz.focused_player_idx == 2
    viz.hovered_player_idx = None
    assert viz.focused_player_idx is None

    # When player 0 (Red GK) is intentionally focused, it works as expected
    viz.inspect_player_idx = 0
    assert viz.focused_player_idx == 0
    surf_p0 = viz.render_frame()
    assert isinstance(surf_p0, pygame.Surface)

    # Boundary: out of bounds inspect_player_idx does not crash render
    viz.inspect_player_idx = 999
    assert viz.focused_player_idx == 999
    surf_oob = viz.render_frame()
    assert isinstance(surf_oob, pygame.Surface)

    # Boundary: empty players list does not crash render with no player focused
    viz.inspect_player_idx = None
    data_empty = _create_mock_replay_data()
    for fr in data_empty["frames"]:
        fr["players"] = []
    viz_empty = ReplayViewer(data_empty, headless=True, show_telemetry=True)
    surf_empty = viz_empty.render_frame()
    assert isinstance(surf_empty, pygame.Surface)
    assert viz_empty.focused_player_idx is None


def test_subframe_accumulator_0_5_and_0_75() -> None:
    """Verify subframe accumulator accuracy for 0.5x and 0.75x slow-mo speeds."""
    data = _create_mock_replay_data()
    viz = ReplayVisualizer(data, headless=True)

    # 1. Test 0.50x playback speed: advances 1 frame every 2 ticks
    viz.playback_speed = 0.50
    viz.seek_to_frame(0)
    assert viz.current_frame_idx == 0

    viz.step_playback()  # tick 1: acc = 0.5 -> 0 frames advanced
    assert viz.current_frame_idx == 0
    viz.step_playback()  # tick 2: acc = 1.0 -> 1 frame advanced
    assert viz.current_frame_idx == 1
    viz.step_playback()  # tick 3: acc = 0.5 -> 0 frames advanced
    assert viz.current_frame_idx == 1
    viz.step_playback()  # tick 4: acc = 1.0 -> 1 frame advanced
    assert viz.current_frame_idx == 2

    # 2. Test 0.75x playback speed: advances 3 frames every 4 ticks
    viz.playback_speed = 0.75
    viz.subframe_accumulator = 0.0
    viz.seek_to_frame(0)
    assert viz.current_frame_idx == 0

    viz.step_playback()  # tick 1: acc = 0.75 -> 0 frames advanced
    assert viz.current_frame_idx == 0
    viz.step_playback()  # tick 2: acc = 1.50 -> 1 frame advanced (acc rem 0.50)
    assert viz.current_frame_idx == 1
    viz.step_playback()  # tick 3: acc = 1.25 -> 1 frame advanced (acc rem 0.25)
    assert viz.current_frame_idx == 2
    viz.step_playback()  # tick 4: acc = 1.00 -> 1 frame advanced (acc rem 0.00)
    assert viz.current_frame_idx == 3


def test_critic_telemetry_toggle_and_display() -> None:
    """Verify Critic V(s) recording, [C] key and chip toggles, diverging ring, and TD-surprise."""
    data = _create_mock_replay_data()
    for i, fr in enumerate(data["frames"]):
        fr["v_pred"] = [round(0.2 * i - 1.0 + p * 0.1, 2) for p in range(12)]

    viz = ReplayVisualizer(data, headless=True)
    assert not viz.show_critic

    # 1. Toggle via keyboard 'C'
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c, mod=0))
    viz.handle_events()
    assert viz.show_critic

    # 2. Toggle via chip 'C:CRIT'
    viz.render_frame()
    chip_rect = viz.telemetry_chip_rects.get("C:CRIT")
    assert chip_rect is not None
    pygame.event.post(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            pos=(chip_rect.centerx, chip_rect.centery),
            button=1,
        )
    )
    viz.handle_events()
    assert not viz.show_critic

    # 3. Turn on and render frame 2
    viz.show_critic = True
    viz.seek_to_frame(2)
    surf = viz.render_frame()
    assert isinstance(surf, pygame.Surface)

    # 4. Verify focused player card TD-surprise calculation
    viz.inspect_player_idx = 3
    viz.render_frame()


def test_reward_breakdown_and_gradient_compass() -> None:
    """Verify reward breakdown card rendering, [G] key and chip toggles, and compass arrows."""
    data = _create_mock_replay_data()
    for fr in data["frames"]:
        bd = []
        gp = []
        for p in range(12):
            mag = 0.05 if p != 0 else 0.0  # Player 0 has flat gradient zone
            gp.append([0.707, 0.707, mag])
            bd.append({
                "r_ball_adv": 0.012,
                "r_pot_i": 0.005,
                "phi_role": 0.62,
                "phi_crowd": 0.08,
                "r_sparse": 0.0,
                "grad_phi_xy": [0.707, 0.707, mag],
            })
        fr["rew_breakdown"] = bd
        fr["grad_phi"] = gp

    viz = ReplayVisualizer(data, headless=True)
    assert not viz.show_gradient

    # 1. Toggle via keyboard 'G'
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_g, mod=0))
    viz.handle_events()
    assert viz.show_gradient

    # 2. Toggle via chip 'G:GRAD'
    viz.render_frame()
    chip_rect = viz.telemetry_chip_rects.get("G:GRAD")
    assert chip_rect is not None
    pygame.event.post(
        pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            pos=(chip_rect.centerx, chip_rect.centery),
            button=1,
        )
    )
    viz.handle_events()
    assert not viz.show_gradient

    # 3. Render frame with gradient compass ON
    viz.show_gradient = True
    surf = viz.render_frame()
    assert isinstance(surf, pygame.Surface)


def test_dynamic_role_tethers_and_presser_badges() -> None:
    """Verify live tactical badges (BALL, 1stP, 2ndP, COVR, CHSE, STUN, LNGE, GK) and role tethers."""
    data = _create_mock_replay_data()
    roles = ["GK", "1stP", "2ndP", "COVR", "CHSE", "BALL", "GK", "COVR", "STUN", "LNGE", "1stP", "2ndP"]
    targets = [
        [50.0, 420.0], [700.0, 420.0], [700.0, 420.0], [400.0, 300.0], [700.0, 420.0], [700.0, 420.0],
        [1350.0, 420.0], [1000.0, 500.0], [900.0, 400.0], [800.0, 300.0], [700.0, 420.0], [700.0, 420.0],
    ]
    for fr in data["frames"]:
        fr["role_state"] = list(roles)
        fr["role_targets"] = [list(t) for t in targets]

    viz = ReplayVisualizer(data, headless=True, show_radar=True, show_telemetry=True)
    surf = viz.render_frame()
    assert isinstance(surf, pygame.Surface)


def test_true_possession_and_timeline_stall_ribbon() -> None:
    """Verify true possession header and 4-color timeline stall ribbon."""
    data = _create_mock_replay_data()
    # Frame 0..4: Team A possession
    for f in range(5):
        data["frames"][f]["players"][2][3] |= 1  # FLAG_HAS_BALL for Team A
    # Frame 5..9: Team B possession
    for f in range(5, 10):
        data["frames"][f]["players"][8][3] |= 1  # FLAG_HAS_BALL for Team B
    # Frame 10..14: Loose moving ball
    for f in range(10, 15):
        data["frames"][f]["ball"] = [700.0, 420.0, -1, 10.0, 5.0]
    # Frame 15..19: Stationary ball (vx=0, vy=0 for 5 frames)
    for f in range(15, 20):
        data["frames"][f]["ball"] = [700.0, 420.0, -1, 0.0, 0.0]

    viz = ReplayVisualizer(data, headless=True)
    assert hasattr(viz, "possession_ribbon_surface")
    assert viz.possession_ribbon_surface is not None
    assert viz.possession_ribbon_surface.get_width() == viz.disp_field_w
    assert viz.possession_ribbon_surface.get_height() == 4

    surf = viz.render_frame()
    assert isinstance(surf, pygame.Surface)


def test_focused_player_action_readout() -> None:
    """Verify action outputs line in focused player card with highlight thresholds."""
    data = _create_mock_replay_data()
    for fr in data["frames"]:
        fr["actions"] = [[0.8, -0.2, 0.65, 0.85] for _ in range(12)]  # Active kick and tackle

    viz = ReplayVisualizer(data, headless=True, show_telemetry=True)
    viz.inspect_player_idx = 5
    surf = viz.render_frame()
    assert isinstance(surf, pygame.Surface)


def test_backward_compatibility_older_replays() -> None:
    """Verify older replays without new telemetry fields load and play smoothly without exceptions."""
    data = _create_mock_replay_data()
    for fr in data["frames"]:
        fr.pop("v_pred", None)
        fr.pop("rew_breakdown", None)
        fr.pop("grad_phi", None)
        fr.pop("role_state", None)
        fr.pop("role_targets", None)
        fr.pop("actions", None)

    viz = ReplayVisualizer(
        data,
        headless=True,
        show_critic=True,
        show_gradient=True,
        show_radar=True,
        show_telemetry=True,
        show_reward_hud=True,
    )
    for f in range(len(data["frames"])):
        viz.seek_to_frame(f)
        surf = viz.render_frame()
        assert isinstance(surf, pygame.Surface)


def test_env_replay_recording_telemetry_fields() -> None:
    """Verify FootballEnv records v_pred, rew_breakdown, flight state, role_state, role_targets, actions."""
    from dataclasses import replace
    import numpy as np
    from football_rl.config import DEFAULT_CONFIG
    from football_rl.env import FootballEnv

    replay_cfg = replace(DEFAULT_CONFIG.replay, enable_recording=True)
    cfg = replace(DEFAULT_CONFIG, replay=replay_cfg)
    env = FootballEnv(cfg)
    env.reset(seed=42)

    acts = np.zeros((12, 4), dtype=np.float64)
    v_preds = np.array([0.5 * (i - 6) for i in range(12)], dtype=np.float32)

    env.step_array(acts, v_preds=v_preds)

    assert env.last_v_preds is not None
    assert len(env.last_v_preds) == 12
    assert env.last_rew_breakdown is not None
    assert len(env.last_rew_breakdown) == 12
    assert "r_ball_adv" in env.last_rew_breakdown[0]
    assert "r_pot_i" in env.last_rew_breakdown[0]
    assert "phi_role" in env.last_rew_breakdown[0]
    assert "phi_crowd" in env.last_rew_breakdown[0]
    assert "grad_phi_xy" not in env.last_rew_breakdown[0]
    assert not hasattr(env, "last_grad_phi")

    assert env.last_role_states is not None
    assert len(env.last_role_states) == 12
    assert env.last_role_targets is not None
    assert len(env.last_role_targets) == 12
    assert env.last_actions is not None
    assert len(env.last_actions) == 12

    # Check replay_frames
    assert len(env.replay_frames) > 0
    rf = env.replay_frames[-1]
    assert "v_pred" in rf
    assert "rew_breakdown" in rf
    assert "grad_phi" not in rf
    assert isinstance(rf["inflight"], bool)
    assert len(rf["pk"]) == 12
    assert "role_state" in rf
    assert "role_targets" in rf
    assert "actions" in rf










def _flight_state_replay_data() -> Dict[str, Any]:
    """Mock replay whose frames carry the flight state needed to rebuild the potential field."""
    data = _create_mock_replay_data()
    for fr in data["frames"]:
        fr["players"][0] = [60.0, 400.0, 0.0, 0]
        fr["ball"] = [700.0, 420.0, -1, 0.0, 0.0]
        fr["box_radii"] = [195.0, 195.0]
        fr["inflight"] = False
        fr["pk"] = [0.0] * 12
        fr["grad_phi"] = [[0.707, 0.707, 0.05] for _ in range(12)]
    return data


def test_gradient_arrows_are_the_true_potential_gradient_when_flight_state_is_recorded() -> None:
    # Keeper A at (60, 400) facing a loose centre-spot ball: its potential is a Gaussian in the distance to the
    # goal-line bisector point (71.5714, 420); the gradient is (0.00074687, 0.00129089), i.e. direction
    # (0.501, 0.866) and magnitude 0.0015.
    viz = ReplayVisualizer(_flight_state_replay_data(), headless=True)
    arrows, legacy = viz.gradient_arrows(0)
    assert not legacy
    assert len(arrows) == 12
    assert arrows[0] == pytest.approx([0.501, 0.866, 0.0015], abs=1e-12)


def test_ball_carrier_has_a_flat_true_gradient() -> None:
    # The carrier potential is the constant carrier baseline, so its true gradient vanishes.
    data = _flight_state_replay_data()
    data["frames"][0]["players"][5] = [700.0, 420.0, 0.0, 1]
    data["frames"][0]["ball"] = [700.0, 420.0, 0, 0.0, 0.0]
    viz = ReplayVisualizer(data, headless=True)
    arrows, legacy = viz.gradient_arrows(0)
    assert not legacy
    assert arrows[5] == [0.0, 0.0, 0.0]


def test_gradient_arrows_are_cached_per_frame_index() -> None:
    viz = ReplayVisualizer(_flight_state_replay_data(), headless=True)
    first, _ = viz.gradient_arrows(3)
    second, _ = viz.gradient_arrows(3)
    assert first is second


def test_replays_without_flight_state_draw_the_stored_arrows_and_are_flagged_legacy() -> None:
    data = _flight_state_replay_data()
    for fr in data["frames"]:
        fr.pop("inflight")
        fr.pop("pk")
    viz = ReplayVisualizer(data, headless=True, show_gradient=True, show_telemetry=True)
    arrows, legacy = viz.gradient_arrows(0)
    assert legacy
    assert arrows[0] == [0.707, 0.707, 0.05]
    assert isinstance(viz.render_frame(), pygame.Surface)
