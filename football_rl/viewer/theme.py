"""Replay viewer palette, player-state flag bits, event colours and the critic-value colour scale."""

from __future__ import annotations

from typing import Dict, Tuple


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


def _get_critic_color(v: float) -> Tuple[int, int, int]:
    """Return a diverging color: Red/Magenta for V < 0, Grey for V ≈ 0, Cyan/Green for V > 0."""
    if v > 0.05:
        t = min(1.0, (v - 0.05) / 1.5)
        r = int(150 + t * (45 - 150))
        g = int(160 + t * (240 - 160))
        b = int(175 + t * (160 - 175))
        return (r, g, b)
    elif v < -0.05:
        t = min(1.0, (-v - 0.05) / 1.5)
        r = int(150 + t * (255 - 150))
        g = int(160 + t * (60 - 160))
        b = int(175 + t * (100 - 175))
        return (r, g, b)
    else:
        return (150, 160, 175)
