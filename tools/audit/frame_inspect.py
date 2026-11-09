"""Dump the full state of selected replay frames (read-only).

For each frame: ball state, every player's position, velocity, facing cosine to the
ball, holder/ghost/stun/lunge flags, presser role state, PBRS target and gradient,
action, step reward and reward breakdown, plus close player pairs.

Frame numbers are raw 60 Hz indices by default; with --viewer they are indices into
the replay viewer's cinematic timeline (what the viewer's frame counter shows).

Usage:
  python tools/audit/frame_inspect.py (REPLAY | --match ID) --frames 760 5531
         [--viewer] [--close 65] [--json OUT]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.audit._common import (  # noqa: E402
    FLAG_GHOST,
    FLAG_HOLDER,
    FLAG_LUNGE,
    FLAG_STUN,
    PLAYERS_PER_TEAM,
    REPO_ROOT,
    SLOT_NAMES,
    TEAM_LABELS,
    ensure_repo_on_path,
    load_json,
    select_replays,
    write_json,
)

__all__ = ["get_frame", "inspect_frame", "main"]


def _viewer_timeline(replay: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand raw frames exactly as the replay viewer does (no window is opened)."""
    ensure_repo_on_path()
    from types import SimpleNamespace

    from football_rl.replay import ReplayViewer  # local import: pulls in pygame

    meta = replay.get("metadata") or {}
    host = SimpleNamespace(field_w=float(meta.get("field_width", 1400.0)))  # only attribute the builder reads
    return ReplayViewer._build_cinematic_timeline(host, list(replay.get("frames") or []))  # type: ignore[arg-type]


def get_frame(replay: Dict[str, Any], index: int, viewer: bool = False) -> Tuple[Dict[str, Any], Optional[int]]:
    """Return (frame, raw_index). raw_index is None for synthetic cinematic viewer frames."""
    raw = list(replay.get("frames") or [])
    seq = _viewer_timeline(replay) if viewer else raw
    if index < 0 or index >= len(seq):
        raise IndexError(f"frame {index} out of range (0..{len(seq) - 1})")
    fr = seq[index]
    if not viewer:
        return fr, index
    if fr.get("cinematic") is not None:
        return fr, None
    by_number = {f.get("frame"): i for i, f in enumerate(raw)}
    return fr, by_number.get(fr.get("frame"))


def _pick(seq: Optional[Sequence[Any]], i: int) -> Any:
    if not seq or i >= len(seq):
        return None
    return seq[i]


def inspect_frame(frame: Dict[str, Any], close_threshold: float = 65.0) -> Dict[str, Any]:
    """Structured dump of one frame dict."""
    b = frame.get("ball") or [0.0, 0.0, -1, 0.0, 0.0]
    bx, by = float(b[0]), float(b[1])
    bvx = float(b[3]) if len(b) > 3 else 0.0
    bvy = float(b[4]) if len(b) > 4 else 0.0
    players_raw = frame.get("players") or []
    holder = None
    players: List[Dict[str, Any]] = []
    for i, p in enumerate(players_raw):
        px, py = float(p[0]), float(p[1])
        ang = float(p[2]) if len(p) > 2 else 0.0
        flags = int(p[3]) if len(p) > 3 else 0
        vx = float(p[4]) if len(p) > 4 else 0.0
        vy = float(p[5]) if len(p) > 5 else 0.0
        d = math.hypot(bx - px, by - py)
        cos_b = None if d < 1e-9 else (math.cos(ang) * (bx - px) + math.sin(ang) * (by - py)) / d
        is_holder = bool(flags & FLAG_HOLDER)
        if is_holder and holder is None:
            holder = i
        players.append({
            "idx": i,
            "team": i // PLAYERS_PER_TEAM,
            "slot": SLOT_NAMES[i % PLAYERS_PER_TEAM],
            "holder": is_holder,
            "ghost": bool(flags & FLAG_GHOST),
            "stun": bool(flags & FLAG_STUN),
            "lunge": bool(flags & FLAG_LUNGE),
            "pos": [px, py],
            "vel": [vx, vy],
            "speed": math.hypot(vx, vy),
            "angle_deg": math.degrees(ang),
            "d_ball": d,
            "cos_to_ball": cos_b,
            "role_state": _pick(frame.get("role_state"), i),
            "target": _pick(frame.get("role_targets"), i),
            "grad": _pick(frame.get("grad_phi"), i),
            "action": _pick(frame.get("actions"), i),
            "reward": _pick(frame.get("player_rewards"), i),
            "breakdown": _pick(frame.get("rew_breakdown"), i),
        })
    pairs = []
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            pi, pj = players[i]["pos"], players[j]["pos"]
            d = math.hypot(pi[0] - pj[0], pi[1] - pj[1])
            if d <= close_threshold:
                pairs.append({"i": i, "j": j, "dist": d})
    pairs.sort(key=lambda q: (q["dist"], q["i"], q["j"]))
    return {
        "frame": frame.get("frame"),
        "time": frame.get("time"),
        "half": frame.get("half"),
        "team_side": frame.get("team_side"),
        "box_radii": frame.get("box_radii"),
        "cinematic": frame.get("cinematic") is not None,
        "events": frame.get("events") or [],
        "ball": {"pos": [bx, by], "vel": [bvx, bvy], "speed": math.hypot(bvx, bvy), "holder": holder,
                 "team_color": b[2] if len(b) > 2 else None},
        "players": players,
        "close_pairs": pairs,
    }


def _f(v: Any, nd: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, (list, tuple)):
        return "(" + ",".join(_f(x, nd) for x in v) + ")"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _lines(label: str, info: Dict[str, Any]) -> List[str]:
    ball = info["ball"]
    out = [
        f"== {label}: raw_frame={info['frame']} t={_f(info['time'])}s half={info['half']} sides={info['team_side']} "
        f"box={info['box_radii']} cinematic={info['cinematic']}",
        f"   ball pos={_f(ball['pos'], 1)} vel={_f(ball['vel'], 1)} |{ball['speed']:.1f}| holder={ball['holder']}",
    ]
    for p in info["players"]:
        fl = "".join(c for c, on in (("B", p["holder"]), ("G", p["ghost"]), ("S", p["stun"]), ("L", p["lunge"])) if on)
        bk = p["breakdown"] or {}
        out.append(
            f"   {p['idx']:2d} {TEAM_LABELS[p['team']][0]}-{p['slot']} [{p['role_state'] or '-':4s}] {fl:3s} "
            f"pos={_f(p['pos'], 1)} v=|{p['speed']:5.1f}| db={p['d_ball']:6.1f} cos_b={_f(p['cos_to_ball'])} "
            f"tgt={_f(p['target'], 1)} grad={_f(p['grad'])} act={_f(p['action'])} r={_f(p['reward'], 4)} "
            f"pot={_f(bk.get('r_pot_i'), 4)} phi={_f(bk.get('phi_role'), 3)} crowd={_f(bk.get('phi_crowd'), 3)}"
        )
    if info["close_pairs"]:
        out.append("   close: " + ", ".join(f"{q['i']}-{q['j']}:{q['dist']:.1f}" for q in info["close_pairs"][:12]))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replay", nargs="?", default=None)
    ap.add_argument("--match", type=int, default=None, help="resolve replay by match id in --replays-dir")
    ap.add_argument("--replays-dir", default=str(REPO_ROOT / "replays"))
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--viewer", action="store_true", help="frame numbers are viewer (cinematic) indices")
    ap.add_argument("--close", type=float, default=65.0, help="close-pair distance threshold")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if args.replay:
        path = Path(args.replay)
    elif args.match is not None:
        sel = select_replays(args.replays_dir, match_id=args.match)
        if not sel:
            print(f"no replay for match {args.match} in {args.replays_dir}", file=sys.stderr)
            return 2
        path = sel[0]
    else:
        ap.error("give a replay path or --match")
        return 2
    replay = load_json(path)
    dumps = []
    for idx in args.frames:
        try:
            fr, raw_idx = get_frame(replay, idx, viewer=args.viewer)
        except IndexError as exc:
            print(f"frame {idx}: {exc}", file=sys.stderr)
            continue
        info = inspect_frame(fr, close_threshold=args.close)
        info["requested_index"] = idx
        info["raw_index"] = raw_idx
        dumps.append(info)
        print("\n".join(_lines(f"{'viewer' if args.viewer else 'raw'}[{idx}]", info)))
    if args.json:
        write_json(args.json, {"replay": str(path), "frames": dumps})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
