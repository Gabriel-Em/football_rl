"""Specification tests for the read-only audit tooling in tools/audit/.

All expected values are hand-computed from tiny synthetic replay / history /
source fixtures. No production formulas are re-implemented here.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audit import frame_inspect, orphan_scan, perf_profile, replay_metrics, run_all, training_progress  # noqa: E402

# ----------------------------------------------------------------------------
# Synthetic replay fixture helpers
# ----------------------------------------------------------------------------
# Team A (indices 0..5) defends the left goal in half 1; team B (6..11) the right.
# Slot order per team: GK, D1, D2, M1, M2, ST.
DEFAULT_A = [(40, 420), (300, 300), (300, 540), (500, 300), (500, 540), (650, 420)]
DEFAULT_B = [(1360, 420), (1100, 540), (1100, 300), (900, 540), (900, 300), (750, 420)]
FAR_BALL = (700.0, 30.0)  # far from every default player position


def make_frame(
    idx: int,
    positions: Optional[Sequence[Sequence[float]]] = None,
    ball: Sequence[float] = FAR_BALL,
    holder: Optional[int] = None,
    team_side: Sequence[int] = (0, 1),
    half: int = 1,
    role_state: Optional[List[str]] = None,
    passes: Sequence[int] = (0, 0),
    events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    pos = list(positions) if positions is not None else DEFAULT_A + DEFAULT_B
    players = []
    for p_i, (x, y) in enumerate(pos):
        flags = 1 if holder == p_i else 0
        players.append([float(x), float(y), 0.0, flags, 0.0, 0.0])
    ball_team = -1 if holder is None else (0 if holder < 6 else 1)
    return {
        "frame": idx + 1,
        "time": round((idx + 1) / 60.0, 3),
        "half": half,
        "team_side": list(team_side),
        "box_radii": [195.0, 195.0],
        "players": players,
        "ball": [float(ball[0]), float(ball[1]), ball_team, 0.0, 0.0],
        "player_rewards": [0.0] * 12,
        "role_state": role_state,
        "grad_phi": None,
        "role_targets": None,
        "actions": None,
        "rew_breakdown": None,
        "stats": {"passes": list(passes)},
        "events": events or [],
    }


def make_replay(frames: List[Dict[str, Any]], match_id: int = 7, generation: int = 3) -> Dict[str, Any]:
    return {
        "metadata": {
            "match_id": match_id,
            "generation": generation,
            "physics_hz": 60,
            "field_width": 1400.0,
            "field_height": 840.0,
            "goal_y_min": 320.0,
            "goal_y_max": 520.0,
            "keeper_box_radius": 195.0,
            "player_radius": 18.0,
            "ball_radius": 10.0,
        },
        "frames": frames,
    }


def with_moves(moves: Dict[int, Sequence[float]]) -> List[Sequence[float]]:
    pos = list(DEFAULT_A + DEFAULT_B)
    for k, v in moves.items():
        pos[k] = v
    return pos


def metrics(replay: Dict[str, Any], **kw: Any) -> Dict[str, Any]:
    return replay_metrics.compute_match_metrics(replay, **kw)


# ----------------------------------------------------------------------------
# replay_metrics
# ----------------------------------------------------------------------------
class TestReplayMetrics:
    def test_empty_match_yields_none_rates_and_zero_counts(self) -> None:
        m = metrics(make_replay([]))
        assert m["n_frames"] == 0
        assert m["ball_hold_rate"] is None
        assert m["mean_cluster_size"] is None
        assert m["cluster_frames"] == 0
        assert m["gk_pickups"] == [0, 0]
        assert m["gk_hold_mean_s"] == [None, None]
        assert m["inversion_rate"] == [None, None]
        assert m["possession_share"] == [None, None]
        assert m["presser_trailing_rate"] == [None, None]
        assert m["gk_facing_blocked_frames"] == [0, 0] and m["gk_dives"] == [0, 0]
        assert m["longest_hold_spell_s"] is None

    def test_gk_pickup_count_and_hold_duration_including_run_at_match_end(self) -> None:
        frames = [
            make_frame(0),
            make_frame(1, holder=0),
            make_frame(2, holder=0),
            make_frame(3, holder=0),
            make_frame(4),
            make_frame(5, holder=0),  # second spell still open at final frame
        ]
        m = metrics(make_replay(frames))
        assert m["gk_pickups"] == [2, 0]
        # spells of 3 and 1 frames -> mean 2 frames at 60 Hz
        assert m["gk_hold_mean_s"][0] == pytest.approx(2.0 / 60.0)
        assert m["gk_hold_mean_s"][1] is None
        assert m["ball_hold_rate"] == pytest.approx(4.0 / 6.0)

    def test_gk_hold_from_first_frame_counts_as_pickup(self) -> None:
        m = metrics(make_replay([make_frame(0, holder=6), make_frame(1, holder=6)]))
        assert m["gk_pickups"] == [0, 1]
        assert m["gk_hold_mean_s"][1] == pytest.approx(2.0 / 60.0)

    def test_gk_at_end_wall_distance_and_pinned_rate_across_side_switch(self) -> None:
        f0 = make_frame(0, positions=with_moves({0: (18, 420), 6: (1382, 420)}))
        # Second half: sides swapped, A now defends the right goal.
        f1 = make_frame(1, positions=with_moves({0: (1382, 420), 6: (1360, 420)}), team_side=(1, 0), half=2)
        m = metrics(make_replay([f0, f1]))
        # A: 18 then 18 -> 18. B: 18 (x=1382 vs right line) then 1360 vs left line -> 1360.
        assert m["gk_dist_goal_line_mean"][0] == pytest.approx(18.0)
        assert m["gk_dist_goal_line_mean"][1] == pytest.approx((18.0 + 1360.0) / 2.0)
        assert m["gk_wall_pinned_rate"][0] == pytest.approx(1.0)
        assert m["gk_wall_pinned_rate"][1] == pytest.approx(0.5)
        # goal centre at y=420 -> same as goal-line distance here
        assert m["gk_dist_goal_center_mean"][0] == pytest.approx(18.0)

    def test_gk_near_miss_window_is_open_below_and_closed_above(self) -> None:
        ball = (100.0, 420.0)  # loose, inside A's box (100 <= 195)
        frames = [
            make_frame(0, positions=with_moves({0: (70, 420)}), ball=ball),  # d=30 -> near miss
            make_frame(1, positions=with_moves({0: (72, 420)}), ball=ball),  # d=28 -> pickup range, not a miss
            make_frame(2, positions=with_moves({0: (62, 420)}), ball=ball),  # d=38 -> window edge, near miss
            make_frame(3, positions=with_moves({0: (60, 420)}), ball=ball),  # d=40 -> outside window
        ]
        m = metrics(make_replay(frames), pickup_distance=28.0, near_miss_window=10.0)
        assert m["gk_near_miss_frames"] == [2, 0]

    def test_role_inversion_strict_margin_opposing_half_and_side_switch(self) -> None:
        frames = [
            make_frame(0, positions=with_moves({1: (751, 300), 5: (720, 420)})),  # A inverted
            make_frame(1, positions=with_moves({1: (750, 300), 5: (720, 420)})),  # tie: exactly margin -> not
            make_frame(2, positions=with_moves({1: (600, 300), 5: (400, 420)})),  # own half -> not
            # B (defends right): ST canon 1400-680=720, D2 canon 1400-649=751 -> inverted; B holds ball
            make_frame(3, positions=with_moves({8: (649, 300), 11: (680, 420)}), holder=11),
        ]
        m = metrics(make_replay(frames), inversion_margin=30.0)
        assert m["inversion_rate"] == [pytest.approx(0.25), pytest.approx(0.25)]
        # Out of possession: A never holds (4 frames, 1 inverted); B is out of possession in frames 0..2 only.
        assert m["inversion_rate_oop"] == [pytest.approx(0.25), pytest.approx(0.0)]

    def test_inversion_after_side_switch_uses_new_own_goal(self) -> None:
        # Half 2: A defends right. A's ST at x=680 (canon 720), D1 at x=649 (canon 751) -> inverted.
        f = make_frame(0, positions=with_moves({1: (649, 300), 5: (680, 420), 11: (1000, 420), 8: (1000, 600)}),
                       team_side=(1, 0), half=2)
        m = metrics(make_replay([f]))
        assert m["inversion_rate"][0] == pytest.approx(1.0)

    def test_cluster_counts_players_exactly_on_radius(self) -> None:
        ball = (700.0, 420.0)
        ring = [(790, 420), (610, 420), (700, 510), (700, 330), (754, 492), (646, 348)]  # all exactly 90 away
        far = [(100, 100), (100, 740), (1300, 100), (1300, 740), (200, 420), (1200, 420)]
        pos0 = ring[:3] + far[:3] + ring[3:] + far[3:]
        pos1 = list(pos0)
        pos1[0] = (790.5, 420)  # 90.5 away -> leaves the cluster
        m = metrics(make_replay([make_frame(0, positions=pos0, ball=ball), make_frame(1, positions=pos1, ball=ball)]),
                    cluster_radius=90.0, cluster_n=6, near_radius=90.0)
        assert m["cluster_frames"] == 1
        assert m["cluster_frame_rate"] == pytest.approx(0.5)
        assert m["mean_cluster_size"] == pytest.approx(5.5)
        # per team within 90: A has 3 then 2, B has 3 then 3
        assert m["near_ball_mean"] == [pytest.approx(2.5), pytest.approx(3.0)]

    def test_width_and_depth_of_outfield_shape(self) -> None:
        m = metrics(make_replay([make_frame(0)]))
        assert m["width_mean"] == [pytest.approx(240.0), pytest.approx(240.0)]
        assert m["depth_mean"] == [pytest.approx(350.0), pytest.approx(350.0)]

    def test_possession_share_and_passes_from_last_frame(self) -> None:
        frames = [
            make_frame(0, holder=3),
            make_frame(1, holder=3),
            make_frame(2, holder=9),
            make_frame(3, passes=(3, 1)),
        ]
        m = metrics(make_replay(frames))
        assert m["possession_share"] == [pytest.approx(0.5), pytest.approx(0.25)]
        assert m["loose_share"] == pytest.approx(0.25)
        assert m["passes"] == [3, 1]

    def test_presser_role_share(self) -> None:
        rs_st = ["GK", "COVR", "COVR", "COVR", "COVR", "1stP"] + ["GK"] + ["COVR"] * 5
        rs_d1 = ["GK", "1stP", "COVR", "COVR", "COVR", "COVR"] + ["GK"] + ["COVR"] * 5
        frames = [make_frame(0, role_state=rs_st), make_frame(1, role_state=rs_st), make_frame(2, role_state=rs_d1)]
        m = metrics(make_replay(frames))
        share_a = m["presser_share"][0]
        assert share_a["ST"] == pytest.approx(2.0 / 3.0)
        assert share_a["D1"] == pytest.approx(1.0 / 3.0)
        assert share_a["M1"] == pytest.approx(0.0)
        assert m["presser_share"][1] == {}

    def test_presser_trailing_rate_strict_margin(self) -> None:
        rs = ["GK", "COVR", "COVR", "COVR", "COVR", "1stP"] + ["GK"] + ["COVR"] * 5
        ball = (500.0, 420.0)
        frames = [
            make_frame(0, positions=with_moves({5: (560, 420)}), ball=ball, role_state=rs),  # trails by 60
            make_frame(1, positions=with_moves({5: (550, 420)}), ball=ball, role_state=rs),  # exactly 50 -> not
            make_frame(2, positions=with_moves({5: (450, 420)}), ball=ball, role_state=rs),  # goal-side
        ]
        m = metrics(make_replay(frames), trail_margin=50.0)
        assert m["presser_trailing_rate"][0] == pytest.approx(1.0 / 3.0)
        assert m["presser_trailing_rate"][1] is None

    def test_gk_facing_blocked_frames(self) -> None:
        ball = (100.0, 420.0)
        f_away = make_frame(0, positions=with_moves({0: (80, 420)}), ball=ball)
        f_away["players"][0][2] = math.pi  # facing -x, ball is at +x -> cos = -1
        f_face = make_frame(1, positions=with_moves({0: (80, 420)}), ball=ball)  # angle 0 faces the ball
        f_far = make_frame(2, positions=with_moves({0: (60, 420)}), ball=ball)
        f_far["players"][0][2] = math.pi  # facing away but 40u away -> not in pickup range
        m = metrics(make_replay([f_away, f_face, f_far]), pickup_distance=28.0)
        assert m["gk_facing_blocked_frames"] == [1, 0]

    def test_gk_dive_starts_counted_from_lunge_flag(self) -> None:
        frames = [make_frame(i) for i in range(5)]
        for i in (1, 2, 4):
            frames[i]["players"][0][3] = 8
        m = metrics(make_replay(frames))
        assert m["gk_dives"] == [2, 0]

    def test_longest_hold_spell_bridges_short_gaps_only(self) -> None:
        seq = [3, 3, None, 3, 3, 3, None, None, None, 3, 9, 3]
        frames = [make_frame(i, holder=h) for i, h in enumerate(seq)]
        m = metrics(make_replay(frames), spell_gap_frames=2)
        assert m["longest_hold_spell_s"] == pytest.approx(6.0 / 60.0)
        assert metrics(make_replay([make_frame(0)]))["longest_hold_spell_s"] == pytest.approx(0.0)

    def test_missing_velocity_columns_are_tolerated(self) -> None:
        f = make_frame(0)
        f["players"] = [p[:4] for p in f["players"]]
        m = metrics(make_replay([f]))
        assert m["n_frames"] == 1

    def test_select_replays_filters_by_match_generation_and_last(self, tmp_path: Path) -> None:
        names = [
            "match_0002_gen0001_score3_0_20261108_044233.json",
            "match_0009_gen0002_score13_5_20261108_044258.json",
            "match_0011_gen0002_score9_0_20261108_044300.json",
        ]
        for n in names:
            (tmp_path / n).write_text("{}")
        (tmp_path / "matches_summary.md").write_text("x")
        sel = replay_metrics.select_replays(tmp_path, last=2)
        assert [p.name for p in sel] == names[1:]
        assert [p.name for p in replay_metrics.select_replays(tmp_path, generation=2)] == names[1:]
        assert [p.name for p in replay_metrics.select_replays(tmp_path, match_id=9)] == [names[1]]
        assert replay_metrics.select_replays(tmp_path, match_id=999) == []

    def test_cli_writes_json_and_concise_summary(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        p = tmp_path / "match_0007_gen0003_score1_0_x.json"
        p.write_text(json.dumps(make_replay([make_frame(0), make_frame(1, holder=0)])))
        out = tmp_path / "m.json"
        rc = replay_metrics.main([str(p), "--json", str(out)])
        assert rc == 0
        data = json.loads(out.read_text())
        assert data["matches"][0]["gk_pickups"] == [1, 0]
        assert len(capsys.readouterr().out.splitlines()) <= 25


# ----------------------------------------------------------------------------
# training_progress
# ----------------------------------------------------------------------------
def _hist_match(gen: int, interest: float, goals: int, passes: int, shots: Any, tackles: int, saves: int,
                loose: float, stat: int) -> Dict[str, Any]:
    return {
        "generation": gen, "interest_score": interest, "goals": goals, "completed_passes": passes,
        "shots": shots, "successful_tackles": tackles, "keeper_saves": saves,
        "true_possession_pct": [10.0, 10.0, loose], "stationary_ball_frames": stat,
    }


class TestTrainingProgress:
    def test_buckets_hand_computed(self) -> None:
        hist = {"matches": [
            _hist_match(1, 10.0, 1, 2, [1, 1], 3, 0, 80.0, 100),
            _hist_match(1, 20.0, 3, 4, [2, 0], 5, 2, 90.0, 300),
            _hist_match(2, 60.0, 2, 0, [0, 0], 1, 1, 70.0, 200),
            _hist_match(3, 5.0, 0, 1, [1, 0], 0, 0, 50.0, 0),
        ]}
        rows = training_progress.bucket_progress(hist, bucket=2)
        assert [(r["gen_start"], r["gen_end"], r["n"]) for r in rows] == [(1, 2, 3), (3, 3, 1)]
        r0 = rows[0]
        assert r0["interest_mean"] == pytest.approx(30.0)
        assert r0["interest_median"] == pytest.approx(20.0)
        assert r0["interest_max"] == pytest.approx(60.0)
        assert r0["goals"] == pytest.approx(2.0)
        assert r0["passes"] == pytest.approx(2.0)
        assert r0["shots"] == pytest.approx(4.0 / 3.0)  # (2 + 2 + 0) / 3
        assert r0["tackles"] == pytest.approx(3.0)
        assert r0["saves"] == pytest.approx(1.0)
        assert r0["loose_pct"] == pytest.approx(80.0)
        assert r0["stationary_frames"] == pytest.approx(200.0)

    def test_empty_history_and_missing_keys(self) -> None:
        assert training_progress.bucket_progress({"matches": []}, bucket=100) == []
        assert training_progress.bucket_progress({}, bucket=100) == []
        rows = training_progress.bucket_progress({"matches": [{"generation": 4, "interest_score": 1.0, "shots": 3}]},
                                                 bucket=10)
        assert rows[0]["shots"] == pytest.approx(3.0)  # scalar shots accepted
        assert rows[0]["goals"] == pytest.approx(0.0)
        assert rows[0]["loose_pct"] is None

    def test_generation_range_filter_and_bucket_validation(self) -> None:
        hist = {"matches": [_hist_match(g, float(g), 0, 0, [0, 0], 0, 0, 0.0, 0) for g in range(1, 11)]}
        rows = training_progress.bucket_progress(hist, bucket=5, gen_min=3, gen_max=7)
        assert [(r["gen_start"], r["gen_end"], r["n"]) for r in rows] == [(3, 7, 5)]
        with pytest.raises(ValueError):
            training_progress.bucket_progress(hist, bucket=0)


# ----------------------------------------------------------------------------
# frame_inspect
# ----------------------------------------------------------------------------
class TestFrameInspect:
    def test_close_pairs_inclusive_threshold_and_deterministic_tie_order(self) -> None:
        pos = with_moves({1: (300, 300), 2: (300, 336), 3: (336, 300)})  # D1-D2 = 36, D1-M1 = 36 (tie)
        rep = make_replay([make_frame(0, positions=pos)])
        out = frame_inspect.inspect_frame(rep["frames"][0], close_threshold=36.0)
        pairs = [(p["i"], p["j"], p["dist"]) for p in out["close_pairs"]]
        assert pairs[:2] == [(1, 2, pytest.approx(36.0)), (1, 3, pytest.approx(36.0))]

    def test_player_on_ball_has_no_facing_cosine_and_missing_telemetry_is_none(self) -> None:
        pos = with_moves({5: (700, 30)})
        out = frame_inspect.inspect_frame(make_frame(0, positions=pos, ball=(700, 30)))
        st = out["players"][5]
        assert st["d_ball"] == pytest.approx(0.0)
        assert st["cos_to_ball"] is None
        assert st["grad"] is None and st["action"] is None and st["role_state"] is None
        assert st["slot"] == "ST" and st["team"] == 0

    def test_facing_cosine_and_holder_flag(self) -> None:
        pos = with_moves({3: (600, 30)})
        f = make_frame(0, positions=pos, ball=(700, 30), holder=3)
        out = frame_inspect.inspect_frame(f)
        assert out["players"][3]["cos_to_ball"] == pytest.approx(1.0)  # angle 0 faces +x toward ball
        assert out["players"][3]["holder"] is True
        assert out["ball"]["holder"] == 3

    def test_frame_index_out_of_range_raises(self) -> None:
        rep = make_replay([make_frame(0)])
        with pytest.raises(IndexError):
            frame_inspect.get_frame(rep, 5)
        with pytest.raises(IndexError):
            frame_inspect.get_frame(make_replay([]), 0)

    def test_viewer_index_maps_through_cinematic_timeline(self) -> None:
        frames = [make_frame(i) for i in range(5)]
        rep = make_replay(frames)
        fr, raw = frame_inspect.get_frame(rep, 3, viewer=True)
        assert raw == 3  # no transition events -> identity mapping
        frames[2]["events"] = [{"type": "GOAL!", "team": 0, "agent": "a_atk_1"}]
        fr, raw = frame_inspect.get_frame(make_replay(frames), 4 + 270, viewer=True)
        assert fr["frame"] == frames[4]["frame"]
        assert raw == 4


# ----------------------------------------------------------------------------
# orphan_scan
# ----------------------------------------------------------------------------
CFG_SRC = '''
from dataclasses import dataclass, field

@dataclass(frozen=True)
class MechCfg:
    used_a: float = 28.0
    unused_b: float = 5.0
    pass_threat_weight: float = 6.0
    pass_threat_coeff: float = 6.0
    flag: bool = True
    zero_x: float = 0.0
    zero_y: float = 0.0
    test_only: float = 3.5

DEFAULT = MechCfg()
'''

MOD_SRC = '''
import numpy as np

class Env:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dead_timer = np.zeros(2)
        self.live = 1.0
        self.buf = [0]
        self.out_only = np.zeros(2)
        self.out_dest = np.zeros(2)

    def step(self):
        self.dead_timer[0] = 1.0
        self.dead_timer.fill(0.0)
        self.dead_timer += 1.0
        np.maximum(0.0, self.out_only, out=self.out_only)
        np.maximum(0.0, self.live, out=self.out_dest)
        x = self.cfg.used_a + getattr(self.cfg, "pass_threat_weight", 0)
        self.buf.append(2)
        return x + self.live + self.buf[0]

    step_alias = step
'''


class TestOrphanScan:
    def _pkg(self, tmp_path: Path) -> Path:
        pkg = tmp_path / "pkg"
        (pkg / "tests").mkdir(parents=True)
        (pkg / "config.py").write_text(CFG_SRC)
        (pkg / "module.py").write_text(MOD_SRC)
        (pkg / "tests" / "test_x.py").write_text("def test_a(cfg):\n    assert cfg.test_only\n")
        return pkg

    def test_unread_config_fields_and_test_only_reads(self, tmp_path: Path) -> None:
        res = orphan_scan.scan(self._pkg(tmp_path))
        assert sorted(res["unread_config_fields"]) == [
            "MechCfg.flag", "MechCfg.pass_threat_coeff", "MechCfg.unused_b", "MechCfg.zero_x", "MechCfg.zero_y",
        ]
        assert res["config_fields_read_only_in_tests"] == ["MechCfg.test_only"]

    def test_write_only_attributes_treat_subscript_mutator_augassign_and_out_kw_as_writes(self, tmp_path: Path) -> None:
        res = orphan_scan.scan(self._pkg(tmp_path))
        # out_only appears both as positional input (a read) and out= target, so it is read;
        # out_dest is only ever an out= target, so it is write-only.
        assert res["write_only_attributes"] == ["Env.dead_timer", "Env.out_dest"]

    def test_duplicate_value_groups_exclude_trivial_values(self, tmp_path: Path) -> None:
        res = orphan_scan.scan(self._pkg(tmp_path))
        groups = res["duplicate_value_groups"]
        assert len(groups) == 1
        g = groups[0]
        assert g["class"] == "MechCfg" and g["value"] == 6.0
        assert g["fields"] == ["pass_threat_coeff", "pass_threat_weight"]
        assert g["similarity"] == pytest.approx(0.5)

    def test_unreferenced_class_alias_is_reported(self, tmp_path: Path) -> None:
        res = orphan_scan.scan(self._pkg(tmp_path))
        assert res["unused_class_aliases"] == ["Env.step_alias"]

    def test_missing_config_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            orphan_scan.scan(tmp_path)

    def test_real_codebase_reports_are_consistent_with_source_text(self) -> None:
        """Every reported write-only attribute must never appear in a plain read form anywhere in the package."""
        src_dir = REPO_ROOT / "football_rl"
        res = orphan_scan.scan(src_dir)
        assert res["files_scanned"] > 5
        text = "\n".join(p.read_text() for p in src_dir.rglob("*.py") if "__pycache__" not in p.parts)
        for item in res["write_only_attributes"]:
            attr = item.split(".", 1)[1]
            assert f"self.{attr}" in text  # it is written somewhere
            assert not re.search(rf'(getattr|hasattr)\([^)]*["\']{attr}["\']', text)  # no string-based reads
        for item in res["unread_config_fields"]:
            field_name = item.split(".", 1)[1]
            assert f".{field_name}" not in text.replace(f"{field_name}:", "")


# ----------------------------------------------------------------------------
# perf_profile
# ----------------------------------------------------------------------------
class TestPerfProfile:
    def test_call_counts_follow_frame_skip_and_recording_flag(self) -> None:
        on = perf_profile.profile_env(steps=3, seed=0, record=True, count_numpy=False)
        off = perf_profile.profile_env(steps=3, seed=0, record=False, count_numpy=False)
        assert on["components"]["_physics_substep"]["calls"] == 12
        assert on["components"]["_compute_potential_shaping"]["calls"] == 3
        assert on["components"]["_record_replay_frame"]["calls"] == 12
        assert off["components"]["_record_replay_frame"]["calls"] == 0
        for comp in on["components"].values():
            assert comp["total_ms"] >= 0.0
        assert on["total_ms_per_step"] > 0.0

    def test_numpy_call_counts_are_seed_deterministic(self) -> None:
        a = perf_profile.profile_env(steps=2, seed=11, record=False, count_numpy=True)
        b = perf_profile.profile_env(steps=2, seed=11, record=False, count_numpy=True)
        assert a["numpy_calls_per_step"] == b["numpy_calls_per_step"]
        assert a["numpy_calls_per_step"] > 0
        assert a["numpy_calls_by_component"] == b["numpy_calls_by_component"]

    def test_invalid_step_count_raises(self) -> None:
        with pytest.raises(ValueError):
            perf_profile.profile_env(steps=0, seed=0)


# ----------------------------------------------------------------------------
# run_all
# ----------------------------------------------------------------------------
class TestRunAll:
    def test_compute_deltas_hand_values_and_missing_keys(self) -> None:
        before = {"ball_hold_rate": 0.2, "gk_pickups_per_match": 6.0, "gone": 1.0}
        after = {"ball_hold_rate": 0.25, "gk_pickups_per_match": 4.0, "new": 2.0}
        d = run_all.compute_deltas(before, after)
        assert d["ball_hold_rate"]["delta"] == pytest.approx(0.05)
        assert d["gk_pickups_per_match"]["delta"] == pytest.approx(-2.0)
        assert d["new"] == {"before": None, "after": 2.0, "delta": None}
        assert d["gone"] == {"before": 1.0, "after": None, "delta": None}
        nan_d = run_all.compute_deltas({"x": None}, {"x": 1.0})
        assert nan_d["x"]["delta"] is None

    def test_end_to_end_baseline_then_delta(self, tmp_path: Path) -> None:
        rdir = tmp_path / "replays"
        rdir.mkdir()
        rep = make_replay([make_frame(0), make_frame(1, holder=0), make_frame(2, holder=0)])
        (rdir / "match_0007_gen0003_score1_0_x.json").write_text(json.dumps(rep))
        hist = tmp_path / "hist.json"
        hist.write_text(json.dumps({"matches": [_hist_match(1, 10.0, 1, 2, [1, 1], 3, 0, 80.0, 100)]}))
        pkg = TestOrphanScan()._pkg(tmp_path)
        out = tmp_path / "metrics"
        args = ["--out-dir", str(out), "--replays-dir", str(rdir), "--history", str(hist),
                "--source-dir", str(pkg), "--skip-perf", "--last", "3"]
        assert run_all.main(args + ["--save-baseline"]) == 0
        assert (out / "latest.json").exists() and (out / "latest.md").exists() and (out / "baseline.json").exists()
        assert run_all.main(args) == 0
        report = json.loads((out / "latest.json").read_text())
        g = report["guards"]
        assert g["ball_hold_rate"] == pytest.approx(2.0 / 3.0)
        assert g["gk_pickups_per_match"] == pytest.approx(1.0)
        assert report["deltas"]["ball_hold_rate"]["delta"] == pytest.approx(0.0)
        md = (out / "latest.md").read_text()
        assert "ball_hold_rate" in md

    def test_missing_inputs_are_reported_not_fatal(self, tmp_path: Path) -> None:
        out = tmp_path / "metrics"
        rc = run_all.main(["--out-dir", str(out), "--replays-dir", str(tmp_path / "none"),
                           "--history", str(tmp_path / "none.json"), "--source-dir", str(tmp_path / "nopkg"),
                           "--skip-perf"])
        assert rc == 0
        report = json.loads((out / "latest.json").read_text())
        assert report["replays"]["n_matches"] == 0
        assert report["history"] is None
        assert len(report["errors"]) >= 2  # missing history + missing source dir
