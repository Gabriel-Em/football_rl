"""Batch archives: archiving the active run with its selected replays, loading, listing and status reports."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import json
import os
import random
import re
import shutil
from typing import Any, Dict, List, Optional

from football_rl.migrations import migrate_team_color_keys


# ==============================================================================
# Batch Archiving, Loading, Resetting & Status Helpers
# ==============================================================================
def reset_active_batch(checkpoint_dir: str, replay_dir: str) -> None:
    """Wipe active `checkpoints/` and `replays/` to start a brand-new batch from scratch."""
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    if os.path.exists(replay_dir):
        shutil.rmtree(replay_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(replay_dir, exist_ok=True)


def select_archive_replays(
    replay_files: List[str],
    replay_dir: Optional[str] = None,
    num_first: int = 5,
    num_mid: int = 5,
    num_last: int = 5,
    seed: Optional[int] = None,
) -> List[str]:
    """Select up to (num_first + num_mid + num_last) match replay files across training generations.

    Retains the first `num_first` generations, the last `num_last` generations, and `num_mid`
    generations randomly spread out in-between, keeping at most one match replay per generation.
    If the total number of match replay files is <= (num_first + num_mid + num_last), all files are kept.
    """
    total_target = num_first + num_mid + num_last
    if len(replay_files) <= total_target:
        return list(replay_files)

    items = []
    for f in replay_files:
        full_path = os.path.join(replay_dir, f) if replay_dir else f
        base = os.path.basename(f)
        m = re.search(r"match_(\d+)_gen(\d+)(?:_score(\d+(?:_\d+)?))?", base)
        if m:
            mid = int(m.group(1))
            gen = int(m.group(2))
            try:
                score = float(m.group(3).replace("_", ".")) if m.group(3) else 0.0
            except (ValueError, TypeError):
                score = 0.0
        else:
            m_gen = re.search(r"gen(\d+)", base)
            m_mid = re.search(r"match_(\d+)", base)
            m_score = re.search(r"score(\d+(?:_\d+)?)", base)
            gen = int(m_gen.group(1)) if m_gen else None
            mid = int(m_mid.group(1)) if m_mid else 0
            try:
                score = float(m_score.group(1).replace("_", ".")) if m_score else 0.0
            except (ValueError, TypeError):
                score = 0.0

        if gen is None and os.path.isfile(full_path):
            try:
                with open(full_path, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                    meta = data.get("metadata", {})
                    gen = meta.get("generation", data.get("generation"))
                    if mid == 0:
                        mid = meta.get("match_id", data.get("match_id", 0))
                    if score == 0.0:
                        score = float(meta.get("interest_score", data.get("interest_score", 0.0)))
            except Exception:
                pass

        if gen is None:
            gen = mid if mid > 0 else 0

        items.append({"file": f, "gen": gen, "mid": mid, "score": score, "is_uniform": "_uniform_" in base})

    gen_to_items = defaultdict(list)
    for item in items:
        gen_to_items[item["gen"]].append(item)

    for gen in gen_to_items:
        gen_to_items[gen].sort(key=lambda x: (x["is_uniform"], x["score"], x["mid"]), reverse=True)

    sorted_gens = sorted(gen_to_items.keys())
    N = len(sorted_gens)

    if N <= total_target:
        chosen_files = [gen_to_items[g][0]["file"] for g in sorted_gens]
        if len(chosen_files) < total_target:
            remaining_items = []
            for g in sorted_gens:
                remaining_items.extend(gen_to_items[g][1:])
            remaining_items.sort(key=lambda x: (x["score"], x["mid"]), reverse=True)
            for item in remaining_items[: total_target - len(chosen_files)]:
                chosen_files.append(item["file"])
        return chosen_files

    first_gens = sorted_gens[:num_first]
    last_gens = sorted_gens[-num_last:]
    in_between = sorted_gens[num_first:-num_last]
    M = len(in_between)

    rng = random.Random(seed) if seed is not None else random
    mid_gens = []
    if M <= num_mid:
        mid_gens = list(in_between)
    else:
        for i in range(num_mid):
            start = int(i * M / num_mid)
            end = int((i + 1) * M / num_mid)
            segment = in_between[start:end]
            mid_gens.append(rng.choice(segment))
    mid_gens.sort()

    chosen_gens = first_gens + mid_gens + last_gens
    return [gen_to_items[g][0]["file"] for g in chosen_gens]


def archive_batch(
    name: Optional[str] = None,
    checkpoint_dir: str = "checkpoints",
    replay_dir: str = "replays",
    archive_root: str = "archives",
    num_first: int = 5,
    num_mid: int = 5,
    num_last: int = 5,
) -> str:
    """Save active `checkpoints/` and `replays/` into `archives/<name>/`.

    When archiving, keeps the first 5 and last 5 generations as well as 5 randomly spread
    out in-between generations, retaining a total of 15 match replays only.
    If `name` is omitted or empty, automatically generates a timestamp-based name
    formatted as `%Y%m%d_%H%M%S`.
    """
    if not name or not name.strip():
        name = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        name = name.strip()

    state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if not os.path.exists(state_path) and not os.path.exists(
        os.path.join(checkpoint_dir, "field_policy.pt")
    ):
        raise FileNotFoundError(
            f"No active training checkpoints found in `{checkpoint_dir}/` to archive."
        )

    dest_dir = os.path.join(archive_root, name)
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    if os.path.exists(checkpoint_dir):
        shutil.copytree(checkpoint_dir, os.path.join(dest_dir, "checkpoints"))

    if os.path.exists(replay_dir):
        dest_rep = os.path.join(dest_dir, "replays")
        os.makedirs(dest_rep, exist_ok=True)

        all_entries = os.listdir(replay_dir)
        replay_files = []
        for entry in all_entries:
            src_path = os.path.join(replay_dir, entry)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, os.path.join(dest_rep, entry), dirs_exist_ok=True)
            elif entry.endswith(".json") and entry != "training_history.json":
                replay_files.append(entry)
            else:
                shutil.copy2(src_path, os.path.join(dest_rep, entry))

        kept_replays = select_archive_replays(
            replay_files,
            replay_dir=replay_dir,
            num_first=num_first,
            num_mid=num_mid,
            num_last=num_last,
        )
        for rf in kept_replays:
            src_file = os.path.join(replay_dir, rf)
            if os.path.exists(src_file):
                shutil.copy2(src_file, os.path.join(dest_rep, rf))

    return dest_dir


def load_archived_batch(
    name: str,
    checkpoint_dir: str = "checkpoints",
    replay_dir: str = "replays",
    archive_root: str = "archives",
) -> str:
    """Restore an archived batch from `archives/<name>/` into active `checkpoints/` and `replays/`."""
    src_dir = os.path.join(archive_root, name)
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Archive `{src_dir}` does not exist.")

    reset_active_batch(checkpoint_dir, replay_dir)
    src_ckpt = os.path.join(src_dir, "checkpoints")
    src_rep = os.path.join(src_dir, "replays")

    if os.path.exists(src_ckpt):
        shutil.copytree(src_ckpt, checkpoint_dir, dirs_exist_ok=True)
    if os.path.exists(src_rep):
        shutil.copytree(src_rep, replay_dir, dirs_exist_ok=True)

    return src_dir


def list_archived_batches(archive_root: str = "archives") -> List[Dict[str, Any]]:
    """Return metadata about all saved training batches in `archives/`."""
    if not os.path.exists(archive_root):
        return []
    entries: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(archive_root)):
        bdir = os.path.join(archive_root, name)
        if not os.path.isdir(bdir):
            continue
        hist_path = os.path.join(bdir, "checkpoints", "training_history.json")
        if os.path.exists(hist_path):
            with open(hist_path, "r", encoding="utf-8") as f:
                h = migrate_team_color_keys(json.load(f))
            matches = h.get("matches", [])
            best_score = max((m.get("interest_score", 0.0) for m in matches), default=0.0)
            entries.append(
                {
                    "name": name,
                    "generations": h.get("current_generation", 0),
                    "matches": h.get("total_matches", len(matches)),
                    "total_env_steps": h.get("total_env_steps", 0),
                    "best_interest": best_score,
                    "last_updated": h.get("last_updated", ""),
                }
            )
        else:
            entries.append(
                {
                    "name": name,
                    "generations": 0,
                    "matches": 0,
                    "total_env_steps": 0,
                    "best_interest": 0.0,
                    "last_updated": "unknown",
                }
            )
    return entries


def print_active_status(checkpoint_dir: str = "checkpoints", replay_dir: str = "replays") -> None:
    """Print the current active batch's generation count, total matches, and top highlights."""
    hist_path = os.path.join(checkpoint_dir, "training_history.json")
    if not os.path.exists(hist_path):
        print("No active training history found in `checkpoints/training_history.json`.")
        print("Run `.venv/bin/python train_ppo.py --total-updates 100` to start training!")
        return

    with open(hist_path, "r", encoding="utf-8") as f:
        h = migrate_team_color_keys(json.load(f))

    matches = h.get("matches", [])
    red_w = sum(1 for m in matches if m["score"][0] > m["score"][1])
    blue_w = sum(1 for m in matches if m["score"][1] > m["score"][0])
    draws = len(matches) - red_w - blue_w
    red_g = sum(int(m["score"][0]) for m in matches)
    blue_g = sum(int(m["score"][1]) for m in matches)

    print("=" * 80)
    print(f"ACTIVE TRAINING BATCH STATUS")
    print("=" * 80)
    print(f"  Generations Trained : {h.get('current_generation', 0)}")
    print(f"  Completed Matches   : {len(matches)}")
    print(
        f"  Series (Red - Blue) : {red_w} Wins - {draws} Draws - {blue_w} Wins "
        f"(Goals: Red {red_g} - {blue_g} Blue)"
    )
    print(f"  Total RL Steps      : {h.get('total_env_steps', 0):,}")
    print(f"  Last Updated        : {h.get('last_updated', 'N/A')}")
    print(f"  Summary File        : {os.path.join(replay_dir, 'matches_summary.md')}")
    print("-" * 80)

    if matches:
        top = sorted(matches, key=lambda m: float(m.get("interest_score", 0.0)), reverse=True)[:10]
        print("  TOP 10 MOST EXCITING MATCHES:")
        print(
            f"  {'Match #':<9} {'Gen':<7} {'Score (R-B)':<13} {'Interest':<10} "
            f"{'Shots':<9} {'Passes':<9} {'Tackles':<9} {'Saves':<7}"
        )
        for m in top:
            sc = f"{m['score'][0]} - {m['score'][1]}"
            sh = f"{m['shots'][0]}/{m['shots'][1]}"
            pa = f"{m['team_passes'][0]}/{m['team_passes'][1]}"
            ta = f"{m['team_tackles'][0]}/{m['team_tackles'][1]}"
            sa = f"{m['team_saves'][0]}/{m['team_saves'][1]}"
            print(
                f"  #{m['match_id']:<8d} {m['generation']:<7d} {sc:<13} "
                f"{m['interest_score']:<10.1f} {sh:<9} {pa:<9} {ta:<9} {sa:<7}"
            )
    print("=" * 80)


def resume_num_envs_warning(checkpoint_dir: str, num_envs: int) -> Optional[str]:
    """Return a warning when the last recorded generation ran with a different `num_envs`.

    Returns None when the history is absent/unreadable, empty, or its last generation record
    carries no `num_envs` (legacy records), or when the values agree.
    """
    history_path = os.path.join(checkpoint_dir, "training_history.json")
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            generations = json.load(f).get("generations") or []
    except (OSError, ValueError, AttributeError):
        return None
    if not generations or not isinstance(generations[-1], dict):
        return None
    recorded = generations[-1].get("num_envs")
    if recorded is None or int(recorded) == int(num_envs):
        return None
    return (
        f"WARNING: resuming with num_envs={num_envs} but generation "
        f"{generations[-1].get('generation', '?')} was trained with num_envs={int(recorded)}. "
        f"Samples per generation and the co-evolution window pacing will differ from the earlier run."
    )
