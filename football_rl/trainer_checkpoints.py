"""Trainer persistence: checkpoints, generation snapshots, history migration backups, match summaries and resume."""

from __future__ import annotations

from collections import deque
from datetime import datetime
import json
import os
import shutil
from typing import Dict, List, Optional, Tuple

import torch

from football_rl.coevolution import COEVO_WINDOW_MATCHES
from football_rl.migrations import (
    TEAM_COLOR_SCHEMA,
    adapt_optimizer_state,
    adapt_policy_weights,
    migrate_team_color_keys,
)
from football_rl.opponent_pool import DEFAULT_POOL_SIZE
from football_rl.policy import ActorCritic, holds_return_statistics, policy_head_of_weights


SNAPSHOT_INTERVAL = 50
SNAPSHOT_POLICY_FILES = (
    "field_policy.pt",
    "keeper_policy.pt",
    "blue_field_policy.pt",
    "blue_keeper_policy.pt",
    "red_field_policy.pt",
    "red_keeper_policy.pt",
)


class CheckpointMixin:
    """Checkpoint saving and loading of `ParameterSharingPPOTrainer`."""

    def save_checkpoints(self) -> Tuple[str, str]:
        """Save Red and Blue policies, optimizers, generation counter, and match summaries to disk."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        field_path = os.path.join(self.checkpoint_dir, "field_policy.pt")
        keeper_path = os.path.join(self.checkpoint_dir, "keeper_policy.pt")
        blue_field_path = os.path.join(self.checkpoint_dir, "blue_field_policy.pt")
        blue_keeper_path = os.path.join(self.checkpoint_dir, "blue_keeper_policy.pt")
        red_field_path = os.path.join(self.checkpoint_dir, "red_field_policy.pt")
        red_keeper_path = os.path.join(self.checkpoint_dir, "red_keeper_policy.pt")
        state_path = os.path.join(self.checkpoint_dir, "training_state.pt")
        history_path = os.path.join(self.checkpoint_dir, "training_history.json")

        torch.save(self.blue_field_policy.state_dict(), field_path)
        torch.save(self.blue_keeper_policy.state_dict(), keeper_path)
        torch.save(self.blue_field_policy.state_dict(), blue_field_path)
        torch.save(self.blue_keeper_policy.state_dict(), blue_keeper_path)
        torch.save(self.red_field_policy.state_dict(), red_field_path)
        torch.save(self.red_keeper_policy.state_dict(), red_keeper_path)

        torch.save(
            {
                "current_generation": self.current_generation,
                "total_env_steps": self.total_env_steps,
                "seed": int(self.seed),
                "config_overrides": dict(self.config_overrides),
                "git_commit": self.git_commit,
                "next_match_id": self._get_max_saved_match_id() + 1,
                "separate_teams": self.separate_teams,
                "shared_keepers": self.shared_keepers,
                "policy_head": self.policy_head,
                "pool_envs": self.pool_envs,
                "pool_size": self.pool_size,
                "replay_every": self.replay_every,
                "replay_kind": self.replay_kind,
                "field_policy": self.blue_field_policy.state_dict(),
                "keeper_policy": self.blue_keeper_policy.state_dict(),
                "field_optimizer": self.blue_field_optimizer.state_dict(),
                "keeper_optimizer": self.blue_keeper_optimizer.state_dict(),
                "blue_field_policy": self.blue_field_policy.state_dict(),
                "blue_keeper_policy": self.blue_keeper_policy.state_dict(),
                "blue_field_optimizer": self.blue_field_optimizer.state_dict(),
                "blue_keeper_optimizer": self.blue_keeper_optimizer.state_dict(),
                "red_field_policy": self.red_field_policy.state_dict(),
                "red_keeper_policy": self.red_keeper_policy.state_dict(),
                "red_field_optimizer": self.red_field_optimizer.state_dict(),
                "red_keeper_optimizer": self.red_keeper_optimizer.state_dict(),
                "match_history": self.match_history,
                "generation_history": self.generation_history,
                "recent_match_scores": [list(score) for score in self.recent_match_scores],
                "team_color_schema": TEAM_COLOR_SCHEMA,
            },
            state_path,
        )

        history_payload = {
            "current_generation": self.current_generation,
            "total_matches": len(self.match_history),
            "total_env_steps": self.total_env_steps,
            "separate_teams": self.separate_teams,
            "shared_keepers": self.shared_keepers,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "team_color_schema": TEAM_COLOR_SCHEMA,
            "matches": self.match_history,
            "generations": self.generation_history,
        }
        self._backup_pre_migration_history(history_path)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history_payload, f, indent=2)

        if self.cfg.replay.enable_recording:
            self._write_matches_summary_markdown()

        self._write_generation_snapshot()

        return field_path, keeper_path

    def _write_generation_snapshot(self) -> Optional[str]:
        """Copy the policy files and training state to `snapshots/genNNNN/` on every 50th completed generation.

        A snapshot that already exists is left untouched, so resuming or saving again at the same
        generation never rewrites it. The directory appears atomically (copy to a temp name, then rename).
        """
        gen = self.current_generation
        if gen <= 0 or gen % SNAPSHOT_INTERVAL != 0:
            return None
        root = os.path.join(self.checkpoint_dir, "snapshots")
        final = os.path.join(root, f"gen{gen:04d}")
        if os.path.isdir(final):
            return final
        staging = final + ".tmp"
        if os.path.isdir(staging):
            shutil.rmtree(staging)
        os.makedirs(staging)
        for name in SNAPSHOT_POLICY_FILES + ("training_state.pt",):
            shutil.copyfile(os.path.join(self.checkpoint_dir, name), os.path.join(staging, name))
        os.replace(staging, final)
        return final

    def _backup_pre_migration_history(self, history_path: str) -> None:
        """Copy an on-disk history that predates the current team-colour schema, once per trainer."""
        if self._history_backup_checked:
            return
        self._history_backup_checked = True
        backup_path = os.path.join(self.checkpoint_dir, "training_history.pre_color_schema.json")
        if not os.path.exists(history_path) or os.path.exists(backup_path):
            return
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                on_disk_schema = json.load(f).get("team_color_schema")
        except ValueError:
            on_disk_schema = None
        if on_disk_schema != TEAM_COLOR_SCHEMA:
            shutil.copy2(history_path, backup_path)

    def _compute_series_summary(self) -> Dict[str, int]:
        """Compute cumulative Red (team A) vs. Blue (team B) wins, draws, and goals across `self.match_history`."""
        red_wins = 0
        blue_wins = 0
        draws = 0
        red_goals = 0
        blue_goals = 0
        for m in self.match_history:
            sr, sb = int(m["score"][0]), int(m["score"][1])
            red_goals += sr
            blue_goals += sb
            if sr > sb:
                red_wins += 1
            elif sb > sr:
                blue_wins += 1
            else:
                draws += 1
        return {
            "red_wins": red_wins,
            "blue_wins": blue_wins,
            "draws": draws,
            "red_goals": red_goals,
            "blue_goals": blue_goals,
        }

    def _write_root_summary_copy(self, summary_path: str, content: str) -> None:
        """Mirror the summary to `./matches_summary.md` when enabled; never writes through a symlink."""
        if not self.write_root_summary:
            return
        root_summary = "matches_summary.md"
        if os.path.islink(root_summary):
            return
        if os.path.abspath(summary_path) == os.path.abspath(root_summary):
            return
        try:
            with open(root_summary, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError:
            pass

    def _write_matches_summary_markdown(self) -> Optional[str]:
        """Write a human-friendly `replays/matches_summary.md` table for quick scanning."""
        os.makedirs(self.cfg.replay.save_dir, exist_ok=True)
        summary_path = os.path.join(self.cfg.replay.save_dir, "matches_summary.md")
        series = self._compute_series_summary()
        mode_str = "Independent Team Policies (Red vs. Blue Co-Evolution)" if self.separate_teams else "Shared Mirror Policies"

        lines: List[str] = [
            "# Match & Generation Summary",
            "",
            f"- **Training Mode:** `{mode_str}`",
            f"- **Generations Trained:** `{self.current_generation}`",
            f"- **Total Completed Matches:** `{len(self.match_history)}`",
            f"- **Head-to-Head Series (Red - Draws - Blue):** `{series['red_wins']} W - {series['draws']} D - {series['blue_wins']} W` (Goals: **Red `{series['red_goals']}` - `{series['blue_goals']}` Blue**)",
            f"- **Total RL Steps:** `{self.total_env_steps:,}`",
            f"- **Last Updated:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
            "",
            "### Quick Commands",
            "- **Watch any match by number (e.g. Match #15):** `.venv/bin/python replay.py 15`",
            "- **List matches or filter in terminal:** `.venv/bin/python replay.py --list` or `.venv/bin/python replay.py --range 1 100 --top 10`",
            "- **Resume training next 100 generations:** `.venv/bin/python train_ppo.py --total-updates 100`",
            "",
        ]

        if not self.match_history:
            lines.append("_No completed matches recorded yet._\n")
            content = "\n".join(lines)
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(content)
            self._write_root_summary_copy(summary_path, content)
            return summary_path

        # 1. Top 15 Most Exciting Matches Leaderboard
        top_matches = sorted(
            self.match_history, key=lambda m: float(m["interest_score"]), reverse=True
        )[:15]
        lines.extend(
            [
                "## Top 15 Most Exciting Matches (by Interest Score)",
                "",
                "| Rank | Match # | Gen | Score (Red - Blue) | Interest | Poss% (R/B) | Shots (R/B) | Passes (R/B) | Tackles (R/B) | Saves (R/B) | Watch Command |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for rank, m in enumerate(top_matches, start=1):
            mid = m["match_id"]
            gen = m["generation"]
            sc = f"**{m['score'][0]} - {m['score'][1]}**"
            inter = f"`{m['interest_score']:.1f}`"
            poss = f"{m['possession_pct'][0]:.0f}% / {m['possession_pct'][1]:.0f}%"
            shots = f"{m['shots'][0]}/{m['shots'][1]}"
            passes = f"{m['team_passes'][0]}/{m['team_passes'][1]}"
            tackles = f"{m['team_tackles'][0]}/{m['team_tackles'][1]}"
            saves = f"{m['team_saves'][0]}/{m['team_saves'][1]}"
            cmd = f"`.venv/bin/python replay.py {mid}`" if m.get("replay_path") else "_(no replay)_"
            lines.append(
                f"| #{rank} | **Match {mid}** | Gen {gen} | {sc} | {inter} | {poss} | {shots} | {passes} | {tackles} | {saves} | {cmd} |"
            )

        # 2. Complete Chronological Match Table (1 .. N)
        lines.extend(
            [
                "",
                "---",
                "",
                "## All Matches (Chronological: Match 1 to " + str(len(self.match_history)) + ")",
                "",
                "| Match # | Gen | Score (Red - Blue) | Interest | Poss% (R/B) | Shots (R/B) | Passes (R/B) | Tackles (R/B) | Saves (R/B) | Fouls (Field 5s/GK 5s) | Replay File |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for m in self.match_history:
            mid = m["match_id"]
            gen = m["generation"]
            sc = f"**{m['score'][0]} - {m['score'][1]}**"
            inter = f"`{m['interest_score']:.1f}`"
            poss = f"{m['possession_pct'][0]:.0f}%/{m['possession_pct'][1]:.0f}%"
            shots = f"{m['shots'][0]}/{m['shots'][1]}"
            passes = f"{m['team_passes'][0]}/{m['team_passes'][1]}"
            tackles = f"{m['team_tackles'][0]}/{m['team_tackles'][1]}"
            saves = f"{m['team_saves'][0]}/{m['team_saves'][1]}"
            f_viol = m.get("field_5s_violations", 0)
            k_viol = m.get("keeper_5s_violations", 0)
            fouls = f"{f_viol}/{k_viol}"
            rfile = os.path.basename(m["replay_path"]) if m["replay_path"] else "_(not saved)_"
            lines.append(
                f"| **Match {mid}** | Gen {gen} | {sc} | {inter} | {poss} | {shots} | {passes} | {tackles} | {saves} | {fouls} | `{rfile}` |"
            )

        lines.append("")
        content = "\n".join(lines)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(content)

        self._write_root_summary_copy(summary_path, content)

        return summary_path

    def _load_adapted_weights(self, policy: ActorCritic, weights: Dict[str, torch.Tensor]) -> None:
        """Load a state dict into `policy`, zero-padding narrower first-layer inputs (`adapt_policy_weights`).

        Raises ValueError when the checkpoint's policy head differs from the network's; heads are not convertible, and
        when it holds return statistics but the network has none. A checkpoint without statistics leaves a value-normalising
        network on its identity statistics.
        """
        saved_head = policy_head_of_weights(weights)
        if saved_head is not None and saved_head != policy.policy_head:
            raise ValueError(
                f"Checkpoint holds {saved_head!r} policies but the configured policy head is {policy.policy_head!r}; "
                "heads cannot be converted."
            )
        if holds_return_statistics(weights) and not policy.value_norm:
            raise ValueError(
                "Checkpoint holds critic return statistics but ppo.value_norm is off; its critic is in "
                "normalised units and cannot be read as raw returns."
            )
        policy.load_state_dict(
            adapt_policy_weights(weights, policy.obs_dim, policy.central_state_dim), strict=False
        )

    def load_checkpoints(self) -> bool:
        """Load Red and Blue policies, optimizers, and cumulative generation/match state."""
        state_path = os.path.join(self.checkpoint_dir, "training_state.pt")
        field_path = os.path.join(self.checkpoint_dir, "field_policy.pt")
        keeper_path = os.path.join(self.checkpoint_dir, "keeper_policy.pt")

        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device, weights_only=False)
            stored_head = state.get("policy_head", "gaussian4")
            if stored_head != self.policy_head:
                raise ValueError(
                    f"Checkpoint was trained with policy head {stored_head!r} but the configured policy head is "
                    f"{self.policy_head!r}; heads cannot be converted."
                )
            stored_overrides = dict(sorted((state.get("config_overrides") or {}).items()))
            if stored_overrides != self.config_overrides:
                if not self.accept_config_change:
                    raise ValueError(
                        f"Config overrides differ from the checkpoint: stored {stored_overrides!r}, "
                        f"requested {self.config_overrides!r}. Pass --accept-config-change to resume anyway."
                    )
                print(
                    f"[Resume] Config overrides changed: stored {stored_overrides!r}, "
                    f"requested {self.config_overrides!r}; continuing with the requested overrides."
                )
            stored_pool = (int(state.get("pool_envs", 0)), int(state.get("pool_size", DEFAULT_POOL_SIZE)))
            requested_pool = (self.pool_envs, self.pool_size)
            if (stored_pool[0] or requested_pool[0]) and stored_pool != requested_pool:
                described = (
                    f"stored pool_envs={stored_pool[0]}, pool_size={stored_pool[1]}, "
                    f"requested pool_envs={requested_pool[0]}, pool_size={requested_pool[1]}"
                )
                if not self.accept_config_change:
                    raise ValueError(
                        f"Pool settings differ from the checkpoint: {described}. "
                        "Pass --accept-config-change to resume anyway."
                    )
                print(f"[Resume] Pool settings changed: {described}; continuing with the requested settings.")
            stored_shared_keepers = bool(state.get("shared_keepers", False))
            if stored_shared_keepers != self.shared_keepers:
                described = f"stored shared_keepers={stored_shared_keepers}, requested shared_keepers={self.shared_keepers}"
                if not self.accept_config_change:
                    raise ValueError(
                        f"Keeper sharing differs from the checkpoint: {described}. "
                        "Pass --accept-config-change to resume anyway."
                    )
                print(f"[Resume] Keeper sharing changed: {described}; continuing with the requested setting.")
            stored_replay_every = int(state.get("replay_every", 1))
            stored_replay_kind = str(state.get("replay_kind", "both"))
            if (stored_replay_every, stored_replay_kind) != (self.replay_every, self.replay_kind):
                print(
                    f"[Resume] Replay cap changed: "
                    f"stored replay_every={stored_replay_every}, replay_kind={stored_replay_kind!r}, "
                    f"requested replay_every={self.replay_every}, replay_kind={self.replay_kind!r}; "
                    "continuing with the requested settings."
                )
            saved_seed = state.get("seed")
            if saved_seed is not None and int(saved_seed) != int(self.seed):
                print(f"[Resume] Checkpoint seed was {int(saved_seed)}; continuing with seed={int(self.seed)}.")
            self.current_generation = int(state.get("current_generation", 0))
            self.total_env_steps = int(state.get("total_env_steps", 0))
            self.match_history = list(state.get("match_history", []))
            self.generation_history = migrate_team_color_keys(
                {
                    "generations": list(state.get("generation_history", [])),
                    "team_color_schema": state.get("team_color_schema"),
                }
            )["generations"]
            self.recent_match_scores = deque(
                ((int(a), int(b)) for a, b in state.get("recent_match_scores", [])),
                maxlen=COEVO_WINDOW_MATCHES,
            )
            self.next_match_id = self._get_max_saved_match_id() + 1

            blue_f_weights = state.get("blue_field_policy", state.get("field_policy"))
            blue_k_weights = state.get("blue_keeper_policy", state.get("keeper_policy"))
            red_f_weights = state.get("red_field_policy", blue_f_weights)
            red_k_weights = state.get("red_keeper_policy", blue_k_weights)

            if blue_f_weights is not None:
                self._load_adapted_weights(self.blue_field_policy, blue_f_weights)
            if blue_k_weights is not None:
                self._load_adapted_weights(self.blue_keeper_policy, blue_k_weights)
            if self.separate_teams:
                if red_f_weights is not None:
                    self._load_adapted_weights(self.red_field_policy, red_f_weights)
                if red_k_weights is not None and not self.shared_keepers:
                    self._load_adapted_weights(self.red_keeper_policy, red_k_weights)

            if "blue_field_optimizer" in state:
                try:
                    self.blue_field_optimizer.load_state_dict(
                        adapt_optimizer_state(state["blue_field_optimizer"], self.blue_field_policy, blue_f_weights)
                    )
                except Exception:
                    pass
            elif "field_optimizer" in state:
                try:
                    self.blue_field_optimizer.load_state_dict(
                        adapt_optimizer_state(state["field_optimizer"], self.blue_field_policy, blue_f_weights)
                    )
                except Exception:
                    pass

            if "blue_keeper_optimizer" in state:
                try:
                    self.blue_keeper_optimizer.load_state_dict(
                        adapt_optimizer_state(state["blue_keeper_optimizer"], self.blue_keeper_policy, blue_k_weights)
                    )
                except Exception:
                    pass
            elif "keeper_optimizer" in state:
                try:
                    self.blue_keeper_optimizer.load_state_dict(
                        adapt_optimizer_state(state["keeper_optimizer"], self.blue_keeper_policy, blue_k_weights)
                    )
                except Exception:
                    pass

            if self.separate_teams:
                if "red_field_optimizer" in state:
                    try:
                        self.red_field_optimizer.load_state_dict(
                            adapt_optimizer_state(state["red_field_optimizer"], self.red_field_policy, red_f_weights)
                        )
                    except Exception:
                        pass
                if "red_keeper_optimizer" in state and not self.shared_keepers:
                    try:
                        self.red_keeper_optimizer.load_state_dict(
                            adapt_optimizer_state(state["red_keeper_optimizer"], self.red_keeper_policy, red_k_weights)
                        )
                    except Exception:
                        pass

            # Re-number active environments so the next match starts cleanly at len(match_history) + 1
            self._reset_all_envs(initial=True)
            self._sync_cpu_policies()
            return True

        loaded_any = False
        if os.path.exists(field_path):
            w = torch.load(field_path, map_location=self.device, weights_only=True)
            self._load_adapted_weights(self.blue_field_policy, w)
            if self.separate_teams:
                self._load_adapted_weights(self.red_field_policy, w)
            loaded_any = True
        if os.path.exists(keeper_path):
            w = torch.load(keeper_path, map_location=self.device, weights_only=True)
            self._load_adapted_weights(self.blue_keeper_policy, w)
            if self.separate_teams:
                self._load_adapted_weights(self.red_keeper_policy, w)
            loaded_any = True
        self._sync_cpu_policies()
        return loaded_any
