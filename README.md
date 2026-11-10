# 6v6 Multi-Agent Reinforcement Learning (MARL) 2D Football

A continuous-space, discrete-time Dec-POMDP multi-agent reinforcement learning sports environment built with `PettingZoo` (`ParallelEnv`), `Gymnasium`, `NumPy`, `PyTorch`, and `Pygame`.

Simulates 6v6 competitive football matches featuring 12 autonomous agents across 4 distinct tactical roles per team, continuous 2D sub-stepped physics, dynamic coordinate canonicalization, and parameter-shared Proximal Policy Optimization (PPO) self-play.

---

## Key Architectural & Engine Features

- **60 Hz Physics & 15 Hz RL Decision Cadence (`frame_skip = 4`)**:
  Matches run for `180.0s` (2,700 RL decision steps / 10,800 physics frames) with a physical side-switch and alternating kickoff at Half-Time (`t = 90.0s`).
- **Pitch Dimensions ($1400.0 \times 840.0\text{ u}$)**:
  Expanded 6v6 pitch featuring $200.0\text{u}$ goal mouths centered at $Y = 420.0$ ($Y \in [320.0, 520.0]$), and $195.0\text{u}$ nominal radius semicircle goalkeeper boxes.
- **Dynamic Semicircle Shrinking & Expansion (Anti-Stall System)**:
  When an unclaimed loose ball rests inside the goalkeeper box and comes to a complete stop ($\le 15.0\text{ u/s}$), after a $2.0\text{s}$ grace period the semicircle boundary smoothly contracts at $55.0\text{ u/s}$ down toward $0.0\text{u}$. This enables outfield attackers to contest deadlocked loose balls cleanly without artificial foul teleports while preventing any premature shrinking when the ball is merely passing through in motion. Upon possession recovery, movement, or clearance, the box smoothly re-expands at $220.0\text{ u/s}$ back to nominal $195.0\text{u}$ radius, gently nudging outfielders outside.
- **76-Dimensional Team-Canonical Observation Space ($\mathbf{o} \in \mathbb{R}^{76}$)**:
  Dynamically transforms coordinates and velocities by $180^\circ$ for whichever team currently defends the right goal, ensuring an invariant forward-attacking perspective ($+X$ toward opponent goal) across both halves:
  - **16 Self Features (`indices 0..15`)**: Canonical position, facing angle vector $(\sin, \cos)$, normalized velocity, score differential, ball possession flag, hold/box timers, tackle/lunge cooldown, ghost/stun timer, relative goal angle and distance, and 3-bit one-hot role vector `[is_def, is_mid, is_atk]`.
  - **60 Entity Features (`indices 16..75`)**: 12 entity slots $\times$ 5 features (Slot 0: Ball with $360^\circ$ awareness; Slots 1..5: 5 Teammates; Slots 6..10: 5 Field Opponents; Slot 11: Enemy Goalkeeper). Peripheral vision fades smoothly from $170^\circ$ to $210^\circ$ FOV.
- **4-Dimensional Continuous Action Space ($\mathbf{a} \in [-1, 1]^4$)**:
  - `action[0]`: Throttle force (with $[-0.35, +0.15]$ deadzone and backpedal multipliers).
  - `action[1]`: Angular steering rate ($270^\circ$/s max body turn rate). Also acts as real-time **kick-steer deflection** ($\pm 35^\circ$) on kick execution.
  - `action[2]`: Kick trigger & power threshold ($> 0.35$ triggers kick with power scaled to $[260, 640]\text{ u/s}$).
  - `action[3]`: Special tactical ability / dynamic slide tackle / goalkeeper holonomic dive.
- **Kick-Steer Deflection ($\pm 35^\circ$) & Passing Mechanics**:
  - Outfielders and keepers can deflect their kick release by up to $\pm 35^\circ$ using `action[1]`, executing instant side-foot passes and angled clearances without waiting for full body reorientation.
  - Frame-0 Dangerous Own-Goal Threat Penalty ($-3.0$) penalizes kicks directed back into one's own net from the defensive third ($X_c \le 350.0\text{u}$).
  - Goalkeeper wall-ricochet / bank clearances reward high-power releases reflecting off touchlines past pressed defensive corridors.
  - Blocked-kick detection checks a forward corridor ($185.0\text{u} \times 30.0\text{u}$) and penalizes kicks fired straight into close defenders ($-0.85$).
- **Through-Balls & Tactical Movement**:
  - Progressive lead passes completed to teammates running in forward stride ($v_{x,c} \ge 90.0\text{ u/s}$) award a shared through-ball stride bonus ($+0.80$, `through_ball_stride_bonus`) and trigger `"THROUGH BALL!"` visual events.
  - Receiving agents compute in-flight lead intercept trajectories $\mathbf{p}_{\text{int}} \in [\mathbf{p}_b, \mathbf{p}_b + 0.8\mathbf{v}_b]$ to meet passes in stride.
  - Half-turn body readiness rewards off-ball support players maintaining forward-facing field orientation ($\cos(\theta_c) \ge -0.10$) while reading the ball.
  - Continuous lateral unmasking reward ($+0.015$) encourages shadowed teammates to slide laterally into clear passing lanes.
- **Heterogeneous Role Profiles & Kinematics**:
  - **Goalkeeper (`ROLE_KEEPER` / 1 per team)**: $180.0\text{ u/s}$ top speed ($0.75\times$ backpedal), $340.0\text{ u/s}$ holonomic dives/lunges in any direction (`0.25s` duration, `1.8s` cooldown), $640.0\text{ u/s}$ boosted clearance speed, $5.0\text{s}$ maximum hold time, and strip-stun recovery.
  - **Center Backs (`ROLE_DEF` / 2 per team)**: $182.0\text{ u/s}$ speed, $510.0\text{ u/s}$ kick launch velocity, widest dynamic tackle reach ($48.0\text{ u}$ nominal radius, 86% success), tactical spreading width ($420 \pm 175\text{ u}$) to build out from deep.
  - **Box-to-Box Midfielders (`ROLE_MID` / 2 per team)**: $195.0\text{ u/s}$ speed (fastest field runners), $560.0\text{ u/s}$ kick velocity, balanced dynamic tackles ($44.0\text{ u}$ radius, 78% success), flank channel spreading, half-turn posture, and link-up play.
  - **Attacking Striker (`ROLE_ATK` / 1 per team)**: $190.0\text{ u/s}$ speed, highest launch kick velocity ($640.0\text{ u/s}$) with **zero kick jitter (pinpoint laser)**, dynamic tackle ($40.0\text{ u}$ radius, 70% success), and prime finishing bonuses inside $320.0\text{ u}$ of goal.
- **Physical Contests, Shielding & Dynamic Tackling**:
  - Dynamic tackle reach decays quadratically with relative closing speed, rewarding controlled defensive positioning over reckless sprint-ram tackles.
  - Contested loose-ball recoveries apply a $0.35$ speed damping and $15.0\text{ u}$ radial separation to defeated opponents within $45.0\text{ u}$, granting physical shielding.
  - Free ball recoveries enjoy a $0.45\text{s}$ post-pickup tackle immunity grace period.
- **PPO Co-Evolution & Vectorized Self-Play (`train_ppo.py`)**:
  - Independent Blue Team (`blue_field_policy` + `blue_keeper_policy`) vs. Red Team (`red_field_policy` + `red_keeper_policy`) actor-critic networks.
  - Rolling co-evolution soft-sync over the last 24 completed matches (independent of `--num-envs`) to prevent runaway dominance and preserve competitive self-play parity.
  - Vectorized parallel environment multiprocessing (`--num-envs 8+`) with monotonic match persistence and `--save-best-only` replay filtering.
- **Interactive 60 FPS Replay Visualizer (`replay.py`)**:
  - Pygame viewer rendering active dynamic semicircle boundaries, player vision cones ($210^\circ$ FOV), role rings, tackle radii, ghost recovery stuns, and scoreboard HUD.
  - Color-coded floating event notifications: `"GOAL!"`, `"SAVE!"`, `"TACKLE!"`, `"THROUGH BALL!"`, `"WALL CLEARANCE!"`, and `"BALL WON!"`.
  - Interactive timeline scrubbing, goal/halftime tick marks, speed controls (`1x`, `2x`, `4x`), and match filtering.

---

## Project Structure

```
football_rl/
├── football_rl/
│   ├── config.py             # Dataclass configurations: physics, mechanics, vision, rewards, PPO
│   ├── env.py                # PettingZoo ParallelEnv: 60Hz physics, Dec-POMDP obs, rules & rewards
│   ├── potentials.py         # Pure per-player potential field (roles, crowding, cover assignment) behind PBRS
│   ├── recording.py          # Preallocated 60 Hz replay recorder; builds JSON frames on demand
│   ├── replay.py             # 60 FPS Pygame replay visualizer & match analyzer
│   ├── train_ppo.py          # Vectorized PPO training loop, parameter sharing, co-evolution
│   └── tests/
│       └── test_env.py       # Automated test suite (70+ unit tests covering physics, symmetry, mechanics)
├── context/                  # Architectural specs, behavioral audits, and tactical masterplans
├── replays/                  # Recorded 60 FPS match replay JSONs and matches_summary.md
├── checkpoints/              # Active policy weights and training_state.pt
├── archives/                 # Saved experiment batches and archived model snapshots
├── pyproject.toml            # Project packaging & dependencies
├── replay.py                 # Root CLI shortcut for replay visualizer
└── train_ppo.py              # Root CLI shortcut for training loop
```

---

## 1. Installation & Environment Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.12, Linux / WSL2 Ubuntu)
- NVIDIA GPU with CUDA 12+ (optional, CPU execution fully supported)

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/Gabriel-Em/football_rl.git
cd football_rl

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package and dependencies in editable mode
pip install --upgrade pip
pip install -e .

# Optional: Install CUDA-accelerated PyTorch (if running on NVIDIA GPU / WSL2)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## 2. Training Workflows (`train_ppo.py`)

### Recommended Multi-Environment Training Run
To train with 8 parallel environments while recording the highest-scoring match of each generation:
```bash
.venv/bin/python train_ppo.py --num-envs 8 --save-best-only --total-updates 100
```

### Resume Training Active Batch
Loads model weights, optimizer states, and rolling statistics from `checkpoints/training_state.pt`:
```bash
.venv/bin/python train_ppo.py --resume --num-envs 8 --save-best-only --total-updates 100
```
*(Tip: Whenever `checkpoints/training_state.pt` exists, running without `--from-scratch` automatically resumes).*

### Start a Brand-New Batch from Scratch
Cleans unarchived active checkpoints and replays, reinitializes network weights, and starts at Generation 1:
```bash
.venv/bin/python train_ppo.py --from-scratch --num-envs 8 --save-best-only --total-updates 100
```

### Check Current Training Progress & Statistics
Inspect generation count, cumulative matches, total steps, and Top 10 most exciting matches in the terminal:
```bash
.venv/bin/python train_ppo.py --status
```

### Parallel & Device Execution Guidelines
- **Multi-Env Vectorization (`--num-envs 8`)**: Significantly increases simulation throughput (~650–700 SPS) by running parallel environment steps across available CPU cores.
- **`--save-best-only`**: Highly recommended when `--num-envs > 1` to save disk space and retain only the single most interesting replay from each generation.
- **Device Support (`--device cpu` / `--device cuda`)**: Automatically selects CUDA if available; mini-batch PPO updates leverage GPU acceleration during policy gradient steps.

### Training CLI Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--total-updates` | `100` | Number of PPO generations to train in this session. |
| `--num-envs` | `8` | Number of parallel environment instances (one generation = `num-envs` full matches). Resuming with a different value than the last recorded generation prints a `WARNING`. |
| `--rollout-steps` | `2700` | RL steps per env per generation (`2700` = 1 full 180s match). |
| `--interest-threshold`| `0.0` | Minimum `interest_score` to export a replay JSON (`0.0` = save all matches). |
| `--save-best-only` | `False` | When running with `num_envs > 1`, only saves the single most interesting match per generation. |
| `--device` | Auto | Execution device (`cpu` or `cuda`, auto-detects CUDA). |
| `--resume` | Auto | Resume from `checkpoints/training_state.pt`. |
| `--from-scratch` | `False` | Reset `checkpoints/` and `replays/` to start at Generation 1. |
| `--shared-teams` | `False` | Force Blue and Red to share the same policy network (mirror self-play). |
| `--seed` | `42` | Random seed. |
| `--status` | `False` | Display current training status, match count, and leaderboard, then exit. |
| `--archive [NAME]` | `None` | Save active checkpoints and replays to `archives/<NAME>/` (or timestamped name if omitted) and exit. |
| `--load-archive <NAME>` | `None` | Restore an experiment archive to active workspace and exit. |
| `--list-archives` | `False` | List all saved experiment archives in `archives/` and exit. |

---

## 3. Watching Replays & Match Analysis (`replay.py`)

### Human-Readable Match Summary
During training, <a href="./replays/matches_summary.md">replays/matches_summary.md</a> is continuously updated with:
1. **Batch Overview**: Total generations, completed matches, and cumulative RL steps.
2. **Top 15 Leaderboard**: Ranked by `interest_score` (goals, shots, saves, tackles, through balls, wall clears).
3. **Chronological Match Table**: Final scores, possession statistics, through-ball counts, and replay viewer commands.

### Scan & Filter Matches in Terminal
```bash
# List all recorded matches and summary metrics
.venv/bin/python replay.py --list

# Filter matches within a generation range (e.g. 100 to 200)
.venv/bin/python replay.py --range 100 200

# Display Top 10 most exciting matches so far
.venv/bin/python replay.py --top 10
```

### Launch Interactive 60 FPS Replay Viewer
```bash
# Watch a match by its sequential Match ID
.venv/bin/python replay.py --match 11802

# Watch the most recently saved replay
.venv/bin/python replay.py

# Watch a specific replay file
.venv/bin/python replay.py replays/match_11802_gen1477_score181_9_20261105_211148.json

# Watch an archived match replay
.venv/bin/python replay.py archives/run_baseline/replays/match_0050_gen0050_score5_2_20261103.json

# Generate and watch a heuristic bot demonstration match
.venv/bin/python replay.py --generate-demo
```

### Viewer Keyboard & Mouse Controls
| Control | Action |
| :--- | :--- |
| `Space` | Pause / Resume playback. |
| `Left / Right` | Scrub `-5.0s / +5.0s` (or single frame when paused or holding `Shift`). |
| `Mouse Click / Drag` | Scrub directly to any point on the bottom timeline (ticks indicate goals, fouls, and HT). |
| `Up / Down` | Increase / decrease playback speed (`0.25x` $\leftrightarrow$ `0.5x` $\leftrightarrow$ `0.75x` $\leftrightarrow$ `1x` $\leftrightarrow$ `2x` $\leftrightarrow$ `4x`). |
| `V` | Toggle player vision cone overlays ($210^\circ$ FOV with peripheral ramp). |
| `R` | Restart replay from kickoff ($t = 0.0\text{s}$). |
| `Q / Esc` | Exit viewer. |

---

## 4. Experiment Archiving & Lifecycle Management

Save training runs (`checkpoints/` + `replays/` + `matches_summary.md`) into self-contained named experiment archives, test parameter variations, and restore older checkpoints at any time:

```bash
# 1. Archive the current batch
.venv/bin/python train_ppo.py --archive run_gen1450_tactical_v1

# 2. List all saved experiment archives
.venv/bin/python train_ppo.py --list-archives

# 3. Modify parameters in football_rl/config.py and run a new experiment
.venv/bin/python train_ppo.py --from-scratch --num-envs 8 --save-best-only --total-updates 100
.venv/bin/python train_ppo.py --archive run_gen100_experimental

# 4. Restore an earlier archive back to the active workspace to continue training
.venv/bin/python train_ppo.py --load-archive run_gen1450_tactical_v1
.venv/bin/python train_ppo.py --resume --num-envs 8 --save-best-only --total-updates 100
```

---

## 5. Automated Verification & Testing

The project maintains a comprehensive automated test suite covering PettingZoo compliance, rotational symmetry, ball kinematics, dynamic semicircle shrinking, kick-steer deflection, through-ball detection, and PPO updates:

```bash
# Run full test suite
.venv/bin/pytest football_rl/tests/test_env.py -v
```

---

## License

This project is licensed under the MIT License - see the <a href="./LICENSE">LICENSE</a> file for details.
