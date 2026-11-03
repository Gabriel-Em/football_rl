# 6v6 Multi-Agent Reinforcement Learning (MARL) 2D Football

A continuous-space, discrete-time Dec-POMDP multi-agent reinforcement learning sports environment built with `PettingZoo` (`ParallelEnv`), `Gymnasium`, `NumPy`, `PyTorch`, and `Pygame`.

Simulates 6v6 competitive football matches featuring 12 autonomous agents across 4 distinct tactical roles per team, continuous physics, dynamic coordinate canonicalization, and parameter-shared Proximal Policy Optimization (PPO) self-play.

---

## Key Architectural Features

- **60 Hz Physics & 15 Hz RL Decision Cadence (`frame_skip = 4`)**:
  Matches run for `180.0s` (2,700 RL decision steps / 10,800 physics frames) with a physical side-switch and alternating kickoff at Half-Time (`t = 90.0s`).
- **Pitch Dimensions ($1400.0 \times 840.0\text{ u}$)**:
  Expanded 6v6 pitch featuring $240.0\text{u}$ goal mouths centered at $Y = 420.0$, and $140.0\text{u}$ radius semicircle goalkeeper boxes.
- **76-Dimensional Team-Canonical Observation Space ($\mathbf{o} \in \mathbb{R}^{76}$)**:
  Dynamically transforms coordinates and velocities by $180^\circ$ for whichever team currently defends the right goal, ensuring an invariant forward-attacking perspective ($+X$ toward opponent goal) across both halves:
  - **16 Self Features (`indices 0..15`)**: Canonical position, facing angle vector $(\sin, \cos)$, normalized velocity, score differential, ball possession flag, hold/box timers, tackle/lunge cooldown, ghost/stun timer, relative goal angle and distance, and 3-bit one-hot role vector `[is_def, is_mid, is_atk]`.
  - **60 Entity Features (`indices 16..75`)**: 12 entity slots $\times$ 5 features (Slot 0: Ball with $360^\circ$ awareness; Slots 1..5: 5 Teammates; Slots 6..10: 5 Field Opponents; Slot 11: Enemy Goalkeeper).
- **4-Dimensional Continuous Action Space ($\mathbf{a} \in [-1, 1]^4$)**:
  Throttle force (`action[0]`), angular steering rate (`action[1]`), kick trigger & power (`action[2]`), and special tactical ability / tackle / keeper lunge (`action[3]`).
- **Heterogeneous Role Profiles & Mechanics**:
  - **Goalkeeper (`ROLE_KEEPER` / 1 per team)**: Confined to $r \le 140.0\text{u}$ semicircle, slowest run speed ($150\text{u/s}$), $430\text{u/s}$ explosive lunges (`0.25s` duration), $5.0\text{s}$ maximum hold violation rule, and loose-ball box recovery.
  - **Center Backs (`ROLE_DEF` / 2 per team)**: $180\text{u/s}$ speed, widest tackle radius ($48.0\text{u}$, 86% success), holding defensive shape and distributing breakout passes.
  - **Box-to-Box Midfielders (`ROLE_MID` / 2 per team)**: $190\text{u/s}$ speed, balanced tackles ($44.0\text{u}$, 78% success), transition play, wide flank channels, and playmaker link bonuses to attackers.
  - **Attacking Striker (`ROLE_ATK` / 1 per team)**: Fastest speed ($205\text{u/s}$), highest kick launch velocity ($640\text{u/s}$) with **zero kick jitter (pinpoint laser)**, and prime-zone finishing bonuses within $320\text{u}$ of goal.
- **PPO Co-Evolution & Self-Play (`train_ppo.py`)**:
  - Independent Blue Team (`blue_field_policy` + `blue_keeper_policy`) vs. Red Team (`red_field_policy` + `red_keeper_policy`) brains.
  - Rolling 3-generation co-evolution soft-sync mechanism to prevent runaway dominance and preserve competitive self-play parity.
  - Supports `--device cpu` and `--device cuda` with automatic device detection.
- **Interactive 60 FPS Replay Visualizer (`replay.py`)**:
  Decoupled Pygame player rendering vision cones, role halos, tackle radii, ghost dashed rings, scoreboard HUD, interactive timeline scrubbing, and floating event notifications.

---

## Project Structure

```
football_rl/
├── config.py             # Dataclass configurations: physics, mechanics, vision, rewards, PPO
├── env.py                # PettingZoo ParallelEnv: 60Hz physics, collision engine, Dec-POMDP obs
├── replay.py             # 60 FPS Pygame replay visualizer & match analyzer
├── train_ppo.py          # Vectorized PPO training loop, parameter sharing, co-evolution
└── tests/
    └── test_env.py       # Automated test suite (physics, symmetry, mechanics, PPO updates)
pyproject.toml            # Project packaging & dependencies
replay.py                 # Root CLI shortcut for replay visualizer
train_ppo.py              # Root CLI shortcut for training loop
```

---

## 1. Installation & Environment Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.12, Linux / WSL2)
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

By default, 1 generation runs 1 full match (`--num-envs 1 --rollout-steps 2700`) and exports a replay (`--interest-threshold 0.0`), meaning **Generation $N$ maps directly to Match #$N$**.

### Start a New Training Batch from Scratch
Wipes unarchived checkpoints in `checkpoints/` and `replays/`, initializes fresh weights, and trains Generations `1` to `100`:
```bash
.venv/bin/python train_ppo.py --from-scratch --total-updates 100
```

### Resume Training Active Batch
Loads weights and optimizers from `checkpoints/training_state.pt` and continues from the latest generation:
```bash
.venv/bin/python train_ppo.py --resume --total-updates 100
```
*(Tip: Whenever `checkpoints/training_state.pt` exists, running without `--from-scratch` automatically resumes).*

### Check Current Training Progress
Inspect current generation count, match statistics, and Top 10 most exciting matches in the terminal:
```bash
.venv/bin/python train_ppo.py --status
```

### Parallel & Device Execution Guidelines
- **Small Parallel Batches (`--num-envs 1..4`)**: **CPU is ~1.8x faster** (~125 SPS on modern multi-core CPU vs ~70 SPS on CUDA) because the continuous 2D physics simulation runs in Python and PCIe transfer overhead dominates small forward passes.
- **Scaled Batches (`--num-envs 8+` with high rollout steps)**: **CUDA** provides acceleration during large PPO mini-batch gradient descent epochs.

```bash
# Recommended CPU training for fast single/dual env simulation:
.venv/bin/python train_ppo.py --device cpu --num-envs 2 --total-updates 100

# Scaled training on GPU:
.venv/bin/python train_ppo.py --device cuda --num-envs 8 --rollout-steps 2700 --total-updates 200
```

### Training CLI Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--total-updates` | `100` | Number of PPO generations to train in this session. |
| `--num-envs` | `1` | Parallel environment instances. |
| `--rollout-steps` | `2700` | RL steps per env per generation (`2700` = 1 full 180s match). |
| `--interest-threshold` | `0.0` | Minimum `interest_score` to export a replay JSON (`0.0` = save all). |
| `--device` | `cuda`/`cpu` | Execution device (`cpu` or `cuda`, auto-detects CUDA). |
| `--resume` | Auto | Resume from `checkpoints/training_state.pt`. |
| `--from-scratch` | `False` | Reset `checkpoints/` and `replays/` to start at Generation 1. |
| `--shared-teams` | `False` | Force Blue and Red to share the same policy network (mirror self-play). |
| `--seed` | `42` | Random seed. |

---

## 3. Watching Replays & Match Analysis (`replay.py`)

### Human-Readable Match Summary
During training, [`replays/matches_summary.md`](replays/matches_summary.md) (also symlinked in root) is continuously updated with:
1. **Batch Overview** (total generations, total matches, total RL steps).
2. **Top 15 Leaderboard** ranked by `interest_score` (goals, passes, saves, tackles, fouls).
3. **Chronological Match Table** with final scores, possession stats, and the direct command to watch each match.

### Scan & Filter Matches in Terminal
```bash
# List all recorded matches and summary metrics
.venv/bin/python replay.py --list

# Filter matches within a generation range (e.g. 1 to 50)
.venv/bin/python replay.py --range 1 50

# Display Top 10 most exciting matches so far
.venv/bin/python replay.py --top 10
```

### Launch Interactive 60 FPS Replay Viewer
```bash
# Watch a match by its Match Number (e.g., Match #1, #25, #100)
.venv/bin/python replay.py 1
.venv/bin/python replay.py 25

# Watch the most recently saved replay
.venv/bin/python replay.py

# Watch an archived replay file
.venv/bin/python replay.py archives/batch_baseline/replays/match_0050_gen0050_score5_2_20261103.json

# Generate and watch a heuristic bot demonstration match
.venv/bin/python replay.py --generate-demo
```

### Viewer Keyboard & Mouse Controls
| Control | Action |
| :--- | :--- |
| `Space` | Pause / Resume playback. |
| `Left / Right` | Scrub `-5s / +5s` (or `-1 / +1` frame when paused / holding `Shift`). |
| `Mouse Click / Drag` | Scrub directly to any point on the bottom timeline (ticks indicate goals / HT). |
| `1 / 2 / 4` | Set playback speed multiplier (`1x`, `2x`, `4x`). |
| `V` | Toggle player vision cone overlays ($210^\circ$ FOV). |
| `R` | Restart replay from kickoff ($t = 0.0\text{s}$). |
| `Q / Esc` | Exit viewer. |

---

## 4. Experiment Archiving & Lifecycle Management

Save training progress (`checkpoints/` + `replays/` + `matches_summary.md`) into named experiment archives, test parameter variations, and restore older checkpoints at any time.

```bash
# 1. Archive the current baseline batch
.venv/bin/python train_ppo.py --archive run_100_baseline

# 2. List all saved experiment archives
.venv/bin/python train_ppo.py --list-archives

# 3. Modify parameters in football_rl/config.py and run a new experiment
.venv/bin/python train_ppo.py --from-scratch --total-updates 100
.venv/bin/python train_ppo.py --archive run_100_progressive_pass

# 4. Restore an earlier archive back to the active workspace to continue training
.venv/bin/python train_ppo.py --load-archive run_100_baseline
.venv/bin/python train_ppo.py --resume --total-updates 100
```

---

## 5. Automated Verification & Testing

Run the automated test suite covering PettingZoo compliance, rotational symmetry, ball physics, role constraints, semi-circle barriers, and PPO parameter-sharing:

```bash
.venv/bin/pytest football_rl/tests/test_env.py -v
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
