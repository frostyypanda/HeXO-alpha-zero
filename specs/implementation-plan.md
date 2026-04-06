# Implementation Plan: Heuristic-Bootstrapped AlphaZero

## Overview

Replace the curriculum-based training with a 3-stage pipeline:
**Heuristic Player → Imitation Learning → AlphaZero Self-Play**

### Timeline Estimate

| Phase | Work | Training | Total |
|-------|------|----------|-------|
| 0: Housekeeping | 30min | — | 30min |
| 1: Heuristic Player | 2-3h dev | 0 | 2-3h |
| 2: Imitation Learning | 1-2h dev | 1-2h gen + 1h train | 3-5h |
| 3: AlphaZero Training | 1h dev | 8-24h training | 9-25h |
| 4: Eval & Polish | 2h | — | 2h |

**Total developer time: ~7-8 hours. Total wall-clock including training: 1-2 days.**

---

## Phase 0: Housekeeping

Move superseded files to `archive/`. No code changes.

**Files to archive:**
```
archive/
  overnight_train.py
  nn_curriculum_train.py
  OVERNIGHT_PLAN.md
  train_hexo.py
  test_curriculum.py
  test_curriculum2.py
  test_seeding.py
  test_training_signal.py
  test_transfer.py
  test_overnight_smoke.py
```

**Keep in place (still used):**
- All `hexo/` files
- All `alphazero/` files
- `play.py`, `play_web.py`
- `config.py`
- All `checkpoints*/` directories
- All log files (reference)

---

## Phase 1: Heuristic Player

**Goal:** A rule-based player that wins >90% against random at full rules.

### Step 1.1: Position Scoring Function

Create `hexo/heuristic.py`.

```python
def score_move(state, q, r, player, game) -> float:
    """Score a single candidate move for the given player."""
```

For each of the 3 hex axes, count:
- Own consecutive stones through (q, r): `own_len`
- Open ends (empty cells at both ends of the line): `open_ends`
- Opponent consecutive stones through (q, r): `opp_len`

Scoring formula:
```
offense = sum over axes: base_score[own_len] * open_end_multiplier[open_ends]
defense = sum over axes: base_score[opp_len] * open_end_multiplier[open_ends]
score = offense + 0.9 * defense
```

Base scores (exponential):
```
base_score = {0: 0, 1: 1, 2: 4, 3: 16, 4: 64, 5+: 1000}
```
These values ensure a 4-in-a-row is treated as ~60x more important than a single stone.

Open-end multiplier: `{0: 0.1, 1: 0.5, 2: 1.0}` (dead line is nearly worthless).

### Step 1.2: Move Selector

```python
def heuristic_select_move(state, game, temperature=1.0) -> int:
    """Select a move using heuristic scores with temperature-controlled sampling."""
```

1. Score all legal moves
2. Apply softmax with temperature
3. Sample from distribution

### Step 1.3: Self-Play Game Generator

```python
def play_heuristic_game(game, temperature=1.0) -> list[tuple]:
    """Play one game, return list of (encoded_state, move_probs, outcome)."""
```

Uses the standard game loop. Records data in the same format as AlphaZero self-play.

### Step 1.4: Tests

`tests/test_heuristic.py`:
- Scoring function produces expected relative ordering for known positions
- Blocking move is top-ranked when opponent threatens win
- Win-completing move is top-ranked when available
- Heuristic beats random >90% in 100 games (integration test)
- Heuristic vs heuristic games are >80% decisive (not all draws)

### Validation Gate
Run 200 heuristic vs heuristic games at full rules (win=5, dist=8).
**Pass criteria:** >70% decisive outcomes, average game length < 100 moves.

---

## Phase 2: Imitation Learning

**Goal:** NN that can guide MCTS productively at full rules.

### Step 2.1: Data Generation Script

Create `alphazero/imitation.py`:

```python
def generate_imitation_data(game, num_games, temperature_range, ...) -> list[tuple]:
    """Generate training data from heuristic self-play."""
```

- Play 10,000 games with temperature sampled from [0.8, 2.0]
- Keep ALL games (wins AND draws)
- Record (encoded_state, heuristic_move_probs, game_outcome) per position
- Filter oversized states (max_dim=60)
- Target: 100k+ training samples

### Step 2.2: Training Script

Create `train_bootstrap.py` (new entry point, replaces `train_hexo.py`):

```
python train_bootstrap.py --phase imitation    # Run imitation learning
python train_bootstrap.py --phase selfplay     # Run AlphaZero self-play
python train_bootstrap.py --phase all          # Run full pipeline
```

Imitation training:
- Load generated data
- Split 90/10 train/val
- Train for 25 epochs with cosine LR decay (0.001 → 0.0001)
- Early stopping if val loss doesn't improve for 5 epochs
- Save checkpoint: `checkpoints_v2/imitation_final.pt`

### Step 2.3: Validation

After imitation training, run automated validation:

1. **NN-only test**: Play 50 games with NN policy (no MCTS). Check that games show
   line-building behavior (longest line in game > random average).

2. **NN+MCTS vs heuristic**: Play 40 games with MCTS (50 sims) against strong heuristic.
   **Pass criteria:** >30% win rate.

3. **Value correlation**: For 1000 random positions from validation set, compare NN value
   prediction to actual game outcome. **Pass criteria:** Pearson correlation > 0.3.

**If validation fails:** generate more data (wider temperature range), train longer, or
adjust heuristic strength.

---

## Phase 3: AlphaZero Self-Play

**Goal:** Model that surpasses heuristic through self-play.

### Step 3.1: Pipeline Configuration

Update `config.py` with v2 defaults:

```python
V2_CONFIG = {
    'win_length': 5,
    'max_placement_dist': 8,
    'num_resBlocks': 10,
    'num_hidden': 128,
    'C': 2,
    'num_searches': 200,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.15,
    'temperature': 1.0,
    'temperature_threshold': 30,  # switch to 0.3 after move 30
    'num_iterations': 200,
    'num_selfPlay_iterations': 50,  # games per worker per iteration
    'num_epochs': 4,
    'batch_size': 128,
    'lr': 0.001,
    'num_workers': 8,
    'replay_buffer_size': 100000,
    'arena_games': 40,
    'arena_threshold': 0.55,
    'max_dim': 60,
}
```

### Step 3.2: Training Loop Modifications

Minimal changes to existing `pipeline.py`:

1. **Load imitation checkpoint** (via `--resume` flag)
2. **Keep all games in replay buffer** (not just wins)
3. **Temperature schedule**: 1.0 for moves 1-30, 0.3 after (configure in self_play.py)
4. **Larger replay buffer**: 100k samples (was 40k)
5. **Cosine LR decay**: decrease lr from 0.001 to 0.0001 over training

### Step 3.3: Training Run

```bash
python train_bootstrap.py --phase selfplay \
    --checkpoint checkpoints_v2/imitation_final.pt \
    --num-iterations 200 \
    --num-workers 8
```

Run overnight (8-24h). Monitor:
- Win rate per iteration (should be >30% from start, rising)
- Policy loss (should decrease from ~3-4 to <2)
- Arena acceptance rate (should accept ~50-70% of iterations)

### Step 3.4: Scaling to win=6

After model plays well at win=5 (beats strong heuristic >90%):

```bash
python train_bootstrap.py --phase selfplay \
    --checkpoint checkpoints_v2/model_best.pt \
    --win-length 6 \
    --num-iterations 100
```

---

## Phase 4: Evaluation & Polish

### Step 4.1: Strength Evaluation

- NN+MCTS (200 sims) vs strong heuristic: target >95% win rate
- NN+MCTS (200 sims) vs NN+MCTS (50 sims): assess sim count impact
- AI vs AI games: visual inspection for strategic quality

### Step 4.2: Web UI Integration

Update `play_web.py` to load the new model:
- Default to v2 best model
- Configurable MCTS simulations (time budget for real-time play)

### Step 4.3: Documentation

- Update CLAUDE.md with new training approach
- Record training results and findings

---

## File Structure After Implementation

```
HeXO-alpha-zero/
  CLAUDE.md
  config.py                    # Updated with V2_CONFIG
  train_bootstrap.py           # NEW: unified entry point (imitation → self-play)
  play.py
  play_web.py
  hexo/
    game.py
    hex_utils.py
    heuristic.py               # NEW: position evaluator + move selector
  alphazero/
    model.py
    mcts.py
    train.py
    self_play.py               # MODIFIED: temperature schedule, keep all games
    inference.py
    arena.py
    pipeline.py                # MODIFIED: replay buffer size, LR decay
    imitation.py               # NEW: data generation + imitation training
  tests/
    test_game.py
    test_mcts.py
    test_model.py
    test_heuristic.py          # NEW: heuristic player tests
    test_imitation.py          # NEW: imitation learning tests
  specs/
    _overview.md
    heuristic-player.md
    imitation-learning.md
    alphazero-training-v2.md
    implementation-plan.md
  checkpoints_v2/              # NEW: v2 training checkpoints
    imitation_final.pt
    model_iter{N}.pt
    model_best.pt
  archive/                     # Superseded files
    overnight_train.py
    nn_curriculum_train.py
    OVERNIGHT_PLAN.md
    train_hexo.py
    test_curriculum.py
    test_curriculum2.py
    test_seeding.py
    test_training_signal.py
    test_transfer.py
    test_overnight_smoke.py
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Heuristic too weak (draws only) | Low | Medium | Tune scoring weights, add randomness asymmetry |
| Imitation model doesn't guide MCTS | Low | High | Validation gate prevents wasted self-play time |
| Self-play stalls (no improvement) | Medium | Medium | Increase sims, tune exploration, try LR warmup |
| VRAM OOM on large boards | Medium | Low | max_dim filter already proven effective |
| Training too slow | Medium | Low | Start at 200 sims, increase if quality is good |

## Success Criteria

The project succeeds when:
1. The model consistently wins against the heuristic player at full rules
2. Self-play games show strategic depth (forks, multi-turn threats, blocking)
3. The model improves over training iterations (measurable via arena)
4. Human players find the AI challenging and "intelligent" via the web UI
