# Implementation Plan: Heuristic-Bootstrapped Gumbel AlphaZero

## Overview

3-stage pipeline: **Heuristic Player → Imitation Learning → Gumbel AlphaZero Self-Play**

Key compute-saving enhancements over vanilla AlphaZero:
- **Gumbel MCTS**: productive training at 32-50 sims (vs 200-800 for standard PUCT)
- **Playout cap randomization**: 75% of moves use quick search → 1.4x more games/hour
- **Progressive simulation**: start fast, increase quality as model improves
- **Train on ALL games**: 10-100x more data than win-only filtering

### Timeline Estimate

| Phase | Dev Time | Training Time | Wall-Clock |
|-------|----------|---------------|------------|
| 0: Housekeeping | 30min | — | 30min |
| 1: Heuristic Player | 2-3h | — | 2-3h |
| 2: Imitation Learning | 1-2h dev | 1-2h data gen + 1h training | 3-5h |
| 3: Gumbel MCTS | 3-4h | — | 3-4h |
| 4: AlphaZero Training | 1h dev | 8-24h training | 9-25h |
| 5: Eval & Polish | 2h | — | 2h |

**Total dev time: ~10-12 hours. Training wall-clock: 1-2 days.**

---

## Phase 0: Housekeeping (DONE)

Superseded files moved to `archive/`. Specs created.

---

## Phase 1: Heuristic Player

**Goal:** Rule-based player that wins >90% against random at full rules.

### Step 1.1: Position Scoring (`hexo/heuristic.py`)

Score each candidate move by analyzing the 3 hex axes:

```python
def score_move(state, q, r, player, game) -> float:
```

**Per-axis analysis:**
- Walk outward from (q, r) along the axis in both directions
- Count own consecutive stones: `own_len`
- Count open ends (empty cells at line endpoints): `open_ends`
- Count opponent consecutive stones (if move blocks): `opp_len`

**Scoring formula:**
```
THREAT_SCORE = {0: 0, 1: 1, 2: 4, 3: 16, 4: 64, 5+: 1000}
OPEN_MULT = {0: 0.1, 1: 0.5, 2: 1.0}

offense = sum over 3 axes: THREAT_SCORE[own_len] * OPEN_MULT[open_ends]
defense = sum over 3 axes: THREAT_SCORE[opp_len] * OPEN_MULT[opp_open_ends]
score = offense + 0.9 * defense + small_random_noise
```

### Step 1.2: Move Selector

```python
def heuristic_select_move(state, game, temperature=1.0) -> tuple[int, np.ndarray]:
    """Returns (action_index, probability_distribution)."""
```

Softmax over scores with temperature. Returns both the sampled action AND the full
probability distribution (needed for imitation learning targets).

### Step 1.3: Game Generator

```python
def play_heuristic_game(game, temp=1.0, max_moves=200) -> list[tuple]:
    """Returns list of (encoded_state, move_probs, outcome) — same format as AlphaZero."""
```

### Step 1.4: Tests (`tests/test_heuristic.py`)

- [ ] Win-completing move ranks highest when available
- [ ] Blocking move ranks highest when opponent threatens win
- [ ] Scores increase exponentially with line length
- [ ] 100 games vs random: >90% win rate
- [ ] 100 heuristic vs heuristic games: >70% decisive (not all draws)
- [ ] Average game length < 100 moves at full rules

### Validation Gate
200 heuristic vs heuristic games at win=5/dist=8.
**Pass:** >70% decisive, avg length < 100 moves. **Fail:** tune scoring weights.

---

## Phase 2: Imitation Learning

**Goal:** NN that guides MCTS productively at full rules.

### Step 2.1: Data Generation (`alphazero/imitation.py`)

```python
def generate_imitation_data(game, num_games=10000) -> list[tuple]:
```

- Play 10,000 heuristic self-play games at win=5, dist=8
- Temperature sampled uniformly from [0.8, 2.0] per game
- Keep ALL games (wins AND draws)
- Record (encoded_state, heuristic_move_probs, outcome) per position
- Filter oversized states (max_dim=60)
- Target: 100k+ training samples

### Step 2.2: Training (`train_bootstrap.py`)

```bash
python train_bootstrap.py imitation          # Run imitation learning only
python train_bootstrap.py selfplay           # Run AlphaZero from checkpoint
python train_bootstrap.py all                # Full pipeline
```

Imitation training:
- 90/10 train/val split
- 25 epochs, cosine LR 0.001 → 0.0001, batch size 128
- Early stopping if val loss doesn't improve for 5 epochs
- Save: `checkpoints_v2/imitation_final.pt`

### Step 2.3: Validation

1. **NN-only play** (50 games, no MCTS): games show line-building (longest line > 3)
2. **NN+MCTS (50 sims) vs strong heuristic** (40 games): **>30% win rate**
3. **Value correlation**: NN value vs outcome on 1000 val positions: **r > 0.3**

**If validation fails:** more data, wider temp range, or add 1-2 ply minimax to heuristic.
**Do NOT proceed** to self-play with a model that fails validation.

---

## Phase 3: Gumbel MCTS Implementation

**Goal:** Replace PUCT selection with Gumbel Sequential Halving.

### Step 3.1: Gumbel Selection (`alphazero/mcts.py` rewrite)

Replace the existing PUCT-based selection with:

**Root move selection (Gumbel):**
1. Get NN policy prior π for all legal moves
2. Sample Gumbel noise: g_i ~ Gumbel(0, 1) for each legal move
3. Compute scores: s_i = log(π_i) + g_i
4. Select top-k moves (k=16 or 32) by score
5. Allocate simulations via Sequential Halving:
   - Round 1: give N/(k * ceil(log2(k))) sims to each of k moves
   - Round 2: keep top k/2 by Q-value, double their sim budget
   - Repeat until 1 move remains
6. The selected move's "completed" Q-value is the policy improvement target

**Non-root selection (standard):**
Below the root, use standard PUCT with NN policy priors (same as before). Gumbel
selection only applies at the root node where we need the policy improvement guarantee.

**Policy target construction:**
For each root move i that was searched, compute:
```
completed_logit_i = log(π_i) + σ(q_i)
```
where σ(q) transforms Q-values to logit scale. The softmax of completed logits is the
improved policy target for training.

### Step 3.2: Playout Cap Randomization

In self-play, for each move:
```python
if random.random() < 0.25:
    sims = full_sim_count       # Full search — contributes to policy target
    include_in_policy_training = True
else:
    sims = full_sim_count // 4  # Quick search — only contributes value target
    include_in_policy_training = False
```

### Step 3.3: Tests (`tests/test_mcts.py` expanded)

- [ ] Gumbel selection with k=16, N=32 sims completes without error
- [ ] Gumbel at 32 sims produces better policy than PUCT at 32 sims (100 positions)
- [ ] Sequential halving correctly eliminates half the candidates each round
- [ ] Completed policy sums to 1.0 and is valid probability distribution
- [ ] Playout cap correctly flags positions for policy/value-only training

### Validation Gate
On 100 positions from heuristic games, compare PUCT vs Gumbel at 32 sims:
**Pass:** Gumbel's top-1 move matches 200-sim PUCT more often than 32-sim PUCT does.

---

## Phase 4: AlphaZero Self-Play Training

**Goal:** Model that surpasses heuristic through self-play.

### Step 4.1: Pipeline Configuration

Update `config.py`:
```python
V2_CONFIG = {
    'win_length': 5,
    'max_placement_dist': 8,
    'num_resBlocks': 10,
    'num_hidden': 128,
    'C': 2,
    'gumbel_top_k': 16,
    'num_searches': 32,          # Starting sims (progressive)
    'playout_cap_fraction': 0.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.15,
    'temperature': 1.0,
    'temperature_threshold': 30,
    'num_iterations': 200,
    'num_selfPlay_iterations': 50,
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

### Step 4.2: Self-Play Modifications

Changes to `alphazero/self_play.py`:
- Use Gumbel MCTS at root, standard PUCT below root
- Apply playout cap randomization per move
- Flag each sample as policy-trainable or value-only
- Keep ALL games (not just wins)

Changes to `alphazero/pipeline.py`:
- Progressive simulation schedule (auto-increase based on arena results)
- Larger replay buffer (100k)
- Cosine LR decay
- Load from imitation checkpoint

### Step 4.3: Training Run

```bash
python train_bootstrap.py selfplay \
    --checkpoint checkpoints_v2/imitation_final.pt \
    --num-iterations 200 \
    --num-workers 8
```

**Expected throughput at 32 sims, 8 workers:** ~15-30 games/hour (vs ~3/hour at 200 sims)
**Expected per-iteration:** ~50 games + training in ~15-20 minutes

Monitor for:
- Decisive game rate >30% from iteration 1 (thanks to warm start)
- Policy loss decreasing (should start ~3-4, drop below 2 by iter 50)
- Arena acceptance ~50-70% of iterations
- Progressive sim increases at natural intervals

### Step 4.4: Scaling to win=6

After model beats strong heuristic >90% at win=5:
```bash
python train_bootstrap.py selfplay \
    --checkpoint checkpoints_v2/model_best.pt \
    --win-length 6 --num-iterations 100
```

---

## Phase 5: Evaluation & Polish

- [ ] NN+Gumbel MCTS (128 sims) vs strong heuristic: target >95% win rate
- [ ] Web UI loads new model, plays responsively
- [ ] Update CLAUDE.md with new training approach and results
- [ ] Record training findings

---

## File Structure After Implementation

```
HeXO-alpha-zero/
  CLAUDE.md
  config.py                    # Updated with V2_CONFIG
  train_bootstrap.py           # NEW: unified entry point
  play.py
  play_web.py
  hexo/
    game.py
    hex_utils.py
    heuristic.py               # NEW: position evaluator + move selector
  alphazero/
    model.py
    mcts.py                    # REWRITTEN: Gumbel selection + Sequential Halving
    train.py                   # MODIFIED: playout cap awareness
    self_play.py               # MODIFIED: Gumbel, playout cap, keep all games
    inference.py
    arena.py
    pipeline.py                # MODIFIED: progressive sims, larger buffer, LR decay
    imitation.py               # NEW: data generation + imitation training
  tests/
    test_game.py
    test_mcts.py               # EXPANDED: Gumbel-specific tests
    test_model.py
    test_heuristic.py          # NEW
    test_imitation.py          # NEW
  specs/
    _overview.md
    001-approach-selection.md
    heuristic-player.md
    imitation-learning.md
    alphazero-training-v2.md
    implementation-plan.md
  checkpoints_v2/              # NEW
    imitation_final.pt
    model_iter{N}.pt
    model_best.pt
  archive/                     # Superseded files (already moved)
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Heuristic too weak (all draws) | Low | Medium | Tune weights, add asymmetry |
| Imitation model fails validation | Low | High | Validation gate prevents wasted self-play |
| Gumbel top-k too small for HeXO | Medium | Medium | Start at k=16, increase if needed |
| Self-play stalls | Low | Medium | Progressive schedule adapts; roll back to good checkpoint |
| VRAM OOM | Medium | Low | max_dim filter proven effective |

## Compute Budget

**Conservative estimate for win=5 training (32→128 sims, 200 iterations):**
- Self-play: ~15 games/hour × 200 iterations × 50 games = ~667 GPU-hours
  → But with 8 workers generating in parallel: ~80 GPU-hours wall-clock
- Training: ~4 epochs × 200 iterations × 2 min = ~27 hours
- **Total wall-clock: ~24-48 hours** (overnight to 2 nights)

**Compared to prior approach:** 14+ hours of overnight training produced 0 usable model.
The new approach should produce a competent model in the first few hours of self-play
thanks to the warm start + Gumbel efficiency.

## Success Criteria

1. Model consistently beats heuristic at full rules
2. Self-play games show strategic depth (forks, threats, blocking)
3. Model improves measurably over training (arena wins)
4. Entire training completes in <48 hours wall-clock
5. No wasted compute — every phase has a validation gate before proceeding
