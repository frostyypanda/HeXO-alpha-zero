# Feature: Imitation Learning (Heuristic Bootstrap)

## Status
draft

## Why this exists

The AlphaZero training loop requires an NN that can guide MCTS productively. All prior
attempts to bootstrap from random play failed at higher complexity levels because:
- Random NN + MCTS → 0 wins → no signal → NN stays random (cold-start death spiral)
- Random play at full rules produces <0.1% wins (sparse reward)
- Curriculum doesn't transfer between difficulty levels

Imitation learning solves this by training the NN to mimic the heuristic player's moves
and game outcomes **before** AlphaZero self-play begins. This gives MCTS a competent
guide from the start.

## What it does

### Data Generation

Generate self-play games using the heuristic player at full rules (win=5, dist=8):

1. For each game:
   - Both sides play using heuristic with medium-strength settings
   - Vary temperature (0.8-2.0) across games for diversity
   - Record: `(encoded_state, move_probabilities, outcome)` for each position
2. Keep ALL games (not just wins) — draws provide valuable position evaluations too
3. Target: 100k+ training samples (achievable with ~5-10k games)
4. Label outcomes: +1 for winner's positions, -1 for loser's, 0 for draws

**Why keep draws too:** Prior approaches discarded draws, leaving <1% of data. Draws
contain valuable information: "this position is roughly equal." The value head needs to
learn that most positions are close to 0, with wins/losses being the exception.

### Training Procedure

Standard supervised learning on the generated data:

1. **Policy target**: Cross-entropy loss between NN policy and heuristic move probabilities
2. **Value target**: MSE loss between NN value and game outcome (+1/-1/0)
3. **Loss**: `L = policy_loss + value_loss` (same as AlphaZero training)
4. Use the existing `AlphaZero.train()` method — the data format is identical

Training hyperparameters:
- Epochs: 20-30 (until validation loss plateaus)
- Learning rate: 0.001 with cosine decay to 0.0001
- Batch size: 128-256 (large batches for stable gradients)
- Hold out 10% data for validation

### Validation Criteria

The imitation-learned model is ready for AlphaZero when:
1. **NN-only play** (no MCTS) shows line-building behavior (visual inspection)
2. **NN + MCTS (50 sims)** beats the heuristic player >30% of the time
3. **Value predictions** correlate with game outcomes (correlation > 0.3)

If criterion 2 fails, train longer or generate more diverse data. Do NOT proceed to
AlphaZero with a model that can't beat the heuristic — it will just collapse.

## Boundaries & edge cases

**What it intentionally does NOT do:**
- No data augmentation (hex grid symmetries could be added later if needed)
- No curriculum — generates all data at target rules directly
- No replay buffer management — single-pass supervised learning
- No opponent modeling — both sides use same heuristic

**Known limitations:**
- Imitation-learned model is bounded by heuristic quality (can't exceed teacher)
- Heuristic's blind spots become the NN's blind spots initially
- These are resolved in the AlphaZero phase when self-play surpasses the teacher

**Data size considerations:**
- 10k games at ~20 moves each = 200k positions = ~2GB in memory
- Fits comfortably in 32GB RAM
- If too large, sample a subset for training (100k samples is plenty)

## Testing & verification

### Key scenarios
- [ ] Generate 1000 games in <5 minutes (performance check)
- [ ] Generated games have >70% decisive outcomes (not all draws)
- [ ] After training, NN-only play produces games with line-building behavior
- [ ] After training, NN + MCTS (50 sims) beats heuristic >30% of the time
- [ ] Policy loss converges below 4.0 (well below random ~6.5)
- [ ] Value loss converges below 0.5

### Edge cases
- All games end in draws (heuristic too balanced): lower one side's strength
- VRAM spike from large boards: use existing max_dim filter (60px)
- Training diverges: reduce learning rate, check data quality

### Automation notes
- Smoke test: 100 games generation + 5 epochs training + arena eval
- Full run: 10k games + 25 epochs + comprehensive arena eval
- Save training curves (loss per epoch) for diagnostics

## Decision log
| Date | Decision | Why | Who |
|------|----------|-----|-----|
| 2026-04-06 | Keep draws in training data | Prior approach discarded draws → <1% of data kept → sparse signal. Draws teach the value head that "most positions are close to 0" | Kei + Claude |
| 2026-04-06 | 100k+ samples target | Phase 1 overnight run had 46k samples and showed solid learning. 100k gives margin for the harder full-rules setting | Kei + Claude |
| 2026-04-06 | Validation gate before AlphaZero | Prevents wasting hours on self-play with a bad model, which is exactly what happened in prior runs | Kei + Claude |
| 2026-04-06 | No hex symmetry augmentation initially | Adds complexity; heuristic can generate unlimited data anyway. Revisit if data diversity is a bottleneck | Kei + Claude |

## Related
- [specs/_overview.md](_overview.md) — epic context
- [specs/heuristic-player.md](heuristic-player.md) — generates the training data
- [specs/alphazero-training-v2.md](alphazero-training-v2.md) — consumes the pre-trained model
- `alphazero/train.py` — `AlphaZero.train()` used for supervised training
- `alphazero/model.py` — ResNet architecture (unchanged)
