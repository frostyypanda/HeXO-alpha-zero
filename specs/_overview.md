# Epic: Heuristic-Bootstrapped AlphaZero Training

## Origin

Multiple training runs (Mar 28-30, 2026) demonstrated that the original curriculum-based
AlphaZero approach cannot bootstrap from random play at higher complexity levels
(win>=4, dist>=3). This epic replaces the curriculum strategy with a heuristic-bootstrap
approach that eliminates the cold-start problem entirely.

## High-Level Vision

1. Build a hand-crafted heuristic player that understands HeXO tactics (line-building,
   blocking, fork threats).
2. Pre-train the neural network via imitation learning on heuristic self-play data.
3. Run standard AlphaZero self-play from the warm-started model at full rules.

This sidesteps every failure mode observed in prior training runs:
- No cold start (NN already knows lines are good before MCTS starts)
- No curriculum needed (go straight to full rules)
- No sparse reward problem (heuristic games produce wins reliably)
- No transfer gap (single-phase training at target rules)

## Child Specs

| Spec | Status | Purpose |
|------|--------|---------|
| [heuristic-player.md](heuristic-player.md) | draft | Rule-based position evaluator and move selector |
| [imitation-learning.md](imitation-learning.md) | draft | Supervised pre-training on heuristic game data |
| [alphazero-training-v2.md](alphazero-training-v2.md) | draft | Updated self-play pipeline from warm model |

## Evidence Base

### What We Know Works
- Game engine: complete, well-tested (29 unit tests), correct rules
- NN architecture: FCN ResNet, GroupNorm, masked pooling — handles variable-size boards
- MCTS: lazy expansion (8x speedup), handles 300+ legal moves efficiently
- Parallel infrastructure: persistent workers, batched GPU inference, hot-reload weights
- Phase 1 training (win=3, dist=2): 72% random win rate, model learned well

### What Failed and Why

| Run | Symptom | Root Cause |
|-----|---------|------------|
| Overnight MCTS, Phase 2+ | 0 wins for 20+ iterations at win=4/dist=3 | Random NN misleads MCTS away from wins |
| NN-only, Phases F-G | 0.6% win rate at win=5, policy loss stuck at ~5.8 | NN-only play without search is too weak |
| NN-only, Phase H | 0 wins for 48+ iterations at win=5/dist=6 | Random wins at 0.05% — no signal at all |
| Transfer test | win=3/dist=1 model → 0% at dist=2 | Spatial patterns don't transfer across scales |

### Validated Data Points
- Random win rates: 72% (w3/d2), 33% (w3/d4), 23% (w4/d3), 2% (w5/d5), 0.05% (w5/d6)
- MCTS + good value head (Phase 1): 12% win rate, improving
- NN pre-training converges in 4-8 epochs when given good data
- Existing 3.0M param model fits in VRAM with room to spare on RTX 4090

## Existing Assets to Reuse

All core infrastructure is kept as-is:
- `hexo/game.py`, `hexo/hex_utils.py` — game engine
- `alphazero/model.py` — neural network
- `alphazero/mcts.py` — Monte Carlo tree search
- `alphazero/self_play.py`, `alphazero/inference.py` — parallel self-play
- `alphazero/pipeline.py` — orchestrator
- `alphazero/arena.py` — model evaluation
- `alphazero/train.py` — training loop
- `play.py`, `play_web.py` — play interfaces
- `config.py` — hyperparameters

## Files to Archive

Moved to `archive/` (not deleted):
- `overnight_train.py` — curriculum MCTS approach (superseded)
- `nn_curriculum_train.py` — NN-only curriculum approach (superseded)
- `OVERNIGHT_PLAN.md` — curriculum plan (superseded)
- `train_hexo.py` — old CLI entry point (replaced by train_bootstrap.py)
- `test_curriculum.py`, `test_curriculum2.py` — curriculum smoke tests
- `test_seeding.py`, `test_training_signal.py`, `test_transfer.py` — ad-hoc tests
- `test_overnight_smoke.py` — overnight smoke test

## Existing Weights

Checkpoints from prior runs exist in `checkpoints/`, `checkpoints_overnight/`,
`checkpoints_nn/`. These were trained on restricted rules (win=3-5, dist=2-5) and
showed poor transfer. The heuristic bootstrap should produce a stronger starting point.
However, we keep all checkpoints for reference — they're small (~12MB each).
