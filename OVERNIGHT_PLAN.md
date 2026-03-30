# Overnight Training Plan

**Start:** ~6:30pm | **End:** ~8am | **Duration:** ~14 hours

## The Problem
MCTS + random NN produces FEWER wins than pure random play. The NN concentrates
search on bad moves, preventing accidental line formation. Result: 100% draws,
zero training signal.

## The Solution: Random Seeding + Curriculum

Each phase:
1. **Seed** with random games (where wins happen naturally without NN)
2. **Pre-train** the value head on this data (learns "these positions → win/loss")
3. **MCTS self-play** takes over — now the NN has SOME understanding, MCTS improves it

## Phase Schedule

| Phase | Win | Dist | Sims | Duration | Purpose |
|-------|-----|------|------|----------|---------|
| 1 | 3 | 2 | 25 | 1h | Bootstrap: lines are good |
| 2 | 4 | 3 | 40 | 2h | Extend to 4-in-a-row |
| 3 | 4 | 5 | 60 | 2h | Expand spatial range |
| 4 | 5 | 6 | 75 | 3h | Target win condition |
| 5 | 5 | 8 | 100 | 6h | Full rules, polish |

## Validated Assumptions (from testing)
- Random play at win=3/dist=2: **70% win rate** → good seed data
- Random play at win=4/dist=3: **16% win rate** → ok seed data (need more games)
- Random play at win=5/dist=5: **2% win rate** → marginal (model should be strong by then)
- Pre-training on random data → MCTS starts producing wins (~20% by iter 4)
- win=3/dist=1 with 25 sims: 100% wins but doesn't transfer to dist=2

## Key Parameters
- **Model:** 10 res blocks, 128 hidden, ~2.8M params, GroupNorm, fully convolutional
- **MCTS:** Lazy expansion, Dirichlet noise at root
- **Temperature:** High early (1.5 → 1.0) for exploration
- **Noise:** High early (eps=0.5 → 0.25) to prevent NN overriding MCTS with bad advice

## What to Look For in Logs
- **v_loss > 0:** Value head is learning (good!)
- **Win rate increasing:** Model is improving
- **p_loss decreasing:** Policy is sharpening
- **All draws:** Stuck (may need intervention — lower win_length or dist)

## Checkpoints
Saved to `checkpoints_overnight/`:
- `phase1_iter0.pt`, `phase1_iter10.pt`, ..., `phase1_final.pt`
- Same pattern for each phase
- `model_final.pt` — the final model after all phases
